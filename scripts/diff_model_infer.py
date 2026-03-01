from __future__ import annotations
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import random
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from types import MethodType
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import monai
from monai.data import DataLoader, partition_dataset
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose
from monai.utils import set_determinism
from tqdm import tqdm

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance, dynamic_infer
from .solver import euler_step, midpoint_step, rk4_step, rk5_step

# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

# ═══════════════════════════════════════════════════════════════════════════════
# 常數
# ═══════════════════════════════════════════════════════════════════════════════

# 後處理反正規化參數（原始碼硬編碼於 run_inference 中）
INTENSITY_A_MIN = -1000
INTENSITY_A_MAX = 1000
INTENSITY_B_MIN = 0
INTENSITY_B_MAX = 1


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════════════════════════


def set_random_seed(seed: Optional[int]) -> int:
    """
    設定隨機種子以確保可重現性。

    Args:
        seed: 隨機種子，若為 None 則自動產生。

    Returns:
        實際使用的隨機種子。
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def _load_json_field(file_path: str, key: str, convert_to_float: bool = True):
    """
    從 JSON 檔案載入指定欄位。

    問題: 原始碼將此函式定義在 prepare_data 內部，每次呼叫 prepare_data 時重建。
    解法: 抽為模組層級函式，避免重複定義。
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    if convert_to_float:
        return torch.FloatTensor(data[key])
    return data[key]


# ═══════════════════════════════════════════════════════════════════════════════
# 模型載入
# ═══════════════════════════════════════════════════════════════════════════════

def compile_unet_model(model, logger):
    compile_module_list = ["conv_in", "down_blocks", "middle_block", "up_blocks", "out"]
    for module_name in compile_module_list:
        module = getattr(model, module_name, None)
        if module is not None:
            # 統一使用 .compile() 進行就地編譯，不分 Module 或 ModuleList
            module.compile(
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,
                backend="inductor",
            )
            logger.info(f"Compiled {module_name} with max-autotune.")

    return model

def compile_autoencoder_model(model, shape, device):
    """
    編譯 autoencoder 的 encode 方法，使用 torch.compile 而非私有 API。
    """
    model = torch.compile(
        model,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
        backend="inductor",
    )
    # warmup: 觸發編譯
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=True, dtype=torch.float32):
            example_inputs = torch.randn(1, 1, *shape, device=device)
            _ = model.decode(example_inputs)
    return model


def load_models(
        args: argparse.Namespace,
        device: torch.device,
        logger: logging.Logger,
) -> tuple:
    """
    載入 autoencoder 與 UNet 模型。

    修正:
      1. autoencoder 的 torch.load 加入 map_location（原始碼缺少）
      2. 加入 weights_only=False（明確指定，避免未來版本警告）
      3. 載入後立即 eval()（原始碼延遲到 run_inference 才設定）
      4. 可選 torch.compile 加速推理
    """
    # ── Autoencoder ──
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path,
        map_location=device,
        weights_only=False,
    )
    if "unet_state_dict" in checkpoint_autoencoder:
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    logger.info(f"checkpoints {args.trained_autoencoder_path} loaded.")

    # ── UNet ──
    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(
        f"{args.model_dir}/{args.model_filename}",
        map_location=device,
        weights_only=False,
    )
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    unet.eval()
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    src_shift_factor = checkpoint["src_shift_factor"]
    src_scale_factor = checkpoint["src_scale_factor"]
    tar_shift_factor = checkpoint["tar_shift_factor"]
    tar_scale_factor = checkpoint["tar_scale_factor"]

    logger.info(f"src_shift_factor -> {src_shift_factor}.")
    logger.info(f"src_scale_factor -> {src_scale_factor}.")
    logger.info(f"tar_shift_factor -> {tar_shift_factor}.")
    logger.info(f"tar_scale_factor -> {tar_scale_factor}.")

    return autoencoder, unet, src_shift_factor, src_scale_factor, tar_shift_factor, tar_scale_factor


# ═══════════════════════════════════════════════════════════════════════════════
# 資料準備
# ═══════════════════════════════════════════════════════════════════════════════


def load_filenames(data_list_path: str, mode: str) -> list:
    """從 JSON 資料清單載入檔名並轉換為 embedding/prediction 路徑。"""
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    return [
        {
            "src_image": item["src_image"].replace(".nii.gz", "_emb.nii.gz"),
            "tar_image": item["tar_image"].replace(".nii.gz", "_emb.nii.gz"),
            "file_name": item["tar_image"].replace(".nii.gz", "_pred.nii.gz"),
        }
        for item in json_data[mode]
    ]


def prepare_file_list(
        filenames: list,
        embedding_base_dir: str,
        mode: str,
        include_body_region: bool,
        include_modality: bool,
) -> list:
    """
    建立包含完整路徑的檔案清單，跳過不存在的檔案。

    修正: 原始碼用 range(len(...)) 迭代，改為直接迭代 item。
    """
    files_list = []
    for item in filenames:
        str_src_img = os.path.join(embedding_base_dir, mode, item["src_image"])
        str_tar_img = os.path.join(embedding_base_dir, mode, item["tar_image"])
        if (not os.path.exists(str_src_img)) and (not os.path.exists(str_tar_img)):
            continue

        str_info = str_src_img + ".json"
        files_i = {
            "src_image": str_src_img,
            "tar_image": str_tar_img,
            "spacing": str_info,
            "out_name": item["file_name"],
        }
        if include_body_region:
            files_i["top_region_index"] = str_info
            files_i["bottom_region_index"] = str_info
        if include_modality:
            files_i["modality"] = str_info
        files_list.append(files_i)
    return files_list


def prepare_data(
        train_files: list,
        cache_rate: float,
        num_workers: int = 2,
        batch_size: int = 1,
        include_body_region: bool = False,
        include_modality: bool = True,
        modality_mapping: Optional[dict] = None,
) -> DataLoader:
    """
    建立 DataLoader。

    修正:
      - pin_memory=True: 加速 CPU→GPU 資料傳輸（需搭配 non_blocking=True）
      - persistent_workers=True: 避免每個 epoch 重啟 worker 進程
    """
    transforms_list = [
        monai.transforms.LoadImaged(keys=["src_image", "tar_image"]),
        monai.transforms.EnsureChannelFirstd(
            keys=["src_image", "tar_image"], channel_dim=-1
        ),
        monai.transforms.Lambdad(
            keys="spacing",
            func=lambda x: _load_json_field(x, "spacing"),
        ),
        monai.transforms.Lambdad(
            keys="spacing",
            func=lambda x: x * 1e2,
        ),
    ]

    if include_body_region:
        transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index",
                func=lambda x: _load_json_field(x, "top_region_index"),
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index",
                func=lambda x: _load_json_field(x, "bottom_region_index"),
            ),
            monai.transforms.Lambdad(
                keys="top_region_index",
                func=lambda x: x * 1e2,
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index",
                func=lambda x: x * 1e2,
            ),
        ]

    if include_modality:
        transforms_list += [
            monai.transforms.Lambdad(
                keys="modality",
                func=lambda x: modality_mapping[
                    _load_json_field(x, "modality", convert_to_float=False)
                ],
            ),
            monai.transforms.EnsureTyped(
                keys=["modality"], dtype=torch.long
            ),
        ]

    train_ds = monai.data.CacheDataset(
        data=train_files,
        transform=Compose(transforms_list),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    use_persistent = num_workers > 0
    return DataLoader(
        train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=use_persistent,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 非同步 I/O
# ═══════════════════════════════════════════════════════════════════════════════


def _save_single_image(
        data: np.ndarray,
        out_spacing: np.ndarray,
        filename: str,
        data_root_dir: str,
        logger: logging.Logger,
) -> None:
    """單一影像的磁碟寫入（在背景執行緒執行）。"""
    try:
        out_affine = np.eye(4)
        for i in range(3):
            out_affine[i, i] = out_spacing[i]
        output_path = os.path.join(data_root_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine=out_affine), output_path)
        logger.info(f"Saved {output_path}.")
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")


def save_images_async(
        executor: ProcessPoolExecutor,
        datas: np.ndarray,
        out_spacings: np.ndarray,
        output_names: list,
        data_root_dir: str,
        logger: logging.Logger,
) -> list[Future]:
    """
    非同步批次寫入影像。

    問題: 原始碼 save_images 為同步寫入，GPU 需等待磁碟 I/O 完成。
    解法: 使用 ThreadPoolExecutor 讓 I/O 在背景執行緒完成。
    """
    futures = []
    for data, out_spacing, filename in zip(datas, out_spacings, output_names):
        fut = executor.submit(
            _save_single_image,
            data,
            out_spacing,
            filename,
            data_root_dir,
            logger,
        )
        futures.append(fut)
    return futures


# ═══════════════════════════════════════════════════════════════════════════════
# UNet 呼叫封裝
# ═══════════════════════════════════════════════════════════════════════════════


def _call_unet(
        unet: torch.nn.Module,
        x: torch.Tensor,
        t: float,
        spacing_tensor: torch.Tensor,
        include_body_region: bool,
        include_modality: bool,
        top_region_index_tensor: Optional[torch.Tensor],
        bottom_region_index_tensor: Optional[torch.Tensor],
        modality_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    純函式形式呼叫 UNet。

    問題: 原始碼在每個 timestep 的迴圈體內定義 model_warpper（拼字錯誤），
          導致每步重複建立閉包，且閉包捕捉外部可變引用有潛在風險。
    解法: 抽為獨立函式，所有依賴透過參數顯式傳入。
    """
    unet_inputs = {
        "x": x,
        "timesteps": torch.Tensor((t,)).repeat(x.shape[0]).to(x.device),
        "spacing_tensor": spacing_tensor,
    }
    if include_body_region:
        unet_inputs["top_region_index_tensor"] = top_region_index_tensor
        unet_inputs["bottom_region_index_tensor"] = bottom_region_index_tensor
    if include_modality:
        unet_inputs["class_labels"] = modality_tensor
    return unet(**unet_inputs)


# ═══════════════════════════════════════════════════════════════════════════════
# 主推理迴圈
# ═══════════════════════════════════════════════════════════════════════════════


def run_inference(
        data_root_dir: str,
        unet: torch.nn.Module,
        autoencoder: torch.nn.Module,
        data_loader: DataLoader,
        src_shift_factor: torch.Tensor,
        src_scale_factor: torch.Tensor,
        tar_shift_factor: torch.Tensor,
        tar_scale_factor: torch.Tensor,
        noise_scheduler: RFlowScheduler,
        inferer: SlidingWindowInferer,
        logger: logging.Logger,
        include_body_region: bool,
        include_modality: bool,
        device: torch.device,
        io_executor: ProcessPoolExecutor,
        cleanup_interval: int = 20,
) -> None:
    """
    執行擴散模型推理並儲存結果。

    修正:
      1. 回傳型別 np.ndarray → None（原始碼無 return 但標註 np.ndarray）
      2. with A and B → with A, B（致命語法錯誤）
      3. SlidingWindowInferer 移至迴圈外 + cache_roi_weight_map=True
      4. model_wrapper 改為 batch 迴圈內建立一次薄封裝（呼叫 _call_unet）
      5. 移除未使用的 tar_images 載入
      6. non_blocking=True 搭配 pin_memory
      7. .cpu().detach() → .float().cpu()（inference_mode 下 detach 多餘）
      8. 非同步 I/O 寫入
      9. 定期清理 Future 與 CUDA cache
    """
    unet.eval()
    autoencoder.eval()

    # 修正: 原始碼在每個 batch 的 autocast 區塊內重建 SlidingWindowInferer
    # 移至迴圈外並啟用權重圖快取，避免重複計算高斯權重

    pending_futures: list[Future] = []
    batch_count = 0

    for eval_data in tqdm(
            data_loader, desc=f"Running inference on device {device}"
    ):
        batch_count += 1

        # 修正: 加入 non_blocking=True 搭配 DataLoader 的 pin_memory=True
        src_images = eval_data["src_image"].to(device, non_blocking=True)
        spacing_tensor = eval_data["spacing"].to(device, non_blocking=True)

        # 修正: 移除原始碼中載入但未使用的 tar_images，節省 GPU 記憶體
        # 原始碼: tar_images = eval_data["tar_image"].to(device)

        # Normalize
        src_images = (src_images - src_shift_factor) * src_scale_factor

        top_region_index_tensor = None
        bottom_region_index_tensor = None
        modality_tensor = None

        if include_body_region:
            top_region_index_tensor = eval_data["top_region_index"].to(
                device, non_blocking=True
            )
            bottom_region_index_tensor = eval_data["bottom_region_index"].to(
                device, non_blocking=True
            )
        if include_modality:
            modality_tensor = eval_data["modality"].to(
                device, non_blocking=True
            )

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat(
            (all_timesteps[1:], torch.tensor([0.0], dtype=all_timesteps.dtype))
        )

        mu_t = src_images

        # 修正: batch 迴圈內建立一次薄封裝，而非在 timestep 迴圈內重複定義
        # 透過 _call_unet 純函式避免閉包捕捉可變引用問題
        def model_wrapper(t, x):
            return _call_unet(
                unet,
                x,
                t,
                spacing_tensor,
                include_body_region,
                include_modality,
                top_region_index_tensor,
                bottom_region_index_tensor,
                modality_tensor,
            )

        # ══ 修正致命錯誤 ══
        # 原始碼: with torch.inference_mode() and torch.autocast(...)
        # Python 的 `and` 運算子對兩個 context manager 求值後只回傳右側，
        # 導致 inference_mode 完全不生效。
        # 修正: 使用逗號分隔，兩者都正確進入 context。
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=True, dtype=torch.float32
        ):
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                mu_t, _ = noise_scheduler.step(model_wrapper, t, mu_t, next_t)

            # Un-normalize
            mu_t = mu_t * (1.0 / tar_scale_factor) + tar_shift_factor

            # Decode latent → image
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=True, dtype=torch.float32
        ):
            predict_images = dynamic_infer(inferer, autoencoder.decode, mu_t)

        # ── Post-process on CPU ──
        # 修正: inference_mode 下不需 .detach()，直接 .float().cpu()
        datas = predict_images.squeeze(1).float().cpu().numpy()
        out_spacings = eval_data["spacing"].cpu().numpy()

        # 反正規化: [b_min, b_max] → [a_min, a_max]
        datas = (
                (datas - INTENSITY_B_MIN)
                / (INTENSITY_B_MAX - INTENSITY_B_MIN)
                * (INTENSITY_A_MAX - INTENSITY_A_MIN)
                + INTENSITY_A_MIN
        )
        datas = np.int16(np.clip(datas, INTENSITY_A_MIN, INTENSITY_A_MAX))

        # 修正: 非同步寫入取代同步 save_images
        futures = save_images_async(
            io_executor,
            datas,
            out_spacings,
            eval_data["out_name"],
            data_root_dir,
            logger,
        )
        pending_futures.extend(futures)

        # 主動釋放 GPU tensor
        del src_images, mu_t, predict_images
        if include_body_region:
            del top_region_index_tensor, bottom_region_index_tensor
        if include_modality:
            del modality_tensor

        # 定期清理: 等待累積的 I/O 完成 + 釋放 CUDA cache
        if batch_count % cleanup_interval == 0:
            for fut in pending_futures:
                fut.result()  # 同時檢查是否有寫入錯誤
            pending_futures.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 等待所有殘餘的非同步 I/O 完成
    for fut in pending_futures:
        fut.result()
    pending_futures.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════


@torch.inference_mode()
def diff_model_infer(
        env_config_path: str,
        model_config_path: str,
        model_def_path: str,
        num_gpus: int,
) -> None:
    """
    擴散模型推理主函式。

    Args:
        env_config_path: 環境設定檔路徑。
        model_config_path: 模型設定檔路徑。
        model_def_path: 模型定義檔路徑。
        num_gpus: 推理使用的 GPU 數量。
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    if "autoencoder_tp_num_splits" in args.diffusion_unet_inference:
        args.autoencoder_def["num_splits"] = args.diffusion_unet_inference[
            "autoencoder_tp_num_splits"
        ]
    args.autoencoder_def["save_mem"] = False
    args.autoencoder_def["norm_float16"] = False
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")

    # ── 隨機種子 ──
    raw_seed = args.diffusion_unet_inference.get("random_seed", None)
    if raw_seed is not None:
        raw_seed += local_rank
    random_seed = set_random_seed(raw_seed)
    logger.info(
        f"Using {device} of {world_size} with random seed: {random_seed}"
    )

    # ── 載入模型 ──
    autoencoder, unet, src_shift_factor, src_scale_factor, tar_shift_factor, tar_scale_factor = load_models(
        args, device, logger
    )
    unet = compile_unet_model(unet, logger)
    autoencoder = compile_autoencoder_model(autoencoder, args.diffusion_unet_inference["slide_window_size"], device)

    # ── Noise scheduler ──
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.step = MethodType(euler_step, noise_scheduler) # Option: euler_step, midpoint_step, rk4_step, rk5_step
    noise_scheduler.set_timesteps(
        num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
        input_img_size_numel=torch.prod(torch.tensor(src_scale_factor.shape[2:])),
    )

    # ── 條件輸入判斷 ──
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    # ── 載入檔案清單 ──
    # 修正: 移除原始碼中載入但未使用的 filenames_test
    filenames_train = load_filenames(args.json_data_list, mode="training")
    filenames_val = load_filenames(args.json_data_list, mode="validation")

    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")
        logger.info(f"num_files_val: {len(filenames_val)}")

    train_files = prepare_file_list(
        filenames_train,
        args.embedding_base_dir,
        "training",
        include_body_region,
        include_modality,
    )
    val_files = prepare_file_list(
        filenames_val,
        args.embedding_base_dir,
        "validation",
        include_body_region,
        include_modality,
    )

    # ── 分散式資料分割 ──
    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files,
            shuffle=False,
            num_partitions=world_size,
            even_divisible=False,
        )[local_rank]
        val_files = partition_dataset(
            data=val_files,
            shuffle=False,
            num_partitions=world_size,
            even_divisible=False,
        )[local_rank]

    # ── DataLoader ──
    train_loader = prepare_data(
        train_files,
        cache_rate=args.diffusion_unet_inference["cache_rate"],
        num_workers=args.diffusion_unet_inference["num_workers"],
        batch_size=args.diffusion_unet_inference["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping,
    )
    val_loader = prepare_data(
        val_files,
        cache_rate=args.diffusion_unet_inference["cache_rate"],
        num_workers=args.diffusion_unet_inference["num_workers"],
        batch_size=args.diffusion_unet_inference["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # ── 非同步 I/O 執行緒池 ──
    io_executor = ProcessPoolExecutor(max_workers=8)
    cleanup_interval = 50

    try:
        # 修正: 原始碼 data = run_inference(...) 但函式無回傳值
        for mode, loader in [("training", train_loader), ("validation", val_loader)]:
            save_dir = os.path.join(args.output_dir, timestamp, mode)
            os.makedirs(save_dir, exist_ok=True)

            inferer = SlidingWindowInferer(
                roi_size=args.diffusion_unet_inference["slide_window_size"],
                sw_batch_size=args.diffusion_unet_inference["sw_batch_size"],
                progress=False,
                mode="gaussian",
                overlap=0.5,
                sw_device=device,
                device=device,
                cache_roi_weight_map=True,
            )

            run_inference(
                data_root_dir=save_dir,
                unet=unet,
                autoencoder=autoencoder,
                data_loader=loader,
                src_shift_factor=src_shift_factor,
                src_scale_factor=src_scale_factor,
                tar_shift_factor=tar_shift_factor,
                tar_scale_factor=tar_scale_factor,
                noise_scheduler=noise_scheduler,
                inferer=inferer,
                logger=logger,
                include_body_region=include_body_region,
                include_modality=include_modality,
                device=device,
                io_executor=io_executor,
                cleanup_interval=cleanup_interval
            )
    finally:
        # 確保所有 I/O 完成後才關閉執行緒池
        io_executor.shutdown(wait=True)

    # ── 分散式清理 ──
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument(
        "-e", "--env_config", type=str, required=True,
        help="Path to the environment configuration file",
    )
    parser.add_argument(
        "-c", "--model_config", type=str, required=True,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "-t", "--model_def", type=str, required=True,
        help="Path to the model definition file",
    )
    parser.add_argument(
        "-g", "--num_gpus", type=int, default=1,
        help="Number of GPUs to use for inference",
    )
    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def, args.num_gpus)

