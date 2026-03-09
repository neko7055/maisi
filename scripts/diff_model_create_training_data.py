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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future
from functools import partial

import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from monai.data import DataLoader, partition_dataset, CacheDataset
from monai.transforms import Compose
from monai.inferers.inferer import SlidingWindowInferer

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .transforms import define_fixed_intensity_transform, SUPPORT_MODALITIES
from .utils import define_instance, dynamic_infer

# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

def compile_model(model, shape, device):
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
            example_inputs = torch.randn(1, 18, *shape, device=device)
            _ = model.encode(example_inputs)
    return model

def expand_first_conv_input_channels(model: torch.nn.Module, k: int):
    """
    将模型第一个卷积层 (self.conv) 的 input channel 复制 k 倍。
    新权重通过重复原始权重 k 次来初始化。
    """
    old_conv = model.conv
    in_channels = old_conv.in_channels * k
    out_channels = old_conv.out_channels

    # 判断卷积类型（2D 或 3D）
    if isinstance(old_conv, torch.nn.Conv3d):
        new_conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )
    elif isinstance(old_conv, torch.nn.Conv2d):
        new_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )
    else:
        raise TypeError(f"Unsupported conv type: {type(old_conv)}")

    # 将原始权重沿 input channel 维度 (dim=1) 重复 k 次，并除以 k 保持输出尺度一致
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.repeat(1, k, *([1] * (old_conv.weight.dim() - 2))) / k)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.conv = new_conv

def load_model(args: argparse.Namespace, device, logger) -> torch.nn.Module:
    # Load model to CPU first, let Accelerate handle movement

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path, map_location=device, weights_only=False
    )
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    expand_first_conv_input_channels(autoencoder.encoder.blocks[0].conv, k=18)
    my_checkpoint_path = args.trained_autoencoder_path.replace(".pt", f"_my.pt")
    if os.path.exists(my_checkpoint_path):
        checkpoint_autoencoder = torch.load(
            my_checkpoint_path, map_location=device, weights_only=True
        )
        logger.info(f"Loading expanded model weights from {my_checkpoint_path}.")
        autoencoder.load_state_dict(checkpoint_autoencoder)
    return autoencoder.to(device)

# ═══════════════════════════════════════════════════════════════════════════════
# Transform 建立
# ═══════════════════════════════════════════════════════════════════════════════

def create_data_transforms(data_type,
                           data_base_dir,
                           embedding_base_dir,
                           modality: str = "unknown") -> Compose:
    keys = ["src_image", "tar_image"]

    # 正規化 modality 字串
    if "mri" in modality:
        modality = "mri"
    if "ct" in modality:
        modality = "ct"

    # if modality in SUPPORT_MODALITIES:
    #     intensity_transforms = define_fixed_intensity_transform(
    #         modality=modality, image_keys=keys
    #     )
    # else:
    #     intensity_transforms = []

    def _build_out_path(filename_str: str):
        """純字串操作：將原始檔名轉為 embedding 輸出路徑。"""
        out_base = filename_str.replace(".gz", "").replace(".nii", "")
        return os.path.join(embedding_base_dir, data_type, out_base + "_emb.nii.gz")

    def _build_full_path(filename_str: str):
        """純字串操作：將相對路徑拼接為完整的檔案路徑。"""
        return os.path.join(data_base_dir, data_type, filename_str)

    def _load_nifti(filepath: str):
        """
        用 nibabel 載入 NIfTI，回傳純 torch.Tensor（CPU, float32）。
        自動處理 channel dimension：
          - (X, Y, Z)    → (1, X, Y, Z)
          - (X, Y, Z, C) → (C, X, Y, Z)
        輸出已為 RAS 方向（nibabel 預設以 header 中的方向載入，
        若需強制 RAS，使用 nibabel.as_closest_canonical）。
        """
        img = nib.load(filepath)
        # 強制轉為 RAS（closest canonical）
        img = nib.as_closest_canonical(img)
        return img


    def _read_affine(img) -> np.ndarray:
        """用 nibabel 直接從 NIfTI 檔案讀取 affine matrix，不使用 MetaTensor。"""
        try:
            return img.affine.astype(np.float64)
        except Exception as e:
            return np.eye(4).astype(np.float64)

    def _nifti_as_tensor(img)-> torch.Tensor:
        """
        用 nibabel 載入 NIfTI，回傳純 torch.Tensor（CPU, float32）。
        自動處理 channel dimension：
          - (X, Y, Z)    → (1, X, Y, Z)
          - (X, Y, Z, C) → (C, X, Y, Z)
        輸出已為 RAS 方向（nibabel 預設以 header 中的方向載入，
        若需強制 RAS，使用 nibabel.as_closest_canonical）。
        """
        data = img.get_fdata(caching='unchanged', dtype=np.float32)
        if data.ndim == 3:
            # (X, Y, Z) → (1, X, Y, Z)
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            # (X, Y, Z, C) → (C, X, Y, Z)
            data = data.transpose(3, 0, 1, 2)

        return torch.from_numpy(data.copy())

    base_transforms = [
        monai.transforms.Lambdad(
            keys=keys,
            func=_build_out_path,
            track_meta=False,
            overwrite=["src_out_path", "tar_out_path"],
        ),
        monai.transforms.Lambdad(
            keys=keys,
            func=_build_full_path,
            track_meta=False,
            overwrite=True,
        ),
        monai.transforms.Lambdad(
            keys=keys,
            func=_load_nifti,
            track_meta=False,
            overwrite=True,
        ),
        monai.transforms.Lambdad(
            keys=keys,
            func=_read_affine,
            track_meta=False,
            overwrite=["src_affine", "tar_affine"],
        ),
        monai.transforms.Lambdad(
            keys=keys,
            func=_nifti_as_tensor,
            track_meta=False,
            overwrite=True,
        ),
    ]

    return Compose(base_transforms)

# ═══════════════════════════════════════════════════════════════════════════════
# 檔案清單建構
# ═══════════════════════════════════════════════════════════════════════════════

def build_file_list(
        json_data: dict,
        data_type: str,
        embedding_base_dir: str,
        logger: logging.Logger,
) -> tuple[list[dict], int]:
    """
    構建檔案列表，預先過濾已存在的 embedding，避免無效 I/O。

    Returns:
        (file_list, skipped_count)
    """
    files_raw = json_data[data_type]
    file_list = []
    skipped = 0

    for item in files_raw:
        src_out_base = item["src_image"].replace(".gz", "").replace(".nii", "")
        tar_out_base = item["tar_image"].replace(".gz", "").replace(".nii", "")

        src_out = os.path.join(
            embedding_base_dir, data_type, src_out_base + "_emb.nii.gz"
        )
        tar_out = os.path.join(
            embedding_base_dir, data_type, tar_out_base + "_emb.nii.gz"
        )

        # 兩個輸出都已存在則跳過
        if os.path.isfile(src_out) and os.path.isfile(tar_out):
            skipped += 1
            continue

        file_list.append(
            {
                "src_image": item["src_image"],
                "tar_image": item["tar_image"],
            }
        )

    return file_list, skipped

# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader 建立（參考 diff_model_infer.py 的 prepare_data）
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
        file_list: list[dict],
        transforms: Compose,
        cache_rate: float,
        num_workers: int = 2,
        batch_size: int = 1,
) -> DataLoader:
    """
    使用 CacheDataset + DataLoader 載入與前處理資料。

    參考 diff_model_infer.py 和 diff_model_train_with_accelerate.py 的做法：
      - CacheDataset 快取前處理結果，避免重複的 LoadImage + Orientation 計算
      - DataLoader 的 pin_memory=True 加速 CPU→GPU 傳輸
      - persistent_workers=True 避免每個 epoch 重啟 worker

    與原始碼差異：
      - 原始碼逐一 item 呼叫 data_transforms()，無法利用多 worker 平行載入
      - 現在使用 CacheDataset 快取 + DataLoader 多 worker 預取

    注意：
      - batch_size=1 是因為醫學影像尺寸可能不一致（無法 stack）
      - 若確定所有影像尺寸一致，可增大 batch_size

    Args:
        file_list: 包含檔案路徑的 dict 列表。
        transforms: 預建立的 Compose transforms。
        cache_rate: CacheDataset 快取比率（0.0~1.0）。
        num_workers: DataLoader worker 數量。
        batch_size: 每批次資料量。

    Returns:
        DataLoader 實例。
    """
    dataset = CacheDataset(
        data=file_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    use_persistent = num_workers > 0
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

def img_save(out_nda, out_path, out_affine, logger):
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_img = nib.Nifti1Image(np.float32(out_nda), affine=out_affine)
        nib.save(out_img, out_path)
        logger.info(f"Saved {out_path}.")
    except Exception as e:
        logger.error(f"Error saving {out_path}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 批次處理（DataLoader batch → encode + save）
# ═══════════════════════════════════════════════════════════════════════════════

def build_ct_channel_params(A, B, levels=None, ks=None):
    """预计算所有 (a, b) 参数张量，只需调用一次。"""
    if ks is None:
        ks = [5]
    if levels is None:
        levels = [1, 3, 5]

    a_list, b_list = [], []
    for k_val in ks:
        for l in levels:
            step = (B - A) / l
            for i in range(l):
                s = A + step * i
                e = s + step
                a_list.append(k_val / (e - s))
                b_list.append(-k_val * (e + s) / (2.0 * (e - s)))

    # (N, 1, 1, 1) 用于与 (1, H, W, D) 广播
    a_t = torch.tensor(a_list, dtype=torch.float64).reshape(-1, 1, 1, 1)
    b_t = torch.tensor(b_list, dtype=torch.float64).reshape(-1, 1, 1, 1)
    return a_t, b_t

def apply_ct_channel_extend(x, a_t, b_t):
    """利用预计算的参数张量，广播计算 sigmoid 扩展通道。"""
    from monai.data import MetaTensor

    meta = x.meta if isinstance(x, MetaTensor) else None

    # 参数移到与 x 相同的设备
    device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
    a_t = a_t.to(device=device)
    b_t = b_t.to(device=device)

    # float64 计算避免溢出，广播: (N,1,1,1) * (1,H,W,D) -> (N,H,W,D)
    x_64 = x.to(dtype=torch.float64)
    z = -(a_t * x_64 + b_t)
    result = torch.sigmoid(z).to(dtype=x.dtype)

    if meta is not None:
        result = MetaTensor(result, meta=meta)

    return result

def process_batch(
        args: argparse.Namespace,
        batch_data: dict,
        autoencoder: torch.nn.Module,
        device: torch.device,
        inferer: SlidingWindowInferer,
        io_executor: ProcessPoolExecutor,
        logger: logging.Logger,
) -> list[Future]:
    a_t, b_t = build_ct_channel_params(A=-200, B=700, levels=[1, 3, 5], ks=[4, 5])
    et_channel = lambda x: apply_ct_channel_extend(x, a_t, b_t)
    infer_fn = lambda input: autoencoder.encode(et_channel(input))
    all_futures = []
    for key in ("src", "tar"):
        pt_nda = batch_data[f"{key}_image"].to(device, non_blocking=True) # size: [B, C, X, Y, Z]
        with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.float32):
            z_mu, z_log_var = dynamic_infer(inferer, infer_fn, pt_nda)
            logger.info(f"z_mu: {z_mu.size()}, {z_mu.dtype}")
        out_ndas = z_mu.float().cpu().numpy().transpose(0, 2, 3, 4, 1).copy() # size: [B, C, X, Y, Z] -> [B, X, Y, Z, C]
        # 主動釋放 GPU tensor
        del pt_nda, z_mu, z_log_var
        futures = [
            io_executor.submit(partial(img_save, logger=logger), a, b, c)
            for a, b, c in zip(out_ndas, batch_data[f"{key}_out_path"], batch_data[f"{key}_affine"].numpy().copy())
        ]
        all_futures.extend(futures)
    return all_futures


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════


@torch.inference_mode()
def diff_model_create_training_data(
        env_config_path: str,
        model_config_path: str,
        model_def_path: str,
        num_gpus: int,
) -> None:
    """
    Create training data for the diffusion model.

    重構重點（相對於原始碼）：
      1. CacheDataset + DataLoader：參考 diff_model_infer.py 和
         diff_model_train_with_accelerate.py，使用 CacheDataset 快取前處理結果、
         DataLoader 多 worker 平行預取，取代逐一 item 的序列處理。
      2. 統一 modality transform：所有資料共用同一個 modality 的 transform pipeline，
         只建立一次 Compose，供 CacheDataset 使用。
      3. 保留原有優化：預過濾已存在 embedding、非同步 I/O、partition_dataset、
         定期 CUDA cache 清理。
      4. pin_memory + non_blocking：加速 CPU→GPU 資料傳輸。
      5. persistent_workers：避免每輪重啟 worker 進程。
    """
    # ── Load config ──
    args = load_config(env_config_path, model_config_path, model_def_path)
    if "autoencoder_tp_num_splits" in args.transform_to_laten.keys():
        args.autoencoder_def["num_splits"] = args.transform_to_laten[
            "autoencoder_tp_num_splits"
        ]
    args.autoencoder_def["save_mem"] = False
    args.autoencoder_def["norm_float16"] = False
    # ── Initialize distributed ──
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data", rk_filter=False)
    logger.info(f"Using device {device}, rank {local_rank}/{world_size}")
    logger.info(
        f"Slide Window Size: {args.transform_to_laten['slide_window_size']}"
    )
    logger.info(f"SW Batch Size: {args.transform_to_laten['sw_batch_size']}")
    logger.info(
        f"autoencoder_tp_num_splits: {args.transform_to_laten['autoencoder_tp_num_splits']}"
    )

    # ── Load autoencoder ──
    autoencoder = load_model(args, device, logger)
    autoencoder.eval()
    autoencoder = compile_model(autoencoder, args.transform_to_laten['slide_window_size'], device)
    logger.info(f"Autoencoder loaded from {args.trained_autoencoder_path}")

    # ── Ensure output dirs exist ──
    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    # ── Load JSON data list ──
    with open(args.json_data_list, "r") as file:
        json_data = json.load(file)

    # ── 讀取統一 modality（所有資料共用）──
    # 從 config 中取得，若未指定則 fallback 為 "unknown"
    global_modality = args.transform_to_laten.get("modality", "unknown")
    logger.info(f"Modality for transforms: {global_modality}")



    # ── 優化: 預建 SlidingWindowInferer，避免每個檔案重建 ──
    inferer = SlidingWindowInferer(
        roi_size=args.transform_to_laten["slide_window_size"],
        sw_batch_size=args.transform_to_laten["sw_batch_size"],
        progress=False,
        mode="gaussian",
        overlap=0.5,
        sw_device=device,
        device=device,
        cache_roi_weight_map=True,
    )

    # ── 優化: ThreadPoolExecutor 非同步 I/O ──
    io_executor = ProcessPoolExecutor(max_workers=8)

    # ── DataLoader 設定（參考 diff_model_infer.py）──
    cache_rate = args.transform_to_laten["cache_rate"]
    dl_num_workers = args.transform_to_laten["num_workers"]
    batch_size = args.transform_to_laten["batch_size"]
    cleanup_interval = 50

    try:
        for data_type in ["training", "validation", "test"]:
            os.makedirs(
                os.path.join(args.embedding_base_dir, data_type), exist_ok=True
            )

            # ── 統一建立 transforms（一次建立，所有資料共用）──
            data_transforms = create_data_transforms(data_type=data_type,
                                                     data_base_dir=args.data_base_dir,
                                                     embedding_base_dir=args.embedding_base_dir,
                                                     modality=global_modality)

            # ── 優化: 預先過濾已存在的 embedding ──
            file_list, skipped = build_file_list(
                json_data,
                data_type,
                args.embedding_base_dir,
                logger,
            )
            logger.info(
                f"[{data_type}] Total: {len(json_data[data_type])}, "
                f"Skipped (already exist): {skipped}, "
                f"To process: {len(file_list)}"
            )

            if len(file_list) == 0:
                logger.info(
                    f"[{data_type}] All embeddings already exist. Skipping."
                )
                continue

            # ── 優化: partition_dataset 分散式分配 ──
            if dist.is_initialized() and world_size > 1:
                file_list = partition_dataset(
                    data=file_list,
                    shuffle=False,
                    num_partitions=world_size,
                    even_divisible=False,
                )[local_rank]
                logger.info(
                    f"[{data_type}] Rank {local_rank} processing {len(file_list)} files."
                )

            # ── 建立 CacheDataset + DataLoader ──
            # 參考 diff_model_infer.py 的 prepare_data：
            #   - CacheDataset 快取前處理，多 worker 平行載入
            #   - pin_memory=True 加速 CPU→GPU
            #   - persistent_workers=True 避免重啟 worker
            #
            # 注意：src_out_path / tar_out_path 是 str，不是影像路徑，
            #   LoadImaged 只會處理 keys=["src_image", "tar_image"]，
            #   其他 key 會被 pass-through 保留在 dict 中。
            data_loader = prepare_data(
                file_list=file_list,
                transforms=data_transforms,
                cache_rate=cache_rate,
                num_workers=dl_num_workers,
                batch_size=batch_size,
            )

            # ── 主處理迴圈（改用 DataLoader 迭代）──
            pending_futures: list[Future] = []
            batch_count = 0

            for batch_data in tqdm(
                    data_loader,
                    desc=f"[Rank {local_rank}] {data_type}",
                    # disable=(local_rank != 0),
            ):
                batch_count += 1

                futures = process_batch(
                    args,
                    batch_data,
                    autoencoder,
                    device,
                    inferer,
                    io_executor,
                    logger,
                )
                pending_futures.extend(futures)

                # 定期清理 GPU 快取並等待累積的 I/O 完成
                if batch_count % cleanup_interval == 0:
                    for fut in pending_futures:
                        fut.result()
                    pending_futures.clear()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 等待本 data_type 所有殘餘 I/O 完成
            for fut in pending_futures:
                fut.result()
            pending_futures.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"[{data_type}] Finished processing on rank {local_rank}."
            )

    finally:
        # 確保所有非同步 I/O 完成
        io_executor.shutdown(wait=True)

    # ── Synchronize & teardown ──
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logger.info("Training data creation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion Model Training Data Creation"
    )
    parser.add_argument(
        "-e",
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "-c",
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "-t",
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "-g",
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )

    args = parser.parse_args()
    diff_model_create_training_data(args.env_config, args.model_config, args.model_def, args.num_gpus)