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
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

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


# ═══════════════════════════════════════════════════════════════════════════════
# Transform 建立
# ═══════════════════════════════════════════════════════════════════════════════


def create_data_transforms(modality: str = "unknown") -> Compose:
    """
    建立統一的資料前處理 pipeline，所有資料共用同一個 modality。

    與原始碼差異：
      - 原始碼在 process_single_item 中每個 item 各建立一次 transforms
      - 現在統一建立一次，供 CacheDataset 使用
      - 保留 affine metadata（LoadImaged 預設行為）

    Args:
        modality: 資料模態，用於選擇對應的強度變換。

    Returns:
        Compose: 組合後的 MONAI transforms。
    """
    keys = ["src_image", "tar_image"]

    # 正規化 modality 字串
    if "mri" in modality:
        modality = "mri"
    if "ct" in modality:
        modality = "ct"

    if modality in SUPPORT_MODALITIES:
        intensity_transforms = define_fixed_intensity_transform(
            modality=modality, image_keys=keys
        )
    else:
        intensity_transforms = []

    base_transforms = [
        monai.transforms.LoadImaged(keys=keys),
        monai.transforms.EnsureChannelFirstd(keys=keys),
        monai.transforms.Orientationd(keys=keys, axcodes="RAS"),
        monai.transforms.EnsureTyped(keys=keys, dtype=torch.float32),
    ]

    return Compose(base_transforms + intensity_transforms)


# ═══════════════════════════════════════════════════════════════════════════════
# 檔案清單建構
# ═══════════════════════════════════════════════════════════════════════════════


def build_file_list(
        json_data: dict,
        data_type: str,
        data_base_dir: str,
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
                "src_image": os.path.join(
                    data_base_dir, data_type, item["src_image"]
                ),
                "tar_image": os.path.join(
                    data_base_dir, data_type, item["tar_image"]
                ),
                "src_out_path": src_out,
                "tar_out_path": tar_out,
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


# ═══════════════════════════════════════════════════════════════════════════════
# 非同步 I/O（與原始碼一致）
# ═══════════════════════════════════════════════════════════════════════════════


def save_embedding_async(
        executor: ThreadPoolExecutor,
        out_nda: np.ndarray,
        affine: np.ndarray,
        out_path: str,
        logger: logging.Logger,
) -> Future:
    """
    使用線程池非同步保存 NIfTI 檔案，讓 GPU 不需等待磁碟 I/O。
    """

    def _save():
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            out_img = nib.Nifti1Image(np.float32(out_nda), affine=affine)
            nib.save(out_img, out_path)
            logger.info(f"Saved {out_path}.")
        except Exception as e:
            logger.error(f"Error saving {out_path}: {e}")

    return executor.submit(_save)


# ═══════════════════════════════════════════════════════════════════════════════
# 單一影像 Encode（從 batch 資料中處理）
# ═══════════════════════════════════════════════════════════════════════════════


def encode_and_save(
        args: argparse.Namespace,
        image_tensor: torch.Tensor,
        affine: np.ndarray,
        out_path: str,
        autoencoder: torch.nn.Module,
        device: torch.device,
        inferer: SlidingWindowInferer,
        io_executor: ThreadPoolExecutor,
        logger: logging.Logger,
) -> Optional[Future]:
    """
    對單一影像 tensor 執行 autoencoder encode 並非同步寫入磁碟。

    與原始碼 process_single_item 差異：
      - 不再負責 transform（已由 DataLoader 完成）
      - 接收已前處理的 tensor，只做 encode + save
      - 個別檔案層級的 skip 檢查仍保留

    Args:
        image_tensor: 前處理後的影像 tensor [C, X, Y, Z]。
        affine: 影像的 affine matrix。
        out_path: 輸出 NIfTI 路徑。
        autoencoder: 已載入權重的 autoencoder。
        device: CUDA 裝置。
        inferer: 預建的 SlidingWindowInferer。
        io_executor: 非同步 I/O 線程池。
        logger: Logger。

    Returns:
        非同步 I/O Future，若跳過則回傳 None。
    """
    # 個別檔案可能其中一個已存在（src 存在但 tar 不存在的情況）
    if os.path.isfile(out_path):
        logger.info(f"Skipping {out_path}, already exists.")
        return None

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # image_tensor 來自 DataLoader: [C, X, Y, Z]
            # autoencoder 需要 [B, C, X, Y, Z]
            pt_nda = image_tensor.float().to(device).unsqueeze(0)

            sw_size = args.transform_to_laten["slide_window_size"]
            if (
                    pt_nda.shape[2] <= sw_size[0]
                    or pt_nda.shape[3] <= sw_size[1]
                    or pt_nda.shape[4] <= sw_size[2]
            ):
                z_mu, _ = autoencoder.encode(pt_nda)
            else:
                z_mu, _ = dynamic_infer(inferer, autoencoder.encode, pt_nda)

            logger.info(f"z_mu: {z_mu.size()}, {z_mu.dtype}")

        # 立即移到 CPU 釋放 GPU 記憶體
        out_nda = z_mu.float().squeeze().cpu().numpy().transpose(1, 2, 3, 0)

        # 主動釋放 GPU tensor
        del pt_nda, z_mu

        # 非同步寫入磁碟
        return save_embedding_async(io_executor, out_nda, affine, out_path, logger)

    except Exception as e:
        logger.error(f"Error encoding for {out_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 批次處理（DataLoader batch → encode + save）
# ═══════════════════════════════════════════════════════════════════════════════


def process_batch(
        args: argparse.Namespace,
        batch_data: dict,
        autoencoder: torch.nn.Module,
        device: torch.device,
        inferer: SlidingWindowInferer,
        io_executor: ThreadPoolExecutor,
        logger: logging.Logger,
) -> list[Future]:
    """
    處理 DataLoader 產出的一個 batch。

    由於醫學影像尺寸可能不一致，batch_size 通常為 1。
    對 batch 中每個 sample 的 src_image 和 tar_image 分別 encode。

    Args:
        batch_data: DataLoader 產出的 dict，包含 tensor 和 metadata。
        autoencoder: 已載入權重的 autoencoder。
        device: CUDA 裝置。
        inferer: 預建的 SlidingWindowInferer。
        io_executor: 非同步 I/O 線程池。
        logger: Logger。

    Returns:
        非同步 I/O Future 列表。
    """
    futures = []
    batch_size = batch_data["src_image"].shape[0]

    for i in range(batch_size):
        for img_key, out_key in [
            ("src_image", "src_out_path"),
            ("tar_image", "tar_out_path"),
        ]:
            image_tensor = batch_data[img_key][i]  # [C, X, Y, Z]

            # 取得 affine：LoadImaged 會將 affine 存在 meta 中
            # CacheDataset + DataLoader 會保留 MetaTensor 的 meta
            if hasattr(image_tensor, "meta") and "affine" in image_tensor.meta:
                affine = image_tensor.meta["affine"].numpy()
            else:
                # fallback: 單位矩陣
                logger.warning(
                    f"No affine found for {img_key}[{i}], using identity."
                )
                affine = np.eye(4)

            # DataLoader batch 中 out_path 是 list of str
            out_path = batch_data[out_key][i]

            fut = encode_and_save(
                args,
                image_tensor,
                affine,
                out_path,
                autoencoder,
                device,
                inferer,
                io_executor,
                logger,
            )
            if fut is not None:
                futures.append(fut)

    return futures


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
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path, map_location=device, weights_only=False
    )
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    autoencoder = torch.compile(
        autoencoder,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )
    logger.info(f"Autoencoder loaded from {args.trained_autoencoder_path}")

    # ── Ensure output dirs exist ──
    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    # ── Load JSON data list ──
    with open(args.json_data_list, "r") as file:
        json_data = json.load(file)

    # ── 讀取統一 modality（所有資料共用）──
    # 從 config 中取得，若未指定則 fallback 為 "unknown"
    global_modality = getattr(args.transform_to_laten, "modality", "unknown")
    logger.info(f"Modality for transforms: {global_modality}")

    # ── 統一建立 transforms（一次建立，所有資料共用）──
    data_transforms = create_data_transforms(modality=global_modality)

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
    io_executor = ThreadPoolExecutor(max_workers=12)

    # ── DataLoader 設定（參考 diff_model_infer.py）──
    cache_rate = args.transform_to_laten["cache_rate"]
    dl_num_workers = args.transform_to_laten["num_workers"]
    batch_size = 1  # 醫學影像尺寸不一致，batch_size=1 避免 collation 錯誤
    cleanup_interval = 50

    try:
        for data_type in ["training", "validation", "test"]:
            os.makedirs(
                os.path.join(args.embedding_base_dir, data_type), exist_ok=True
            )

            # ── 優化: 預先過濾已存在的 embedding ──
            file_list, skipped = build_file_list(
                json_data,
                data_type,
                args.data_base_dir,
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
                    disable=(local_rank != 0),
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