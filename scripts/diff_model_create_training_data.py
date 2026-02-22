n
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

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from monai.data import partition_dataset
from monai.transforms import Compose
from monai.inferers.inferer import SlidingWindowInferer

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .transforms import define_fixed_intensity_transform, SUPPORT_MODALITIES
from .utils import define_instance, dynamic_infer


def create_transforms(dim: Optional[tuple] = None, modality: str = 'unknown') -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.
        modality (str): Data modality for intensity transforms.

    Returns:
        Compose: Composed MONAI transforms.
    """
    keys = ["src_image", "tar_image"]
    if 'mri' in modality:
        modality = 'mri'
    if 'ct' in modality:
        modality = 'ct'

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

    if dim:
        return Compose(
            base_transforms + intensity_transforms +
            [monai.transforms.Resized(keys=keys, spatial_size=dim, mode="trilinear")]
        )
    else:
        return Compose(base_transforms + intensity_transforms)


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

        src_out = os.path.join(embedding_base_dir, data_type, src_out_base + "_emb.nii.gz")
        tar_out = os.path.join(embedding_base_dir, data_type, tar_out_base + "_emb.nii.gz")

        # 兩個輸出都已存在則跳過
        if os.path.isfile(src_out) and os.path.isfile(tar_out):
            skipped += 1
            continue

        file_list.append({
            "src_image": os.path.join(data_base_dir, data_type, item["src_image"]),
            "tar_image": os.path.join(data_base_dir, data_type, item["tar_image"]),
            "src_out_path": src_out,
            "tar_out_path": tar_out,
            "modality": item.get("modality", "unknown"),
        })

    return file_list, skipped


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


def process_single_item(
    item: dict,
    autoencoder: torch.nn.Module,
    device: torch.device,
    inferer: SlidingWindowInferer,
    io_executor: ThreadPoolExecutor,
    logger: logging.Logger,
) -> list[Future]:
    """
    處理單一資料項（包含 src_image 與 tar_image），
    對每個尚未產生 embedding 的影像執行 autoencoder encode，
    並非同步寫入磁碟。

    Args:
        item: 包含路徑與輸出路徑的 dict
        autoencoder: 已載入權重的 autoencoder model
        device: CUDA 裝置
        inferer: 預建的 SlidingWindowInferer（複用避免重建開銷）
        io_executor: 非同步 I/O 用的 ThreadPoolExecutor
        logger: Logger

    Returns:
        非同步 I/O Future 列表
    """
    futures = []
    modality = item["modality"]
    data_transforms = create_transforms(dim=None, modality=modality)

    try:
        transformed_data = data_transforms({
            "src_image": item["src_image"],
            "tar_image": item["tar_image"],
        })
    except Exception as e:
        logger.error(f"Error loading data - src: {item['src_image']}, tar: {item['tar_image']}: {e}")
        return futures

    for key, out_key in [("src_image", "src_out_path"), ("tar_image", "tar_out_path")]:
        out_path = item[out_key]

        # 個別檢查：可能其中一個已存在
        if os.path.isfile(out_path):
            logger.info(f"Skipping {out_path}, already exists.")
            continue

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # [1, 1, X, Y, Z]
                pt_nda = (
                    torch.from_numpy(transformed_data[key].numpy().squeeze())
                    .float()
                    .to(device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                z_mu, _ = dynamic_infer(inferer, autoencoder.encode, pt_nda)
                logger.info(f"{key}, z_mu: {z_mu.size()}, {z_mu.dtype}")

            # 立即移到 CPU 釋放 GPU 記憶體
            out_nda = z_mu.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            affine = transformed_data[key].meta["affine"].numpy()

            # 主動釋放 GPU tensor
            del pt_nda, z_mu

            # 非同步寫入磁碟
            fut = save_embedding_async(io_executor, out_nda, affine, out_path, logger)
            futures.append(fut)

        except Exception as e:
            logger.error(f"Error processing {item[key]}: {e}")
            continue

    return futures


@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_gpus: int,
) -> None:
    """
    Create training data for the diffusion model.

    Optimizations applied:
      1. Pre-filtering: skip files whose embeddings already exist (resume-friendly)
      2. Reusable SlidingWindowInferer: built once, reused for all files
      3. bfloat16 mixed precision: faster encode with less memory
      4. Async disk I/O: ThreadPoolExecutor overlaps NIfTI writes with GPU compute
      5. partition_dataset: balanced multi-GPU workload distribution (from MONAI)
      6. Periodic CUDA cache clearing: prevents memory fragmentation
    """
    # ── Load config ──
    args = load_config(env_config_path, model_config_path, model_def_path)
    if "autoencoder_tp_num_splits" in args.diffusion_unet_inference.keys():
        args.autoencoder_def["num_splits"] = args.diffusion_unet_inference["autoencoder_tp_num_splits"]

    # ── Initialize distributed ──
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}, rank {local_rank}/{world_size}")

    # ── Load autoencoder ──
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path, map_location=device, weights_only=False
    )
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    logger.info(f"Autoencoder loaded from {args.trained_autoencoder_path}")

    # ── Ensure output dirs exist ──
    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    # ── Load JSON data list ──
    with open(args.json_data_list, "r") as file:
        json_data = json.load(file)

    # ── 優化: 預建 SlidingWindowInferer，避免每個檔案重建 ──
    inferer = SlidingWindowInferer(
        roi_size=args.transform_to_laten["slide_window_size"],
        sw_batch_size=args.transform_to_laten["sw_batch_size"],
        progress=False,
        mode="gaussian",
        overlap=0.5,
        sw_device=device,
        device=device,
    )

    # ── 優化: ThreadPoolExecutor 非同步 I/O ──
    io_executor = ThreadPoolExecutor(max_workers=4)

    try:
        for data_type in ["training", "validation", "test"]:
            os.makedirs(os.path.join(args.embedding_base_dir, data_type), exist_ok=True)

            # ── 優化: 預先過濾已存在的 embedding ──
            file_list, skipped = build_file_list(
                json_data, data_type, args.data_base_dir, args.embedding_base_dir, logger
            )
            logger.info(
                f"[{data_type}] Total: {len(json_data[data_type])}, "
                f"Skipped (already exist): {skipped}, "
                f"To process: {len(file_list)}"
            )

            if len(file_list) == 0:
                logger.info(f"[{data_type}] All embeddings already exist. Skipping.")
                continue

            # ── 優化: partition_dataset 分散式分配（參考 infer & train 的做法）──
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

            # ── 主處理迴圈 ──
            pending_futures: list[Future] = []

            for idx, item in enumerate(tqdm(
                file_list,
                desc=f"[Rank {local_rank}] {data_type}",
                disable=(local_rank != 0),
            )):
                futures = process_single_item(
                    item, autoencoder, device, inferer, io_executor, logger
                )
                pending_futures.extend(futures)

                # 定期清理 GPU 快取並等待累積的 I/O 完成，避免記憶體爆炸
                if (idx + 1) % 50 == 0:
                    # 等待已累積的寫入完成
                    for fut in pending_futures:
                        fut.result()
                    pending_futures.clear()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 等待本 data_type 所有殘餘 I/O 完成
            for fut in pending_futures:
                fut.result()
            pending_futures.clear()

            logger.info(f"[{data_type}] Finished processing on rank {local_rank}.")

    finally:
        # 確保所有非同步 I/O 完成
        io_executor.shutdown(wait=True)

    # ── Synchronize & teardown ──
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logger.info("Training data creation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Data Creation")
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