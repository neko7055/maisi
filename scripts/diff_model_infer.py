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

INTENSITY_A_MIN = -1000
INTENSITY_A_MAX = 1000
INTENSITY_B_MIN = 0
INTENSITY_B_MAX = 1

def set_random_seed(seed: Optional[int]) -> int:
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def _load_json_field(file_path: str, key: str, convert_to_float: bool = True):
    with open(file_path, "r") as f:
        data = json.load(f)
    if convert_to_float:
        return torch.FloatTensor(data[key])
    return data[key]

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
            example_inputs = torch.randn(1, 4, *shape, device=device)
            _ = model.decode(example_inputs)
    return model

def load_autoencoder(args: argparse.Namespace, device, logger) -> torch.nn.Module:
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path, map_location=device, weights_only=False
    )
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    return autoencoder.to(device)

def load_models(
        args: argparse.Namespace,
        device: torch.device,
        logger: logging.Logger,
) -> tuple:
    autoencoder = load_autoencoder(args, device, logger)

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(
        f"{args.model_dir}/{args.model_filename}",
        map_location=device,
        weights_only=False,
    )
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    unet.eval()
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    shift_factor = checkpoint["shift_factor"]
    scale_factor = checkpoint["scale_factor"]

    logger.info(f"shift_factor -> {shift_factor}.")
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, shift_factor, scale_factor

def load_filenames(data_list_path: str, mode: str) -> list:
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

def _save_single_image(
        data: np.ndarray,
        out_spacing: np.ndarray,
        filename: str,
        data_root_dir: str,
        logger: logging.Logger,
) -> None:
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

def _call_unet(
        unet: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        spacing_tensor: torch.Tensor,
        include_body_region: bool,
        include_modality: bool,
        top_region_index_tensor: Optional[torch.Tensor],
        bottom_region_index_tensor: Optional[torch.Tensor],
        modality_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    unet_inputs = {
        "x": x,
        "timesteps": t,
        "spacing_tensor": spacing_tensor,
    }
    if include_body_region:
        unet_inputs["top_region_index_tensor"] = top_region_index_tensor
        unet_inputs["bottom_region_index_tensor"] = bottom_region_index_tensor
    if include_modality:
        unet_inputs["class_labels"] = modality_tensor
    return unet(**unet_inputs)

def run_inference(
        data_root_dir: str,
        unet: torch.nn.Module,
        autoencoder: torch.nn.Module,
        data_loader: DataLoader,
        shift_factor: torch.Tensor,
        scale_factor: torch.Tensor,
        noise_scheduler: RFlowScheduler,
        inferer: SlidingWindowInferer,
        logger: logging.Logger,
        include_body_region: bool,
        include_modality: bool,
        device: torch.device,
        io_executor: ProcessPoolExecutor,
        cleanup_interval: int = 20,
) -> None:
    unet.eval()
    autoencoder.eval()

    pending_futures: list[Future] = []
    batch_count = 0

    for eval_data in tqdm(
            data_loader, desc=f"Running inference on device {device}"
    ):
        batch_count += 1

        # 修正: 加入 non_blocking=True 搭配 DataLoader 的 pin_memory=True
        src_images = eval_data["src_image"].to(device, non_blocking=True)
        tar_images = eval_data["tar_image"].to(device, non_blocking=True)
        spacing_tensor = eval_data["spacing"].to(device, non_blocking=True)

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

        def model_wrapper(t, x):
            return _call_unet(unet,
                              x,
                              torch.Tensor((t,)).repeat(x.shape[0]).to(x.device),
                              spacing_tensor,
                              include_body_region,
                              include_modality,
                              top_region_index_tensor,
                              bottom_region_index_tensor,
                              modality_tensor)  * scale_factor + shift_factor

        with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=True, dtype=torch.float32
        ):
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                mu_t, _ = noise_scheduler.step(model_wrapper, t, mu_t, next_t)

        with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=True, dtype=torch.float32
        ):
            predict_images = dynamic_infer(inferer, autoencoder.decode, mu_t)

        datas = predict_images.squeeze(1).float().cpu().numpy()
        out_spacings = eval_data["spacing"].cpu().numpy()

        datas = (
                (datas - INTENSITY_B_MIN)
                / (INTENSITY_B_MAX - INTENSITY_B_MIN)
                * (INTENSITY_A_MAX - INTENSITY_A_MIN)
                + INTENSITY_A_MIN
        )
        datas = np.int16(np.clip(datas, INTENSITY_A_MIN, INTENSITY_A_MAX))

        futures = save_images_async(
            io_executor,
            datas,
            out_spacings,
            eval_data["out_name"],
            data_root_dir,
            logger,
        )
        pending_futures.extend(futures)

        del src_images, mu_t, predict_images
        if include_body_region:
            del top_region_index_tensor, bottom_region_index_tensor
        if include_modality:
            del modality_tensor

        if batch_count % cleanup_interval == 0:
            for fut in pending_futures:
                fut.result()
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
    args = load_config(env_config_path, model_config_path, model_def_path)
    if "autoencoder_tp_num_splits" in args.diffusion_unet_inference:
        args.autoencoder_def["num_splits"] = args.diffusion_unet_inference[
            "autoencoder_tp_num_splits"
        ]
    args.autoencoder_def["save_mem"] = False
    args.autoencoder_def["norm_float16"] = False
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")

    raw_seed = args.diffusion_unet_inference.get("random_seed", None)
    if raw_seed is not None:
        raw_seed += local_rank
    random_seed = set_random_seed(raw_seed)
    logger.info(
        f"Using {device} of {world_size} with random seed: {random_seed}"
    )

    autoencoder, unet, shift_factor, scale_factor = load_models(
        args, device, logger
    )
    autoencoder = compile_autoencoder_model(autoencoder, args.diffusion_unet_inference["slide_window_size"], device)

    # ── Noise scheduler ──
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.step = MethodType(rk5_step, noise_scheduler) # Option: euler_step, midpoint_step, rk4_step, rk5_step
    noise_scheduler.set_timesteps(
        num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
        input_img_size_numel=torch.prod(torch.tensor(scale_factor.shape[2:])),
    )

    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    filenames_train = load_filenames(args.json_data_list, mode="training")
    filenames_val = load_filenames(args.json_data_list, mode="validation")
    filenames_test = load_filenames(args.json_data_list, mode="test")

    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")
        logger.info(f"num_files_val: {len(filenames_val)}")
        logger.info(f"num_files_test: {len(filenames_test)}")

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
    test_files = prepare_file_list(
        filenames_test,
        args.embedding_base_dir,
        "test",
        include_body_region,
        include_modality,
    )

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
        test_files = partition_dataset(
            data=test_files,
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
    test_loader = prepare_data(
        test_files,
        cache_rate=args.diffusion_unet_inference["cache_rate"],
        num_workers=args.diffusion_unet_inference["num_workers"],
        batch_size=args.diffusion_unet_inference["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping,
    )

    if dist.is_available() and dist.is_initialized():
        obj_list = [None]
        if local_rank == 0:
            obj_list[0] = datetime.now().strftime("%Y%m%d%H%M%S")
        dist.broadcast_object_list(obj_list, src=0)
        timestamp = obj_list[0]
        dist.barrier()
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    io_executor = ProcessPoolExecutor(max_workers=8)
    cleanup_interval = 50
    save_dir_base = os.path.join(args.output_dir, timestamp)

    try:
        for mode, loader in [("validation", val_loader), ("test", test_loader), ("training", train_loader)]:
            save_dir = os.path.join(save_dir_base, mode)
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
                shift_factor=shift_factor,
                scale_factor=scale_factor,
                noise_scheduler=noise_scheduler,
                inferer=inferer,
                logger=logger,
                include_body_region=include_body_region,
                include_modality=include_modality,
                device=device,
                io_executor=io_executor,
                cleanup_interval=cleanup_interval
            )
            dist.barrier()
    finally:
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

