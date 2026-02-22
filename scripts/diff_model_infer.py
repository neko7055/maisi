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
import logging
import os, json
import random
from datetime import datetime
from functools import partial
from types import MethodType

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import monai
import monai
import torch
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose
from monai.inferers.inferer import SlidingWindowInferer
from monai.utils import set_determinism
from tqdm import tqdm

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .sample import ReconModel, check_input_ct
from .utils import define_instance, dynamic_infer
from .solver import euler_step, midpoint_step, rk4_step, rk5_step


def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path)
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    logger.info(f"checkpoints {args.trained_autoencoder_path} loaded.")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    shift_factor = checkpoint["shift_factor"]
    logger.info(f"scale_factor -> {shift_factor}.")
    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, shift_factor, scale_factor

def load_filenames(data_list_path: str, mode: str) -> list:
    # (Same as original function)
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data[mode]
    return [{"src_image": _item["src_image"].replace(".nii.gz", "_emb.nii.gz"),
             "tar_image": _item["tar_image"].replace(".nii.gz", "_emb.nii.gz"),
             "file_name": _item["tar_image"].replace(".nii.gz", "_pred.nii.gz"),} for _item in filenames_train]

def prepare_file_list(filenames, embedding_base_dir, mode, include_body_region, include_modality):
    # Prepare file list
    files_list = []
    for _i in range(len(filenames)):
        str_src_img = os.path.join(embedding_base_dir, mode, filenames[_i]["src_image"])
        str_tar_img = os.path.join(embedding_base_dir, mode, filenames[_i]["tar_image"])
        out_name = filenames[_i]["file_name"]
        if (not os.path.exists(str_src_img)) and (not os.path.exists(str_tar_img)):
            continue

        str_info = os.path.join(embedding_base_dir, mode, filenames[_i]["src_image"]) + ".json"
        files_i = {"src_image": str_src_img,
                   "tar_image": str_tar_img,
                   "spacing": str_info,
                   "out_name": out_name}
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
        modality_mapping: dict = None
) -> DataLoader:
    # (Modified: removed device arg, rely on Accelerator or manual to() later)

    def _load_data_from_file(file_path, key, convert_to_float=True):
        with open(file_path) as f:
            if convert_to_float:
                return torch.FloatTensor(json.load(f)[key])
            else:
                return json.load(f)[key]

    train_transforms_list = [
        monai.transforms.LoadImaged(keys=["src_image", "tar_image"]),
        monai.transforms.EnsureChannelFirstd(keys=["src_image", "tar_image"], channel_dim=-1),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
    ]
    if include_body_region:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index", func=lambda x: _load_data_from_file(x, "top_region_index")
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: _load_data_from_file(x, "bottom_region_index")
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]
    if include_modality:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="modality", func=lambda x: modality_mapping[_load_data_from_file(x, "modality", False)]
            ),
            monai.transforms.EnsureTyped(keys=['modality'], dtype=torch.long),
        ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=False)

def run_inference(
    unet: torch.nn.Module,
    autoencoder: torch.nn.Module,
    data_loader: DataLoader,
    shift_factor: torch.Tensor,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    logger: logging.Logger,
    include_body_region,
    include_modality,
    device: torch.device,
) -> np.ndarray:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        modality_tensor (torch.Tensor): Modality tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """


    unet.eval()
    autoencoder.eval()

    # Iterate over loader
    for eval_data in data_loader:

        src_images = eval_data["src_image"].to(device)
        tar_images = eval_data["tar_image"].to(device)

        src_images = (src_images - shift_factor) * scale_factor

        if include_body_region:
            top_region_index_tensor = eval_data["top_region_index"].to(device)
            bottom_region_index_tensor = eval_data["bottom_region_index"].to(device)
        if include_modality:
            modality_tensor = eval_data["modality"].to(device)

        spacing_tensor = eval_data["spacing"].to(device)

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0.0], dtype=all_timesteps.dtype)))

        mu_t = src_images
        with torch.inference_mode():
            for t, next_t in zip(all_timesteps, all_next_timesteps):

                def model_warpper(t, x):
                    unet_inputs = {
                        "x": x,
                        "timesteps": torch.Tensor((t,)).repeat(x.shape[0]).to(x.device),
                        "spacing_tensor": spacing_tensor,
                    }
                    if include_body_region:
                        unet_inputs.update({
                            "top_region_index_tensor": top_region_index_tensor,
                            "bottom_region_index_tensor": bottom_region_index_tensor,
                        })
                    if include_modality:
                        unet_inputs.update({"class_labels": modality_tensor})
                    model_output = unet(**unet_inputs)
                    return model_output

                mu_t, _ = noise_scheduler.step(model_warpper, t, mu_t, next_t)

                # Logging only on main process (simplified checks)
                # if accelerator.is_main_process:
                #    logger.info(...)
            mu_t = mu_t * (1 / scale_factor) + shift_factor  # Un-normalize for loss calculation

            inferer = SlidingWindowInferer(
                        roi_size=[320, 320, 160],
                        sw_batch_size=1,
                        progress=False,
                        mode="gaussian",
                        overlap=0.4,
                        sw_device=device,
                        device=device,
                    )
            predict_images = dynamic_infer(inferer, autoencoder.decode, mu_t)
            data = predict_images.cpu().detach().numpy()
            a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
            data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
            data = np.clip(data, a_min, a_max)
            asd/2
        return np.int16(data)


def save_image(
    data,
    out_spacing: tuple,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data : Synthetic image data.
        output_size (tuple): Output size of the image.
        out_spacing (tuple): Spacing of the output image.
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"Saved {output_path}.")


@torch.inference_mode()
def diff_model_infer(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int) -> None:
    """
    Main function to run the diffusion model inference.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if "random_seed" in args.diffusion_unet_inference.keys()
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    # if local_rank == 0:
    #     logger.info(f"[config] ckpt_filepath -> {ckpt_filepath}.")
    #     logger.info(f"[config] random_seed -> {random_seed}.")
    #     logger.info(f"[config] output_prefix -> {output_prefix}.")
    #     logger.info(f"[config] output_size -> {output_size}.")
    #     logger.info(f"[config] out_spacing -> {out_spacing}.")

    autoencoder, unet, shift_factor, scale_factor = load_models(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.step = MethodType(euler_step, noise_scheduler)
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

    # Prepare file list
    train_files = prepare_file_list(filenames_train, args.embedding_base_dir, "training", include_body_region,
                                    include_modality)
    val_files = prepare_file_list(filenames_val, args.embedding_base_dir, "validation", include_body_region,
                                  include_modality)
    # test_files = prepare_file_list(filenames_test,args.embedding_base_dir, "test", include_body_region, include_modality)

    if torch.distributed.is_initialized():
        train_files = partition_dataset(
            data=train_files,
            shuffle=False,
            num_partitions=torch.distributed.get_world_size(),
            even_divisible=True
        )[local_rank]

        val_files = partition_dataset(
            data=val_files,
            shuffle=False,
            num_partitions=torch.distributed.get_world_size(),
            even_divisible=True
        )[local_rank]

    # Create DataLoader with local subset
    train_loader = prepare_data(
        train_files,
        args.diffusion_unet_inference["cache_rate"],
        batch_size=args.diffusion_unet_inference["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping
    )

    val_loader = prepare_data(
        val_files,
        args.diffusion_unet_inference["cache_rate"],
        batch_size=args.diffusion_unet_inference["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping
    )

    torch.set_float32_matmul_precision("highest")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    for mode in ["training"]:
        save_dir = os.path.join(args.output_dir, timestamp, mode)
        os.makedirs(save_dir, exist_ok=True)
        data = run_inference(
            unet,
        autoencoder,
        train_loader,
        shift_factor,
        scale_factor,
        noise_scheduler,
        logger,
        include_body_region,
        include_modality,
        device
        )
    save_image(data, output_size, out_spacing, output_path, logger)

    # ---- gather & persist ----
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument("-e","--env_config", type=str, required=True)
    parser.add_argument("-c","--model_config", type=str, required=True)
    parser.add_argument("-t","--model_def", type=str, required=True)
    parser.add_argument(
        "-g",
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for training"
    )

    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def, args.num_gpus)