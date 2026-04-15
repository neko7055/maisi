# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (License text omitted for brevity)

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Optional
import gc

import nibabel as nib
import numpy as np
import monai
import torch
from accelerate import Accelerator  # Import Accelerate
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose
from focal_frequency_loss import FocalFrequencyLoss as FFL

from .optimizer import Lion, Lookahead
from .diff_model_setting import load_config, setup_logging
from .interpolator import linear_interpolate, triangular_interpolate, enc_dec_interpolate, polynomial_interpolate, spacial_interpolate
from .solver import euler_step, midpoint_step, rk4_step, rk5_step
from .ssim import _ssim_3D
from .utils import define_instance


# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

def cosine_similarity_loss(y_t, y_prime_t):
    y_t = torch.flatten(y_t, start_dim=1)  # Flatten all dimensions except batch
    y_prime_t = torch.flatten(y_prime_t, start_dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(y_t, y_prime_t, dim=1)
    return 1 - torch.mean(cos_sim)

def x_sigmoid_loss(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(ey_t * torch.tanh(ey_t / 2))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        return x_sigmoid_loss(y_t, y_prime_t)


class MSXSigmoidLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [4 / 7, 2 / 7, 1 / 7]
        self.weights = weights

    def forward(self, y_t, y_prime_t):
        loss = self.weights[0] * x_sigmoid_loss(y_t, y_prime_t)
        for w in self.weights[1:]:
            y_t = torch.nn.functional.avg_pool3d(y_t, kernel_size=2, stride=2)
            y_prime_t = torch.nn.functional.avg_pool3d(y_prime_t, kernel_size=2, stride=2)
            loss += w * x_sigmoid_loss(y_t, y_prime_t)
        return loss


class MSSSIM(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [4 / 7, 2 / 7, 1 / 7]
        self.weights = weights

    def forward(self, y_t, y_prime_t):
        loss = self.weights[0] * (1 - _ssim_3D(y_t, y_prime_t, 3)) / 2
        for w in self.weights[1:]:
            y_t = torch.nn.functional.avg_pool3d(y_t, kernel_size=2, stride=2)
            y_prime_t = torch.nn.functional.avg_pool3d(y_prime_t, kernel_size=2, stride=2)
            loss += w * (1 - _ssim_3D(y_t, y_prime_t, 3)) / 2
        return loss


def ffl_loss_3d(pred, target):
    ffl_loss_fn = FFL(loss_weight=1.0, alpha=1.0)
    B, C, H, W, D = pred.shape
    pred = pred.permute(0, 4, 1, 2, 3)
    target = target.permute(0, 4, 1, 2, 3)

    pred_2d = pred.reshape(-1, C, H, W)
    target_2d = target.reshape(-1, C, H, W)

    loss = ffl_loss_fn(pred_2d, target_2d)
    return loss


class SpectralL1Loss(torch.nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.fft_dims = (-2, -1) if dim == 2 else (-3, -2, -1)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfftn(pred, dim=self.fft_dims, norm="ortho")
        target_fft = torch.fft.rfftn(target, dim=self.fft_dims, norm="ortho")

        loss_real = torch.nn.functional.l1_loss(pred_fft.real, target_fft.real)
        loss_imag = torch.nn.functional.l1_loss(pred_fft.imag, target_fft.imag)

        loss_spectral = loss_real + loss_imag

        return loss_spectral

def augment_modality_label(modality_tensor, prob=0.1):
    # (Same as original function)
    mask_ct = torch.logical_and((modality_tensor < 8), (modality_tensor >= 2))
    prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[torch.logical_and(mask_ct, prob_ct)] = 1

    mask_mri = (modality_tensor >= 9)
    prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[torch.logical_and(mask_mri, prob_mri)] = 8

    mask_zero = torch.rand(modality_tensor.size(), device=modality_tensor.device) > prob
    modality_tensor = modality_tensor * mask_zero.long()

    return modality_tensor


def prepare_file_list(filenames, embedding_base_dir, mode, include_body_region, include_modality):
    # Prepare file list
    files_list = []
    for _i in range(len(filenames)):
        str_src_img = os.path.join(embedding_base_dir, mode, filenames[_i]["src_image"])
        str_tar_img = os.path.join(embedding_base_dir, mode, filenames[_i]["tar_image"])
        if (not os.path.exists(str_src_img)) and (not os.path.exists(str_tar_img)):
            continue

        str_info = os.path.join(embedding_base_dir, mode, filenames[_i]["src_image"]) + ".json"
        files_i = {"src_image": str_src_img,
                   "tar_image": str_tar_img,
                   "spacing": str_info}
        if include_body_region:
            files_i["top_region_index"] = str_info
            files_i["bottom_region_index"] = str_info
        if include_modality:
            files_i["modality"] = str_info
        files_list.append(files_i)
    return files_list


def load_filenames(data_list_path: str, mode: str) -> list:
    # (Same as original function)
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data[mode]
    return [{"src_image": _item["src_image"].replace(".nii.gz", "_emb.nii.gz"),
             "tar_image": _item["tar_image"].replace(".nii.gz", "_emb.nii.gz")} for _item in filenames_train]


def prepare_data(
        train_files: list,
        cache_rate: float,
        num_workers: int = 2,
        batch_size: int = 1,
        include_body_region: bool = False,
        include_modality: bool = True,
        modality_mapping: dict = None
) -> DataLoader:
    def _load_nifti(filepath: str):
        img = nib.load(filepath)
        return img

    def _read_affine(img) -> np.ndarray:
        try:
            return img.affine.astype(np.float64)
        except Exception as e:
            return np.eye(4).astype(np.float64)

    def _nifti_as_tensor(img)-> torch.Tensor:
        data = img.get_fdata(caching='unchanged', dtype=np.float32)
        if data.ndim == 3:
            # (X, Y, Z) → (1, X, Y, Z)
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            # (X, Y, Z, C) → (C, X, Y, Z)
            data = data.transpose(3, 0, 1, 2)

        return torch.from_numpy(data.copy())

    def _load_data_from_file(file_path, key, convert_to_float=True):
        with open(file_path) as f:
            if convert_to_float:
                return torch.FloatTensor(json.load(f)[key])
            else:
                return json.load(f)[key]

    train_transforms_list = [
        monai.transforms.Lambdad(
            keys=["src_image", "tar_image"],
            func=_load_nifti,
            track_meta=False,
            overwrite=True,
        ),
        monai.transforms.Lambdad(
            keys=["src_image", "tar_image"],
            func=_read_affine,
            track_meta=False,
            overwrite=["src_affine", "tar_affine"],
        ),
        monai.transforms.Lambdad(
            keys=["src_image", "tar_image"],
            func=_nifti_as_tensor,
            track_meta=False,
            overwrite=True,
        ),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing"), track_meta=False,),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2, track_meta=False,),
    ]
    if include_body_region:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index", func=lambda x: _load_data_from_file(x, "top_region_index"), track_meta=False,
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: _load_data_from_file(x, "bottom_region_index"), track_meta=False,
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2, track_meta=False,),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2, track_meta=False,),
        ]
    if include_modality:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="modality", func=lambda x: modality_mapping[_load_data_from_file(x, "modality", False)],
                track_meta=False,
            ),
            monai.transforms.EnsureTyped(keys=['modality'], dtype=torch.long, track_meta=False,),
        ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )
    use_persistent = num_workers > 0
    return DataLoader(train_ds,
                      num_workers=num_workers,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      persistent_workers=use_persistent, )


def load_unet(args: argparse.Namespace, accelerator: Accelerator, logger: logging.Logger) -> torch.nn.Module:
    # Load model to CPU first, let Accelerate handle movement
    unet = define_instance(args, "diffusion_unet_def")

    # Optional: Convert BatchNorm to SyncBatchNorm for DDP
    if accelerator.num_processes > 1:
        unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
    else:
        # Load checkpoint on CPU map_location
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location="cpu", weights_only=False)
        unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=False)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet


def calculate_scale_factor(train_files, accelerator: Accelerator, logger: logging.Logger) -> tuple[
    torch.Tensor, torch.Tensor]:
    def _calculate(tensor_list):
        if len(tensor_list) > 0:
            all_data = torch.stack(tensor_list, dim=0).to(accelerator.device, dtype=torch.float64)  # [B, C, D, H, W]
            #mean = torch.mean(all_data, dim=0, keepdim=True)
            #std = torch.std(all_data, dim=0, keepdim=True, correction=0)
            shift_factor = torch.zeros(*all_data.shape[2:], device=accelerator.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)#mean  # median_data
            scale_factor = torch.ones(*all_data.shape[2:], device=accelerator.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)#1 / std  # mad
        else:
            # Fallback if a process has no data (unlikely with proper partition)
            scale_factor = torch.tensor(1.0, device=accelerator.device, dtype=torch.float32)
            shift_factor = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
        return shift_factor, scale_factor

    # (Calculates scale factor locally then syncs across processes)
    data_transforms_list = [
        monai.transforms.LoadImaged(keys=["src_image", "tar_image"]),
        monai.transforms.EnsureChannelFirstd(keys=["src_image", "tar_image"], channel_dim=-1),
        monai.transforms.EnsureTyped(keys=["src_image", "tar_image"], dtype=torch.float32)
    ]
    data_transforms = Compose(data_transforms_list)
    tensor_list = []

    # Only process local files
    for d in train_files:
        d_transformed = data_transforms(d)
        tensor_list.append(d_transformed["tar_image"] - d_transformed["src_image"])

    shift_factor, scale_factor = _calculate(tensor_list)

    shift_factor = accelerator.reduce(shift_factor, reduction="mean").float()
    scale_factor = accelerator.reduce(scale_factor, reduction="mean").float()

    # replace inf/nan with finite numbers to avoid issues in training
    scale_factor = torch.where(torch.isfinite(scale_factor), scale_factor,
                               torch.tensor(1.0, device=scale_factor.device, dtype=torch.float32))
    shift_factor = torch.where(torch.isfinite(shift_factor), shift_factor,
                               torch.tensor(0.0, device=shift_factor.device, dtype=torch.float32))

    logger.info(f"Scale factor is valid -> {torch.isfinite(scale_factor).all().item()}.")
    logger.info(f"Scale factor shape -> {scale_factor.shape}.")
    logger.info(f"scale_factor -> {scale_factor}.")
    logger.info(f"Shift factor is valid -> {torch.isfinite(shift_factor).all().item()}.")
    logger.info(f"Shift factor shape -> {shift_factor.shape}.")
    logger.info(f"shift_factor -> {shift_factor}.")

    return shift_factor, scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    # optimizer = Lion(model.parameters(), lr=lr)
    # optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)
    optimizer = Lookahead(optimizer=optimizer, alpha=0.5)
    return optimizer


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


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


def evaluate(
        unet: torch.nn.Module,
        data_loader: DataLoader,
        shift_factor: torch.Tensor,
        scale_factor: torch.Tensor,
        noise_scheduler: torch.nn.Module,
        accelerator: Accelerator,
        logger: logging.Logger,
        include_body_region,
        include_modality,
) -> torch.Tensor:
    # Handle DDP wrapping access

    if accelerator.is_main_process:
        logger.info(f"Evaluating.")

    _iter = 0
    loss_torch = torch.zeros(4, dtype=torch.float, device=accelerator.device)

    unet.eval()

    # Iterate over loader
    for eval_data in data_loader:
        _iter += 1

        # Move data to device manually (since we didn't pass loader to accelerator.prepare)
        # This is because we pre-partitioned for CacheDataset memory efficiency.
        device = accelerator.device

        src_images = eval_data["src_image"].to(device, non_blocking=True)
        tar_images = eval_data["tar_image"].to(device, non_blocking=True)

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

        spacing_tensor = eval_data["spacing"].to(device, non_blocking=True)

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0.0], dtype=all_timesteps.dtype)))
        all_timesteps, all_next_timesteps = torch.flip(all_next_timesteps, dims=[-1]), torch.flip(all_timesteps,
                                                                                                  dims=[-1])
        mu_t_tf = src_images
        mu_t = src_images
        with torch.inference_mode():
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                def model_wrapper(t, x):
                    return _call_unet(unet,
                                      x,
                                      torch.Tensor((t,)).repeat(x.shape[0]).to(x.device),
                                      spacing_tensor,
                                      include_body_region,
                                      include_modality,
                                      top_region_index_tensor,
                                      bottom_region_index_tensor,
                                      modality_tensor) * scale_factor + shift_factor

                def dry_run(t, x):
                    mu_t_gt, d_mu_t_gt = noise_scheduler.add_noise(src_images, tar_images,
                                                                   torch.Tensor((t,)).repeat(x.shape[0]).to(device,
                                                                                                            non_blocking=True),
                                                                   force_no_noise=True)
                    return mu_t_gt, d_mu_t_gt

                dt_org = next_t - t
                dt = dt_org / noise_scheduler.num_train_timesteps
                mu_t, _ = noise_scheduler.step(model_wrapper, t, mu_t, next_t)

                mu_t_gt, d_mu_t_gt = dry_run(t, mu_t_tf)
                v_pred_tf = model_wrapper(t, mu_t_gt)
                mu_t_tf = mu_t_tf + dt * v_pred_tf
                loss_pose = torch.nn.functional.mse_loss(mu_t, mu_t_tf, reduction='mean') * dt
                loss_torch[2] += loss_pose.item()
            loss = torch.nn.functional.mse_loss(mu_t, tar_images, reduction='mean')
            loss_tf = torch.nn.functional.mse_loss(mu_t_tf, tar_images, reduction='mean')
            loss_torch[0] += loss.item()
            loss_torch[1] += loss_tf.item()
            loss_torch[-1] += 1.0

    # Reduce loss for logging
    loss_torch = accelerator.reduce(loss_torch, reduction="sum")
    return loss_torch


def train_one_epoch(
        epoch: int,
        unet: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
        loss_pt: torch.nn.Module,
        shift_factor: torch.Tensor,
        scale_factor: torch.Tensor,
        noise_scheduler: torch.nn.Module,
        accelerator: Accelerator,
        logger: logging.Logger,
        include_body_region,
        include_modality,
        time_batch_size: int,
        gradient_accumulation_steps: int
) -> torch.Tensor:
    # Handle DDP wrapping access
    if accelerator.is_main_process:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=accelerator.device)

    unet.train()

    # Iterate over loader
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]
        _iter += 1

        # Move data to device manually (since we didn't pass loader to accelerator.prepare)
        # This is because we pre-partitioned for CacheDataset memory efficiency.
        device = accelerator.device

        src_images = train_data["src_image"].to(device, non_blocking=True)
        tar_images = train_data["tar_image"].to(device, non_blocking=True)

        spacing_tensor = train_data["spacing"].to(device, non_blocking=True)
        top_region_index_tensor = None
        bottom_region_index_tensor = None
        modality_tensor = None

        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(
                device, non_blocking=True
            )
            bottom_region_index_tensor = train_data["bottom_region_index"].to(
                device, non_blocking=True
            )
        if include_modality:
            modality_tensor = train_data["modality"].to(
                device, non_blocking=True
            )

        # Logic remains same
        assert isinstance(noise_scheduler, RFlowScheduler)
        loss_float = 0
        for _ in range(time_batch_size):
            timesteps = noise_scheduler.sample_timesteps(src_images)
            with accelerator.accumulate(unet), accelerator.autocast():
                mu_t_gt, d_mu_t_gt = noise_scheduler.add_noise(src_images, tar_images, timesteps)
                d_mu_t = _call_unet(unet,
                                    mu_t_gt,
                                    timesteps,
                                    spacing_tensor,
                                    include_body_region,
                                    include_modality,
                                    top_region_index_tensor,
                                    bottom_region_index_tensor,
                                    modality_tensor) * scale_factor + shift_factor
                loss = loss_pt(d_mu_t.float(), d_mu_t_gt.float())
                loss_float += loss.item() / time_batch_size
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_torch[0] += loss_float
        loss_torch[1] += 1.0

        # if accelerator.is_main_process:
        if _iter % gradient_accumulation_steps == 0:
            float_loss = torch.tensor(loss_float, dtype=torch.float32, device=accelerator.device)
            accelerator.wait_for_everyone()
            avg_loss = accelerator.reduce(float_loss, reduction="mean")
            avg_float_loss = avg_loss.item()
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter // gradient_accumulation_steps,
                                              len(train_loader) // gradient_accumulation_steps, avg_float_loss,
                    current_lr
                )
            )

        # Reduce loss for logging
    loss_torch = accelerator.reduce(loss_torch, reduction="sum")
    optimizer.optimizer.apply_lookahead_steps()
    lr_scheduler.step()
    return loss_torch


def save_checkpoint(
        epoch: int,
        unet: torch.nn.Module,
        loss_torch_epoch: float,
        num_train_timesteps: int,
        shift_factor: torch.Tensor,
        scale_factor: torch.Tensor,
        ckpt_folder: str,
        args: argparse.Namespace,
        accelerator: Accelerator
) -> None:
    # Wait for everyone before saving
    accelerator.wait_for_everyone()

    # Unwrap model to get clean state dict
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Only save on main process
    if accelerator.is_main_process:
        torch.save(
            {
                "epoch": epoch + 1,
                "loss": loss_torch_epoch,
                "num_train_timesteps": num_train_timesteps,
                "shift_factor": shift_factor,
                "scale_factor": scale_factor,
                "unet_state_dict": unwrapped_unet.state_dict(),
            },
            f"{ckpt_folder}/{args.model_filename}",
        )


def diff_model_train(
        env_config_path: str, model_config_path: str, model_def_path: str
) -> None:
    # Initialize Accelerator
    # mixed_precision can be "no", "fp16", "bf16".
    # It is recommended to configure this via `accelerate config` CLI or pass arg here.
    args = load_config(env_config_path, model_config_path, model_def_path)
    accelerator = Accelerator(gradient_accumulation_steps=1 * args.diffusion_unet_train["gradient_accumulation_steps"] *\
                                                              args.diffusion_unet_train["time_batch_size"],
                              step_scheduler_with_optimizer=False)

    logger = setup_logging("training", rk_filter=True)

    # Log device info
    logger.info(f"Process {accelerator.process_index} using device: {accelerator.device}")

    if accelerator.is_main_process:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Load UNet (Move to device logic handled by prepare, but we load first)
    unet = load_unet(args, accelerator, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.step = MethodType(rk5_step,
                                      noise_scheduler)  # Option: euler_step, midpoint_step, rk4_step, rk5_step
    noise_scheduler.add_noise = MethodType(partial(spacial_interpolate, add_noise=True),
                                           noise_scheduler)  # Option: linear_interpolate, triangular_interpolate, enc_dec_interpolate, polynomial_interpolate

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
    if accelerator.is_main_process:
        logger.info(f"num_files_train: {len(filenames_train)}")
        logger.info(f"num_files_val: {len(filenames_val)}")
        logger.info(f"num_files_test: {len(filenames_test)}")

    # Prepare file list
    train_files = prepare_file_list(filenames_train, args.embedding_base_dir, "training", include_body_region,
                                    include_modality)
    val_files = prepare_file_list(filenames_val, args.embedding_base_dir, "validation", include_body_region,
                                  include_modality)

    # Partition dataset BEFORE creating CacheDataset to save RAM
    # Accelerate makes this easy by giving us num_processes and process_index
    train_files = partition_dataset(
        data=train_files,
        shuffle=True,
        num_partitions=accelerator.num_processes,
        even_divisible=True
    )[accelerator.process_index]

    val_files = partition_dataset(
        data=val_files,
        shuffle=False,
        num_partitions=accelerator.num_processes,
        even_divisible=False
    )[accelerator.process_index]

    # Calculate scale factor locally then sync
    shift_factor, scale_factor = calculate_scale_factor(train_files, accelerator, logger)
    noise_scheduler.set_timesteps(
        num_inference_steps=args.diffusion_unet_train["num_validation_steps"],
        input_img_size_numel=torch.prod(torch.tensor(scale_factor.shape[2:])),
    )

    # Create DataLoader with local subset
    train_loader = prepare_data(
        train_files,
        cache_rate=args.diffusion_unet_train["cache_rate"],
        num_workers=args.diffusion_unet_train["num_workers"],
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping
    )

    val_loader = prepare_data(
        val_files,
        cache_rate=args.diffusion_unet_train["cache_rate"],
        num_workers=args.diffusion_unet_train["num_workers"],
        batch_size=args.diffusion_unet_train["validation_batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping
    )

    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    # Calculate steps based on local dataset size (approximate)
    total_steps = args.diffusion_unet_train["n_epochs"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = MSXSigmoidLoss().to(accelerator.device)

    # Prepare everything with Accelerate
    # NOTE: We do NOT pass train_loader here because we manually partitioned the dataset
    # for CacheDataset efficiency. Accelerate would try to shard it again or replace sampler.
    # We will manually handle device movement in the loop.
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )
    accelerator.wait_for_everyone()
    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        start_time = time.perf_counter()
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            shift_factor,
            scale_factor,
            noise_scheduler,
            accelerator,
            logger,
            include_body_region,
            include_modality,
            args.diffusion_unet_train["time_batch_size"],
            args.diffusion_unet_train["gradient_accumulation_steps"]
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        loss_torch = loss_torch.tolist()

        # Calculate average loss
        loss_torch_epoch = loss_torch[0] / loss_torch[1]

        if accelerator.is_main_process:
            logger.info(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, time taken: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
        accelerator.free_memory()
        if (epoch + 1) % 10 == 0 or epoch == args.diffusion_unet_train["n_epochs"] - 1 or epoch == 0:
            start_time = time.perf_counter()
            eval_loss_torch = evaluate(
                unet,
                val_loader,
                shift_factor,
                scale_factor,
                noise_scheduler,
                accelerator,
                logger,
                include_body_region,
                include_modality,
            )
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            eval_loss_torch = eval_loss_torch / eval_loss_torch[-1]
            eval_loss_torch = eval_loss_torch.tolist()
            formatted = ", ".join(f"{x:.8f}" for x in eval_loss_torch)

            if accelerator.is_main_process:
                logger.info(
                    f"epoch {epoch + 1} average mse loss on validation set: {formatted}, time taken: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.")
            accelerator.free_memory()
        save_checkpoint(
            epoch,
            unet,
            loss_torch_epoch,
            args.noise_scheduler["num_train_timesteps"],
            shift_factor,
            scale_factor,
            args.model_dir,
            args,
            accelerator
        )

    logger.info("Training finished")
    del train_loader, val_loader, unet, optimizer, lr_scheduler, noise_scheduler
    accelerator.free_memory()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training with Accelerate")
    parser.add_argument(
        "-e", "--env_config_path", type=str, default="./configs/environment_maisi_diff_model.json",
    )
    parser.add_argument(
        "-c", "--model_config_path", type=str, default="./configs/config_maisi_diff_model.json",
    )
    parser.add_argument(
        "-t", "--model_def_path", type=str, default="./configs/config_maisi.json",
    )

    parser.add_argument(
        "-g",
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training"
    )
    # Removed manual GPU args, handled by 'accelerate launch'

    args = parser.parse_args()
    diff_model_train(args.env_config_path, args.model_config_path, args.model_def_path)
