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

import monai
import torch
from accelerate import Accelerator  # Import Accelerate
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose

from .optimizer import Lion, Lookahead
from .diff_model_setting import load_config, setup_logging
from .interpolator import linear_interpolate, triangular_interpolate, enc_dec_interpolate
from .solver import euler_step, midpoint_step, rk4_step, rk5_step
from .ssim import SSIM3D
from .utils import define_instance

# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t / 2))


class Loss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        # self.l2_loss = torch.nn.MSELoss()
        # self.ssim_8 = SSIM3D(window_size=9)
        # self.ssim_16 = SSIM3D(window_size=17)
        # self.ssim_32 = SSIM3D(window_size=33)
        # self.ssim_64 = SSIM3D(window_size=65)
        self.ssim_15 = SSIM3D(window_size=15)
        $self.xsigmoidloss = XSigmoidLoss()

    def forward(self, outputs, targets):
        # ssim = self.ssim_64(outputs, targets) * 0.4 + \
        #        self.ssim_32(outputs, targets) * 0.3 + \
        #        self.ssim_16(outputs, targets) * 0.2 + \
        #        self.ssim_8(outputs, targets) * 0.1

        ssim = self.ssim_15(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        return ssim * 0.8 + l1 * 0.2


def compile_unet_model(model):
    compile_module_list = ["conv_in", "down_blocks", "middle_block", "up_blocks", "out"]

    for module_name in compile_module_list:
        module = getattr(model, module_name, None)
        if module is not None:
            # 統一使用 .compile() 進行就地編譯，不分 Module 或 ModuleList
            module.compile(
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,  # 若需動態解析度請改為 True
                backend="inductor",
            )

    return model

def augment_modality_label(modality_tensor, prob=0.1):
    # (Same as original function)
    mask_ct = (modality_tensor < 8) and (modality_tensor >= 2)
    prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_ct & prob_ct] = 1

    mask_mri = (modality_tensor >= 9)
    prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_mri & prob_mri] = 8

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
    use_persistent = num_workers > 0
    return DataLoader(train_ds,
                      num_workers=num_workers,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      persistent_workers=use_persistent,)


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
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def _calculate(tensor_list):
        if len(tensor_list) > 0:
            all_data = torch.stack(tensor_list, dim=0).to(accelerator.device, dtype=torch.float64) # [B, C, D, H, W]
            mean = torch.mean(all_data, dim=[0, 2, 3, 4], keepdim=True)
            std = torch.std(all_data, dim=[0, 2, 3, 4], keepdim=True, correction=0)
            #median_data = torch.quantile(all_data, 0.5, dim=1, keepdim=False)
            #mad = torch.quantile(torch.abs(all_data - median_data), 0.5, dim=1, keepdim=False) * 1.4826
            shift_factor = mean#median_data
            scale_factor = 1 / std#mad
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
    src_tensor_list = []
    tar_tensor_list = []

    # Only process local files
    for d in train_files:
        d_transformed = data_transforms(d)
        src_tensor_list.append(d_transformed["src_image"])
        tar_tensor_list.append(d_transformed["tar_image"])


    src_shift_factor, src_scale_factor = _calculate(src_tensor_list)
    tar_shift_factor, tar_scale_factor = _calculate(tar_tensor_list)

    src_shift_factor = accelerator.reduce(src_shift_factor, reduction="mean").float()
    src_scale_factor = accelerator.reduce(src_scale_factor, reduction="mean").float()

    tar_shift_factor = accelerator.reduce(tar_shift_factor, reduction="mean").float()
    tar_scale_factor = accelerator.reduce(tar_scale_factor, reduction="mean").float()

    # replace inf/nan with finite numbers to avoid issues in training
    src_scale_factor = torch.where(torch.isfinite(src_scale_factor), src_scale_factor, torch.tensor(1.0, device=src_scale_factor.device, dtype=torch.float32))
    src_shift_factor = torch.where(torch.isfinite(src_shift_factor), src_shift_factor,torch.tensor(0.0, device=src_shift_factor.device, dtype=torch.float32))
    tar_scale_factor = torch.where(torch.isfinite(tar_scale_factor), tar_scale_factor, torch.tensor(1.0, device=tar_scale_factor.device, dtype=torch.float32))
    tar_shift_factor = torch.where(torch.isfinite(tar_shift_factor), tar_shift_factor, torch.tensor(0.0, device=tar_shift_factor.device, dtype=torch.float32))

    logger.info(f"Src Scale factor is valid -> {torch.isfinite(src_scale_factor).all().item()}.")
    logger.info(f"Src scale_factor -> {src_scale_factor}.")
    logger.info(f"Src Shift factor is valid -> {torch.isfinite(src_shift_factor).all().item()}.")
    logger.info(f"Src shift_factor -> {src_shift_factor}.")

    logger.info(f"Src Scale factor is valid -> {torch.isfinite(tar_scale_factor).all().item()}.")
    logger.info(f"Src scale_factor -> {tar_scale_factor}.")
    logger.info(f"Src Shift factor is valid -> {torch.isfinite(tar_shift_factor).all().item()}.")
    logger.info(f"Src shift_factor -> {tar_shift_factor}.")
    return src_shift_factor, src_scale_factor, tar_shift_factor, tar_scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    optimizer = Lion(model.parameters(), lr=lr)
    optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    #ptimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

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

def evaluate(
        unet: torch.nn.Module,
        data_loader: DataLoader,
        src_shift_factor: torch.Tensor,
        src_scale_factor: torch.Tensor,
        tar_shift_factor: torch.Tensor,
        tar_scale_factor: torch.Tensor,
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
    loss_torch = torch.zeros(2, dtype=torch.float, device=accelerator.device)

    unet.eval()

    # Iterate over loader
    for eval_data in data_loader:
        _iter += 1

        # Move data to device manually (since we didn't pass loader to accelerator.prepare)
        # This is because we pre-partitioned for CacheDataset memory efficiency.
        device = accelerator.device

        src_images = eval_data["src_image"].to(device, non_blocking=True)
        tar_images = eval_data["tar_image"].to(device, non_blocking=True)

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

        spacing_tensor = eval_data["spacing"].to(device, non_blocking=True)

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0.0], dtype=all_timesteps.dtype)))

        mu_t = src_images
        with torch.inference_mode():
            for t, next_t in zip(all_timesteps, all_next_timesteps):
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
                mu_t, _ = noise_scheduler.step(model_wrapper, t, mu_t, next_t)

                # Logging only on main process (simplified checks)
                # if accelerator.is_main_process:
                #    logger.info(...)
            mu_t = mu_t * (1 / tar_scale_factor) + tar_shift_factor  # Un-normalize for loss calculation
            loss = torch.nn.functional.mse_loss(mu_t, tar_images, reduction='mean')

            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0

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
        src_shift_factor: torch.Tensor,
        src_scale_factor: torch.Tensor,
        tar_shift_factor: torch.Tensor,
        tar_scale_factor: torch.Tensor,
        noise_scheduler: torch.nn.Module,
        accelerator: Accelerator,
        logger: logging.Logger,
        include_body_region,
        include_modality,
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

        src_images = (src_images - src_shift_factor) * src_scale_factor
        tar_images = (tar_images - tar_shift_factor) * tar_scale_factor

        spacing_tensor = train_data["spacing"].to(device, non_blocking=True)

        # Logic remains same
        assert isinstance(noise_scheduler, RFlowScheduler), "Currently we only support RFlowScheduler for training, please check your config and model definition."
        timesteps = noise_scheduler.sample_timesteps(src_images)

        mu_t, d_mu_t = noise_scheduler.add_noise(src_images, tar_images, timesteps)

        unet_inputs = {
            "x": mu_t,
            "timesteps": timesteps,
            "spacing_tensor": spacing_tensor,
        }

        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(device, non_blocking=True)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device, non_blocking=True)
            unet_inputs.update({
                "top_region_index_tensor": top_region_index_tensor,
                "bottom_region_index_tensor": bottom_region_index_tensor,
            })

        if include_modality:
            modality_tensor = train_data["modality"].to(device, non_blocking=True)
            modality_tensor = augment_modality_label(modality_tensor).to(device)
            unet_inputs.update({"class_labels": modality_tensor})

        # Accelerate handles mixed precision automatically if configured
        with accelerator.accumulate(unet):
            model_output = unet(**unet_inputs)
            loss = loss_pt(model_output, d_mu_t)

            # Replaced backward with accelerator
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if accelerator.is_main_process:
            if _iter % gradient_accumulation_steps == 0:
                logger.info(
                    "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                        str(datetime.now())[:19], epoch + 1, _iter // gradient_accumulation_steps,
                                                  len(train_loader) // gradient_accumulation_steps, loss.item(),
                        current_lr
                    )
                )

    # Reduce loss for logging
    loss_torch = accelerator.reduce(loss_torch, reduction="sum")
    lr_scheduler.step()
    return loss_torch


def save_checkpoint(
        epoch: int,
        unet: torch.nn.Module,
        loss_torch_epoch: float,
        num_train_timesteps: int,
        src_shift_factor: torch.Tensor,
        src_scale_factor: torch.Tensor,
        tar_shift_factor: torch.Tensor,
        tar_scale_factor: torch.Tensor,
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
                "src_shift_factor": src_shift_factor,
                "src_scale_factor": src_scale_factor,
                "tar_shift_factor": tar_shift_factor,
                "tar_scale_factor": tar_scale_factor,
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
    accelerator = Accelerator(gradient_accumulation_steps=args.diffusion_unet_train["gradient_accumulation_steps"],
                              step_scheduler_with_optimizer=False)

    logger = setup_logging("training")

    # Log device info
    logger.info(f"Process {accelerator.process_index} using device: {accelerator.device}")

    if accelerator.is_main_process:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Load UNet (Move to device logic handled by prepare, but we load first)
    unet = load_unet(args, accelerator, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.step = MethodType(midpoint_step, noise_scheduler) # Option: euler_step, midpoint_step, rk4_step, rk5_step
    noise_scheduler.add_noise = MethodType(partial(linear_interpolate, add_noise=False), noise_scheduler) # Option: linear_interpolate, triangular_interpolate, enc_dec_interpolate

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
    src_shift_factor, src_scale_factor, tar_shift_factor, tar_scale_factor = calculate_scale_factor(train_files, accelerator, logger)
    noise_scheduler.set_timesteps(
        num_inference_steps=args.diffusion_unet_train["num_validation_steps"],
        input_img_size_numel=torch.prod(torch.tensor(src_scale_factor.shape[2:])),
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
    loss_pt = Loss().to(accelerator.device)

    # Prepare everything with Accelerate
    # NOTE: We do NOT pass train_loader here because we manually partitioned the dataset
    # for CacheDataset efficiency. Accelerate would try to shard it again or replace sampler.
    # We will manually handle device movement in the loop.
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )
    unet = compile_unet_model(unet)


    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        start_time = time.perf_counter()
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            src_shift_factor,
            src_scale_factor,
            tar_shift_factor,
            tar_scale_factor,
            noise_scheduler,
            accelerator,
            logger,
            include_body_region,
            include_modality,
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
        #if (epoch + 1) % 10 == 0 or epoch == args.diffusion_unet_train["n_epochs"] - 1:
        start_time = time.perf_counter()
        eval_loss_torch = evaluate(
            unet,
            val_loader,
            src_shift_factor,
            src_scale_factor,
            tar_shift_factor,
            tar_scale_factor,
            noise_scheduler,
            accelerator,
            logger,
            include_body_region,
            include_modality,
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        eval_loss_torch = eval_loss_torch.tolist()
        eval_loss_torch_epoch = eval_loss_torch[0] / eval_loss_torch[1]

        if accelerator.is_main_process:
            logger.info(
                f"epoch {epoch + 1} average mse loss on validation set: {eval_loss_torch_epoch:.4f}, time taken: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.")

        save_checkpoint(
            epoch,
            unet,
            loss_torch_epoch,
            args.noise_scheduler["num_train_timesteps"],
            src_shift_factor,
            src_scale_factor,
            tar_shift_factor,
            tar_scale_factor,
            args.model_dir,
            args,
            accelerator
        )

    logger.info("Training finished")
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
