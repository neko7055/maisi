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
from torch.distributions import Normal
from accelerate import Accelerator  # Import Accelerate
from monai.data import DataLoader, partition_dataset
from monai.networks.nets import PatchDiscriminator
from monai.transforms import Compose

from .optimizer import Lion, Lookahead
from .diff_model_setting import load_config, setup_logging
from .ssim import _ssim_3D
from .utils import define_instance
from .transforms import VAE_Transform

# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

def sliced_energy_loss(X, num_projections=50):
    """
    利用隨機投影，將高維轉為 1D 後，使用 1D 的精確封閉解
    """
    B, d = X.shape

    directions = torch.randn(d, num_projections, device=X.device)
    directions = directions / torch.norm(directions, dim=0, keepdim=True)

    X_proj = torch.matmul(X, directions)  # (B, num_projections)

    normal = Normal(0.0, 1.0)
    phi = torch.exp(normal.log_prob(X_proj))  # PDF
    Phi = normal.cdf(X_proj)  # CDF

    # 第一項：E[|X - Z|] 的封閉解，對 batch 取平均
    term1 = (X_proj * (2 * Phi - 1) + 2 * phi).mean(dim=0)  # (num_projections,)

    # 第二項：E[|X - X'|]，使用排序法避免 O(B^2) 記憶體
    X_sorted, _ = torch.sort(X_proj, dim=0)  # (B, num_projections)
    # 排序後 E[|X_i - X_j|] = (2*rank - B - 1) * X_sorted / B^2 的求和
    ranks = torch.arange(1, B + 1, device=X.device, dtype=X.dtype).unsqueeze(1)
    dist_XX = (2 * (2 * ranks - B - 1) * X_sorted).sum(dim=0) / (B * B)  # (num_projections,)

    # 第三項：E[|Z - Z'|] = 2/sqrt(pi)
    term3 = 2.0 / (torch.pi ** 0.5)

    loss = (2 * term1 - dist_XX - term3).mean()
    return loss

def kl_loss(z_mu, z_sigma):
    # KL divergence between N(z_mu, z_sigma^2) and N(0, 1)
    # z_mu = (b,c, d, h, w), z_sigma = (b,c, d, h, w)
    kl = -0.5 * torch.mean(1 + torch.log(z_sigma.pow(2) + 1e-8) - z_mu.pow(2) - z_sigma.pow(2), dim=[1, 2, 3, 4])
    return kl.mean()

def x_sigmoid_loss(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(ey_t * torch.tanh(ey_t / 2))

class Loss(torch.nn.Module):
    def __init__(self,weights=None, window_size=3):
        super().__init__()
        if weights is None:
            weights = [4/7, 2/7, 1/7]
        self.weights = weights
        self.window_size = window_size

    def _loss(self, y_t, y_prime_t):
        mse_loss = torch.nn.functional.mse_loss(y_t, y_prime_t)
        loss = (1 - _ssim_3D(y_t, y_prime_t, window_size=self.window_size)) / 2 * 0.8 + x_sigmoid_loss(y_t, y_prime_t) * 0.2
        return loss + mse_loss

    def forward(self, y_t, y_prime_t):
        loss = self.weights[0] * self._loss(y_t, y_prime_t)
        for w in self.weights[1:]:
            y_t = torch.nn.functional.avg_pool3d(y_t, kernel_size=2, stride=2)
            y_prime_t = torch.nn.functional.avg_pool3d(y_prime_t, kernel_size=2, stride=2)
            loss += w * self._loss(y_t, y_prime_t)
        return loss

def prepare_file_list(filenames, data_base_dir, mode):
    # Prepare file list
    files_list = []
    for _i in range(len(filenames)):
        str_src_img = os.path.join(data_base_dir, mode, filenames[_i]["src_image"])
        str_tar_img = os.path.join(data_base_dir, mode, filenames[_i]["tar_image"])
        files_i = {"src_image": str_src_img,
                   "tar_image": str_tar_img,
                   "class": "ct"}
        files_list.append(files_i)
    return files_list

def load_filenames(data_list_path: str, mode: str) -> list:
    # (Same as original function)
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data[mode]
    return [{"src_image": _item["src_image"],
             "tar_image": _item["tar_image"]} for _item in filenames_train]

def prepare_data(
        train_files: list,
        data_transforms: VAE_Transform,
        cache_rate: float,
        num_workers: int = 2,
        batch_size: int = 1,
) -> DataLoader:
    # (Modified: removed device arg, rely on Accelerator or manual to() later)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=data_transforms, cache_rate=cache_rate, num_workers=num_workers
    )
    use_persistent = num_workers > 0
    return DataLoader(train_ds,
                      num_workers=num_workers,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      persistent_workers=use_persistent,)

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

def load_model(args: argparse.Namespace, accelerator: Accelerator, logger) -> torch.nn.Module:
    # Load model to CPU first, let Accelerate handle movement

    autoencoder = define_instance(args, "autoencoder_def")
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path, map_location="cpu", weights_only=False
    )
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)

    # Optional: Convert BatchNorm to SyncBatchNorm for DDP
    if accelerator.num_processes > 1:
        autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
    expand_first_conv_input_channels(autoencoder.encoder.blocks[0].conv, k=18)
    # if args.trained_autoencoder_path.replace(".pt", f"_my.pt") exists, load it instead (for resuming training or using pre-expanded model)
    my_checkpoint_path = args.trained_autoencoder_path.replace(".pt", f"_my.pt")
    if os.path.exists(my_checkpoint_path):
        checkpoint_autoencoder = torch.load(
            my_checkpoint_path, map_location="cpu", weights_only=True
        )
        logger.info(f"Loading expanded model weights from {my_checkpoint_path}.")
        autoencoder.load_state_dict(checkpoint_autoencoder)
    return autoencoder


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    #optimizer = Lion(model.parameters(), lr=lr)
    #optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)
    optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    return optimizer


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    #return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)

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

def train_one_epoch(
        epoch: int,
        autoencoder: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
        loss_pt: torch.nn.Module,
        accelerator: Accelerator,
        logger: logging.Logger,
        gradient_accumulation_steps: int
) -> torch.Tensor:
    # Handle DDP wrapping access

    if accelerator.is_main_process:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=accelerator.device)

    autoencoder.train()
    a_t, b_t = build_ct_channel_params(A=-200, B=700, levels=[1, 3, 5], ks=[4, 5])
    et_channel = lambda x: apply_ct_channel_extend(x, a_t, b_t)
    # Iterate over loader
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]
        _iter += 1

        # Move data to device manually (since we didn't pass loader to accelerator.prepare)
        # This is because we pre-partitioned for CacheDataset memory efficiency.
        device = accelerator.device

        src_images = train_data["src_image"].to(device, non_blocking=True).contiguous()
        tar_images = train_data["tar_image"].to(device, non_blocking=True).contiguous()
        src_images = et_channel(src_images)
        tar_images = et_channel(tar_images)
        with accelerator.accumulate(autoencoder), accelerator.autocast():
            src_reconstruction, src_z_mu, src_z_sigma = autoencoder(src_images)
            tar_reconstruction, tar_z_mu, tar_z_sigma = autoencoder(tar_images)

            src_reconstruction = src_reconstruction * 2000 - 1000
            tar_reconstruction = tar_reconstruction * 2000 - 1000

            # reconstruction loss
            src_reconstruction = et_channel(src_reconstruction)
            tar_reconstruction = et_channel(tar_reconstruction)

            src_loss = loss_pt(src_reconstruction.flatten(start_dim=0, end_dim=1),
                               src_images.flatten(start_dim=0, end_dim=1))
            tar_loss = loss_pt(tar_reconstruction.flatten(start_dim=0, end_dim=1),
                               tar_images.flatten(start_dim=0, end_dim=1))
            recon_loss = (src_loss + tar_loss) / 2

            # KL divergence loss
            z_mu, z_sigma = torch.cat([src_z_mu, tar_z_mu], dim=0), torch.cat([src_z_sigma, tar_z_sigma], dim=0)
            z_mu_mean = z_mu.mean(dim=0, keepdim=True)
            z_var_mean = z_sigma.pow(2).mean(dim=0, keepdim=True) + (z_mu - z_mu_mean).pow(2).mean(dim=0, keepdim=True)
            z_sigma_mean = torch.sqrt(z_var_mean)
            kl = kl_loss(z_mu_mean, z_sigma_mean)

            # MMD loss
            src_eps = torch.randn(128, *src_z_mu.shape, device=src_z_mu.device, dtype=src_z_mu.dtype)
            src_z_samples = src_z_mu.unsqueeze(0) + src_z_sigma.unsqueeze(0) * src_eps  # (k, B, C, D, H, W)
            src_z_samples = src_z_samples.reshape(128 * src_z_mu.shape[0], *src_z_mu.shape[1:])  # (k*B, C, D, H, W)
            tar_eps = torch.randn(128, *tar_z_mu.shape, device=tar_z_mu.device, dtype=tar_z_mu.dtype)
            tar_z_samples = tar_z_mu.unsqueeze(0) + tar_z_sigma.unsqueeze(0) * tar_eps  # (k, B, C, D, H, W)
            tar_z_samples = tar_z_samples.reshape(128 * tar_z_mu.shape[0], *tar_z_mu.shape[1:])  # (k*B, C, D, H, W)
            src_b, src_c, src_d, src_h, src_w = src_z_samples.shape
            tar_b, tar_c, tar_d, tar_h, tar_w = tar_z_samples.shape
            src_z_samples = src_z_samples.view(src_b, src_c * src_d * src_h * src_w)
            tar_z_samples = tar_z_samples.view(tar_b, src_c * tar_d * tar_h * tar_w)
            cat_z_samples = torch.cat([src_z_samples, tar_z_samples], dim=0)
            mmd_loss = sliced_energy_loss(cat_z_samples, num_projections=65536)

            loss = recon_loss + mmd_loss + kl
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if accelerator.is_main_process:
            if _iter % gradient_accumulation_steps == 0:
                logger.info(
                    "[{0}] epoch {1}, iter {2}/{3}, recon loss: {4:.4f}, mmd loss: {5:.4f}, kl loss: {6:.4f}, lr: {7:.12f}.".format(
                        str(datetime.now())[:19],
                        epoch + 1,
                        _iter // gradient_accumulation_steps,
                        len(train_loader) // gradient_accumulation_steps,
                        recon_loss.item(),
                        mmd_loss.item(),
                        kl.item(),
                        current_lr
                    )
                )

    # Reduce loss for logging
    loss_torch = accelerator.reduce(loss_torch, reduction="sum")
    lr_scheduler.step()
    return loss_torch


def save_checkpoint(
        autoencoder: torch.nn.Module,
        args: argparse.Namespace,
        accelerator: Accelerator
) -> None:
    # Wait for everyone before saving
    accelerator.wait_for_everyone()

    # Unwrap model to get clean state dict
    unwrapped_autoencoder = accelerator.unwrap_model(autoencoder)

    # Only save on main process
    if accelerator.is_main_process:
        torch.save(unwrapped_autoencoder.state_dict(), args.trained_autoencoder_path.replace(".pt", f"_my.pt"))
def diff_model_train(
        env_config_path: str, model_config_path: str, model_def_path: str
) -> None:
    # Initialize Accelerator
    # mixed_precision can be "no", "fp16", "bf16".
    # It is recommended to configure this via `accelerate config` CLI or pass arg here.
    args = load_config(env_config_path, model_config_path, model_def_path)
    accelerator = Accelerator(gradient_accumulation_steps=args.vae_train["autoencoder_train"]["gradient_accumulation_steps"],
                              step_scheduler_with_optimizer=False)

    args.autoencoder_def["num_splits"] = 1
    args.autoencoder_def["save_mem"] = False
    args.autoencoder_def["norm_float16"] = False
    logger = setup_logging("VAE training", rk_filter=False)

    # Log device info
    logger.info(f"Process {accelerator.process_index} using device: {accelerator.device}")

    if accelerator.is_main_process:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Load UNet (Move to device logic handled by prepare, but we load first)
    autoencoder = load_model(args, accelerator, logger)

    filenames_train = load_filenames(args.json_data_list, mode="training")
    filenames_val = load_filenames(args.json_data_list, mode="validation")
    filenames_test = load_filenames(args.json_data_list, mode="test")
    if accelerator.is_main_process:
        logger.info(f"num_files_train: {len(filenames_train)}")
        logger.info(f"num_files_val: {len(filenames_val)}")
        logger.info(f"num_files_test: {len(filenames_test)}")

    # Prepare file list
    train_files = prepare_file_list(filenames_train, args.data_base_dir, "training")
    val_files = prepare_file_list(filenames_val, args.data_base_dir, "validation")

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

    # Create DataLoader with local subset
    train_transforms = VAE_Transform(
        is_train=True,
        random_aug=args.vae_train["data_option"]["random_aug"],  # whether apply random data augmentation for training
        k=4,  # patches should be divisible by k
        patch_size=args.vae_train["autoencoder_train"]["patch_size"],
        val_patch_size=args.vae_train["autoencoder_train"]["patch_size"],
        output_dtype=torch.float32,  # final data type
        spacing_type=args.vae_train["data_option"]["spacing_type"],
        spacing=args.vae_train["data_option"]["spacing"],
        image_keys=["src_image", "tar_image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    train_loader = prepare_data(
        train_files,
        data_transforms=train_transforms,
        cache_rate=args.vae_train["autoencoder_train"]["cache_rate"],
        num_workers=args.vae_train["autoencoder_train"]["num_workers"],
        batch_size=args.vae_train["autoencoder_train"]["batch_size"],
    )
    val_transforms = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        val_patch_size=args.vae_train["autoencoder_train"]["patch_size"],  # if None, will validate on whole image volume
        output_dtype=torch.float32,  # final data type
        image_keys=["src_image", "tar_image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )

    val_loader = prepare_data(
        val_files,
        data_transforms=val_transforms,
        cache_rate=args.vae_train["autoencoder_train"]["cache_rate"],
        num_workers=args.vae_train["autoencoder_train"]["num_workers"],
        batch_size=args.vae_train["autoencoder_train"]["batch_size"],
    )

    optimizer = create_optimizer(autoencoder, args.vae_train["autoencoder_train"]["lr"])

    # Calculate steps based on local dataset size (approximate)
    total_steps = args.vae_train["autoencoder_train"]["n_epochs"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = Loss().to(accelerator.device)

    # Prepare everything with Accelerate
    # NOTE: We do NOT pass train_loader here because we manually partitioned the dataset
    # for CacheDataset efficiency. Accelerate would try to shard it again or replace sampler.
    # We will manually handle device movement in the loop.
    autoencoder, optimizer, lr_scheduler = accelerator.prepare(
        autoencoder, optimizer, lr_scheduler
    )

    for epoch in range(args.vae_train["autoencoder_train"]["n_epochs"]):
        start_time = time.perf_counter()
        loss_torch = train_one_epoch(
            epoch,
            autoencoder,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            accelerator,
            logger,
            args.vae_train["autoencoder_train"]["gradient_accumulation_steps"]
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        loss_torch = loss_torch.tolist()

        # Calculate average loss
        loss_torch_epoch = loss_torch[0] / loss_torch[1]

        if accelerator.is_main_process:
            logger.info(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, time taken: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
        save_checkpoint(autoencoder, args, accelerator)
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
