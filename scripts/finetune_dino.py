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
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

import monai
import torch
from accelerate import Accelerator  # Import Accelerate

from .diff_model_setting import load_config, setup_logging
from .infonce_loss import GaussianInfoNCELoss
from .inv_mlp import MLPInvertible

class SinLU(torch.nn.Module):
    def __init__(self):
        super(SinLU,self).__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return torch.sigmoid(torch.exp(self.beta)*x)*(x+torch.exp(self.a)*torch.sin(torch.exp(self.b)*x))

class ConvSwiGLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.conv_b = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.act = SinLU()
    def forward(self, x):
        a, b = self.conv_a(x), self.conv_b(x)
        return self.act(a) * b

class CNNAdapter(torch.nn.Module):
    def __init__(self, adapter_dim):
        super(CNNAdapter, self).__init__()
        #self.in_norm = torch.nn.InstanceNorm2d(1)
        #self.param = torch.nn.Parameter(torch.zeros(3))
        self.adapter = torch.nn.Sequential(
            ConvSwiGLU(1, adapter_dim, kernel_size=7, padding=3),
            torch.nn.Dropout2d(),
            ConvSwiGLU(adapter_dim, adapter_dim, kernel_size=5, padding=2),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(adapter_dim, 3, kernel_size=3, padding=1),
            #torch.nn.InstanceNorm2d(1, affine=True),
            torch.nn.Softsign(),
        )
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x = torch.cat([x, self.in_norm(x), torch.pow(self.sigmoid(x * torch.exp(self.param[0] + self.param[1])),torch.exp(self.param[2]))],dim=1)
        # x = self.in_norm(x)
        return (self.adapter(x / 1000.0) + 1) / 2 # Scale to [0, 1] before adapter

class DinoWithAdapter(torch.nn.Module):
    def __init__(self, dino_repo, dino_model, dino_weights, adapter_dim=64):
        super(DinoWithAdapter, self).__init__()
        self.dino_model = torch.hub.load(
            repo_or_dir=dino_repo,
            model=dino_model,
            source="local",
            weights=dino_weights,
        )
        for p in self.dino_model.parameters():
            p.requires_grad = False
        self.adapter = CNNAdapter(adapter_dim=adapter_dim)
        #self.register_buffer('mean',torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        #self.register_buffer('std',torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.log(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))
        self.distance_head = MLPInvertible(384)

    def forward(self, x):
        x = self.adapter(x)  # Scale to [0, 1] before adapter
        x = (x - self.mean) / torch.exp(self.std)
        features = self.dino_model.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        return self.distance_head(cls_token)

# ================= 1. 定義隨機切片邏輯 =================

class PairedSliceCacheDataset(torch.utils.data.Dataset):
    """
    從已快取的 3D 資料（每個 tensor 形狀假設為 (C, D, H, W)）中隨機抽取 N 個 2D slices。
    效能優化：先抽所有 (axis, index) 組合，對於同一 axis 批次使用一次 index_select（對每個 key 各一次）。
    注意：此實作不會自動 resize slices；若不同 axis 的切片尺寸不一致，stack 會失敗。
    """
    def __init__(self, cache_dataset: torch.utils.data.Dataset, num_slices: int, keys: Sequence[str], size: int):
        """
        Args:
            cache_dataset: 已處理並快取於 RAM 的 3D 資料集（每 item 為 dict of torch.Tensor，shape = (C, D, H, W)）
            num_slices: 每個 item 要隨機抽取的 slice 數量 N
            keys: 要從 item dict 中抽取的 key 列表（例如 ["image", "label"]）
            size: 在每個 axis 中可抽的索引上界（假設每個 axis 的長度至少為 size）
        """
        self.cache_dataset = cache_dataset
        self.num_slices = int(num_slices)
        self.keys = list(keys)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.cache_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1) 取得快取 item（每個 tensor 預期為 (C,D,H,W)）
        item_3d: Dict[str, torch.Tensor] = self.cache_dataset[idx]

        N = self.num_slices
        # 2) 一次性抽樣 N 個 (axis, slice_index)
        axes = np.random.randint(0, 3, size=N)           # values in {0,1,2}
        slice_indices = np.random.randint(0, self.size, size=N)

        # 3) 為每個 key 建立一個暫存 list (長度 N) 用以存放對應位置的 slice
        out_slices_per_key: Dict[str, List[torch.Tensor]] | Dict[str, List[None]] = {
            k: [None] * N for k in self.keys
        }

        # 4) 對三個 axis 分別處理（每個 axis 對每個 key 做一次 index_select）
        for axis in (0, 1, 2):
            positions = np.where(axes == axis)[0]  # 哪些抽樣位置選到此 axis
            if positions.size == 0:
                continue

            idxs = slice_indices[positions]  # numpy array of shape (M,)
            # 逐 key 做一次 index_select（保留 device）
            for k in self.keys:
                tensor = item_3d[k]
                if tensor.dim() != 4:
                    raise ValueError(f"Expected tensor with 4 dims (C,D,H,W) for key '{k}', got shape {tensor.shape}")

                device = tensor.device
                idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=device)  # shape (M,)
                tensor_dim = axis + 1  # map axis (0/1/2) -> tensor dim (1/2/3)

                # selected shape: same as tensor but length at tensor_dim becomes M
                selected = torch.index_select(tensor, dim=tensor_dim, index=idxs_t)

                # 將 selected 的 M 維移到第一維，得到 (M, C, H, W)
                permute_order = [tensor_dim, 0] + [d for d in range(4) if d not in (tensor_dim, 0)]
                slices = selected.permute(*permute_order).contiguous()  # (M, C, H, W)

                # 放回對應的原始位置
                for j, pos in enumerate(positions):
                    out_slices_per_key[k][pos] = slices[j]

        # 5) 確認每個位置都被填滿，並將 list stack 成張量 (N, C, H, W)
        output: Dict[str, torch.Tensor] = {}
        for k in self.keys:
            lst = out_slices_per_key[k]
            # 檢查是否有未填入的元素
            missing = [i for i, s in enumerate(lst) if s is None]
            if missing:
                raise RuntimeError(f"Missing slices for key '{k}' at positions {missing}")

            # 檢查 shape 是否一致（若不一致，stack 會 raise）
            # 直接 stack（若形狀不一致，torch.stack 會報錯）
            output[k] = torch.stack(lst, dim=0)

        return output

def prepare_file_list(filenames,data_base_dir, mode):
    # Prepare file list
    files_list = []
    for _i in range(len(filenames)):
        str_src_img = os.path.join(data_base_dir, mode, filenames[_i]["src_image"])
        str_tar_img = os.path.join(data_base_dir, mode, filenames[_i]["tar_image"])
        if (not os.path.exists(str_src_img)) and (not os.path.exists(str_tar_img)):
            continue

        files_i = {"src_image": str_src_img,
                   "tar_image": str_tar_img,}
        files_list.append(files_i)
    return files_list


def load_filenames(data_list_path: str, mode: str) -> list:
    # (Same as original function)
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data[mode]
    return filenames_train

def prepare_data(
    train_files: list,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
) -> monai.data.DataLoader:
    # (Modified: removed device arg, rely on Accelerator or manual to() later)

    keys = ["src_image", "tar_image"]
    train_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=keys),
            monai.transforms.EnsureChannelFirstd(keys=keys),
            monai.transforms.Orientationd(keys=keys, axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )

    train_ds = PairedSliceCacheDataset(
        monai.data.CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        ),
        num_slices=8,
        keys=["src_image", "tar_image"],
        size=512,
    )


    return monai.data.DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True)


def load_model(
    args: argparse.Namespace, accelerator: Accelerator, logger: logging.Logger
) -> torch.nn.Module:
    # Load model to CPU first, let Accelerate handle movement
    model = DinoWithAdapter(
        args.dinov3_repo_path,
        args.dinov3_model_type,
        args.dinov3_model_path,
    )

    # # Optional: Convert BatchNorm to SyncBatchNorm for DDP
    # if accelerator.num_processes > 1:
    #     unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
    #
    # if args.existing_ckpt_filepath is None:
    #     logger.info("Training from scratch.")
    # else:
    #     # Load checkpoint on CPU map_location
    #     checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location="cpu", weights_only=False)
    #     unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=False)
    #     logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return model



def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

def train_one_epoch(
        epoch: int,
        model: torch.nn.Module,
        train_loader: monai.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
        loss_pt: GaussianInfoNCELoss,
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

    model.train()

    # Iterate over loader
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]
        _iter += 1

        # Move data to device manually (since we didn't pass loader to accelerator.prepare)
        # This is because we pre-partitioned for CacheDataset memory efficiency.
        device = accelerator.device

        src_images = train_data["src_image"].to(device)
        tar_images = train_data["tar_image"].to(device)
        b, n, c, h, w = src_images.shape
        # print(b,n ,c,h,w)

        # Accelerate handles mixed precision automatically if configured
        with accelerator.accumulate(model):
            src_model_output = model(src_images.view(-1, c, h, w))  # Reshape to (b*n, c, h, w)
            tar_model_output = model(tar_images.view(-1, c, h, w))
            src_model_output = src_model_output.view(b, n, -1)  # Reshape back to (b, n, feature_dim)
            tar_model_output = tar_model_output.view(b, n, -1)

            loss = loss_pt.forward_symmetric(src_model_output,tar_model_output)

            # Replaced backward with accelerator
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if accelerator.is_main_process:
            if _iter % gradient_accumulation_steps == 0:
                logger.info(
                    "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                        str(datetime.now())[:19], epoch + 1, _iter//gradient_accumulation_steps, len(train_loader)//gradient_accumulation_steps, loss.item(), current_lr
                    )
                )

    # Reduce loss for logging
    loss_torch = accelerator.reduce(loss_torch, reduction="sum")
    return loss_torch

def inference_adapter(
    epoch: int,
    args: argparse.Namespace,
    model: torch.nn.Module,
    test_loader: monai.data.DataLoader,
    accelerator: Accelerator,
):
    model.eval()
    data_idx = 0
    os.makedirs(f"{args.work_dir}/dino_infer/epoch_{epoch + 1}", exist_ok=True)
    with torch.no_grad() and torch.inference_mode():
        for test_data in test_loader:
            device = accelerator.device
            # pick the first slice from each batch item for inference (shape: (b, c, h, w))
            src_images = test_data["src_image"].to(device)[:, 0, :, :, :]
            tar_images = test_data["tar_image"].to(device)[:, 0, :, :, :]
            # tar_images = tar_images - src_images
            # inference and to CPU for later processing
            src_model_output = (model.module.adapter(src_images) * 255).byte().cpu()
            tar_model_output = (model.module.adapter(tar_images) * 255).byte().cpu()
            # ct clip and normalize for visualization
            src_images = torch.clamp(src_images, -1000, 1000) / 1000.0
            src_images = (src_images + 1) / 2.0 * 255.0
            src_images = src_images.repeat(1, 3, 1, 1)
            src_images = src_images.byte().cpu()

            tar_images = torch.clamp(tar_images, -1000, 1000) / 1000.0
            tar_images = (tar_images + 1) / 2.0 * 255.0
            tar_images = tar_images.repeat(1, 3, 1, 1)
            tar_images = tar_images.byte().cpu()

            # save to png
            for i in range(src_images.shape[0]):
                # fix idx with precess index and epoch to avoid overwrite
                idx_offset = (
                    accelerator.process_index * len(test_loader.dataset) + data_idx
                )
                save_path_src = f"{args.work_dir}/dino_infer/epoch_{epoch + 1}/src_image_{idx_offset}.png"
                save_path_tar = f"{args.work_dir}/dino_infer/epoch_{epoch + 1}//tar_image_{idx_offset}.png"
                save_path_src_origin = f"{args.work_dir}/dino_infer/epoch_{epoch + 1}//src_image_origin_{idx_offset}.png"
                save_path_tar_origin = f"{args.work_dir}/dino_infer/epoch_{epoch + 1}//tar_image_origin_{idx_offset}.png"

                pil_img_src = Image.fromarray(
                    src_model_output[i].permute(1, 2, 0).numpy()
                )
                pil_img_src.save(save_path_src)

                pil_img_tar = Image.fromarray(
                    tar_model_output[i].permute(1, 2, 0).numpy()
                )
                pil_img_tar.save(save_path_tar)
                pil_img_src_org = Image.fromarray(
                    src_images[i].permute(1, 2, 0).numpy()
                )
                pil_img_src_org.save(save_path_src_origin)
                pil_img_tar_org = Image.fromarray(
                    tar_images[i].permute(1, 2, 0).numpy()
                )
                pil_img_tar_org.save(save_path_tar_origin)
                data_idx += 1
    model.train()


def save_checkpoint(
        epoch: int,
        model: torch.nn.Module,
        loss_torch_epoch: float,
        num_train_timesteps: int,
        ckpt_folder: str,
        args: argparse.Namespace,
        accelerator: Accelerator
) -> None:
    # Wait for everyone before saving
    accelerator.wait_for_everyone()

    # Unwrap model to get clean state dict
    unwrapped_model = accelerator.unwrap_model(model)

    # Only save on main process
    if accelerator.is_main_process:
        torch.save(
            {
                "epoch": epoch + 1,
                "loss": loss_torch_epoch,
                "num_train_timesteps": num_train_timesteps,
                "model_state_dict": unwrapped_model.state_dict(),
            },
            f"{ckpt_folder}/{args.dinov3_adapter_filename}_{epoch+1}.pt",
        )


def diff_model_train(
        env_config_path: str, model_config_path: str, model_def_path: str
) -> None:
    # Initialize Accelerator
    # mixed_precision can be "no", "fp16", "bf16".
    # It is recommended to configure this via `accelerate config` CLI or pass arg here.
    args = load_config(env_config_path, model_config_path, model_def_path)
    accelerator = Accelerator(gradient_accumulation_steps=args.dino_finetune["gradient_accumulation_steps"])


    logger = setup_logging("training dino dapter")

    # Log device info
    logger.info(f"Process {accelerator.process_index} using device: {accelerator.device}")

    if accelerator.is_main_process:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Load UNet (Move to device logic handled by prepare, but we load first)
    model = load_model(args, accelerator, logger)

    filenames_train = load_filenames(args.json_data_list, mode="training")
    filenames_val = load_filenames(args.json_data_list, mode="validation")
    filenames_test = load_filenames(args.json_data_list, mode="test")
    if accelerator.is_main_process:
        logger.info(f"num_files_train: {len(filenames_train)}")
        logger.info(f"num_files_val: {len(filenames_val)}")
        logger.info(f"num_files_test: {len(filenames_test)}")

    # Prepare file list
    train_files = prepare_file_list(filenames_train,args.data_base_dir, "training")
    val_files = prepare_file_list(filenames_val,args.data_base_dir, "validation")

    # Partition dataset BEFORE creating CacheDataset to save RAM
    # Accelerate makes this easy by giving us num_processes and process_index
    train_files = monai.data.partition_dataset(
        data=train_files,
        shuffle=True,
        num_partitions=accelerator.num_processes,
        even_divisible=True
    )[accelerator.process_index]

    val_files = monai.data.partition_dataset(
        data=val_files,
        shuffle=True,
        num_partitions=accelerator.num_processes,
        even_divisible=True
    )[accelerator.process_index]

    # Create DataLoader with local subset
    train_loader = prepare_data(
        train_files,
        args.dino_finetune["cache_rate"],
        batch_size=args.dino_finetune["batch_size"],
        num_workers=8,
    )

    val_loader = prepare_data(
        val_files,
        args.dino_finetune["cache_rate"],
        batch_size=args.dino_finetune["batch_size"],
        num_workers=8,
    )

    optimizer = create_optimizer(model, args.dino_finetune["lr"])

    # Calculate steps based on local dataset size (approximate)
    total_steps = (args.dino_finetune["n_epochs"] * len(train_loader.dataset)) / args.dino_finetune["batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = GaussianInfoNCELoss().to(accelerator.device) # GaussianInfoNCELoss

    # Prepare everything with Accelerate
    # NOTE: We do NOT pass train_loader here because we manually partitioned the dataset
    # for CacheDataset efficiency. Accelerate would try to shard it again or replace sampler.
    # We will manually handle device movement in the loop.
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    torch.set_float32_matmul_precision("highest")

    for epoch in range(args.dino_finetune["n_epochs"]):
        start_time = time.perf_counter()
        loss_torch = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            accelerator,
            logger,
            args.dino_finetune["gradient_accumulation_steps"]
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        loss_torch = loss_torch.tolist()

        # Calculate average loss
        loss_torch_epoch = loss_torch[0] / loss_torch[1]

        if accelerator.is_main_process:
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, time taken: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")

        inference_adapter(epoch, args, model, val_loader, accelerator)
        save_checkpoint(
            epoch,
            model,
            loss_torch_epoch,
            args.noise_scheduler["num_train_timesteps"],
            args.model_dir,
            args,
            accelerator
        )

    logger.info("Dino Adapter Training finished")
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
