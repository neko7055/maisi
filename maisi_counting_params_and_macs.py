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
import torch
from monai.bundle import ConfigParser
from thop import clever_format, profile

def define_instance(args: Namespace, instance_def_key: str) -> Any:
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)

def initialize_distributed(num_gpus: int) -> tuple:
    local_rank = 0
    world_size = 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return local_rank, world_size, device

def load_unet(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    return unet


if __name__ == "__main__":
    # load config
    model_def_path = "configs/config_network_rflow.json"
    args = argparse.Namespace()
    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    for k, v in model_def.items():
        setattr(args, k, v)
    print(args)
    local_rank, world_size, device = initialize_distributed(1)

    unet = load_unet(args, device)

    torch.set_float32_matmul_precision("highest")

    dummy_input = torch.randn(1, 4, 64, 128, 128).to(device)
    timesteps = torch.rand(1).to(device)
    spacing_tensor = torch.tensor([[1.0, 1.0, 1.0]]).to(device)
    modality_tensor = torch.tensor([0]).to(device)
    macs, params = profile(unet, inputs=(dummy_input,
                                         timesteps,
                                         None,
                                         modality_tensor,
                                         None,
                                         None,
                                         None,
                                         None,
                                         spacing_tensor))
    # Format the numbers into a readable format (e.g., 4.14 GMac, 25.56 MParams)
    macs_readable, params_readable = clever_format([macs, params], "%.3f")

    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
