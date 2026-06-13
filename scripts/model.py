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
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from functools import reduce, partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable, cast

import numpy as np
import torch
from monai.networks.blocks import SpatialAttentionBlock
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.type_conversion import convert_to_tensor

from .layers import PatchEmbed, TransformerBlock, FinalLayer, legendre_time_embedding, generate_3d_legendre_pe


class VisionTransformer(nn.Module):
    def __init__(
            self,
            *,
            patch_size=(2, 2, 2),
            in_chans: int = 3,
            out_chans: int = 3,
            embed_dim: int = 768,
            temb_channels: int = 256,
            depth: int = 12,
            num_heads: int = 12,
            ffn_ratio: int = 4,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            ffn_bias: bool = True,
            legendre_max_degree: int = 21,
            device: Any | None = None,
            **ignored_kwargs,
    ):
        super().__init__()
        del ignored_kwargs

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.legendre_max_degree = legendre_max_degree
        self.pe_dim = math.comb(legendre_max_degree + 3, 3)

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            TransformerBlock(
                dim=embed_dim,
                pe_dim=self.pe_dim,
                num_heads=num_heads,
                temb_channels=temb_channels,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        self.out_channels = out_chans
        self.final_layer = FinalLayer(embed_dim,
                                      self.patch_size,
                                      self.out_channels,
                                      temb_channels, )

    def _get_pe(self, H, W, D, device):
        x_coords = torch.linspace(-1, 1, steps=H, device=device)
        y_coords = torch.linspace(-1, 1, steps=W, device=device)
        z_coords = torch.linspace(-1, 1, steps=D, device=device)
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        coords = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
        embeddings = generate_3d_legendre_pe(coords, self.legendre_max_degree)
        embeddings = embeddings.view(H, W, D, -1)
        embeddings = embeddings.permute(-1, 0, 1, 2).unsqueeze(0)
        return embeddings

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[Tensor, Any] | Tensor:
        x = self.patch_embed(x)
        B, C, H, W, D = x.shape

        pe = self._get_pe(H, W, D, x.device)
        for _, blk in enumerate(self.blocks):
            x = blk(x, emb, pe=pe)
        x = self.final_layer(x, emb)
        x = x.reshape(shape=(B, self.out_channels,
                             self.final_layer.patch_size[0],
                             self.final_layer.patch_size[1],
                             self.final_layer.patch_size[2],
                             H, W, D,))
        x = torch.einsum('ncpqkhwd->nchpwqdk', x)
        x = x.reshape(shape=(x.shape[0],
                             self.out_channels,
                             H * self.final_layer.patch_size[0],
                             W * self.final_layer.patch_size[1],
                             D * self.final_layer.patch_size[2]))
        return x

class Net(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_emb_dim: int,
        time_embed_dim: int,
        include_spacing_input: bool = False,
    ):
        super().__init__()
        self.include_spacing_input = include_spacing_input

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_emb_dim = cond_emb_dim
        self.time_embed_dim = time_embed_dim



        # time
        self.time_embed = self._create_embedding_module(self.time_embed_dim, cond_emb_dim)

        if self.include_spacing_input:
            self.spacing_layer = self._create_embedding_module(3, self.cond_emb_dim)
            self.cond_emb_dim += self.cond_emb_dim

        self.vit = VisionTransformer(patch_size=(2, 2, 2),
                                     in_chans=4,
                                     out_chans=4,
                                     embed_dim=1024,
                                     temb_channels=self.cond_emb_dim,
                                     num_heads=16)


    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim),
                              nn.SiLU(),
                              nn.Linear(embed_dim, embed_dim))
        return model

    def _get_time_and_class_embedding(self, x, timesteps):
        t_emb = legendre_time_embedding(timesteps, self.time_embed_dim)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)
        return emb

    def _get_input_embeddings(self, emb, spacing):
        if self.include_spacing_input:
            _emb = self.spacing_layer(spacing)
            emb = torch.cat((emb, _emb), dim=1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        spacing_tensor: torch.Tensor | None = None,
        **ignored_kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x: Input tensor of shape (N, C, SpatialDims).
            timesteps: Timestep tensor of shape (N,).
            context: Context tensor of shape (N, 1, ContextDim).
            class_labels: Class labels tensor of shape (N,).
            down_block_additional_residuals: Additional residual tensors for down blocks of shape (N, C, FeatureMapsDims).
            mid_block_additional_residual: Additional residual tensor for mid block of shape (N, C, FeatureMapsDims).
            top_region_index_tensor: Tensor representing top region index of shape (N, 4).
            bottom_region_index_tensor: Tensor representing bottom region index of shape (N, 4).
            spacing_tensor: Tensor representing spacing of shape (N, 3).

        Returns:
            A tensor representing the output of the UNet model.
        """
        # x: torch.Tensor = convert_to_tensor(x)
        emb = self._get_time_and_class_embedding(x, timesteps)
        emb = self._get_input_embeddings(emb, spacing_tensor)
        h = self.vit(x, emb)
        h_tensor: torch.Tensor = convert_to_tensor(h)
        return h_tensor

