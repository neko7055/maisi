# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def make_2tuple(x: tuple[int, int] | int):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return x, x

def make_3tuple(x: tuple[int, int, int] | int):
    if isinstance(x, tuple):
        assert len(x) == 3
        return x

    assert isinstance(x, int)
    return x, x, x

def legendre_time_embedding(timesteps: torch.Tensor, embedding_dim: int):
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    x = 2.0 * timesteps - 1.0
    out = torch.zeros(x.shape[0], embedding_dim+1, device=x.device, dtype=x.dtype)
    out[:, 0] = 1.0
    out[:, 1] = x
    for n in range(1, embedding_dim):
        out[:, n + 1] = ((2 * n + 1) * x * out[:, n] - n * out[:, n - 1]) / (n + 1)
    out = out[:, 1:]
    return out


def generate_3d_legendre_pe(coords: torch.Tensor, max_degree: int) -> torch.Tensor:
    """
    生成三維勒讓德多項式位置編碼 (3D Legendre Polynomial Encoding)

    Args:
        coords: 形狀為 (N, 3) 的 Tensor，代表 N 個空間點的 (x, y, z) 座標。
                數值必須歸一化在 [-1, 1] 之間。
        max_degree: 容許的最大總階數 L_max (i + j + k <= L_max)。

    Returns:
        torch.Tensor: 形狀為 (N, C) 的特徵矩陣，C 為有效組合的數量。
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    # 1. 預先計算並儲存三個軸的一維勒讓德多項式
    # p_1d 的形狀為 (max_degree + 1, N, 3)，最後一個維度對應 x, y, z
    p_1d = torch.zeros(max_degree + 1, N, 3, device=device, dtype=dtype)

    # P_0(x) = 1
    p_1d[0] = 1.0

    if max_degree >= 1:
        # P_1(x) = x
        p_1d[1] = coords

    # 利用 Bonnet 遞迴公式計算更高階的 1D 多項式，確保數值穩定
    for n in range(1, max_degree):
        p_1d[n + 1] = ((2 * n + 1) * coords * p_1d[n] - n * p_1d[n - 1]) / (n + 1)

    # 2. 進行張量積組合 (Tensor Product) 並加入截斷條件
    features = []

    # 窮舉所有可能的 (i, j, k) 階數組合
    for i in range(max_degree + 1):
        for j in range(max_degree + 1):
            for k in range(max_degree + 1):
                # 總階數截斷：確保 i + j + k 不超過 max_degree，以避免維度與高頻雜訊爆炸
                if i + j + k <= max_degree:
                    # P_{i,j,k}(x,y,z) = P_i(x) * P_j(y) * P_k(z)
                    term = p_1d[i, :, 0] * p_1d[j, :, 1] * p_1d[k, :, 2]
                    features.append(term.unsqueeze(1))

    # 3. 將所有合法的特徵向量拼接成單一矩陣
    # 輸出形狀為 (N, C)
    out = torch.cat(features, dim=1)
    return out

def layer_norm(x, eps=1e-6):
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) * torch.rsqrt(s + eps)
    return x

class AdaLN(nn.Module):
    def __init__(self, num_channels,
                 emb_dim,
                 eps=1e-6,
                 act_fn: Callable[..., nn.Module]=nn.SiLU):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(emb_dim, num_channels * 2)
        if act_fn is None:
            self.act = None
        else:
            self.act = act_fn()

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        while len(emb.shape) < len(x.shape):
            emb = emb.unsqueeze(-1)
        scale, shift = torch.chunk(emb, 2, dim=1)
        h = layer_norm(x, self.eps)
        return h * (1 + scale) + shift

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        ffn_ratio: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv3d(in_features, in_features * ffn_ratio, kernel_size=1, stride=1, padding=0, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(in_features * ffn_ratio, in_features, kernel_size=1, stride=1, padding=0, bias=bias, device=device)
        self._init_weights()


    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        torch.nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

@torch.compile(mode="max-autotune", fullgraph=True)
def linear_attn(Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                eps: float = 1e-5):
    # Step 1: phi() kernel，讓數值非負
    Q_phi = F.elu(Q) + 1.0  # (B, heads, head_dim,  H, W, D)
    K_phi = F.elu(K) + 1.0  # (B, heads, head_dim,  H, W, D)
    KV = torch.einsum("b h d x y z, b h v x y z -> b h d v", K_phi, V)
    numerator = torch.einsum("b h d x y z, b h d v -> b h v x y z", Q_phi, KV)
    K_sum = K_phi.sum(dim=[-3, -2, -1])
    Z = torch.einsum("b h d x y z, b h d -> b h x y z", Q_phi, K_sum)  # (B, heads, N_q)
    Z = Z + eps
    out = numerator / Z.unsqueeze(2)
    return out

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pe_dim = pe_dim
        #self.pe_proj = nn.Conv3d(self.pe_dim, dim, kernel_size=1, stride=1, padding=0, bias=False, device=device)
        self.q_pe_proj = nn.Conv3d(self.pe_dim, dim, kernel_size=1, stride=1, padding=0, bias=False, device=device)
        self.k_pe_proj = nn.Conv3d(self.pe_dim, dim, kernel_size=1, stride=1, padding=0, bias=False, device=device)
        self.qkv = nn.Conv3d(dim, 3 * dim, kernel_size=1,padding=0, bias=qkv_bias, device=device)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=proj_bias, device=device)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        torch.nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H, W, D)
        q, k, v = torch.unbind(qkv, 1)
        q_pe = self.q_pe_proj(pe).view(1, self.num_heads, self.head_dim, H, W, D)
        k_pe = self.k_pe_proj(pe).view(1, self.num_heads, self.head_dim, H, W, D)
        q = q + q_pe
        k = k + k_pe
        #attn_out = linear_attn(q, k, v)
        q, k, v = q.flatten(-3).permute(0, 3, 1, 2), k.flatten(-3).permute(0, 3, 1, 2), v.flatten(-3).permute(0, 3, 1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v).permute(0, 2, 3, 1).contiguous()
        attn_out = attn_out.view(B, C, H, W, D)
        attn_out = self.proj(attn_out)
        return attn_out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        num_heads: int,
        temb_channels: int,
        ffn_ratio: int = 4,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        ffn_bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLN(dim, temb_channels)
        self.attn = SelfAttention(
            dim,
            pe_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            device=device,
        )
        self.norm2 = AdaLN(dim, temb_channels)
        self.mlp = Mlp(
            in_features=dim,
            ffn_ratio=ffn_ratio,
            act_layer=partial(nn.GELU, approximate="tanh"),
            bias=ffn_bias,
            device=device,
        )
    def forward(self, x: torch.Tensor, emb:torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        x_attn = x + self.attn(self.norm1(x, emb), pe=pe)
        x_ffn = x_attn + self.mlp(self.norm2(x_attn, emb))
        return x_ffn


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size,
                 patch_size: Union[int, Tuple[int, int, int]],
                 out_channels,
                 temb_channels: int,):
        super().__init__()
        self.patch_size = make_3tuple(patch_size)
        #kernel_size = (self.patch_size[0] * 3, self.patch_size[1] * 3, self.patch_size[2] * 3)
        #padding_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
        self.norm = AdaLN(hidden_size, temb_channels)
        self.linear = nn.Conv3d(hidden_size,
                                self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * out_channels,#
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.linear.weight, 0)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, emb):
        x = self.linear(self.norm(x, emb))
        return x

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 4,
        embed_dim: int = 768,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        patch_HWD: Tuple[int, int, int] = make_3tuple(patch_size)

        self.patch_size = patch_HWD
        kernel_size = (self.patch_size[0]*3, self.patch_size[1]*3, self.patch_size[2]*3)
        padding_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2])

        #kernel_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
        #padding_size = (0, 0, 0)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_HWD, padding=padding_size)
    #     self._init_weights()
    #
    # def _init_weights(self):
    #    w = self.proj.weight.data
    #    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #    if self.proj.bias is not None:
    #        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x

def generate_patch_mask_3d(b, h, w, d, mask_ratio=0.5, device="cuda"):
    mask = torch.bernoulli(
        torch.full((b,1, h, w, d), mask_ratio, device=device)
    )
    return mask

def patch_mask_to_voxel_mask(mask, pt, ph, pw):
    """
    mask: (B, Nt, Nh, Nw)
    return:
        voxel_mask: (B, 1, T, H, W)
    """

    B, Nt, Nh, Nw = mask.shape

    # expand to patch volume
    mask = mask[:, :, :, :, None, None, None]

    mask = mask.expand(
        B, Nt, Nh, Nw,
        pt, ph, pw
    )

    # rearrange to voxel space
    mask = mask.permute(0, 1, 4, 2, 5, 3, 6)

    mask = mask.reshape(
        B,
        Nt * pt,
        Nh * ph,
        Nw * pw
    )

    return mask.float()

class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        patch_size = (4,4,4),
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
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
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
                                      patch_size,
                                      self.out_channels,
                                      temb_channels,)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))

    def _get_pe(self, H, W, D, device):
        x_coords = torch.linspace(-1, 1, steps=H, device=device)
        y_coords = torch.linspace(-1, 1, steps=W, device=device)
        z_coords = torch.linspace(-1, 1, steps=D, device=device)
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        coords = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
        embeddings = generate_3d_legendre_pe(coords, self.legendre_max_degree)
        embeddings = embeddings.view(H, W, D, -1)
        embeddings = embeddings.permute(-1,0,1,2).unsqueeze(0)
        return embeddings


    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[Tensor, Any] | Tensor:
        x = self.patch_embed(x, emb)
        B, C, H, W, D = x.shape

        if self.training:
            mask = generate_patch_mask_3d(B, H, W, D, mask_ratio=0.5, device=x.device)
            x = x * (1-mask) + self.mask_token * mask
        pe = self._get_pe(H, W, D, x.device)
        for _, blk in enumerate(self.blocks):
            x = blk(x, emb, pe=pe)
        x = self.final_layer(x, emb)
        x = x.reshape(shape=(B,self.out_channels,
                             self.patch_embed.patch_size[0], self.patch_embed.patch_size[1],self.patch_embed.patch_size[2],
                             H, W, D,))
        x = torch.einsum('ncpqkhwd->nchpwqdk', x)
        x = x.reshape(shape=(x.shape[0],
                                self.out_channels,
                                H * self.patch_embed.patch_size[0],
                                W * self.patch_embed.patch_size[1],
                                D * self.patch_embed.patch_size[2]))
        return x
