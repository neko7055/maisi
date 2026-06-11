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

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True, fullgraph=True)
def lift_core_dwt(even: torch.Tensor, odd: torch.Tensor, dim: int):
    alpha = -1.586134342059924
    beta = -0.052980118572961
    gamma = 0.882911075530934
    delta = 0.443506852043971
    K = 1.149604398860241

    odd = odd + alpha * (even + torch.roll(even, shifts=-1, dims=dim))
    even = even + beta * (odd + torch.roll(odd, shifts=1, dims=dim))
    odd = odd + gamma * (even + torch.roll(even, shifts=-1, dims=dim))
    even = even + delta * (odd + torch.roll(odd, shifts=1, dims=dim))

    return even * K, odd * (1.0 / K)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True, fullgraph=True)
def lift_core_idwt(low: torch.Tensor, high: torch.Tensor, dim: int):
    alpha = -1.586134342059924
    beta = -0.052980118572961
    gamma = 0.882911075530934
    delta = 0.443506852043971
    K = 1.149604398860241

    even = low * (1.0 / K)
    odd = high * K

    even = even - delta * (odd + torch.roll(odd, shifts=1, dims=dim))
    odd = odd - gamma * (even + torch.roll(even, shifts=-1, dims=dim))
    even = even - beta * (odd + torch.roll(odd, shifts=1, dims=dim))
    odd = odd - alpha * (even + torch.roll(even, shifts=-1, dims=dim))

    return even, odd

def split_even_odd(x: torch.Tensor, dim: int):
    # 優化策略 1: 零拷貝視圖 (Zero-copy view) 取代 index_select
    shape = list(x.shape)
    shape[dim] = shape[dim] // 2
    shape.insert(dim + 1, 2)
    x_view = x.reshape(shape)
    return x_view.select(dim + 1, 0), x_view.select(dim + 1, 1)


def merge_even_odd(even: torch.Tensor, odd: torch.Tensor, dim: int):
    # 優化策略 2: 使用 C++ 底層高度優化的 stack 與 reshape，取代空矩陣賦值
    shape = list(even.shape)
    shape[dim] = shape[dim] * 2
    return torch.stack([even, odd], dim=dim + 1).reshape(shape)

class CDF97DWT3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, d, h, w = x.shape
        if d % 2 or h % 2 or w % 2:
            raise ValueError("D, H, W must be even for CDF 9/7 DWT.")

        # 針對 D, H, W 三個維度依序執行 Lifting 分解
        L_d, H_d = self.lift(x, dim=2)

        LL, LH = self.lift(L_d, dim=3)
        HL, HH = self.lift(H_d, dim=3)

        LLL, LLH = self.lift(LL, dim=4)
        LHL, LHH = self.lift(LH, dim=4)
        HLL, HLH = self.lift(HL, dim=4)
        HHL, HHH = self.lift(HH, dim=4)

        # 將 8 個 Sub-band 疊合成與您原本網路架構相容的型態 (N, C, 8, D/2, H/2, W/2)
        # out = torch.stack([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=2)
        out = (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
        return out

    def lift(self, x, dim):
        # 使用極速拆分與融合運算核心
        even, odd = split_even_odd(x, dim)
        return lift_core_dwt(even, odd, dim)


class CDF97IDWT3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coeffs):
        # 取出 8 個 Sub-band
        # LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = torch.unbind(coeffs, dim=2)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = coeffs

        # 依照完全相反順序 Inverse Lifting (W -> H -> D)
        LL = self.ilift(LLL, LLH, dim=4)
        LH = self.ilift(LHL, LHH, dim=4)
        HL = self.ilift(HLL, HLH, dim=4)
        HH = self.ilift(HHL, HHH, dim=4)

        L_d = self.ilift(LL, LH, dim=3)
        H_d = self.ilift(HL, HH, dim=3)

        x = self.ilift(L_d, H_d, dim=2)
        return x

    def ilift(self, low, high, dim):
        # 使用極速還原與交疊核心
        even, odd = lift_core_idwt(low, high, dim)
        return merge_even_odd(even, odd, dim)

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

class WaveletSelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            num_wt_apply=2,
            device=None,
    ) -> None:
        super().__init__()
        self.dwt = CDF97DWT3D()
        self.idwt = CDF97IDWT3D()
        self.layer_norm = nn.LayerNorm(dim)
        self.num_wt_apply = num_wt_apply
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, device=device)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.bandwidth_pe_shift = nn.Parameter(torch.zeros(1,8**self.num_wt_apply,dim).to(device))
        self.bandwidth_pe_scale = nn.Parameter(torch.ones(1,8**self.num_wt_apply,dim).to(device))
    def _attn(self, x):
        qkv = self.qkv(self.bandwidth_pe_scale * x + self.bandwidth_pe_shift)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = self.proj(attn_out)
        return attn_out
    def apply_dwt(self, F):
        for _ in range(self.num_wt_apply):
            F = torch.cat(self.dwt(F), dim=1)
        F = torch.chunk(F, 8 ** self.num_wt_apply, dim=1)
        return F
    def apply_idwt(self, F):
        F = torch.cat(F, dim=1)
        for i in range(self.num_wt_apply):
            F = torch.chunk(F, 8, dim=1)
            F = self.idwt(F)
        return F
    def forward(self, x: torch.Tensor):
        x = self.apply_dwt(x)
        x = torch.stack(x, dim=-1)
        B, C, H, W, D, N = x.shape
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, N, C)
        x = x + self._attn(self.layer_norm(x))
        x = x.view(B, H, W, D, N, C).permute(0, 5, 1, 2, 3, 4).contiguous()
        x = torch.unbind(x, dim=-1)
        x = self.apply_idwt(x)
        return x

class FourierGlobalFilter(nn.Module):
    def __init__(
        self,
        dim: int,
        pe_dim: int,
    ):
        super().__init__()
        self.pe_dim = pe_dim
        self.attn_map = nn.Conv3d(self.pe_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.zeros_(self.attn_map.weight)
        if self.attn_map.bias is not None:
            nn.init.zeros_(self.attn_map.bias)
    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False, dtype=torch.float32):
            attn = self.attn_map(pe)
            attn = torch.fft.rfftn(attn, dim=(-3, -2, -1), norm="ortho")
            x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
            x = x * attn
            x = torch.fft.irfftn(x, s=(H, W, D), dim=(-3, -2, -1), norm="ortho")
        return x

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
        self.norm1_1 = AdaLN(dim, temb_channels)
        self.wavelet_attn = WaveletSelfAttention(dim,
                                                 num_heads=num_heads,
                                                 qkv_bias=qkv_bias,
                                                 proj_bias=proj_bias,
                                                 device=device, )
        self.norm1_2 = AdaLN(dim, temb_channels)
        self.fft_filter = FourierGlobalFilter(
            dim,
            pe_dim,
        )
        self.norm2 = AdaLN(dim, temb_channels)
        self.mlp = Mlp(
            in_features=dim,
            ffn_ratio=ffn_ratio,
            act_layer=partial(nn.GELU, approximate="tanh"),
            bias=ffn_bias,
            device=device,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        x_attn = self.wavelet_attn(self.norm1_1(x, emb)) + self.fft_filter(self.norm1_2(x, emb), pe=pe)
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