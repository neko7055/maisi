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


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    def forward(self, x):
        x = layer_norm(x, self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
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

# ----------------- 效能優化區塊 (Lifting Scheme) -----------------
def lift_core_dwt(even: torch.Tensor, odd: torch.Tensor, dim: int):
    # CDF 5/3 (Bior 2.2) 提升係數
    alpha = -0.5
    beta = 0.25
    K = 1.4142135623730951  # math.sqrt(2)

    # Step 1: Predict
    odd = odd + alpha * (even + torch.roll(even, shifts=-1, dims=dim))

    # Step 2: Update
    even = even + beta * (odd + torch.roll(odd, shifts=1, dims=dim))

    # Step 3: Scale
    return even * K, odd * (1.0 / K)

def lift_core_idwt(low: torch.Tensor, high: torch.Tensor, dim: int):
    alpha = -0.5
    beta = 0.25
    K = 1.4142135623730951  # math.sqrt(2)

    # Step 1: Inverse Scale
    even = low * (1.0 / K)
    odd = high * K

    # Step 2: Inverse Update (反向減去 Update)
    even = even - beta * (odd + torch.roll(odd, shifts=1, dims=dim))

    # Step 3: Inverse Predict (反向減去 Predict)
    odd = odd - alpha * (even + torch.roll(even, shifts=-1, dims=dim))

    return even, odd


def split_even_odd(x: torch.Tensor, dim: int):
    # 零拷貝拆分陣列 (視圖操作)
    shape = list(x.shape)
    shape[dim] = shape[dim] // 2
    shape.insert(dim + 1, 2)
    x_view = x.reshape(shape)
    return x_view.select(dim + 1, 0), x_view.select(dim + 1, 1)


def merge_even_odd(even: torch.Tensor, odd: torch.Tensor, dim: int):
    # C++ 底層高速記憶體交錯排放
    shape = list(even.shape)
    shape[dim] = shape[dim] * 2
    return torch.stack([even, odd], dim=dim + 1).reshape(shape)


# ------------------------------------------------
# ==============================================================================
# 3. 空間堆疊工具 (處理張量拼接與拆分，以維持與原圖大小相同)
# ==============================================================================
def pack_octants(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
    """將 8 個次頻帶拼成一個完整的 3D 區塊"""
    top_left = torch.cat([LLL, LLH], dim=4)  # W 軸
    top_right = torch.cat([LHL, LHH], dim=4)
    top = torch.cat([top_left, top_right], dim=3)  # H 軸

    bot_left = torch.cat([HLL, HLH], dim=4)
    bot_right = torch.cat([HHL, HHH], dim=4)
    bot = torch.cat([bot_left, bot_right], dim=3)

    return torch.cat([top, bot], dim=2)  # D 軸


def unpack_octants(x):
    """將 1 個完整 3D 區塊拆分為 8 個次頻帶"""
    d2, h2, w2 = x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2
    LLL = x[:, :, :d2, :h2, :w2]
    LLH = x[:, :, :d2, :h2, w2:]
    LHL = x[:, :, :d2, h2:, :w2]
    LHH = x[:, :, :d2, h2:, w2:]
    HLL = x[:, :, d2:, :h2, :w2]
    HLH = x[:, :, d2:, :h2, w2:]
    HHL = x[:, :, d2:, h2:, :w2]
    HHH = x[:, :, d2:, h2:, w2:]
    return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

class CDF53DWT3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, d, h, w = x.shape
        if d % 2 or h % 2 or w % 2:
            raise ValueError("D, H, W must be even for CDF 5/3 DWT.")

        # 三個維度分別執行分解
        L_d, H_d = self.lift(x, dim=2)

        LL, LH = self.lift(L_d, dim=3)
        HL, HH = self.lift(H_d, dim=3)

        LLL, LLH = self.lift(LL, dim=4)
        LHL, LHH = self.lift(LH, dim=4)
        HLL, HLH = self.lift(HL, dim=4)
        HHL, HHH = self.lift(HH, dim=4)

        return (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)

    def lift(self, x, dim):
        even, odd = split_even_odd(x, dim)
        return lift_core_dwt(even, odd, dim)


class CDF53IDWT3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coeffs):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = coeffs

        # 反向組合 (W -> H -> D 順序)
        LL = self.ilift(LLL, LLH, dim=4)
        LH = self.ilift(LHL, LHH, dim=4)
        HL = self.ilift(HLL, HLH, dim=4)
        HH = self.ilift(HHL, HHH, dim=4)

        L_d = self.ilift(LL, LH, dim=3)
        H_d = self.ilift(HL, HH, dim=3)

        x = self.ilift(L_d, H_d, dim=2)
        return x

    def ilift(self, low, high, dim):
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
        attn = self.attn_map(pe)
        attn = torch.fft.rfftn(attn, dim=(-3, -2, -1), norm="ortho")
        x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        x = x * attn
        x = torch.fft.irfftn(x, s=(H, W, D), dim=(-3, -2, -1), norm="ortho")
        return x

class WaveletSelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            pe_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            wavelet_levels=2,
            device=None,
    ) -> None:
        super().__init__()
        self.dwt = CDF53DWT3D()
        self.idwt = CDF53IDWT3D()
        self.levels = wavelet_levels
        self.total_bands = 7 * self.levels + 1
        self.wavelet_feature_transform = nn.ModuleList([nn.Sequential(nn.Conv3d(dim, dim,1,bias=True),
                                                      LayerNorm(dim)) for _ in range(self.total_bands)])
        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        self.pe_conv = nn.Conv3d(pe_dim, dim, kernel_size=1, padding=0, bias=False)
        self.wavelet_embedding = nn.Parameter(torch.randn(self.total_bands, dim), requires_grad=True)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Conv3d(dim, 3 * dim, kernel_size=1, bias=qkv_bias, device=device)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=proj_bias, device=device)
        self.channel_gate = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, bias=True, device=device),
                                          nn.LeakyReLU(),
                                          nn.Conv3d(dim, 1, kernel_size=1, bias=True, device=device),
                                          nn.Sigmoid())
        self._init_weights()
    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        torch.nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    def _attn(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        qkv = self.qkv(x + pe)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H, W, D)
        q, k, v = torch.unbind(qkv, 1)
        q, k, v = q.flatten(-3).permute(0, 3, 1, 2), k.flatten(-3).permute(0, 3, 1, 2), v.flatten(-3).permute(0, 3, 1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v).permute(0, 2, 3, 1).contiguous()
        attn_out = attn_out.view(B, C, H, W, D)
        attn_out = self.proj(attn_out)
        return attn_out
    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        pe = self.pe_conv(pe)
        high_bands_list = []
        pe_bands_list = []
        # 1. 遞迴拆解，收集各階的高頻部分

        for i in range(self.levels):
            bands = self.dwt(x)
            x = bands[0]  # LLL 留作下一輪的輸入
            high_bands = (self.wavelet_feature_transform[i * 7 + 1](bands[1]),
                          self.wavelet_feature_transform[i * 7 + 2](bands[2]),
                          self.wavelet_feature_transform[i * 7 + 3](bands[3]),
                          self.wavelet_feature_transform[i * 7 + 4](bands[4]),
                          self.wavelet_feature_transform[i * 7 + 5](bands[5]),
                          self.wavelet_feature_transform[i * 7 + 6](bands[6]),
                          self.wavelet_feature_transform[i * 7 + 7](bands[7]))
            high_bands_list.append(high_bands)


            pe = self.avg_pool(pe)
            pe_bands = [pe + self.wavelet_embedding[i * 7 + 1][None, :, None, None, None],
                        pe + self.wavelet_embedding[i * 7 + 2][None, :, None, None, None],
                        pe + self.wavelet_embedding[i * 7 + 3][None, :, None, None, None],
                        pe + self.wavelet_embedding[i * 7 + 4][None, :, None, None, None],
                        pe + self.wavelet_embedding[i * 7 + 5][None, :, None, None, None],
                        pe + self.wavelet_embedding[i * 7 + 6][None, :, None, None, None],
                        pe + self.wavelet_embedding[i * 7 + 7][None, :, None, None, None]]
            pe_bands_list.append(pe_bands)
        # 2. 函數式組裝 (Functional Pack)，由最小張量拼回原本大小
        x = self.wavelet_feature_transform[0](x)
        pe = pe + self.wavelet_embedding[0][None, :, None, None, None]
        for high_bands, pe_bands in zip(reversed(high_bands_list), reversed(pe_bands_list)):
            x = pack_octants(x, *high_bands)
            pe = pack_octants(pe, *pe_bands)

        x = self.channel_gate(x) * self._attn(x, pe)

        d, h, w = x.shape[2:]
        scale = 2 ** self.levels
        curr_out = x[:, :, :d // scale, :h // scale, :w // scale]
        # 2. 從內到外逐層還原
        for level in reversed(range(1, self.levels + 1)):
            scale_l = 2 ** (level - 1)
            active_d, active_h, active_w = d // scale_l, h // scale_l, w // scale_l

            # 將該層所在的包裝區塊讀出
            region = x[:, :, :active_d, :active_h, :active_w]

            # 拆解出該層的高頻部份 (不需要理會丟出來的原本 LLL，我們用剛才重建好的 curr_out 替換)
            _, LLH, LHL, LHH, HLL, HLH, HHL, HHH = unpack_octants(region)

            # 將上層還原好的 LLL 與這層的高頻結合，做一次 IDWT
            curr_out = self.idwt((curr_out, LLH, LHL, LHH, HLL, HLH, HHL, HHH))
        return curr_out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        num_heads: int,
        temb_channels: int,
        ffn_ratio: int = 4,
        wavelet_levels: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1_1 = AdaLN(dim, temb_channels)
        self.wavelet_attn = WaveletSelfAttention(dim,
                                                 pe_dim,
                                                 num_heads=num_heads,
                                                 qkv_bias=qkv_bias,
                                                 proj_bias=proj_bias,
                                                 wavelet_levels=wavelet_levels,
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

    def forward(self, x: torch.Tensor, emb:torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        x_attn = x + self.wavelet_attn(self.norm1_1(x, emb), pe=pe) + self.fft_filter(self.norm1_2(x, emb), pe=pe)
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
        patch_levels: int = 1,
        in_chans: int = 4,
        embed_dim: int = 768,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_levels = patch_levels

        self.patch_size = (2**self.patch_levels, 2**self.patch_levels, 2**self.patch_levels)
        self.dwt = CDF53DWT3D()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d((8**self.patch_levels) * in_chans,
                              embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
    #     self._init_weights()
    #
    # def _init_weights(self):
    #    w = self.proj.weight.data
    #    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #    if self.proj.bias is not None:
    #        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for _ in range(self.patch_levels):
            x = torch.cat(self.dwt(x),dim=1)
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        patch_levels = 1,
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
            patch_levels=patch_levels,
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