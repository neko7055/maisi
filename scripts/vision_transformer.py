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
import torch.utils.checkpoint as cp

from .dwt import CDF97DWT3D, CDF97IDWT3D

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

def modulate(x, shift, scale):
    return x * (1 + scale[...,None,None,None]) + shift[...,None,None,None]

class WTConv(nn.Module):
    def __init__(
            self,
            dim: int,
            wavelet_levels=2,
    ) -> None:
        super().__init__()
        self.dwt = CDF97DWT3D()
        self.idwt = CDF97IDWT3D()
        self.levels = wavelet_levels
        self.dim = dim
        self.base_conv = nn.Conv3d(dim, dim, groups=dim, kernel_size=3,stride=1,padding=1, bias=False)
        self.convs = nn.ModuleList([nn.Conv3d(dim, dim, groups=dim,
                                              kernel_size=3,stride=1,padding=1, bias=False) for _ in range(8 * self.levels)])
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands_list = []
        bs_conv = self.base_conv(x)
        bands=[x]
        for i in range(self.levels):
            bands = self.dwt(bands[0])
            bands_list.append(bands)
        x = 0.0
        for i,bands in enumerate(reversed(bands_list)):
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = bands
            LLL = x + self.convs[i*8 + 0](LLL)
            LLH = self.convs[i*8 + 1](LLH)
            LHL = self.convs[i*8 + 2](LHL)
            LHH = self.convs[i*8 + 3](LHH)
            HLL = self.convs[i*8 + 4](HLL)
            HLH = self.convs[i*8 + 5](HLH)
            HHL = self.convs[i*8 + 6](HHL)
            HHH = self.convs[i*8 + 7](HHH)
            x = self.idwt((LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH))
        return bs_conv + x

class WPConv(nn.Module):
    def __init__(
            self,
            dim: int,
            wavelet_levels=2,
    ) -> None:
        super().__init__()
        self.dwt = CDF97DWT3D()
        self.idwt = CDF97IDWT3D()
        self.levels = wavelet_levels
        self.dim = dim
        self.dws = nn.ModuleList([nn.Conv3d(dim * (8 ** i), dim * (8 ** i), groups=dim * (8 ** i),
                           kernel_size=3,stride=1,padding=1, bias=False) for i in range(1,self.levels+1)])
        self.dw = nn.Conv3d(dim, dim, groups=dim,
                           kernel_size=3,stride=1,padding=1, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.dw(x)]
        for i in range(self.levels):
            x = torch.cat(self.dwt(x),dim=1)
            y.append(self.dws[i](x))
        x = y.pop()
        for _ in range(self.levels):
            x = self.idwt(torch.chunk(x,8,dim=1)) + y.pop()
        return x

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

@torch.compile(mode="default", fullgraph=True)
def linear_attn(Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                eps: float = 1e-5):
    # Q, K, V 輸入大小皆為: (B, heads, head_dim, H, W, D)
    H, W, D = Q.shape[-3:]
    N = H * W * D
    # Step 1: phi() kernel，讓數值非負
    Q_phi = F.elu(Q) + 1.0  # (B, heads, head_dim,  H, W, D)
    K_phi = F.elu(K) + 1.0  # (B, heads, head_dim,  H, W, D)
    KV = torch.einsum("b h d x y z, b h v x y z -> b h d v", K_phi, V) / N
    numerator = torch.einsum("b h d x y z, b h d v -> b h v x y z", Q_phi, KV)
    K_sum = K_phi.mean(dim=[-3, -2, -1])
    Z = torch.einsum("b h d x y z, b h d -> b h x y z", Q_phi, K_sum)
    Z = Z + eps
    out = numerator / Z.unsqueeze(2)
    return out

@torch.compile(mode="default", fullgraph=True)
def linear_attn_enhance(Q: torch.Tensor,
                        K: torch.Tensor,
                        V: torch.Tensor,
                        eps: float = 1e-5):
    # Q, K, V 輸入大小皆為: (B, heads, head_dim, H, W, D)
    H, W, D = Q.shape[-3:]
    N = H * W * D
    # Step 1: phi() kernel，確保數值非負
    Q_phi = F.elu(Q) + 1.0
    K_phi = F.elu(K) + 1.0

    # === 分子計算最佳化 ===
    # 將 Q_phi @ T 的計算展開成 8 項獨立的張量乘法
    # 先計算 K 與 V 在空間上的 Sum 降維，再乘上 Q，完全避免產生 head_dim * head_dim 的超大空間張量

    # 0. 無額外加總項 (Identity)
    # 數學等價於 Q_phi @ (K_phi * V)
    QK = (Q_phi * K_phi).sum(dim=2, keepdim=True)
    N0 = QK * V / N

    # 1. 僅對 X (dim=-3) 進行加總
    KV_x = torch.einsum("b h d x y z, b h v x y z -> b h d v y z", K_phi, V) / N
    N_x = torch.einsum("b h d x y z, b h d v y z -> b h v x y z", Q_phi, KV_x)

    # 2. 僅對 Y (dim=-2) 進行加總
    KV_y = torch.einsum("b h d x y z, b h v x y z -> b h d v x z", K_phi, V) / N
    N_y = torch.einsum("b h d x y z, b h d v x z -> b h v x y z", Q_phi, KV_y)

    # 3. 僅對 Z (dim=-1) 進行加總
    KV_z = torch.einsum("b h d x y z, b h v x y z -> b h d v x y", K_phi, V) / N
    N_z = torch.einsum("b h d x y z, b h d v x y -> b h v x y z", Q_phi, KV_z)

    # 4. 對 X, Y 進行加總
    KV_xy = torch.einsum("b h d x y z, b h v x y z -> b h d v z", K_phi, V) / N
    N_xy = torch.einsum("b h d x y z, b h d v z -> b h v x y z", Q_phi, KV_xy)

    # 5. 對 X, Z 進行加總
    KV_xz = torch.einsum("b h d x y z, b h v x y z -> b h d v y", K_phi, V) / N
    N_xz = torch.einsum("b h d x y z, b h d v y -> b h v x y z", Q_phi, KV_xz)

    # 6. 對 Y, Z 進行加總
    KV_yz = torch.einsum("b h d x y z, b h v x y z -> b h d v x", K_phi, V) / N
    N_yz = torch.einsum("b h d x y z, b h d v x -> b h v x y z", Q_phi, KV_yz)

    # 7. 對 X, Y, Z 皆進行加總
    KV_xyz = torch.einsum("b h d x y z, b h v x y z -> b h d v", K_phi, V) / N
    N_xyz = torch.einsum("b h d x y z, b h d v -> b h v x y z", Q_phi, KV_xyz)

    # 加總 8 種空間維度組合
    numerator = N0 + N_x + N_y + N_z + N_xy + N_xz + N_yz + N_xyz

    # === 分母計算 ===
    # 分母的 K_sum 維度只有 (B, heads, head_dim, H, W, D)，沒有膨脹問題，保留原本的高效寫法
    K_sum = K_phi / N
    K_sum = K_sum + K_sum.sum(dim=-3, keepdim=True)
    K_sum = K_sum + K_sum.sum(dim=-2, keepdim=True)
    K_sum = K_sum + K_sum.sum(dim=-1, keepdim=True)

    Z = torch.einsum("b h d x y z, b h d x y z -> b h x y z", Q_phi, K_sum)
    Z = Z + eps

    # Z.unsqueeze(2) 將形狀變為 (B, heads, 1, H, W, D) 以便 broadcast 給分子
    out = numerator / Z.unsqueeze(2)
    return out

class LinearAttention(nn.Module):
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
        self.conv = WTConv(dim, 3)
        self.qkv = nn.Conv3d(dim, 3 * dim, kernel_size=1,padding=0, bias=qkv_bias, device=device)
        self.fft_dw = FourierGlobalFilter(dim, pe_dim)
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
        qkv = qkv.view(B, 3, C, H, W, D)
        q, k, v = torch.unbind(qkv, 1)
        q = q + self.conv(q)
        k = k + self.conv(k)
        q = q.view(B, self.num_heads, self.head_dim, H, W, D)
        k = k.view(B, self.num_heads, self.head_dim, H, W, D)
        v = v.view(B, self.num_heads, self.head_dim, H, W, D)
        attn_out = linear_attn_enhance(q, k, v)
        attn_out = attn_out.view(B, C, H, W, D) + self.fft_dw(v.view(B, C, H, W, D), pe)
        attn_out = self.proj(attn_out)
        return attn_out

class FourierGlobalFilter(nn.Module):
    def __init__(
        self,
        dim: int,
        pe_dim: int,
    ):
        super().__init__()
        self.pe_dim = pe_dim
        self.dim = dim
        self.attn_map = nn.Conv3d(self.pe_dim, self.dim, kernel_size=1, stride=1, padding=0, bias=False)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.zeros_(self.attn_map.weight)
        if self.attn_map.bias is not None:
            nn.init.zeros_(self.attn_map.bias)
    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        weight = self.attn_map(pe)
        x = x.float()
        weight = weight.float()
        weight = torch.fft.rfftn(weight, dim=(-3, -2, -1), norm="ortho")
        x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        x = x * weight
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
        self.attn = LinearAttention(
            dim,
            pe_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            device=device,
        )
        self.mlp = Mlp(dim,
                       ffn_ratio,
                       act_layer=partial(nn.GELU, approximate="tanh"),
                       bias=ffn_bias,
                       device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_channels, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
    def forward(self, x: torch.Tensor, emb:torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa,
        shift_mlp, scale_mlp, gate_mlp)= self.adaLN_modulation(emb).chunk(6, dim=1)
        x = x + gate_msa[...,None,None,None] * self.attn(modulate(layer_norm(x), shift_msa, scale_msa), pe=pe)
        x = x + gate_mlp[...,None,None,None] * self.mlp(modulate(layer_norm(x), shift_mlp, scale_mlp))
        return x

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
        self.linear = nn.Conv3d(hidden_size,
                                self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * out_channels,#
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_channels, 2 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.linear.weight, 0)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, emb):
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = self.linear(modulate(layer_norm(x), shift, scale))
        return x

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 4,
        embed_dim: int = 768,
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
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            *,
            patch_size=(2, 2, 2),
            in_chans: int = 3,
            out_chans: int = 3,
            embed_dim: int = 768,
            temb_channels: int = 256,
            num_heads: int = 8,
            depth: int = 12,
            ffn_ratio: int = 4,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            legendre_max_degree: int = 8,
            device: Any | None = None,
            **ignored_kwargs,
    ):
        super().__init__()
        del ignored_kwargs
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
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
        self._pe_cache = dict()

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
        if (H, W, D) not in self._pe_cache.keys():
            self._pe_cache[(H, W, D)] = self._get_pe(H, W, D, torch.device('cpu'))
        pe = self._pe_cache[(H, W, D)].to(x.device)
        for _, blk in enumerate(self.blocks):
            x = cp.checkpoint(blk,x, emb, pe, use_reentrant=False)
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