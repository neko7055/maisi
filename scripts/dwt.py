import math
import torch
from torch import nn
import triton
import triton.language as tl

# ==============================================================================
# 0. 精準常數快取系統 (純 PyTorch 浮點數，避開 Triton AST 污染)
# ==============================================================================
_CONSTANTS_CACHE = {}


def get_cdf97_constants_tensor(dtype, device):
    key = (dtype, device)
    if key in _CONSTANTS_CACHE:
        return _CONSTANTS_CACHE[key]

    alpha = -1.586134342059924
    beta = -0.052980118572961
    gamma = 0.882911075530934
    delta = 0.443506852043971
    K = 1.1496043988602418
    inv_K = 1.0 / K

    C1 = alpha * beta * gamma
    C2 = alpha + gamma + 3.0 * C1
    C3 = beta * gamma
    C3_mid = 1.0 + 2.0 * C3

    D1 = delta * C1
    D2 = alpha * beta + delta * (C1 + C2)
    D3 = 1.0 + 2.0 * alpha * beta + 2.0 * delta * C2
    D4 = delta * C3
    D5 = beta + delta * (1.0 + 3.0 * C3)

    tensor = torch.tensor([
        C1, C2, C3, C3_mid, D1, D2, D3, D4, D5, K, inv_K
    ], dtype=torch.float64, device=device).to(dtype)

    _CONSTANTS_CACHE[key] = tensor
    return tensor


# ==============================================================================
# 1. Triton Autotune 參數配置 (加入 1024 與 512 侵略性配置)
# ==============================================================================
def get_dwt_autotune_configs():
    return [
        # ==============================================================================
        # [Tier 1] 半精度 (BF16/FP16) 與資料中心卡專武區
        # FP32 在多數消費級顯卡上跑這區可能會 OOM，但 Autotuner 會自動忽略並往下找。
        # 這是為了確保在 BF16 下能榨乾所有 Shared Memory。
        # ==============================================================================
        triton.Config({'BLOCK_B': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_B': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 512}, num_warps=4, num_stages=2), # FP32 在高階卡的極限

        # ==============================================================================
        # [Tier 2] FP32 甜蜜點 & BF16 深層管線區 (最常被選中)
        # 256 是泛用性最高的 Block Size。
        # num_stages=4 對 BF16 是神級配置，能完美掩蓋記憶體讀取延遲。
        # ==============================================================================
        triton.Config({'BLOCK_B': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8, num_stages=3), # FP32 最佳平衡點
        triton.Config({'BLOCK_B': 256}, num_warps=4, num_stages=3),

        # ==============================================================================
        # [Tier 3] 極限延遲掩蓋區 (針對 dim=4 記憶體不連續問題)
        # 在 3D DWT 處理 W 維度時，記憶體 stride 跳躍極大。
        # 縮小 BLOCK_B 到 128，並將 num_stages 撐到 5 或 6，強迫硬體提早預載資料。
        # ==============================================================================
        triton.Config({'BLOCK_B': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=4),

        # ==============================================================================
        # [Tier 4] 高佔有率 (Occupancy) 保底區
        # 當 B_TOTAL 總量不夠填滿 GPU 所有的 SM 核心時 (例如 B=1, C=16)。
        # 切碎 BLOCK_B 可以生出更多 Thread Blocks，讓所有核心都有事做。
        # ==============================================================================
        triton.Config({'BLOCK_B': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_B': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_B': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 32}, num_warps=2, num_stages=3),

        # ==============================================================================
        # [Tier 5] 防爆保底
        # ==============================================================================
        triton.Config({'BLOCK_B': 16}, num_warps=2, num_stages=2),
    ]


# ==============================================================================
# 2. Triton Kernels
# ==============================================================================

@triton.autotune(configs=get_dwt_autotune_configs(), key=['B_TOTAL', 'N_HALF'])
@triton.jit
def cdf97_dwt_fwd_kernel(
        x_ptr, l_ptr, h_ptr, const_ptr,
        B_TOTAL: tl.constexpr, N_HALF: tl.constexpr, B_INNER: tl.constexpr,
        stride_x_o, stride_x_n, stride_x_i,
        stride_l_o, stride_l_n, stride_l_i,
        stride_h_o, stride_h_n, stride_h_i,
        BLOCK_N: tl.constexpr,
        BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    idx_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = idx_b < B_TOTAL

    idx_o = idx_b // B_INNER
    idx_i = idx_b % B_INNER

    x_base = x_ptr + idx_o * stride_x_o + idx_i * stride_x_i
    l_base = l_ptr + idx_o * stride_l_o + idx_i * stride_l_i
    h_base = h_ptr + idx_o * stride_h_o + idx_i * stride_h_i

    idx_n = tl.arange(0, BLOCK_N)
    mask_n = idx_n < N_HALF
    mask_2d = mask_b[:, None] & mask_n[None, :]

    idx_m2 = (idx_n - 2 + 2 * N_HALF) % N_HALF
    idx_m1 = (idx_n - 1 + 2 * N_HALF) % N_HALF
    idx_0 = idx_n
    idx_p1 = (idx_n + 1) % N_HALF
    idx_p2 = (idx_n + 2) % N_HALF

    e_m2 = tl.load(x_base[:, None] + (2 * idx_m2[None, :]) * stride_x_n, mask=mask_2d)
    e_m1 = tl.load(x_base[:, None] + (2 * idx_m1[None, :]) * stride_x_n, mask=mask_2d)
    e_0 = tl.load(x_base[:, None] + (2 * idx_0[None, :]) * stride_x_n, mask=mask_2d)
    e_p1 = tl.load(x_base[:, None] + (2 * idx_p1[None, :]) * stride_x_n, mask=mask_2d)
    e_p2 = tl.load(x_base[:, None] + (2 * idx_p2[None, :]) * stride_x_n, mask=mask_2d)

    o_m2 = tl.load(x_base[:, None] + (2 * idx_m2[None, :] + 1) * stride_x_n, mask=mask_2d)
    o_m1 = tl.load(x_base[:, None] + (2 * idx_m1[None, :] + 1) * stride_x_n, mask=mask_2d)
    o_0 = tl.load(x_base[:, None] + (2 * idx_0[None, :] + 1) * stride_x_n, mask=mask_2d)
    o_p1 = tl.load(x_base[:, None] + (2 * idx_p1[None, :] + 1) * stride_x_n, mask=mask_2d)

    C1 = tl.load(const_ptr + 0)
    C2 = tl.load(const_ptr + 1)
    C3 = tl.load(const_ptr + 2)
    C3_mid = tl.load(const_ptr + 3)
    D1 = tl.load(const_ptr + 4)
    D2 = tl.load(const_ptr + 5)
    D3 = tl.load(const_ptr + 6)
    D4 = tl.load(const_ptr + 7)
    D5 = tl.load(const_ptr + 8)
    K = tl.load(const_ptr + 9)
    inv_K = tl.load(const_ptr + 10)

    L = D1 * e_m2 + D2 * e_m1 + D3 * e_0 + D2 * e_p1 + D1 * e_p2 + D4 * o_m2 + D5 * o_m1 + D5 * o_0 + D4 * o_p1
    H = C1 * e_m1 + C2 * e_0 + C2 * e_p1 + C1 * e_p2 + C3 * o_m1 + C3_mid * o_0 + C3 * o_p1

    tl.store(l_base[:, None] + idx_n[None, :] * stride_l_n, L * K, mask=mask_2d)
    tl.store(h_base[:, None] + idx_n[None, :] * stride_h_n, H * inv_K, mask=mask_2d)


@triton.autotune(configs=get_dwt_autotune_configs(), key=['B_TOTAL', 'N_HALF'])
@triton.jit
def cdf97_dwt_bwd_kernel(
        dl_ptr, dh_ptr, dx_ptr, const_ptr,
        B_TOTAL: tl.constexpr, N_HALF: tl.constexpr, B_INNER: tl.constexpr,
        stride_l_o, stride_l_n, stride_l_i,
        stride_h_o, stride_h_n, stride_h_i,
        stride_x_o, stride_x_n, stride_x_i,
        BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    idx_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = idx_b < B_TOTAL

    idx_o = idx_b // B_INNER
    idx_i = idx_b % B_INNER

    l_base = dl_ptr + idx_o * stride_l_o + idx_i * stride_l_i
    h_base = dh_ptr + idx_o * stride_h_o + idx_i * stride_h_i
    x_base = dx_ptr + idx_o * stride_x_o + idx_i * stride_x_i

    idx_n = tl.arange(0, BLOCK_N)
    mask_n = idx_n < N_HALF
    mask_2d = mask_b[:, None] & mask_n[None, :]

    idx_m2 = (idx_n - 2 + 2 * N_HALF) % N_HALF
    idx_m1 = (idx_n - 1 + 2 * N_HALF) % N_HALF
    idx_0 = idx_n
    idx_p1 = (idx_n + 1) % N_HALF
    idx_p2 = (idx_n + 2) % N_HALF

    C1 = tl.load(const_ptr + 0)
    C2 = tl.load(const_ptr + 1)
    C3 = tl.load(const_ptr + 2)
    C3_mid = tl.load(const_ptr + 3)
    D1 = tl.load(const_ptr + 4)
    D2 = tl.load(const_ptr + 5)
    D3 = tl.load(const_ptr + 6)
    D4 = tl.load(const_ptr + 7)
    D5 = tl.load(const_ptr + 8)
    K = tl.load(const_ptr + 9)
    inv_K = tl.load(const_ptr + 10)

    dL_m2 = tl.load(l_base[:, None] + idx_m2[None, :] * stride_l_n, mask=mask_2d) * K
    dL_m1 = tl.load(l_base[:, None] + idx_m1[None, :] * stride_l_n, mask=mask_2d) * K
    dL_0 = tl.load(l_base[:, None] + idx_0[None, :] * stride_l_n, mask=mask_2d) * K
    dL_p1 = tl.load(l_base[:, None] + idx_p1[None, :] * stride_l_n, mask=mask_2d) * K
    dL_p2 = tl.load(l_base[:, None] + idx_p2[None, :] * stride_l_n, mask=mask_2d) * K

    dH_m2 = tl.load(h_base[:, None] + idx_m2[None, :] * stride_h_n, mask=mask_2d) * inv_K
    dH_m1 = tl.load(h_base[:, None] + idx_m1[None, :] * stride_h_n, mask=mask_2d) * inv_K
    dH_0 = tl.load(h_base[:, None] + idx_0[None, :] * stride_h_n, mask=mask_2d) * inv_K
    dH_p1 = tl.load(h_base[:, None] + idx_p1[None, :] * stride_h_n, mask=mask_2d) * inv_K

    dx_even = D1 * dL_m2 + D2 * dL_m1 + D3 * dL_0 + D2 * dL_p1 + D1 * dL_p2 + C1 * dH_m2 + C2 * dH_m1 + C2 * dH_0 + C1 * dH_p1
    dx_odd = D4 * dL_m1 + D5 * dL_0 + D5 * dL_p1 + D4 * dL_p2 + C3 * dH_m1 + C3_mid * dH_0 + C3 * dH_p1

    tl.store(x_base[:, None] + (2 * idx_n[None, :]) * stride_x_n, dx_even, mask=mask_2d)
    tl.store(x_base[:, None] + (2 * idx_n[None, :] + 1) * stride_x_n, dx_odd, mask=mask_2d)


@triton.autotune(configs=get_dwt_autotune_configs(), key=['B_TOTAL', 'N_HALF'])
@triton.jit
def cdf97_idwt_fwd_kernel(
        l_ptr, h_ptr, x_ptr, const_ptr,
        B_TOTAL: tl.constexpr, N_HALF: tl.constexpr, B_INNER: tl.constexpr,
        stride_l_o, stride_l_n, stride_l_i,
        stride_h_o, stride_h_n, stride_h_i,
        stride_x_o, stride_x_n, stride_x_i,
        BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    idx_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = idx_b < B_TOTAL

    idx_o = idx_b // B_INNER
    idx_i = idx_b % B_INNER

    l_base = l_ptr + idx_o * stride_l_o + idx_i * stride_l_i
    h_base = h_ptr + idx_o * stride_h_o + idx_i * stride_h_i
    x_base = x_ptr + idx_o * stride_x_o + idx_i * stride_x_i

    idx_n = tl.arange(0, BLOCK_N)
    mask_n = idx_n < N_HALF
    mask_2d = mask_b[:, None] & mask_n[None, :]

    idx_m2 = (idx_n - 2 + 2 * N_HALF) % N_HALF
    idx_m1 = (idx_n - 1 + 2 * N_HALF) % N_HALF
    idx_0 = idx_n
    idx_p1 = (idx_n + 1) % N_HALF
    idx_p2 = (idx_n + 2) % N_HALF

    C1 = tl.load(const_ptr + 0)
    C2 = tl.load(const_ptr + 1)
    C3 = tl.load(const_ptr + 2)
    C3_mid = tl.load(const_ptr + 3)
    D1 = tl.load(const_ptr + 4)
    D2 = tl.load(const_ptr + 5)
    D3 = tl.load(const_ptr + 6)
    D4 = tl.load(const_ptr + 7)
    D5 = tl.load(const_ptr + 8)
    K = tl.load(const_ptr + 9)
    inv_K = tl.load(const_ptr + 10)

    E_m1 = tl.load(l_base[:, None] + idx_m1[None, :] * stride_l_n, mask=mask_2d) * inv_K
    E_0 = tl.load(l_base[:, None] + idx_0[None, :] * stride_l_n, mask=mask_2d) * inv_K
    E_p1 = tl.load(l_base[:, None] + idx_p1[None, :] * stride_l_n, mask=mask_2d) * inv_K
    E_p2 = tl.load(l_base[:, None] + idx_p2[None, :] * stride_l_n, mask=mask_2d) * inv_K

    O_m2 = tl.load(h_base[:, None] + idx_m2[None, :] * stride_h_n, mask=mask_2d) * K
    O_m1 = tl.load(h_base[:, None] + idx_m1[None, :] * stride_h_n, mask=mask_2d) * K
    O_0 = tl.load(h_base[:, None] + idx_0[None, :] * stride_h_n, mask=mask_2d) * K
    O_p1 = tl.load(h_base[:, None] + idx_p1[None, :] * stride_h_n, mask=mask_2d) * K
    O_p2 = tl.load(h_base[:, None] + idx_p2[None, :] * stride_h_n, mask=mask_2d) * K

    x_even = C3 * E_m1 + C3_mid * E_0 + C3 * E_p1 - D4 * O_m2 - D5 * O_m1 - D5 * O_0 - D4 * O_p1
    x_odd = -C1 * E_m1 - C2 * E_0 - C2 * E_p1 - C1 * E_p2 + D1 * O_m2 + D2 * O_m1 + D3 * O_0 + D2 * O_p1 + D1 * O_p2

    tl.store(x_base[:, None] + (2 * idx_n[None, :]) * stride_x_n, x_even, mask=mask_2d)
    tl.store(x_base[:, None] + (2 * idx_n[None, :] + 1) * stride_x_n, x_odd, mask=mask_2d)


@triton.autotune(configs=get_dwt_autotune_configs(), key=['B_TOTAL', 'N_HALF'])
@triton.jit
def cdf97_idwt_bwd_kernel(
        dx_ptr, dl_ptr, dh_ptr, const_ptr,
        B_TOTAL: tl.constexpr, N_HALF: tl.constexpr, B_INNER: tl.constexpr,
        stride_x_o, stride_x_n, stride_x_i,
        stride_l_o, stride_l_n, stride_l_i,
        stride_h_o, stride_h_n, stride_h_i,
        BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    idx_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = idx_b < B_TOTAL

    idx_o = idx_b // B_INNER
    idx_i = idx_b % B_INNER

    x_base = dx_ptr + idx_o * stride_x_o + idx_i * stride_x_i
    l_base = dl_ptr + idx_o * stride_l_o + idx_i * stride_l_i
    h_base = dh_ptr + idx_o * stride_h_o + idx_i * stride_h_i

    idx_n = tl.arange(0, BLOCK_N)
    mask_n = idx_n < N_HALF
    mask_2d = mask_b[:, None] & mask_n[None, :]

    idx_m2 = (idx_n - 2 + 2 * N_HALF) % N_HALF
    idx_m1 = (idx_n - 1 + 2 * N_HALF) % N_HALF
    idx_0 = idx_n
    idx_p1 = (idx_n + 1) % N_HALF
    idx_p2 = (idx_n + 2) % N_HALF

    de_m1 = tl.load(x_base[:, None] + (2 * idx_m1[None, :]) * stride_x_n, mask=mask_2d)
    de_0 = tl.load(x_base[:, None] + (2 * idx_0[None, :]) * stride_x_n, mask=mask_2d)
    de_p1 = tl.load(x_base[:, None] + (2 * idx_p1[None, :]) * stride_x_n, mask=mask_2d)
    de_p2 = tl.load(x_base[:, None] + (2 * idx_p2[None, :]) * stride_x_n, mask=mask_2d)

    do_m2 = tl.load(x_base[:, None] + (2 * idx_m2[None, :] + 1) * stride_x_n, mask=mask_2d)
    do_m1 = tl.load(x_base[:, None] + (2 * idx_m1[None, :] + 1) * stride_x_n, mask=mask_2d)
    do_0 = tl.load(x_base[:, None] + (2 * idx_0[None, :] + 1) * stride_x_n, mask=mask_2d)
    do_p1 = tl.load(x_base[:, None] + (2 * idx_p1[None, :] + 1) * stride_x_n, mask=mask_2d)
    do_p2 = tl.load(x_base[:, None] + (2 * idx_p2[None, :] + 1) * stride_x_n, mask=mask_2d)

    C1 = tl.load(const_ptr + 0)
    C2 = tl.load(const_ptr + 1)
    C3 = tl.load(const_ptr + 2)
    C3_mid = tl.load(const_ptr + 3)
    D1 = tl.load(const_ptr + 4)
    D2 = tl.load(const_ptr + 5)
    D3 = tl.load(const_ptr + 6)
    D4 = tl.load(const_ptr + 7)
    D5 = tl.load(const_ptr + 8)
    K = tl.load(const_ptr + 9)
    inv_K = tl.load(const_ptr + 10)

    dE = C3 * de_m1 + C3_mid * de_0 + C3 * de_p1 - C1 * do_m2 - C2 * do_m1 - C2 * do_0 - C1 * do_p1
    dO = -D4 * de_m1 - D5 * de_0 - D5 * de_p1 - D4 * de_p2 + D1 * do_m2 + D2 * do_m1 + D3 * do_0 + D2 * do_p1 + D1 * do_p2

    tl.store(l_base[:, None] + idx_n[None, :] * stride_l_n, dE * inv_K, mask=mask_2d)
    tl.store(h_base[:, None] + idx_n[None, :] * stride_h_n, dO * K, mask=mask_2d)


# ==============================================================================
# 3. PyTorch autograd.Function 包裝器
# ==============================================================================

class TritonCDF97DWTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        if dim < 0: dim += x.ndim
        ctx.dim = dim
        ctx.orig_shape = list(x.shape)

        if not x.is_contiguous(): x = x.contiguous()

        N = x.shape[dim]
        N_HALF = N // 2

        B_outer = math.prod(x.shape[:dim]) if dim > 0 else 1
        B_inner = math.prod(x.shape[dim + 1:]) if dim < x.ndim - 1 else 1
        B_TOTAL = B_outer * B_inner

        x_3d = x.view(B_outer, N, B_inner)

        out_shape = list(x.shape)
        out_shape[dim] = N_HALF
        L = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        H = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        L_3d = L.view(B_outer, N_HALF, B_inner)
        H_3d = H.view(B_outer, N_HALF, B_inner)

        BLOCK_N = triton.next_power_of_2(N_HALF)

        # Grid 與動態 BLOCK_B 連動
        grid = lambda meta: (triton.cdiv(B_TOTAL, meta['BLOCK_B']),)

        constants = get_cdf97_constants_tensor(x.dtype, x.device)

        cdf97_dwt_fwd_kernel[grid](
            x_3d, L_3d, H_3d, constants,
            B_TOTAL, N_HALF, B_inner,
            x_3d.stride(0), x_3d.stride(1), x_3d.stride(2),
            L_3d.stride(0), L_3d.stride(1), L_3d.stride(2),
            H_3d.stride(0), H_3d.stride(1), H_3d.stride(2),
            BLOCK_N=BLOCK_N
        )
        return L, H

    @staticmethod
    def backward(ctx, grad_L, grad_H):
        if grad_L is None or grad_H is None: return None, None

        dim = ctx.dim
        N = ctx.orig_shape[dim]
        N_HALF = N // 2

        if not grad_L.is_contiguous(): grad_L = grad_L.contiguous()
        if not grad_H.is_contiguous(): grad_H = grad_H.contiguous()

        B_outer = math.prod(ctx.orig_shape[:dim]) if dim > 0 else 1
        B_inner = math.prod(ctx.orig_shape[dim + 1:]) if dim < len(ctx.orig_shape) - 1 else 1
        B_TOTAL = B_outer * B_inner

        grad_L_3d = grad_L.view(B_outer, N_HALF, B_inner)
        grad_H_3d = grad_H.view(B_outer, N_HALF, B_inner)

        grad_x = torch.empty(ctx.orig_shape, device=grad_L.device, dtype=grad_L.dtype)
        grad_x_3d = grad_x.view(B_outer, N, B_inner)

        BLOCK_N = triton.next_power_of_2(N_HALF)
        grid = lambda meta: (triton.cdiv(B_TOTAL, meta['BLOCK_B']),)

        constants = get_cdf97_constants_tensor(grad_L.dtype, grad_L.device)

        cdf97_dwt_bwd_kernel[grid](
            grad_L_3d, grad_H_3d, grad_x_3d, constants,
            B_TOTAL, N_HALF, B_inner,
            grad_L_3d.stride(0), grad_L_3d.stride(1), grad_L_3d.stride(2),
            grad_H_3d.stride(0), grad_H_3d.stride(1), grad_H_3d.stride(2),
            grad_x_3d.stride(0), grad_x_3d.stride(1), grad_x_3d.stride(2),
            BLOCK_N=BLOCK_N
        )
        return grad_x, None


class TritonCDF97IDWTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, L, H, dim):
        if dim < 0: dim += L.ndim
        ctx.dim = dim
        ctx.orig_shape_L = list(L.shape)

        if not L.is_contiguous(): L = L.contiguous()
        if not H.is_contiguous(): H = H.contiguous()

        N_HALF = L.shape[dim]
        N = N_HALF * 2

        B_outer = math.prod(L.shape[:dim]) if dim > 0 else 1
        B_inner = math.prod(L.shape[dim + 1:]) if dim < L.ndim - 1 else 1
        B_TOTAL = B_outer * B_inner

        L_3d = L.view(B_outer, N_HALF, B_inner)
        H_3d = H.view(B_outer, N_HALF, B_inner)

        out_shape = list(L.shape)
        out_shape[dim] = N
        x = torch.empty(out_shape, device=L.device, dtype=L.dtype)
        x_3d = x.view(B_outer, N, B_inner)

        BLOCK_N = triton.next_power_of_2(N_HALF)
        grid = lambda meta: (triton.cdiv(B_TOTAL, meta['BLOCK_B']),)

        constants = get_cdf97_constants_tensor(L.dtype, L.device)

        cdf97_idwt_fwd_kernel[grid](
            L_3d, H_3d, x_3d, constants,
            B_TOTAL, N_HALF, B_inner,
            L_3d.stride(0), L_3d.stride(1), L_3d.stride(2),
            H_3d.stride(0), H_3d.stride(1), H_3d.stride(2),
            x_3d.stride(0), x_3d.stride(1), x_3d.stride(2),
            BLOCK_N=BLOCK_N
        )
        return x

    @staticmethod
    def backward(ctx, grad_x):
        if grad_x is None: return None, None, None

        dim = ctx.dim
        N_HALF = ctx.orig_shape_L[dim]
        N = N_HALF * 2

        if not grad_x.is_contiguous(): grad_x = grad_x.contiguous()

        B_outer = math.prod(ctx.orig_shape_L[:dim]) if dim > 0 else 1
        B_inner = math.prod(ctx.orig_shape_L[dim + 1:]) if dim < len(ctx.orig_shape_L) - 1 else 1
        B_TOTAL = B_outer * B_inner

        grad_x_3d = grad_x.view(B_outer, N, B_inner)

        grad_L = torch.empty(ctx.orig_shape_L, device=grad_x.device, dtype=grad_x.dtype)
        grad_H = torch.empty(ctx.orig_shape_L, device=grad_x.device, dtype=grad_x.dtype)
        grad_L_3d = grad_L.view(B_outer, N_HALF, B_inner)
        grad_H_3d = grad_H.view(B_outer, N_HALF, B_inner)

        BLOCK_N = triton.next_power_of_2(N_HALF)
        grid = lambda meta: (triton.cdiv(B_TOTAL, meta['BLOCK_B']),)

        constants = get_cdf97_constants_tensor(grad_x.dtype, grad_x.device)

        cdf97_idwt_bwd_kernel[grid](
            grad_x_3d, grad_L_3d, grad_H_3d, constants,
            B_TOTAL, N_HALF, B_inner,
            grad_x_3d.stride(0), grad_x_3d.stride(1), grad_x_3d.stride(2),
            grad_L_3d.stride(0), grad_L_3d.stride(1), grad_L_3d.stride(2),
            grad_H_3d.stride(0), grad_H_3d.stride(1), grad_H_3d.stride(2),
            BLOCK_N=BLOCK_N
        )
        return grad_L, grad_H, None


# ==============================================================================
# 4. 隨插即用的 PyTorch Modules
# ==============================================================================

class CDF97DWT3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, d, h, w = x.shape
        if d % 2 or h % 2 or w % 2:
            raise ValueError("D, H, W must be even for CDF 9/7 DWT.")

        L_d, H_d = TritonCDF97DWTFunction.apply(x, 2)

        LL, LH = TritonCDF97DWTFunction.apply(L_d, 3)
        HL, HH = TritonCDF97DWTFunction.apply(H_d, 3)

        LLL, LLH = TritonCDF97DWTFunction.apply(LL, 4)
        LHL, LHH = TritonCDF97DWTFunction.apply(LH, 4)
        HLL, HLH = TritonCDF97DWTFunction.apply(HL, 4)
        HHL, HHH = TritonCDF97DWTFunction.apply(HH, 4)

        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH


class CDF97IDWT3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coeffs):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = coeffs

        LL = TritonCDF97IDWTFunction.apply(LLL, LLH, 4)
        LH = TritonCDF97IDWTFunction.apply(LHL, LHH, 4)
        HL = TritonCDF97IDWTFunction.apply(HLL, HLH, 4)
        HH = TritonCDF97IDWTFunction.apply(HHL, HHH, 4)

        L_d = TritonCDF97IDWTFunction.apply(LL, LH, 3)
        H_d = TritonCDF97IDWTFunction.apply(HL, HH, 3)

        x = TritonCDF97IDWTFunction.apply(L_d, H_d, 2)
        return x

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator()
    x = torch.randn(1, 4, 32, 32, 32, dtype=torch.float64, device=device)

    dwt = CDF97DWT3D()#.to(torch.float32)
    idwt = CDF97IDWT3D()#.to(torch.float32)
    coeffs = dwt(x)
    x_rec = idwt(coeffs)

    max_err = torch.max(torch.abs(x - x_rec)).item()
    mean_err = torch.mean(torch.abs(x - x_rec)).item()

    print(f"Max reconstruction error: {max_err:.2e}")
    print(f"Mean reconstruction error: {mean_err:.2e}")

    is_perfect = max_err < 1e-10
    print(f"Perfect reconstruction: {is_perfect}")

    x = torch.randn(
        1, 1, 4, 4, 4,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    for d in range(2,5):
        ok_1 = torch.autograd.gradcheck(
            lambda z: TritonCDF97DWTFunction.apply(z, d)[0],
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

        ok_2 = torch.autograd.gradcheck(
            lambda z: TritonCDF97DWTFunction.apply(z, d)[1],
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

        print("DWT gradcheck:", ok_1, ok_2)

        L = torch.randn(
            1, 1, 4, 4, 2,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        H = torch.randn_like(L, requires_grad=True)

        ok = torch.autograd.gradcheck(
            lambda l, h: TritonCDF97IDWTFunction.apply(l, h, d),
            (L, H),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

        print("IDWT gradcheck:", ok)