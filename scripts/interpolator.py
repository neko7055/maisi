import math
import torch

def linear_interpolate(self, sources, targets, timesteps, add_noise=False, eps=1e-08) :
    timepoints = timesteps.float() / self.num_train_timesteps
    assert sources.shape == targets.shape
    # expand timepoint to noise shape
    if sources.ndim == 5:
        timepoints = timepoints[..., None, None, None, None]
    elif sources.ndim == 4:
        timepoints = timepoints[..., None, None, None]
    else:
        raise ValueError(f"noise tensor has to be 4D or 5D tensor, yet got shape of {sources.shape}")

    mu_t = (1 - timepoints) * sources + timepoints * targets
    d_mu_t = targets - sources
    if add_noise:
        z_coef = torch.sqrt(2 * timepoints * (1 - timepoints))
        dz_coef = (1 - 2 * timepoints) / torch.clamp(z_coef, min=eps)
        noise = torch.randn_like(mu_t)
        return mu_t + z_coef * noise, d_mu_t + dz_coef * noise
    else:
        return mu_t, d_mu_t

def polynomial_interpolate(self, sources, targets, timesteps, add_noise=False, sigma_0=1, sigma_1=0.00001) :
    timepoints = timesteps.float() / self.num_train_timesteps
    assert sources.shape == targets.shape
    # expand timepoint to noise shape
    if sources.ndim == 5:
        timepoints = timepoints[..., None, None, None, None]
    elif sources.ndim == 4:
        timepoints = timepoints[..., None, None, None]
    else:
        raise ValueError(f"noise tensor has to be 4D or 5D tensor, yet got shape of {sources.shape}")

    f_t = 2 * timepoints ** 5 - 4.5 * timepoints ** 4 + 2 * timepoints ** 3 + 0.5 * timepoints ** 2 + timepoints
    df_t = 10 * timepoints ** 4 - 18 * timepoints ** 3 + 6 * timepoints ** 2 + timepoints + 1
    mu_t = sources + f_t * (targets - sources)
    d_mu_t = df_t * (targets - sources)
    if add_noise:
        a = -3
        g_t = (1-torch.exp(-a * timepoints)) / (1-math.exp(-a))
        dg_t = (a * torch.exp(-a * timepoints)) / (1-math.exp(-a))
        noise = torch.randn_like(mu_t)
        z_coef = sigma_0 + g_t * (sigma_1 - sigma_0)
        dz_coef = dg_t * (sigma_1 - sigma_0)
        return mu_t + z_coef * noise, d_mu_t + dz_coef * noise
    else:
        return mu_t, d_mu_t

def triangular_interpolate(self, sources, targets, timesteps, add_noise=False, eps=1e-08) :
    timepoints = timesteps.float() / self.num_train_timesteps
    assert sources.shape == targets.shape
    # expand timepoint to noise shape
    if sources.ndim == 5:
        timepoints = timepoints[..., None, None, None, None]
    elif sources.ndim == 4:
        timepoints = timepoints[..., None, None, None]
    else:
        raise ValueError(f"noise tensor has to be 4D or 5D tensor, yet got shape of {sources.shape}")

    alpha = torch.pi / 2
    c_t, s_t = torch.cos(alpha * timepoints), torch.sin(alpha * timepoints)
    mu_t = c_t * sources + s_t * targets
    d_mu_t = alpha * (c_t * targets - s_t * sources)

    if add_noise:
        gamma = torch.sqrt(2 * timepoints * (1 - timepoints))
        d_gamma = (1 - 2 * timepoints) / torch.clamp(gamma, min=eps)
        z_coef = gamma
        dz_coef = d_gamma
        noise = torch.randn_like(mu_t)
        return torch.sqrt(1-gamma**2) * mu_t + z_coef * noise, (((1-gamma**2) * d_mu_t - gamma * d_gamma * mu_t) / torch.sqrt(1-gamma**2)) + dz_coef * noise
    else:
        return mu_t, d_mu_t

def enc_dec_interpolate(self, sources, targets, timesteps, add_noise=True) :
    assert add_noise, "enc-dec interpolation requires noise to be added, yet add_noise is set to False"
    timepoints = timesteps.float() / self.num_train_timesteps
    assert sources.shape == targets.shape
    # expand timepoint to noise shape
    if sources.ndim == 5:
        timepoints = timepoints[..., None, None, None, None]
    elif sources.ndim == 4:
        timepoints = timepoints[..., None, None, None]
    else:
        raise ValueError(f"noise tensor has to be 4D or 5D tensor, yet got shape of {sources.shape}")
    mu_coef = torch.cos(torch.pi * timepoints) ** 2
    d_mu_coef = -torch.pi * torch.sin(2 * torch.pi * timepoints)
    z_coef = torch.sin(torch.pi * timepoints) ** 2
    d_z_coef = torch.pi * torch.sin(2 * torch.pi * timepoints)
    mask = (timepoints < 0.5).to(torch.float32)
    images = sources * mask + targets * (1 - mask)
    mu_t = mu_coef * images
    d_mu_t = d_mu_coef * images
    noise = torch.randn_like(mu_t)
    return mu_t + z_coef * noise, d_mu_t + d_z_coef * noise

def spacial_interpolate(
    self,x0, x1, t,freq_a=25,freq_b=0.5, a_max=3.0, eps=1e-8, add_noise=True, sigma_0=1, sigma_1=0.00001, force_no_noise=False
):
    """
    x0, x1: [B, C, D, H, W] 或 [B, D, H, W]
    t:      [B]
    return:
        xt:    same shape as x0
        dx_dt: same shape as x0, 表示每個 sample 各自對應 t[b] 的偏微分
    """
    t = t.float() / self.num_train_timesteps
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape")
    if t.ndim != 1 or t.shape[0] != x0.shape[0]:
        raise ValueError("t must have shape [B]")

    B = x0.shape[0]
    D, H, W = x0.shape[-3:]
    device, dtype = x0.device, x0.dtype
    t = t.to(device=device, dtype=dtype)

    F0 = torch.fft.rfftn(x0, dim=(-3, -2, -1), norm="ortho")
    F1 = torch.fft.rfftn(x1, dim=(-3, -2, -1), norm="ortho")
    dF = F1 - F0

    # 2. 計算正規化頻率半徑 rho (0 ~ 1)
    fz = torch.fft.fftfreq(D, d=1.0, device=device).to(dtype).view(1, D, 1, 1)
    fy = torch.fft.fftfreq(H, d=1.0, device=device).to(dtype).view(1, 1, H, 1)
    fx = torch.fft.rfftfreq(W, d=1.0, device=device).to(dtype).view(1, 1, 1, W // 2 + 1)

    rho = torch.sqrt(fz ** 2 + fy ** 2 + fx ** 2)
    rho = rho / (rho.max() + eps)

    # 調整 rho 的形狀以支援 broadcast (補齊 batch 與 channel 維度)
    # 若輸入是 [B, C, D, H, W]，rho 變為 [1, 1, D, H, W_r]
    # 若輸入是 [B, D, H, W]，rho 變為 [1, D, H, W_r]
    view_shape = [1] * (x0.ndim - 3) + [D, H, W // 2 + 1]
    rho = rho.view(*view_shape)

    # 3) 根據 rho 計算每個頻率點的 alpha 參數
    alpha = torch.zeros_like(rho)

    mask_low = rho < freq_a
    mask_mid = (rho >= freq_a) & (rho <= freq_b)
    mask_high = rho > freq_b

    # f(t): rho < a
    alpha = torch.where(mask_low, torch.full_like(alpha, -a_max), alpha)
    # g(t): a <= rho <= b
    alpha_mid = -a_max * (freq_b - rho) / (freq_b - freq_a)
    alpha = torch.where(mask_mid, alpha_mid, alpha)
    # h(t): rho > b
    alpha_high = a_max * (rho - freq_b) / (1.0 - freq_b)
    alpha = torch.where(mask_high, alpha_high, alpha)

    # 4) 準備 t
    t_view = t.view(B, *([1] * (x0.ndim - 1)))

    # 5) 計算插值權重 M 與 微分 dM_dt (包含 Taylor 展開處理 alpha 趨近 0)
    mask_small = alpha.abs() < 1e-4

    # --- 正常計算 (alpha 不為 0) ---
    exp_alpha = torch.exp(-alpha)
    exp_alphat = torch.exp(-alpha * t_view)
    denom = 1.0 - exp_alpha

    M_normal = (1.0 - exp_alphat) / denom
    dM_dt_normal = alpha * exp_alphat / denom

    # --- 泰勒展開 (alpha 趨近 0，發生在 rho 接近 b 的交界處) ---
    M_small = t_view + 0.5 * alpha * t_view * (t_view - 1.0)
    dM_dt_small = 1.0 + alpha * (t_view - 0.5)

    M = torch.where(mask_small, M_small, M_normal)
    dM_dt = torch.where(mask_small, dM_dt_small, dM_dt_normal)

    # 6) 頻域組合與微分
    Y = F0 + M * dF
    dY_dt = dM_dt * dF

    # 6. 轉回空間域
    mu_t = torch.fft.irfftn(Y, s=(D, H, W), dim=(-3, -2, -1), norm="ortho")
    d_mu_t = torch.fft.irfftn(dY_dt, s=(D, H, W), dim=(-3, -2, -1), norm="ortho")
    if force_no_noise:
        return mu_t, d_mu_t

    if add_noise:
        t = t[..., None, None, None, None]
        a = -a_max
        g_t = (1 - torch.exp(-a * t)) / (1 - math.exp(-a))
        dg_t = (a * torch.exp(-a * t)) / (1 - math.exp(-a))
        noise = torch.randn_like(mu_t)
        z_coef = sigma_0 + g_t * (sigma_1 - sigma_0)
        dz_coef = dg_t * (sigma_1 - sigma_0)
        return mu_t + z_coef * noise, d_mu_t + dz_coef * noise
    else:
        return mu_t, d_mu_t