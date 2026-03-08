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

def polynomial_interpolate(self, sources, targets, timesteps, add_noise=False) :
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
        sqrt05 = 0.7071067811865476
        z_coef = (-54 * timepoints ** 6 +
                  158 * timepoints ** 5 -
                  151.5 * timepoints ** 4 +
                  46 * timepoints ** 3 +
                  0.5 * timepoints ** 2 +
                  timepoints) * sqrt05

        dz_coef = (-326 * timepoints ** 5 +
                  790 * timepoints ** 4 -
                  606 * timepoints ** 3 +
                  138 * timepoints ** 2 +
                  timepoints +
                  1) * sqrt05

        noise = torch.randn_like(mu_t)
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