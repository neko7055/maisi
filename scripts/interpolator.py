import torch

def linear_interpolate(self, sources, targets, timesteps, add_noise=False) :
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
        dz_coef = (1 - 2 * timepoints) / torch.sqrt(2 * timepoints * (1 - timepoints))
        noise = torch.randn_like(mu_t)
        return mu_t + z_coef * noise, d_mu_t + dz_coef * noise
    else:
        return mu_t, d_mu_t

def triangular_interpolate(self, sources, targets, timesteps, add_noise=False) :
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
        d_gamma = (1 - 2 * timepoints) / torch.sqrt(2 * timepoints * (1 - timepoints))
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