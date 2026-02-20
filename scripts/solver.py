import torch

def construct_rk4(dtype):
    c = torch.tensor([0.5, 0.5, 1.0], dtype=dtype)
    a = [
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([0.0, 0.5], dtype=dtype),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
    ]
    b = torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=dtype)
    return c, a, b

def construct_ralston4(dtype):
    sqrt_5 = 2.236067977499789696409173668731276235440
    c = torch.tensor([2/5, (14 - 3*sqrt_5)/16, 1.0], dtype=dtype)
    a = [
        torch.tensor([2/5], dtype=dtype),
        torch.tensor([(-2889 + 1428*sqrt_5)/1024, (3785 - 1620*sqrt_5)/1024], dtype=dtype),
        torch.tensor([(-3365 + 2094*sqrt_5)/6040, (-975 - 3046*sqrt_5)/2552, (467040 + 203968*sqrt_5)/240845], dtype=dtype),
    ]
    b = torch.tensor([(263 + 24*sqrt_5)/1812, (125 - 1000*sqrt_5)/3828, (3426304 + 1661952*sqrt_5)/5924787, (30 - 4*sqrt_5)/123], dtype=dtype)
    return c, a, b

def construct_rk5(dtype):
    c = torch.tensor([1/3, 2/5, 1, 2/3, 4/5], dtype=dtype)
    a = [
        torch.tensor([1/3], dtype=dtype),
        torch.tensor([4/25, 6/25], dtype=dtype),
        torch.tensor([1/4, -3, 15/4], dtype=dtype),
        torch.tensor([2/27, 10/9, -50/81, 8/81], dtype=dtype),
        torch.tensor([2/25, 12/25, 2/15, 8/75, 0], dtype=dtype),
    ]
    b = torch.tensor([23/192, 0, 125/192, 0, -27/64, 125/192], dtype=dtype)
    return c, a, b


def euler_step(self, f, timestep: float, sample: torch.Tensor, next_timestep: float | None = None):
    if next_timestep is not None:
        next_timestep = next_timestep
        dt: float = (
                float(timestep - next_timestep) / self.num_train_timesteps
        )  # Now next_timestep is guaranteed to be int
    else:
        dt = (
            1.0 / float(self.num_inference_steps) if self.num_inference_steps > 0 else 0.0
        )  # Avoid division by zero
    v_pred = f(timestep, sample)
    pred_post_sample = sample + v_pred * dt
    pred_original_sample = sample + v_pred * timestep / self.num_train_timesteps
    return pred_post_sample, pred_original_sample

def midpoint_step(self, f, timestep: float, sample: torch.Tensor, next_timestep: float | None = None):
    if next_timestep is not None:
        next_timestep = next_timestep
        dt: float = (
                float(timestep - next_timestep) / self.num_train_timesteps
        )  # Now next_timestep is guaranteed to be int
    else:
        dt = (
            1.0 / float(self.num_inference_steps) if self.num_inference_steps > 0 else 0.0
        )  # Avoid division by zero
    x_mid = sample + 0.5 * dt * f(timestep, sample)
    v_pred = f(timestep + 0.5 * dt, x_mid)
    pred_post_sample = sample + v_pred * dt
    pred_original_sample = sample + v_pred * timestep / self.num_train_timesteps
    return pred_post_sample, pred_original_sample

def rk4_step(self, f, timestep: float, sample: torch.Tensor, next_timestep: float | None = None):
    if next_timestep is not None:
        next_timestep = next_timestep
        dt: float = (
                float(timestep - next_timestep) / self.num_train_timesteps
        )  # Now next_timestep is guaranteed to be int
    else:
        dt = (
            1.0 / float(self.num_inference_steps) if self.num_inference_steps > 0 else 0.0
        )  # Avoid division by zero
    c, a, b= construct_ralston4(torch.float64)
    k1 = f(timestep, sample)
    k2 = f(timestep + c[0] * dt, sample + dt * a[0] * k1)
    k3 = f(timestep + c[1] * dt, sample + dt * (a[1][0] * k1 + a[1][1] * k2))
    k4 = f(timestep + c[2] * dt, sample + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
    v_pred = (
            b[0] * k1
            + b[1] * k2
            + b[2] * k3
            + b[3] * k4
        )
    pred_post_sample = sample + v_pred * dt
    pred_original_sample = sample + v_pred * timestep / self.num_train_timesteps
    return pred_post_sample, pred_original_sample

def rk5_step(self, f, timestep: float, sample: torch.Tensor, next_timestep: float | None = None):
    if next_timestep is not None:
        next_timestep = next_timestep
        dt: float = (
                float(timestep - next_timestep) / self.num_train_timesteps
        )  # Now next_timestep is guaranteed to be int
    else:
        dt = (
            1.0 / float(self.num_inference_steps) if self.num_inference_steps > 0 else 0.0
        )  # Avoid division by zero
    c, a, b = construct_rk5(torch.float64)
    k1 = f(timestep, sample)
    k2 = f(timestep + c[0] * dt, sample + dt * a[0] * k1)
    k3 = f(timestep + c[1] * dt, sample + dt * (a[1][0] * k1 + a[1][1] * k2))
    k4 = f(timestep + c[2] * dt, sample + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
    k5 = f(timestep + c[3] * dt, sample + dt * (a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3 + a[3][3] * k4))
    k6 = f(timestep + c[4] * dt, sample + dt * (a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4 + a[4][4] * k5))
    v_pred = (
            b[0] * k1
            + b[1] * k2
            + b[2] * k3
            + b[3] * k4
            + b[4] * k5
            + b[5] * k6
        )
    pred_post_sample = sample + v_pred * dt
    pred_original_sample = sample + v_pred * timestep / self.num_train_timesteps
    return pred_post_sample, pred_original_sample

if __name__ == "__main__":
    # Example usage
    c, a, b = construct_rk5(torch.float64)
    print(sum(b))  # Should be 1.0 for a valid Runge-Kutta method