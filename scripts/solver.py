def construct_rk4():
    c = [0.5, 0.5, 1.0]
    a = [
        [0.5],
        [0.0, 0.5],
        [0.0, 0.0, 1.0],
    ]
    b = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
    return c, a, b


def construct_ralston4():
    sqrt_5 = 2.236067977499789696409173668731276235440
    c = [2.0 / 5.0, (14.0 - 3.0 * sqrt_5) / 16.0, 1.0]
    a = [
        [2.0 / 5.0],
        [(-2889.0 + 1428.0 * sqrt_5) / 1024.0, (3785.0 - 1620.0 * sqrt_5) / 1024.0],
        [(-3365.0 + 2094.0 * sqrt_5) / 6040.0, (-975.0 - 3046.0 * sqrt_5) / 2552.0,
         (467040.0 + 203968.0 * sqrt_5) / 240845.0],
    ]
    b = [(263.0 + 24.0 * sqrt_5) / 1812.0, (125.0 - 1000.0 * sqrt_5) / 3828.0,
         (3426304.0 + 1661952.0 * sqrt_5) / 5924787.0, (30.0 - 4.0 * sqrt_5) / 123.0]
    return c, a, b


def construct_rk5():
    c = [1.0 / 3.0, 2.0 / 5.0, 1.0, 2.0 / 3.0, 4.0 / 5.0]
    a = [
        [1.0 / 3.0],
        [4.0 / 25.0, 6.0 / 25.0],
        [1.0 / 4.0, -3.0, 15.0 / 4.0],
        [2.0 / 27.0, 10.0 / 9.0, -50.0 / 81.0, 8.0 / 81.0],
        [2.0 / 25.0, 12.0 / 25.0, 2.0 / 15.0, 8.0 / 75.0, 0.0],
    ]
    b = [23.0 / 192.0, 0.0, 125.0 / 192.0, 0.0, -27.0 / 64.0, 125.0 / 192.0]
    return c, a, b


def euler_step(f, timestep, dt, sample):
    v_pred = f(timestep, sample)
    pred_post_sample = sample + v_pred * dt
    return pred_post_sample

def midpoint_step(f, timestep, dt, sample):
    x_mid = sample + 0.5 * dt * f(timestep, sample)
    v_pred = f(timestep + 0.5 * dt, x_mid)
    pred_post_sample = sample + v_pred * dt
    return pred_post_sample


def rk4_step(f, timestep, dt, sample):
    c, a, b = construct_ralston4()
    k1 = f(timestep, sample)
    k2 = f(timestep + c[0] * dt, sample + dt * a[0][0] * k1)
    k3 = f(timestep + c[1] * dt, sample + dt * (a[1][0] * k1 + a[1][1] * k2))
    k4 = f(timestep + c[2] * dt, sample + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
    v_pred = (
            b[0] * k1
            + b[1] * k2
            + b[2] * k3
            + b[3] * k4
    )
    pred_post_sample = sample + v_pred * dt
    return pred_post_sample


def rk5_step(f, timestep, dt, sample):
    c, a, b = construct_rk5()
    k1 = f(timestep, sample)
    k2 = f(timestep + c[0] * dt, sample + dt * a[0][0] * k1)
    k3 = f(timestep + c[1] * dt, sample + dt * (a[1][0] * k1 + a[1][1] * k2))
    k4 = f(timestep + c[2] * dt, sample + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
    k5 = f(timestep + c[3] * dt, sample + dt * (a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3 + a[3][3] * k4))
    k6 = f(timestep + c[4] * dt,
           sample + dt * (a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4 + a[4][4] * k5))
    v_pred = (
            b[0] * k1
            + b[1] * k2
            + b[2] * k3
            + b[3] * k4
            + b[4] * k5
            + b[5] * k6
    )
    pred_post_sample = sample + v_pred * dt
    return pred_post_sample


if __name__ == "__main__":
    # Example usage
    c, a, b = construct_ralston4()
    print(sum(b))  # Should be 1.0 for a valid Runge-Kutta method
