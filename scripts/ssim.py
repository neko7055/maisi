import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size):
    mean = window_size / 2
    var = (window_size ** 2 - 1) / 12.0
    gauss = torch.Tensor([exp(-(x - mean) ** 2 / float(2 * var)) for x in range(window_size)])
    # gauss = torch.ones(window_size, dtype=torch.float32)
    return gauss / gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

#@torch.compile(mode="max-autotune", backend="inductor", dynamic=False, fullgraph=False)
def _ssim_3D(img1, img2, window, window_size, channel):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    concat = torch.cat([img1, img2, img1 * img1, img2 * img2, img1 * img2], dim=1)
    window_all = window.repeat(5, 1, 1, 1, 1)
    out = F.conv3d(concat, window_all, padding='same', groups=channel * 5)
    mu1, mu2, sigma1_sq, sigma2_sq, sigma12 = out.chunk(5, dim=1)
    #mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    #mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = sigma1_sq - mu1_sq
    sigma2_sq = sigma2_sq - mu2_sq
    sigma12 = sigma12 - mu1_mu2
    #sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    #sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    #sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2


    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11,channel=4):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window_3D(window_size, self.channel))

    def forward(self, img1, img2):
        return _ssim_3D(img1, img2, self.window, self.window_size, self.channel)

class MS_SSIM3D(torch.nn.Module):
    def __init__(self, weights, window_size=11,channel=4):
        super(MS_SSIM3D, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window_3D(window_size, self.channel))
        self.weights = weights

    def forward(self, img1, img2):
        ssim = 0.0
        for w in self.weights:
            ssim += w * _ssim_3D(img1, img2, self.window, self.window_size, self.channel)
            img1 = F.avg_pool3d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool3d(img2, kernel_size=2, stride=2)
        return ssim


def ssim3D(img1, img2, window_size=11):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel)