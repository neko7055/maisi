import torch
import torch.nn.functional as F

def _ssim_3D(img1, img2, window_size):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool3d(img1, kernel_size=window_size, stride=1, padding=0)
    mu2 = F.avg_pool3d(img2, kernel_size=window_size, stride=1, padding=0)
    second_mu1 = F.avg_pool3d(img1 * img1, kernel_size=window_size, stride=1, padding=0)
    second_mu2 = F.avg_pool3d(img2 * img2, kernel_size=window_size, stride=1, padding=0)
    second_mu12 = F.avg_pool3d(img1 * img2, kernel_size=window_size, stride=1, padding=0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = second_mu1 - mu1_sq
    sigma2_sq = second_mu2 - mu2_sq
    sigma12 = second_mu12 - mu1_mu2


    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM3D, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return _ssim_3D(img1, img2, self.window_size)

class MS_SSIM3D(torch.nn.Module):
    def __init__(self, weights, window_size=11,channel=4):
        super(MS_SSIM3D, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.weights = weights

    def forward(self, img1, img2):
        ssim = _ssim_3D(img1, img2, self.window_size) * self.weights[0]
        for w in self.weights[1:]:
            img1 = F.avg_pool3d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool3d(img2, kernel_size=2, stride=2)
            ssim += w * _ssim_3D(img1, img2, self.window_size)
        return ssim