import torch
import torch.nn.functional as F

class SSIM(torch.nn.Module):
    def __init__(self,channels, window_size=11):
        super(SSIM, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.norm = torch.nn.InstanceNorm3d(self.channels, affine=False)
    def forward(self, X, Y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        X = F.sigmoid(self.norm(X))
        Y = F.sigmoid(self.norm(Y))
        X = F.pad(X, (pad, pad, pad, pad, pad, pad), mode="circular")
        Y = F.pad(Y, (pad, pad, pad, pad, pad, pad), mode="circular")
        mu1 = F.avg_pool3d(X, (self.window_size, self.window_size, self.window_size))
        mu2 = F.avg_pool3d(Y, (self.window_size, self.window_size, self.window_size))

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        mu_sq_sum = mu1_sq + mu2_sq
        molecular = 2 * mu1_mu2 + C1
        denominator = mu_sq_sum + C1
        l = molecular / denominator

        # 計算變異數 (sigma^2) 與共變異數 (sigma_12)
        sigma1_sq = F.avg_pool3d(X.pow(2), (self.window_size, self.window_size, self.window_size)) - mu1_sq
        sigma2_sq = F.avg_pool3d(Y.pow(2), (self.window_size, self.window_size, self.window_size)) - mu2_sq
        sigma12 = F.avg_pool3d(X * Y, (self.window_size, self.window_size, self.window_size)) - mu1 * mu2
        sigma_sq_sum = sigma1_sq + sigma2_sq
        molecular = 2 * sigma12 + C2
        denominator = sigma_sq_sum + C2
        cs = molecular / denominator
        # 結構相似性與對比度特徵
        ssim_map = l * cs
        return ssim_map.mean()

if __name__ == '__main__':
    # ssim loss test
    torch.manual_seed(0)
    ssim_fn = SSIM(channels=4, window_size=3)
    loss_fn = LOSS(channels=4, window_size=3)
    x = torch.randn(1, 4, 32, 32, 32, dtype=torch.float32, requires_grad=True,)
    y = torch.randn(1, 4, 32, 32, 32, dtype=torch.float32)#x.detach().clone()#
    optimizer = torch.optim.Adam([x], lr=1e-3)
    for i in range(100000):
        optimizer.zero_grad()
        loss = (1-ssim_fn(x,y))/2 * 0.8 + 0.2 * (x.abs().mean(dim=[1,2,3,4]) - y.abs().mean(dim=[1,2,3,4])).abs().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mse = (x - y).pow(2).mean().item()
            ssim = ssim_fn(x, y)
            print(f"Step {i+1}: Loss={loss.item():.6f}, MSE={mse:.6f}, SSIM={ssim.item():.6f}")
