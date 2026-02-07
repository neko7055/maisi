import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianInfoNCELoss(nn.Module):
    def __init__(self, sigma=1.0, temperature=1.0):
        """
        Args:
            sigma: Gaussian kernel bandwidth (σ)
            temperature: InfoNCE temperature (τ)
        """
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature
        self.gamma = 1.0 / (2.0 * sigma ** 2)

    def _pairwise_sq_dist(self, x, y):
        """
        x: (K, D)
        y: (K, D)
        return: (K, K) pairwise squared distance
        """
        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # (K, 1)
        y_sq = torch.sum(y ** 2, dim=1, keepdim=True)  # (K, 1)
        return x_sq + y_sq.T - 2.0 * torch.matmul(x, y.T)

    def forward(self, features_nc, features_c):
        """
        Args:
            features_nc: (B, N, D)
            features_c:  (B, N, D)
        """
        # Flatten safely
        flat_nc = features_nc.reshape(-1, features_nc.shape[-1])
        flat_c = features_c.reshape(-1, features_c.shape[-1])

        K = flat_nc.shape[0]

        # Pairwise squared distances
        sq_dist = self._pairwise_sq_dist(flat_nc, flat_c)

        # Gaussian InfoNCE logits
        logits = -self.gamma * sq_dist / self.temperature  # (K, K)

        # Labels: positive pairs on diagonal
        labels = torch.arange(K, device=logits.device)

        # InfoNCE via CrossEntropy
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward_symmetric(self, features_nc, features_c):
        """
        Symmetric version: NC→C and C→NC
        """
        loss_nc = self.forward(features_nc, features_c)
        loss_c = self.forward(features_c, features_nc)
        return 0.5 * (loss_nc + loss_c)


# ================= 使用範例 =================
if __name__ == "__main__":
    # 假設輸入
    batch_size, n_slices, embed_dim = 4, 16, 128
    features_nc = torch.randn(batch_size, n_slices, embed_dim).cuda()
    features_c = torch.randn(batch_size, n_slices, embed_dim).cuda()

    # 初始化 Loss (sigma 控制 kernel 的寬度)
    # sigma 小 -> kernel 窄 -> 只有非常近的才被視為相似
    # sigma 大 -> kernel 寬 -> 較遠的也被視為相似
    criterion = GaussianInfoNCELoss(sigma=1.0).cuda()

    # 計算 Loss
    loss = criterion(features_nc, features_c)
    print(f"Gaussian InfoNCE Loss: {loss.item():.4f}")

    # 雙向版本
    loss_sym = criterion.forward_symmetric(features_nc, features_c)
    print(f"Symmetric Gaussian InfoNCE: {loss_sym.item():.4f}")
