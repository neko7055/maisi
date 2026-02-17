import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim

import numpy as np

class SinLU(nn.Module):
    def __init__(self):
        super(SinLU,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
    def forward(self,x):
        return torch.sigmoid(x)*(x+self.a*torch.sin(self.b*x))

# ======= 可逆線性層 =======
class InvertibleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 任意矩陣 M，通過 matrix_exp 生成可逆權重 W
        self.M = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        W = torch.matrix_exp(self.M)  # W 永遠可逆
        return x @ W.T + self.bias  # 注意 PyTorch batch 運算順序


# ======= 可逆線性層 (LU + exp 重參數化) =======
class InvertibleLinearLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # P, L, U 分別參數化
        # 初始化用隨機矩陣
        #self.P = torch.eye(dim)  # 固定 permutation，簡化
        self.register_buffer("P", torch.eye(dim))  # P 不參與訓練
        self.L = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.U = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # 對角元素單獨參數化，確保非零
        self.log_diag_L = nn.Parameter(torch.zeros(dim))
        self.log_diag_U = nn.Parameter(torch.zeros(dim))

        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Lower triangular with ones on diagonal
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.exp(self.log_diag_L))
        # Upper triangular with positive diagonal
        U = torch.triu(self.U, diagonal=1) + torch.diag(torch.exp(self.log_diag_U))
        W = self.P @ L @ U  # 最終可逆權重
        return x @ W.T + self.bias


# ======= Coupling 層 =======
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim, change_order=False):
        """
        dim: 輸入維度，必須能整除 2
        hidden_dim: g(s/t) 層的隱藏維度
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for coupling layer"
        self.dim = dim
        self.d = dim // 2
        self.change_order = change_order

        # s(x1) 和 t(x1) 的神經網路
        self.net = nn.Sequential(
            nn.Linear(self.d, hidden_dim),
            SinLU(),
            InvertibleLinearLU(hidden_dim),
            SinLU(),
            nn.Linear(hidden_dim, self.d * 2)  # 同時輸出 scale 和 shift
        )

    def forward(self, x):
        if self.change_order:
            x2, x1 = x[:, :self.d], x[:, self.d:]
        else:
            x1, x2 = x[:, :self.d], x[:, self.d:]
        st = self.net(x1)
        s, t = st[:, :self.d], st[:, self.d:]
        s = torch.nn.functional.softsign(s)  # 限制 scale 避免太大
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=1) if not self.change_order else torch.cat([y2, y1], dim=1)

# ======= 單層 MLP =======
class MLPInvertible(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = InvertibleLinearLU(dim)
        self.ffn = nn.Sequential(AffineCoupling(dim, hidden_dim=dim, change_order=False),
                                 AffineCoupling(dim, hidden_dim=dim, change_order=True),
                                 AffineCoupling(dim, hidden_dim=dim, change_order=False),
                                 AffineCoupling(dim, hidden_dim=dim, change_order=True),)
        self.activation = nn.Softsign()
        self.out_linear = InvertibleLinearLU(dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.ffn(x)
        x = self.activation(x)
        x = self.out_linear(x)
        return x


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 生成分類數據集
    np.random.seed(0)
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    # add dirty data (change labels of 10% data)
    n_dirty = int(0.25 * len(y))
    dirty_indices = np.random.choice(len(y), n_dirty, replace=False)
    y[dirty_indices] = 1 - y[dirty_indices]  # flip labels
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    model = torch.nn.Sequential(MLPInvertible(dim=2),MLPInvertible(dim=2), nn.Linear(2,2))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=0.0001)
    # 訓練模型
    n = 5000
    for epoch in range(n):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{n}], Loss: {loss.item()}')

    # 繪製決策邊界
    draw_decision_boundary(model, X.numpy(), y.numpy())
