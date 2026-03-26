import torch
from torch.optim import Optimizer
from collections import defaultdict
from typing import List, Tuple

class Lion(Optimizer):
    """
    Lion gradient optimizer created by Google Research.
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-4,
        *,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Initialize the Optimizer.

        Args:
            params (List[torch.Tensor]): Parameters of model to optimize.
            lr (float): Learning rate of the optimizer, default is 1e-4.
            betas (Tuple[float, float]): Coefficients used for computing running averages
                of gradient and its square, default values are 0.9, 0.99.
            weight_decay (float): Weight decay coefficient, default is 0.0.
        """

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate {lr}, value must be positive!")

        for index, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(
                    f"Invalid beta value at index {index}: {beta}, "
                    "value must be between 0.0 and 1.0!"
                )

        # if not 0.0 <= weight_decay < 1.0:
        #     raise ValueError(
        #         f"Invalid weight_decay {weight_decay}, ",
        #         "value must be between 0.0 and 1.0!",
        #     )

        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure: callable = None) -> float | None:
        """
        Performs a single optimization step.

        Args:
            closure (callable): Optional, a closure that reevaluates the model
                and returns the loss value.

        Returns:
            Union[float, None]: Loss value if closure was provided, else None.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for params in group["params"]:
                if params.grad is None:
                    continue

                # Perform weights decay.
                params.data.mul_(1 - group["lr"] * group["weight_decay"])

                gradient = params.grad
                state = self.state[params]

                # Initialize state.
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(params)

                exponential_average = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Update the weights.
                update = exponential_average * beta1 + gradient * (1 - beta1)
                params.add_(update.sign_(), alpha=-group["lr"])

                # Update the exponential average.
                exponential_average.mul_(beta2).add_(gradient, alpha=1 - beta2)

        return loss

class Lookahead(Optimizer):
    def __init__(self, optimizer, alpha=0.5, pullback_momentum="pullback"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        assert pullback_momentum in ["reset", "pullback", "none"]

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Capture initial slow weights (theta_0); device may be CPU here.
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = p.data.clone()

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Save both inner optimizer state and lookahead state
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lookahead_state': dict(self.state),  # convert defaultdict -> dict
        }

    def load_state_dict(self, state_dict):
        # Restore lookahead state if present; fallback for older checkpoints
        if 'lookahead_state' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.state = defaultdict(dict, state_dict['lookahead_state'])  # restore defaultdict
        else:
            self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Swap to slow weights for evaluation."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = p.data.clone()
                if param_state['cached_params'].device != p.device:
                    param_state['cached_params'] = param_state['cached_params'].to(p.device)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @torch.no_grad()
    def apply_lookahead_steps(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                # Device guard: ensure cached_params matches p.device
                if param_state['cached_params'].device != p.device:
                    param_state['cached_params'] = param_state['cached_params'].to(p.device)

                # 更新慢權重，並將快權重同步為新的慢權重
                p.data.mul_(self.alpha).add_(param_state['cached_params'],
                                             alpha=1.0 - self.alpha)
                param_state['cached_params'].copy_(p.data)

                # 處理動量 (Momentum)
                if self.pullback_momentum == "pullback":
                    for momentum_buffer in ["exp_avg", "exp_avg_sq"]:
                        if momentum_buffer not in self.optimizer.state[p]:
                            continue

                        cached_name = "cached_" + momentum_buffer
                        internal_momentum = self.optimizer.state[p][momentum_buffer]

                        if cached_name not in param_state:
                            param_state[cached_name] = internal_momentum.clone()
                        elif param_state[cached_name].device != p.device:
                            param_state[cached_name] = param_state[cached_name].to(p.device)

                        # 線性插值動量
                        internal_momentum.mul_(self.alpha).add_(
                            param_state[cached_name], alpha=1.0 - self.alpha)
                        param_state[cached_name] = internal_momentum.clone()

                elif self.pullback_momentum == "reset":
                    for momentum_buffer in ["exp_avg", "exp_avg_sq"]:
                        # 【修正處 1】: 必須檢查內部的優化器狀態
                        if momentum_buffer not in self.optimizer.state[p]:
                            continue

                        # 【修正處 2】: 使用 in-place 的 .zero_() 防止破壞記憶體參考
                        self.optimizer.state[p][momentum_buffer].zero_()

    def step(self, closure=None):
        """Performs a single Lookahead optimization step."""
        loss = self.optimizer.step(closure)

        return loss