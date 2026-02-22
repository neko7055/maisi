import torch
from torch.optim import Optimizer
from collections import defaultdict


class Lion(Optimizer):
    r"""Implements Lion algorithm."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign_(), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        assert pullback_momentum in ["reset", "pullback", "none"]

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
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
            'step_counter': self.step_counter,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Save both inner optimizer state and lookahead state
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lookahead_state': dict(self.state),  # convert defaultdict -> dict
            'step_counter': self.step_counter
        }

    def load_state_dict(self, state_dict):
        # Restore lookahead state if present; fallback for older checkpoints
        if 'lookahead_state' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.state = defaultdict(dict, state_dict['lookahead_state'])  # restore defaultdict
            self.step_counter = state_dict['step_counter']
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

    def step(self, closure=None):
        """Performs a single Lookahead optimization step."""
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]

                    # Device guard: ensure cached_params matches p.device
                    if param_state['cached_params'].device != p.device:
                        param_state['cached_params'] = param_state['cached_params'].to(p.device)

                    p.data.mul_(self.alpha).add_(param_state['cached_params'],
                                                 alpha=1.0 - self.alpha)
                    param_state['cached_params'].copy_(p.data)

                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        if "cached_mom" not in param_state:
                            param_state["cached_mom"] = internal_momentum.clone()
                        elif param_state["cached_mom"].device != p.device:
                            param_state["cached_mom"] = param_state["cached_mom"].to(p.device)

                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            param_state["cached_mom"], alpha=1.0 - self.alpha)
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"].clone()
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss