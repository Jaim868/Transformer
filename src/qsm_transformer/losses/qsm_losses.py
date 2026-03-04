from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QSMCompositeLoss(nn.Module):
    """Loss for ill-posed QSM inverse problems.

    Combines:
    - L1 reconstruction term for accurate susceptibility estimate
    - Gradient consistency term for edge/lesion sharpness
    - Smoothness regularization for better stability
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _gradient_map(x: torch.Tensor) -> torch.Tensor:
        gx = x[..., 1:, :] - x[..., :-1, :]
        gy = x[..., :, 1:] - x[..., :, :-1]
        gx = F.pad(gx, (0, 0, 0, 1))
        gy = F.pad(gy, (0, 1, 0, 0))
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon = F.l1_loss(pred, target)
        grad = F.l1_loss(self._gradient_map(pred), self._gradient_map(target))
        smooth = self._gradient_map(pred).mean()
        return self.alpha * recon + self.beta * grad + self.gamma * smooth
