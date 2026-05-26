"""
Policy learning losses: MSE and control-aware composite loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


class ControlAwareLoss(nn.Module):
    """
    loss = w_action * MSE(a, a*)
         + w_instab * Var(pred across batch)
         + w_change * ||a - a_prev||^2
         + w_dose * ReLU(dose - threshold)

    Penalizes abrupt changes and excessive dosing for stable control.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        dose_threshold: float = 50.0,
    ) -> None:
        super().__init__()
        w = weights or {}
        self.w_action = w.get("action", 1.0)
        self.w_instability = w.get("instability", 0.2)
        self.w_aggressive = w.get("aggressive_change", 0.15)
        self.w_excessive = w.get("excessive_dose", 0.1)
        self.dose_threshold = dose_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_err = nn.functional.mse_loss(pred, target)

        # Batch instability: high variance in predictions
        instability = pred.var(dim=0).mean()

        aggressive = torch.tensor(0.0, device=pred.device)
        if prev_pred is not None and prev_pred.shape == pred.shape:
            aggressive = ((pred - prev_pred) ** 2).mean()

        # Dose proxy: flowrate * duration
        dose = pred[:, 0] * pred[:, 1]
        excessive = torch.relu(dose - self.dose_threshold).mean()

        return (
            self.w_action * action_err
            + self.w_instability * instability
            + self.w_aggressive * aggressive
            + self.w_excessive * excessive
        )
