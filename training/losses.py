"""
Policy learning losses: Weighted masked MSE for handling class imbalance.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


class ActionPairWeightedHuberLoss(nn.Module):
    """
    Sample-weighted Huber loss based on joint action pair frequencies.
    
    Addresses severe action imbalance where ~66.6% of training samples are (0,0) actions.
    Uses inverse frequency weighting on joint (flowrate, duration) action pairs to
    increase the contribution of rare but biologically important dosing decisions.
    
    Key features:
    1. Computes action pair frequencies from training data
    2. Assigns sample weights based on inverse frequency of action pairs
    3. Normalizes weights to maintain average weight close to 1.0
    4. Clips maximum weight to 3-5x majority weight to avoid excessive gradients
    5. Uses SmoothL1 (Huber) loss for robustness to outliers
    6. Reports action frequencies, weights, and weighted vs unweighted losses
    
    This preserves the teacher-student correspondence while reducing underdosing bias.
    """

    def __init__(
        self,
        max_weight_multiplier: float = 4.0,
        beta: float = 1.0,
        quantize_bins: int = 10,
    ) -> None:
        """
        Args:
            max_weight_multiplier: Maximum weight as multiple of majority class weight (default: 4.0)
            beta: Huber loss beta parameter (default: 1.0)
            quantize_bins: Number of bins to quantize continuous actions for frequency computation
                          (default: 10). Set to None to use exact continuous values.
        """
        super().__init__()
        self.max_weight_multiplier = max_weight_multiplier
        self.beta = beta
        self.quantize_bins = quantize_bins
        self._weights_computed = False
        
        # Storage for computed weights and statistics
        self.action_pair_weights: Dict[Tuple[float, float], float] = {}
        self.sample_weights: Optional[np.ndarray] = None
        self.action_pair_freq: Optional[Counter] = None
        self.majority_weight: float = 1.0
        self.average_weight: float = 1.0
        self.weight_stats: Dict[str, float] = {}

    def _quantize_action(self, flowrate: float, duration: float) -> Tuple[float, float]:
        """Quantize continuous actions to bins for frequency computation."""
        if self.quantize_bins is None:
            return (flowrate, duration)
        
        # Quantize flowrate to bins (0 to 5.0 in steps of 0.5)
        flowrate_bin = round(flowrate / 0.5) * 0.5
        flowrate_bin = max(0.0, min(5.0, flowrate_bin))
        
        # Quantize duration to bins (0 to 30.0 in steps of 3.0)
        duration_bin = round(duration / 3.0) * 3.0
        duration_bin = max(0.0, min(30.0, duration_bin))
        
        return (flowrate_bin, duration_bin)

    def compute_weights_from_data(self, y_train: torch.Tensor) -> np.ndarray:
        """
        Compute sample weights from training data based on action pair frequencies.
        
        Args:
            y_train: (N, 2) - training targets (flowrate, duration)
        
        Returns:
            sample_weights: (N,) - computed sample weights
        """
        y_np = y_train.detach().cpu().numpy()
        n_samples = len(y_np)
        
        # Compute action pair frequencies
        action_pairs = []
        for i in range(n_samples):
            fr, dur = y_np[i]
            pair = self._quantize_action(fr, dur)
            action_pairs.append(pair)
        
        self.action_pair_freq = Counter(action_pairs)
        
        # Compute inverse frequency weights
        total_samples = n_samples
        raw_weights = {}
        for pair, count in self.action_pair_freq.items():
            # Inverse frequency weighting
            raw_weights[pair] = total_samples / count
        
        # Set majority class weight to 1.0
        majority_pair = self.action_pair_freq.most_common(1)[0][0]
        self.majority_weight = raw_weights[majority_pair]
        
        # Normalize all weights relative to majority
        for pair in raw_weights:
            raw_weights[pair] /= self.majority_weight
        
        # Clip maximum weight
        max_allowed_weight = self.max_weight_multiplier
        for pair in raw_weights:
            raw_weights[pair] = min(raw_weights[pair], max_allowed_weight)
        
        self.action_pair_weights = raw_weights
        
        # Assign weights to each sample
        sample_weights = np.array([raw_weights[pair] for pair in action_pairs])
        
        # Normalize to maintain average weight close to 1.0
        self.average_weight = sample_weights.mean()
        sample_weights /= self.average_weight
        
        self.sample_weights = sample_weights
        self._weights_computed = True
        
        # Compute statistics for reporting
        self.weight_stats = {
            'min_weight': float(sample_weights.min()),
            'max_weight': float(sample_weights.max()),
            'mean_weight': float(sample_weights.mean()),
            'std_weight': float(sample_weights.std()),
            'majority_pair': majority_pair,
            'majority_freq': self.action_pair_freq[majority_pair],
            'num_unique_pairs': len(self.action_pair_freq),
            'zero_action_freq': self.action_pair_freq.get((0.0, 0.0), 0),
            'zero_action_fraction': self.action_pair_freq.get((0.0, 0.0), 0) / total_samples,
        }
        
        return sample_weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: (batch, 2) - predicted (flowrate, duration)
            target: (batch, 2) - target (flowrate, duration)
        
        Returns:
            Dict with total loss, component losses, and statistics
        """
        if not self._weights_computed:
            raise ValueError(
                "Weights not computed. Call compute_weights_from_data(y_train) first."
            )
        
        batch_size = pred.shape[0]
        device = pred.device
        
        # Compute sample weights for this batch
        batch_sample_weights = []
        for i in range(batch_size):
            fr, dur = target[i].detach().cpu().numpy()
            pair = self._quantize_action(fr, dur)
            weight = self.action_pair_weights.get(pair, 1.0) / self.average_weight
            batch_sample_weights.append(weight)
        
        batch_sample_weights = torch.tensor(
            batch_sample_weights, dtype=torch.float32, device=device
        ).unsqueeze(1)  # (batch, 1)
        
        # Compute SmoothL1 loss per sample and per output
        huber_loss = nn.functional.smooth_l1_loss(
            pred, target, reduction='none', beta=self.beta
        )  # (batch, 2)
        
        # Sum losses across outputs for each sample
        sample_losses = huber_loss.sum(dim=1, keepdim=True)  # (batch, 1)
        
        # Apply sample weights
        weighted_sample_losses = batch_sample_weights * sample_losses
        
        # Compute weighted batch mean
        total_loss = weighted_sample_losses.mean()
        
        # Also compute unweighted loss for comparison
        unweighted_loss = sample_losses.mean()
        
        # Compute per-dimension losses (unweighted for monitoring)
        flowrate_loss = huber_loss[:, 0].mean()
        duration_loss = huber_loss[:, 1].mean()
        
        # Compute weighted per-dimension losses
        weighted_flowrate_loss = (huber_loss[:, 0] * batch_sample_weights.squeeze()).mean()
        weighted_duration_loss = (huber_loss[:, 1] * batch_sample_weights.squeeze()).mean()
        
        return {
            'total_loss': total_loss,
            'unweighted_loss': unweighted_loss,
            'flowrate_loss': flowrate_loss,
            'duration_loss': duration_loss,
            'weighted_flowrate_loss': weighted_flowrate_loss,
            'weighted_duration_loss': weighted_duration_loss,
            'batch_weight_mean': batch_sample_weights.mean().item(),
            'weight_stats': self.weight_stats,
        }


class WeightedMaskedHuberLoss(nn.Module):
    """
    Weighted masked Huber loss for handling class imbalance in regression.
    
    The dataset has ~88.56% zero-action labels (no intervention). Plain MSE
    would cause the model to collapse to predicting (0, 0). Huber loss (SmoothL1)
    is more stable for sparse regression as it reduces gradient explosion on outliers.
    
    This loss:
    1. Assigns higher weight to non-zero action samples independently for flowrate and duration
    2. Assigns lower weight to zero action samples
    3. Weights are computed automatically from training data separately for each action
    4. Uses Huber loss (SmoothL1) for better convergence on sparse targets
    
    This preserves the teacher-student correspondence: both oracle and learner
    predict continuous (flowrate, duration) outputs.
    """

    def __init__(
        self,
        flowrate_positive_weight: Optional[float] = None,
        flowrate_zero_weight: Optional[float] = None,
        duration_positive_weight: Optional[float] = None,
        duration_zero_weight: Optional[float] = None,
        auto_compute_weights: bool = True,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.flowrate_positive_weight = flowrate_positive_weight
        self.flowrate_zero_weight = flowrate_zero_weight
        self.duration_positive_weight = duration_positive_weight
        self.duration_zero_weight = duration_zero_weight
        self.auto_compute_weights = auto_compute_weights
        self._weights_computed = False
        self.beta = beta

    def compute_weights_from_data(self, y_train: torch.Tensor) -> None:
        """
        Compute sample weights from training data to handle class imbalance.
        Computes separate weights for flowrate and duration since they have different distributions.
        
        Args:
            y_train: (N, 2) - training targets (flowrate, duration)
        """
        # Compute weights for flowrate (column 0)
        flowrate_positive_mask = (y_train[:, 0] > 0).float()
        n_flowrate_positive = flowrate_positive_mask.sum().item()
        n_total = len(y_train)
        n_flowrate_zero = n_total - n_flowrate_positive
        
        if n_flowrate_positive > 0 and n_flowrate_zero > 0:
            self.flowrate_positive_weight = n_total / (2 * n_flowrate_positive)
            self.flowrate_zero_weight = n_total / (2 * n_flowrate_zero)
        else:
            self.flowrate_positive_weight = 1.0
            self.flowrate_zero_weight = 1.0
        
        # Compute weights for duration (column 1)
        duration_positive_mask = (y_train[:, 1] > 0).float()
        n_duration_positive = duration_positive_mask.sum().item()
        n_duration_zero = n_total - n_duration_positive
        
        if n_duration_positive > 0 and n_duration_zero > 0:
            self.duration_positive_weight = n_total / (2 * n_duration_positive)
            self.duration_zero_weight = n_total / (2 * n_duration_zero)
        else:
            self.duration_positive_weight = 1.0
            self.duration_zero_weight = 1.0
        
        self._weights_computed = True

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: (batch, 2) - predicted (flowrate, duration)
            target: (batch, 2) - target (flowrate, duration)
        
        Returns:
            Dict with total loss and loss components
        """
        if self.auto_compute_weights and not self._weights_computed:
            raise ValueError(
                "Weights not computed. Call compute_weights_from_data(y_train) first, "
                "or set auto_compute_weights=False and provide explicit weights."
            )
        
        # Compute per-dimension weights
        flowrate_weights = torch.where(
            target[:, 0:1] > 0,
            torch.tensor(self.flowrate_positive_weight, device=pred.device),
            torch.tensor(self.flowrate_zero_weight, device=pred.device),
        )
        duration_weights = torch.where(
            target[:, 1:2] > 0,
            torch.tensor(self.duration_positive_weight, device=pred.device),
            torch.tensor(self.duration_zero_weight, device=pred.device),
        )
        weights = torch.cat([flowrate_weights, duration_weights], dim=1)
        
        # Compute weighted Huber loss (SmoothL1)
        huber_loss = nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=self.beta)
        weighted_huber_loss = weights * huber_loss
        loss = weighted_huber_loss.mean()
        
        # Also compute per-dimension loss for monitoring
        flowrate_loss = (huber_loss[:, 0] * flowrate_weights.squeeze()).mean()
        duration_loss = (huber_loss[:, 1] * duration_weights.squeeze()).mean()
        
        return {
            'total_loss': loss,
            'flowrate_loss': flowrate_loss,
            'duration_loss': duration_loss,
            'flowrate_positive_weight': self.flowrate_positive_weight,
            'flowrate_zero_weight': self.flowrate_zero_weight,
            'duration_positive_weight': self.duration_positive_weight,
            'duration_zero_weight': self.duration_zero_weight,
        }


class ControlAwareLoss(nn.Module):
    """
    Control-aware loss with gradient regularization.
    
    Inspect each component to determine if it's EC-based or pure regularization:
    - action: MSE loss on predictions
    - instability: batch variance - pure regularization
    - aggressive_change: difference from previous action - pure regularization
    - excessive_dose: penalty for large doses - EC-based (remove)
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
        # excessive_dose is EC-based - set to 0 by default
        self.w_excessive = w.get("excessive_dose", 0.0)
        self.dose_threshold = dose_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_err = nn.functional.mse_loss(pred, target)

        # Batch instability: high variance in predictions (pure regularization)
        instability = pred.var(dim=0).mean()

        # Aggressive change penalty (pure regularization)
        aggressive = torch.tensor(0.0, device=pred.device)
        if prev_pred is not None and prev_pred.shape == pred.shape:
            aggressive = ((pred - prev_pred) ** 2).mean()

        # Excessive dose penalty (EC-based - removed by default)
        dose = pred[:, 0] * pred[:, 1]
        excessive = torch.relu(dose - self.dose_threshold).mean()

        return (
            self.w_action * action_err
            + self.w_instability * instability
            + self.w_aggressive * aggressive
            + self.w_excessive * excessive
        )
