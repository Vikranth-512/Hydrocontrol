"""
Reserve-Ratio Model Predictive Control (MPC) baseline controller.

A lightweight, receding-horizon predictive controller regulating intracellular
nutrient reserve ratio rho = R / (B * C_int) toward a target setpoint.

Key Architecture:
- Primary feedback variable: reserve ratio rho in [0, 1].
- Predictive model: online regularized autoregressive model with exogenous dosing lags.
    Delta_rho_hat_t = c + a*(rho_t - rho_ref) + sum_{j=0}^{L-1} b_j * u_tilde_{t-j}
- Online parameter estimation: regularized Ridge regression with fixed Gaussian prior on a
  sliding history window; robust against lack of excitation and constant histories.
- Action generation: deterministic, bounded candidate sequence library (<= 30 unique sequences)
  over prediction horizon H=5, respecting actuator limits and cooldown constraints.
- Objective function: finite-horizon cost balancing tracking error, low-reserve penalty,
  dosing effort, and action change penalties.
- Receding horizon: optimizes sequence over H steps, applies first action, and replans.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from simulation.state_utils import compute_rho


# Canonical actuator reference values for normalization
MAX_FLOWRATE: float = 5.0    # mL/min
MAX_DURATION: float = 30.0   # seconds
MAX_DOSE_VOLUME: float = (MAX_FLOWRATE * MAX_DURATION) / 60.0  # 2.5 mL


@dataclass
class MPCReserveConfig:
    """Configuration for reserve-ratio Model Predictive Control baseline."""

    # Setpoint and safety thresholds
    target_reserve_ratio: float = 0.50     # Configured reference operating point rho_ref
    minimum_reserve_ratio: float = 0.20    # Starvation boundary rho_min from ecosystem_control
    
    # Prediction and search horizon
    prediction_horizon: int = 5           # Horizon H in controller decision steps (5 steps = 300s = 5 min)
    max_candidate_sequences: int = 30     # Bounded candidate sequence library limit
    action_history_length: int = 3        # Number of historical dosing lags L in model (lags 0, 1, 2)
    
    # Objective function weights
    tracking_weight: float = 1.0          # w_rho: weight on (rho_hat - rho_ref)^2
    low_reserve_weight: float = 5.0       # w_low: weight on max(0, rho_min - rho_hat)^2
    dose_effort_weight: float = 0.10      # w_u: weight on u_tilde^2
    action_change_weight: float = 0.05    # w_delta_u: weight on (u_tilde_k - u_tilde_{k-1})^2
    
    # Online model estimation parameters
    ridge_lambda: float = 1.0             # L2 regularization strength for online parameter update
    history_window_length: int = 60       # Window size W of past transitions used for ridge regression
    
    # Default model prior parameters theta_0 = [c_0, a_0, b_0, b_1, b_2]
    prior_c: float = -0.002               # Unperturbed baseline drift per step (~ -0.002 due to maintenance)
    prior_a: float = -0.02                # Local reserve feedback damping
    prior_b: Tuple[float, ...] = (0.05, 0.05, 0.03)  # Dosing lag responses across lags 0, 1, 2
    
    # Actuator specifications and discrete levels
    min_time_between_doses: float = 120.0 # Actuator cooldown in seconds (2 steps of dt=60s)
    low_flowrate: float = 2.0             # Moderate intervention flowrate (mL/min)
    low_duration: float = 15.0            # Moderate intervention duration (s)
    critical_flowrate: float = 4.0        # Severe intervention flowrate (mL/min)
    critical_duration: float = 20.0       # Severe intervention duration (s)
    no_flowrate: float = 0.0              # Zero intervention flowrate
    no_duration: float = 0.0              # Zero intervention duration
    
    # Internal capacity for state_utils
    internal_capacity: float = 0.5


def normalize_action(flowrate: float, duration: float) -> float:
    """
    Normalize action (flowrate, duration) into scalar effort u_tilde in [0.0, 1.0].
    
    Normalized relative to maximum single dose volume (2.5 mL):
      u_tilde = (flowrate * duration / 60.0) / MAX_DOSE_VOLUME
    """
    vol = max(0.0, float(flowrate)) * max(0.0, float(duration)) / 60.0
    return float(np.clip(vol / MAX_DOSE_VOLUME, 0.0, 1.0))


class MPCReserveController:
    """
    Lightweight receding-horizon Model Predictive Controller regulating reserve ratio.

    Follows standard controller lifecycle:
      - reset(): re-initializes online model, action history, and cooldown timers
      - compute_from_reserve(rho, ...): plans and selects optimal next action
      - compute(...): generic dispatch shim compatible with all baseline callers
    """

    def __init__(self, config: Optional[MPCReserveConfig] = None) -> None:
        self.config = config or MPCReserveConfig()
        
        # Action definitions
        cfg = self.config
        self.action_levels: List[Tuple[float, float]] = [
            (cfg.no_flowrate, cfg.no_duration),            # Level 0: No dose
            (cfg.low_flowrate, cfg.low_duration),          # Level 1: Low dose
            (cfg.critical_flowrate, cfg.critical_duration), # Level 2: High dose
        ]
        self.normalized_levels: List[float] = [
            normalize_action(fr, dur) for fr, dur in self.action_levels
        ]

        # Parameter dimensions
        self.L: int = cfg.action_history_length
        self.d: int = 2 + self.L  # [c, a, b_0, ..., b_{L-1}]
        
        # Build prior parameter vector theta_0
        b_priors = list(cfg.prior_b)[:self.L]
        if len(b_priors) < self.L:
            b_priors.extend([0.02] * (self.L - len(b_priors)))
        self.theta_0 = np.array([cfg.prior_c, cfg.prior_a] + b_priors, dtype=np.float64)

        # Precompute deterministic candidate action sequence library
        self._candidate_library = self._generate_candidate_library()

        # State and diagnostics (initialized in reset)
        self.reset()

    def reset(self) -> None:
        """Reset internal controller state, history buffers, and model estimates."""
        cfg = self.config
        self.time_since_last_dose: float = float("inf")
        self.last_commanded_action: Tuple[float, float] = (0.0, 0.0)
        self.last_commanded_norm_u: float = 0.0
        self.prev_rho: Optional[float] = None
        
        # Sliding action history buffer of commanded actions: [u_{t-1}, u_{t-2}, ...]
        self.action_history: deque[float] = deque([0.0] * self.L, maxlen=self.L)
        
        # Online regression history buffer: stores (feature_vector_x, delta_rho_y)
        self.regression_history: deque[Tuple[np.ndarray, float]] = deque(
            maxlen=cfg.history_window_length
        )
        
        # Current active parameter estimate
        self.current_theta: np.ndarray = self.theta_0.copy()
        
        # Runtime diagnostics
        self.last_decision_latency_ms: float = 0.0
        self.last_candidates_evaluated: int = 0
        self.total_decisions: int = 0
        self.total_decision_time_ms: float = 0.0

    def _generate_candidate_library(self) -> List[List[Tuple[float, float]]]:
        """
        Construct a deterministic, bounded library of candidate action sequences.
        
        Respects:
          - Maximum candidate sequences limit (<= 30)
          - Actuator cooldown (no two consecutive non-zero doses)
          - Balanced coverage: zero, single pulses at each horizon step,
            two-pulse sequences, and cooldown-compliant multi-pulses.
        """
        cfg = self.config
        H = cfg.prediction_horizon
        act = self.action_levels
        
        candidates: List[List[Tuple[float, float]]] = []
        
        # 1. Constant no-dose baseline
        candidates.append([act[0]] * H)
        
        # 2. Single pulses at each step k in [0, H-1]
        for k in range(H):
            # Low pulse
            seq_low = [act[0]] * H
            seq_low[k] = act[1]
            candidates.append(seq_low)
            
            # High pulse
            seq_high = [act[0]] * H
            seq_high[k] = act[2]
            candidates.append(seq_high)
            
        # 3. Two pulses separated by cooldown (separated by at least one zero step)
        for i in range(H):
            for j in range(i + 2, H):
                # Low then Low
                s_ll = [act[0]] * H
                s_ll[i], s_ll[j] = act[1], act[1]
                candidates.append(s_ll)
                
                # High then High
                s_hh = [act[0]] * H
                s_hh[i], s_hh[j] = act[2], act[2]
                candidates.append(s_hh)
                
                # Low then High
                s_lh = [act[0]] * H
                s_lh[i], s_lh[j] = act[1], act[2]
                candidates.append(s_lh)
                
                # High then Low
                s_hl = [act[0]] * H
                s_hl[i], s_hl[j] = act[2], act[1]
                candidates.append(s_hl)

        # 4. Three pulses (only possible if H >= 5, e.g. steps 0, 2, 4)
        if H >= 5:
            s3_low = [act[0]] * H
            s3_low[0], s3_low[2], s3_low[4] = act[1], act[1], act[1]
            candidates.append(s3_low)
            
            s3_high = [act[0]] * H
            s3_high[0], s3_high[2], s3_high[4] = act[2], act[2], act[2]
            candidates.append(s3_high)

        # Deduplicate while preserving order
        unique_candidates: List[List[Tuple[float, float]]] = []
        seen = set()
        for c in candidates:
            c_tuple = tuple(c)
            if c_tuple not in seen:
                seen.add(c_tuple)
                unique_candidates.append(c)

        # Cap at configured max candidate sequences
        return unique_candidates[:cfg.max_candidate_sequences]

    def _update_model(self, current_rho: float) -> None:
        """
        Update approximate response model using previous transition observation.
        
        Uses regularized Ridge regression with fixed Gaussian prior:
          theta_hat = (X^T X + lambda*I)^(-1) (X^T y + lambda*theta_0)
        Guarantees finite, bounded coefficients without matrix singularity.
        """
        cfg = self.config
        
        # If we have a previous observation, record transition (x_{t-1}, delta_rho_{t-1})
        if self.prev_rho is not None:
            delta_rho = current_rho - self.prev_rho
            
            # Feature vector x = [1.0, prev_rho - rho_ref, u_{t-1}, u_{t-2}, ..., u_{t-L}]
            x_feat = np.zeros(self.d, dtype=np.float64)
            x_feat[0] = 1.0
            x_feat[1] = self.prev_rho - cfg.target_reserve_ratio
            for j in range(self.L):
                x_feat[2 + j] = self.action_history[j] if j < len(self.action_history) else 0.0

            # Guard against non-finite values
            if np.all(np.isfinite(x_feat)) and np.isfinite(delta_rho):
                self.regression_history.append((x_feat, delta_rho))

        # Update theta if we have sufficient observations
        if len(self.regression_history) >= 2:
            X = np.array([item[0] for item in self.regression_history], dtype=np.float64)
            y = np.array([item[1] for item in self.regression_history], dtype=np.float64)
            
            # Ridge normal equations: (X^T X + lambda * I) theta = X^T y + lambda * theta_0
            reg_matrix = X.T @ X + cfg.ridge_lambda * np.eye(self.d)
            rhs = X.T @ y + cfg.ridge_lambda * self.theta_0
            
            try:
                theta_est = np.linalg.solve(reg_matrix, rhs)
                
                # Check for numerical validity
                if np.all(np.isfinite(theta_est)):
                    # Conservative physical parameter bounding:
                    # c in [-0.05, 0.02] (drift)
                    # a in [-0.20, 0.00] (negative feedback damping)
                    # b_j in [0.00, 0.25] (positive nutrient dosing effect)
                    theta_bounded = theta_est.copy()
                    theta_bounded[0] = np.clip(theta_bounded[0], -0.05, 0.02)
                    theta_bounded[1] = np.clip(theta_bounded[1], -0.20, 0.00)
                    for j in range(self.L):
                        theta_bounded[2 + j] = np.clip(theta_bounded[2 + j], 0.00, 0.25)
                    self.current_theta = theta_bounded
                else:
                    self.current_theta = self.theta_0.copy()
            except np.linalg.LinAlgError:
                self.current_theta = self.theta_0.copy()
        else:
            self.current_theta = self.theta_0.copy()

    def _predict_trajectory(
        self,
        current_rho: float,
        candidate_u_seq: List[float],
    ) -> np.ndarray:
        """
        Roll out approximate response model over horizon H for a normalized candidate action sequence.
        
        Returns:
            predicted_rho: array of length H containing [rho_hat_1, ..., rho_hat_H].
        """
        cfg = self.config
        H = cfg.prediction_horizon
        theta = self.current_theta
        c = theta[0]
        a = theta[1]
        b = theta[2:2 + self.L]
        
        predicted_rho = np.zeros(H, dtype=np.float64)
        rho_k = current_rho
        
        # Build history of actions for lag retrieval:
        # past actions: [u_{t-L}, ..., u_{t-2}, u_{t-1}]
        hist_list = list(reversed(list(self.action_history)))
        
        for k in range(H):
            # Form dosing lag inputs for step k
            u_lags = np.zeros(self.L, dtype=np.float64)
            for j in range(self.L):
                idx = k - j
                if idx >= 0:
                    u_lags[j] = candidate_u_seq[idx]
                else:
                    hist_idx = len(hist_list) + idx
                    u_lags[j] = hist_list[hist_idx] if 0 <= hist_idx < len(hist_list) else 0.0

            # Delta_rho_hat = c + a*(rho_k - rho_ref) + sum_j(b_j * u_lag_j)
            delta_rho_hat = c + a * (rho_k - cfg.target_reserve_ratio) + np.dot(b, u_lags)
            
            # Next state
            rho_next = rho_k + delta_rho_hat
            # Gentle numerical safeguard against divergence
            if not np.isfinite(rho_next):
                rho_next = float("nan")
            else:
                rho_next = float(np.clip(rho_next, 0.0, 1.2))
                
            predicted_rho[k] = rho_next
            rho_k = rho_next
            
        return predicted_rho

    def _evaluate_objective(
        self,
        current_rho: float,
        candidate_seq: List[Tuple[float, float]],
        candidate_norm_u: List[float],
        cooldown_active: bool,
    ) -> float:
        """
        Evaluate MPC objective function for candidate sequence.
        
        Returns:
            Scalar objective J or float('inf') if invalid / infeasible.
        """
        cfg = self.config
        H = cfg.prediction_horizon
        
        # Feasibility check: if cooldown is active, candidate cannot dose at step 0
        if cooldown_active and (candidate_seq[0][0] > 0.0 or candidate_seq[0][1] > 0.0):
            return float("inf")
        
        # Cooldown check across candidate sequence (cannot have consecutive non-zero doses)
        for k in range(H - 1):
            if (candidate_seq[k][0] > 0.0 or candidate_seq[k][1] > 0.0) and (
                candidate_seq[k + 1][0] > 0.0 or candidate_seq[k + 1][1] > 0.0
            ):
                return float("inf")

        # Predict reserve ratio trajectory
        pred_rho = self._predict_trajectory(current_rho, candidate_norm_u)
        if not np.all(np.isfinite(pred_rho)):
            return float("inf")

        # 1. Reserve ratio tracking error penalty: sum_{k=1}^H w_rho * (rho_hat - rho_ref)^2
        tracking_penalty = cfg.tracking_weight * np.sum(
            (pred_rho - cfg.target_reserve_ratio) ** 2
        )

        # 2. Low-reserve risk penalty: sum_{k=1}^H w_low * max(0, rho_min - rho_hat)^2
        low_reserve_deficit = np.maximum(0.0, cfg.minimum_reserve_ratio - pred_rho)
        low_reserve_penalty = cfg.low_reserve_weight * np.sum(low_reserve_deficit ** 2)

        # 3. Dosing effort penalty: sum_{k=0}^{H-1} w_u * u_tilde_k^2
        norm_u_arr = np.array(candidate_norm_u, dtype=np.float64)
        effort_penalty = cfg.dose_effort_weight * np.sum(norm_u_arr ** 2)

        # 4. Action change penalty: sum_{k=0}^{H-1} w_delta_u * (u_tilde_k - u_tilde_{k-1})^2
        prev_u = self.last_commanded_norm_u
        u_with_prev = np.concatenate(([prev_u], norm_u_arr))
        delta_u = np.diff(u_with_prev)
        action_change_penalty = cfg.action_change_weight * np.sum(delta_u ** 2)

        total_cost = tracking_penalty + low_reserve_penalty + effort_penalty + action_change_penalty
        return float(total_cost) if np.isfinite(total_cost) else float("inf")

    def compute_from_reserve(
        self,
        rho: float,
        time_since_last_dose: Optional[float] = None,
        dt: float = 60.0,
    ) -> Tuple[float, float]:
        """
        Evaluate MPC control decision given reserve ratio observation rho in [0, 1].

        Args:
            rho: Intracellular reserve ratio observation.
            time_since_last_dose: Optional actuator cooldown timer in seconds.
            dt: Controller decision interval in seconds (default 60.0s).

        Returns:
            (flowrate mL/min, duration s)
        """
        start_time = time.perf_counter()
        cfg = self.config

        # Numerical input safeguards
        if not np.isfinite(rho):
            # Non-finite observation fallback
            return cfg.no_flowrate, cfg.no_duration
        safe_rho = float(np.clip(rho, 0.0, 1.0))

        # Check actuator cooldown
        tsld = (
            time_since_last_dose
            if time_since_last_dose is not None
            else self.time_since_last_dose
        )
        cooldown_active = tsld < cfg.min_time_between_doses

        # Step 2: Online model estimation update
        self._update_model(safe_rho)

        # Step 3-6: Evaluate candidate sequences
        best_cost = float("inf")
        best_action = (cfg.no_flowrate, cfg.no_duration)
        best_norm_u = 0.0
        candidates_evaluated = 0

        for candidate_seq in self._candidate_library:
            candidates_evaluated += 1
            cand_norm_u = [normalize_action(fr, dur) for fr, dur in candidate_seq]
            
            cost = self._evaluate_objective(
                safe_rho,
                candidate_seq,
                cand_norm_u,
                cooldown_active=cooldown_active,
            )

            # Deterministic selection: strictly lower finite cost
            if cost < best_cost:
                best_cost = cost
                best_action = candidate_seq[0]
                best_norm_u = cand_norm_u[0]
            elif cost == best_cost and cost < float("inf"):
                # Tie-breaking: prefer lower dosing effort
                if cand_norm_u[0] < best_norm_u:
                    best_action = candidate_seq[0]
                    best_norm_u = cand_norm_u[0]

        # Step 7: Receding horizon execution — execute only the first action
        selected_action = best_action
        selected_norm_u = best_norm_u

        # Update controller internal state
        if selected_action[0] > 0.0 and selected_action[1] > 0.0:
            self.time_since_last_dose = 0.0
        else:
            self.time_since_last_dose = tsld + dt

        self.last_commanded_action = selected_action
        self.last_commanded_norm_u = selected_norm_u
        self.action_history.appendleft(selected_norm_u)
        self.prev_rho = safe_rho

        # Timing and diagnostics
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_decision_latency_ms = elapsed_ms
        self.last_candidates_evaluated = candidates_evaluated
        self.total_decisions += 1
        self.total_decision_time_ms += elapsed_ms

        return selected_action

    def compute_from_state(
        self,
        state: Any,
        internal_capacity: float = 0.5,
        dt: float = 60.0,
    ) -> Tuple[float, float]:
        """Evaluate MPC decision given a TankState or state-like object."""
        rho = compute_rho(state, internal_capacity)
        tsld = getattr(state, "time_since_last_dose", None)
        return self.compute_from_reserve(rho, time_since_last_dose=tsld, dt=dt)

    def compute(self, val: Any, *args: Any, **kwargs: Any) -> Tuple[float, float]:
        """
        Generic dispatch shim for supported controller interface:
          - compute_from_reserve(rho) if val is a float or dict with 'reserve_ratio'/'rho'
          - Direct TankState objects are evaluated via compute_from_state(val)
        
        Guarded against hidden-state access:
          If a dictionary is passed without 'reserve_ratio' or 'rho', fallback to 
          hidden state (e.g. val['state']) is prohibited and raises ValueError.
        """
        if isinstance(val, dict):
            if "reserve_ratio" in val:
                tsld = kwargs.pop("time_since_last_dose", val.get("time_since_last_dose", None))
                return self.compute_from_reserve(float(val["reserve_ratio"]), time_since_last_dose=tsld, **kwargs)
            if "rho" in val:
                tsld = kwargs.pop("time_since_last_dose", val.get("time_since_last_dose", None))
                return self.compute_from_reserve(float(val["rho"]), time_since_last_dose=tsld, **kwargs)
            # Guard against hidden-state leakage:
            if "state" in val:
                raise ValueError(
                    "MPCReserveController disallows accessing latent environment state via 'state'. "
                    "Input dictionary must supply observed 'reserve_ratio' or 'rho'."
                )
            return self.config.no_flowrate, self.config.no_duration

        if hasattr(val, "algae_biomass") or hasattr(val, "internal_reserve"):
            return self.compute_from_state(val, **kwargs)

        try:
            return self.compute_from_reserve(float(val), **kwargs)
        except (ValueError, TypeError):
            return self.config.no_flowrate, self.config.no_duration

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[float, float]:
        return self.compute(*args, **kwargs)

    @property
    def average_decision_latency_ms(self) -> float:
        """Average decision latency in milliseconds."""
        return (
            self.total_decision_time_ms / self.total_decisions
            if self.total_decisions > 0
            else 0.0
        )
