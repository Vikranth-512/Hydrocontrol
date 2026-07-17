"""
Ecosystem Regulation cost function for optimization-based label generation.

Replaces EC setpoint-tracking objective with biological health preservation:

J_total = λ_h  × J_health          (ecosystem viability — irreversibility guard)
        + λ_d  × J_damage           (rate of harm + accumulated harm)
        + λ_r  × J_reserve          (starvation prevention + reserve stability)
        + λ_br × J_biomass_rate     (bloom prevention — upward growth trend only)
        + λ_n  × J_nutrient         (dosing efficiency — pure cost, no reserve coupling)
        + λ_a  × J_action           (control smoothness)
        +        V(s_T)             (terminal sustainability cost)

Lower total is better.

EC is no longer an objective variable. It is retained as a secondary diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from simulation.dynamics import TankDynamicsParams, TankState

_INVALID_COST = 1e6
_INVALID_TOTAL = 1e9


def _ensure_finite(value: float, name: str, debug: bool = False) -> float:
    if np.isfinite(value):
        return float(value)
    if debug:
        raise FloatingPointError(f"Invalid cost term: {name} = {value}")
    return _INVALID_COST


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LabelingObjectiveConfig:
    """Weights and thresholds for the ecosystem regulation objective."""

    # Biological thresholds
    health_floor: float = 0.70          # health below this → elevated mortality
    reserve_floor: float = 0.20         # starvation boundary (~100 steps reserve)
    b_ref: float = 120.0                # normalizer for biomass rate cost
    b_bloom_threshold: float = 300.0    # terminal cost bloom threshold
    rho_setpoint: float = 0.50          # reference operating point (Droop half-quota)
    internal_capacity: float = 0.5      # mirrors dynamics param

    # Cost term weights (λ) - Hierarchical sustainability penalties
    health: float = 1.5
    damage: float = 1.0
    reserve: float = 1.2
    biomass_rate: float = 0.6         # Medium penalty: acceleration/velocity
    biomass_envelope: float = 0.3     # Weak penalty: safe operating zone
    maintenance: float = 2.0          # Strong penalty: root cause of detritus
    detritus: float = 1.0             # Future dissolved penalty
    dissolved_inventory: float = 1.0  # Sustained tank loading
    dissolved_accumulation: float = 3.0 # Very strong penalty: tank poisoning
    osmotic: float = 5.0              # Dominant penalty: uptake shutdown
    nutrient_cost: float = 0.25
    action_smoothness: float = 0.15
    
    # Mechanistic Parameters from Dynamics
    maintenance_rate: float = 0.0005
    osmotic_safe_threshold: float = 0.2

    # Sub-term weights
    reserve_setpoint_weight: float = 2.0      # weight on setpoint tracking in J_reserve
    reserve_low_weight: float = 2.0          # weight on (rho_target - rho)^2 for depleted reserves
    reserve_high_weight: float = 0.5         # weight on (rho - rho_target)^2 for excessive reserves
    reserve_oscillation_weight: float = 0.5   # var(rho) inside J_reserve
    damage_rate_weight: float = 2.0           # Δdamage vs accumulated damage
    nutrient_quadratic_weight: float = 0.5    # (avg_dose)^2 in J_nutrient

    # Rollout fractions
    tail_fraction: float = 0.35        # tail window for late-horizon metrics

    # Misc
    horizon: int = 200
    debug_mode: bool = False

    # Legacy fields retained for backward compatibility (not used in cost)
    ec_target: float = 1.2
    ec_safe_min: float = 0.4
    ec_safe_max: float = 2.5

    @classmethod
    def from_yaml(
        cls,
        labeling: Dict[str, Any],
        sim: Dict[str, Any],
        dyn: Dict[str, Any],
        eco: Optional[Dict[str, Any]] = None,
    ) -> "LabelingObjectiveConfig":
        eco = eco or {}
        w = labeling.get("weights", {})

        def _f(key: str, default: float, *dicts) -> float:
            for d in dicts:
                if key in d:
                    return float(d[key])
            return default

        return cls(
            # Thresholds (ecosystem_control > labeling > defaults)
            health_floor=_f("health_floor", 0.70, eco, labeling),
            reserve_floor=_f("reserve_floor", 0.20, eco, labeling),
            b_ref=_f("b_ref", 120.0, eco, labeling),
            b_bloom_threshold=_f("b_bloom_threshold", 300.0, eco, labeling),
            rho_setpoint=_f("rho_setpoint", 0.50, eco, labeling),
            internal_capacity=_f("internal_capacity", 0.5, eco, dyn),
            # Weights
            health=float(w.get("health", 1.5)),
            damage=float(w.get("damage", 1.0)),
            reserve=float(w.get("reserve", 1.2)),
            biomass_rate=float(w.get("biomass_rate", 0.6)),
            biomass_envelope=float(w.get("biomass_envelope", 0.3)),
            maintenance=float(w.get("maintenance", 2.0)),
            detritus=float(w.get("detritus", 1.0)),
            dissolved_inventory=float(w.get("dissolved_inventory", 1.0)),
            dissolved_accumulation=float(w.get("dissolved_accumulation", 3.0)),
            osmotic=float(w.get("osmotic", 5.0)),
            nutrient_cost=float(w.get("nutrient_cost", 0.25)),
            action_smoothness=float(w.get("action_smoothness", 0.15)),
            
            # Mechanistic Parameters
            maintenance_rate=float(dyn.get("maintenance_cost", 0.0005)),
            osmotic_safe_threshold=float(dyn.get("osmotic_safe_threshold", 0.2)),
            # Sub-term weights
            reserve_setpoint_weight=float(labeling.get("reserve_setpoint_weight", 2.0)),
            reserve_low_weight=float(labeling.get("reserve_low_weight", 2.0)),
            reserve_high_weight=float(labeling.get("reserve_high_weight", 0.5)),
            reserve_oscillation_weight=float(labeling.get("reserve_oscillation_weight", 0.5)),
            damage_rate_weight=float(labeling.get("damage_rate_weight", 2.0)),
            nutrient_quadratic_weight=float(labeling.get("nutrient_quadratic_weight", 0.5)),
            # Rollout
            tail_fraction=float(labeling.get("tail_fraction", 0.35)),
            horizon=int(labeling.get("horizon_steps", 200)),
            debug_mode=bool(labeling.get("debug_mode", False)),
            # Legacy
            ec_target=float(sim.get("ec_target", 1.2)),
            ec_safe_min=float(sim.get("ec_safe_min", 0.4)),
            ec_safe_max=float(sim.get("ec_safe_max", 2.5)),
        )


# ---------------------------------------------------------------------------
# Cost breakdown dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Per-candidate ecosystem cost terms (unweighted raw values)."""

    health: float = 0.0         # J_health
    damage: float = 0.0         # J_damage
    reserve: float = 0.0        # J_reserve
    biomass_rate: float = 0.0   # J_biomass_rate (velocity/acceleration)
    biomass_envelope: float = 0.0 # J_biomass_envelope
    maintenance: float = 0.0    # J_maintenance
    detritus: float = 0.0       # J_detritus
    dissolved_inventory: float = 0.0 # J_dissolved_inventory
    dissolved_accumulation: float = 0.0 # J_dissolved_accumulation
    osmotic: float = 0.0        # J_osmotic
    nutrient: float = 0.0       # J_nutrient
    action: float = 0.0         # J_action
    terminal: float = 0.0       # V(s_T)
    safety: float = 0.0         # hard infeasibility penalties
    total: float = 0.0

    # Diagnostic fields (not in cost)
    reserve_mean: float = 0.0
    health_mean: float = 0.0
    damage_mean: float = 0.0
    biomass_mean: float = 0.0

    def weighted_total(self, cfg: LabelingObjectiveConfig) -> float:
        return (
            cfg.health         * self.health
            + cfg.damage       * self.damage
            + cfg.reserve      * self.reserve
            + cfg.biomass_rate * self.biomass_rate
            + cfg.biomass_envelope * self.biomass_envelope
            + cfg.maintenance  * self.maintenance
            + cfg.detritus     * self.detritus
            + cfg.dissolved_inventory * self.dissolved_inventory
            + cfg.dissolved_accumulation * self.dissolved_accumulation
            + cfg.osmotic      * self.osmotic
            + cfg.nutrient_cost * self.nutrient
            + cfg.action_smoothness * self.action
            + self.terminal
            + self.safety
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rollout state trace helpers
# ---------------------------------------------------------------------------

def compute_rho(state: TankState, internal_capacity: float) -> float:
    """Reserve ratio: clipped to [0, 1]."""
    B = max(state.algae_biomass, 1e-6)
    return float(np.clip(state.internal_reserve / (B * internal_capacity), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def _health_cost(health_trace: np.ndarray, final_health: float, cfg: LabelingObjectiveConfig) -> float:
    """
    J_health = mean_t[ max(0, H_floor - H(t)) ] + 3.0 * terminal deficit.

    Leverages the health_index which already integrates damage hysteresis,
    repair kinetics, and starvation effects from the biology layer.
    Terminal multiplier (×3) encodes irreversibility of collapse.
    """
    H_floor = cfg.health_floor
    deficit = np.maximum(0.0, H_floor - health_trace)
    running = float(np.mean(deficit))
    terminal = 3.0 * max(0.0, H_floor - final_health)
    return running + terminal


def _damage_cost(damage_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    J_damage = mean(D) + λ_drate × mean(max(0, ΔD))

    Running cost (accumulated harm) + rate cost (active deterioration).
    Rate term fires only when damage is actively worsening, not when stable.
    Asymmetric: stable damage incurs no additional rate penalty.
    """
    accumulated = float(np.mean(damage_trace))
    delta_d = np.diff(damage_trace, prepend=damage_trace[0])
    rate = float(np.mean(np.maximum(0.0, delta_d)))
    return accumulated + cfg.damage_rate_weight * rate


def _reserve_cost(rho_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    J_reserve = λ_starv × mean[ max(0, rho_floor - rho)^2 ] + λ_setpoint × [w_low × max(0, rho_target - rho)^2 + w_high × max(0, rho - rho_target)^2] + λ_osc × var(rho)

    Three components:
    1. Starvation penalty: smooth gradient away from starvation boundary
    2. Asymmetric setpoint tracking: penalize low reserves more heavily than high reserves
    3. Oscillation penalty: stability of reserves

    Asymmetric weighting reflects biology: depleted reserves (rho < target) are more dangerous than excessive reserves (rho > target).
    Normalized by reserve_floor for starvation, rho_setpoint for setpoint deviations.
    """
    floor_deficit = np.maximum(0.0, cfg.reserve_floor - rho_trace)
    norm_deficit = floor_deficit / max(cfg.reserve_floor, 1e-6)
    starvation_cost = float(np.mean(norm_deficit ** 2))
    
    # Asymmetric setpoint tracking: penalize low reserves more heavily than high reserves
    low_deficit = np.maximum(0.0, cfg.rho_setpoint - rho_trace) / max(cfg.rho_setpoint, 1e-6)
    high_excess = np.maximum(0.0, rho_trace - cfg.rho_setpoint) / max(cfg.rho_setpoint, 1e-6)
    setpoint_cost = float(np.mean(cfg.reserve_low_weight * low_deficit ** 2 + cfg.reserve_high_weight * high_excess ** 2))
    
    oscillation_cost = float(np.var(rho_trace))
    
    return starvation_cost + cfg.reserve_setpoint_weight * setpoint_cost + cfg.reserve_oscillation_weight * oscillation_cost


def _biomass_envelope_cost(biomass_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    J_biomass_envelope: Weak penalty for operating in high-density regions.
    Provides a soft operating envelope without a hard cliff.
    """
    safe_threshold = cfg.b_bloom_threshold * 0.55
    overage = np.maximum(0.0, biomass_trace - safe_threshold)
    return float(np.mean((overage / 40.0) ** 2))


def _biomass_rate_cost(biomass_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    J_biomass_rate: Medium penalty for accelerating growth.
    Penalizes positive velocity and positive acceleration indicating impending bloom.
    Normalized by the biological maximum growth rate.
    """
    if len(biomass_trace) < 3:
        return 0.0
    vel = np.diff(biomass_trace, prepend=biomass_trace[0])
    accel = np.diff(vel, prepend=vel[0])
    
    # Max theoretical velocity is approx b_ref * 0.012 (max_growth_rate)
    max_vel = max(cfg.b_ref * 0.012, 1e-6)
    norm_vel = np.maximum(0.0, vel) / max_vel
    
    # Max acceleration is a fraction of max velocity
    norm_accel = np.maximum(0.0, accel) / (max_vel * 0.1)
    
    return float(np.mean(norm_vel**2 + norm_accel**2))


def _maintenance_cost(maint_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    J_maintenance: Strong penalty for high maintenance burden (precursor to detritus).
    Normalized by the approximate maintenance cost of a healthy B_ref.
    Nonlinear to severely penalize extremely high maintenance levels.
    """
    baseline_maint = cfg.b_ref * cfg.maintenance_rate
    if baseline_maint < 1e-6:
        return 0.0
    norm_maint = np.maximum(0.0, maint_trace) / baseline_maint
    return float(np.mean(norm_maint ** 1.5))


def _detritus_cost(detritus_trace: np.ndarray) -> float:
    """
    J_detritus: Penalty for future dissolved nutrients currently in the dead pool.
    Penalizes both inventory and accumulation of detritus.
    Normalized so a 10g dead pool or 0.05g/min growth yields 1.0 cost.
    """
    if len(detritus_trace) < 2:
        return 0.0
    mean_val = float(np.mean(detritus_trace))
    delta = np.diff(detritus_trace, prepend=detritus_trace[0])
    slope_val = float(np.mean(np.maximum(0.0, delta)))
    
    norm_mean = mean_val / 10.0
    norm_slope = slope_val / 0.05
    return norm_mean + norm_slope


def _dissolved_inventory_cost(dissolved_trace: np.ndarray) -> float:
    """
    J_dissolved_inventory: Sustained tank loading penalty.
    Normalized against osmotic_half_effect (5.0).
    """
    return float(np.mean(dissolved_trace)) / 5.0


def _dissolved_accumulation_cost(dissolved_trace: np.ndarray) -> float:
    """
    J_dissolved_accumulation: Very strong penalty for dissolved nutrient accumulation.
    Normalized against max physical entry rate (approx 0.1g/min).
    """
    if len(dissolved_trace) < 2:
        return 0.0
    delta = np.diff(dissolved_trace, prepend=dissolved_trace[0])
    return float(np.mean(np.maximum(0.0, delta))) / 0.10


def _osmotic_cost(osmotic_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    J_osmotic: Dominant penalty for direct osmotic stress (causes immediate uptake shutdown).
    """
    return float(np.mean(np.maximum(0.0, osmotic_trace - cfg.osmotic_safe_threshold) ** 2))


def _nutrient_cost(
    dose_trace: np.ndarray,
    cfg: LabelingObjectiveConfig,
) -> float:
    """
    J_nutrient = mean(dose) + λ_quad × (mean_dose)^2

    Pure dosing cost — decoupled from reserve state.
    J_reserve separately determines when dosing is worthwhile.
    Quadratic term super-linearly penalizes sustained high-rate dosing.
    """
    mean_dose = float(np.mean(dose_trace))
    linear = mean_dose
    quadratic = cfg.nutrient_quadratic_weight * (mean_dose ** 2)
    return linear + quadratic


def _action_cost(flowrate: float, duration: float, state: TankState) -> float:
    """
    J_action = (Δflowrate_norm)^2 + (Δduration_norm)^2

    Rate-of-change penalty for actuator wear and control chattering.
    Normalized to [0, 1] range to avoid overshadowing biological costs.
    """
    d_fr = abs(flowrate - state.prev_flowrate) / 5.0    # max flowrate
    d_dur = abs(duration - state.prev_duration) / 30.0  # max duration
    spike = max(0.0, flowrate * duration / 60.0 - 1.5) ** 2 * 0.1
    return d_fr ** 2 + d_dur ** 2 + spike


def _terminal_sustainability_cost(
    final_s: TankState,
    rho_trace: np.ndarray,
    health_trace: np.ndarray,
    biomass_trace: np.ndarray,
    dissolved_trace: np.ndarray,
    maint_trace: np.ndarray,
    osmotic_trace: np.ndarray,
    cfg: LabelingObjectiveConfig,
) -> float:
    """
    V(s_T) — extrapolates impending collapse at horizon end using terminal slopes.
    """
    rho_T = compute_rho(final_s, cfg.internal_capacity)
    
    # Calculate trends over the last 10% of the trace
    tail_n = max(2, int(len(rho_trace) * 0.1))
    
    if len(rho_trace) >= tail_n:
        rho_slope = float(rho_trace[-1] - rho_trace[-tail_n])
        biomass_slope = float(biomass_trace[-1] - biomass_trace[-tail_n])
        dissolved_slope = float(dissolved_trace[-1] - dissolved_trace[-tail_n])
        maint_tail = np.mean(maint_trace[-tail_n:])
    else:
        rho_slope = biomass_slope = dissolved_slope = 0.0
        maint_tail = maint_trace[-1] if len(maint_trace) > 0 else 0.0

    # Base state risks
    norm_deficit = max(0.0, cfg.reserve_floor - rho_T) / max(cfg.reserve_floor, 1e-6)
    starvation_risk = norm_deficit ** 2 * 5.0  # Terminal starvation is very risky
    
    # Trajectory slopes indicating impending collapse
    decline_risk = 2.0 * max(0.0, -rho_slope)
    bloom_momentum = 5.0 * max(0.0, biomass_slope / max(cfg.b_ref, 1.0))
    dissolved_momentum = 5.0 * max(0.0, dissolved_slope)
    
    # Terminal mechanistic triggers
    maint_risk = (maint_tail / max(cfg.b_ref * cfg.maintenance_rate, 1e-6)) * 2.0
    
    osmotic_T = osmotic_trace[-1] if len(osmotic_trace) > 0 else 0.0
    osmotic_risk = max(0.0, osmotic_T - cfg.osmotic_safe_threshold) ** 2 * 5.0

    return starvation_risk + decline_risk + bloom_momentum + dissolved_momentum + maint_risk + osmotic_risk


def _hard_safety(final_s: TankState) -> float:
    """Hard penalty for degenerate terminal states (near-zero biomass or NaN)."""
    if final_s.algae_biomass < 1.0:
        return 50.0   # complete biomass collapse
    if final_s.health_index < 0.05:
        return 30.0   # near-certain irreversible death
    return 0.0


# ---------------------------------------------------------------------------
# Main rollout evaluator
# ---------------------------------------------------------------------------

def evaluate_rollout(
    eco_trace: List[Dict[str, float]],
    final_state: TankState,
    initial_state: TankState,
    flowrate: float,
    duration: float,
    cfg: LabelingObjectiveConfig,
) -> CostBreakdown:
    """
    Compute full ecosystem cost breakdown for one candidate action rollout.

    eco_trace: list of per-step dicts with keys:
        ec, rho, health, damage, biomass, dose
    """
    dbg = cfg.debug_mode
    H = len(eco_trace)

    if H == 0:
        bd = CostBreakdown(safety=_INVALID_COST, total=_INVALID_TOTAL)
        return bd

    health_arr  = np.array([s["health"]  for s in eco_trace], dtype=np.float64)
    damage_arr  = np.array([s["damage"]  for s in eco_trace], dtype=np.float64)
    rho_arr     = np.array([s["rho"]     for s in eco_trace], dtype=np.float64)
    biomass_arr = np.array([s["biomass"] for s in eco_trace], dtype=np.float64)
    dose_arr    = np.array([s["dose"]    for s in eco_trace], dtype=np.float64)
    diss_arr    = np.array([s["dissolved"] for s in eco_trace], dtype=np.float64)
    maint_arr   = np.array([s["maint_cost"] for s in eco_trace], dtype=np.float64)
    osmotic_arr = np.array([s["osmotic"] for s in eco_trace], dtype=np.float64)
    detritus_arr = np.array([s.get("detritus", 0.0) for s in eco_trace], dtype=np.float64)

    j_health  = _health_cost(health_arr, float(final_state.health_index), cfg)
    j_damage  = _damage_cost(damage_arr, cfg)
    j_reserve = _reserve_cost(rho_arr, cfg)
    j_biomass = _biomass_rate_cost(biomass_arr, cfg)
    j_env     = _biomass_envelope_cost(biomass_arr, cfg)
    j_maint   = _maintenance_cost(maint_arr, cfg)
    j_det     = _detritus_cost(detritus_arr)
    j_diss_inv = _dissolved_inventory_cost(diss_arr)
    j_diss_acc = _dissolved_accumulation_cost(diss_arr)
    j_osmotic = _osmotic_cost(osmotic_arr, cfg)
    j_nutrient = _nutrient_cost(dose_arr, cfg)
    j_action   = _action_cost(flowrate, duration, initial_state)
    j_terminal = _terminal_sustainability_cost(final_state, rho_arr, health_arr, biomass_arr, diss_arr, maint_arr, osmotic_arr, cfg)
    j_safety   = _hard_safety(final_state)

    terms = {
        "health": j_health, "damage": j_damage, "reserve": j_reserve,
        "biomass_rate": j_biomass, "biomass_envelope": j_env,
        "maintenance": j_maint, "detritus": j_det,
        "dissolved_inventory": j_diss_inv, "dissolved_accumulation": j_diss_acc,
        "osmotic": j_osmotic, "nutrient": j_nutrient, "action": j_action,
        "terminal": j_terminal, "safety": j_safety,
    }
    for name, val in terms.items():
        terms[name] = _ensure_finite(val, name, debug=dbg)

    if dbg:
        print(
            f"  [COST] flow={flowrate:.2f} dur={duration:.1f} "
            + " ".join(f"{k}={terms[k]:.4f}" for k in terms)
        )

    bd = CostBreakdown(
        health=terms["health"],
        damage=terms["damage"],
        reserve=terms["reserve"],
        biomass_rate=terms["biomass_rate"],
        biomass_envelope=terms["biomass_envelope"],
        maintenance=terms["maintenance"],
        detritus=terms["detritus"],
        dissolved_inventory=terms["dissolved_inventory"],
        dissolved_accumulation=terms["dissolved_accumulation"],
        osmotic=terms["osmotic"],
        nutrient=terms["nutrient"],
        action=terms["action"],
        terminal=terms["terminal"],
        safety=terms["safety"],
        # Diagnostics
        reserve_mean=float(np.mean(rho_arr)),
        health_mean=float(np.mean(health_arr)),
        damage_mean=float(np.mean(damage_arr)),
        biomass_mean=float(np.mean(biomass_arr)),
    )
    bd.total = bd.weighted_total(cfg)
    if not np.isfinite(bd.total):
        bd.total = _INVALID_TOTAL
    return bd
