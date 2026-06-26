"""
Nonlinear algae-tank dynamics (v3) — active equilibrium with smooth EC, strong
thermal coupling, and semi-independent turbidity evolution.

Design goals:
  - EC: soft saturation, assimilation lag, damped second-order recovery (no vertical jumps)
  - Temperature: Q10 + Gaussian optimal-zone efficiency shapes all rates
  - Turbidity: logistic biomass with long memory, lagged growth, partial EC decoupling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np

DisturbanceInput = Union[float, "DisturbanceSpec"]


@dataclass
class DisturbanceSpec:
    """Multi-channel disturbances propagated through the tank."""

    ec_shock: float = 0.0
    temp_shock: float = 0.0
    mixing_noise: float = 0.0
    uptake_multiplier: float = 1.0
    actuator_efficiency: float = 1.0
    turbidity_shock: float = 0.0  # sediment / scattering — does not change EC


def _parse_disturbance(d: DisturbanceInput) -> DisturbanceSpec:
    if isinstance(d, DisturbanceSpec):
        return d
    return DisturbanceSpec(ec_shock=float(d))


@dataclass
class TankDynamicsParams:
    """YAML-driven parameters for the v3 dynamical model."""

    ec_target: float = 1.2
    ec_healthy_min: float = 0.75
    ec_instability_threshold: float = 1.55
    ec_soft_limit: float = 2.2
    ec_hard_limit: float = 3.0

    baseline_ec_depletion: float = 0.0038
    biological_uptake_rate: float = 0.0035
    starvation_acceleration: float = 0.006
    nutrient_to_ec_gain: float = 0.095
    immediate_absorption_fraction: float = 0.28
    delay_kernel: Tuple[float, ...] = (0.15, 0.25, 0.30, 0.20, 0.10)

    # EC smoothness & damping
    ec_assimilation_tau: float = 0.18
    max_queue_release_rate: float = 0.048
    release_damping: float = 0.85
    ec_spring_stiffness: float = 0.015
    ec_damping: float = 0.22
    memory_to_ec_gain: float = 0.04

    # Temperature
    temp_ref: float = 25.0
    temp_opt: float = 26.0
    temp_sigma: float = 5.0
    q10_coefficient: float = 2.2
    temp_coupling_strength: float = 2.0
    ambient_temp_mean: float = 22.0
    temp_relaxation_rate: float = 0.025
    turbidity_temp_exponent: float = 1.35

    # Turbidity / biomass (partially decoupled from EC)
    turbidity_carrying_capacity: float = 220.0
    turbidity_growth_rate: float = 0.016
    turbidity_mortality_rate: float = 0.0045
    biomass_memory_alpha: float = 0.025
    turbidity_growth_lag_tau: float = 0.06
    turbidity_ec_coupling: float = 0.25
    turbidity_stochastic_std: float = 0.4
    algae_sensor_tau: float = 0.045
    bloom_turbidity_noise: float = 1.2

    # Biological inertia
    health_lag_tau: float = 0.035
    nutrient_memory_alpha: float = 0.06

    # Failure modes (mild — avoid hard jumps)
    nutrient_overshoot_penalty: float = 0.04
    oscillation_sensitivity: float = 0.05
    cumulative_nutrient_limit: float = 50.0
    overshoot_ec_gain: float = 0.02

    drift_rate: float = 1.0e-5
    ec_decay_jitter: float = 0.0
    gain_jitter: float = 0.0

    @property
    def delay_steps(self) -> int:
        return len(self.delay_kernel)

    @classmethod
    def from_config(cls, dyn: Dict[str, Any], ec_target: float = 1.2) -> "TankDynamicsParams":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {"ec_target": ec_target}
        for key in fields:
            if key == "ec_target" or key == "delay_kernel":
                continue
            if key in dyn:
                kwargs[key] = dyn[key]
        if "delay_kernel" in dyn:
            kwargs["delay_kernel"] = tuple(dyn["delay_kernel"])
        return cls(**kwargs)

    @classmethod
    def sample_random(cls, base: "TankDynamicsParams", rng: np.random.Generator) -> "TankDynamicsParams":
        d = base.__dict__.copy()
        d["baseline_ec_depletion"] = base.baseline_ec_depletion * rng.uniform(0.85, 1.15)
        d["biological_uptake_rate"] = base.biological_uptake_rate * rng.uniform(0.85, 1.15)
        d["nutrient_to_ec_gain"] = base.nutrient_to_ec_gain * rng.uniform(0.9, 1.1)
        d["ambient_temp_mean"] = base.ambient_temp_mean + rng.normal(0, 1.0)
        d["turbidity_growth_rate"] = base.turbidity_growth_rate * rng.uniform(0.75, 1.25)
        d["temp_coupling_strength"] = base.temp_coupling_strength * rng.uniform(0.9, 1.15)
        d["ec_decay_jitter"] = rng.normal(0, 0.0003)
        d["gain_jitter"] = rng.normal(0, 0.004)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TankState:
    """Environment + hidden dynamical state."""

    water_temp: float
    ec: float
    turbidity: float
    prev_flowrate: float
    prev_duration: float
    time_since_last_dose: float
    ph: float = 7.2
    dissolved_oxygen: float = 8.0
    ambient_temp: float = 22.0
    cumulative_nutrients: float = 0.0
    step_index: int = 0
    absorption_queue: np.ndarray = field(default_factory=lambda: np.zeros(5))
    algae_biomass: float = 80.0
    nutrient_memory: float = 0.0
    biomass_memory: float = 0.5
    biomass_growth_drive: float = 0.5
    health_index: float = 1.0
    ec_velocity: float = 0.0
    assimilation_pool: float = 0.0
    oscillation_phase: float = 0.0

    def as_dict(self) -> dict:
        return {
            "water_temp": self.water_temp,
            "ec": self.ec,
            "turbidity": self.turbidity,
            "prev_flowrate": self.prev_flowrate,
            "prev_duration": self.prev_duration,
            "time_since_last_dose": self.time_since_last_dose,
            "ph": self.ph,
            "dissolved_oxygen": self.dissolved_oxygen,
            "ambient_temp": self.ambient_temp,
            "cumulative_nutrients": self.cumulative_nutrients,
            "step_index": self.step_index,
            "algae_biomass": self.algae_biomass,
            "nutrient_memory": self.nutrient_memory,
            "biomass_memory": self.biomass_memory,
            "health_index": self.health_index,
            "pending_absorption": float(np.sum(self.absorption_queue)),
            "ec_velocity": self.ec_velocity,
        }

    @classmethod
    def create_initial(
        cls,
        params: TankDynamicsParams,
        ec: float | None = None,
        turbidity: float | None = None,
        water_temp: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> "TankState":
        rng = rng or np.random.default_rng()
        ec0 = ec if ec is not None else params.ec_target + rng.normal(0, 0.06)
        turb0 = turbidity if turbidity is not None else 90.0 + rng.uniform(0, 35)
        wt = water_temp if water_temp is not None else params.ambient_temp_mean + rng.normal(0, 0.5)
        return cls(
            water_temp=float(wt),
            ec=float(ec0),
            turbidity=float(turb0),
            prev_flowrate=0.0,
            prev_duration=0.0,
            time_since_last_dose=999.0,
            ph=7.2 + rng.normal(0, 0.05),
            dissolved_oxygen=8.0 + rng.normal(0, 0.2),
            ambient_temp=params.ambient_temp_mean,
            cumulative_nutrients=0.0,
            step_index=0,
            absorption_queue=np.zeros(params.delay_steps, dtype=np.float64),
            algae_biomass=float(turb0),
            nutrient_memory=0.0,
            biomass_memory=0.5,
            biomass_growth_drive=0.5,
            health_index=1.0,
            ec_velocity=0.0,
            assimilation_pool=0.0,
            oscillation_phase=0.0,
        )


def q10_metabolism_factor(temp: float, params: TankDynamicsParams) -> float:
    """Q10 scaling with configurable strength."""
    delta = (temp - params.temp_ref) / 10.0
    raw = params.q10_coefficient ** delta
    centered = 1.0 + params.temp_coupling_strength * (raw - 1.0)
    return float(np.clip(centered, 0.25, 3.5))


def thermal_efficiency(temp: float, params: TankDynamicsParams) -> float:
    """
    Bell-shaped thermal efficiency (optimal growth zone).

    Combines Arrhenius-like Q10 with Gaussian penalty away from temp_opt.
    """
    z = (temp - params.temp_opt) / max(params.temp_sigma, 0.5)
    gaussian = float(np.exp(-0.5 * z * z))
    q10 = q10_metabolism_factor(temp, params)
    return gaussian * q10


def soft_ec_saturation_factor(ec: float, params: TankDynamicsParams) -> float:
    """
    Diminishing nutrient→EC effectiveness as EC approaches soft limit.

    Uses smooth rational saturation: (1 - (ec/soft)^2)_+ with floor.
    """
    ratio = max(0.0, ec) / params.ec_soft_limit
    return np.clip(1.0 - ratio**2, 0.15, 1.0)


def _soft_clip_ec(ec, params):
    return ec


def _inject_delayed_dose(
    queue: np.ndarray,
    dose_mass: float,
    kernel: Tuple[float, ...],
    immediate_frac: float,
) -> Tuple[np.ndarray, float]:
    immediate = immediate_frac * dose_mass
    delayed_mass = (1.0 - immediate_frac) * dose_mass
    k = np.asarray(kernel, dtype=np.float64)
    k = k / (k.sum() + 1e-12)
    queue = queue.copy()
    n = min(len(queue), len(k))
    for i in range(n):
        queue[i] += delayed_mass * k[i]
    return queue, immediate


def _release_absorption(
    queue: np.ndarray,
    params: TankDynamicsParams,
    dt_scale: float,
) -> Tuple[np.ndarray, float]:
    """FIFO release with per-step cap and exponential queue damping."""
    queue = queue * params.release_damping
    raw_release = float(queue[0]) if len(queue) else 0.0
    cap = params.max_queue_release_rate * max(dt_scale, 0.01)
    released = min(raw_release, cap)
    if len(queue) > 0:
        queue[0] = max(0.0, queue[0] - released)
    if len(queue) > 1:
        queue = np.roll(queue, -1)
        queue[-1] = 0.0
    return queue, released


def step_dynamics(
    state: TankState,
    flowrate: float,
    duration: float,
    dt: float,
    params: TankDynamicsParams,
    disturbance: DisturbanceInput = 0.0,
    rng: np.random.Generator | None = None,
) -> TankState:
    """
    Advance one timestep with smooth EC, thermal metabolism, and decoupled turbidity.
    """
    dist = _parse_disturbance(disturbance)
    rng = rng or np.random.default_rng()
    dt_scale = dt / 60.0

    dose_mass = flowrate * duration / 60.0 * dist.actuator_efficiency
    cumulative = state.cumulative_nutrients + dose_mass

    nutrient_memory = (
        (1.0 - params.nutrient_memory_alpha) * state.nutrient_memory
        + params.nutrient_memory_alpha * dose_mass
    )

    queue = np.asarray(state.absorption_queue, dtype=np.float64)
    if len(queue) != params.delay_steps:
        queue = np.zeros(params.delay_steps, dtype=np.float64)
    queue, immediate_mass = _inject_delayed_dose(
        queue, dose_mass, params.delay_kernel, params.immediate_absorption_fraction
    )
    queue, released_mass = _release_absorption(queue, params, dt_scale)

    thermal = thermal_efficiency(state.water_temp, params)
    temp_factor = q10_metabolism_factor(state.water_temp, params)

    # --- EC depletion (stronger at high temperature) ---
    baseline_drain = (params.baseline_ec_depletion + params.ec_decay_jitter) * thermal
    proportional_uptake = (
    params.biological_uptake_rate
    * np.clip(state.ec, 0.0, None)
    * thermal
    * dist.uptake_multiplier
    )
    starvation = 0.0
    if state.ec < params.ec_healthy_min:
        deficit = params.ec_healthy_min - state.ec
        starvation = params.starvation_acceleration * deficit * thermal

    depletion = (baseline_drain + proportional_uptake + starvation) * dt_scale

    # --- Nutrient influx with soft saturation (prevents runaway jumps) ---
    sat = soft_ec_saturation_factor(state.ec, params)
    gain = (params.nutrient_to_ec_gain + params.gain_jitter) * thermal * sat
    raw_influx = gain * (immediate_mass + released_mass)
    memory_influx = params.memory_to_ec_gain * nutrient_memory * thermal * sat * dt_scale
    assimilation_pool = (
        (1.0 - params.ec_assimilation_tau) * state.assimilation_pool
        + params.ec_assimilation_tau * (raw_influx + memory_influx)
    )

    # --- Second-order: influx drives velocity; damping only (no passive spring to target) ---
    # Optional weak restoring only when well-fed (avoids artificial collapse prevention)
    restoring = 0.0
    if state.health_index > 0.85 and state.ec < params.ec_target:
        restoring = params.ec_spring_stiffness * (params.ec_target - state.ec)
    ec_velocity = state.ec_velocity + dt_scale * (
        assimilation_pool - depletion + restoring - params.ec_damping * state.ec_velocity
    )

    ec_new = state.ec + dt_scale * ec_velocity + dist.ec_shock * 0.5
    ec_new += dist.mixing_noise * 0.005

    # Mild overshoot oscillation (damped, not additive runaway)
    phase = state.oscillation_phase + 0.25 * dt_scale
    if state.ec > params.ec_instability_threshold:
        excess = state.ec - params.ec_instability_threshold
        ec_new += (
            params.overshoot_ec_gain * excess * dt_scale
            + params.oscillation_sensitivity * excess * np.sin(phase) * dt_scale
        )

    if cumulative > params.cumulative_nutrient_limit:
        pass

    ec_new = _soft_clip_ec(ec_new, params)

    ec_new = max(0.0, ec_new)

    # --- Health & biomass memory (slow; decouples turbidity from instant EC) ---
    ec_ratio = float(np.clip(state.ec / params.ec_target, 0.0, 1.5))
    target_health = float(np.clip(ec_ratio, 0.0, 1.0))
    health_index = (
        (1.0 - params.health_lag_tau) * state.health_index
        + params.health_lag_tau * target_health
    )

    nutrient_signal = float(np.clip(nutrient_memory / 5.0, 0.0, 1.5))
    biomass_memory = (
        (1.0 - params.biomass_memory_alpha) * state.biomass_memory
        + params.biomass_memory_alpha * (0.6 * health_index + 0.4 * nutrient_signal)
    )

    # Growth drive lags EC: uses memory + health, only partial instant EC
    instant_ec_drive = params.turbidity_ec_coupling * max(
        0.0, (state.ec - params.ec_healthy_min * 0.5) / params.ec_target
    )
    target_drive = float(np.clip(0.55 * biomass_memory + 0.35 * health_index + instant_ec_drive, 0.0, 1.2))
    biomass_growth_drive = (
        (1.0 - params.turbidity_growth_lag_tau) * state.biomass_growth_drive
        + params.turbidity_growth_lag_tau * target_drive
    )

    # --- Algae biomass (logistic, temperature-sensitive, semi-independent) ---
    B = state.algae_biomass
    K = params.turbidity_carrying_capacity
    temp_growth = thermal ** params.turbidity_temp_exponent
    r = params.turbidity_growth_rate * biomass_growth_drive * temp_growth
    mortality = params.turbidity_mortality_rate * (1.2 - 0.5 * health_index) * (2.0 - temp_growth * 0.5)

    stochastic = rng.normal(0, params.turbidity_stochastic_std) * np.sqrt(dt_scale)
    dB = (r * B * (1.0 - B / K) - mortality * B) * dt_scale + stochastic
    if state.ec > params.ec_instability_threshold:
        dB += params.bloom_turbidity_noise * np.sin(phase * 1.2) * dt_scale

    algae_biomass = float(np.clip(B + dB, 5.0, K * 1.05))

    # Turbidity: lagged biomass + immediate fraction of sediment shock (EC unchanged)
    immediate_turb = 0.55 * dist.turbidity_shock
    turbidity_target = algae_biomass + 0.45 * dist.turbidity_shock
    turbidity_new = (
        (1.0 - params.algae_sensor_tau) * state.turbidity
        + params.algae_sensor_tau * turbidity_target
        + immediate_turb
    )
    turbidity_new = float(np.clip(turbidity_new, 0.0, K * 1.2))

    # --- Temperature (ambient pull + shocks) ---
    ambient_pull = params.temp_relaxation_rate * (
        params.ambient_temp_mean - state.water_temp
    )
    water_temp_new = state.water_temp + ambient_pull * dt_scale + dist.temp_shock

    ph_new = state.ph - 0.0025 * dose_mass + 0.0001 * (7.0 - state.ph)
    do_new = (
        state.dissolved_oxygen
        - 0.01 * dose_mass * temp_factor
        + 0.0003 * (8.5 - state.dissolved_oxygen)
    )

    time_since = 0.0 if dose_mass > 1e-6 else state.time_since_last_dose + dt

    return TankState(
        water_temp=float(water_temp_new),
        ec=ec_new,
        turbidity=turbidity_new,
        prev_flowrate=flowrate,
        prev_duration=duration,
        time_since_last_dose=time_since,
        ph=float(np.clip(ph_new, 5.5, 9.0)),
        dissolved_oxygen=float(np.clip(do_new, 2.0, 12.0)),
        ambient_temp=state.ambient_temp,
        cumulative_nutrients=cumulative,
        step_index=state.step_index + 1,
        absorption_queue=queue,
        algae_biomass=algae_biomass,
        nutrient_memory=nutrient_memory,
        biomass_memory=biomass_memory,
        biomass_growth_drive=biomass_growth_drive,
        health_index=health_index,
        ec_velocity=ec_velocity,
        assimilation_pool=assimilation_pool,
        oscillation_phase=float(phase),
    )


def rollout_horizon(
    state: TankState,
    flowrate: float,
    duration: float,
    horizon: int,
    dt: float,
    params: TankDynamicsParams,
) -> Tuple[np.ndarray, TankState]:
    ec_trace = []
    s = state
    for k in range(horizon):
        fr = flowrate if k == 0 else 0.0
        dur = duration if k == 0 else 0.0
        s = step_dynamics(s, fr, dur, dt, params)
        ec_trace.append(s.ec)
    return np.array(ec_trace, dtype=np.float64), s


def simulate_open_loop(
    params: TankDynamicsParams,
    length: int,
    dt: float,
    actions: List[Tuple[float, float]] | None = None,
    initial_state: TankState | None = None,
    disturbances: List[DisturbanceInput] | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    s = initial_state or TankState.create_initial(params, rng=rng)
    actions = actions or [(0.0, 0.0)] * length
    disturbances = disturbances or [0.0] * length
    rng = rng or np.random.default_rng()

    hist: Dict[str, List[float]] = {
        "ec": [],
        "turbidity": [],
        "water_temp": [],
        "pending_absorption": [],
        "health_index": [],
        "algae_biomass": [],
        "biomass_memory": [],
    }
    for t in range(length):
        fr, dur = actions[t] if t < len(actions) else (0.0, 0.0)
        d = disturbances[t] if t < len(disturbances) else 0.0
        hist["ec"].append(s.ec)
        hist["turbidity"].append(s.turbidity)
        hist["water_temp"].append(s.water_temp)
        hist["pending_absorption"].append(float(np.sum(s.absorption_queue)))
        hist["health_index"].append(s.health_index)
        hist["algae_biomass"].append(s.algae_biomass)
        hist["biomass_memory"].append(s.biomass_memory)
        s = step_dynamics(s, fr, dur, dt, params, disturbance=d, rng=rng)

    return {k: np.array(v) for k, v in hist.items()}


def max_ec_step_change(ec_trace: np.ndarray) -> float:
    """Diagnostic: largest single-step |ΔEC| (should stay modest after v3)."""
    if len(ec_trace) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(ec_trace))))
