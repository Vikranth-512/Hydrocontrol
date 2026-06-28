"""
Nonlinear algae-tank dynamics (v4) — Mechanistic biological process model.

Design goals:
  - True physical process model (open-loop).
  - Explicit mass conservation of nutrients.
  - Biomass driven by internal nutrient reserves and Monod kinetics.
  - Sensor models strictly decoupled from internal plant state.
  - No hidden control equilibria or restoring springs.
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
    turbidity_shock: float = 0.0


def _parse_disturbance(d: DisturbanceInput) -> DisturbanceSpec:
    if isinstance(d, DisturbanceSpec):
        return d
    return DisturbanceSpec(ec_shock=float(d))


@dataclass
class TankDynamicsParams:
    """YAML-driven parameters for the mechanistic bioprocess model."""

    # Sensor Models
    sensor_gain_ec: float = 0.8
    biomass_optical_factor: float = 2.2
    algae_sensor_tau: float = 0.045
    
    # Transport
    immediate_absorption_fraction: float = 0.28
    delay_kernel: Tuple[float, ...] = (0.15, 0.25, 0.30, 0.20, 0.10)
    max_queue_release_rate: float = 0.048
    release_damping: float = 0.85
    background_dilution_rate: float = 0.002

    # Biology (Uptake & Reserve)
    maximum_uptake_rate: float = 0.015
    half_saturation_mass: float = 1.0
    internal_capacity: float = 0.5  # Max reserve per unit of biomass
    
    # Biology (Growth & Maintenance)
    maximum_growth_rate: float = 0.012
    growth_yield: float = 0.6       # Fraction of consumed reserve that becomes biomass
    maintenance_cost: float = 0.001
    biomass_nutrient_content: float = 0.1 # Fraction of biomass that is structural nutrient
    
    # Biology (Mortality & Recycling)
    mortality_rate: float = 0.0005
    mineralization_rate: float = 0.002
    
    # Biology (Health & Damage)
    damage_rate: float = 0.001
    repair_rate: float = 0.0005
    osmotic_half_effect: float = 5.0
    
    # Temperature
    temp_ref: float = 25.0
    temp_opt: float = 26.0
    temp_sigma: float = 5.0
    q10_coefficient: float = 2.2
    temperature_rate_modifier: float = 2.0
    ambient_temp_mean: float = 22.0
    temp_relaxation_rate: float = 0.025
    
    # Jitter
    ec_decay_jitter: float = 0.0
    gain_jitter: float = 0.0

    @property
    def delay_steps(self) -> int:
        return len(self.delay_kernel)

    @classmethod
    def from_config(cls, dyn: Dict[str, Any]) -> "TankDynamicsParams":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for key in fields:
            if key == "delay_kernel":
                continue
            if key in dyn:
                kwargs[key] = dyn[key]
        if "delay_kernel" in dyn:
            kwargs["delay_kernel"] = tuple(dyn["delay_kernel"])
        return cls(**kwargs)

    @classmethod
    def sample_random(cls, base: "TankDynamicsParams", rng: np.random.Generator) -> "TankDynamicsParams":
        d = base.__dict__.copy()
        d["background_dilution_rate"] = base.background_dilution_rate * rng.uniform(0.85, 1.15)
        d["maximum_uptake_rate"] = base.maximum_uptake_rate * rng.uniform(0.85, 1.15)
        d["maximum_growth_rate"] = base.maximum_growth_rate * rng.uniform(0.85, 1.15)
        d["ambient_temp_mean"] = base.ambient_temp_mean + rng.normal(0, 1.0)
        d["temperature_rate_modifier"] = base.temperature_rate_modifier * rng.uniform(0.9, 1.15)
        d["ec_decay_jitter"] = rng.normal(0, 0.0003)
        d["gain_jitter"] = rng.normal(0, 0.004)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TankState:
    """Environment + hidden dynamical state."""
    
    water_temp: float
    ph: float = 7.2
    dissolved_oxygen: float = 8.0
    ambient_temp: float = 22.0
    prev_flowrate: float = 0.0
    prev_duration: float = 0.0
    time_since_last_dose: float = 999.0
    cumulative_nutrients: float = 0.0
    cumulative_dilution: float = 0.0
    step_index: int = 0
    
    # Hidden physical state
    nutrient_queue: np.ndarray = field(default_factory=lambda: np.zeros(5))
    dissolved_nutrient_mass: float = 1.5
    algae_biomass: float = 80.0
    internal_reserve: float = 10.0
    dead_biomass_pool: float = 0.0
    health_index: float = 1.0
    damage_index: float = 0.0
    
    # Cached sensor observations (readonly mapping from physical state)
    ec: float = 1.2
    turbidity: float = 176.0

    def as_dict(self) -> dict:
        return {
            "water_temp": self.water_temp,
            "ph": self.ph,
            "dissolved_oxygen": self.dissolved_oxygen,
            "ambient_temp": self.ambient_temp,
            "prev_flowrate": self.prev_flowrate,
            "prev_duration": self.prev_duration,
            "time_since_last_dose": self.time_since_last_dose,
            "cumulative_nutrients": self.cumulative_nutrients,
            "cumulative_dilution": self.cumulative_dilution,
            "step_index": self.step_index,
            "dissolved_nutrient_mass": self.dissolved_nutrient_mass,
            "algae_biomass": self.algae_biomass,
            "internal_reserve": self.internal_reserve,
            "dead_biomass_pool": self.dead_biomass_pool,
            "health_index": self.health_index,
            "damage_index": self.damage_index,
            "ec": self.ec,
            "turbidity": self.turbidity,
            "pending_nutrients": float(np.sum(self.nutrient_queue)),
        }

    def compute_total_mass(self, params: TankDynamicsParams) -> float:
        """Helper to compute total nutrient mass in the system for conservation checking."""
        return (
            self.dissolved_nutrient_mass
            + self.internal_reserve
            + self.algae_biomass * params.biomass_nutrient_content
            + self.dead_biomass_pool
            + float(np.sum(self.nutrient_queue))
            + self.cumulative_dilution
        )

    @classmethod
    def create_initial(
        cls,
        params: TankDynamicsParams,
        dissolved_mass: float | None = None,
        biomass: float | None = None,
        water_temp: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> "TankState":
        rng = rng or np.random.default_rng()
        dm0 = dissolved_mass if dissolved_mass is not None else 1.5 + rng.normal(0, 0.06)
        b0 = biomass if biomass is not None else 80.0 + rng.uniform(0, 10)
        wt = water_temp if water_temp is not None else params.ambient_temp_mean + rng.normal(0, 0.5)
        
        s = cls(
            water_temp=float(wt),
            ph=7.2 + rng.normal(0, 0.05),
            dissolved_oxygen=8.0 + rng.normal(0, 0.2),
            ambient_temp=params.ambient_temp_mean,
            nutrient_queue=np.zeros(params.delay_steps, dtype=np.float64),
            dissolved_nutrient_mass=float(dm0),
            algae_biomass=float(b0),
            internal_reserve=float(b0 * params.internal_capacity * 0.5), # start half full
            dead_biomass_pool=0.0,
            health_index=1.0,
            damage_index=0.0,
        )
        s.ec = params.sensor_gain_ec * s.dissolved_nutrient_mass
        s.turbidity = params.biomass_optical_factor * s.algae_biomass
        return s


def q10_metabolism_factor(temp: float, params: TankDynamicsParams) -> float:
    delta = (temp - params.temp_ref) / 10.0
    raw = params.q10_coefficient ** delta
    centered = 1.0 + params.temperature_rate_modifier * (raw - 1.0)
    return float(np.clip(centered, 0.25, 3.5))


def thermal_efficiency(temp: float, params: TankDynamicsParams) -> float:
    z = (temp - params.temp_opt) / max(params.temp_sigma, 0.5)
    gaussian = float(np.exp(-0.5 * z * z))
    q10 = q10_metabolism_factor(temp, params)
    return gaussian * q10


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
    # Removed mass-destroying queue damping
    raw_release = float(queue[0]) if len(queue) else 0.0
    cap = params.max_queue_release_rate * max(dt_scale, 0.01)
    released = min(raw_release, cap)
    if len(queue) > 0:
        leftover = max(0.0, queue[0] - released)
    else:
        leftover = 0.0
        
    if len(queue) > 1:
        # Shift the queue forward
        queue[:-1] = queue[1:]
        queue[-1] = 0.0
        # Add leftover back to the new front so it isn't lost
        queue[0] += leftover
    elif len(queue) == 1:
        queue[0] = leftover

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
    Advance one timestep using strict mechanistic progression and mass conservation.
    """
    dist = _parse_disturbance(disturbance)
    rng = rng or np.random.default_rng()
    dt_scale = dt / 60.0

    # 1. Pump Action & Transport Delay
    dose_mass = flowrate * duration / 60.0 * dist.actuator_efficiency
    cumulative = state.cumulative_nutrients + dose_mass
    
    queue = np.asarray(state.nutrient_queue, dtype=np.float64)
    if len(queue) != params.delay_steps:
        queue = np.zeros(params.delay_steps, dtype=np.float64)
    queue, immediate_mass = _inject_delayed_dose(
        queue, dose_mass, params.delay_kernel, params.immediate_absorption_fraction
    )
    queue, released_mass = _release_absorption(queue, params, dt_scale)

    # Global Modifiers
    thermal_growth = thermal_efficiency(state.water_temp, params)
    thermal_respiration = q10_metabolism_factor(state.water_temp, params)
    
    osmotic_stress = (state.dissolved_nutrient_mass / params.osmotic_half_effect)**2
    osmotic_factor = 1.0 / (1.0 + osmotic_stress)

    # 2. Mixing & Dissolved Nutrient Pool
    # First-order exponential decay updates for dilution and mineralization
    dil_rate = (params.background_dilution_rate + params.ec_decay_jitter) * thermal_respiration
    dilution = state.dissolved_nutrient_mass * (1.0 - np.exp(-dil_rate * dt_scale))
    
    min_rate = params.mineralization_rate * thermal_respiration
    mineralized = state.dead_biomass_pool * (1.0 - np.exp(-min_rate * dt_scale))

    dissolved_new = state.dissolved_nutrient_mass + immediate_mass + released_mass + mineralized - dilution

    # 3. Uptake Kinetics (Monod)
    max_reserve = state.algae_biomass * params.internal_capacity
    reserve_deficit = max(0.0, max_reserve - state.internal_reserve)
    reserve_inhibition = reserve_deficit / max(max_reserve, 1e-6)
    monod = dissolved_new / (params.half_saturation_mass + dissolved_new)

    uptake_rate = (
        params.maximum_uptake_rate 
        * thermal_growth 
        * state.health_index 
        * osmotic_factor 
        * reserve_inhibition 
        * monod
        * dist.uptake_multiplier
    )
    uptake_mass = uptake_rate * state.algae_biomass * dt_scale
    uptake_mass = min(uptake_mass, dissolved_new)
    
    dissolved_new -= uptake_mass
    reserve_new = state.internal_reserve + uptake_mass

    # 4. Maintenance (Consumes reserve, waste recycled to dissolved pool)
    maintenance_cost = params.maintenance_cost * state.algae_biomass * thermal_respiration * dt_scale
    actual_maintenance = min(maintenance_cost, reserve_new)
    reserve_new -= actual_maintenance
    maintenance_deficit = maintenance_cost - actual_maintenance
    dissolved_new += actual_maintenance # Nutrients recycled via maintenance waste

    # 5. Biomass Growth
    reserve_ratio = reserve_new / max(state.algae_biomass * params.internal_capacity, 1e-6)
    growth_drive = reserve_ratio * thermal_growth * osmotic_factor
    growth_amount = params.maximum_growth_rate * growth_drive * state.algae_biomass * dt_scale

    nutrient_cost_per_biomass = params.biomass_nutrient_content / params.growth_yield
    required_reserve = growth_amount * nutrient_cost_per_biomass

    if required_reserve > reserve_new:
        growth_amount = reserve_new / nutrient_cost_per_biomass
        required_reserve = reserve_new

    reserve_new -= required_reserve
    biomass_new = state.algae_biomass + growth_amount
    waste = required_reserve - (growth_amount * params.biomass_nutrient_content)
    dissolved_new += waste # Respiration/growth waste recycled

    # 6. Mortality
    # Mortality increases with heat/cold stress, using Arrhenius/Q10 behavior
    thermal_stress = thermal_respiration
    mortality_rate = params.mortality_rate * thermal_stress * (1.0 + osmotic_stress) * (2.0 - state.health_index)
    mortality_amount = biomass_new * (1.0 - np.exp(-mortality_rate * dt_scale))

    biomass_new -= mortality_amount
    dead_nutrient = mortality_amount * params.biomass_nutrient_content
    proportional_reserve = reserve_new * (mortality_amount / max(biomass_new + mortality_amount, 1e-6))
    
    reserve_new -= proportional_reserve
    dead_pool_new = state.dead_biomass_pool - mineralized + dead_nutrient + proportional_reserve

    # 7. Health Evolution
    # Damage uses dimensionless intensive ratio (recalibrated weights for timescale)
    deficit_ratio = maintenance_deficit / max(maintenance_cost, 1e-6)
    damage_inc = (deficit_ratio * 1.0 + osmotic_stress * 2.0) * params.damage_rate * dt_scale
    
    # Repair is based on maintenance flux being met
    repair_availability = actual_maintenance / max(maintenance_cost, 1e-6)
    repair_inc = params.repair_rate * repair_availability * thermal_respiration * dt_scale

    damage_new = state.damage_index + damage_inc - repair_inc
    damage_new = float(np.clip(damage_new, 0.0, 1.0))
    health_new = 1.0 - damage_new

    # 8. Sensor Output (Strictly read-only)
    ec_new = params.sensor_gain_ec * dissolved_new + dist.mixing_noise + dist.ec_shock
    turbidity_target = params.biomass_optical_factor * biomass_new + dist.turbidity_shock
    turbidity_new = (
        (1.0 - params.algae_sensor_tau) * state.turbidity
        + params.algae_sensor_tau * turbidity_target
    )

    # Env states
    ambient_pull_factor = 1.0 - np.exp(-params.temp_relaxation_rate * dt_scale)
    water_temp_new = state.water_temp + (params.ambient_temp_mean - state.water_temp) * ambient_pull_factor + dist.temp_shock

    ph_new = state.ph - 0.0025 * dose_mass + 0.0001 * (7.0 - state.ph)
    do_new = state.dissolved_oxygen - 0.01 * dose_mass * thermal_growth + 0.0003 * (8.5 - state.dissolved_oxygen)

    time_since = 0.0 if dose_mass > 1e-6 else state.time_since_last_dose + dt

    return TankState(
        water_temp=float(water_temp_new),
        ph=float(np.clip(ph_new, 5.5, 9.0)),
        dissolved_oxygen=float(np.clip(do_new, 2.0, 12.0)),
        ambient_temp=state.ambient_temp,
        prev_flowrate=flowrate,
        prev_duration=duration,
        time_since_last_dose=time_since,
        cumulative_nutrients=cumulative,
        cumulative_dilution=state.cumulative_dilution + dilution,
        step_index=state.step_index + 1,
        
        nutrient_queue=queue,
        dissolved_nutrient_mass=dissolved_new,
        algae_biomass=biomass_new,
        internal_reserve=reserve_new,
        dead_biomass_pool=dead_pool_new,
        health_index=health_new,
        damage_index=damage_new,
        
        ec=ec_new,
        turbidity=turbidity_new,
    )


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
        "dissolved_nutrient_mass": [],
        "algae_biomass": [],
        "internal_reserve": [],
        "dead_biomass_pool": [],
        "health_index": [],
        "ec": [],
        "turbidity": [],
        "water_temp": [],
        "total_mass": [],
    }
    for t in range(length):
        fr, dur = actions[t] if t < len(actions) else (0.0, 0.0)
        d = disturbances[t] if t < len(disturbances) else 0.0
        
        hist["dissolved_nutrient_mass"].append(s.dissolved_nutrient_mass)
        hist["algae_biomass"].append(s.algae_biomass)
        hist["internal_reserve"].append(s.internal_reserve)
        hist["dead_biomass_pool"].append(s.dead_biomass_pool)
        hist["health_index"].append(s.health_index)
        hist["ec"].append(s.ec)
        hist["turbidity"].append(s.turbidity)
        hist["water_temp"].append(s.water_temp)
        hist["total_mass"].append(s.compute_total_mass(params))
        
        s = step_dynamics(s, fr, dur, dt, params, disturbance=d, rng=rng)

    return {k: np.array(v) for k, v in hist.items()}
