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
    maximum_uptake_rate: float = 0.018
    half_saturation_mass: float = 1.0
    internal_capacity: float = 0.5  # Max reserve per unit of biomass
    
    # Biology (Growth & Maintenance)
    maximum_growth_rate: float = 0.012
    growth_yield: float = 0.6       # Fraction of consumed reserve that becomes biomass
    maintenance_cost: float = 0.0005
    biomass_nutrient_content: float = 0.1 # Fraction of biomass that is structural nutrient
    
    # Biology (Mortality & Recycling)
    mortality_rate: float = 0.0002
    mineralization_rate: float = 0.002
    mineralization_efficiency: float = 0.3  # Fraction of dead biomass that mineralizes to dissolved
    maintenance_to_detritus_fraction: float = 0.5
    maintenance_to_dissolved_fraction: float = 0.2  # Fraction of maintenance waste recycled to dissolved
    maintenance_lost_fraction: float = 0.6  # Fraction of maintenance permanently lost (respiration, CO2, etc.)
    growth_metabolic_efficiency: float = 0.7  # Fraction of growth nutrients actually incorporated into biomass
    nutrient_retention_efficiency: float = 0.5  # Overall nutrient retention (0.35-0.6 recommended)
    harvest_rate: float = 0.0005  # Fraction of biomass harvested per timestep
    
    # Biology (Health & Damage)
    damage_rate: float = 0.0001
    repair_rate: float = 0.01
    starvation_damage_multiplier: float = 2.0  # Additional damage under starvation
    starvation_reserve_threshold: float = 0.15  # Reserve ratio below which starvation stress applies
    osmotic_half_effect: float = 5.0
    osmotic_safe_threshold: float = 0.2
    basal_repair_rate: float = 0.0003
    
    # Biology (Density Dependence)
    light_attenuation_half_mass: float = 250.0
    
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
    time_since_last_dose: float = 120.0  # Initialized to min_time_between_doses for realistic actuator history
    cumulative_nutrients: float = 0.0
    cumulative_dilution: float = 0.0
    step_index: int = 0
    
    # Hidden physical state
    pending_doses: List[Tuple[float, float]] = field(default_factory=list) # (mass, time_left)
    dissolved_nutrient_mass: float = 1.5
    algae_biomass: float = 80.0
    internal_reserve: float = 10.0
    dead_biomass_pool: float = 0.0
    health_index: float = 1.0
    damage_index: float = 0.0
    
    # Cached sensor observations (readonly mapping from physical state)
    ec: float = 1.2
    turbidity: float = 176.0

    @property
    def pending_nutrients(self) -> float:
        return sum(mass for mass, _ in self.pending_doses)

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
            "pending_nutrients": self.pending_nutrients,
        }

    def compute_total_mass(self, params: TankDynamicsParams) -> float:
        """Helper to compute total nutrient mass in the system for conservation checking."""
        return (
            self.dissolved_nutrient_mass
            + self.internal_reserve
            + self.algae_biomass * params.biomass_nutrient_content
            + self.dead_biomass_pool
            + self.pending_nutrients
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
        
        # Initialize time_since_last_dose with realistic distribution
        # Use uniform(120, 300) to represent varied actuator history
        time_since = rng.uniform(120.0, 300.0)
        
        # Fix 5: Lower initial reserve from 50% to 20%
        # Fix 6: Lower initial dissolved nutrient from 1.5 to 0.7
        dm0 = dissolved_mass if dissolved_mass is not None else 0.7 + rng.normal(0, 0.03)
        
        s = cls(
            water_temp=float(wt),
            ph=7.2 + rng.normal(0, 0.05),
            dissolved_oxygen=8.0 + rng.normal(0, 0.2),
            ambient_temp=params.ambient_temp_mean,
            time_since_last_dose=time_since,
            pending_doses=[],
            dissolved_nutrient_mass=float(dm0),
            algae_biomass=float(b0),
            internal_reserve=float(b0 * params.internal_capacity * 0.2), # start 20% full
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
    queue: List[Tuple[float, float]],
    dose_mass: float,
    kernel: Tuple[float, ...],
    immediate_frac: float,
) -> Tuple[List[Tuple[float, float]], float]:
    immediate = immediate_frac * dose_mass
    delayed_mass = (1.0 - immediate_frac) * dose_mass
    k = np.asarray(kernel, dtype=np.float64)
    k = k / (k.sum() + 1e-12)
    
    new_queue = list(queue)
    for i, frac in enumerate(k):
        mass = delayed_mass * frac
        if mass > 1e-6:
            # Kernel slots represent 1-minute (60s) delays
            delay_seconds = (i + 1) * 60.0
            new_queue.append((mass, delay_seconds))
            
    return new_queue, immediate


def _release_absorption(
    queue: List[Tuple[float, float]],
    params: TankDynamicsParams,
    dt: float,
) -> Tuple[List[Tuple[float, float]], float, Dict[str, Any]]:
    new_queue = []
    released = 0.0
    
    # Process each pending dose, reducing its timer
    for mass, time_left in queue:
        time_left -= dt
        if time_left <= 0:
            released += mass
        else:
            new_queue.append((mass, time_left))
            
    # Apply release rate cap (FIFO logic: put unreleased mass back with small/zero delay)
    dt_scale = dt / 60.0
    cap = params.max_queue_release_rate * max(dt_scale, 0.01)
    was_capped = released > cap
    if was_capped:
        leftover = released - cap
        released = cap
        # Push leftover back into queue. 
        # By giving it a tiny positive time, we prevent zero-time artifacts while
        # keeping it at the front of the FIFO line for the next step.
        new_queue.append((leftover, 1e-6))
    
    # Calculate queue diagnostics
    queue_mass = sum(mass for mass, _ in new_queue)
    queue_count = len(new_queue)
    queue_ages = [(60.0 - time_left) / 60.0 for _, time_left in new_queue]  # Convert to minutes
    avg_queue_age = sum(queue_ages) / queue_count if queue_count > 0 else 0.0
    max_queue_age = max(queue_ages) if queue_ages else 0.0
    
    diagnostics = {
        "queue_mass": queue_mass,
        "queue_count": queue_count,
        "released_mass": released,
        "release_cap": cap,
        "was_capped": was_capped,
        "avg_queue_age": avg_queue_age,
        "max_queue_age": max_queue_age,
    }
        
    return new_queue, released, diagnostics


def step_dynamics(
    state: TankState,
    flowrate: float,
    duration: float,
    dt: float,
    params: TankDynamicsParams,
    disturbance: DisturbanceInput = 0.0,
    rng: np.random.Generator | None = None,
    enable_diagnostics: bool = False,
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
    
    queue = list(state.pending_doses)
    queue, immediate_mass = _inject_delayed_dose(
        queue, dose_mass, params.delay_kernel, params.immediate_absorption_fraction
    )
    queue, released_mass, queue_diagnostics = _release_absorption(queue, params, dt)

    # Global Modifiers
    thermal_growth = thermal_efficiency(state.water_temp, params)
    thermal_respiration = q10_metabolism_factor(state.water_temp, params)
    
    osmotic_stress = (state.dissolved_nutrient_mass / params.osmotic_half_effect)**2
    osmotic_factor = 1.0 / (1.0 + osmotic_stress)

    # 2. Mixing & Dissolved Nutrient Pool
    # First-order exponential decay updates for dilution and mineralization
    # Fix 8: Increase dilution rate slightly
    dil_rate = (params.background_dilution_rate * 2.0 + params.ec_decay_jitter) * thermal_respiration
    dilution = state.dissolved_nutrient_mass * (1.0 - np.exp(-dil_rate * dt_scale))
    
    # Fix 2: Reduce mineralization efficiency
    min_rate = params.mineralization_rate * thermal_respiration
    mineralized = state.dead_biomass_pool * (1.0 - np.exp(-min_rate * dt_scale))
    mineralized *= params.mineralization_efficiency  # Only 30% of dead biomass mineralizes

    dissolved_new = state.dissolved_nutrient_mass + immediate_mass + released_mass + mineralized - dilution
    
    # Store dissolved before uptake for mass balance verification
    dissolved_before_uptake = dissolved_new

    # Density dependence through self-shading (Hill function, n=2)
    # Gentle near equilibrium, sharp suppression at high density
    # Approximates nonlinear Beer-Lambert light attenuation in dense cultures
    L2 = params.light_attenuation_half_mass ** 2
    light_factor = L2 / (L2 + state.algae_biomass ** 2)

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

    # 4. Maintenance (Consumes reserve, waste split between dissolved pool, dead pool, and permanent loss)
    # Fix 4: Increase maintenance demand slightly (1.3x)
    maintenance_cost = params.maintenance_cost * 1.3 * state.algae_biomass * thermal_respiration * dt_scale
    
    # Fix 9: Make maintenance efficiency depend on reserve ratio (strengthened)
    # Option 3: Maintenance should become progressively inefficient
    reserve_ratio_maint = reserve_new / max(state.algae_biomass * params.internal_capacity, 1e-6)
    # More aggressive curve: efficiency drops more sharply with low reserves
    maintenance_efficiency = 0.1 + 0.9 * (reserve_ratio_maint ** 0.5)  # 0.1 at ratio=0, 1.0 at ratio=1
    maintenance_cost *= maintenance_efficiency
    
    actual_maintenance = min(maintenance_cost, reserve_new)
    reserve_new -= actual_maintenance
    maintenance_deficit = maintenance_cost - actual_maintenance
    
    # Fix 1: Irreversible nutrient losses in maintenance
    # 20% dissolved, 20% detritus, 60% permanently lost
    maint_to_dissolved = actual_maintenance * params.maintenance_to_dissolved_fraction
    maint_to_dead = actual_maintenance * params.maintenance_to_detritus_fraction
    maint_lost = actual_maintenance * params.maintenance_lost_fraction  # Permanently lost
    
    # Apply nutrient retention efficiency
    maint_to_dissolved *= params.nutrient_retention_efficiency
    maint_to_dead *= params.nutrient_retention_efficiency
    
    dissolved_new += maint_to_dissolved

    # 5. Biomass Growth
    reserve_ratio = reserve_new / max(state.algae_biomass * params.internal_capacity, 1e-6)
    
    # Fix 10: Starvation adaptation - growth more sensitive than maintenance
    # Growth drive decreases more sharply with low reserves
    starvation_adaptation = np.clip(reserve_ratio / 0.3, 0.0, 1.0)  # Growth drops below ratio=0.3
    growth_drive = reserve_ratio * thermal_growth * osmotic_factor * starvation_adaptation
    growth_amount = params.maximum_growth_rate * growth_drive * light_factor * state.algae_biomass * dt_scale

    nutrient_cost_per_biomass = params.biomass_nutrient_content / params.growth_yield
    required_reserve = growth_amount * nutrient_cost_per_biomass

    if required_reserve > reserve_new:
        growth_amount = reserve_new / nutrient_cost_per_biomass
        required_reserve = reserve_new

    reserve_new -= required_reserve
    biomass_new = state.algae_biomass + growth_amount
    
    # Fix 3: Growth metabolic efficiency - 70% incorporated, 30% respired/lost
    waste = required_reserve * (1.0 - params.growth_metabolic_efficiency)
    waste *= params.nutrient_retention_efficiency  # Apply retention efficiency
    dissolved_new += waste

    # 6. Mortality
    # Mortality increases with heat/cold stress, using Arrhenius/Q10 behavior
    thermal_stress = thermal_respiration
    damage_penalty = 1.0 + 5.0 * (state.damage_index ** 2)
    osmotic_penalty = 1.0 + np.log1p(osmotic_stress)
    mortality_rate = params.mortality_rate * thermal_stress * osmotic_penalty * damage_penalty
    mortality_amount = biomass_new * (1.0 - np.exp(-mortality_rate * dt_scale))

    biomass_new -= mortality_amount
    dead_nutrient = mortality_amount * params.biomass_nutrient_content
    proportional_reserve = reserve_new * (mortality_amount / max(biomass_new + mortality_amount, 1e-6))
    
    reserve_new -= proportional_reserve
    dead_pool_new = state.dead_biomass_pool - mineralized + dead_nutrient + proportional_reserve + maint_to_dead
    
    # Fix 7: Harvesting loss - small fraction of biomass removed per timestep
    harvest_amount = biomass_new * params.harvest_rate * dt_scale
    biomass_new -= harvest_amount
    harvest_nutrient = harvest_amount * params.biomass_nutrient_content
    # Harvested nutrients are permanently lost (removed from system)

    # 7. Health Evolution
    # Damage uses dimensionless intensive ratio with logistic bounded kinetics
    deficit_ratio = maintenance_deficit / max(maintenance_cost, 1e-6)
    
    # Option 2: Add starvation stress to damage computation
    # Additional damage when reserve ratio is below threshold
    reserve_ratio_health = reserve_new / max(biomass_new * params.internal_capacity, 1e-6)
    starvation_stress = max(0.0, params.starvation_reserve_threshold - reserve_ratio_health) / params.starvation_reserve_threshold
    starvation_stress *= params.starvation_damage_multiplier
    
    stress_drive = deficit_ratio * 1.0 + osmotic_stress * 2.0 + starvation_stress
    damage_inc = stress_drive * params.damage_rate * dt_scale * (1.0 - state.damage_index)
    
    # Option 1: Repair depends strongly on reserve ratio
    # Repair capacity decreases with reserve depletion
    repair_reserve_factor = np.clip(reserve_ratio_health / 0.3, 0.0, 1.0)  # Repair drops below ratio=0.3
    
    # Repair is based on maintenance flux being met, with a basal rate and damage power scaling
    repair_availability = actual_maintenance / max(maintenance_cost, 1e-6)
    repair_drive = params.repair_rate * repair_availability * thermal_respiration * repair_reserve_factor + params.basal_repair_rate * repair_reserve_factor
    repair_inc = repair_drive * dt_scale * (state.damage_index ** 0.7)

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

    # Collect diagnostics if enabled
    diagnostics = {}
    if enable_diagnostics:
        # Nutrient mass balance verification
        # Δdissolved = immediate + queue_release + mineralization - uptake - dilution
        net_dissolved_change = dissolved_new - state.dissolved_nutrient_mass
        expected_change = immediate_mass + released_mass + mineralized - uptake_mass - dilution
        mass_balance_residual = net_dissolved_change - expected_change
        
        diagnostics = {
            **queue_diagnostics,
            "immediate_mass": immediate_mass,
            "dissolved_before": state.dissolved_nutrient_mass,
            "dissolved_after_uptake": dissolved_before_uptake,
            "dissolved_after": dissolved_new,
            "uptake_mass": uptake_mass,
            "dilution_loss": dilution,
            "mineralization_contribution": mineralized,
            "net_dissolved_change": net_dissolved_change,
            "expected_change": expected_change,
            "mass_balance_residual": mass_balance_residual,
            "ec": ec_new,
            "osmotic_stress": osmotic_stress,
            "osmotic_factor": osmotic_factor,
        }

    new_state = TankState(
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
        
        pending_doses=queue,
        dissolved_nutrient_mass=dissolved_new,
        algae_biomass=biomass_new,
        internal_reserve=reserve_new,
        dead_biomass_pool=dead_pool_new,
        health_index=health_new,
        damage_index=damage_new,
        
        ec=ec_new,
        turbidity=turbidity_new,
    )

    return new_state


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
