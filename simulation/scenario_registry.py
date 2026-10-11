"""
Authoritative Central Scenario Registry for Hydroponic Nutrient Regulation Benchmark.

Single source of truth defining all 12 operational regimes across:
- Environmental Stress
- Observation Degradation
- Actuation & Transport Limitations

References canonical simulator, disturbance, and environment implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class ScenarioSpec:
    """Specification of an operational regime in the benchmark."""

    scenario_id: str
    display_name: str
    family: str  # "Environmental Stress", "Observation Degradation", "Actuation & Transport"
    primary_affected_subsystem: str  # "environmental_biology", "sensor_observations", "actuator_delivery", "physical_transport", "sensor_telemetry"
    latent_state_affected: bool
    observation_affected: bool
    actuator_affected: bool
    temporal_profile: str  # "continuous", "sustained", "transient", "delayed", "late", "periodic"
    source_file: str
    implementation_reference: str
    description: str


# Canonical list of all 12 benchmark scenarios in authoritative evaluation order.
# Scenarios 0-3 MUST be normal, heatwave, nutrient_depletion, cold_shock to guarantee
# exact seed-namespace preservation and backward compatibility.
BENCHMARK_SCENARIOS: Dict[str, ScenarioSpec] = {
    "normal": ScenarioSpec(
        scenario_id="normal",
        display_name="Normal (Baseline)",
        family="Environmental Stress",
        primary_affected_subsystem="environmental_biology",
        latent_state_affected=False,
        observation_affected=False,
        actuator_affected=False,
        temporal_profile="continuous",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='normal')",
        description="Baseline open-loop dynamics with ambient temperature relaxation, correlated mixing noise, and background stochastic pulses.",
    ),
    "heatwave": ScenarioSpec(
        scenario_id="heatwave",
        display_name="Heatwave (+4.0°C)",
        family="Environmental Stress",
        primary_affected_subsystem="environmental_biology",
        latent_state_affected=True,
        observation_affected=False,
        actuator_affected=False,
        temporal_profile="sustained",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='heatwave')",
        description="Sustained positive thermal disturbance (+4.0°C ambient shift for 40 steps) inducing elevated biological uptake rate (1.2x).",
    ),
    "nutrient_depletion": ScenarioSpec(
        scenario_id="nutrient_depletion",
        display_name="Nutrient Depletion",
        family="Environmental Stress",
        primary_affected_subsystem="environmental_biology",
        latent_state_affected=True,
        observation_affected=False,
        actuator_affected=False,
        temporal_profile="sustained",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='nutrient_depletion')",
        description="Negative dissolved nutrient step shock (-0.048 mS/cm) paired with elevated biological consumption demand (1.35x uptake) for 30 steps.",
    ),
    "cold_shock": ScenarioSpec(
        scenario_id="cold_shock",
        display_name="Cold Shock (-3.5°C)",
        family="Environmental Stress",
        primary_affected_subsystem="environmental_biology",
        latent_state_affected=True,
        observation_affected=False,
        actuator_affected=False,
        temporal_profile="sustained",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='cold_shock')",
        description="Severe negative thermal disturbance (-3.5°C for 25 steps) causing Q10 metabolic suppression and reduced uptake rate (0.85x).",
    ),
    "noisy_sensors": ScenarioSpec(
        scenario_id="noisy_sensors",
        display_name="Noisy Sensors (3x Noise)",
        family="Observation Degradation",
        primary_affected_subsystem="sensor_observations",
        latent_state_affected=False,
        observation_affected=True,
        actuator_affected=False,
        temporal_profile="continuous",
        source_file="simulation/environment.py",
        implementation_reference="AlgaeTankEnvironment._observe() with 3.0x noise_std",
        description="Tripled Gaussian observation noise standard deviation across all sensor channels (EC, temperature, turbidity) without altering latent states.",
    ),
    "delayed_response": ScenarioSpec(
        scenario_id="delayed_response",
        display_name="Delayed Response",
        family="Actuation & Transport",
        primary_affected_subsystem="physical_transport",
        latent_state_affected=True,
        observation_affected=False,
        actuator_affected=True,
        temporal_profile="delayed",
        source_file="simulation/synthetic_generator.py",
        implementation_reference="TankDynamicsParams(immediate_absorption=0.08, delay_kernel=(0.1, 0.25, 0.35, 0.30))",
        description="Extended physical mixing and transport delay: immediate nutrient absorption reduced from 50% to 8%, with a 4-stage absorption lag kernel.",
    ),
    "actuator_failure": ScenarioSpec(
        scenario_id="actuator_failure",
        display_name="Actuator Failure (35% Eff)",
        family="Actuation & Transport",
        primary_affected_subsystem="actuator_delivery",
        latent_state_affected=False,
        observation_affected=False,
        actuator_affected=True,
        temporal_profile="late",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='actuator_failure') with actuator_eff=0.35 for t > T/2",
        description="Mechanical dosing pump failure occurring after mid-horizon (t > T/2), dropping delivered dose mass to 35% of commanded quantity.",
    ),
    "actuator_saturation": ScenarioSpec(
        scenario_id="actuator_saturation",
        display_name="Actuator Saturation",
        family="Actuation & Transport",
        primary_affected_subsystem="actuator_delivery",
        latent_state_affected=False,
        observation_affected=False,
        actuator_affected=True,
        temporal_profile="continuous",
        source_file="simulation_runner/closed_loop_eval.py",
        implementation_reference="ClosedLoopEvaluator.run_episode() clamping fr<=1.0, dur<=5.0",
        description="Severe hardware pump authority limit clamping commanded actions to max flowrate 1.0 mL/min and max duration 5.0 s.",
    ),
    "sediment": ScenarioSpec(
        scenario_id="sediment",
        display_name="Sediment Plume",
        family="Observation Degradation",
        primary_affected_subsystem="sensor_observations",
        latent_state_affected=False,
        observation_affected=True,
        actuator_affected=False,
        temporal_profile="transient",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='sediment')",
        description="Optical sensor obscuration by particulate sediment plume (+18.0 NTU turbidity shock for 15 steps) without biological biomass alteration.",
    ),
    "missing_data": ScenarioSpec(
        scenario_id="missing_data",
        display_name="Missing Data (EC NaN)",
        family="Observation Degradation",
        primary_affected_subsystem="sensor_telemetry",
        latent_state_affected=False,
        observation_affected=True,
        actuator_affected=False,
        temporal_profile="periodic",
        source_file="simulation_runner/closed_loop_eval.py",
        implementation_reference="ClosedLoopEvaluator.run_episode() injecting obs_dict['ec']=NaN every 17 steps",
        description="Periodic telemetry dropout replacing raw EC sensor readings with NaN every 17 timesteps to test observation processing robustness.",
    ),
    "temp_spike": ScenarioSpec(
        scenario_id="temp_spike",
        display_name="Temperature Spike",
        family="Environmental Stress",
        primary_affected_subsystem="environmental_biology",
        latent_state_affected=True,
        observation_affected=False,
        actuator_affected=False,
        temporal_profile="transient",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='temp_spike')",
        description="Acute positive temperature pulse (+0.06°C/step for 40 steps, uptake 1.2x) testing recovery from rapid transient thermal shocks.",
    ),
    "ec_drop": ScenarioSpec(
        scenario_id="ec_drop",
        display_name="EC Drop",
        family="Environmental Stress",
        primary_affected_subsystem="environmental_biology",
        latent_state_affected=True,
        observation_affected=False,
        actuator_affected=False,
        temporal_profile="transient",
        source_file="simulation/disturbances.py",
        implementation_reference="DisturbanceGenerator.sample(scenario='ec_drop')",
        description="Acute dissolved nutrient concentration drop (-0.048 mS/cm for 30 steps, uptake 1.35x) testing sudden nutrient deficit recovery.",
    ),
}


def get_scenario_registry() -> Dict[str, ScenarioSpec]:
    """Return dictionary of all 12 authoritative operational regimes."""
    return dict(BENCHMARK_SCENARIOS)


def get_scenario_ids() -> List[str]:
    """Return list of all 12 canonical scenario IDs in order."""
    return list(BENCHMARK_SCENARIOS.keys())


def get_scenario_spec(scenario_id: str) -> ScenarioSpec:
    """Retrieve specification for a given scenario ID."""
    if scenario_id not in BENCHMARK_SCENARIOS:
        raise KeyError(f"Unknown scenario ID: {scenario_id}. Must be one of {list(BENCHMARK_SCENARIOS.keys())}")
    return BENCHMARK_SCENARIOS[scenario_id]


def get_scenario_orthogonality_matrix() -> pd.DataFrame:
    """Generate scenario orthogonality audit matrix."""
    rows = []
    for sc_id, spec in BENCHMARK_SCENARIOS.items():
        rows.append({
            "scenario_id": spec.scenario_id,
            "display_name": spec.display_name,
            "family": spec.family,
            "primary_affected_subsystem": spec.primary_affected_subsystem,
            "latent_state_affected": spec.latent_state_affected,
            "observation_affected": spec.observation_affected,
            "actuator_affected": spec.actuator_affected,
            "temporal_profile": spec.temporal_profile,
            "source_file": spec.source_file,
        })
    return pd.DataFrame(rows)


def validate_scenario_registry() -> bool:
    """Validate registry integrity, completeness, and ordering."""
    ids = get_scenario_ids()
    assert len(ids) == 12, f"Expected exactly 12 scenarios, got {len(ids)}"
    assert len(set(ids)) == 12, "Scenario IDs must be unique"
    # Verify first 4 scenarios are preserved for regression
    assert ids[:4] == ["normal", "heatwave", "nutrient_depletion", "cold_shock"], (
        f"First 4 scenarios must be normal, heatwave, nutrient_depletion, cold_shock. Got {ids[:4]}"
    )
    return True


def get_multiseed_evaluation_scenarios() -> List[str]:
    """
    Return the list of scenarios for multiseed long-horizon evaluation.
    
    This excludes 'missing_data' from the full registry, resulting in 11 scenarios:
    normal, heatwave, nutrient_depletion, cold_shock, noisy_sensors, delayed_response,
    actuator_failure, actuator_saturation, sediment, temp_spike, ec_drop
    """
    all_scenarios = get_scenario_ids()
    # Exclude missing_data from multiseed evaluation
    multiseed_scenarios = [s for s in all_scenarios if s != "missing_data"]
    return multiseed_scenarios
