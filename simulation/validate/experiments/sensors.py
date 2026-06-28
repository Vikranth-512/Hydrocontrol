"""
V. Sensor Validation Experiments

Verify the sensor layer is fully decoupled from plant dynamics:
  - EC linearity and gain
  - Turbidity lag measurement
  - Sensor never influences biology
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import TankDynamicsParams, step_dynamics
from simulation.validate.common import Experiment, make_initial_state, simulate_and_record
from simulation.validate.metrics import cross_correlation


# ---------------------------------------------------------------------------
# SEN-01  EC Linearity & Gain Verification
# ---------------------------------------------------------------------------
def run_ec_linearity(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Verify EC = sensor_gain * dissolved_nutrient_mass (+ noise terms)."""
    dt = 60.0
    length = 2000
    rng = np.random.default_rng(42)
    actions = [(rng.uniform(0, 3.0), rng.uniform(0, 20.0)) for _ in range(length)]
    s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)

    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0, rng=rng)
    df.to_csv(output_dir / "ec_linearity.csv", index=False)

    # Compute expected EC and residual
    expected_ec = params.sensor_gain_ec * df["dissolved_nutrient_mass"]
    residual = df["ec"] - expected_ec

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].scatter(df["dissolved_nutrient_mass"], df["ec"], s=1, alpha=0.3)
    axes[0].plot([0, df["dissolved_nutrient_mass"].max()],
                 [0, params.sensor_gain_ec * df["dissolved_nutrient_mass"].max()],
                 "r--", label=f"Gain = {params.sensor_gain_ec}")
    axes[0].set_xlabel("Dissolved Nutrient Mass")
    axes[0].set_ylabel("EC")
    axes[0].set_title("EC vs Dissolved Mass")
    axes[0].legend()
    axes[1].plot(df["time_min"], residual, linewidth=0.5)
    axes[1].set_title("EC Residual (EC - gain × dissolved)")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(output_dir / "ec_linearity.png", dpi=150)
    plt.close(fig)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_residual = float(np.max(np.abs(residual)))

    return {
        "status": "PASS" if rmse < 0.5 else "FAIL",
        "metrics": {
            "EC RMSE vs Linear Model": rmse,
            "Max Residual": max_residual,
        },
    }


# ---------------------------------------------------------------------------
# SEN-02  Turbidity Lag Measurement
# ---------------------------------------------------------------------------
def run_turbidity_lag(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 1000
    actions = [(2.0, 15.0)] * length
    s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
    df.to_csv(output_dir / "turbidity_lag.csv", index=False)

    # True optical density
    true_od = params.biomass_optical_factor * df["algae_biomass"]
    lag_residual = df["turbidity"] - true_od

    # Cross-correlation to measure lag
    lags, corr = cross_correlation(true_od.values, df["turbidity"].values, max_lag=50)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(df["time_min"], true_od, label="True Optical Density")
    axes[0].plot(df["time_min"], df["turbidity"], label="Turbidity Sensor")
    axes[0].set_title("Turbidity Lag")
    axes[0].set_xlabel("Time (min)")
    axes[0].legend()
    axes[1].plot(lags, corr)
    axes[1].set_title("Cross-Correlation (True OD ↔ Turbidity)")
    axes[1].set_xlabel("Lag (steps)")
    axes[1].set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "turbidity_lag.png", dpi=150)
    plt.close(fig)

    peak_lag = int(lags[np.argmax(corr)])
    peak_corr = float(np.max(corr))

    return {
        "status": "PASS" if peak_corr > 0.9 else "FAIL",
        "metrics": {
            "Peak Lag (steps)": peak_lag,
            "Peak Correlation": peak_corr,
            "Sensor Tau": params.algae_sensor_tau,
        },
    }


# ---------------------------------------------------------------------------
# SEN-03  Sensor Decoupling Proof
# ---------------------------------------------------------------------------
def run_sensor_decoupling(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """
    Run two identical scenarios where sensor_gain differs.
    If sensors are decoupled, physical states (dissolved, biomass, reserve, health)
    should be IDENTICAL despite different EC readings.
    """
    dt = 60.0
    length = 500
    actions = [(2.0, 15.0)] * length

    # Scenario A: normal gain
    s0_a = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    df_a = simulate_and_record(params, length, dt, actions=actions, initial_state=s0_a)

    # Scenario B: doubled gain
    import copy
    params_b = copy.deepcopy(params)
    params_b.sensor_gain_ec = params.sensor_gain_ec * 2.0
    s0_b = make_initial_state(params_b, dissolved_mass=2.0, biomass=100.0)
    df_b = simulate_and_record(params_b, length, dt, actions=actions, initial_state=s0_b)

    df_a.to_csv(output_dir / "sensor_decoupling_A.csv", index=False)
    df_b.to_csv(output_dir / "sensor_decoupling_B.csv", index=False)

    # Physical states should be identical
    physical_cols = ["dissolved_nutrient_mass", "algae_biomass", "internal_reserve", "health_index"]
    max_diffs = {}
    for col in physical_cols:
        diff = np.abs(df_a[col].values - df_b[col].values)
        max_diffs[f"Max Delta_{col}"] = float(np.max(diff))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), physical_cols):
        ax.plot(df_a["time_min"], df_a[col], label="Gain=0.8")
        ax.plot(df_b["time_min"], df_b[col], "--", label="Gain=1.6")
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Sensor Decoupling Proof: Physical States Must Be Identical")
    fig.tight_layout()
    fig.savefig(output_dir / "sensor_decoupling.png", dpi=150)
    plt.close(fig)

    all_zero = all(v < 1e-9 for v in max_diffs.values())
    return {
        "status": "PASS" if all_zero else "FAIL",
        "metrics": max_diffs,
        "warnings": [] if all_zero else ["Sensor gain changes physical state! This is a critical coupling leak."],
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="SEN-01", name="EC Linearity & Gain", category="V. Sensor Validation",
        hypothesis="EC is a linear function of dissolved_nutrient_mass with gain = sensor_gain_ec.",
        execute=run_ec_linearity,
        metrics=["EC RMSE vs Linear Model", "Max Residual"], plots=["ec_linearity.png"],
    ),
    Experiment(
        id="SEN-02", name="Turbidity Lag", category="V. Sensor Validation",
        hypothesis="Turbidity tracks biomass optical density through a first-order lag filter.",
        execute=run_turbidity_lag,
        metrics=["Peak Lag", "Peak Correlation"], plots=["turbidity_lag.png"],
    ),
    Experiment(
        id="SEN-03", name="Sensor Decoupling Proof", category="V. Sensor Validation",
        hypothesis="Changing sensor_gain_ec has zero effect on physical states (dissolved, biomass, reserve, health).",
        execute=run_sensor_decoupling,
        metrics=["Max Delta_dissolved_nutrient_mass", "Max Delta_algae_biomass"], plots=["sensor_decoupling.png"],
    ),
]
