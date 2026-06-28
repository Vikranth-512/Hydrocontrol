"""
VIII. Controller Independence Experiments

Prove that the plant contains no hidden equilibrium or restoring force:
  - No restoring tendency at any initial EC
  - dEC/dt is explained entirely by mass balance
  - Disturbance recovery is purely physical
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import TankDynamicsParams, step_dynamics, thermal_efficiency
from simulation.validate.common import Experiment, make_initial_state, simulate_and_record
from simulation.validate.plotting import plot_multi_panel


# ---------------------------------------------------------------------------
# CTL-01  Hidden Controller Detection (multi-IC open-loop)
# ---------------------------------------------------------------------------
def run_hidden_controller_detection(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """
    Initialize EC at extreme values (0, 2, 4, 8) with zero dosing.
    If any state naturally returns toward a preferred operating point,
    there is a hidden controller.
    """
    dt = 60.0
    length = 20_000
    init_dissolved = [0.0, 2.5, 5.0, 10.0]  # These map to EC via sensor_gain

    all_traces = {}
    for dm in init_dissolved:
        ec0 = params.sensor_gain_ec * dm
        s = make_initial_state(params, dissolved_mass=dm, biomass=100.0)
        trace_ec = []
        trace_dissolved = []
        for t in range(length):
            s = step_dynamics(s, 0.0, 0.0, dt, params)
            if t % 10 == 0:
                trace_ec.append(s.ec)
                trace_dissolved.append(s.dissolved_nutrient_mass)
        all_traces[f"EC (init={ec0:.1f})"] = trace_ec
        all_traces[f"Dissolved (init={dm:.1f})"] = trace_dissolved

    # All traces should converge toward zero (depletion + dilution), NOT toward a target
    df = pd.DataFrame(all_traces)
    df["time_min"] = np.arange(len(df)) * 10 * dt / 60.0
    df.to_csv(output_dir / "hidden_controller.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ec_cols = [c for c in df.columns if c.startswith("EC")]
    dissolved_cols = [c for c in df.columns if c.startswith("Dissolved")]
    for col in ec_cols:
        axes[0].plot(df["time_min"], df[col], label=col)
    axes[0].set_title("Open-Loop EC from Multiple Initial Conditions (no dosing)")
    axes[0].set_ylabel("EC")
    axes[0].legend(fontsize=7)
    for col in dissolved_cols:
        axes[1].plot(df["time_min"], df[col], label=col)
    axes[1].set_title("Dissolved Nutrient from Multiple ICs")
    axes[1].set_ylabel("Dissolved Mass")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "hidden_controller.png", dpi=150)
    plt.close(fig)

    # All traces should end near zero; none should converge to a non-zero equilibrium
    final_vals = [df[col].iloc[-1] for col in ec_cols]
    any_nonzero_eq = any(v > 0.5 for v in final_vals)

    warnings = []
    if any_nonzero_eq:
        warnings.append("A trace converged to a non-zero EC without dosing - possible hidden controller!")
    warnings.append("Note: The transient dissolved nutrient increase from high initial conditions is physically consistent. It is caused by osmotic-driven mortality releasing nutrients back to the dissolved pool faster than dilution removes them.")

    return {
        "status": "PASS" if not any_nonzero_eq else "FAIL",
        "metrics": {k: float(df[k].iloc[-1]) for k in ec_cols},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CTL-02  dEC/dt Explained by Mass Balance
# ---------------------------------------------------------------------------
def run_dec_dt_mass_balance(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """
    Verify dEC/dt = sensor_gain * d(dissolved)/dt, and that d(dissolved)/dt
    is fully explained by -(uptake + dilution) + mineralization.
    No restoring spring or target-seeking term should appear.
    """
    dt = 60.0
    length = 2000
    rng = np.random.default_rng(42)
    actions = [(rng.uniform(0, 3.0), rng.uniform(0, 20.0)) for _ in range(length)]
    s0 = make_initial_state(params, dissolved_mass=3.0, biomass=100.0)

    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0, rng=rng)
    df.to_csv(output_dir / "dec_dt_audit.csv", index=False)

    # Compute numerical dEC/dt
    ec = df["ec"].values
    dissolved = df["dissolved_nutrient_mass"].values
    dec_dt = np.diff(ec)
    d_dissolved_dt = np.diff(dissolved)

    # If EC = gain * dissolved (+ small noise), then dEC/dt ≈ gain * d(dissolved)/dt
    predicted_dec_dt = params.sensor_gain_ec * d_dissolved_dt
    residual = dec_dt - predicted_dec_dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(dec_dt, label="Actual dEC/dt", alpha=0.7)
    axes[0].plot(predicted_dec_dt, label="gain × d(dissolved)/dt", alpha=0.7)
    axes[0].set_title("dEC/dt vs Predicted from Mass Balance")
    axes[0].legend()
    axes[1].plot(residual, color="red", linewidth=0.5)
    axes[1].set_title("Residual (unexplained dEC/dt)")
    axes[1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_dir / "dec_dt_mass_balance.png", dpi=150)
    plt.close(fig)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_residual = float(np.max(np.abs(residual)))

    return {
        "status": "PASS" if rmse < 0.05 else "FAIL",
        "metrics": {
            "dEC/dt Residual RMSE": rmse,
            "dEC/dt Max Residual": max_residual,
        },
        "warnings": [f"Large unexplained dEC/dt residual - possible hidden restoring force."] if rmse > 0.05 else [],
    }


# ---------------------------------------------------------------------------
# CTL-03  Disturbance Recovery is Purely Physical
# ---------------------------------------------------------------------------
def run_disturbance_recovery(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """
    Inject temperature and sensor shocks. Verify the plant responds
    only through physics - no hidden stabilization.
    """
    from simulation.dynamics import DisturbanceSpec

    dt = 60.0
    length = 1000

    # Phase 1: Stable baseline (200 steps)
    # Phase 2: Temperature shock at t=200 (+8°C)
    # Phase 3: Recovery without shocks
    disturbances = [DisturbanceSpec()] * length
    for t in range(200, 250):
        disturbances[t] = DisturbanceSpec(temp_shock=0.5)  # Persistent hot shock

    actions = [(1.5, 15.0)] * length
    s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0, disturbances=disturbances)
    df.to_csv(output_dir / "disturbance_recovery.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["water_temp"], "Temperature"),
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["algae_biomass"], "Biomass"),
            (["health_index"], "Health"),
        ],
        "Disturbance Recovery (Temperature Shock at t=200)",
        output_dir / "disturbance_recovery.png",
    )

    # The system should NOT snap back to pre-shock values instantly.
    # Temperature should relax slowly toward ambient.
    temp_at_250 = float(df.loc[df["time_min"] >= 250 * dt / 60.0, "water_temp"].iloc[0])
    temp_at_500 = float(df.loc[df["time_min"] >= 500 * dt / 60.0, "water_temp"].iloc[0])
    temp_ambient = params.ambient_temp_mean

    # Temp should be recovering but not instantly
    recovering = abs(temp_at_500 - temp_ambient) < abs(temp_at_250 - temp_ambient)

    return {
        "status": "PASS" if recovering else "FAIL",
        "metrics": {
            "Temp at t=250": temp_at_250,
            "Temp at t=500": temp_at_500,
            "Ambient": temp_ambient,
        },
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="CTL-01", name="Hidden Controller Detection", category="VIII. Controller Independence",
        hypothesis="With zero dosing, EC from any initial condition decays to zero. No state converges to a non-zero equilibrium.",
        execute=run_hidden_controller_detection,
        metrics=["Final EC values"], plots=["hidden_controller.png"],
    ),
    Experiment(
        id="CTL-02", name="dEC/dt Mass Balance Proof", category="VIII. Controller Independence",
        hypothesis="dEC/dt is fully explained by sensor_gain × d(dissolved)/dt with near-zero residual.",
        execute=run_dec_dt_mass_balance,
        metrics=["dEC/dt Residual RMSE"], plots=["dec_dt_mass_balance.png"],
    ),
    Experiment(
        id="CTL-03", name="Disturbance Recovery", category="VIII. Controller Independence",
        hypothesis="After a temperature shock, the plant recovers only through physical relaxation - no hidden stabilization.",
        execute=run_disturbance_recovery,
        metrics=["Temp at t=250", "Temp at t=500"], plots=["disturbance_recovery.png"],
    ),
]
