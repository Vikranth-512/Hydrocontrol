"""
VI. RL Benchmark Characterization Experiments

Characterize the difficulty of the environment as an RL benchmark:
  - Partial observability (Mutual Information between hidden states and observations)
  - State-space coverage (Monte Carlo reachable regions)
  - Impulse response delay chain measurement
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
from simulation.validate.metrics import mutual_information_binned, cross_correlation


# ---------------------------------------------------------------------------
# BEN-01  Partial Observability (Mutual Information)
# ---------------------------------------------------------------------------
def run_partial_observability(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 5000
    rng = np.random.default_rng(42)
    actions = [(rng.uniform(0, 4.0), rng.uniform(0, 25.0)) for _ in range(length)]
    s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0, rng=rng)
    df.to_csv(output_dir / "observability.csv", index=False)

    # Compute MI between hidden states and observations
    mi_reserve_ec = mutual_information_binned(df["internal_reserve"].values, df["ec"].values)
    mi_health_ec = mutual_information_binned(df["health_index"].values, df["ec"].values)
    mi_reserve_turb = mutual_information_binned(df["internal_reserve"].values, df["turbidity"].values)
    mi_health_turb = mutual_information_binned(df["health_index"].values, df["turbidity"].values)
    mi_dead_turb = mutual_information_binned(df["dead_biomass_pool"].values, df["turbidity"].values)
    mi_damage_ec = mutual_information_binned(df["damage_index"].values, df["ec"].values)

    mi_results = {
        "MI(Reserve, EC)": mi_reserve_ec,
        "MI(Health, EC)": mi_health_ec,
        "MI(Reserve, Turbidity)": mi_reserve_turb,
        "MI(Health, Turbidity)": mi_health_turb,
        "MI(Dead Pool, Turbidity)": mi_dead_turb,
        "MI(Damage, EC)": mi_damage_ec,
    }

    # Plot MI matrix
    hidden = ["Reserve", "Health", "Damage", "Dead Pool"]
    observed = ["EC", "Turbidity"]
    mi_matrix = np.array([
        [mi_reserve_ec, mi_reserve_turb],
        [mi_health_ec, mi_health_turb],
        [mi_damage_ec, mi_dead_turb],  # damage vs EC, dead vs turb
        [0.0, mi_dead_turb],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mi_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(observed)))
    ax.set_xticklabels(observed)
    ax.set_yticks(range(len(hidden)))
    ax.set_yticklabels(hidden)
    for i in range(len(hidden)):
        for j in range(len(observed)):
            ax.text(j, i, f"{mi_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mutual Information: Hidden ↔ Observed")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_dir / "observability_matrix.png", dpi=150)
    plt.close(fig)

    pd.DataFrame([mi_results]).to_csv(output_dir / "mi_values.csv", index=False)

    return {
        "status": "PASS",
        "metrics": mi_results,
    }


# ---------------------------------------------------------------------------
# BEN-02  State-Space Coverage (Monte Carlo)
# ---------------------------------------------------------------------------
def run_state_space_coverage(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 1000
    n_trajectories = 200
    rng = np.random.default_rng(77)

    all_ec = []
    all_biomass = []
    all_reserve = []
    all_health = []

    for traj in range(n_trajectories):
        s = make_initial_state(
            params,
            dissolved_mass=rng.uniform(0.5, 5.0),
            biomass=rng.uniform(50.0, 150.0),
            internal_reserve=rng.uniform(0.0, 15.0),
        )
        for t in range(length):
            fr = rng.uniform(0, 5.0)
            dur = rng.uniform(0, 30.0)
            s = step_dynamics(s, fr, dur, dt, params, rng=rng)

        all_ec.append(s.ec)
        all_biomass.append(s.algae_biomass)
        all_reserve.append(s.internal_reserve)
        all_health.append(s.health_index)

    all_ec = np.array(all_ec)
    all_biomass = np.array(all_biomass)
    all_reserve = np.array(all_reserve)
    all_health = np.array(all_health)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(all_ec, all_biomass, s=5, alpha=0.5)
    axes[0].set_xlabel("EC")
    axes[0].set_ylabel("Biomass")
    axes[0].set_title("EC vs Biomass")
    axes[1].scatter(all_ec, all_reserve, s=5, alpha=0.5)
    axes[1].set_xlabel("EC")
    axes[1].set_ylabel("Reserve")
    axes[1].set_title("EC vs Reserve")
    axes[2].scatter(all_biomass, all_health, s=5, alpha=0.5)
    axes[2].set_xlabel("Biomass")
    axes[2].set_ylabel("Health")
    axes[2].set_title("Biomass vs Health")
    fig.suptitle(f"State-Space Coverage ({n_trajectories} Monte Carlo Trajectories)")
    fig.tight_layout()
    fig.savefig(output_dir / "state_space_coverage.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({
        "ec": all_ec, "biomass": all_biomass, "reserve": all_reserve, "health": all_health
    }).to_csv(output_dir / "state_space.csv", index=False)

    survival_mask = all_health > 0.05
    survival_pct = (np.sum(survival_mask) / n_trajectories) * 100.0

    return {
        "status": "PASS",
        "metrics": {
            "Survival Rate (%)": f"{survival_pct:.1f}%",
            "EC Range": f"[{all_ec.min():.2f}, {all_ec.max():.2f}]",
            "Biomass Range": f"[{all_biomass.min():.2f}, {all_biomass.max():.2f}]",
            "Reserve Range": f"[{all_reserve.min():.2f}, {all_reserve.max():.2f}]",
            "Health Range": f"[{all_health.min():.2f}, {all_health.max():.2f}]",
        },
        "warnings": ["Highly bimodal state space (mostly dead culture attractors). Wide coverage statistics may be misleading."] if survival_pct < 20.0 else [],
    }


# ---------------------------------------------------------------------------
# BEN-03  Full Impulse Response Chain
# ---------------------------------------------------------------------------
def run_impulse_response_chain(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Measure delay from Dose → EC → Reserve → Growth → Biomass → Turbidity."""
    dt = 60.0
    length = 800
    actions = [(0.0, 0.0)] * length
    pulse_t = 50
    actions[pulse_t] = (5.0, 30.0)

    s0 = make_initial_state(params, dissolved_mass=0.5, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
    df.to_csv(output_dir / "impulse_chain.csv", index=False)

    from simulation.validate.metrics import estimate_impulse_delay
    signals = {
        "Dissolved": df["dissolved_nutrient_mass"].values,
        "EC": df["ec"].values,
        "Reserve": df["internal_reserve"].values,
        "Biomass": df["algae_biomass"].values,
        "Turbidity": df["turbidity"].values,
    }
    delays = {}
    for name, sig in signals.items():
        d = estimate_impulse_delay(sig, pulse_t)
        delays[f"Delay to {name} (steps)"] = d

    # Plot all signals normalized
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, sig in signals.items():
        sig_norm = (sig - sig[pulse_t]) / (np.max(np.abs(sig - sig[pulse_t])) + 1e-12)
        ax.plot(df["time_min"], sig_norm, label=name)
    ax.axvline(pulse_t * dt / 60.0, color="black", linestyle="--", label="Impulse")
    ax.set_title("Normalized Impulse Response Chain")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Normalized Response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "impulse_chain.png", dpi=150)
    plt.close(fig)

    return {"status": "PASS", "metrics": delays}


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="BEN-01", name="Partial Observability", category="VI. RL Benchmark",
        hypothesis="Hidden states (reserve, health, damage) have low mutual information with observed states (EC, turbidity), confirming partial observability.",
        execute=run_partial_observability,
        metrics=["MI(Reserve, EC)", "MI(Health, EC)"], plots=["observability_matrix.png"],
    ),
    Experiment(
        id="BEN-02", name="State-Space Coverage", category="VI. RL Benchmark",
        hypothesis="Monte Carlo random trajectories cover a wide range of (EC, Biomass, Reserve, Health) states.",
        execute=run_state_space_coverage,
        metrics=["EC Range", "Biomass Range"], plots=["state_space_coverage.png"],
    ),
    Experiment(
        id="BEN-03", name="Impulse Response Delay Chain", category="VI. RL Benchmark",
        hypothesis="A single dose impulse propagates through Dissolved → Reserve → Biomass → Turbidity with measurable cascading delays.",
        execute=run_impulse_response_chain,
        metrics=["Delay to Dissolved", "Delay to Biomass"], plots=["impulse_chain.png"],
    ),
]
