import numpy as np
import pandas as pd
from pathlib import Path

from simulation.dynamics import (
TankDynamicsParams,
TankState,
step_dynamics,
soft_ec_saturation_factor,
_soft_clip_ec,
)

from simulation.disturbances import DisturbanceSpec

REPORT_DIR = Path("audit_report")
REPORT_DIR.mkdir(exist_ok=True)

def reconstruct_ec_terms(state, flowrate, duration, dt, params, disturbance):


    dist = disturbance
    dt_scale = dt / 60.0

    dose_mass = flowrate * duration / 60.0 * dist.actuator_efficiency

    cumulative = state.cumulative_nutrients + dose_mass

    nutrient_memory = (
        (1.0 - params.nutrient_memory_alpha) * state.nutrient_memory
        + params.nutrient_memory_alpha * dose_mass
    )

    queue = np.asarray(state.absorption_queue, dtype=np.float64)

    from simulation.dynamics import (
        _inject_delayed_dose,
        _release_absorption,
        thermal_efficiency,
        q10_metabolism_factor,
    )

    queue, immediate_mass = _inject_delayed_dose(
        queue,
        dose_mass,
        params.delay_kernel,
        params.immediate_absorption_fraction,
    )

    queue, released_mass = _release_absorption(
        queue,
        params,
        dt_scale,
    )

    thermal = thermal_efficiency(
        state.water_temp,
        params,
    )

    baseline_drain = (
        params.baseline_ec_depletion
        + params.ec_decay_jitter
    ) * thermal

    proportional_uptake = (
    params.biological_uptake_rate
    * np.clip(state.ec, 0.0, None)
    * thermal
    * dist.uptake_multiplier
    )

    starvation = 0.0

    if state.ec < params.ec_healthy_min:
        starvation = (
            params.starvation_acceleration
            * (params.ec_healthy_min - state.ec)
            * thermal
        )

    depletion = (
        baseline_drain
        + proportional_uptake
        + starvation
    ) * dt_scale

    sat = soft_ec_saturation_factor(
        state.ec,
        params,
    )

    gain = (
        params.nutrient_to_ec_gain
        + params.gain_jitter
    ) * thermal * sat

    raw_influx = gain * (
        immediate_mass + released_mass
    )

    memory_influx = (
        params.memory_to_ec_gain
        * nutrient_memory
        * thermal
        * sat
        * dt_scale
    )

    assimilation_pool = (
        (1.0 - params.ec_assimilation_tau)
        * state.assimilation_pool
        + params.ec_assimilation_tau
        * (raw_influx + memory_influx)
    )

    restoring = 0.0

    if (
        state.health_index > 0.85
        and state.ec < params.ec_target
    ):
        restoring = (
            params.ec_spring_stiffness
            * (params.ec_target - state.ec)
        )

    ec_velocity = (
        state.ec_velocity
        + dt_scale
        * (
            assimilation_pool
            - depletion
            + restoring
            - params.ec_damping * state.ec_velocity
        )
    )

    overshoot_effect = 0.0

    if state.ec > params.ec_instability_threshold:

        excess = (
            state.ec
            - params.ec_instability_threshold
        )

        phase = (
        state.oscillation_phase
        + 0.25 * dt_scale
    )


        overshoot_effect = (
            params.overshoot_ec_gain
            * excess
            * dt_scale
            +
            params.oscillation_sensitivity
            * excess
            * np.sin(phase)
            * dt_scale
        )

    penalty_effect = 0.0

    if cumulative > params.cumulative_nutrient_limit:

        penalty_effect = 0.0

    print(
    "final ec:",
    state.ec
    )

    print(
        "final assimilation_pool:",
        state.assimilation_pool
    )

    print(
        "final ec_velocity:",
        state.ec_velocity
    )

    return {
        "dose_mass": dose_mass,
        "baseline_drain": baseline_drain * dt_scale,
        "uptake": proportional_uptake * dt_scale,
        "starvation": starvation * dt_scale,
        "depletion": depletion,
        "raw_influx": raw_influx,
        "memory_influx": memory_influx,
        "assimilation_pool": assimilation_pool,
        "restoring": restoring,
        "ec_velocity": ec_velocity,
        "shock_effect": dist.ec_shock * 0.5,
        "mixing_effect": dist.mixing_noise * 0.005,
        "overshoot_effect": overshoot_effect,
        "penalty_effect": penalty_effect,
    }
    
    

def contribution_audit():

    params = TankDynamicsParams()

    state = TankState.create_initial(
        params,
        ec=params.ec_target,
    )

    rows = []

    for step in range(10000):

        flowrate = 0.5
        duration = 10

        terms = reconstruct_ec_terms(
            state,
            flowrate,
            duration,
            60,
            params,
            DisturbanceSpec(),
        )

        next_state = step_dynamics(
            state,
            flowrate,
            duration,
            60,
            params,
        )

        clip_effect = (
            next_state.ec
            -
            (
                state.ec
                + terms["ec_velocity"]
            )
        )

        terms["clip_effect"] = clip_effect

        terms["delta_ec"] = (
            next_state.ec
            - state.ec
        )

        rows.append(terms)

        state = next_state

    df = pd.DataFrame(rows)

    print()
    print("=" * 80)
    print("EC DYNAMICS BALANCE AUDIT")
    print("=" * 80)

    print(
        "baseline depletion:",
        df["baseline_drain"].mean(),
    )

    print(
        "uptake:",
        df["uptake"].mean(),
    )

    print(
        "starvation:",
        df["starvation"].mean(),
    )

    print(
        "raw influx:",
        df["raw_influx"].mean(),
    )

    print(
        "memory influx:",
        df["memory_influx"].mean(),
    )

    print(
        "assimilation pool:",
        df["assimilation_pool"].mean(),
    )

    print(
        "ec velocity:",
        df["ec_velocity"].mean(),
    )

    total_depletion = (
        df["baseline_drain"]
        + df["uptake"]
        + df["starvation"]
    )

    total_influx = (
        df["raw_influx"]
        + df["memory_influx"]
    )

    print()

    print(
        "total depletion:",
        total_depletion.mean(),
    )

    print(
        "total influx:",
        total_influx.mean(),
    )

    print(
        "net influx minus depletion:",
        (
            total_influx
            - total_depletion
        ).mean(),
    )

    print()

    print(
        "mean assimilation - depletion:",
        (
            df["assimilation_pool"]
            - total_depletion
        ).mean(),
    )

    print(
        "mean ec velocity:",
        df["ec_velocity"].mean(),
    )

    print(
        "final ec velocity:",
        df["ec_velocity"].iloc[-1],
    )

    print(
        "mean delta ec:",
        df["delta_ec"].mean(),
    )

    print(
        "final delta ec:",
        df["delta_ec"].iloc[-1],
    )

    print("=" * 80)
    print()

    ranking = []

    for col in [
        "raw_influx",
        "memory_influx",
        "depletion",
        "overshoot_effect",
        "penalty_effect",
        "shock_effect",
        "mixing_effect",
        "clip_effect",
    ]:

        ranking.append(
            {
                "mechanism": col,
                "mean_abs":
                    df[col].abs().mean(),
                "max_abs":
                    df[col].abs().max(),
                "cumulative":
                    df[col].sum(),
            }
        )

    ranking_df = (
        pd.DataFrame(ranking)
        .sort_values(
            "mean_abs",
            ascending=False,
        )
    )

    ranking_df.to_csv(
        REPORT_DIR /
        "contribution_ranking.csv",
        index=False,
    )

    df.to_csv(
        REPORT_DIR /
        "full_contributions.csv",
        index=False,
    )

    return df
    

def saturation_audit():

    params = TankDynamicsParams()

    rows = []

    for ec in [
        0.5,
        1.0,
        1.2,
        1.5,
        2.0,
        3.0,
    ]:

        sat = soft_ec_saturation_factor(
            ec,
            params,
        )

        clipped = _soft_clip_ec(
            ec,
            params,
        )

        rows.append(
            {
                "ec": ec,
                "sat_factor": sat,
                "soft_clip": clipped,
                "saturation_active": sat != 1.0,
                "clipping_active": clipped != ec,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        REPORT_DIR / "saturation_audit.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print("SATURATION AUDIT")
    print("=" * 60)

    print(df.to_string(index=False))

    if not df["saturation_active"].any():
        print()
        print("WARNING: soft_ec_saturation_factor disabled")

    if not df["clipping_active"].any():
        print("WARNING: _soft_clip_ec disabled")

def dead_parameter_audit():

    params = TankDynamicsParams()

    print()
    print("=" * 60)
    print("DEAD PARAMETER AUDIT")
    print("=" * 60)

    dead_params = [
        "ec_soft_limit",
        "ec_hard_limit",
        "nutrient_overshoot_penalty",
    ]

    for p in dead_params:
        print(
            f"{p:35s}",
            getattr(params, p),
        )

    print()
    print(
        "These parameters currently appear unused "
        "in step_dynamics()."
    )

def equilibrium_drift_audit():

    params = TankDynamicsParams()

    state = TankState.create_initial(
        params,
        ec=params.ec_target,
    )

    for _ in range(20000):

        state = step_dynamics(
            state,
            0.5,
            10,
            60,
            params,
        )

    equilibrium_ec = state.ec

    trace = []

    for _ in range(10000):

        state = step_dynamics(
            state,
            0.5,
            10,
            60,
            params,
        )

        trace.append(state.ec)

    trace = np.asarray(trace)

    print()
    print("=" * 60)
    print("EQUILIBRIUM DRIFT AUDIT")
    print("=" * 60)

    print(
        "equilibrium_ec:",
        equilibrium_ec,
    )

    print(
        "tail_mean:",
        trace[-1000:].mean(),
    )

    print(
        "drift:",
        trace[-1000:].mean()
        - equilibrium_ec,
    )

def open_loop_audit():


    params = TankDynamicsParams()

    tests = {
        "zero": (0.0, 0.0),
        "low": (0.25, 10),
        "medium": (0.5, 10),
        "high": (1.0, 10),
    }

    results = []

    for name, action in tests.items():

        state = TankState.create_initial(
            params,
            ec=params.ec_target,
        )

        ec_trace = []

        for _ in range(5000):

            state = step_dynamics(
                state,
                action[0],
                action[1],
                60,
                params,
            )

            ec_trace.append(
                state.ec
            )

        ec_trace = np.array(ec_trace)

        results.append(
            {
                "test": name,
                "steady_state":
                    ec_trace[-500:].mean(),
                "tail_mean":
                    ec_trace[-1000:].mean(),
                "max_ec":
                    ec_trace.max(),
                "min_ec":
                    ec_trace.min(),
            }
        )

    pd.DataFrame(results).to_csv(
        REPORT_DIR /
        "open_loop_results.csv",
        index=False,

    )


def long_horizon_audit():

    params = TankDynamicsParams()

    horizons = [
        1000,
        2500,
        5000,
        10000,
        20000,
    ]

    rows = []

    for horizon in horizons:

        state = TankState.create_initial(
            params,
            ec=params.ec_target,
        )

        trace = []

        for _ in range(horizon):

            state = step_dynamics(
                state,
                0.5,
                10,
                60,
                params,
            )

            trace.append(state.ec)

        trace = np.array(trace)

        rows.append(
            {
                "horizon": horizon,
                "mean_ec":
                    trace.mean(),
                "tail_mean":
                    trace[-500:].mean(),
                "drift":
                    trace[-500:].mean()
                    -
                    trace[:500].mean(),
            }
        )

    pd.DataFrame(rows).to_csv(
        REPORT_DIR /
        "long_horizon.csv",
        index=False,
    )
    

def artifact_summary():

    print()
    print("=" * 60)
    print("ENVIRONMENT ARTIFACT AUDIT")
    print("=" * 60)

    print()
    print("Active EC mechanisms")

    print("- assimilation_pool")
    print("- ec_velocity")
    print("- biological uptake")
    print("- starvation depletion")
    print("- thermal efficiency")
    print("- delayed absorption queue")

    print()
    print("Disabled mechanisms")

    print("- soft_ec_saturation_factor() returns 1.0")
    print("- _soft_clip_ec() returns EC unchanged")
    print("- cumulative nutrient penalty disabled")

    print()
    print("Potential dominant hidden states")

    print("- assimilation_pool")
    print("- ec_velocity")
    print("- nutrient_memory")
    print("- absorption_queue")

def starvation_audit():

    params = TankDynamicsParams()

    rows = []

    for ec in [
        -100.0,
        -50.0,
        -25.0,
        -10.0,
        -5.0,
        -1.0,
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.2,
        1.5,
        2.0,
    ]:

        thermal = 1.0

        starvation = 0.0

        if ec < params.ec_healthy_min:

            deficit = (
                params.ec_healthy_min
                - ec
            )

            starvation = (
                params.starvation_acceleration
                * deficit
                * thermal
            )

        baseline = (
            params.baseline_ec_depletion
            * thermal
        )

        uptake = (
        params.biological_uptake_rate
        * max(ec, 0.0)
        * thermal
        )

        total_depletion = (
            baseline
            + uptake
            + starvation
        )

        rows.append(
            {
                "ec": ec,
                "deficit":
                    max(
                        0.0,
                        params.ec_healthy_min - ec
                    ),
                "baseline":
                    baseline,
                "uptake":
                    uptake,
                "starvation":
                    starvation,
                "total_depletion":
                    total_depletion,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        REPORT_DIR /
        "starvation_audit.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print("STARVATION AUDIT")
    print("=" * 60)

    print(df.to_string(index=False))

def negative_ec_runaway_test():

    params = TankDynamicsParams()

    starting_values = [
        -100.0,
        -50.0,
        -10.0,
        -1.0,
        0.0,
        0.25,
        0.5,
    ]

    rows = []

    for start_ec in starting_values:

        state = TankState.create_initial(
            params,
            ec=start_ec,
        )

        trace = []

        for _ in range(100):

            state = step_dynamics(
                state,
                0.0,
                0.0,
                60,
                params,
            )

            trace.append(state.ec)

        rows.append(
            {
                "start_ec": start_ec,
                "final_ec": state.ec,
                "delta":
                    state.ec - start_ec,
                "min_ec":
                    np.min(trace),
                "max_ec":
                    np.max(trace),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        REPORT_DIR /
        "negative_ec_runaway.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print("NEGATIVE EC RUNAWAY TEST")
    print("=" * 60)

    print(df.to_string(index=False))

def equilibrium_audit():

    params = TankDynamicsParams()

    state = TankState.create_initial(
        params,
        ec=params.ec_target,
    )

    for _ in range(20000):

        state = step_dynamics(
            state,
            0.5,
            10,
            60,
            params,
        )

    terms = reconstruct_ec_terms(
        state,
        0.5,
        10,
        60,
        params,
        DisturbanceSpec(),
    )

    print()
    print("=" * 60)
    print("EQUILIBRIUM AUDIT")
    print("=" * 60)

    print("EC:", state.ec)

    print()

    for k, v in terms.items():

        print(
            f"{k:20s}",
            v,
        )

    print()

    print(
        "assimilation_minus_depletion:",
        terms["assimilation_pool"]
        - terms["depletion"]
    )

    print(
        "ec_velocity:",
        state.ec_velocity
    )

    print(
        "health_index:",
        state.health_index
    )

    print(
        "nutrient_memory:",
        state.nutrient_memory
    )
def velocity_integrator_audit():

    params = TankDynamicsParams()

    state = TankState.create_initial(
        params,
        ec=params.ec_target,
    )

    rows = []

    for _ in range(5000):

        forcing = (
            state.assimilation_pool
            -
            (
                params.baseline_ec_depletion
                +
                params.biological_uptake_rate
                * max(state.ec, 0.0)
            )
        )

        rows.append(
            {
                "ec": state.ec,
                "velocity": state.ec_velocity,
                "forcing": forcing,
            }
        )

        state = step_dynamics(
            state,
            0.5,
            10,
            60,
            params,
        )

    df = pd.DataFrame(rows)

    print()
    print("=" * 60)
    print("VELOCITY INTEGRATOR AUDIT")
    print("=" * 60)

    print(
        "mean forcing:",
        df["forcing"].mean(),
    )

    print(
        "mean velocity:",
        df["velocity"].mean(),
    )

    print(
        "final forcing:",
        df["forcing"].iloc[-1],
    )

    print(
        "final velocity:",
        df["velocity"].iloc[-1],
    )

if __name__ == "__main__":

    artifact_summary()

    contribution_audit()

    saturation_audit()

    dead_parameter_audit()

    open_loop_audit()

    long_horizon_audit()

    starvation_audit()

    negative_ec_runaway_test()

    equilibrium_audit()

    equilibrium_drift_audit()

    velocity_integrator_audit()

    print()
    print("Audit reports saved to:")
    print(REPORT_DIR.resolve())

