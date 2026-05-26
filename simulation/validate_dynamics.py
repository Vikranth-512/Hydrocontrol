"""
Validate v3 simulator: smooth EC, thermal sensitivity, decoupled turbidity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from controllers.pid_controller import PIDController, PIDGains
from controllers.rule_based_controller import RuleBasedController, RuleBasedConfig
from simulation.dynamics import (
    DisturbanceSpec,
    TankDynamicsParams,
    TankState,
    max_ec_step_change,
    simulate_open_loop,
    step_dynamics,
    thermal_efficiency,
)
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig


def _periodic_good_policy(t: int, interval: int = 7) -> Tuple[float, float]:
    return (2.5, 14.0) if t % interval == 0 else (0.0, 0.0)


def _aggressive_policy(t: int) -> Tuple[float, float]:
    return (5.0, 30.0) if t % 4 == 0 else (0.0, 0.0)


def run_validation_suite(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = config.get("simulation", {})
    dyn = config.get("dynamics", {})
    ev = config.get("evaluation", {})
    dt = sim.get("dt_seconds", 60.0)
    length = min(400, ev.get("closed_loop_steps", 400))
    ec_target = sim.get("ec_target", 1.2)
    params = TankDynamicsParams.from_config(dyn, ec_target=ec_target)
    t_axis = np.arange(length) * dt / 60.0
    results: Dict[str, Any] = {}
    env_cfg = EnvironmentConfig(
        dt_seconds=dt,
        ec_target=ec_target,
        ec_safe_min=sim.get("ec_safe_min", 0.4),
        ec_safe_max=sim.get("ec_safe_max", 2.5),
        flowrate_min=0.0,
        flowrate_max=sim.get("flowrate_max", 5.0),
        duration_max=sim.get("duration_max", 30.0),
        min_time_between_doses=sim.get("min_time_between_doses", 120.0),
    )
    pid_cfg = ev.get("pid", {})

    # 1. No control collapse
    hist_none = simulate_open_loop(params, length, dt, actions=[(0.0, 0.0)] * length)
    results["no_control_final_ec"] = float(hist_none["ec"][-1])
    results["no_control_ec_drop"] = float(hist_none["ec"][0] - hist_none["ec"][-1])
    results["max_ec_step_no_control"] = max_ec_step_change(hist_none["ec"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_axis, hist_none["ec"], label="EC")
    axes[0].axhline(ec_target, color="r", linestyle="--")
    axes[0].set_ylabel("EC")
    axes[0].set_title("Without control: depletion (smooth transitions)")
    axes[1].plot(t_axis, hist_none["turbidity"], color="C2", label="Turbidity")
    axes[1].plot(t_axis, hist_none["algae_biomass"], "--", alpha=0.6, label="Biomass (internal)")
    axes[1].set_ylabel("NTU")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_no_control_collapse.png", dpi=150)
    plt.close(fig)

    # 2. Good control — PID closed-loop equilibrium
    s0 = TankState.create_initial(params, ec=ec_target)
    env_good = AlgaeTankEnvironment(env_cfg, params)
    env_good.reset(initial_state=s0)
    pid_good = PIDController(
        setpoint=ec_target,
        gains=PIDGains(
            kp=pid_cfg.get("kp", 2.5),
            ki=pid_cfg.get("ki", 0.08),
            kd=pid_cfg.get("kd", 0.4),
        ),
        flowrate_max=env_cfg.flowrate_max,
        duration_max=env_cfg.duration_max,
        dt=dt,
    )
    ec_good, turb_good, mem_good, fr_good = [], [], [], []
    for _ in range(length):
        ec_val = env_good.state.ec if env_good.state else ec_target
        fr, dur = pid_good.compute(ec_val)
        ec_good.append(ec_val)
        turb_good.append(env_good.state.turbidity if env_good.state else 0.0)
        mem_good.append(env_good.state.biomass_memory if env_good.state else 0.0)
        fr_good.append(fr)
        env_good.step((fr, dur))
    hist_good = {
        "ec": np.array(ec_good),
        "turbidity": np.array(turb_good),
        "biomass_memory": np.array(mem_good),
        "flowrate": np.array(fr_good),
    }
    results["good_control_ec_mean"] = float(np.mean(hist_good["ec"][-80:]))
    results["max_ec_step_good_control"] = max_ec_step_change(hist_good["ec"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_axis, hist_good["ec"])
    axes[0].axhline(ec_target, color="r", linestyle="--")
    axes[0].set_ylabel("EC")
    axes[0].set_title("PID closed-loop: regulated equilibrium")
    axes[1].plot(t_axis, hist_good["turbidity"], label="Turbidity")
    axes[1].plot(t_axis, hist_good["biomass_memory"], "--", label="Biomass memory")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "02_good_control_equilibrium.png", dpi=150)
    plt.close(fig)

    # 3. Aggressive dosing — mild overshoot (not vertical saturation)
    hist_bad = simulate_open_loop(
        params, length, dt, actions=[_aggressive_policy(t) for t in range(length)]
    )
    results["bad_control_overshoot"] = float(np.max(hist_bad["ec"] - ec_target))
    results["max_ec_step_aggressive"] = max_ec_step_change(hist_bad["ec"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, hist_bad["ec"])
    ax.axhline(ec_target, color="r", linestyle="--")
    ax.axhline(params.ec_soft_limit, color="orange", linestyle=":", label="Soft limit")
    ax.set_title("Aggressive dosing: damped overshoot (not hard jump to ceiling)")
    ax.set_xlabel("Time (min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_bad_control_overshoot.png", dpi=150)
    plt.close(fig)

    # 4. Delayed absorption
    s = TankState.create_initial(params, ec=ec_target)
    impulse_ec, pending = [], []
    for t in range(50):
        fr, dur = (3.0, 15.0) if t == 0 else (0.0, 0.0)
        s = step_dynamics(s, fr, dur, dt, params)
        impulse_ec.append(s.ec)
        pending.append(float(np.sum(s.absorption_queue)))
    results["max_ec_step_impulse"] = max_ec_step_change(np.array(impulse_ec))

    tt = np.arange(len(impulse_ec)) * dt / 60.0
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(tt, impulse_ec, "o-", markersize=3)
    ax1.set_ylabel("EC")
    ax2 = ax1.twinx()
    ax2.plot(tt, pending, "s--", color="C1", alpha=0.8)
    ax2.set_ylabel("Pending mass")
    ax1.set_title("Delayed absorption: gradual EC ramp")
    ax1.set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_delayed_absorption.png", dpi=150)
    plt.close(fig)

    # 5. Temperature strongly affects depletion rate
    temps = np.linspace(14, 33, 15)
    ec_end, turb_end, thermal_eff = [], [], []
    ec_mid_by_temp = []
    for temp in temps:
        s = TankState.create_initial(params, ec=ec_target, water_temp=temp)
        h = simulate_open_loop(params, 150, dt, actions=[(0.0, 0.0)] * 150, initial_state=s)
        ec_end.append(h["ec"][-1])
        ec_mid_by_temp.append(h["ec"][60] if len(h["ec"]) > 60 else h["ec"][-1])
        turb_end.append(h["turbidity"][-1])
        thermal_eff.append(thermal_efficiency(temp, params))

    results["temp_ec_range"] = float(max(ec_mid_by_temp) - min(ec_mid_by_temp))
    results["temp_ec_final_range"] = float(max(ec_end) - min(ec_end))
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(temps, ec_end, "o-", label="Final EC (no control)")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Final EC")
    ax2 = ax1.twinx()
    ax2.plot(temps, turb_end, "s-", color="C2", label="Final turbidity")
    ax2.set_ylabel("Final turbidity")
    ax1.set_title("Temperature shapes depletion & biomass trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "05_temperature_coupling.png", dpi=150)
    plt.close(fig)

    # 6. Same PID at different temperatures → different closed-loop outcomes
    def run_temp_pid(temp: float) -> float:
        s0 = TankState.create_initial(params, ec=ec_target, water_temp=temp)
        env = AlgaeTankEnvironment(env_cfg, params)
        env.reset(initial_state=s0)
        pid = PIDController(
            setpoint=ec_target,
            gains=PIDGains(
                kp=pid_cfg.get("kp", 2.5),
                ki=pid_cfg.get("ki", 0.08),
                kd=pid_cfg.get("kd", 0.4),
            ),
            flowrate_max=env_cfg.flowrate_max,
            duration_max=env_cfg.duration_max,
            dt=dt,
        )
        trace = []
        for _ in range(200):
            ec_val = env.state.ec if env.state else ec_target
            trace.append(ec_val)
            env.step(pid.compute(ec_val))
        return float(np.mean(trace[-40:]))

    def run_temp_pid_trace(temp: float) -> np.ndarray:
        s0 = TankState.create_initial(params, ec=ec_target, water_temp=temp)
        env = AlgaeTankEnvironment(env_cfg, params)
        env.reset(initial_state=s0)
        pid = PIDController(
            setpoint=ec_target,
            gains=PIDGains(
                kp=pid_cfg.get("kp", 2.5),
                ki=pid_cfg.get("ki", 0.08),
                kd=pid_cfg.get("kd", 0.4),
            ),
            flowrate_max=env_cfg.flowrate_max,
            duration_max=env_cfg.duration_max,
            dt=dt,
        )
        trace = []
        for _ in range(200):
            ec_val = env.state.ec if env.state else ec_target
            trace.append(ec_val)
            env.step(pid.compute(ec_val))
        return np.array(trace)

    trace_cold = run_temp_pid_trace(17.0)
    trace_hot = run_temp_pid_trace(30.0)
    err_cold = float(np.mean(np.abs(trace_cold - ec_target)))
    err_hot = float(np.mean(np.abs(trace_hot - ec_target)))
    results["temp_policy_ec_cold"] = float(np.mean(trace_cold[-40:]))
    results["temp_policy_ec_hot"] = float(np.mean(trace_hot[-40:]))
    results["temp_policy_spread"] = abs(err_hot - err_cold)

    # 7. Turbidity partially independent (sediment shock)
    disturbances = [DisturbanceSpec(turbidity_shock=45.0 if t == 80 else 0.0) for t in range(length)]
    hist_sed = simulate_open_loop(
        params, length, dt, actions=[(0.0, 0.0)] * length, disturbances=disturbances
    )
    ec_delta_at_shock = abs(hist_sed["ec"][81] - hist_sed["ec"][79])
    turb_delta_at_shock = abs(hist_sed["turbidity"][81] - hist_sed["turbidity"][79])
    results["sediment_ec_delta"] = float(ec_delta_at_shock)
    results["sediment_turbidity_delta"] = float(turb_delta_at_shock)
    lag_corr = float(np.corrcoef(hist_sed["ec"], hist_sed["turbidity"])[0, 1])
    results["ec_turbidity_correlation"] = lag_corr

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_axis, hist_sed["ec"], label="EC")
    axes[0].axvline(80 * dt / 60, color="gray", linestyle=":", label="Sediment event")
    axes[0].set_ylabel("EC")
    axes[1].plot(t_axis, hist_sed["turbidity"], color="C2", label="Turbidity")
    axes[1].set_ylabel("Turbidity")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend()
    axes[0].set_title("Sediment spike: turbidity moves, EC largely unaffected")
    fig.tight_layout()
    fig.savefig(output_dir / "07_turbidity_independence.png", dpi=150)
    plt.close(fig)

    # 8. Closed-loop controllers
    def closed_loop(name: str, water_temp: float = 22.0) -> np.ndarray:
        s0 = TankState.create_initial(params, ec=ec_target, water_temp=water_temp)
        env = AlgaeTankEnvironment(env_cfg, params)
        env.reset(initial_state=s0)
        trace = []
        pid = PIDController(
            setpoint=ec_target,
            gains=PIDGains(
                kp=pid_cfg.get("kp", 2.5),
                ki=pid_cfg.get("ki", 0.08),
                kd=pid_cfg.get("kd", 0.4),
            ),
            flowrate_max=env_cfg.flowrate_max,
            duration_max=env_cfg.duration_max,
            dt=dt,
        )
        rb = RuleBasedController(RuleBasedConfig(ec_target=ec_target))
        for t in range(length):
            ec_val = env.state.ec if env.state else ec_target
            if name == "none":
                act = (0.0, 0.0)
            elif name == "pid":
                act = pid.compute(ec_val)
            else:
                act = rb.compute(ec_val)
            trace.append(ec_val)
            env.step(act)
        return np.array(trace)

    ec_none = closed_loop("none")
    ec_pid = closed_loop("pid")
    ec_pid_hot = closed_loop("pid", water_temp=30.0)
    ec_pid_cold = closed_loop("pid", water_temp=17.0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_axis, ec_none, label="No control")
    ax.plot(t_axis, ec_pid, label="PID (22°C)")
    ax.plot(t_axis, ec_pid_hot, label="PID (30°C)", alpha=0.8)
    ax.plot(t_axis, ec_pid_cold, label="PID (17°C)", alpha=0.8)
    ax.axhline(ec_target, color="k", linestyle="--")
    ax.legend()
    ax.set_title("Controller behavior varies with temperature")
    fig.tight_layout()
    fig.savefig(output_dir / "06_closed_loop_controllers.png", dpi=150)
    plt.close(fig)

    results["closed_loop_final_ec"] = {
        "none": float(ec_none[-1]),
        "pid_22C": float(ec_pid[-1]),
        "pid_30C": float(ec_pid_hot[-1]),
        "pid_17C": float(ec_pid_cold[-1]),
    }

    # Pass criteria (v3)
    results["validation_passed"] = (
        results["no_control_ec_drop"] > 0.12
        and results["no_control_final_ec"] < params.ec_healthy_min
        and results["good_control_ec_mean"] > 0.35
        and results["good_control_ec_mean"] > results["no_control_final_ec"] + 0.2
        and results["max_ec_step_aggressive"] < 0.45
        and results["max_ec_step_impulse"] < 0.35
        and results["temp_ec_range"] > 0.08
        and results["temp_policy_spread"] > 0.01
        and results["sediment_turbidity_delta"] > 1.0
        and results["sediment_ec_delta"] < 0.15
        and abs(results["ec_turbidity_correlation"]) < 0.98
    )

    with open(output_dir / "validation_summary.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results
