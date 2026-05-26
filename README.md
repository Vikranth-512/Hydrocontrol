Learned Nutrient Dosing Control for Algae Cultivation

A research-oriented control system for autonomous algae nutrient regulation using nonlinear ecosystem simulation, optimization-based labeling, and temporal deep learning.

The project models real-world aquatic dynamics including delayed nutrient absorption, biological uptake, thermal effects, turbidity evolution, and environmental disturbances. A tuned classical PID controller and an LSTM-based learned policy are evaluated in closed loop to study the tradeoff between conservative stability-focused control and adaptive precision regulation.

The pipeline includes:

nonlinear ecosystem simulation,
synthetic trajectory generation,
optimization-generated control labels,
LSTM sequence modeling,
PID auto-tuning,
disturbance robustness testing,
and publication-style evaluation/visualization.

The goal is not simply to outperform classical control, but to characterize when learned controllers become advantageous in delayed, nonlinear, multi-regime ecological systems.

This is **not** a forecasting model. It learns a control policy using optimization-generated pseudo-labels and LSTM sequence modeling.

## Simulator (v3 — Smooth EC, Strong Thermal, Decoupled Turbidity)

**v3 refinements** over active-equilibrium v2:
- **EC**: soft saturation, assimilation lag, capped queue release, second-order damping (no vertical jumps to ceiling)
- **Temperature**: Gaussian optimal zone × Q10; materially changes depletion and control difficulty
- **Turbidity**: biomass memory, lagged growth drive, sediment shocks independent of EC

## Simulator (v2 — Active Equilibrium)

The tank is an **actively regulated** nonlinear system:

- **Without dosing**: EC depletes continuously; turbidity declines (starvation).
- **With correct dosing**: EC tracks target; algae/turbidity stabilizes via logistic growth.
- **With poor dosing**: delayed overshoot, oscillations, nutrient waste.

Key mechanisms: delayed absorption queue, Q10 temperature metabolism, EC-coupled turbidity, biological inertia, instability above `ec_instability_threshold`.

Validate behavior:

```bash
python main.py --config configs/default.yaml --stage validate_dynamics
```

## PID tuning (systematic baseline optimization)

```bash
# Full tuning (~72 coarse + refinement + validation)
python main.py --config configs/default.yaml --stage tune_pid

# Quick smoke test
python main.py --config configs/pid_tune_quick.yaml --stage tune_pid
```

Outputs: `data/processed/pid_tuning_results.json`, `figures/pid_tuning/`, `data/processed/pid_tuned_gains.yaml`

The enhanced PID includes anti-windup, derivative filtering, deadband, rate limiting, and output saturation. Tuning uses a weighted composite score over EC MAE, overshoot, oscillation, nutrient use, collapse fraction, and control smoothness.

Figures are written to `figures/dynamics_validation/`.

## Architecture

```
simulation/          → tank dynamics, synthetic trajectories, optimization labels
preprocessing/       → features, sliding windows, normalization
models/              → LSTM policy (PyTorch)
controllers/         → PID and rule-based baselines
training/            → losses, training loop, metrics
simulation_runner/   → closed-loop evaluation
visualization/       → publication plots
configs/             → YAML experiment config
main.py              → end-to-end orchestration
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml --stage all
```

### Individual stages

```bash
python main.py --stage generate    # synthetic CSV trajectories
python main.py --stage label       # optimization-based labels
python main.py --stage preprocess  # features + sequences + scalers
python main.py --stage train       # LSTM policy training
python main.py --stage evaluate    # offline + closed-loop benchmarks
python main.py --stage export      # TorchScript / ONNX export
```

## Outputs

| Path | Description |
|------|-------------|
| `data/synthetic/` | Raw trajectory CSVs |
| `data/processed/` | Labeled data, sequences, scalers |
| `checkpoints/` | Best model weights, export artifacts |
| `figures/` | Training curves, EC trajectories, controller comparisons |
| `data/processed/evaluation_results.json` | Metrics summary |

## Control Policy

- **Input**: Window of engineered sensor features `(seq_len, n_features)`
- **Output**: `(flowrate, duration)`
- **Labels**: Short-horizon constrained search minimizing EC error, nutrient cost, instability, overshoot

## Reproducibility

- Fixed seeds in `configs/default.yaml`
- Trajectory-level train/val/test splits (no window leakage)
- Train-only scaler fitting
- Experiment metadata JSON per run

## Citation / Research

Designed for publication experiments: modular configs, baseline controllers, robustness scenarios, and closed-loop metrics (overshoot, settling time, nutrient efficiency).
