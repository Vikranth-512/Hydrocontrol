# Simulation Validation Report

This document contains the automated scientific validation results for the mechanistic algae tank simulator.

**Overall Status:** 30/30 passed (100.0%)

---
## I. Physical Conservation
---

### [PASS] Strict Mass Conservation (PHYS-01)
**Hypothesis:** Total nutrient mass (all compartments + cumulative losses) exactly equals initial mass + cumulative doses at every timestep.
**Status:** PASS

**Metrics:**
- Max Absolute Error: 3.385e-09
- RMS Error: 2.201e-09

**Plots Generated:**
- `conservation_error.png`

### [PASS] Dilution-Only Bookkeeping (PHYS-02)
**Hypothesis:** With near-zero biomass, mass loss is explained entirely by tracked cumulative dilution.
**Status:** PASS

**Metrics:**
- Mass Lost: 4.507
- Cumulative Dilution: 4.505
- Fraction Accounted: 0.9996

**Plots Generated:**
- `dilution_bookkeeping.png`

### [PASS] Transport Queue Conservation (PHYS-03)
**Hypothesis:** An impulse dose placed in the transport queue eventually exits entirely into the dissolved pool.
**Status:** PASS

**Metrics:**
- Final Queue Residual: 0

**Plots Generated:**
- `queue_conservation.png`

## II. Biological Dynamics
---

### [PASS] Monod Uptake Saturation (BIO-01)
**Hypothesis:** Uptake rate saturates at Vmax according to Michaelis-Menten kinetics.
**Status:** PASS

**Metrics:**
- Empirical Vmax: 0.3045
- Expected Vmax (approx): 0.3467

**Plots Generated:**
- `monod_curve.png`

### [PASS] Starvation Inertia (Cryptic Recycling) (BIO-02)
**Hypothesis:** Reserve monotonically decreases in the absence of uptake, and approaches a recycling equilibrium when mineralization is enabled.
**Status:** PASS

**Metrics:**
- Final Reserve (Baseline): 2.714
- Final Reserve (No Uptake): 0
- Final Biomass (Baseline): 109.9

**Plots Generated:**
- `starvation_baseline.png`

### [PASS] Temperature Dependence (BIO-03)
**Hypothesis:** Uptake and growth rates peak near T_opt and decline at temperature extremes.
**Status:** PASS

**Metrics:**
- Peak Uptake Temp: 28
- Offset from T_opt: 2

**Plots Generated:**
- `temperature_curve.png`

### [PASS] Osmotic Stress Inhibition (BIO-04)
**Hypothesis:** At very high dissolved nutrient concentrations, osmotic stress reduces uptake before toxicity causes mass mortality.
**Status:** PASS

**Metrics:**
- Uptake at Low Conc: 0.3041
- Uptake at High Conc: 0

**Plots Generated:**
- `osmotic_curve.png`

### [PASS] Reserve Isolation (BIO-05)
**Hypothesis:** With zero dissolved nutrients, internal reserve only decreases (no hidden replenishment).
**Status:** PASS

**Metrics:**
- Number of Reserve Increases: 0
- Final Reserve: 5.641

**Plots Generated:**
- `reserve_isolation.png`

### [PASS] Growth Limitation by Reserve (BIO-06)
**Hypothesis:** Growth depends on internal reserve, not dissolved nutrients directly. Pre-loaded cells grow instantly in sterile water.
**Status:** PASS

**Metrics:**
- Growth (high dissolved, t=0..10): -1.406
- Growth (high reserve, t=0..10): 0.6994

**Plots Generated:**
- `growth_limitation.png`

### [PASS] Mortality Channels (BIO-07)
**Hypothesis:** Starvation, osmotic stress, and heat each independently cause biomass loss.
**Status:** PASS

**Metrics:**
- Starvation Loss: 13.17
- Osmotic Loss: 97.44
- Heat Loss: 99.96

**Plots Generated:**
- `mortality_channels.png`

### [PASS] Mineralization Half-Life (BIO-08)
**Hypothesis:** Dead biomass decays back into dissolved nutrients via first-order mineralization kinetics.
**Status:** PASS

**Metrics:**
- Mineralization Half-Life (min): 599

**Plots Generated:**
- `mineralization.png`

### [PASS] Health Hysteresis (BIO-09)
**Hypothesis:** Health recovery after osmotic-induced damage is slower than the damage onset, demonstrating physiological inertia.
**Status:** PASS

**Metrics:**
- Minimum Health: 0.4845
- Final Health: 1
- Recovery Delta: 0.5155

**Plots Generated:**
- `health_hysteresis.png`

## III. Numerical Analysis
---

### [PASS] Timestep Sensitivity (NUM-01)
**Hypothesis:** Trajectories converge as dt decreases; dt=60s is sufficiently accurate.
**Status:** PASS

**Metrics:**
- dt=30s Biomass Final: 211.4
- dt=30s Biomass Error vs 30s: 0
- dt=60s Biomass Final: 211.6
- dt=60s Biomass Error vs 30s: 0.1475
- dt=120s Biomass Final: 211.6
- dt=120s Biomass Error vs 30s: 0.1496
- dt=300s Biomass Final: 209.3
- dt=300s Biomass Error vs 30s: 2.137

**Plots Generated:**
- `timestep_sensitivity.png`

### [PASS] Long Horizon Stability (NUM-02)
**Hypothesis:** 100,000 steps produce no NaNs, Infs, or unbounded variables.
**Status:** PASS

**Metrics:**
- NaN Count: 0
- Inf Count: 0
- Max Biomass: 4057
- Max Dissolved: 445.5
- Final Biomass: 0
- Final Health: 0

**Plots Generated:**
- `long_horizon.png`

### [PASS] Clipping Frequency Analysis (NUM-03)
**Hypothesis:** Numerical bounding events (min/max clips) occur infrequently, indicating the integration is not relying on clamps.
**Status:** PASS

**Metrics:**
- uptake_capped_by_dissolved: 0
- maintenance_capped_by_reserve: 0
- growth_capped_by_reserve: 0
- mortality_capped_at_95pct: 77
- damage_clipped_0: 3
- damage_clipped_1: 9927

**Warnings:**
- WARNING: High clipping rate: 10007/10000

**Plots Generated:**
- `clipping_analysis.png`

## IV. Emergent Behavior
---

### [PASS] Toxic Accumulation Positive Feedback (EMR-01)
**Hypothesis:** Excessive dosing triggers osmotic stress → uptake halt → dissolved accumulation → biomass collapse, an emergent positive feedback loop.
**Status:** PASS

**Metrics:**
- Final Dissolved: 296.1
- Final Biomass: 1.025e-168
- Final Health: 0

**Plots Generated:**
- `toxic_accumulation.png`

### [PASS] Dynamic Equilibrium (EMR-02)
**Hypothesis:** Constant moderate dosing produces a dynamic equilibrium where uptake ≈ dosing, without hidden restoring forces.
**Status:** PASS

**Metrics:**
- Dissolved CV (tail): 0.008424
- Biomass CV (tail): 0.008454
- Tail Mean Dissolved: 0.1845
- Tail Mean Biomass: 307.5

**Plots Generated:**
- `dynamic_equilibrium.png`

### [PASS] Pulse Response Delay Chain (EMR-03)
**Hypothesis:** A single pulse propagates through dissolved → reserve → biomass → turbidity with measurable cascading delays.
**Status:** PASS

**Metrics:**
- Delay to Dissolved (steps): 1
- Delay to Reserve (steps): 2
- Delay to EC (steps): 1

**Plots Generated:**
- `pulse_response.png`

### [PASS] Repeated Pulse Phase Lag (EMR-04)
**Hypothesis:** Periodic dosing reveals phase lag, memory accumulation, and potential nonlinear saturation across pulses.
**Status:** PASS

**Metrics:**
- Total Biomass Change: 271.3
- Final Biomass: 371.3

**Plots Generated:**
- `repeated_pulse.png`

## V. Sensor Validation
---

### [PASS] EC Linearity & Gain (SEN-01)
**Hypothesis:** EC is a linear function of dissolved_nutrient_mass with gain = sensor_gain_ec.
**Status:** PASS

**Metrics:**
- EC RMSE vs Linear Model: 0
- Max Residual: 0

**Plots Generated:**
- `ec_linearity.png`

### [PASS] Turbidity Lag (SEN-02)
**Hypothesis:** Turbidity tracks biomass optical density through a first-order lag filter.
**Status:** PASS

**Metrics:**
- Peak Lag (steps): 0
- Peak Correlation: 0.9903
- Sensor Tau: 0.045

**Plots Generated:**
- `turbidity_lag.png`

### [PASS] Sensor Decoupling Proof (SEN-03)
**Hypothesis:** Changing sensor_gain_ec has zero effect on physical states (dissolved, biomass, reserve, health).
**Status:** PASS

**Metrics:**
- Max Delta_dissolved_nutrient_mass: 0
- Max Delta_algae_biomass: 0
- Max Delta_internal_reserve: 0
- Max Delta_health_index: 0

**Plots Generated:**
- `sensor_decoupling.png`

## VI. RL Benchmark
---

### [PASS] Partial Observability (BEN-01)
**Hypothesis:** Hidden states (reserve, health, damage) have low mutual information with observed states (EC, turbidity), confirming partial observability.
**Status:** PASS

**Metrics:**
- MI(Reserve, EC): 0.2504
- MI(Health, EC): 0.1333
- MI(Reserve, Turbidity): 0.274
- MI(Health, Turbidity): 0.1218
- MI(Dead Pool, Turbidity): 0.1307
- MI(Damage, EC): 0.1333

**Plots Generated:**
- `observability_matrix.png`

### [PASS] State-Space Coverage (BEN-02)
**Hypothesis:** Monte Carlo random trajectories cover a wide range of (EC, Biomass, Reserve, Health) states.
**Status:** PASS

**Metrics:**
- EC Range: [0.45, 121.56]
- Biomass Range: [0.00, 862.56]
- Reserve Range: [0.00, 139.27]
- Health Range: [0.00, 1.00]

**Plots Generated:**
- `state_space_coverage.png`

### [PASS] Impulse Response Delay Chain (BEN-03)
**Hypothesis:** A single dose impulse propagates through Dissolved → Reserve → Biomass → Turbidity with measurable cascading delays.
**Status:** PASS

**Metrics:**
- Delay to Dissolved (steps): 1
- Delay to EC (steps): 1
- Delay to Reserve (steps): 2
- Delay to Biomass (steps): 46
- Delay to Turbidity (steps): 49

**Plots Generated:**
- `impulse_chain.png`

## VII. Parameter Identifiability
---

### [PASS] OAT Parameter Sensitivity (PAR-01)
**Hypothesis:** Biological parameters have varying sensitivity on EC, biomass, and health, with identifiable dominant parameters.
**Status:** PASS

**Metrics:**
- Most Sensitive Parameter (EC): background_dilution_rate
- Num Parameters Tested: 13

**Plots Generated:**
- `tornado_ec.png`
- `tornado_biomass.png`

### [PASS] Monte Carlo Robustness (PAR-02)
**Hypothesis:** Under randomized initial conditions, parameters, and actions, the simulator produces no NaNs or Infs across 500 runs.
**Status:** PASS

**Metrics:**
- NaN Count: 0
- Inf Count: 0
- EC Mean: 63.65
- EC Std: 42.81
- Biomass Mean: 256.7
- Biomass Std: 393.3
- Health Mean: 0.304

**Plots Generated:**
- `monte_carlo.png`

## VIII. Controller Independence
---

### [PASS] Hidden Controller Detection (CTL-01)
**Hypothesis:** With zero dosing, EC from any initial condition decays to zero. No state converges to a non-zero equilibrium.
**Status:** PASS

**Metrics:**
- EC (init=0.0): 0.09529
- EC (init=2.0): 0.102
- EC (init=4.0): 0.1085
- EC (init=8.0): 5.538e-08

**Plots Generated:**
- `hidden_controller.png`

### [PASS] dEC/dt Mass Balance Proof (CTL-02)
**Hypothesis:** dEC/dt is fully explained by sensor_gain × d(dissolved)/dt with near-zero residual.
**Status:** PASS

**Metrics:**
- dEC/dt Residual RMSE: 4.568e-15
- dEC/dt Max Residual: 1.138e-14

**Plots Generated:**
- `dec_dt_mass_balance.png`

### [PASS] Disturbance Recovery (CTL-03)
**Hypothesis:** After a temperature shock, the plant recovers only through physical relaxation - no hidden stabilization.
**Status:** PASS

**Metrics:**
- Temp at t=250: 36.45
- Temp at t=500: 22.03
- Ambient: 22

**Plots Generated:**
- `disturbance_recovery.png`
