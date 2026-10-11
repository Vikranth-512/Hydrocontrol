# Hydroponic Nutrient-Regulation Benchmark Control Audit

This report presents an audit of the closed-loop nutrient-regulation benchmark and its 3,500-step multi-seed evaluation pipeline (4 controllers x 4 scenarios x 30 seeds = 480 rollouts).

The primary objective is to resolve three core questions:
1. **How often does each controller fail or collapse?**
2. **Why does the reserve-ratio rule controller achieve high biomass with relatively low dosing, and does that reveal a benchmark problem?**
3. **What additional scenario and disturbance regimes are implemented in the repository beyond the four currently evaluated?**

---

## 1. Failure and Collapse Probability

### 1.1 Definitions and Statistical Unit
- **Statistical Unit**: The episode (N = 30 per scenario-controller cell, N = 120 per aggregate controller). Timesteps are never treated as independent Bernoulli trials.
- **Episode Failure (P_failure)**: Early termination triggered by simulator safety or extinction boundaries (B_t <= 0 or H_t <= 0).
- **Biomass Collapse (P_collapse)**: Defined as terminal biomass B_final < 50.0 g/L, matching the established benchmark collapse threshold in `simulation/labeling_objective.py` (line 440: *"complete biomass collapse"*). Secondary check: B_final < 20.0 g/L (`biomass_persistence` threshold from `training/evaluation.py`).
- **Confidence Intervals**: 95% Wilson score binomial proportion confidence intervals bounded in [0.0, 1.0].

### 1.2 Scenario-Specific Breakdown (N = 30 seeds per cell)

| Scenario | Controller | N | Failures | P(Failure) [95% CI] | Collapses (B<50) | P(Collapse) [95% CI] | Mean B (g/L) | Median B (g/L) | IQR B (g/L) | Min B | Max B | Early Term |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| normal | no_action | 30 | 0 | 0.000 [0.000, 0.114] | 30 | 1.000 [0.886, 1.000] | 6.21 | 6.18 | 0.35 | 5.82 | 6.61 | 0 |
| normal | rule_based_reserve | 30 | 0 | 0.000 [0.000, 0.114] | 6 | 0.200 [0.095, 0.373] | 222.69 | 277.84 | 10.59 | 0.97 | 282.86 | 0 |
| normal | pid | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 216.97 | 216.69 | 3.70 | 211.66 | 224.40 | 0 |
| normal | lstm | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 135.01 | 125.77 | 47.45 | 68.43 | 199.66 | 0 |
| heatwave | no_action | 30 | 0 | 0.000 [0.000, 0.114] | 30 | 1.000 [0.886, 1.000] | 6.16 | 6.12 | 0.32 | 5.84 | 6.49 | 0 |
| heatwave | rule_based_reserve | 30 | 0 | 0.000 [0.000, 0.114] | 6 | 0.200 [0.095, 0.373] | 221.35 | 275.90 | 20.99 | 0.95 | 287.08 | 0 |
| heatwave | pid | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 216.29 | 216.11 | 3.63 | 210.74 | 224.77 | 0 |
| heatwave | lstm | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 132.21 | 134.02 | 35.50 | 76.95 | 215.74 | 0 |
| nutrient_depletion | no_action | 30 | 0 | 0.000 [0.000, 0.114] | 30 | 1.000 [0.886, 1.000] | 6.21 | 6.23 | 0.22 | 5.84 | 6.51 | 0 |
| nutrient_depletion | rule_based_reserve | 30 | 0 | 0.000 [0.000, 0.114] | 3 | 0.100 [0.035, 0.256] | 248.18 | 277.10 | 16.69 | 0.95 | 288.18 | 0 |
| nutrient_depletion | pid | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 216.25 | 216.01 | 4.83 | 209.85 | 221.92 | 0 |
| nutrient_depletion | lstm | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 140.62 | 138.20 | 47.98 | 81.05 | 232.86 | 0 |
| cold_shock | no_action | 30 | 0 | 0.000 [0.000, 0.114] | 30 | 1.000 [0.886, 1.000] | 6.28 | 6.31 | 0.42 | 5.80 | 6.75 | 0 |
| cold_shock | rule_based_reserve | 30 | 0 | 0.000 [0.000, 0.114] | 21 | 0.700 [0.521, 0.833] | 81.77 | 1.03 | 255.46 | 0.90 | 284.70 | 0 |
| cold_shock | pid | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 216.04 | 216.52 | 3.68 | 209.65 | 221.09 | 0 |
| cold_shock | lstm | 30 | 0 | 0.000 [0.000, 0.114] | 0 | 0.000 [0.000, 0.114] | 147.02 | 144.33 | 28.39 | 98.11 | 201.92 | 0 |

### 1.3 Aggregate Controller Results (N = 120 episodes per controller)

| Controller | N_total | Failures | P(Failure) [95% CI] | Collapses (B<50) | P(Collapse) [95% CI] | Mean B (g/L) | Median B (g/L) | IQR B (g/L) | Min B | Max B | Early Term |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| no_action | 120 | 0 | 0.000 [0.000, 0.031] | 120 | 1.000 [0.969, 1.000] | 6.21 | 6.20 | 0.34 | 5.80 | 6.75 | 0 |
| rule_based_reserve | 120 | 0 | 0.000 [0.000, 0.031] | 36 | 0.300 [0.225, 0.387] | 193.50 | 274.04 | 278.77 | 0.90 | 288.18 | 0 |
| pid | 120 | 0 | 0.000 [0.000, 0.031] | 0 | 0.000 [0.000, 0.031] | 216.39 | 216.27 | 3.43 | 209.65 | 224.77 | 0 |
| lstm | 120 | 0 | 0.000 [0.000, 0.031] | 0 | 0.000 [0.000, 0.031] | 138.72 | 137.21 | 41.42 | 68.43 | 232.86 | 0 |

### 1.4 Empirical Failure Analysis Findings
1. **Zero Early Simulator Terminations**: No controller experienced early simulator termination (P_failure = 0.0 [0.000, 0.031] across all 480 episodes). Every trajectory completed its full 3,500-step nominal horizon.
2. **Extreme Fragility and Bimodality in Reserve-Ratio Rule**:
   - In aggregate, `rule_based_reserve` suffers a **30.0% collapse probability** (36 / 120 episodes, 95% Wilson CI: [0.225, 0.387]).
   - In `cold_shock`, `rule_based_reserve` collapses in **70.0% of episodes** (21 / 30, 95% CI: [0.521, 0.833]), with a median terminal biomass of only **1.03 g/L** and an IQR of **255.46 g/L**.
   - In `normal`, it collapses in 6 / 30 episodes (20.0%).
   - In `heatwave`, it collapses in 6 / 30 episodes (20.0%).
   - In `nutrient_depletion`, it collapses in 3 / 30 episodes (10.0%).
   - Terminal biomass for `rule_based_reserve` spans from 0.90 g/L to 288.18 g/L: when it succeeds, it reaches ~278 g/L, but in nearly one-third of all rollouts it collapses completely to < 1.0 g/L.
3. **Zero Collapses in PID and LSTM**:
   - Both `pid` and `lstm` achieved **0 collapses across all 120 episodes** (P_collapse = 0.0 [0.000, 0.031]).
   - `pid` maintains extremely tight biomass regulation (mean 216.39 g/L, median 216.27 g/L, IQR 3.43 g/L).
   - `lstm` sustains safe biological survival across all seeds (mean 138.72 g/L, median 137.21 g/L, IQR 41.42 g/L, minimum 68.43 g/L).
4. **Open-Loop Extinction**: `no_action` experiences **100% biomass collapse** (120 / 120, median 6.20 g/L).

---

## 2. Reserve-Rule Mechanistic Audit

### 2.1 Nutrient Flow and Mass Balance Invariant
The simulator (`simulation/dynamics.py`) enforces physical nutrient conservation across 6 discrete compartments:
$$\\text{Total Mass} = M_{\\text{dissolved}} + M_{\\text{reserve}} + M_{\\text{biomass}} + M_{\\text{detritus}} + M_{\\text{pending}} + M_{\\text{loss}}$$
where:
- $M_{\\text{dissolved}} = \\text{dissolved\\_nutrient\\_mass}$
- $M_{\\text{reserve}} = \\text{internal\\_reserve}$
- $M_{\\text{biomass}} = B \\times \\text{biomass\\_nutrient\\_content} = 0.10 \\times B$
- $M_{\\text{detritus}} = \\text{dead\\_biomass\\_pool} \\times 0.10$
- $M_{\\text{pending}} = \\sum \\text{pending\\_doses}$
- $M_{\\text{loss}} = \\text{cumulative\\_dilution} + \\text{harvest\\_loss} + \\text{unretention\\_losses}$

#### Quantitative Trajectory Audit Across Representative Episodes:

| Metric | Normal (Seed 42: Success) | Normal (Seed 43: Collapse) | Cold Shock (Seed 42: Collapse) |
| :--- | :--- | :--- | :--- |
| **Final Biomass** | **279.62 g/L** | **0.98 g/L** | **1.05 g/L** |
| **Outcome** | High Biomass Success | Complete Collapse | Complete Collapse |
| **Total Dosing Interventions** | 714 | 1,167 | 1,167 |
| **Cumulative Dose Volume** | 364.5 mL | 598.5 mL | 596.0 mL |
| **Mean Pulse Volume** | 0.51 mL | 0.51 mL | 0.51 mL |
| **Initial Compartment Mass** | 19.09 g | 17.62 g | 19.09 g |
| **Final Retained Nutrient Mass** | 68.97 g | 72.73 g | 69.88 g |
| **Final Dissolved Nutrient** | **0.03 g** | **64.80 g** | **61.84 g** |
| **Final Electrical Conductivity** | **0.03 mS/cm** | **51.84 mS/cm** | **49.48 mS/cm** |
| **EC MAE** | 1.01 mS/cm | **31.78 mS/cm** | **32.27 mS/cm** |
| **Final Health Index** | 1.00 | **0.26** | **0.30** |

### 2.2 Mathematical Coupling: d(rho)/dB = -rho / B
The reserve-ratio rule evaluates actions directly on the normalized intracellular quota ratio:
$$\\rho = \\frac{R}{B \\cdot C_{\\text{int}}}, \\quad C_{\\text{int}} = 0.50$$
Partial derivatives with respect to reserve mass R and biomass B:
$$\\frac{\\partial \\rho}{\\partial R} = \\frac{1}{B \\cdot C_{\\text{int}}} > 0$$
$$\\frac{\\partial \\rho}{\\partial B} = -\\frac{R}{B^2 C_{\\text{int}}} = -\\frac{\\rho}{B} < 0$$

#### Implications for Control:
1. **Growth Acts as an Automatic Feedforward Trigger**:
   When biomass B expands, $\\rho$ automatically drops at rate $-\\rho / B$ even if intracellular nutrient mass R remains constant or increases. For example, as B grows from 80 g/L to 200 g/L, $\\rho$ is diluted by a factor of 2.5. When $\\rho$ crosses below the threshold $\\rho < 0.46$, the controller automatically triggers moderate dosing ((2.0 mL/min, 15 s)). The controller is therefore not merely tracking nutrient levels; it is responding directly to population expansion.
2. **Fatal Positive Feedback Loop During Stress (No EC Ceiling)**:
   When adverse conditions (such as cold shock or stochastic damage) inhibit cellular uptake, R drops. The controller observes $\\rho < 0.20$ (critical threshold) and commands maximal dosing ((4.0 mL/min, 20 s)) at every single cooldown step (120 s).
   However, because the rule controller is **blind to EC and water temperature**, it continues pumping fertilizer into a culture that cannot absorb it. As dissolved nutrient accumulates, osmotic stress ($(\\text{dissolved} / 5.0)^2$) shuts down Monod uptake completely ($Monod \\to 0$). This creates a fatal death spiral:
   $$\\text{Uptake drops} \\implies \\rho < 0.20 \\implies \\text{Blind Max Dosing} \\implies \\text{Toxic EC (51.8 mS/cm)} \\implies \\text{Total Extinction}$$
   This explains why in collapsed seeds, the reserve rule actually dosed **64% more nutrient** (598.5 mL vs 364.5 mL) while killing the culture.

### 2.3 Controller Information Access Table

| Controller | EC | Turbidity | Biomass B | Reserve R | Reserve Ratio rho | Water Temp | Prev Action | Latent State Access / Information Asymmetry |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **no_action** | No (Ignored) | No (Ignored) | No (Ignored) | No (Ignored) | No (Ignored) | No (Ignored) | No (Ignored) | Open-loop baseline (zero intervention, receives no state feedback) |
| **rule_based_reserve** | No (Ignored in reserve mode) | No | Yes (Privileged denominator normalization) | Yes (Privileged numerator state) | Yes (Direct canonical latent state: R / (B * 0.5)) | No (Temperature-blind) | Yes (Cooldown timer tsld) | Direct access to unobservable true latent quota ratio rho = R / (B * C_int). Completely bypasses sensor noise, sensor bias, and measurement delay. |
| **pid** | Yes (Soft ceiling override: throttles pump above EC=1.5) | No | Yes (Privileged biomass scaling: min(1.0, B/80)) | Yes (via state object) | Yes (via state object) | No | Yes (Internal rate limiting and anti-windup) | Receives true state object to compute rho, but adds explicit biomass and EC safety overrides. Bypasses sensor noise for primary controlled variable. |
| **lstm** | Yes (Noisy sensor + rolling stats) | Yes (Noisy optical sensor) | No (Hidden biological state, unobserved) | No (Hidden biological state, unobserved) | No (Static fallback 0.5 during evaluation) | Yes (Noisy sensor) | Yes (Observed prev_flowrate, prev_dur) | Strict partial observability: operates entirely on noisy, biased sensor readings and engineered rolling features. Has NO access to true internal quota or biomass. |

#### Critical Findings on Information Asymmetry:
- **`rule_based_reserve` receives privileged unobservable state**: It computes decisions directly from $\\rho_t = R_t / (B_t \\cdot 0.5)$, which is derived from true simulator latent states `internal_reserve` and `algae_biomass`. It operates with zero measurement noise, zero sensor bias, and zero observation delay.
- **`pid` also receives latent state**: It accesses `TankState` to evaluate $\\rho_t$, but explicitly applies two stabilizing heuristic safeguards:
  1. `biomass_scale = min(1.0, B / 80.0)` (throttles pump if biomass drops).
  2. `ec_scale = min(1.0, (2.0 - ec) / 0.5)` (cuts dosing above 1.5 mS/cm to prevent osmotic poisoning).
- **`lstm` operates under strict partial observability**: It receives only noisy, biased sensor readings (`water_temp`, `ec`, `turbidity`, and rolling statistics). It has **no access** to true biomass or intracellular reserves. In fact, `feature_engineering.py` provides a static fallback of $\\rho = 0.5$ during closed-loop evaluation because internal reserve is not observable.

### 2.4 Why High Biomass Occurs at Low Dose (The Role of Recycling)
In benign scenarios where the culture does not collapse:
1. **High Internal Recycling**:
   - Maintenance waste recycles 20% back into the dissolved nutrient pool and 50% into the dead pool (`maintenance_to_dissolved_fraction = 0.2`, `maintenance_to_detritus_fraction = 0.5`).
   - Mineralization returns 30% of dead biomass nutrients back into dissolved nutrients (`mineralization_efficiency = 0.3`).
2. **Self-Shading Density Saturation**:
   Light attenuation follows a Hill function:
   $$f_{\\text{light}}(B) = \\frac{L^2}{L^2 + B^2}, \\quad L = 250.0\\text{ g/L}$$
   Near $B \\approx 270\\text{--}280\\text{ g/L}$, light attenuation throttles growth. Once the culture reaches this equilibrium, cell division matches basal mortality, and internal recycling sustains the existing population. Dosing is only required to replace net irreversible losses (dilution and metabolic waste), which requires only ~364 mL across 3,500 steps.

### 2.5 Potential Benchmark Simplifications
The audit identifies the following benchmark structural characteristics:
1. **Large Reserve Buffer Capacity**: $C_{\\text{int}} = 0.50$ allows cells to store up to 50% of their dry weight in intracellular reserves, creating significant biological inertia that shields simple threshold controllers from short-term fluctuations.
2. **High Recycling Efficiency**: Substantial internal recycling of maintenance and dead biomass allows closed-loop systems to self-sustain with minimal external inputs once peak density is reached.
3. **Absence of EC Safety Constraints in Heuristic Baseline**: The production reserve rule lacks an EC ceiling, causing catastrophic runaway toxicity in 30% of episodes.

---

## 3. Scenario Inventory

Exhaustive inventory of all scenario and disturbance configurations discovered across `simulation/disturbances.py`, `simulation/synthetic_generator.py`, and `simulation_runner/closed_loop_eval.py`:

| Scenario Name | Source File | Definition Entity | Disturbances Involved | Benchmark Status |
| :--- | :--- | :--- | :--- | :--- |
| `normal` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Baseline correlated mixing noise, slow seasonal thermal drift, background stochastic pulses | Yes (Primary) |
| `heatwave` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Sustained positive temperature shock (+4.0 C), uptake multiplier (1.2x) | Yes (Primary) |
| `nutrient_depletion` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | External nutrient depletion (negative EC shock) and elevated uptake (1.35x) | Yes (Primary) |
| `cold_shock` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Severe negative thermal shock (-3.5 C) and metabolic slowdown (0.85x uptake) | Yes (Primary) |
| `temp_spike` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Short transient positive temperature spike (alias/variant of heatwave logic) | No (Unevaluated) |
| `ec_drop` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Direct acute drop in dissolved nutrient concentration (alias of nutrient_depletion) | No (Unevaluated) |
| `noisy_sensors` | `simulation/synthetic_generator.py` | `SyntheticTrajectoryGenerator.generate_trajectory()` | 3x elevated Gaussian measurement noise on all sensor channels (EC, temp, turbidity) | No (Unevaluated) |
| `delayed_response` | `simulation/synthetic_generator.py` | `SyntheticTrajectoryGenerator.generate_trajectory()` | Severe transport delay: immediate absorption reduced from 50% to 8%, delayed kernel extended | No (Unevaluated) |
| `actuator_failure` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Pump mechanical degradation cutting effective dosing mass delivery to 35% | No (Unevaluated) |
| `actuator_saturation / saturation` | `simulation_runner/closed_loop_eval.py` | `ClosedLoopEvaluator.run_episode()` | Severe hardware clamping of pump: flowrate capped at 1.0 mL/min, duration capped at 5.0 s | No (Unevaluated) |
| `sediment / turbidity_event` | `simulation/disturbances.py` | `DisturbanceGenerator.sample()` | Turbidity sensor blinding / particulate sediment plume (+18.0 NTU) | No (Unevaluated) |
| `missing_data` | `simulation_runner/closed_loop_eval.py` | `ClosedLoopEvaluator.run_episode()` | Intermittent sensor dropout: EC reading is replaced with NaN every 17 timesteps | No (Unevaluated) |

---

## 4. Current Evaluation Coverage

| Dimension | Count | Coverage % | Details |
| :--- | :--- | :--- | :--- |
| Total Implemented Scenarios/Regimes | 12 | 100.0% | All distinct named regimes in disturbances.py, synthetic_generator.py, closed_loop_eval.py |
| Scenarios in 3500-Step Benchmark | 4 | 33.3% | normal, heatwave, nutrient_depletion, cold_shock (4 out of 12) |
| Implemented but Unevaluated Regimes | 8 | 66.7% | noisy_sensors, delayed_response, actuator_failure, saturation, sediment, missing_data, temp_spike, ec_drop |

### Key Finding on Evaluation Coverage:
The 3,500-step multi-seed evaluation currently covers only **4 out of 12** implemented operational regimes (33.3% coverage).
Eight distinct regimes are fully implemented in code but remain unevaluated in the benchmark paper:
1. `noisy_sensors` (3x measurement noise)
2. `delayed_response` (8% immediate absorption, extended 4-stage delay kernel)
3. `actuator_failure` (35% pump mechanical efficiency degradation)
4. `actuator_saturation` (flowrate capped at 1.0 mL/min, duration at 5.0 s)
5. `sediment` (+18.0 NTU turbidity sensor blinding plume)
6. `missing_data` (EC sensor dropout every 17 steps)
7. `temp_spike` (short acute thermal pulse)
8. `ec_drop` (acute dissolved nutrient drop)

---

## 5. Findings

Based strictly on code inspection, simulator equations, and empirical multi-seed data:

1. **The Reserve Rule Does NOT Invalidate Learned Control**:
   The apparent superiority of `rule_based_reserve` in previous single-seed reports was an artifact of looking only at successful seeds. Under rigorous 120-episode evaluation, `rule_based_reserve` has an unacceptable **30.0% collapse rate** (and **70.0% in cold shock**), whereas both PID and LSTM achieve **0% collapse**.
2. **Information Asymmetry is Significant**:
   The reserve-ratio rule controller receives privileged, unobservable ground-truth state variables ($\rho = R / (B \\cdot 0.5)$), completely bypassing sensor noise, bias, and measurement delay. In contrast, the LSTM operates under strict partial observability from noisy sensor streams.
3. **Reserve Ratio Couples Directly to Biomass**:
   Because $\\partial\\rho / \\partial B = -\\rho / B$, the reserve ratio acts as a population-growth detector, automatically triggering dosing as the culture expands. However, the absence of an EC ceiling makes this feedback fatal when cellular uptake is impaired.
4. **Low Dosing at High Biomass is a Legitimate Consequence of Internal Recycling**:
   In benign regimes, high biomass (280 g/L) requires little external dosing because mineralization (30%) and maintenance recycling (20% dissolved, 50% detritus) return nutrients to the culture, and light attenuation (L = 250 g/L) halts runaway expansion.
5. **Significant Benchmark Expansion Headroom Exists**:
   Two-thirds of the repository's implemented disturbance regimes (`noisy_sensors`, `delayed_response`, `actuator_failure`, `actuator_saturation`, `sediment`, `missing_data`) are already coded but have not yet been evaluated in the 3,500-step benchmark.
