# Myopic 1-Step Greedy Controller -- Temporal Complexity Diagnostic

## Purpose

This diagnostic evaluates whether meaningful performance in the algae tank benchmark depends on anticipating delayed consequences (long-horizon temporal complexity).

- **If the 1-step controller performs close to existing long-horizon controllers**: provides evidence that the control problem may be relatively myopic (low temporal complexity).
- **If the 1-step controller performs substantially worse** (especially in final biomass, collapse probability, or control effort): provides evidence that delayed dynamics materially affect control difficulty.

> Scientific interpretation is left to the reviewer.  No automated complexity label is assigned.

## Evaluation Configuration

- **Myopic Controller**: `myopic_1step` -- 1-step greedy, deterministic; identical objective and action set to the supervisory labeler
- **Scenarios**: `normal`, `heatwave`, `nutrient_depletion`, `cold_shock`
- **Seeds per scenario**: 30 (same seed schedule as primary benchmark)
- **Horizon**: 3500 steps (60 s/step -> 58.33 h)
- **CRN pairing**: Identical initial states, disturbances, and sensor noise innovations (seed arithmetic reproduced from MultiSeedEvaluator)
- **Action space**: 61 discrete candidates (flowrate x duration, identical to OptimizationLabeler)

## EC MAE (mS/cm)

Mean [95% Bootstrap CI] across 30 seeds per scenario.

| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 1.1156 [1.1145, 1.1166] | 7.3504 [3.2730, 12.2413] | 1.0881 [1.0822, 1.0938] | 0.9949 [0.9893, 1.0011] |
| **heatwave** | 1.1158 [1.1147, 1.1169] | 7.6172 [3.4915, 12.5752] | 1.0879 [1.0824, 1.0933] | 0.9910 [0.9877, 0.9944] |
| **nutrient_depletion** | 1.1163 [1.1153, 1.1172] | 4.6573 [1.6451, 7.9844] | 1.0879 [1.0824, 1.0932] | 0.9933 [0.9885, 0.9985] |
| **cold_shock** | 1.1152 [1.1143, 1.1163] | 23.4059 [18.2331, 28.5069] | 1.0836 [1.0784, 1.0891] | 0.9905 [0.9868, 0.9946] |

## Final Biomass (g/L)

Mean [95% Bootstrap CI] across 30 seeds per scenario.

| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 23.1545 [22.7791, 23.5362] | 222.6929 [177.3769, 259.8341] | 216.9685 [216.0301, 217.9498] | 135.0061 [122.7699, 147.3777] |
| **heatwave** | 22.4593 [22.1223, 22.7879] | 221.3545 [176.9666, 258.5137] | 216.2931 [215.2422, 217.3690] | 132.2142 [122.9924, 142.0210] |
| **nutrient_depletion** | 22.9122 [22.5195, 23.2971] | 248.1808 [218.3589, 274.9933] | 216.2485 [215.0083, 217.4239] | 140.6235 [128.6281, 153.0715] |
| **cold_shock** | 23.3030 [22.9078, 23.7007] | 81.7745 [37.2011, 127.2440] | 216.0418 [215.0948, 216.9032] | 147.0202 [138.5793, 155.7264] |

## Total Dose (mL)

Mean [95% Bootstrap CI] across 30 seeds per scenario.

| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 195.9306 [192.6861, 199.3224] | 410.6111 [380.0943, 447.3738] | 874.1950 [871.4843, 876.8095] | 1123.3610 [1077.1228, 1170.1212] |
| **heatwave** | 192.7792 [190.3083, 195.2598] | 411.8833 [380.8163, 449.7782] | 873.4988 [870.0868, 876.8407] | 1126.8321 [1089.3327, 1163.7822] |
| **nutrient_depletion** | 194.1639 [191.5124, 196.7628] | 388.2778 [365.5554, 412.5058] | 872.2022 [868.7857, 875.5230] | 1135.8687 [1089.8657, 1183.1065] |
| **cold_shock** | 196.6153 [193.4430, 199.7737] | 527.8944 [488.6992, 566.5836] | 873.0011 [870.3508, 875.4309] | 1169.2291 [1141.2230, 1197.9700] |

## Final Reserve (rho)

Mean [95% Bootstrap CI] across 30 seeds per scenario.

| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 0.1560 [0.1553, 0.1566] | 0.2408 [0.2297, 0.2507] | 0.2251 [0.2244, 0.2259] | 0.1095 [0.0920, 0.1290] |
| **heatwave** | 0.1558 [0.1553, 0.1564] | 0.2414 [0.2288, 0.2524] | 0.2256 [0.2249, 0.2264] | 0.0929 [0.0848, 0.1051] |
| **nutrient_depletion** | 0.1558 [0.1553, 0.1563] | 0.2487 [0.2398, 0.2559] | 0.2248 [0.2238, 0.2256] | 0.1196 [0.1002, 0.1417] |
| **cold_shock** | 0.1556 [0.1549, 0.1561] | 0.1978 [0.1833, 0.2129] | 0.2252 [0.2245, 0.2259] | 0.1154 [0.0982, 0.1351] |

## Collapse Probability (Final Biomass < 50 g/L)

Wilson 95% CI.  Primary benchmark data from `reports/multiseed_long_horizon/episode_results.csv` (read-only).

| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 1.000 [0.886, 1.000] (30/30) | 0.200 [0.095, 0.373] (6/30) | 0.000 [0.000, 0.114] (0/30) | 0.000 [0.000, 0.114] (0/30) |
| **heatwave** | 1.000 [0.886, 1.000] (30/30) | 0.200 [0.095, 0.373] (6/30) | 0.000 [0.000, 0.114] (0/30) | 0.000 [0.000, 0.114] (0/30) |
| **nutrient_depletion** | 1.000 [0.886, 1.000] (30/30) | 0.100 [0.035, 0.256] (3/30) | 0.000 [0.000, 0.114] (0/30) | 0.000 [0.000, 0.114] (0/30) |
| **cold_shock** | 1.000 [0.886, 1.000] (30/30) | 0.700 [0.521, 0.833] (21/30) | 0.000 [0.000, 0.114] (0/30) | 0.000 [0.000, 0.114] (0/30) |

## Termination Summary

| Scenario | Episodes | Early Terminations | Termination Rate |
| :--- | :---: | :---: | :---: |
| normal | 30 | 0 | 0.000 |
| heatwave | 30 | 0 | 0.000 |
| nutrient_depletion | 30 | 0 | 0.000 |
| cold_shock | 30 | 0 | 0.000 |

## Scientific Interpretation Guidance

Compare each metric column (myopic vs reserve rule, PID, LSTM) across scenarios:

- **Similar EC MAE across all controllers**: Regulation may not require long-horizon anticipation.
- **Substantially lower final biomass or higher collapse probability for myopic**: Suggests delayed uptake / reserve kinetics require anticipation beyond 1 step.
- **Similar or higher total dose for myopic**: Greedy dosing may be less efficient -- overdosing without awareness of osmotic stress accumulation.
- **Scenario-specific divergence** (e.g., cold_shock only): Temporal complexity is conditionally present under physiological stress.

---
*Report generated by `evaluation/myopic_controller.py`.*
*Primary benchmark data read-only from `reports/multiseed_long_horizon/`.*
*This diagnostic does NOT modify any existing pipeline or result.*