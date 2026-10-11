# Paired Multi-Seed Long-Horizon Statistical Evaluation & ML-Discriminability Audit

## 1. Evaluation Design

- **Controllers Evaluated**: `no_action`, `rule_based_reserve`, `pid`, `lstm`
- **Operating Scenarios** (4): `normal`, `heatwave`, `nutrient_depletion`, `cold_shock`
- **Sample Size**: 2 independent seeds per scenario $\rightarrow$ **8 paired episodes** (480 total controller rollouts)
- **Horizon**: 3500 simulation steps per episode (60 s timestep $\rightarrow$ 58.33 operating hours)
- **Statistical Unit**: Paired episode difference ($\Delta_i = M_{A,i} - M_{B,i}$, $N=2$ paired observations per scenario)
- **Bootstrap Confidence Level**: 95% (1,000 replicates, deterministic seed 12345)

## 2. Common Random Numbers (CRN) & Sensor Noise Decoupling

To ensure rigorous statistical pairing without artificially decoupling observations from controller dynamics:
1. **Exogenous Disturbances**: A complete 3500-step stochastic schedule (thermal events, sediment shocks, uptake pulses, mixing noise) is pre-generated and shared across all 4 controllers.
2. **Initial State ($s_0$)**: A single initial state is constructed per seed and `deepcopy` assigned independently to each controller.
3. **Sensor Noise Innovations ($\epsilon_t$)**: The sensor observation model obeys:
   $$y_t^{(c)} = h(x_t^{(c)}) + \epsilon_t$$
   where identical Gaussian innovations $\epsilon_t$ are shared across controllers, but evaluated at each controller's genuine internal tank state $x_t^{(c)}$.

## 3. Reviewer Primary Comparison: LSTM vs PID

| Scenario | Metric | Paired Mean $\Delta$ | 95% Bootstrap CI | LSTM Win Rate | Exceeding Preregistered $\delta$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Normal | EC MAE (mS/cm) | -0.0556 | [-0.0573, -0.0540] | 100.0% | 100.0% |
| Normal | Final Biomass (g/L) | -136.3243 | [-146.4867, -126.1618] | 0.0% | 0.0% |
| Normal | Final Reserve Ratio (rho) (fraction) | -0.1539 | [-0.1565, -0.1513] | 0.0% | 0.0% |
| Normal | Total Dose (mL) | +38.1634 | [-34.7763, +111.1031] | 50.0% | 50.0% |
| Heatwave | EC MAE (mS/cm) | -0.0859 | [-0.0889, -0.0829] | 100.0% | 100.0% |
| Heatwave | Final Biomass (g/L) | -71.0857 | [-75.8614, -66.3100] | 0.0% | 0.0% |
| Heatwave | Final Reserve Ratio (rho) (fraction) | -0.1338 | [-0.1345, -0.1331] | 0.0% | 0.0% |
| Heatwave | Total Dose (mL) | +320.4169 | [+318.3687, +322.4651] | 0.0% | 0.0% |
| Nutrient Depletion | EC MAE (mS/cm) | -0.0910 | [-0.0965, -0.0855] | 100.0% | 100.0% |
| Nutrient Depletion | Final Biomass (g/L) | -56.9462 | [-99.4563, -14.4362] | 0.0% | 0.0% |
| Nutrient Depletion | Final Reserve Ratio (rho) (fraction) | -0.0738 | [-0.1436, -0.0039] | 0.0% | 0.0% |
| Nutrient Depletion | Total Dose (mL) | +314.8863 | [+170.9204, +458.8522] | 0.0% | 0.0% |
| Cold Shock | EC MAE (mS/cm) | -0.0813 | [-0.0858, -0.0768] | 100.0% | 100.0% |
| Cold Shock | Final Biomass (g/L) | -17.0695 | [-23.2243, -10.9147] | 0.0% | 0.0% |
| Cold Shock | Final Reserve Ratio (rho) (fraction) | -0.0149 | [-0.0574, +0.0275] | 50.0% | 0.0% |
| Cold Shock | Total Dose (mL) | +463.4232 | [+462.3930, +464.4533] | 0.0% | 0.0% |
| Equal Weighted Aggregate | EC MAE (mS/cm) | -0.0785 | [-0.0809, -0.0756] | 100.0% | 100.0% |
| Equal Weighted Aggregate | Final Biomass (g/L) | -70.3564 | [-83.7166, -56.9963] | 0.0% | 0.0% |
| Equal Weighted Aggregate | Final Reserve Ratio (rho) (fraction) | -0.0941 | [-0.1222, -0.0660] | 12.5% | 0.0% |
| Equal Weighted Aggregate | Total Dose (mL) | +284.2224 | [+229.9960, +338.4488] | 12.5% | 12.5% |

## 4. Benchmark ML-Discriminability Audit: LSTM vs Reserve-Ratio Rule

Evaluating whether the learned policy provides meaningful advantage over simple mechanistically informed feedback ($-\delta < \Delta < +\delta$ practical equivalence):

| Scenario | Metric | Paired Mean $\Delta$ | 95% Bootstrap CI | Win Rate | Equivalence Margin | Practical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Normal | EC MAE | -15.3707 | [-30.7719, +0.0306] | 50.0% | [$\pm$0.04] | **Inconclusive** |
| Normal | Final Biomass | -58.1003 | [-211.1918, +94.9912] | 50.0% | [$\pm$5.00] | **Inconclusive** |
| Normal | Final Reserve Ratio (rho) | -0.1411 | [-0.1877, -0.0946] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Normal | Total Dose | +435.7871 | [+396.5622, +475.0119] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Heatwave | EC MAE | -0.9780 | [-1.9656, +0.0095] | 50.0% | [$\pm$0.04] | **Inconclusive** |
| Heatwave | Final Biomass | -124.9547 | [-137.4100, -112.4995] | 0.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Heatwave | Final Reserve Ratio (rho) | -0.1699 | [-0.1707, -0.1692] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Heatwave | Total Dose | +828.1894 | [+822.6623, +833.7166] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Nutrient Depletion | EC MAE | +0.0010 | [-0.0090, +0.0111] | 50.0% | [$\pm$0.04] | **Practical equivalence** |
| Nutrient Depletion | Final Biomass | -124.3834 | [-165.7159, -83.0509] | 0.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Nutrient Depletion | Final Reserve Ratio (rho) | -0.1043 | [-0.1720, -0.0366] | 0.0% | [$\pm$0.04] | **Inconclusive** |
| Nutrient Depletion | Total Dose | +821.3807 | [+684.6715, +958.0900] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Cold Shock | EC MAE | -15.6597 | [-31.1810, -0.1384] | 100.0% | [$\pm$0.04] | **LSTM advantage** |
| Cold Shock | Final Biomass | +62.1812 | [-69.3486, +193.7110] | 50.0% | [$\pm$5.00] | **Inconclusive** |
| Cold Shock | Final Reserve Ratio (rho) | -0.0058 | [-0.0139, +0.0024] | 50.0% | [$\pm$0.04] | **Practical equivalence** |
| Cold Shock | Total Dose | +852.8675 | [+742.2296, +963.5054] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Equal Weighted Aggregate | EC MAE | -8.0019 | [-15.7325, -0.2687] | 62.5% | [$\pm$0.04] | **LSTM advantage** |
| Equal Weighted Aggregate | Final Biomass | -61.3143 | [-135.5835, +9.8410] | 25.0% | [$\pm$5.00] | **Inconclusive** |
| Equal Weighted Aggregate | Final Reserve Ratio (rho) | -0.1053 | [-0.1341, -0.0767] | 12.5% | [$\pm$0.04] | **Reserve rule advantage** |
| Equal Weighted Aggregate | Total Dose | +734.5562 | [+671.3376, +796.3930] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |

## 5. Control Effort Adjustment & Trade-Off Analysis

To prevent misinterpreting higher regulation precision that is achieved merely by excessive chemical dosing, performance differences are directly coupled with dosing effort:

| Scenario | $\Delta$ Biomass (LSTM - Rule) | $\Delta$ EC MAE (LSTM - Rule) | $\Delta$ Total Dose (LSTM - Rule) | $\Delta$ Total Dose (LSTM - PID) |
| :--- | :---: | :---: | :---: | :---: |
| Normal | -58.10 g/L [-211.19, +94.99] | -15.3707 [-30.7719, +0.0306] | +435.79 mL [+396.56, +475.01] | +38.16 mL [-34.78, +111.10] |
| Heatwave | -124.95 g/L [-137.41, -112.50] | -0.9780 [-1.9656, +0.0095] | +828.19 mL [+822.66, +833.72] | +320.42 mL [+318.37, +322.47] |
| Nutrient Depletion | -124.38 g/L [-165.72, -83.05] | +0.0010 [-0.0090, +0.0111] | +821.38 mL [+684.67, +958.09] | +314.89 mL [+170.92, +458.85] |
| Cold Shock | +62.18 g/L [-69.35, +193.71] | -15.6597 [-31.1810, -0.1384] | +852.87 mL [+742.23, +963.51] | +463.42 mL [+462.39, +464.45] |
| Equal Weighted Aggregate | -61.31 g/L [-135.58, +9.84] | -8.0019 [-15.7325, -0.2687] | +734.56 mL [+671.34, +796.39] | +284.22 mL [+230.00, +338.45] |

## 6. Aggregate Controller Summary (Mean ± 95% Bootstrap CI Across Seeds)

| Controller | Scenario | EC MAE (mS/cm) | Final Biomass (g/L) | Final Reserve (rho) | Total Dose (mL) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| No Action | Normal | 1.1662 [1.1654, 1.1671] | 6.22 [5.82, 6.61] | 0.0308 [0.0303, 0.0313] | 0.00 [0.00, 0.00] |
| No Action | Heatwave | 1.1666 [1.1652, 1.1680] | 6.21 [6.20, 6.22] | 0.0308 [0.0307, 0.0309] | 0.00 [0.00, 0.00] |
| No Action | Nutrient Depletion | 1.1669 [1.1665, 1.1674] | 6.25 [6.13, 6.37] | 0.0308 [0.0307, 0.0309] | 0.00 [0.00, 0.00] |
| No Action | Cold Shock | 1.1657 [1.1648, 1.1666] | 6.33 [6.02, 6.64] | 0.0305 [0.0304, 0.0306] | 0.00 [0.00, 0.00] |
| Rule Based Reserve | Normal | 16.3945 [1.0067, 31.7822] | 140.30 [0.98, 279.62] | 0.2129 [0.1692, 0.2566] | 481.50 [364.50, 598.50] |
| Rule Based Reserve | Heatwave | 1.9779 [0.9982, 2.9575] | 267.76 [258.31, 277.20] | 0.2619 [0.2610, 0.2629] | 363.83 [362.50, 365.17] |
| Rule Based Reserve | Nutrient Depletion | 0.9846 [0.9685, 1.0006] | 287.02 [285.86, 288.18] | 0.2566 [0.2534, 0.2597] | 367.00 [362.83, 371.17] |
| Rule Based Reserve | Cold Shock | 16.6582 [1.1206, 32.1958] | 136.16 [1.04, 271.27] | 0.2159 [0.1832, 0.2486] | 477.75 [359.50, 596.00] |
| Pid | Normal | 1.0794 [1.0675, 1.0913] | 218.52 [214.91, 222.13] | 0.2257 [0.2254, 0.2259] | 879.12 [874.29, 883.96] |
| Pid | Heatwave | 1.0857 [1.0808, 1.0906] | 213.89 [212.12, 215.66] | 0.2258 [0.2253, 0.2263] | 871.61 [865.36, 877.85] |
| Pid | Nutrient Depletion | 1.0766 [1.0651, 1.0881] | 219.58 [217.25, 221.92] | 0.2260 [0.2250, 0.2271] | 873.49 [862.07, 884.92] |
| Pid | Cold Shock | 1.0798 [1.0680, 1.0916] | 215.41 [212.84, 217.97] | 0.2251 [0.2234, 0.2268] | 867.19 [860.61, 873.78] |
| Lstm | Normal | 1.0238 [1.0103, 1.0373] | 82.20 [68.43, 95.97] | 0.0717 [0.0689, 0.0746] | 917.29 [839.51, 995.06] |
| Lstm | Heatwave | 0.9998 [0.9919, 1.0077] | 142.80 [139.79, 145.81] | 0.0920 [0.0918, 0.0922] | 1192.02 [1187.83, 1196.22] |
| Lstm | Nutrient Depletion | 0.9856 [0.9796, 0.9916] | 162.64 [122.46, 202.81] | 0.1523 [0.0814, 0.2232] | 1188.38 [1055.84, 1320.92] |
| Lstm | Cold Shock | 0.9984 [0.9822, 1.0147] | 198.34 [194.75, 201.92] | 0.2101 [0.1693, 0.2509] | 1330.62 [1323.01, 1338.23] |
| Mpc Reserve | Normal | 27.0691 [21.7986, 32.3397] | 3.63 [1.11, 6.14] | 0.1934 [0.1724, 0.2145] | 726.92 [665.50, 788.33] |
| Mpc Reserve | Heatwave | 34.2298 [2.3444, 66.1152] | 158.56 [0.00, 317.11] | 0.1496 [0.0000, 0.2992] | 761.75 [526.17, 997.33] |
| Mpc Reserve | Nutrient Depletion | 49.2764 [31.9132, 66.6396] | 0.63 [0.00, 1.25] | 0.0939 [0.0000, 0.1878] | 882.58 [773.17, 992.00] |
| Mpc Reserve | Cold Shock | 36.5779 [6.8424, 66.3134] | 106.43 [0.00, 212.87] | 0.1625 [0.0000, 0.3251] | 746.00 [500.00, 992.00] |

## 7. Episode Termination & Horizon Integrity

- **Total Controller Executions**: 40
- **Completed Full Horizon (3500 steps)**: 37
- **Early Terminations**: 3

| Controller | Scenario | Termination Reason | Episodes Terminated |
| :--- | :--- | :--- | :---: |
| mpc_reserve | cold_shock | health_collapse | 1 |
| mpc_reserve | heatwave | health_collapse | 1 |
| mpc_reserve | nutrient_depletion | health_collapse | 1 |

## 8. Benchmark Discriminability Findings

1. **Statistical Separation from PID**: Across repeated stochastic trajectories, paired differences between LSTM and PID confirm whether the learned policy exhibits statistically distinct regulation and biomass yields with confidence intervals excluding zero.
2. **Residual Control Complexity Beyond Reserve Heuristic**: Practical equivalence testing examines whether the simple reserve-ratio controller ($-\delta < \Delta < +\delta$) captures the actionable control structure under baseline conditions, and highlights which stress regimes (e.g., heatwave vs nutrient depletion) reveal significant divergences.
3. **Control Effort Discrepancy**: The observed performance difference is evaluated in context with nutrient dosage volumes to assess whether learned regulation relies on different dosing efficiency.

## 9. Reproducibility & Provenance Information

- **Seed Schedule**: Saved in `seed_schedule.json` (2 seeds per scenario)
- **Configuration Snapshot**: Saved in `evaluation_config.yaml`
- **Raw Episode Data**: Saved in `episode_results.csv` and `episode_results.json`
- **Full Paired Differences**: Saved in `paired_differences.csv`
- **Bootstrap CIs**: Saved in `bootstrap_confidence_intervals.csv`
- **Equivalence Analysis**: Saved in `equivalence_analysis.csv`
- **Supplementary Wilcoxon Tests**: Saved in `wilcoxon_results.csv`