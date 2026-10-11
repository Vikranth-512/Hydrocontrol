# Paired Multi-Seed Long-Horizon Statistical Evaluation & ML-Discriminability Audit

## 1. Evaluation Design

- **Controllers Evaluated**: `no_action`, `rule_based_reserve`, `pid`, `lstm`, `mpc_reserve`
- **Operating Scenarios** (11): `normal`, `heatwave`, `nutrient_depletion`, `cold_shock`, `noisy_sensors`, `delayed_response`, `actuator_failure`, `actuator_saturation`, `sediment`, `temp_spike`, `ec_drop`
- **Sample Size**: 30 independent seeds per scenario $\rightarrow$ **330 paired episodes** (480 total controller rollouts)
- **Horizon**: 3500 simulation steps per episode (60 s timestep $\rightarrow$ 58.33 operating hours)
- **Statistical Unit**: Paired episode difference ($\Delta_i = M_{A,i} - M_{B,i}$, $N=30$ paired observations per scenario)
- **Bootstrap Confidence Level**: 95% (5,000 replicates, deterministic seed 12345)

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
| Normal | EC MAE (mS/cm) | -0.0931 | [-0.1009, -0.0853] | 100.0% | 100.0% |
| Normal | Final Biomass (g/L) | -81.9625 | [-94.4075, -69.4260] | 0.0% | 0.0% |
| Normal | Final Reserve Ratio (rho) (fraction) | -0.1156 | [-0.1333, -0.0960] | 10.0% | 0.0% |
| Normal | Total Dose (mL) | +249.1660 | [+202.1824, +296.4299] | 3.3% | 3.3% |
| Heatwave | EC MAE (mS/cm) | -0.0969 | [-0.1026, -0.0913] | 100.0% | 100.0% |
| Heatwave | Final Biomass (g/L) | -84.0788 | [-93.2898, -74.1924] | 3.3% | 0.0% |
| Heatwave | Final Reserve Ratio (rho) (fraction) | -0.1327 | [-0.1406, -0.1206] | 3.3% | 0.0% |
| Heatwave | Total Dose (mL) | +253.3333 | [+216.7901, +289.9451] | 0.0% | 0.0% |
| Nutrient Depletion | EC MAE (mS/cm) | -0.0946 | [-0.1018, -0.0872] | 100.0% | 96.7% |
| Nutrient Depletion | Final Biomass (g/L) | -75.6250 | [-87.2921, -63.4828] | 3.3% | 3.3% |
| Nutrient Depletion | Final Reserve Ratio (rho) (fraction) | -0.1051 | [-0.1247, -0.0831] | 10.0% | 0.0% |
| Nutrient Depletion | Total Dose (mL) | +263.6665 | [+217.7901, +310.1990] | 3.3% | 0.0% |
| Cold Shock | EC MAE (mS/cm) | -0.0931 | [-0.0981, -0.0881] | 100.0% | 100.0% |
| Cold Shock | Final Biomass (g/L) | -69.0216 | [-77.4226, -60.2607] | 0.0% | 0.0% |
| Cold Shock | Final Reserve Ratio (rho) (fraction) | -0.1098 | [-0.1271, -0.0900] | 6.7% | 0.0% |
| Cold Shock | Total Dose (mL) | +296.2279 | [+268.7863, +324.4594] | 0.0% | 0.0% |
| Noisy Sensors | EC MAE (mS/cm) | -0.0536 | [-0.0622, -0.0449] | 100.0% | 76.7% |
| Noisy Sensors | Final Biomass (g/L) | -146.1326 | [-151.5920, -140.1855] | 0.0% | 0.0% |
| Noisy Sensors | Final Reserve Ratio (rho) (fraction) | -0.1498 | [-0.1510, -0.1485] | 0.0% | 0.0% |
| Noisy Sensors | Total Dose (mL) | -105.2595 | [-136.1662, -73.6742] | 86.7% | 86.7% |
| Delayed Response | EC MAE (mS/cm) | -0.0964 | [-0.1023, -0.0902] | 100.0% | 100.0% |
| Delayed Response | Final Biomass (g/L) | -81.7804 | [-90.4200, -72.8686] | 0.0% | 0.0% |
| Delayed Response | Final Reserve Ratio (rho) (fraction) | -0.1238 | [-0.1370, -0.1079] | 0.0% | 0.0% |
| Delayed Response | Total Dose (mL) | +247.0697 | [+215.1991, +279.4768] | 0.0% | 0.0% |
| Actuator Failure | EC MAE (mS/cm) | -0.0763 | [-0.0824, -0.0702] | 100.0% | 96.7% |
| Actuator Failure | Final Biomass (g/L) | -39.1505 | [-47.8114, -29.7850] | 6.7% | 3.3% |
| Actuator Failure | Final Reserve Ratio (rho) (fraction) | -0.0682 | [-0.0756, -0.0595] | 3.3% | 0.0% |
| Actuator Failure | Total Dose (mL) | +199.8824 | [+163.6087, +237.3326] | 3.3% | 3.3% |
| Actuator Saturation | EC MAE (mS/cm) | +0.1480 | [+0.1459, +0.1501] | 0.0% | 0.0% |
| Actuator Saturation | Final Biomass (g/L) | -34.7214 | [-36.2191, -33.1976] | 0.0% | 0.0% |
| Actuator Saturation | Final Reserve Ratio (rho) (fraction) | +0.0721 | [+0.0686, +0.0754] | 100.0% | 100.0% |
| Actuator Saturation | Total Dose (mL) | -59.1668 | [-61.8127, -56.5180] | 100.0% | 100.0% |
| Sediment | EC MAE (mS/cm) | -0.1002 | [-0.1068, -0.0935] | 100.0% | 100.0% |
| Sediment | Final Biomass (g/L) | -72.6194 | [-82.5314, -63.1023] | 0.0% | 0.0% |
| Sediment | Final Reserve Ratio (rho) (fraction) | -0.1146 | [-0.1297, -0.0971] | 3.3% | 0.0% |
| Sediment | Total Dose (mL) | +277.4630 | [+234.3306, +316.6332] | 3.3% | 3.3% |
| Temp Spike | EC MAE (mS/cm) | -0.0975 | [-0.1045, -0.0904] | 100.0% | 100.0% |
| Temp Spike | Final Biomass (g/L) | -85.5676 | [-94.2811, -76.6491] | 0.0% | 0.0% |
| Temp Spike | Final Reserve Ratio (rho) (fraction) | -0.1276 | [-0.1392, -0.1133] | 0.0% | 0.0% |
| Temp Spike | Total Dose (mL) | +245.2242 | [+212.2089, +277.8385] | 0.0% | 0.0% |
| Ec Drop | EC MAE (mS/cm) | -0.0957 | [-0.1012, -0.0900] | 100.0% | 100.0% |
| Ec Drop | Final Biomass (g/L) | -78.6749 | [-89.2133, -68.2347] | 0.0% | 0.0% |
| Ec Drop | Final Reserve Ratio (rho) (fraction) | -0.1185 | [-0.1325, -0.1021] | 0.0% | 0.0% |
| Ec Drop | Total Dose (mL) | +270.9684 | [+231.5838, +309.6375] | 0.0% | 0.0% |
| Equal Weighted Aggregate | EC MAE (mS/cm) | -0.0681 | [-0.0701, -0.0662] | 90.9% | 88.2% |
| Equal Weighted Aggregate | Final Biomass (g/L) | -77.2122 | [-79.9401, -74.4382] | 1.2% | 0.6% |
| Equal Weighted Aggregate | Final Reserve Ratio (rho) (fraction) | -0.0994 | [-0.1036, -0.0951] | 12.4% | 9.1% |
| Equal Weighted Aggregate | Total Dose (mL) | +194.4159 | [+183.4565, +204.8340] | 18.2% | 17.9% |

## 4. Benchmark ML-Discriminability Audit: LSTM vs Reserve-Ratio Rule

Evaluating whether the learned policy provides meaningful advantage over simple mechanistically informed feedback ($-\delta < \Delta < +\delta$ practical equivalence):

| Scenario | Metric | Paired Mean $\Delta$ | 95% Bootstrap CI | Win Rate | Equivalence Margin | Practical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Normal | EC MAE | -6.3555 | [-11.2476, -2.2786] | 66.7% | [$\pm$0.04] | **LSTM advantage** |
| Normal | Final Biomass | -87.6869 | [-126.0761, -44.6702] | 20.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Normal | Final Reserve Ratio (rho) | -0.1313 | [-0.1513, -0.1100] | 3.3% | [$\pm$0.04] | **Reserve rule advantage** |
| Normal | Total Dose | +712.7499 | [+654.5418, +771.4094] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Heatwave | EC MAE | -6.6262 | [-11.5858, -2.5004] | 76.7% | [$\pm$0.04] | **LSTM advantage** |
| Heatwave | Final Biomass | -89.1403 | [-128.3216, -43.6321] | 20.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Heatwave | Final Reserve Ratio (rho) | -0.1485 | [-0.1626, -0.1319] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Heatwave | Total Dose | +714.9487 | [+670.9982, +758.0718] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Nutrient Depletion | EC MAE | -3.6641 | [-6.9891, -0.6563] | 76.7% | [$\pm$0.04] | **LSTM advantage** |
| Nutrient Depletion | Final Biomass | -107.5573 | [-136.9754, -70.5634] | 10.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Nutrient Depletion | Final Reserve Ratio (rho) | -0.1290 | [-0.1507, -0.1032] | 3.3% | [$\pm$0.04] | **Reserve rule advantage** |
| Nutrient Depletion | Total Dose | +747.5909 | [+699.3393, +794.8774] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Cold Shock | EC MAE | -22.4154 | [-27.5182, -17.2420] | 96.7% | [$\pm$0.04] | **LSTM advantage** |
| Cold Shock | Final Biomass | +65.2458 | [+17.2232, +111.3537] | 70.0% | [$\pm$5.00] | **LSTM advantage** |
| Cold Shock | Final Reserve Ratio (rho) | -0.0823 | [-0.1060, -0.0570] | 13.3% | [$\pm$0.04] | **Reserve rule advantage** |
| Cold Shock | Total Dose | +641.3346 | [+598.5897, +686.8228] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Noisy Sensors | EC MAE | -4.6878 | [-8.8816, -1.4022] | 50.0% | [$\pm$0.04] | **LSTM advantage** |
| Noisy Sensors | Final Biomass | -165.8720 | [-196.6351, -127.7955] | 13.3% | [$\pm$5.00] | **Reserve rule advantage** |
| Noisy Sensors | Final Reserve Ratio (rho) | -0.1719 | [-0.1807, -0.1617] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Noisy Sensors | Total Dose | +372.0709 | [+327.1338, +413.6118] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Delayed Response | EC MAE | -7.6856 | [-12.6050, -3.4612] | 83.3% | [$\pm$0.04] | **LSTM advantage** |
| Delayed Response | Final Biomass | -75.7771 | [-115.9066, -31.0043] | 23.3% | [$\pm$5.00] | **Reserve rule advantage** |
| Delayed Response | Final Reserve Ratio (rho) | -0.1371 | [-0.1568, -0.1152] | 6.7% | [$\pm$0.04] | **Reserve rule advantage** |
| Delayed Response | Total Dose | +703.3142 | [+651.5815, +753.3187] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Actuator Failure | EC MAE | -0.0444 | [-0.1387, +0.0125] | 23.3% | [$\pm$0.04] | **Inconclusive** |
| Actuator Failure | Final Biomass | -82.8920 | [-91.6971, -73.4878] | 0.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Actuator Failure | Final Reserve Ratio (rho) | -0.1024 | [-0.1099, -0.0939] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Actuator Failure | Total Dose | +761.9755 | [+726.5335, +798.4284] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Actuator Saturation | EC MAE | -3.6680 | [-3.6999, -3.6367] | 100.0% | [$\pm$0.04] | **LSTM advantage** |
| Actuator Saturation | Final Biomass | -4.4835 | [-6.1714, -2.7336] | 16.7% | [$\pm$5.00] | **Inconclusive** |
| Actuator Saturation | Final Reserve Ratio (rho) | -0.2723 | [-0.2979, -0.2477] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Actuator Saturation | Total Dose | +42.0244 | [+39.4926, +44.5498] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Sediment | EC MAE | -1.3986 | [-3.6552, -0.1403] | 80.0% | [$\pm$0.04] | **LSTM advantage** |
| Sediment | Final Biomass | -122.7489 | [-140.7472, -99.3871] | 3.3% | [$\pm$5.00] | **Reserve rule advantage** |
| Sediment | Final Reserve Ratio (rho) | -0.1444 | [-0.1600, -0.1265] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Sediment | Total Dose | +777.4552 | [+732.8477, +817.6965] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Temp Spike | EC MAE | -7.9059 | [-12.9370, -3.6092] | 90.0% | [$\pm$0.04] | **LSTM advantage** |
| Temp Spike | Final Biomass | -77.4288 | [-115.7836, -34.0411] | 23.3% | [$\pm$5.00] | **Reserve rule advantage** |
| Temp Spike | Final Reserve Ratio (rho) | -0.1430 | [-0.1581, -0.1260] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Temp Spike | Total Dose | +698.6873 | [+644.3953, +750.6610] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Ec Drop | EC MAE | -3.7310 | [-7.1818, -0.6853] | 80.0% | [$\pm$0.04] | **LSTM advantage** |
| Ec Drop | Final Biomass | -108.6836 | [-137.5174, -75.3426] | 10.0% | [$\pm$5.00] | **Reserve rule advantage** |
| Ec Drop | Final Reserve Ratio (rho) | -0.1420 | [-0.1582, -0.1238] | 0.0% | [$\pm$0.04] | **Reserve rule advantage** |
| Ec Drop | Total Dose | +756.2906 | [+711.8728, +798.7956] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |
| Equal Weighted Aggregate | EC MAE | -6.1984 | [-7.3310, -5.1114] | 74.8% | [$\pm$0.04] | **LSTM advantage** |
| Equal Weighted Aggregate | Final Biomass | -77.9113 | [-88.0230, -67.4723] | 19.1% | [$\pm$5.00] | **Reserve rule advantage** |
| Equal Weighted Aggregate | Final Reserve Ratio (rho) | -0.1459 | [-0.1514, -0.1402] | 2.4% | [$\pm$0.04] | **Reserve rule advantage** |
| Equal Weighted Aggregate | Total Dose | +629.8584 | [+616.1724, +643.1985] | 0.0% | [$\pm$10.00] | **Reserve rule advantage** |

## 5. Control Effort Adjustment & Trade-Off Analysis

To prevent misinterpreting higher regulation precision that is achieved merely by excessive chemical dosing, performance differences are directly coupled with dosing effort:

| Scenario | $\Delta$ Biomass (LSTM - Rule) | $\Delta$ EC MAE (LSTM - Rule) | $\Delta$ Total Dose (LSTM - Rule) | $\Delta$ Total Dose (LSTM - PID) |
| :--- | :---: | :---: | :---: | :---: |
| Normal | -87.69 g/L [-126.08, -44.67] | -6.3555 [-11.2476, -2.2786] | +712.75 mL [+654.54, +771.41] | +249.17 mL [+202.18, +296.43] |
| Heatwave | -89.14 g/L [-128.32, -43.63] | -6.6262 [-11.5858, -2.5004] | +714.95 mL [+671.00, +758.07] | +253.33 mL [+216.79, +289.95] |
| Nutrient Depletion | -107.56 g/L [-136.98, -70.56] | -3.6641 [-6.9891, -0.6563] | +747.59 mL [+699.34, +794.88] | +263.67 mL [+217.79, +310.20] |
| Cold Shock | +65.25 g/L [+17.22, +111.35] | -22.4154 [-27.5182, -17.2420] | +641.33 mL [+598.59, +686.82] | +296.23 mL [+268.79, +324.46] |
| Noisy Sensors | -165.87 g/L [-196.64, -127.80] | -4.6878 [-8.8816, -1.4022] | +372.07 mL [+327.13, +413.61] | -105.26 mL [-136.17, -73.67] |
| Delayed Response | -75.78 g/L [-115.91, -31.00] | -7.6856 [-12.6050, -3.4612] | +703.31 mL [+651.58, +753.32] | +247.07 mL [+215.20, +279.48] |
| Actuator Failure | -82.89 g/L [-91.70, -73.49] | -0.0444 [-0.1387, +0.0125] | +761.98 mL [+726.53, +798.43] | +199.88 mL [+163.61, +237.33] |
| Actuator Saturation | -4.48 g/L [-6.17, -2.73] | -3.6680 [-3.6999, -3.6367] | +42.02 mL [+39.49, +44.55] | -59.17 mL [-61.81, -56.52] |
| Sediment | -122.75 g/L [-140.75, -99.39] | -1.3986 [-3.6552, -0.1403] | +777.46 mL [+732.85, +817.70] | +277.46 mL [+234.33, +316.63] |
| Temp Spike | -77.43 g/L [-115.78, -34.04] | -7.9059 [-12.9370, -3.6092] | +698.69 mL [+644.40, +750.66] | +245.22 mL [+212.21, +277.84] |
| Ec Drop | -108.68 g/L [-137.52, -75.34] | -3.7310 [-7.1818, -0.6853] | +756.29 mL [+711.87, +798.80] | +270.97 mL [+231.58, +309.64] |
| Equal Weighted Aggregate | -77.91 g/L [-88.02, -67.47] | -6.1984 [-7.3310, -5.1114] | +629.86 mL [+616.17, +643.20] | +194.42 mL [+183.46, +204.83] |

## 6. Aggregate Controller Summary (Mean ± 95% Bootstrap CI Across Seeds)

| Controller | Scenario | EC MAE (mS/cm) | Final Biomass (g/L) | Final Reserve (rho) | Total Dose (mL) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| No Action | Normal | 1.1662 [1.1659, 1.1665] | 6.21 [6.13, 6.29] | 0.0305 [0.0304, 0.0307] | 0.00 [0.00, 0.00] |
| No Action | Heatwave | 1.1664 [1.1660, 1.1667] | 6.16 [6.09, 6.22] | 0.0307 [0.0305, 0.0310] | 0.00 [0.00, 0.00] |
| No Action | Nutrient Depletion | 1.1665 [1.1660, 1.1669] | 6.21 [6.14, 6.27] | 0.0307 [0.0305, 0.0309] | 0.00 [0.00, 0.00] |
| No Action | Cold Shock | 1.1659 [1.1655, 1.1663] | 6.28 [6.19, 6.36] | 0.0306 [0.0303, 0.0308] | 0.00 [0.00, 0.00] |
| No Action | Noisy Sensors | 1.1662 [1.1658, 1.1666] | 6.17 [6.11, 6.22] | 0.0305 [0.0304, 0.0307] | 0.00 [0.00, 0.00] |
| No Action | Delayed Response | 1.1659 [1.1653, 1.1664] | 6.30 [6.23, 6.37] | 0.0304 [0.0302, 0.0306] | 0.00 [0.00, 0.00] |
| No Action | Actuator Failure | 1.1661 [1.1656, 1.1664] | 6.32 [6.25, 6.39] | 0.0306 [0.0304, 0.0308] | 0.00 [0.00, 0.00] |
| No Action | Actuator Saturation | 1.1661 [1.1657, 1.1666] | 6.27 [6.19, 6.34] | 0.0306 [0.0304, 0.0308] | 0.00 [0.00, 0.00] |
| No Action | Sediment | 1.1658 [1.1654, 1.1662] | 6.29 [6.22, 6.37] | 0.0305 [0.0303, 0.0307] | 0.00 [0.00, 0.00] |
| No Action | Temp Spike | 1.1659 [1.1654, 1.1664] | 6.18 [6.10, 6.26] | 0.0307 [0.0305, 0.0309] | 0.00 [0.00, 0.00] |
| No Action | Ec Drop | 1.1666 [1.1662, 1.1669] | 6.28 [6.20, 6.36] | 0.0307 [0.0305, 0.0308] | 0.00 [0.00, 0.00] |
| Rule Based Reserve | Normal | 7.3504 [3.2730, 12.2413] | 222.69 [177.38, 259.83] | 0.2408 [0.2297, 0.2507] | 410.61 [380.09, 447.37] |
| Rule Based Reserve | Heatwave | 7.6172 [3.4915, 12.5752] | 221.35 [176.97, 258.51] | 0.2414 [0.2288, 0.2524] | 411.88 [380.82, 449.78] |
| Rule Based Reserve | Nutrient Depletion | 4.6573 [1.6451, 7.9844] | 248.18 [218.36, 274.99] | 0.2487 [0.2398, 0.2559] | 388.28 [365.56, 412.51] |
| Rule Based Reserve | Cold Shock | 23.4059 [18.2331, 28.5069] | 81.77 [37.20, 127.24] | 0.1978 [0.1833, 0.2129] | 527.89 [488.70, 566.58] |
| Rule Based Reserve | Noisy Sensors | 5.7195 [2.4350, 9.9183] | 235.70 [198.72, 265.71] | 0.2469 [0.2367, 0.2558] | 394.94 [371.36, 425.94] |
| Rule Based Reserve | Delayed Response | 8.6737 [4.4524, 13.5931] | 211.10 [167.37, 249.38] | 0.2382 [0.2254, 0.2498] | 418.41 [386.94, 456.80] |
| Rule Based Reserve | Actuator Failure | 1.0359 [0.9800, 1.1293] | 222.38 [221.44, 223.23] | 0.1997 [0.1995, 0.1999] | 396.88 [393.68, 400.06] |
| Rule Based Reserve | Actuator Saturation | 4.7855 [4.7545, 4.8170] | 30.59 [29.45, 31.74] | 0.5065 [0.4817, 0.5315] | 93.35 [93.24, 93.47] |
| Rule Based Reserve | Sediment | 2.3891 [1.1302, 4.6493] | 265.88 [244.70, 278.52] | 0.2548 [0.2484, 0.2596] | 371.31 [362.72, 387.42] |
| Rule Based Reserve | Temp Spike | 8.8971 [4.5987, 13.9284] | 207.36 [162.78, 246.48] | 0.2404 [0.2275, 0.2520] | 419.29 [388.19, 457.36] |
| Rule Based Reserve | Ec Drop | 4.7250 [1.6783, 8.1732] | 246.07 [213.84, 273.20] | 0.2480 [0.2382, 0.2560] | 388.00 [364.98, 412.53] |
| Pid | Normal | 1.0881 [1.0822, 1.0938] | 216.97 [216.03, 217.95] | 0.2251 [0.2244, 0.2259] | 874.19 [871.48, 876.81] |
| Pid | Heatwave | 1.0879 [1.0824, 1.0933] | 216.29 [215.24, 217.37] | 0.2256 [0.2249, 0.2264] | 873.50 [870.09, 876.84] |
| Pid | Nutrient Depletion | 1.0879 [1.0824, 1.0932] | 216.25 [215.01, 217.42] | 0.2248 [0.2238, 0.2256] | 872.20 [868.79, 875.52] |
| Pid | Cold Shock | 1.0836 [1.0784, 1.0891] | 216.04 [215.09, 216.90] | 0.2252 [0.2245, 0.2259] | 873.00 [870.35, 875.43] |
| Pid | Noisy Sensors | 1.0853 [1.0801, 1.0909] | 215.96 [214.95, 216.99] | 0.2247 [0.2240, 0.2255] | 872.27 [869.83, 874.70] |
| Pid | Delayed Response | 1.0845 [1.0796, 1.0894] | 217.10 [216.16, 218.05] | 0.2249 [0.2242, 0.2257] | 874.65 [872.16, 877.04] |
| Pid | Actuator Failure | 1.0677 [1.0623, 1.0732] | 178.64 [178.00, 179.27] | 0.1655 [0.1644, 0.1665] | 958.98 [954.54, 963.21] |
| Pid | Actuator Saturation | 0.9695 [0.9677, 0.9714] | 60.83 [60.40, 61.26] | 0.1621 [0.1609, 0.1632] | 194.54 [193.96, 195.13] |
| Pid | Sediment | 1.0906 [1.0834, 1.0976] | 215.75 [214.62, 216.85] | 0.2250 [0.2243, 0.2257] | 871.30 [868.71, 873.96] |
| Pid | Temp Spike | 1.0888 [1.0832, 1.0945] | 215.50 [214.46, 216.52] | 0.2250 [0.2243, 0.2258] | 872.75 [870.02, 875.38] |
| Pid | Ec Drop | 1.0898 [1.0838, 1.0951] | 216.06 [214.90, 217.22] | 0.2245 [0.2239, 0.2251] | 873.32 [870.16, 876.58] |
| Lstm | Normal | 0.9949 [0.9893, 1.0011] | 135.01 [122.77, 147.38] | 0.1095 [0.0920, 0.1290] | 1123.36 [1077.12, 1170.12] |
| Lstm | Heatwave | 0.9910 [0.9877, 0.9944] | 132.21 [122.99, 142.02] | 0.0929 [0.0848, 0.1051] | 1126.83 [1089.33, 1163.78] |
| Lstm | Nutrient Depletion | 0.9933 [0.9885, 0.9985] | 140.62 [128.63, 153.07] | 0.1196 [0.1002, 0.1417] | 1135.87 [1089.87, 1183.11] |
| Lstm | Cold Shock | 0.9905 [0.9868, 0.9946] | 147.02 [138.58, 155.73] | 0.1154 [0.0982, 0.1351] | 1169.23 [1141.22, 1197.97] |
| Lstm | Noisy Sensors | 1.0317 [1.0249, 1.0382] | 69.83 [63.98, 76.11] | 0.0750 [0.0738, 0.0762] | 767.01 [735.84, 799.42] |
| Lstm | Delayed Response | 0.9881 [0.9840, 0.9920] | 135.32 [126.53, 144.31] | 0.1011 [0.0876, 0.1171] | 1121.72 [1089.61, 1154.08] |
| Lstm | Actuator Failure | 0.9914 [0.9874, 0.9955] | 139.49 [131.02, 148.68] | 0.0972 [0.0897, 0.1058] | 1158.86 [1124.27, 1194.46] |
| Lstm | Actuator Saturation | 1.1175 [1.1163, 1.1187] | 26.11 [24.73, 27.51] | 0.2341 [0.2310, 0.2371] | 135.38 [132.88, 137.87] |
| Lstm | Sediment | 0.9904 [0.9852, 0.9959] | 143.13 [133.01, 152.76] | 0.1104 [0.0955, 0.1276] | 1148.76 [1105.35, 1187.83] |
| Lstm | Temp Spike | 0.9912 [0.9876, 0.9949] | 129.93 [120.99, 139.25] | 0.0974 [0.0859, 0.1117] | 1117.98 [1084.92, 1150.78] |
| Lstm | Ec Drop | 0.9941 [0.9902, 0.9980] | 137.38 [126.94, 147.72] | 0.1060 [0.0919, 0.1223] | 1144.29 [1105.62, 1181.82] |
| Mpc Reserve | Normal | 41.6239 [31.9612, 51.1404] | 68.32 [24.34, 114.85] | 0.1167 [0.0722, 0.1622] | 816.49 [745.06, 885.63] |
| Mpc Reserve | Heatwave | 35.7718 [26.6832, 44.5967] | 87.63 [43.24, 141.97] | 0.1472 [0.1035, 0.1904] | 776.36 [711.47, 837.80] |
| Mpc Reserve | Nutrient Depletion | 30.3181 [19.9166, 41.1692] | 149.02 [94.32, 204.48] | 0.1671 [0.1159, 0.2155] | 720.88 [641.63, 801.02] |
| Mpc Reserve | Cold Shock | 43.1113 [34.7932, 51.1831] | 40.47 [8.72, 79.58] | 0.1175 [0.0766, 0.1593] | 824.22 [760.91, 884.67] |
| Mpc Reserve | Noisy Sensors | 38.0246 [27.7378, 47.8928] | 103.31 [53.37, 158.52] | 0.1337 [0.0865, 0.1831] | 782.64 [707.28, 855.83] |
| Mpc Reserve | Delayed Response | 37.5693 [27.3540, 48.4837] | 117.64 [63.22, 172.04] | 0.1338 [0.0832, 0.1824] | 779.31 [704.66, 858.96] |
| Mpc Reserve | Actuator Failure | 36.4429 [24.7374, 48.0802] | 106.11 [60.62, 153.38] | 0.1336 [0.0899, 0.1776] | 1060.14 [901.55, 1218.40] |
| Mpc Reserve | Actuator Saturation | 4.7936 [4.7627, 4.8245] | 29.70 [28.56, 30.86] | 0.5288 [0.5035, 0.5532] | 93.66 [93.55, 93.77] |
| Mpc Reserve | Sediment | 28.9584 [18.4095, 40.0921] | 175.32 [119.78, 230.57] | 0.1708 [0.1186, 0.2198] | 716.39 [640.78, 796.93] |
| Mpc Reserve | Temp Spike | 38.2982 [28.0485, 48.7846] | 107.29 [55.21, 162.38] | 0.1326 [0.0825, 0.1802] | 793.99 [719.47, 871.38] |
| Mpc Reserve | Ec Drop | 23.6480 [13.8575, 33.9662] | 184.28 [127.75, 241.35] | 0.1986 [0.1518, 0.2428] | 686.91 [614.02, 763.22] |

## 7. Episode Termination & Horizon Integrity

- **Total Controller Executions**: 1650
- **Completed Full Horizon (3500 steps)**: 1532
- **Early Terminations**: 118

| Controller | Scenario | Termination Reason | Episodes Terminated |
| :--- | :--- | :--- | :---: |
| mpc_reserve | cold_shock | health_collapse | 14 |
| mpc_reserve | delayed_response | health_collapse | 15 |
| mpc_reserve | ec_drop | health_collapse | 8 |
| mpc_reserve | heatwave | health_collapse | 11 |
| mpc_reserve | noisy_sensors | health_collapse | 15 |
| mpc_reserve | normal | health_collapse | 16 |
| mpc_reserve | nutrient_depletion | health_collapse | 12 |
| mpc_reserve | sediment | health_collapse | 12 |
| mpc_reserve | temp_spike | health_collapse | 15 |

## 8. Benchmark Discriminability Findings

1. **Statistical Separation from PID**: Across repeated stochastic trajectories, paired differences between LSTM and PID confirm whether the learned policy exhibits statistically distinct regulation and biomass yields with confidence intervals excluding zero.
2. **Residual Control Complexity Beyond Reserve Heuristic**: Practical equivalence testing examines whether the simple reserve-ratio controller ($-\delta < \Delta < +\delta$) captures the actionable control structure under baseline conditions, and highlights which stress regimes (e.g., heatwave vs nutrient depletion) reveal significant divergences.
3. **Control Effort Discrepancy**: The observed performance difference is evaluated in context with nutrient dosage volumes to assess whether learned regulation relies on different dosing efficiency.

## 9. Reproducibility & Provenance Information

- **Seed Schedule**: Saved in `seed_schedule.json` (30 seeds per scenario)
- **Configuration Snapshot**: Saved in `evaluation_config.yaml`
- **Raw Episode Data**: Saved in `episode_results.csv` and `episode_results.json`
- **Full Paired Differences**: Saved in `paired_differences.csv`
- **Bootstrap CIs**: Saved in `bootstrap_confidence_intervals.csv`
- **Equivalence Analysis**: Saved in `equivalence_analysis.csv`
- **Supplementary Wilcoxon Tests**: Saved in `wilcoxon_results.csv`