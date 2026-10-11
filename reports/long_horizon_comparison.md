# 3500-Step Long-Horizon Controller Benchmark Comparison

| Controller | EC MAE | EC RMSE | Final EC | Final Biomass | Final Reserve Ratio | Intervention Fraction | Total Dose | Effective Horizon | Terminated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No-action | 1.1654 | 1.1666 | 0.0149 | 6.61 | 0.0303 | 0.0000 | 0.00 | 3500 | NO |
| Rule-based | 1.0067 | 1.0740 | 0.0262 | 279.62 | 0.2566 | 0.2040 | 364.50 | 3500 | NO |
| PID | 1.0913 | 1.1207 | 0.0200 | 214.91 | 0.2254 | 0.7369 | 874.29 | 3500 | NO |
| LSTM | 1.0374 | 1.0760 | 0.0044 | 68.98 | 0.0692 | 0.3871 | 831.35 | 3500 | NO |

### Controller Information-Set Audit

| Controller | Reserve Ratio (rho) | Biomass (B) | EC | 17-Feature Sensor History | Nature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **No-action** | — | — | — | — | Uncontrolled baseline, open-loop |
| **Rule-based** | Yes | — | — | — | Reserve-ratio feedback heuristic |
| **PID** | Yes | Yes (scale) | Yes (headroom) | — | Classical feedback control |
| **LSTM** | Yes | (via features) | Yes | Yes | Learned policy (imitation) |
