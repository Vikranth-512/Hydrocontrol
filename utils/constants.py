"""
Controller observation constants.

This module defines the single source of truth for numerical constants used
across the controller observation pipeline (feature engineering, PID control,
closed-loop evaluation, and validation tests).

N_OBS_TARGET derivation
-----------------------
Derived from the benchmark reference operating regime using the following procedure:

  1. Reference regime: synthetic training trajectories, normal scenario only,
     filtered to steps where true_ec in [0.9, 1.5]  (the benchmark EC target window).
  2. Observable ratio computed at each such step:
       N_obs = true_ec / max(turbidity, EPSILON)
  3. Target = median(N_obs) over the reference window.

Result from data/synthetic/all_trajectories.csv (n=127 reference steps):
  median(N_obs) = 0.006006
  mean(N_obs)   = 0.006104

This is consistent with the physics-based estimate:
  EC_ref / Turbidity_ref = 1.2 / (2.2 x 80) = 1.2 / 176 approx 0.006818
The small difference reflects actual sensor dynamics (turbidity lag, biomass
distribution in training data).

The target is fixed from training data. It was NOT selected by optimising PID
performance on validation or test trajectories.

MIN_VALID_TURBIDITY derivation
-------------------------------
From data/synthetic/all_trajectories.csv (200 trajectories, all scenarios):
  turbidity min  = 151.25    (full dataset)
  turbidity p1   = 176.30
  turbidity p5   = 181.16

Turbidity below 150 never occurs in the synthetic dataset.
Sensor model: turbidity approx biomass_optical_factor x biomass = 2.2 x biomass
  turbidity = 150 corresponds to biomass approx 68 g/L (near-crash threshold)

MIN_VALID_TURBIDITY = 50.0 is chosen as a conservative control-validity
threshold below which the ratio is treated as unreliable. This is below any
observed training value and provides headroom for scenarios outside the
training distribution (severe culture crash).

EPSILON vs MIN_VALID_TURBIDITY
-------------------------------
EPSILON = 1e-8: purely a numerical division safeguard. Does not affect PID
  control behaviour in any realistic operating condition.
MIN_VALID_TURBIDITY = 50.0: control-validity threshold. When turbidity falls
  below this value, the ratio is clamped to avoid runaway PID output from a
  physically unreliable measurement.
"""

# ---------------------------------------------------------------------------
# Numerical division safeguard
# ---------------------------------------------------------------------------
EPSILON: float = 1e-8

# ---------------------------------------------------------------------------
# Reference operating point (for N_OBS_TARGET derivation)
# ---------------------------------------------------------------------------
EC_TARGET: float = 1.2           # benchmark dosing reference (mS/cm)
TURBIDITY_REFERENCE: float = 176.0   # biomass_optical_factor x nominal_biomass (2.2 x 80)

# ---------------------------------------------------------------------------
# N_obs PID setpoint -- derived from reference-regime simulation
# ---------------------------------------------------------------------------
#   median(EC / turbidity) over normal-scenario steps where true_ec in [0.9, 1.5]
#   Source: data/synthetic/all_trajectories.csv, n=127 reference steps
N_OBS_TARGET: float = 0.006006

# ---------------------------------------------------------------------------
# Control-validity turbidity threshold
# ---------------------------------------------------------------------------
#   Below this value the EC/turbidity ratio becomes unreliable as a control
#   signal (very low biomass / culture crash regime).
#   Turbidity min in training data: 151.25; this threshold is well below that.
#   Sensor model: turbidity = 2.2 x biomass  ->  50.0 approx biomass 22.7 g/L
MIN_VALID_TURBIDITY: float = 50.0

# Maximum credible N_obs (clamp output when turbidity is very low)
# EC_max_safe (approx 2.5) / MIN_VALID_TURBIDITY = 2.5/50 = 0.05
N_OBS_MAX: float = 0.05

# ---------------------------------------------------------------------------
# Canonical controller observation feature list (POMDP observation contract)
# ---------------------------------------------------------------------------
OBSERVABLE_FEATURES = [
    "water_temp",
    "ec",
    "turbidity",
    "prev_flowrate",
    "prev_duration",
    "time_since_last_dose",
    "ph",
    "dissolved_oxygen",
    "ambient_temp",
    "delta_ec",
    "delta_temp",
    "delta_turbidity",
    "rolling_avg_ec",
    "rolling_std_ec",
    "ec_error",
    "ec_turbidity_ratio",   # observable external nutrient-to-biomass proxy
    "dosing_frequency",
]

# Hidden state names that must NEVER appear in the controller observation vector
_FORBIDDEN_CONTROLLER_FEATURES = frozenset([
    "reserve_ratio",
    "internal_reserve",
    "rho",
    "health_index",
    "damage_index",
    "dead_biomass_pool",
])
