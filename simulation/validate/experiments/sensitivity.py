"""
VII. Parameter Identifiability — PAR-03
Normalized Parameter Sensitivity Matrix Analysis

PURPOSE
-------
Provides quantitative local sensitivity evidence to support or qualify the
manuscript's parameter-identifiability claim.

MATHEMATICAL DEFINITION
-----------------------
For each mechanistic parameter theta_j (nominal value p_j) and each
observable trajectory feature y_i (nominal value at baseline), the
normalized (dimensionless, relative) local sensitivity is:

    S_ij = (theta_j / y_i) * (d y_i / d theta_j)

estimated using a central finite difference:

    S_ij ≈ [y_i(p_j + Delta) - y_i(p_j - Delta)] / (2 * Delta)
            * (p_j / y_scale_i)

where y_scale_i = max(|y_i_nominal|, minimum_output_scale) provides
a stable scale guard for near-zero outputs.

This normalization makes S_ij dimensionless and directly comparable
across parameters and outputs with different physical units and magnitudes.

IMPORTANT DISTINCTIONS
----------------------
* This analysis characterises LOCAL sensitivity at the benchmark's
  nominal operating point under the tested conditions.
* A full-rank, well-conditioned S_obs provides evidence of
  LOCAL PRACTICAL IDENTIFIABILITY for the evaluated parameter subset.
* It does NOT establish structural (global) identifiability.
* It does NOT prove that all parameters are independently recoverable
  from real-world data subject to measurement noise.
* Claims are automatically moderated based on the actual numerical results.

OBSERVABLES
-----------
Primary (sensor-observable) matrix S_obs (6 x n_params):
    ec_mean, ec_final, ec_std,
    turbidity_mean, turbidity_final, turbidity_std

Supplementary latent-state diagnostic matrix S_lat (3 x n_params):
    biomass_mean, biomass_final, biomass_std

DISTINCT FROM PAR-01 (OAT)
---------------------------
PAR-01 asks: "How does each output respond when a single parameter is changed?"
PAR-03 asks: "How do all observable outputs respond to all parameters in a common
              normalized representation, and are the sensitivity directions
              jointly distinguishable across the observable space?"

PAR-03 adds on top of OAT logic:
  - Common dimensionless normalization across all outputs and parameters
  - Full matrix representation enabling SVD/rank/conditioning analysis
  - Sensitivity-vector collinearity (absolute cosine similarity)
  - Multi-scenario aggregation with explicit RMS formula
  - Perturbation convergence analysis across multiple delta fractions
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from simulation.dynamics import TankDynamicsParams, step_dynamics
from simulation.validate.common import Experiment, make_initial_state, simulate_and_record


# ---------------------------------------------------------------------------
# Configuration constants (mirrored from configs/default.yaml block)
# These are the defaults; the YAML block takes precedence when loaded.
# ---------------------------------------------------------------------------

#: Relative perturbation fractions to evaluate
DEFAULT_PERTURBATION_FRACTIONS: List[float] = [0.005, 0.01, 0.02]

#: Minimum output magnitude used as y_scale guard (prevents divide-by-near-zero)
DEFAULT_MIN_OUTPUT_SCALE: float = 0.01

#: Absolute cosine-similarity threshold for flagging collinear parameter pairs
#: Uses |cos(theta)| >= threshold to catch both parallel and anti-parallel directions.
DEFAULT_COLLINEARITY_THRESHOLD: float = 0.95

#: Relative singular-value threshold for numerical rank determination.
#: A singular value sigma_k is treated as zero if sigma_k / sigma_max < tol.
DEFAULT_RANK_TOLERANCE: float = 1.0e-4  # relative

#: Fixed random seed for all scenario trajectories
DEFAULT_RANDOM_SEED: int = 42

#: Simulation timestep (seconds)
DT: float = 60.0

#: Trajectory length per scenario (steps)
SCENARIO_LENGTH: int = 800


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------

#: Explicit list of mechanistic simulator parameters to include.
#: Each entry: (attribute_name, lower_bound_clamp, category_label)
#:
#: Parameters excluded from this list:
#:   sensor_gain_ec, biomass_optical_factor, algae_sensor_tau
#:       -> sensor-layer parameters; do not affect biological dynamics
#:   ec_decay_jitter, gain_jitter
#:       -> stochastic noise parameters; zero at nominal
#:   release_damping, immediate_absorption_fraction, delay_kernel, max_queue_release_rate
#:       -> transport/actuator parameters with narrow identifiable range;
#:          delay_kernel is a tuple (not a scalar), excluded from scalar perturbation
#:   starvation_damage_multiplier, starvation_reserve_threshold, osmotic_safe_threshold
#:       -> threshold/switch parameters; sensitivity is near-discontinuous
#:   basal_repair_rate
#:       -> second-order repair term dominated by repair_rate
#:   temp_ref, temp_opt, temp_sigma, temperature_rate_modifier, ambient_temp_mean, temp_relaxation_rate
#:       -> environmental/reference parameters; not biological model parameters
#:   harvest_rate
#:       -> operational parameter, not biological mechanism
#:   maintenance_to_detritus_fraction, maintenance_to_dissolved_fraction, maintenance_lost_fraction
#:       -> fractions that sum to 1 with hard constraint; not independently identifiable
#:   growth_metabolic_efficiency, nutrient_retention_efficiency
#:       -> combined efficiency terms; partially redundant with growth_yield
PAR03_PARAMETERS: List[Tuple[str, float, str]] = [
    # (attribute_name,               lower_clamp,   category)
    ("maximum_uptake_rate",          1e-6,          "Uptake"),
    ("half_saturation_mass",         1e-3,          "Uptake"),
    ("internal_capacity",            0.01,          "Reserve"),
    ("maximum_growth_rate",          1e-6,          "Growth"),
    ("growth_yield",                 0.01,          "Growth"),
    ("maintenance_cost",             1e-7,          "Maintenance"),
    ("biomass_nutrient_content",     0.01,          "Structural"),
    ("mortality_rate",               1e-7,          "Mortality"),
    ("mineralization_rate",          1e-6,          "Recycling"),
    ("mineralization_efficiency",    0.01,          "Recycling"),
    ("damage_rate",                  1e-8,          "Health"),
    ("repair_rate",                  1e-5,          "Health"),
    ("osmotic_half_effect",          0.1,           "Osmotic"),
    ("light_attenuation_half_mass",  1.0,           "Light/Density"),
    ("q10_coefficient",              1.0,           "Temperature"),
    ("background_dilution_rate",     1e-6,          "Transport"),
]

PARAM_NAMES: List[str] = [p[0] for p in PAR03_PARAMETERS]
PARAM_LOWER_CLAMPS: Dict[str, float] = {p[0]: p[1] for p in PAR03_PARAMETERS}
PARAM_CATEGORIES: Dict[str, str] = {p[0]: p[2] for p in PAR03_PARAMETERS}


# ---------------------------------------------------------------------------
# Observable feature selection
# ---------------------------------------------------------------------------

#: Primary (sensor-observable) features — used for S_obs matrix.
#: These are quantities measurable by the benchmark's sensor interface.
OBSERVABLE_FEATURES: List[str] = [
    "ec_mean",
    "ec_final",
    "ec_std",
    "turbidity_mean",
    "turbidity_final",
    "turbidity_std",
]

#: Supplementary latent-state diagnostic features — used for S_lat matrix.
#: Biomass is a hidden state NOT directly observable in the benchmark.
#: Reported separately; does not substitute for observable-based analysis.
LATENT_FEATURES: List[str] = [
    "biomass_mean",
    "biomass_final",
    "biomass_std",
]

ALL_FEATURES: List[str] = OBSERVABLE_FEATURES + LATENT_FEATURES

# ---------------------------------------------------------------------------
# Publication-quality label maps
# ---------------------------------------------------------------------------

#: Human-readable labels for observable feature names (y-axis of heatmaps).
OBSERVABLE_LABELS: Dict[str, str] = {
    "ec_mean":           "EC mean",
    "ec_final":          "EC final",
    "ec_std":            "EC SD",
    "turbidity_mean":    "Turbidity mean",
    "turbidity_final":   "Turbidity final",
    "turbidity_std":     "Turbidity SD",
    "biomass_mean":      "Biomass mean",
    "biomass_final":     "Biomass final",
    "biomass_std":       "Biomass SD",
}

#: Concise mathematical / manuscript notation for the 16 analysed parameters
#: (x-axis of heatmaps). Names follow the notation used in the dynamics model
#: docstrings and the manuscript parameter table.
PARAM_LABELS: Dict[str, str] = {
    "maximum_uptake_rate":          r"$V_{\mathrm{uptake,max}}$",
    "half_saturation_mass":         r"$K_s$",
    "internal_capacity":            r"$C_{\mathrm{int,max}}$",
    "maximum_growth_rate":          r"$\mu_{\mathrm{growth,max}}$",
    "growth_yield":                 r"$Y_{\mathrm{growth}}$",
    "maintenance_cost":             r"$C_{\mathrm{maint}}$",
    "biomass_nutrient_content":     r"$C_{\mathrm{N,bio}}$",
    "mortality_rate":               r"$k_{\mathrm{mort}}$",
    "mineralization_rate":          r"$k_{\mathrm{min}}$",
    "mineralization_efficiency":    r"$\eta_{\mathrm{min}}$",
    "damage_rate":                  r"$k_{\mathrm{damage}}$",
    "repair_rate":                  r"$k_{\mathrm{repair}}$",
    "osmotic_half_effect":          r"$C_{\mathrm{osm,half}}$",
    "light_attenuation_half_mass":  r"$M_{\mathrm{light,half}}$",
    "q10_coefficient":              r"$Q_{10}$",
    "background_dilution_rate":     r"$k_{\mathrm{dil,bg}}$",
}


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

#: Each scenario: dict with keys understood by _run_scenario_trajectory.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "normal": {
        "description": "Nominal operating conditions, moderate continuous dosing.",
        "dissolved_mass": 2.0,
        "biomass": 100.0,
        "water_temp": 22.0,
        "actions": [(1.0, 10.0)] * SCENARIO_LENGTH,
    },
    "heatwave": {
        "description": "Elevated temperature (33°C), tests thermal parameter sensitivity.",
        "dissolved_mass": 2.0,
        "biomass": 100.0,
        "water_temp": 33.0,
        "actions": [(1.0, 10.0)] * SCENARIO_LENGTH,
    },
    "nutrient_depletion": {
        "description": "Low initial dissolved nutrients, tests uptake and reserve dynamics.",
        "dissolved_mass": 0.3,
        "biomass": 100.0,
        "water_temp": 22.0,
        "actions": [(0.2, 5.0)] * SCENARIO_LENGTH,
    },
    "cold_shock": {
        "description": "Cold temperature (14°C), tests Q10 and thermal response.",
        "dissolved_mass": 2.0,
        "biomass": 100.0,
        "water_temp": 14.0,
        "actions": [(1.0, 10.0)] * SCENARIO_LENGTH,
    },
}


# ---------------------------------------------------------------------------
# Core trajectory and feature utilities
# ---------------------------------------------------------------------------

def _run_scenario_trajectory(
    params: TankDynamicsParams,
    scenario_cfg: Dict[str, Any],
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Run a single deterministic trajectory for a given scenario configuration.

    The random seed is fixed so that nominal and perturbed runs share an
    identical stochastic disturbance realization. The ONLY intended difference
    between paired runs is the single perturbed parameter.

    Parameters
    ----------
    params : TankDynamicsParams
        Model parameters (may be perturbed relative to nominal).
    scenario_cfg : dict
        Scenario definition from SCENARIOS.
    seed : int
        Fixed RNG seed for determinism.

    Returns
    -------
    pd.DataFrame
        Full trajectory with one row per timestep.
    """
    rng = np.random.default_rng(seed)
    s0 = make_initial_state(
        params,
        dissolved_mass=scenario_cfg["dissolved_mass"],
        biomass=scenario_cfg["biomass"],
        water_temp=scenario_cfg["water_temp"],
    )
    return simulate_and_record(
        params,
        length=SCENARIO_LENGTH,
        dt=DT,
        actions=scenario_cfg["actions"],
        initial_state=s0,
        rng=rng,
    )


def _extract_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract trajectory-level summary features from a simulation DataFrame.

    Returns all observable and latent features. NaN values are preserved
    so that downstream code can detect simulation failures.
    """
    feats: Dict[str, float] = {}

    for col, prefix in [("ec", "ec"), ("turbidity", "turbidity"), ("algae_biomass", "biomass")]:
        vals = df[col].values if col in df.columns else np.array([np.nan])
        feats[f"{prefix}_mean"] = float(np.mean(vals))
        feats[f"{prefix}_final"] = float(vals[-1])
        feats[f"{prefix}_std"] = float(np.std(vals))

    return feats


# ---------------------------------------------------------------------------
# Finite-difference sensitivity computation
# ---------------------------------------------------------------------------

def _perturb_params(
    base_params: TankDynamicsParams,
    pname: str,
    delta: float,
    lower_clamp: float,
) -> Tuple[TankDynamicsParams, float, bool]:
    """
    Create a parameter copy perturbed by +delta, clamped to valid domain.

    Returns (perturbed_params, actual_delta_used, was_clamped).
    """
    p_nom = float(getattr(base_params, pname))
    p_new = p_nom + delta
    clamped = False

    if p_new < lower_clamp:
        p_new = lower_clamp
        clamped = True

    p_mod = copy.deepcopy(base_params)
    setattr(p_mod, pname, p_new)
    actual_delta = p_new - p_nom
    return p_mod, actual_delta, clamped


def _central_fd_sensitivity(
    params: TankDynamicsParams,
    pname: str,
    scenario_cfg: Dict[str, Any],
    perturbation_fraction: float,
    seed: int,
    min_output_scale: float = DEFAULT_MIN_OUTPUT_SCALE,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute normalized central finite-difference sensitivity for one parameter
    across all observable and latent features in one scenario.

    Sensitivity definition (dimensionless):
        S_ij = (p_j / y_scale_i) * (y_i_hi - y_i_lo) / (p_j_hi - p_j_lo)

    where y_scale_i = max(|y_i_nominal|, min_output_scale).

    Returns
    -------
    sensitivities : dict {feature_name -> S_ij}
    metadata : dict with diagnostics (actual deltas, clamp flags, etc.)
    """
    lower_clamp = PARAM_LOWER_CLAMPS[pname]
    p_nom = float(getattr(params, pname))
    raw_delta = p_nom * perturbation_fraction

    # Positive perturbation
    p_hi_mod, delta_hi, clamped_hi = _perturb_params(params, pname, +raw_delta, lower_clamp)
    # Negative perturbation
    p_lo_mod, delta_lo_neg, clamped_lo = _perturb_params(params, pname, -raw_delta, lower_clamp)

    delta_lo = abs(delta_lo_neg)  # actual magnitude of negative perturbation
    asymmetric = clamped_hi or clamped_lo

    # Run trajectories — SAME seed guarantees identical disturbance realization
    df_hi = _run_scenario_trajectory(p_hi_mod, scenario_cfg, seed=seed)
    df_lo = _run_scenario_trajectory(p_lo_mod, scenario_cfg, seed=seed)
    df_nom = _run_scenario_trajectory(params, scenario_cfg, seed=seed)

    feats_hi = _extract_features(df_hi)
    feats_lo = _extract_features(df_lo)
    feats_nom = _extract_features(df_nom)

    # Effective denominator: p_hi - p_lo = delta_hi + delta_lo (both positive magnitudes)
    effective_denom = delta_hi + delta_lo
    if effective_denom < 1e-15:
        # Perturbation was entirely clipped — cannot compute sensitivity
        sensitivities = {f: np.nan for f in ALL_FEATURES}
        meta = {
            "parameter": pname,
            "p_nominal": p_nom,
            "perturbation_fraction": perturbation_fraction,
            "delta_hi": delta_hi,
            "delta_lo": delta_lo,
            "asymmetric": asymmetric,
            "valid": False,
            "reason": "effective_denominator_zero",
        }
        return sensitivities, meta

    sensitivities: Dict[str, float] = {}
    for feat in ALL_FEATURES:
        y_nom = feats_nom.get(feat, np.nan)
        y_hi = feats_hi.get(feat, np.nan)
        y_lo = feats_lo.get(feat, np.nan)

        if any(np.isnan(v) for v in [y_nom, y_hi, y_lo]):
            sensitivities[feat] = np.nan
            continue

        # Scale guard: use max of absolute nominal value and minimum scale
        y_scale = max(abs(y_nom), min_output_scale)

        # Normalized central FD:  S = (p_nom / y_scale) * (y_hi - y_lo) / (p_hi - p_lo)
        dy = y_hi - y_lo
        s_ij = (p_nom / y_scale) * (dy / effective_denom)
        sensitivities[feat] = float(s_ij)

    meta = {
        "parameter": pname,
        "p_nominal": p_nom,
        "perturbation_fraction": perturbation_fraction,
        "raw_delta": raw_delta,
        "delta_hi": delta_hi,
        "delta_lo": delta_lo,
        "effective_denom": effective_denom,
        "asymmetric": asymmetric,
        "clamped_hi": clamped_hi,
        "clamped_lo": clamped_lo,
        "valid": True,
    }
    return sensitivities, meta


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def _build_matrix(
    params: TankDynamicsParams,
    scenario_name: str,
    perturbation_fraction: float,
    seed: int,
    min_output_scale: float = DEFAULT_MIN_OUTPUT_SCALE,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Build the full sensitivity matrix for one scenario at one perturbation level.

    Returns
    -------
    S_obs : np.ndarray  shape (n_obs_features, n_params)
        Primary observable sensitivity matrix.
    S_full : np.ndarray  shape (n_all_features, n_params)
        Full matrix including latent-state features.
    row_meta : list of metadata dicts (one per parameter)
    """
    scenario_cfg = SCENARIOS[scenario_name]
    n_params = len(PARAM_NAMES)
    n_obs = len(OBSERVABLE_FEATURES)
    n_all = len(ALL_FEATURES)

    S_full = np.full((n_all, n_params), np.nan)
    row_meta: List[Dict[str, Any]] = []

    for j, pname in enumerate(PARAM_NAMES):
        sens, meta = _central_fd_sensitivity(
            params, pname, scenario_cfg,
            perturbation_fraction, seed, min_output_scale,
        )
        meta["scenario"] = scenario_name
        row_meta.append(meta)

        for i, feat in enumerate(ALL_FEATURES):
            S_full[i, j] = sens.get(feat, np.nan)

    S_obs = S_full[:n_obs, :]
    return S_obs, S_full, row_meta


# ---------------------------------------------------------------------------
# SVD and matrix diagnostics
# ---------------------------------------------------------------------------

def _svd_diagnostics(
    S: np.ndarray,
    rank_tol_relative: float = DEFAULT_RANK_TOLERANCE,
    label: str = "",
) -> Dict[str, Any]:
    """
    Compute SVD-based diagnostics for a sensitivity matrix.

    Rank is determined using a RELATIVE threshold:
        sigma_k is considered nonzero if sigma_k / sigma_max >= rank_tol_relative.

    This avoids the ambiguity of absolute thresholds when matrix entries
    span orders of magnitude.

    Parameters
    ----------
    S : np.ndarray
        Sensitivity matrix, shape (n_outputs, n_params).
    rank_tol_relative : float
        Fraction of largest singular value below which sigma is treated as zero.
    label : str
        Diagnostic label for reporting.
    """
    # Drop rows/cols that are entirely NaN (invalid perturbations)
    valid_rows = ~np.all(np.isnan(S), axis=1)
    valid_cols = ~np.all(np.isnan(S), axis=0)
    S_clean = np.nan_to_num(S[np.ix_(valid_rows, valid_cols)], nan=0.0)

    n_rows, n_cols = S_clean.shape
    if n_rows == 0 or n_cols == 0:
        return {"error": "empty_matrix_after_nan_removal", "label": label}

    try:
        U, sigma, Vt = np.linalg.svd(S_clean, full_matrices=False)
    except np.linalg.LinAlgError as e:
        return {"error": str(e), "label": label}

    sigma_max = float(sigma[0]) if len(sigma) > 0 else 0.0

    if sigma_max < 1e-15:
        # Effectively zero matrix
        return {
            "label": label,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "singular_values": sigma.tolist(),
            "sigma_max": 0.0,
            "sigma_min": 0.0,
            "numerical_rank": 0,
            "rank_deficiency": n_cols,
            "condition_number": float("inf"),
            "rank_tol_relative": rank_tol_relative,
            "rank_fraction": 0.0,
        }

    # Relative rank determination
    numerical_rank = int(np.sum(sigma / sigma_max >= rank_tol_relative))
    rank_deficiency = n_cols - numerical_rank
    sigma_min_nonzero = float(sigma[numerical_rank - 1]) if numerical_rank > 0 else 0.0
    condition_number = sigma_max / sigma_min_nonzero if sigma_min_nonzero > 1e-15 else float("inf")

    return {
        "label": label,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_params": n_cols,
        "n_outputs": n_rows,
        "singular_values": [float(s) for s in sigma],
        "sigma_max": float(sigma_max),
        "sigma_min_nonzero": sigma_min_nonzero,
        "numerical_rank": numerical_rank,
        "rank_deficiency": rank_deficiency,
        "rank_fraction": float(numerical_rank) / n_cols,
        "condition_number": condition_number,
        "rank_tol_relative": rank_tol_relative,
    }


# ---------------------------------------------------------------------------
# Collinearity analysis
# ---------------------------------------------------------------------------

def _collinearity_analysis(
    S: np.ndarray,
    param_names: List[str],
    threshold: float = DEFAULT_COLLINEARITY_THRESHOLD,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute pairwise absolute cosine similarity between parameter sensitivity
    vectors (columns of S).

    Uses |cos(theta)| >= threshold to identify both parallel (S_j ~ S_k)
    and anti-parallel (S_j ~ -S_k) collinear pairs, as both represent
    parameter directions that are difficult to distinguish independently.

    Parameters
    ----------
    S : np.ndarray  shape (n_outputs, n_params)
    param_names : list of str
    threshold : float   |cos| threshold for flagging (default 0.95)

    Returns
    -------
    collinear_pairs : pd.DataFrame  with flagged parameter pairs
    cosine_matrix : np.ndarray  shape (n_params, n_params)  full |cos| matrix
    """
    n_params = S.shape[1]
    # Replace NaN with 0 for cosine computation
    S_clean = np.nan_to_num(S, nan=0.0)

    # Column norms
    norms = np.linalg.norm(S_clean, axis=0)  # shape (n_params,)
    # Safe normalization
    safe_norms = np.where(norms > 1e-15, norms, 1.0)

    S_normed = S_clean / safe_norms[np.newaxis, :]  # shape (n_outputs, n_params)
    cosine_matrix = np.abs(S_normed.T @ S_normed)  # |cos|, shape (n_params, n_params)
    # Diagonal is 1 by construction; set to NaN to suppress self-pairs
    np.fill_diagonal(cosine_matrix, np.nan)

    rows = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            cos_ij = float(cosine_matrix[i, j])
            if np.isnan(cos_ij):
                continue
            rows.append({
                "parameter_a": param_names[i],
                "parameter_b": param_names[j],
                "abs_cosine_similarity": cos_ij,
                "flagged": cos_ij >= threshold,
            })

    df = pd.DataFrame(rows).sort_values("abs_cosine_similarity", ascending=False) if rows else pd.DataFrame()
    np.fill_diagonal(cosine_matrix, 1.0)  # Restore diagonal for plotting
    return df, cosine_matrix


# ---------------------------------------------------------------------------
# Perturbation convergence analysis
# ---------------------------------------------------------------------------

def _perturbation_convergence(
    matrices: Dict[float, np.ndarray],
    param_names: List[str],
    feature_names: List[str],
    epsilon: float = 1e-3,
) -> pd.DataFrame:
    """
    Compare sensitivity estimates across perturbation fractions to assess
    whether the finite-difference approximation has converged locally.

    For each (output, parameter) entry, computes a relative deviation:

        E = |S(frac_max) - S(frac_min)| / max(|S(frac_min)|, epsilon)

    Small E indicates the sensitivity estimate is stable and genuinely local.
    Large E may indicate nonlinearity, numerical noise, or a near-discontinuity.

    Parameters
    ----------
    matrices : dict {perturbation_fraction -> np.ndarray shape (n_outputs, n_params)}
    param_names, feature_names : list of str
    epsilon : float  guard for near-zero reference values

    Returns
    -------
    pd.DataFrame  with per-parameter statistics
    """
    fracs = sorted(matrices.keys())
    if len(fracs) < 2:
        return pd.DataFrame()

    frac_min = fracs[0]
    frac_max = fracs[-1]
    S_min = matrices[frac_min]
    S_max = matrices[frac_max]

    # All intermediate matrices for sign-consistency check
    rows = []
    for j, pname in enumerate(param_names):
        col_values = []  # sensitivities at all fractions for this param (avg over outputs)
        e_vals = []

        for i, feat in enumerate(feature_names):
            s_ref = S_min[i, j]
            s_end = S_max[i, j]
            if np.isnan(s_ref) or np.isnan(s_end):
                continue
            e = abs(s_end - s_ref) / max(abs(s_ref), epsilon)
            e_vals.append(e)

        # Sign consistency across all fracs for this parameter column
        sign_consistent = True
        for k_idx in range(len(feature_names)):
            signs_across_fracs = []
            for frac in fracs:
                v = matrices[frac][k_idx, j]
                if not np.isnan(v) and abs(v) > epsilon:
                    signs_across_fracs.append(np.sign(v))
            if len(set(signs_across_fracs)) > 1:
                sign_consistent = False

        rows.append({
            "parameter": pname,
            "category": PARAM_CATEGORIES.get(pname, "unknown"),
            "max_relative_deviation": float(np.max(e_vals)) if e_vals else np.nan,
            "mean_relative_deviation": float(np.mean(e_vals)) if e_vals else np.nan,
            "sign_consistent": sign_consistent,
            "n_valid_outputs": len(e_vals),
            "frac_min": frac_min,
            "frac_max": frac_max,
        })

    return pd.DataFrame(rows).sort_values("max_relative_deviation", ascending=False)


# ---------------------------------------------------------------------------
# RMS aggregation across scenarios
# ---------------------------------------------------------------------------

def _rms_aggregate(
    matrices_by_scenario: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Aggregate sensitivity matrices across scenarios using element-wise RMS:

        S_RMS_ij = sqrt( (1/N) * sum_k S_ij_k^2 )

    where k indexes scenarios and N is the number of scenarios.

    Note: RMS removes sign information. Sign information is preserved
    separately in per-scenario matrices and the convergence table.

    Parameters
    ----------
    matrices_by_scenario : dict {scenario_name -> np.ndarray (n_out, n_params)}

    Returns
    -------
    np.ndarray  RMS-aggregated matrix, same shape as inputs.
    """
    stacked = np.array(list(matrices_by_scenario.values()))  # (N, n_out, n_params)
    return float(1.0) * np.sqrt(np.nanmean(stacked ** 2, axis=0))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_heatmap(
    S: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    title: str,
    save_path: Path,
    vmax: Optional[float] = None,
    symmetric: bool = True,
    show_colorbar: bool = True,
) -> None:
    """
    Publication-quality diverging heatmap of the sensitivity matrix.

    Uses RdBu_r colormap centred at zero so that positive/negative/near-zero
    sensitivities are immediately apparent. Annotates cell values when the
    matrix is small enough to remain readable.

    Parameters
    ----------
    S : np.ndarray  shape (n_outputs, n_params)
    row_names : observable feature labels (y-axis)
    col_names : parameter labels (x-axis)
    symmetric : if True, colorbar is symmetric around zero.
    show_colorbar : if False, the colorbar is omitted from the figure.
    """
    n_rows, n_cols = S.shape
    fig_w = max(8, n_cols * 0.65 + 2)
    fig_h = max(4, n_rows * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    S_plot = np.nan_to_num(S, nan=0.0)

    if vmax is None:
        abs_max = np.max(np.abs(S_plot))
        vmax = abs_max if abs_max > 1e-15 else 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if symmetric else None
    cmap = "RdBu_r"

    im = ax.imshow(
        S_plot, cmap=cmap, norm=norm, aspect="auto",
        interpolation="nearest",
    )

    # Annotate cells if matrix is small enough
    annotate = n_rows * n_cols <= 200
    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = S_plot[i, j]
                text_color = "black" if abs(val) < 0.6 * vmax else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_names, fontsize=9)
    ax.set_xlabel("Parameter", fontsize=10)
    ax.set_ylabel("Observable Feature", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Normalized Sensitivity S_ij", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_singular_values(
    svd_results: Dict[str, Dict[str, Any]],
    save_path: Path,
) -> None:
    """Bar chart of singular values for each scenario (observable matrix)."""
    scenarios = list(svd_results.keys())
    n = len(scenarios)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        res = svd_results[sc]
        if "error" in res or "singular_values" not in res:
            ax.set_title(f"{sc}\n(SVD error)")
            continue
        sv = res["singular_values"]
        rank = res["numerical_rank"]
        ax.bar(range(len(sv)), sv, color=["#2196F3" if k < rank else "#90CAF9" for k in range(len(sv))])
        ax.set_title(f"{sc}\nrank={rank}, κ={res['condition_number']:.1f}", fontsize=9)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Singular Values — Observable Sensitivity Matrix (S_obs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_collinearity(
    cosine_matrix: np.ndarray,
    param_names: List[str],
    threshold: float,
    save_path: Path,
) -> None:
    """Heatmap of absolute cosine similarity between parameter sensitivity vectors."""
    fig_s = max(6, len(param_names) * 0.45 + 1)
    fig, ax = plt.subplots(figsize=(fig_s, fig_s))

    im = ax.imshow(cosine_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=7)
    ax.set_title(
        f"Parameter Sensitivity-Vector Collinearity\n"
        f"|cos(θ)| — threshold={threshold:.2f}",
        fontsize=10,
    )

    # Annotate cells
    if len(param_names) <= 20:
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                v = cosine_matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if v < 0.7 else "white")

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("|cos(θ)|", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_parameter_ranking(ranking_df: pd.DataFrame, save_path: Path) -> None:
    """Horizontal bar chart showing mean absolute sensitivity across all scenarios."""
    if ranking_df.empty:
        return
    df = ranking_df.sort_values("rms_abs_sensitivity", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    ax.barh(df["parameter"], df["rms_abs_sensitivity"], color="#1976D2", alpha=0.85)
    ax.set_xlabel("RMS |Normalized Sensitivity| (aggregated)", fontsize=9)
    ax.set_title("Parameter Ranking by Aggregate Sensitivity Magnitude\n(observable outputs only)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Auto-generated interpretation language
# ---------------------------------------------------------------------------

def _generate_interpretation(
    svd_main: Dict[str, Any],
    collinear_pairs_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    collinearity_threshold: float,
) -> str:
    """
    Generate scientifically accurate interpretation text from numerical results.

    The language is conservative: it reports what the analysis demonstrates
    and explicitly states what it does NOT establish.
    """
    lines: List[str] = []

    # Rank / conditioning
    if "error" in svd_main:
        lines.append(
            "SVD analysis encountered an error; rank and conditioning "
            "could not be determined."
        )
    else:
        rank = svd_main.get("numerical_rank", 0)
        n_params = svd_main.get("n_params", len(PARAM_NAMES))
        kappa = svd_main.get("condition_number", float("inf"))
        rank_frac = svd_main.get("rank_fraction", 0.0)

        if rank == n_params:
            distinguishability = "strong"
        elif rank >= int(0.75 * n_params):
            distinguishability = "moderate"
        else:
            distinguishability = "limited"

        lines.append(
            f"The normalized observable sensitivity matrix (S_obs) exhibited "
            f"numerical rank {rank} out of {n_params} parameters "
            f"(rank fraction {rank_frac:.2f}) under the relative threshold "
            f"{svd_main.get('rank_tol_relative', DEFAULT_RANK_TOLERANCE)}. "
            f"The condition number was κ = {kappa:.1f}, indicating "
            f"{distinguishability} local distinguishability for the evaluated "
            f"parameter subset under the tested operating conditions."
        )

    # Collinearity
    if collinear_pairs_df is not None and not collinear_pairs_df.empty:
        flagged = collinear_pairs_df[collinear_pairs_df["flagged"]]
        if not flagged.empty:
            pairs_str = "; ".join(
                f"({r['parameter_a']}, {r['parameter_b']}, |cos|={r['abs_cosine_similarity']:.3f})"
                for _, r in flagged.head(5).iterrows()
            )
            lines.append(
                f"Several parameter pairs exhibited high absolute cosine similarity "
                f"(|cos(θ)| ≥ {collinearity_threshold:.2f}): {pairs_str}. "
                f"These parameters may be locally confounded under the available "
                f"observable set, limiting independent parameter recovery for those pairs."
            )
        else:
            lines.append(
                f"No parameter pairs exceeded the absolute cosine-similarity threshold "
                f"of {collinearity_threshold:.2f}, indicating that the parameter "
                f"sensitivity directions are mutually distinguishable in the observable space."
            )

    # Convergence
    if convergence_df is not None and not convergence_df.empty:
        unstable = convergence_df[convergence_df["max_relative_deviation"] > 0.5]
        if not unstable.empty:
            unstable_params = ", ".join(unstable["parameter"].head(5).tolist())
            lines.append(
                f"Perturbation convergence analysis identified {len(unstable)} parameter(s) "
                f"with relative deviation > 0.50 across perturbation fractions "
                f"({unstable_params}). Sensitivity estimates for these parameters "
                f"should be interpreted cautiously."
            )
        else:
            lines.append(
                "Perturbation convergence analysis showed stable sensitivity estimates "
                "across all evaluated perturbation fractions, supporting the local character "
                "of the finite-difference approximation."
            )

    lines.append(
        "IMPORTANT: This analysis characterises local sensitivity at the benchmark's "
        "nominal operating point. It provides evidence of local practical identifiability "
        "for the evaluated subset under the tested conditions. It does NOT establish "
        "structural or global identifiability, and it does not guarantee parameter "
        "recoverability from real experimental data subject to measurement noise."
    )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main PAR-03 execution function
# ---------------------------------------------------------------------------

def run_sensitivity_matrix(
    output_dir: Path,
    params: TankDynamicsParams,
) -> Dict[str, Any]:
    """
    PAR-03: Normalized Parameter Sensitivity Matrix Analysis.

    Executes the full analysis across 4 scenarios and 3 perturbation fractions,
    generates all CSV outputs and figures, and returns a status/metrics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    warnings: List[str] = []
    assertions_failed: List[str] = []

    perturbation_fractions = DEFAULT_PERTURBATION_FRACTIONS
    collinearity_threshold = DEFAULT_COLLINEARITY_THRESHOLD
    rank_tol = DEFAULT_RANK_TOLERANCE
    seed = DEFAULT_RANDOM_SEED
    min_output_scale = DEFAULT_MIN_OUTPUT_SCALE

    n_params = len(PARAM_NAMES)
    n_obs = len(OBSERVABLE_FEATURES)
    n_all = len(ALL_FEATURES)

    # -----------------------------------------------------------------------
    # Step 1: Verify all requested parameters exist
    # -----------------------------------------------------------------------
    for pname in PARAM_NAMES:
        if not hasattr(params, pname):
            assertions_failed.append(f"Parameter '{pname}' does not exist on TankDynamicsParams.")

    if assertions_failed:
        return {
            "status": "FAIL",
            "metrics": {},
            "warnings": assertions_failed,
        }

    # -----------------------------------------------------------------------
    # Step 2: Collect sensitivity matrices for all scenarios × perturbations
    # -----------------------------------------------------------------------
    # S_obs_by_scenario_frac[scenario][frac] = np.ndarray (n_obs, n_params)
    # S_full_by_scenario_frac[scenario][frac] = np.ndarray (n_all, n_params)
    S_obs_by_scenario_frac: Dict[str, Dict[float, np.ndarray]] = {sc: {} for sc in SCENARIOS}
    S_full_by_scenario_frac: Dict[str, Dict[float, np.ndarray]] = {sc: {} for sc in SCENARIOS}
    all_row_meta: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for scenario_name in SCENARIOS:
        for frac in perturbation_fractions:
            S_obs, S_full, row_meta = _build_matrix(
                params, scenario_name, frac, seed, min_output_scale,
            )
            S_obs_by_scenario_frac[scenario_name][frac] = S_obs
            S_full_by_scenario_frac[scenario_name][frac] = S_full
            all_row_meta.extend(row_meta)

            # Accumulate long-form rows
            for j, pname in enumerate(PARAM_NAMES):
                for i, feat in enumerate(ALL_FEATURES):
                    is_obs = feat in OBSERVABLE_FEATURES
                    long_rows.append({
                        "scenario": scenario_name,
                        "perturbation_fraction": frac,
                        "output": feat,
                        "observable": is_obs,
                        "parameter": pname,
                        "category": PARAM_CATEGORIES.get(pname, "unknown"),
                        "sensitivity": S_full[i, j],
                    })

            # Check for NaN/Inf
            if np.any(np.isnan(S_obs)) or np.any(np.isinf(S_obs)):
                n_nan = int(np.sum(np.isnan(S_obs)))
                warnings.append(
                    f"[{scenario_name}, frac={frac}] S_obs has {n_nan} NaN entries. "
                    f"Some parameter perturbations may be invalid."
                )

    # -----------------------------------------------------------------------
    # Step 3: Primary perturbation level for reporting (frac=0.01)
    # -----------------------------------------------------------------------
    primary_frac = 0.01
    if primary_frac not in perturbation_fractions:
        primary_frac = perturbation_fractions[len(perturbation_fractions) // 2]

    # -----------------------------------------------------------------------
    # Step 4: SVD analysis — primary on observable matrix, supplementary on full
    # -----------------------------------------------------------------------
    svd_obs_by_scenario: Dict[str, Dict[str, Any]] = {}
    svd_full_by_scenario: Dict[str, Dict[str, Any]] = {}

    for scenario_name in SCENARIOS:
        S_obs_p = S_obs_by_scenario_frac[scenario_name][primary_frac]
        S_full_p = S_full_by_scenario_frac[scenario_name][primary_frac]
        svd_obs_by_scenario[scenario_name] = _svd_diagnostics(
            S_obs_p, rank_tol, label=f"S_obs [{scenario_name}]"
        )
        svd_full_by_scenario[scenario_name] = _svd_diagnostics(
            S_full_p, rank_tol, label=f"S_full [{scenario_name}]"
        )

    # -----------------------------------------------------------------------
    # Step 5: RMS aggregation across scenarios
    # -----------------------------------------------------------------------
    # Primary observable matrix — RMS across scenarios at primary perturbation
    S_obs_rms = _rms_aggregate({sc: S_obs_by_scenario_frac[sc][primary_frac] for sc in SCENARIOS})
    S_full_rms = _rms_aggregate({sc: S_full_by_scenario_frac[sc][primary_frac] for sc in SCENARIOS})

    svd_obs_agg = _svd_diagnostics(S_obs_rms, rank_tol, label="S_obs [RMS-aggregated]")
    svd_full_agg = _svd_diagnostics(S_full_rms, rank_tol, label="S_full [RMS-aggregated]")

    # -----------------------------------------------------------------------
    # Step 6: Collinearity — on RMS observable matrix
    # -----------------------------------------------------------------------
    collinear_df, cosine_matrix = _collinearity_analysis(
        S_obs_rms, PARAM_NAMES, collinearity_threshold
    )
    flagged_pairs = (
        collinear_df[collinear_df["flagged"]].to_dict("records")
        if collinear_df is not None and not collinear_df.empty
        else []
    )

    # -----------------------------------------------------------------------
    # Step 7: Perturbation convergence — on observable matrix, normal scenario
    # -----------------------------------------------------------------------
    convergence_df = _perturbation_convergence(
        {frac: S_obs_by_scenario_frac["normal"][frac] for frac in perturbation_fractions},
        PARAM_NAMES,
        OBSERVABLE_FEATURES,
    )

    # -----------------------------------------------------------------------
    # Step 8: Parameter ranking
    # -----------------------------------------------------------------------
    ranking_rows = []
    for j, pname in enumerate(PARAM_NAMES):
        col_vals = S_obs_rms[:, j]
        valid = col_vals[~np.isnan(col_vals)]
        ranking_rows.append({
            "parameter": pname,
            "category": PARAM_CATEGORIES.get(pname, "unknown"),
            "rms_abs_sensitivity": float(np.sqrt(np.mean(valid ** 2))) if len(valid) > 0 else np.nan,
            "mean_abs_sensitivity": float(np.mean(np.abs(valid))) if len(valid) > 0 else np.nan,
            "max_abs_sensitivity": float(np.max(np.abs(valid))) if len(valid) > 0 else np.nan,
            "n_obs_features": int(np.sum(~np.isnan(col_vals))),
        })
    ranking_df = pd.DataFrame(ranking_rows).sort_values("rms_abs_sensitivity", ascending=False)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    # -----------------------------------------------------------------------
    # Step 9: Robustness table (per-frac SVD summary)
    # -----------------------------------------------------------------------
    robustness_rows = []
    for sc in SCENARIOS:
        for frac in perturbation_fractions:
            svd_res = _svd_diagnostics(
                S_obs_by_scenario_frac[sc][frac], rank_tol,
                label=f"S_obs [{sc}, frac={frac}]",
            )
            robustness_rows.append({
                "scenario": sc,
                "perturbation_fraction": frac,
                "numerical_rank": svd_res.get("numerical_rank", np.nan),
                "rank_fraction": svd_res.get("rank_fraction", np.nan),
                "condition_number": svd_res.get("condition_number", np.nan),
                "sigma_max": svd_res.get("sigma_max", np.nan),
                "sigma_min_nonzero": svd_res.get("sigma_min_nonzero", np.nan),
            })
    robustness_df = pd.DataFrame(robustness_rows)

    # -----------------------------------------------------------------------
    # Step 10: Write all CSVs
    # -----------------------------------------------------------------------
    # 10a. Long-form
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_dir / "sensitivity_matrix_long.csv", index=False)

    # 10b. Matrix-form (normal scenario, primary fraction, observable only)
    S_obs_normal = S_obs_by_scenario_frac["normal"][primary_frac]
    obs_matrix_df = pd.DataFrame(
        S_obs_normal, index=OBSERVABLE_FEATURES, columns=PARAM_NAMES
    )
    obs_matrix_df.index.name = "output"
    obs_matrix_df.to_csv(output_dir / "sensitivity_matrix_normal.csv")

    # 10c. Aggregated RMS matrix (observable)
    rms_matrix_df = pd.DataFrame(
        S_obs_rms, index=OBSERVABLE_FEATURES, columns=PARAM_NAMES
    )
    rms_matrix_df.index.name = "output"
    rms_matrix_df.to_csv(output_dir / "sensitivity_matrix.csv")

    # 10d. Singular values (all scenarios, primary fraction)
    sv_rows = []
    for sc, svd_res in svd_obs_by_scenario.items():
        if "singular_values" in svd_res:
            for k, sv in enumerate(svd_res["singular_values"]):
                sv_rows.append({
                    "scenario": sc, "matrix": "S_obs",
                    "index": k, "singular_value": sv,
                    "numerical_rank": svd_res.get("numerical_rank", np.nan),
                })
    for sc, svd_res in svd_full_by_scenario.items():
        if "singular_values" in svd_res:
            for k, sv in enumerate(svd_res["singular_values"]):
                sv_rows.append({
                    "scenario": sc, "matrix": "S_full (supplementary)",
                    "index": k, "singular_value": sv,
                    "numerical_rank": svd_res.get("numerical_rank", np.nan),
                })
    pd.DataFrame(sv_rows).to_csv(output_dir / "sensitivity_singular_values.csv", index=False)

    # 10e. Collinearity
    if collinear_df is not None and not collinear_df.empty:
        collinear_df.to_csv(output_dir / "sensitivity_collinearity.csv", index=False)

    # 10f. Parameter ranking
    ranking_df.to_csv(output_dir / "parameter_ranking.csv", index=False)

    # 10g. Perturbation robustness
    robustness_df.to_csv(output_dir / "perturbation_robustness.csv", index=False)

    # 10h. Convergence
    if not convergence_df.empty:
        convergence_df.to_csv(output_dir / "perturbation_convergence.csv", index=False)

    # 10i. Perturbation metadata
    meta_df = pd.DataFrame(all_row_meta)
    meta_df.to_csv(output_dir / "perturbation_metadata.csv", index=False)

    # -----------------------------------------------------------------------
    # Step 11: Figures
    # -----------------------------------------------------------------------
    # Global vmax for consistent coloring across scenario heatmaps
    all_obs_vals = np.concatenate([
        S_obs_by_scenario_frac[sc][primary_frac].ravel() for sc in SCENARIOS
    ])
    vmax_global = float(np.nanmax(np.abs(all_obs_vals))) if len(all_obs_vals) > 0 else 1.0
    vmax_global = max(vmax_global, 0.01)

    # Per-scenario observable heatmaps
    for sc in SCENARIOS:
        _plot_heatmap(
            S_obs_by_scenario_frac[sc][primary_frac],
            OBSERVABLE_FEATURES,
            PARAM_NAMES,
            f"PAR-03: Observable Sensitivity Matrix — {sc}\n"
            f"(S_obs, Δ={primary_frac*100:.1f}%, normalized S_ij=(θ_j/y_i)·Δy_i/Δθ_j)",
            output_dir / f"sensitivity_heatmap_{sc}.png",
            vmax=vmax_global,
        )

    # Aggregated RMS heatmap — publication-quality labels, no colorbar
    _agg_row_labels = [
        OBSERVABLE_LABELS.get(f, f) for f in OBSERVABLE_FEATURES
    ]
    _agg_col_labels = [
        PARAM_LABELS.get(p, p) for p in PARAM_NAMES
    ]
    _plot_heatmap(
        S_obs_rms,
        _agg_row_labels,
        _agg_col_labels,
        "Aggregated Normalized Sensitivity Across Four Operating Regimes",
        output_dir / "sensitivity_heatmap_aggregated.png",
        vmax=vmax_global,
        show_colorbar=False,
    )

    # Singular value bar charts
    _plot_singular_values(svd_obs_by_scenario, output_dir / "sensitivity_singular_values.png")

    # Collinearity heatmap
    if cosine_matrix is not None:
        _plot_collinearity(
            cosine_matrix, PARAM_NAMES, collinearity_threshold,
            output_dir / "sensitivity_collinearity.png",
        )

    # Parameter ranking
    _plot_parameter_ranking(ranking_df, output_dir / "parameter_ranking.png")

    # -----------------------------------------------------------------------
    # Step 12: Generate interpretation text
    # -----------------------------------------------------------------------
    interpretation = _generate_interpretation(
        svd_obs_agg, collinear_df, convergence_df, collinearity_threshold,
    )

    # -----------------------------------------------------------------------
    # Step 13: metadata.json and summary.json
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start

    metadata = {
        "validation": "PAR-03",
        "name": "Normalized Parameter Sensitivity Matrix Analysis",
        "category": "VII. Parameter Identifiability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "random_seed": seed,
        "dt_seconds": DT,
        "scenario_length_steps": SCENARIO_LENGTH,
        "scenarios": list(SCENARIOS.keys()),
        "perturbation_fractions": perturbation_fractions,
        "primary_perturbation_fraction": primary_frac,
        "parameters": PARAM_NAMES,
        "parameter_categories": PARAM_CATEGORIES,
        "observable_features": OBSERVABLE_FEATURES,
        "latent_features": LATENT_FEATURES,
        "min_output_scale": min_output_scale,
        "collinearity_threshold": collinearity_threshold,
        "rank_tolerance_relative": rank_tol,
        "normalization": (
            "S_ij = (theta_j / y_scale_i) * (y_i_hi - y_i_lo) / (p_j_hi - p_j_lo), "
            "y_scale_i = max(|y_i_nominal|, min_output_scale)"
        ),
        "deterministic": True,
        "warnings": warnings,
    }

    summary = {
        "validation": "PAR-03",
        "analysis": "normalized_local_parameter_sensitivity",
        "parameter_count": n_params,
        "observable_feature_count": n_obs,
        "latent_feature_count": len(LATENT_FEATURES),
        "scenario_count": len(SCENARIOS),
        "perturbation_fractions": perturbation_fractions,
        "primary_perturbation_fraction": primary_frac,
        # Aggregated SVD (observable matrix)
        "numerical_rank_obs_aggregated": svd_obs_agg.get("numerical_rank", None),
        "rank_fraction_obs_aggregated": svd_obs_agg.get("rank_fraction", None),
        "condition_number_obs_aggregated": svd_obs_agg.get("condition_number", None),
        "sigma_max_obs_aggregated": svd_obs_agg.get("sigma_max", None),
        "sigma_min_nonzero_obs_aggregated": svd_obs_agg.get("sigma_min_nonzero", None),
        "singular_values_obs_aggregated": svd_obs_agg.get("singular_values", []),
        # Per-scenario SVD (observable matrix)
        "svd_obs_per_scenario": {
            sc: {
                "numerical_rank": svd_obs_by_scenario[sc].get("numerical_rank", None),
                "rank_fraction": svd_obs_by_scenario[sc].get("rank_fraction", None),
                "condition_number": svd_obs_by_scenario[sc].get("condition_number", None),
            }
            for sc in SCENARIOS
        },
        # Collinearity
        "high_collinearity_pairs": [
            {"param_a": r["parameter_a"], "param_b": r["parameter_b"],
             "abs_cosine": round(r["abs_cosine_similarity"], 4)}
            for r in flagged_pairs
        ],
        "collinearity_threshold": collinearity_threshold,
        # Robustness
        "perturbation_convergence_summary": (
            convergence_df[["parameter", "max_relative_deviation", "sign_consistent"]]
            .to_dict("records")
            if not convergence_df.empty
            else []
        ),
        # Top/bottom parameters
        "strongest_parameters": (
            ranking_df.head(3)["parameter"].tolist()
            if not ranking_df.empty else []
        ),
        "weakest_parameters": (
            ranking_df.tail(3)["parameter"].tolist()
            if not ranking_df.empty else []
        ),
        "deterministic": True,
        "interpretation": interpretation,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 14: Validation assertions
    # -----------------------------------------------------------------------
    if np.any(np.isnan(S_obs_rms)) or np.any(np.isinf(S_obs_rms)):
        assertions_failed.append("Aggregated S_obs contains NaN or Inf values.")

    all_files = [
        "sensitivity_matrix.csv", "sensitivity_matrix_normal.csv",
        "sensitivity_matrix_long.csv", "sensitivity_singular_values.csv",
        "parameter_ranking.csv", "perturbation_robustness.csv",
        "sensitivity_heatmap_aggregated.png", "metadata.json", "summary.json",
    ]
    for fname in all_files:
        if not (output_dir / fname).exists():
            assertions_failed.append(f"Expected output file not found: {fname}")

    # Verify SVD succeeded for at least one scenario
    any_svd_ok = any("numerical_rank" in r for r in svd_obs_by_scenario.values())
    if not any_svd_ok:
        assertions_failed.append("SVD failed for all scenarios.")

    # -----------------------------------------------------------------------
    # Step 15: Build return dict
    # -----------------------------------------------------------------------
    agg_rank = svd_obs_agg.get("numerical_rank", None)
    agg_kappa = svd_obs_agg.get("condition_number", None)
    agg_rank_frac = svd_obs_agg.get("rank_fraction", None)
    top_param = ranking_df.iloc[0]["parameter"] if not ranking_df.empty else "N/A"

    status = "FAIL" if assertions_failed else "PASS"

    metrics: Dict[str, Any] = {
        "Observable Matrix Rank (aggregated)": agg_rank,
        "Rank Fraction": f"{agg_rank_frac:.2f}" if agg_rank_frac is not None else "N/A",
        "Condition Number (S_obs agg)": f"{agg_kappa:.1f}" if agg_kappa and agg_kappa < 1e9 else "ill-conditioned",
        "Collinear Parameter Pairs": len(flagged_pairs),
        "Strongest Parameter (RMS)": top_param,
        "Num Parameters Analyzed": n_params,
        "Num Scenarios": len(SCENARIOS),
    }

    return {
        "status": status,
        "metrics": metrics,
        "warnings": warnings + assertions_failed,
        "summary_json_path": str(output_dir / "summary.json"),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTERED_EXPERIMENTS = [
    Experiment(
        id="PAR-03",
        name="Normalized Parameter Sensitivity Matrix",
        category="VII. Parameter Identifiability",
        hypothesis=(
            "The normalized observable sensitivity matrix (S_obs) is non-trivially ranked "
            "and the parameter sensitivity directions are mutually distinguishable under "
            "the tested operating conditions, providing local practical-identifiability "
            "evidence for the evaluated parameter subset."
        ),
        execute=run_sensitivity_matrix,
        metrics=[
            "Observable Matrix Rank (aggregated)",
            "Rank Fraction",
            "Condition Number (S_obs agg)",
            "Collinear Parameter Pairs",
        ],
        plots=[
            "sensitivity_heatmap_normal.png",
            "sensitivity_heatmap_aggregated.png",
            "sensitivity_singular_values.png",
            "sensitivity_collinearity.png",
            "parameter_ranking.png",
        ],
    ),
]
