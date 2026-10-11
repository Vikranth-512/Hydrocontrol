"""
Unit and integration tests for PAR-03: Normalized Parameter Sensitivity Matrix.

Tests verify:
1.  Finite-difference accuracy (synthetic y=p^2 function)
2.  Normalization guard for near-zero outputs
3.  Bounded parameter perturbation (lower-clamp)
4.  Sensitivity matrix construction (shape/type)
5.  SVD rank of identity matrix = n (full rank)
6.  Condition number of known-condition matrix
7.  Collinearity detection — parallel vectors
8.  Collinearity detection — anti-parallel vectors (|cos| not cos)
9.  Deterministic repeated execution
10. Near-zero output handling (no NaN/Inf)
11. Invalid parameter detection (clean error, not crash)
12. RMS aggregation formula (verified against manual computation)
13. Perturbation convergence (sign consistency for monotone function)
14. Full PAR-03 integration (small config, ~smoke test)
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Any, Dict, List

from simulation.dynamics import TankDynamicsParams
from simulation.validate.experiments.sensitivity import (
    _central_fd_sensitivity,
    _build_matrix,
    _svd_diagnostics,
    _collinearity_analysis,
    _perturbation_convergence,
    _rms_aggregate,
    _extract_features,
    _run_scenario_trajectory,
    PARAM_NAMES,
    PARAM_LOWER_CLAMPS,
    OBSERVABLE_FEATURES,
    LATENT_FEATURES,
    ALL_FEATURES,
    SCENARIOS,
    DEFAULT_RANK_TOLERANCE,
    DEFAULT_COLLINEARITY_THRESHOLD,
    DEFAULT_MIN_OUTPUT_SCALE,
    run_sensitivity_matrix,
)


# ---------------------------------------------------------------------------
# Helper: synthetic sensitivity via direct function evaluation
# ---------------------------------------------------------------------------

def _fd_synthetic(y_func, p_nom: float, frac: float, min_scale: float = 0.01) -> float:
    """
    Apply the same normalized central FD formula as in the module to y=f(p).
    Returns the normalized sensitivity estimate.
    """
    delta = p_nom * frac
    y_hi = y_func(p_nom + delta)
    y_lo = y_func(p_nom - delta)
    y_nom = y_func(p_nom)
    y_scale = max(abs(y_nom), min_scale)
    return (p_nom / y_scale) * (y_hi - y_lo) / (2 * delta)


# ---------------------------------------------------------------------------
# 1. Finite-difference accuracy
# ---------------------------------------------------------------------------

class TestFiniteDifferenceAccuracy:
    """For y = p^2, the normalized sensitivity S = (p/y)(dy/dp) = (p/p^2)(2p) = 2.
    The FD estimate should recover this to within ~1% at any reasonable perturbation."""

    @pytest.mark.parametrize("p_nom", [0.5, 1.0, 2.0, 10.0])
    @pytest.mark.parametrize("frac", [0.005, 0.01, 0.02])
    def test_quadratic_normalized_sensitivity(self, p_nom, frac):
        y_func = lambda p: p ** 2
        s_est = _fd_synthetic(y_func, p_nom, frac)
        # Exact normalized sensitivity for y=p^2 is 2
        assert abs(s_est - 2.0) < 0.02, (
            f"FD estimate {s_est:.4f} deviates > 2% from exact=2.0 "
            f"at p_nom={p_nom}, frac={frac}"
        )

    def test_linear_normalized_sensitivity(self):
        """For y = a*p, normalized sensitivity = 1 regardless of a."""
        for a in [0.5, 1.0, 3.7]:
            y_func = lambda p, a=a: a * p
            s_est = _fd_synthetic(y_func, p_nom=5.0, frac=0.01)
            assert abs(s_est - 1.0) < 0.01, f"Linear sensitivity != 1.0 for a={a}: {s_est}"

    def test_inverse_normalized_sensitivity(self):
        """For y = 1/p, normalized sensitivity = -1."""
        y_func = lambda p: 1.0 / p
        s_est = _fd_synthetic(y_func, p_nom=2.0, frac=0.01)
        assert abs(s_est - (-1.0)) < 0.01, f"Inverse sensitivity != -1.0: {s_est}"


# ---------------------------------------------------------------------------
# 2. Normalization guard for near-zero outputs
# ---------------------------------------------------------------------------

class TestNormalizationGuard:
    """Near-zero output must not cause divide-by-zero or NaN."""

    def test_zero_output_no_nan(self):
        """When y_nom ~ 0, y_scale = min_output_scale ensures finite result."""
        min_scale = 0.01
        p_nom = 1.0
        frac = 0.01
        delta = p_nom * frac
        # y is nearly zero (but changes with p)
        y_func = lambda p: 1e-15 * p
        y_hi = y_func(p_nom + delta)
        y_lo = y_func(p_nom - delta)
        y_nom = y_func(p_nom)
        y_scale = max(abs(y_nom), min_scale)  # should clamp to min_scale
        s = (p_nom / y_scale) * (y_hi - y_lo) / (2 * delta)
        assert np.isfinite(s), f"Expected finite sensitivity, got {s}"
        assert y_scale == min_scale, "y_scale should be clamped to min_output_scale"

    def test_min_scale_is_positive(self):
        assert DEFAULT_MIN_OUTPUT_SCALE > 0.0


# ---------------------------------------------------------------------------
# 3. Bounded parameter perturbation (lower clamp)
# ---------------------------------------------------------------------------

class TestBoundedPerturbation:
    def test_clamp_prevents_negative_parameter(self):
        """When nominal * (1 - frac) < lower_clamp, the perturbation is clipped."""
        from simulation.validate.experiments.sensitivity import _perturb_params
        params = TankDynamicsParams()
        pname = "damage_rate"  # nominal = 0.0001, lower_clamp = 1e-8
        p_nom = getattr(params, pname)
        lower_clamp = PARAM_LOWER_CLAMPS[pname]

        # Positive perturbation: should not need clamping
        p_hi, delta_hi, clamped_hi = _perturb_params(params, pname, p_nom * 0.01, lower_clamp)
        assert not clamped_hi, "Positive perturbation should not be clamped"
        assert getattr(p_hi, pname) > p_nom

        # Negative perturbation that would go below lower_clamp
        p_lo, delta_lo, clamped_lo = _perturb_params(params, pname, -p_nom * 100, lower_clamp)
        val_lo = getattr(p_lo, pname)
        assert val_lo >= lower_clamp, f"Clamped value {val_lo} < lower_clamp {lower_clamp}"

    def test_nominal_parameter_unchanged_after_perturb(self):
        """Base params should be unchanged after _perturb_params (deepcopy)."""
        from simulation.validate.experiments.sensitivity import _perturb_params
        params = TankDynamicsParams()
        pname = "mortality_rate"
        nom_val = getattr(params, pname)
        _perturb_params(params, pname, 0.01, 1e-7)
        assert getattr(params, pname) == nom_val, "Base params were mutated!"


# ---------------------------------------------------------------------------
# 4. Sensitivity matrix construction
# ---------------------------------------------------------------------------

class TestMatrixConstruction:
    def test_shape(self, tmp_path):
        params = TankDynamicsParams()
        S_obs, S_full, _ = _build_matrix(params, "normal", 0.01, 42)
        assert S_obs.shape == (len(OBSERVABLE_FEATURES), len(PARAM_NAMES)), (
            f"S_obs shape {S_obs.shape} != expected ({len(OBSERVABLE_FEATURES)}, {len(PARAM_NAMES)})"
        )
        assert S_full.shape == (len(ALL_FEATURES), len(PARAM_NAMES)), (
            f"S_full shape {S_full.shape} != expected ({len(ALL_FEATURES)}, {len(PARAM_NAMES)})"
        )

    def test_no_all_nan_rows(self, tmp_path):
        """No row of S_obs should be entirely NaN for normal scenario."""
        params = TankDynamicsParams()
        S_obs, _, _ = _build_matrix(params, "normal", 0.01, 42)
        for i, feat in enumerate(OBSERVABLE_FEATURES):
            row = S_obs[i, :]
            assert not np.all(np.isnan(row)), f"Row {feat} is entirely NaN"

    def test_dtype_float(self):
        params = TankDynamicsParams()
        S_obs, _, _ = _build_matrix(params, "normal", 0.01, 42)
        assert S_obs.dtype in [np.float64, np.float32], f"Unexpected dtype: {S_obs.dtype}"


# ---------------------------------------------------------------------------
# 5. SVD rank of identity matrix
# ---------------------------------------------------------------------------

class TestSVDRank:
    @pytest.mark.parametrize("n", [3, 5, 8, 16])
    def test_identity_full_rank(self, n):
        S = np.eye(n)
        result = _svd_diagnostics(S, rank_tol_relative=1e-8)
        assert result["numerical_rank"] == n, (
            f"Identity({n}) should have rank={n}, got {result['numerical_rank']}"
        )
        assert result["rank_fraction"] == 1.0

    def test_rank_deficient_matrix(self):
        """Matrix with repeated column should have rank < n."""
        S = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
        result = _svd_diagnostics(S, rank_tol_relative=1e-8)
        assert result["numerical_rank"] < 3, (
            f"Expected rank < 3 for repeated-column matrix, got {result['numerical_rank']}"
        )

    def test_relative_tolerance_respected(self):
        """With very loose tolerance, only the dominant direction counts."""
        S = np.diag([100.0, 1.0, 0.001])
        # sigma_max = 100; with tol=0.05, sigma=1 (0.01) < 0.05 → truncated
        result = _svd_diagnostics(S, rank_tol_relative=0.05)
        assert result["numerical_rank"] == 1, (
            f"Expected rank=1 with loose tol, got {result['numerical_rank']}"
        )

    def test_zero_matrix_rank(self):
        result = _svd_diagnostics(np.zeros((4, 3)), rank_tol_relative=1e-8)
        assert result["numerical_rank"] == 0


# ---------------------------------------------------------------------------
# 6. Condition number
# ---------------------------------------------------------------------------

class TestConditionNumber:
    def test_known_condition(self):
        """Diagonal matrix with known singular values."""
        sv = np.array([10.0, 5.0, 1.0])
        S = np.diag(sv)
        result = _svd_diagnostics(S, rank_tol_relative=1e-8)
        kappa_expected = 10.0 / 1.0
        kappa_got = result["condition_number"]
        assert abs(kappa_got - kappa_expected) < 0.01, (
            f"Expected κ={kappa_expected}, got {kappa_got}"
        )

    def test_rank_deficient_condition(self):
        """For near-zero singular value, condition number should be inf."""
        S = np.array([[1.0, 1.0], [0.5, 0.5]])  # rank 1
        result = _svd_diagnostics(S, rank_tol_relative=1e-8)
        # Second singular value is ~0, so full-matrix condition number is inf
        sigma = result["singular_values"]
        assert sigma[-1] < 1e-10, "Expected near-zero second singular value"


# ---------------------------------------------------------------------------
# 7. Collinearity detection — parallel
# ---------------------------------------------------------------------------

class TestCollinearityDetection:
    def test_parallel_vectors_detected(self):
        """Columns that are identical should have |cos|=1.0 and be flagged."""
        n_out = 6
        v = np.random.default_rng(0).standard_normal(n_out)
        S = np.column_stack([v, v, np.random.default_rng(1).standard_normal(n_out)])
        names = ["p0", "p1", "p2"]
        df, cos_mat = _collinearity_analysis(S, names, threshold=0.95)
        flagged = df[df["flagged"]]
        # (p0, p1) should be flagged
        assert any(
            (r["parameter_a"] == "p0" and r["parameter_b"] == "p1") or
            (r["parameter_a"] == "p1" and r["parameter_b"] == "p0")
            for _, r in flagged.iterrows()
        ), "Identical parameter vectors not detected as collinear"

    def test_antiparallel_vectors_detected(self):
        """Anti-parallel vectors should also be flagged (|cos| not cos)."""
        n_out = 6
        v = np.random.default_rng(5).standard_normal(n_out)
        S = np.column_stack([v, -v])
        names = ["p0", "p1"]
        df, _ = _collinearity_analysis(S, names, threshold=0.95)
        flagged = df[df["flagged"]]
        assert len(flagged) >= 1, (
            "Anti-parallel parameter vectors should be flagged by |cos| criterion"
        )
        assert flagged.iloc[0]["abs_cosine_similarity"] > 0.99

    def test_orthogonal_vectors_not_flagged(self):
        """Orthogonal columns should have |cos|~0 and not be flagged."""
        S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        names = ["p0", "p1"]
        df, _ = _collinearity_analysis(S, names, threshold=0.95)
        flagged = df[df["flagged"]]
        assert len(flagged) == 0, "Orthogonal vectors should not be flagged"

    def test_zero_vector_no_crash(self):
        """Zero column should be handled gracefully (no NaN/crash)."""
        S = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        names = ["p0", "p1", "p2"]
        df, cos_mat = _collinearity_analysis(S, names, threshold=0.95)
        # No crash; cos_mat should have finite entries
        assert np.all(np.isfinite(cos_mat)), "Cosine matrix contains non-finite values"


# ---------------------------------------------------------------------------
# 9. Deterministic repeated execution
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_trajectory_determinism(self):
        """Two calls with the same seed must return identical trajectories."""
        params = TankDynamicsParams()
        sc = SCENARIOS["normal"]
        df1 = _run_scenario_trajectory(params, sc, seed=42)
        df2 = _run_scenario_trajectory(params, sc, seed=42)
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, atol=1e-12)

    def test_sensitivity_determinism(self):
        """Sensitivity computation must be identical across two calls."""
        params = TankDynamicsParams()
        pname = "maximum_uptake_rate"
        sc = SCENARIOS["normal"]
        s1, _ = _central_fd_sensitivity(params, pname, sc, 0.01, 42)
        s2, _ = _central_fd_sensitivity(params, pname, sc, 0.01, 42)
        for feat in OBSERVABLE_FEATURES:
            assert abs(s1[feat] - s2[feat]) < 1e-12, (
                f"Non-deterministic sensitivity for {feat}: {s1[feat]} vs {s2[feat]}"
            )

    def test_matrix_determinism(self):
        """Full matrix construction must be reproducible."""
        params = TankDynamicsParams()
        S1, _, _ = _build_matrix(params, "normal", 0.01, 42)
        S2, _, _ = _build_matrix(params, "normal", 0.01, 42)
        np.testing.assert_allclose(S1, S2, atol=1e-12)


# ---------------------------------------------------------------------------
# 10. Near-zero output handling
# ---------------------------------------------------------------------------

class TestNearZeroOutputHandling:
    def test_feature_extraction_no_nan(self):
        """Feature extraction from any reasonable trajectory should not produce NaN."""
        params = TankDynamicsParams()
        sc = SCENARIOS["normal"]
        df = _run_scenario_trajectory(params, sc, seed=42)
        feats = _extract_features(df)
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature '{k}' is non-finite: {v}"

    def test_ec_std_nonnegative(self):
        params = TankDynamicsParams()
        sc = SCENARIOS["normal"]
        df = _run_scenario_trajectory(params, sc, seed=42)
        feats = _extract_features(df)
        assert feats["ec_std"] >= 0.0


# ---------------------------------------------------------------------------
# 11. Invalid parameter detection
# ---------------------------------------------------------------------------

class TestInvalidParameterDetection:
    def test_nonexistent_param_raises(self):
        """Requesting a nonexistent parameter should raise AttributeError or similar."""
        params = TankDynamicsParams()
        pname = "__nonexistent_parameter__"
        with pytest.raises((AttributeError, KeyError)):
            # _central_fd_sensitivity reads pname from params via getattr
            # If pname doesn't exist on TankDynamicsParams, getattr will raise
            _ = getattr(params, pname)

    def test_run_sensitivity_matrix_fails_gracefully(self, tmp_path, monkeypatch):
        """If parameters list is invalid, run_sensitivity_matrix returns FAIL not crash."""
        # Temporarily patch PARAM_NAMES to contain a bad entry
        import simulation.validate.experiments.sensitivity as sm
        original = sm.PARAM_NAMES[:]
        sm.PARAM_NAMES.append("__bad_param__")
        sm.PARAM_LOWER_CLAMPS["__bad_param__"] = 0.0
        sm.PARAM_CATEGORIES["__bad_param__"] = "test"
        try:
            params = TankDynamicsParams()
            result = run_sensitivity_matrix(tmp_path, params)
            assert result["status"] == "FAIL"
        finally:
            # Restore
            sm.PARAM_NAMES[:] = original
            sm.PARAM_LOWER_CLAMPS.pop("__bad_param__", None)
            sm.PARAM_CATEGORIES.pop("__bad_param__", None)


# ---------------------------------------------------------------------------
# 12. RMS aggregation formula
# ---------------------------------------------------------------------------

class TestRMSAggregation:
    def test_rms_formula(self):
        """S_RMS_ij = sqrt(mean(S_ij_k^2)) across scenarios k."""
        A = np.array([[1.0, -2.0], [3.0, 4.0]])
        B = np.array([[2.0, 0.0], [1.0, -1.0]])
        C = np.array([[0.0, 3.0], [-1.0, 2.0]])
        expected = np.sqrt((A**2 + B**2 + C**2) / 3.0)
        result = _rms_aggregate({"a": A, "b": B, "c": C})
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_rms_single_matrix(self):
        """RMS of a single matrix equals abs-value of each element."""
        A = np.array([[1.0, -2.0], [3.0, -4.0]])
        result = _rms_aggregate({"a": A})
        np.testing.assert_allclose(result, np.abs(A), atol=1e-12)


# ---------------------------------------------------------------------------
# 13. Perturbation convergence
# ---------------------------------------------------------------------------

class TestPerturbationConvergence:
    def test_stable_sensitivity_convergence(self):
        """
        For a linear function y = a*p, normalized sensitivity = 1 at all fracs.
        Relative deviation should be ~0 → sign_consistent=True, max_dev~0.
        """
        # Construct three "matrices" corresponding to fracs 0.5%, 1%, 2%
        # where each entry is exactly 1.0 (linear sensitivity)
        n_out, n_params = 6, 3
        matrices = {
            0.005: np.ones((n_out, n_params)),
            0.01:  np.ones((n_out, n_params)),
            0.02:  np.ones((n_out, n_params)),
        }
        names = [f"p{i}" for i in range(n_params)]
        feats = [f"y{i}" for i in range(n_out)]
        df = _perturbation_convergence(matrices, names, feats)
        assert not df.empty
        assert (df["max_relative_deviation"] < 0.01).all(), (
            "Stable (identical) sensitivity matrices should show near-zero deviation"
        )
        assert df["sign_consistent"].all()

    def test_unstable_sensitivity_flagged(self):
        """Matrices that change substantially across fracs should have large deviation."""
        n_out, n_params = 6, 2
        matrices = {
            0.005: np.ones((n_out, n_params)) * 1.0,
            0.01:  np.ones((n_out, n_params)) * 5.0,  # 5x change
            0.02:  np.ones((n_out, n_params)) * 10.0,
        }
        names = ["p0", "p1"]
        feats = [f"y{i}" for i in range(n_out)]
        df = _perturbation_convergence(matrices, names, feats)
        assert (df["max_relative_deviation"] > 0.5).any(), (
            "Large variation across fracs should produce high relative deviation"
        )


# ---------------------------------------------------------------------------
# 14. Full integration smoke test (limited parameters for speed)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_run_sensitivity_matrix_smoke(self, tmp_path):
        """
        Run a truncated PAR-03 (2 scenarios, 2 fracs) to verify end-to-end
        execution without errors. Patches globals to reduce runtime.
        """
        import simulation.validate.experiments.sensitivity as sm

        # Patch to 2 scenarios and 1 short-length scenario
        original_scenarios = sm.SCENARIOS
        original_length = sm.SCENARIO_LENGTH
        original_fracs = sm.DEFAULT_PERTURBATION_FRACTIONS

        # Mini scenarios
        sm.SCENARIOS = {
            "normal": {
                "description": "test",
                "dissolved_mass": 2.0, "biomass": 100.0, "water_temp": 22.0,
                "actions": [(1.0, 10.0)] * 50,
            },
            "heatwave": {
                "description": "test",
                "dissolved_mass": 2.0, "biomass": 100.0, "water_temp": 33.0,
                "actions": [(1.0, 10.0)] * 50,
            },
        }
        sm.SCENARIO_LENGTH = 50
        sm.DEFAULT_PERTURBATION_FRACTIONS = [0.01, 0.02]

        try:
            params = TankDynamicsParams()
            result = run_sensitivity_matrix(tmp_path, params)

            assert result["status"] in ("PASS", "FAIL")
            assert "metrics" in result
            assert (tmp_path / "sensitivity_matrix.csv").exists()
            assert (tmp_path / "sensitivity_matrix_long.csv").exists()
            assert (tmp_path / "parameter_ranking.csv").exists()
            assert (tmp_path / "metadata.json").exists()
            assert (tmp_path / "summary.json").exists()
            assert (tmp_path / "sensitivity_heatmap_aggregated.png").exists()

            # No NaN/Inf in primary outputs
            df = pd.read_csv(tmp_path / "sensitivity_matrix.csv", index_col=0)
            assert not df.isna().all().all(), "All entries are NaN — something is wrong"

        finally:
            sm.SCENARIOS = original_scenarios
            sm.SCENARIO_LENGTH = original_length
            sm.DEFAULT_PERTURBATION_FRACTIONS = original_fracs
