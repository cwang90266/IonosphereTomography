# -*- coding: utf-8 -*-
"""
Pytest suite for Parameterization.py.

Organized to mirror the module verification plan (Module_Verification_Plan.md,
section 4.3): round trips, finite-difference Jacobian checks per parameterization
style, and Parameterized_EDPSamples wiring (construction, error diagnostics,
NetCDF round trip). Several tests are explicit regressions for bugs found only
by running the code (not by reading it):

- the ANCHOR forward model silently treating log10(NmF2)/log10(NmE) as if they
  were already linear densities (~10 orders of magnitude off);
- the PCA maps never re-applying the mean get_PCA subtracts internally, so even
  full-rank reconstruction silently dropped the ensemble mean;
- get_PCA's retaining_threshold direction and the .shape()/.keys() typos that
  made every PCA/ANCHOR code path raise before it could even run.

Run with:  /opt/anaconda3/bin/python3 -m pytest Parameterization/ -q
(the project's numpy/xarray/scipy/tqdm/pyproj/cartopy stack lives in the
base conda environment, not the system python3 -- see conftest.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

import Parameterization as P
from edp_samples import EDPSamples


# ===========================================================================
# Shared helpers and fixtures
# ===========================================================================

def relative_error(analytic: np.ndarray, numeric: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Elementwise |analytic - numeric| / max(|analytic|, |numeric|, floor)."""
    denom = np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), floor)
    return np.abs(analytic - numeric) / denom


@pytest.fixture(scope="module")
def altitude_full_span() -> np.ndarray:
    """Altitude grid spanning all four ANCHOR regions (E-layer through topside)."""
    return np.linspace(80.0, 700.0, 200)


@pytest.fixture(scope="module")
def single_anchor_params() -> np.ndarray:
    """
    One hand-picked, well-separated ANCHOR parameter vector (order matches
    ANCHOR_PARAM_LABELS): log10_NmF2, hmF2, H0, gamma, B0, B1, log10_NmE, hmE.
    """
    return np.array([11.0, 300.0, 60.0, 0.5, 90.0, 1.5, 10.0, 105.0])


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="module")
def small_anchor_ensemble(rng):
    """
    (8, n_geo, n_sample) ensemble of random-but-physical ANCHOR parameter
    vectors, small enough that per-profile ANCHOR fitting stays fast.
    """
    n_geo, n_sample = 2, 3
    params = np.empty((8, n_geo, n_sample))
    for ig in range(n_geo):
        for isamp in range(n_sample):
            params[:, ig, isamp] = [
                rng.uniform(10.5, 11.5), rng.uniform(250, 350), rng.uniform(40, 80),
                rng.uniform(0.3, 0.8), rng.uniform(60, 120), rng.uniform(1.0, 2.5),
                rng.uniform(9.5, 10.5), rng.uniform(95, 115),
            ]
    return params


@pytest.fixture(scope="module")
def synthetic_edpsamples(small_anchor_ensemble):
    """
    A small, self-consistent EDPSamples built without the Fortran driver:
    EDPs are generated analytically by the ANCHOR forward model itself, so
    downstream ANCHOR fits have a known, in-family ground truth to recover.
    """
    n_alt = 50
    altitude = np.linspace(100.0, 700.0, n_alt)
    geo, mesh = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)
    n_geo = geo.shape[0]
    n_sample = small_anchor_ensemble.shape[2]

    # Broadcast the (8, 2, 3) fixture ensemble out to the actual mesh's n_geo
    # by tiling/truncating -- keeps the fixture's params reusable regardless
    # of exactly how many vertices genRectangularArea produces.
    reps = int(np.ceil(n_geo / small_anchor_ensemble.shape[1]))
    params = np.tile(small_anchor_ensemble, (1, reps, 1))[:, :n_geo, :]

    edps = P._ne_profile_derivatives(altitude, params, partial=False)
    feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), n_geo, n_sample))

    sp = pd.DataFrame({
        "hour": [12] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}

    return EDPSamples(
        DateTime="2026-01-01T12:00:00", geo_type="Rectangle",
        altitude=altitude, sampling_parameters=sp,
        minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5,
        edps=edps, feature_edps=feature_edps,
    )


ALL_STYLES = ["density_10ex", "ANCHOR", "PCA_1D", "PCA_1D_10ex", "PCA_3D", "PCA_3D_10ex"]
PCA_STYLES = ["PCA_1D", "PCA_1D_10ex", "PCA_3D", "PCA_3D_10ex"]


def _hyper_params_for(style: str) -> dict | None:
    if style in PCA_STYLES:
        return {"retaining_threshold": 0.999}
    return None


@pytest.fixture(scope="module")
def parameterized_by_style(synthetic_edpsamples):
    """One Parameterized_EDPSamples per style, built once and reused read-only."""
    return {
        style: P.Parameterized_EDPSamples(
            synthetic_edpsamples, style=style, hyper_params=_hyper_params_for(style)
        )
        for style in ALL_STYLES
    }


# ===========================================================================
# density_10ex
# ===========================================================================

class TestDensity10ex:
    def test_round_trip_above_floor(self):
        density = np.array([1e5, 1e8, 1e10, 1e11, 3.7e11])
        recon = P.density_10ex(P.log10_param(density.copy()))
        assert np.allclose(recon, density, rtol=1e-10)

    def test_clamps_below_floor(self):
        density = np.array([1.0, 100.0, 1e3])   # all below 10**4
        recon = P.density_10ex(P.log10_param(density.copy()))
        assert np.allclose(recon, 1e4)

    def test_does_not_mutate_input(self):
        density = np.array([1.0, 1e5, 1e10])
        original = density.copy()
        _ = P.log10_param(density)
        assert np.array_equal(density, original), "log10_param mutated its input array"

        log10_density = np.array([1.0, 4.0, 10.0])
        original = log10_density.copy()
        _ = P.density_10ex(log10_density)
        assert np.array_equal(log10_density, original), "density_10ex mutated its input array"

    def test_jacobian_matches_finite_difference(self):
        p = np.array([4.5, 8.0, 11.0])   # strictly above the floor
        eps = 1e-6
        fd = (P.density_10ex(p + eps) - P.density_10ex(p - eps)) / (2 * eps)
        analytic = P.Jacobian_density_10ex(p)
        assert np.allclose(fd, analytic, rtol=1e-4)

    def test_jacobian_is_zero_below_floor(self):
        # density_10ex is constant (clamped) below the floor, so its true
        # local derivative there is zero -- not ln(10)*floor.
        p = np.array([1.0, 3.0, 3.9999])
        jac = P.Jacobian_density_10ex(p, minlog10Density=4.0)
        assert np.allclose(jac, 0.0)


# ===========================================================================
# ANCHOR forward model
# ===========================================================================

class TestAnchorForwardModel:
    def test_shape_single_profile(self, altitude_full_span, single_anchor_params):
        Ne = P._ne_profile_derivatives(altitude_full_span, single_anchor_params, partial=False)
        assert Ne.shape == altitude_full_span.shape

    def test_shape_ensemble(self, altitude_full_span, small_anchor_ensemble):
        Ne = P._ne_profile_derivatives(altitude_full_span, small_anchor_ensemble, partial=False)
        assert Ne.shape == (len(altitude_full_span),) + small_anchor_ensemble.shape[1:]

    def test_density_is_nonnegative(self, altitude_full_span, small_anchor_ensemble):
        Ne = P._ne_profile_derivatives(altitude_full_span, small_anchor_ensemble, partial=False)
        assert np.all(Ne >= 0.0)

    def test_magnitude_matches_log10_nmf2(self, altitude_full_span, single_anchor_params):
        """
        Regression: NmF2/NmE are stored as log10(density); the forward model
        must exponentiate them before use. A prior version of this function
        used the raw stored value directly, giving Ne ~ 11 instead of ~1e11
        for log10_NmF2 = 11.0.
        """
        Ne = P._ne_profile_derivatives(altitude_full_span, single_anchor_params, partial=False)
        expected_order = 10.0 ** single_anchor_params[0]   # ~1e11
        assert Ne.max() > 0.1 * expected_order
        assert Ne.max() < 10.0 * expected_order

    def test_continuous_across_region_boundaries(self, single_anchor_params):
        hmE, h_ST, hmF2 = P.anchor_region_boundaries(single_anchor_params)
        fine_alt = np.linspace(80.0, 700.0, 5000)
        Ne = P._ne_profile_derivatives(fine_alt, single_anchor_params, partial=False)
        jumps = np.abs(np.diff(Ne))
        step_scale = np.median(np.abs(Ne)) * 0.05 + 1.0
        # No single altitude step should produce a discontinuous jump far
        # larger than neighbouring steps -- catches a mismatched region
        # formula at hmE/h_ST/hmF2 without requiring an exact tolerance.
        assert jumps.max() < 50 * np.median(jumps) + step_scale

    def test_partial_false_returns_density_only(self, altitude_full_span, single_anchor_params):
        result = P._ne_profile_derivatives(altitude_full_span, single_anchor_params, partial=False)
        assert isinstance(result, np.ndarray)

    def test_partial_true_returns_density_and_jacobian(self, altitude_full_span, single_anchor_params):
        Ne, dNe_dP = P._ne_profile_derivatives(altitude_full_span, single_anchor_params, partial=True)
        assert Ne.shape == altitude_full_span.shape
        assert dNe_dP.shape == (len(altitude_full_span), 8)
        # Regression: a prior version left Ne all-zero on this code path
        # (the real Jacobian was computed but the density fill was skipped).
        assert np.any(Ne > 0)

    def test_invalid_leading_dimension_raises(self, altitude_full_span):
        with pytest.raises(ValueError):
            P._ne_profile_derivatives(altitude_full_span, np.zeros(7), partial=False)


# ===========================================================================
# ANCHOR Jacobian: finite-difference checks, one region-spanning case per parameter
# ===========================================================================

# Per-parameter step size and looseness tuned to each parameter's natural scale.
_ANCHOR_FD_EPS = {
    "log10_NmF2": 1e-4, "hmF2": 1e-3, "H0": 1e-3, "gamma": 1e-4,
    "B0": 1e-3, "B1": 1e-4, "log10_NmE": 1e-4, "hmE": 1e-3,
}


class TestAnchorJacobian:
    @pytest.mark.parametrize("label", P.ANCHOR_PARAM_LABELS)
    def test_matches_finite_difference(self, altitude_full_span, single_anchor_params, label):
        k = P.ANCHOR_PARAM_LABELS.index(label)
        eps = _ANCHOR_FD_EPS[label]

        _, dNe_dP = P._ne_profile_derivatives(altitude_full_span, single_anchor_params, partial=True)
        analytic = dNe_dP[:, k]

        p_plus = single_anchor_params.copy(); p_plus[k] += eps
        p_minus = single_anchor_params.copy(); p_minus[k] -= eps
        Ne_plus = P._ne_profile_derivatives(altitude_full_span, p_plus, partial=False)
        Ne_minus = P._ne_profile_derivatives(altitude_full_span, p_minus, partial=False)
        fd = (Ne_plus - Ne_minus) / (2 * eps)

        err = relative_error(analytic, fd, floor=1.0)
        # A handful of points sit exactly on a region boundary, where the
        # piecewise model's derivative is only continuous, not smooth,
        # inflating the two-sided finite difference; require agreement
        # almost everywhere rather than at the single worst point.
        assert np.percentile(err, 99) < 0.01, (
            f"{label}: 99th-percentile relative error {np.percentile(err, 99):.4g} "
            f"(max {err.max():.4g} at altitude {altitude_full_span[np.argmax(err)]:.1f} km)"
        )


# ===========================================================================
# ANCHOR region boundaries / bisection
# ===========================================================================

class TestAnchorRegionBoundaries:
    def test_hst_matches_independent_root_finder(self, single_anchor_params):
        NmF2 = 10.0 ** single_anchor_params[0]
        hmF2 = single_anchor_params[1]
        B0 = single_anchor_params[4]
        B1 = single_anchor_params[5]
        NmE = 10.0 ** single_anchor_params[6]

        def g(x):
            return NmF2 * np.exp(-(x ** B1)) / np.cosh(x) - NmE

        x_ref = brentq(g, 1e-6, 100.0)
        h_ST_ref = hmF2 - x_ref * B0

        _, h_ST, _ = P.anchor_region_boundaries(single_anchor_params)
        assert h_ST == pytest.approx(h_ST_ref, abs=0.05)

    def test_boundaries_ordered(self, small_anchor_ensemble):
        hmE, h_ST, hmF2 = P.anchor_region_boundaries(small_anchor_ensemble)
        assert np.all(hmE <= h_ST + 1e-9)
        assert np.all(h_ST <= hmF2 + 1e-9)

    def test_scalar_params_return_python_floats(self, single_anchor_params):
        hmE, h_ST, hmF2 = P.anchor_region_boundaries(single_anchor_params)
        assert all(isinstance(v, float) for v in (hmE, h_ST, hmF2))


# ===========================================================================
# ANCHOR inversion (_fit_iri_params / _fit_iri_params_ensemble)
# ===========================================================================

class TestAnchorFit:
    def test_recovers_profile_within_tolerance(self, altitude_full_span, single_anchor_params):
        """
        Fit-then-forward round trip: the fitted parameters need not match the
        truth exactly (the fit is a heuristic nonlinear optimization with
        weakly-identifiable parameters like B0/gamma), but regenerating a
        profile from them should closely reproduce the original in log10-space.
        """
        Ne_true = P._ne_profile_derivatives(altitude_full_span, single_anchor_params, partial=False)
        fitted = P._fit_iri_params(Ne_true, altitude_full_span, P.ANCHOR_DEFAULT_BOUNDS)
        Ne_fit = P._ne_profile_derivatives(altitude_full_span, fitted, partial=False)

        log_rmse = np.sqrt(np.nanmean(
            (np.log10(np.maximum(Ne_true, 1.0)) - np.log10(np.maximum(Ne_fit, 1.0))) ** 2
        ))
        assert log_rmse < 0.1

    def test_ensemble_wrapper_shape(self, altitude_full_span, small_anchor_ensemble):
        Ne = P._ne_profile_derivatives(altitude_full_span, small_anchor_ensemble, partial=False)
        fitted = P._fit_iri_params_ensemble(Ne, altitude_full_span, P.ANCHOR_DEFAULT_BOUNDS,
                                             show_progress=False)
        assert fitted.shape == small_anchor_ensemble.shape

    def test_fitted_params_respect_bounds(self, altitude_full_span, small_anchor_ensemble):
        Ne = P._ne_profile_derivatives(altitude_full_span, small_anchor_ensemble, partial=False)
        fitted = P._fit_iri_params_ensemble(Ne, altitude_full_span, P.ANCHOR_DEFAULT_BOUNDS,
                                             show_progress=False)
        for k, label in enumerate(P.ANCHOR_PARAM_LABELS):
            lo, hi = P.ANCHOR_DEFAULT_BOUNDS[label]
            assert np.all(fitted[k] >= lo - 1e-9) and np.all(fitted[k] <= hi + 1e-9)


# ===========================================================================
# PCA machinery
# ===========================================================================

class TestGetPCA:
    def test_retains_dominant_modes_by_cumulative_variance(self, rng):
        # A rank-3 ensemble embedded in 40-D state space: three orthogonal
        # directions carrying essentially all the variance, plus tiny noise.
        n_state, n_sample = 40, 12
        basis = np.linalg.qr(rng.normal(size=(n_state, 3)))[0]
        coeffs = rng.normal(scale=10.0, size=(3, n_sample))
        edps = basis @ coeffs + rng.normal(scale=1e-6, size=(n_state, n_sample))

        PCA, mean, diag = P.get_PCA(edps, 0.999)
        assert diag.n_retained <= 4   # 3 real modes (+ maybe 1 for the noise floor)
        assert diag.cumulative_variance_ratio[diag.n_retained - 1] >= 0.999

    def test_more_components_never_increase_reconstruction_error(self, rng):
        edps = rng.normal(size=(30, 20)) + 500.0
        errs = []
        for thr in (0.3, 0.6, 0.9, 0.99, 0.999):
            PCA, mean, diag = P.get_PCA(edps, thr)
            recon = P.PCA2EDP_1D(P.EDP2PCA_1D(edps, PCA, mean=mean), PCA, mean=mean)
            errs.append(float(np.sqrt(np.mean((edps - recon) ** 2))))
        assert all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1))

    def test_full_retention_round_trip_preserves_mean(self, rng):
        """
        Regression: PCA2EDP_1D/EDP2PCA_1D previously ignored the mean
        get_PCA subtracts internally, so even full-rank reconstruction
        silently dropped the ensemble's mean profile.
        """
        edps = rng.normal(size=(25, 15)) + 500.0   # mean far from zero
        PCA, mean, diag = P.get_PCA(edps, 1.0)
        recon = P.PCA2EDP_1D(P.EDP2PCA_1D(edps, PCA, mean=mean), PCA, mean=mean)
        assert recon.mean() == pytest.approx(edps.mean(), abs=1e-6)
        assert np.sqrt(np.mean((edps - recon) ** 2)) < 1e-6

    def test_mean_none_is_backward_compatible(self, rng):
        """mean=None (the default) must reproduce the pre-mean-fix behaviour
        exactly -- a caller that already handles centering itself, or an
        externally-fit zero-mean basis, shouldn't be forced to pass one.
        Uses a full-rank basis so the round trip is exact regardless of mean
        handling (a truncated basis would lose information on its own,
        independent of the mean question this test targets)."""
        edps_zero_mean = np.array([[1.0, -1.0], [2.0, -2.0], [0.5, -0.5]])
        PCA = np.eye(3)   # full rank: lossless for any input, mean or not
        state = P.EDP2PCA_1D(edps_zero_mean, PCA, mean=None, linear=True)
        recon = P.PCA2EDP_1D(state, PCA, mean=None, linear=True)
        assert np.allclose(recon, edps_zero_mean)


class TestPCAJacobians1D:
    @pytest.mark.parametrize("linear", [True, False])
    def test_matches_finite_difference(self, rng, linear):
        n_alt, nPCA, nPts = 40, 4, 3
        PCA = rng.normal(size=(n_alt, nPCA))
        mean = rng.normal(scale=5.0, size=n_alt)
        state = rng.normal(size=(nPCA, nPts))

        jac = P.PCA2EDP_1D_map(state, PCA, mean=mean, linear=linear)
        eps = 1e-6
        max_err = 0.0
        for k in range(nPCA):
            for j in range(nPts):
                sp = state.copy(); sp[k, j] += eps
                sm = state.copy(); sm[k, j] -= eps
                fd = (P.PCA2EDP_1D(sp, PCA, mean=mean, linear=linear)[:, j]
                      - P.PCA2EDP_1D(sm, PCA, mean=mean, linear=linear)[:, j]) / (2 * eps)
                max_err = max(max_err, np.max(relative_error(jac[:, k, j], fd)))
        assert max_err < 1e-5


class TestPCAJacobians3D:
    @pytest.mark.parametrize("linear", [True, False])
    def test_matches_finite_difference(self, rng, linear):
        n_alt, n_geo, nPCA, nSample = 15, 6, 3, 2
        PCA = rng.normal(size=(n_alt, n_geo, nPCA))
        mean = rng.normal(scale=5.0, size=(n_alt, n_geo))
        state = rng.normal(size=(nPCA, nSample))

        jac = P.PCA2EDP_3D_map(state, PCA, mean=mean, linear=linear)
        eps = 1e-6
        max_err = 0.0
        for k in range(nPCA):
            for j in range(nSample):
                sp = state.copy(); sp[k, j] += eps
                sm = state.copy(); sm[k, j] -= eps
                fd = (P.PCA2EDP_3D(sp, PCA, mean=mean, linear=linear)[:, :, j]
                      - P.PCA2EDP_3D(sm, PCA, mean=mean, linear=linear)[:, :, j]) / (2 * eps)
                max_err = max(max_err, np.max(relative_error(jac[:, :, k, j], fd)))
        assert max_err < 1e-5


# ===========================================================================
# EDP_Parameterization registry
# ===========================================================================

class TestEDPParameterizationRegistry:
    def test_density_10ex_default_hyper_params(self):
        ep = P.EDP_Parameterization(style="density_10ex", hyper_params=None)
        assert ep.hyper_params["minlog10Density"] == 4.0

    def test_density_10ex_empty_dict_also_gets_defaults(self):
        """Regression: hyper_params={} (falsy, not None) previously skipped
        default-filling entirely, raising KeyError deep inside get_density."""
        ep = P.EDP_Parameterization(style="density_10ex", hyper_params={})
        assert ep.hyper_params["minlog10Density"] == 4.0

    def test_anchor_default_bounds(self):
        ep = P.EDP_Parameterization(style="ANCHOR", hyper_params=None)
        assert ep.hyper_params == P.ANCHOR_DEFAULT_BOUNDS

    def test_anchor_partial_override_merges_with_defaults(self):
        ep = P.EDP_Parameterization(style="ANCHOR", hyper_params={"hmF2": [150.0, 500.0]})
        assert ep.hyper_params["hmF2"] == [150.0, 500.0]
        assert ep.hyper_params["B0"] == P.ANCHOR_DEFAULT_BOUNDS["B0"]  # untouched default

    @pytest.mark.parametrize("style", PCA_STYLES)
    def test_pca_style_requires_pca_in_hyper_params(self, style):
        with pytest.raises(ValueError):
            P.EDP_Parameterization(style=style, hyper_params={})

    def test_pca_1d_wrong_ndim_raises(self, rng):
        with pytest.raises(ValueError):
            P.EDP_Parameterization(style="PCA_1D", hyper_params={"PCA": rng.normal(size=(5, 3, 2))})

    def test_pca_3d_wrong_ndim_raises(self, rng):
        with pytest.raises(ValueError):
            P.EDP_Parameterization(style="PCA_3D", hyper_params={"PCA": rng.normal(size=(5, 3))})

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError):
            P.EDP_Parameterization(style="not_a_style", hyper_params=None)

    def test_density_10ex_reconstruction_error_near_zero(self):
        ep = P.EDP_Parameterization(style="density_10ex")
        density = np.array([1e5, 1e8, 1e10, 1e11])
        residual, density_hat = ep.reconstruction_error(density)
        assert np.allclose(residual, 0.0, atol=1e-3)
        assert np.allclose(density_hat, density, rtol=1e-6)


# ===========================================================================
# Parameterized_EDPSamples
# ===========================================================================

class TestParameterizedEDPSamples:
    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_construction_produces_expected_param_vec_dims(self, parameterized_by_style, style):
        pe = parameterized_by_style[style]
        assert "param_vec" in pe.EDPSamples.data_vars
        assert pe.style == style

    def test_does_not_mutate_original_edpsamples(self, synthetic_edpsamples):
        original_vars = set(synthetic_edpsamples.data_vars)
        _ = P.Parameterized_EDPSamples(synthetic_edpsamples, style="density_10ex")
        assert set(synthetic_edpsamples.data_vars) == original_vars, (
            "Parameterized_EDPSamples must not add variables to the caller's "
            "original EDPSamples object"
        )

    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_error_summary_has_expected_keys(self, parameterized_by_style, style):
        summary = parameterized_by_style[style].error_summary()
        for key in ("style", "rmse_overall", "rmse_by_altitude", "max_abs_error"):
            assert key in summary
        if style == "ANCHOR":
            assert "rmse_by_region" in summary
            for region in ("topside", "bottomside", "intermediate", "E_layer"):
                assert region in summary["rmse_by_region"]
        if style in PCA_STYLES:
            assert "n_retained_components" in summary

    @pytest.mark.parametrize("style", ["density_10ex", "PCA_1D", "PCA_3D"])
    def test_netcdf_round_trip_preserves_param_vec(self, parameterized_by_style, style, tmp_path):
        pe = parameterized_by_style[style]
        path = tmp_path / f"{style}.nc"
        pe.saveNetCDF(str(path))
        pe2 = P.Parameterized_EDPSamples.fromNetCDF(str(path))

        assert pe2.style == style
        pv1 = pe.EDPSamples["param_vec"].to_numpy()
        pv2 = pe2.EDPSamples["param_vec"].to_numpy()
        assert pv1.shape == pv2.shape
        assert np.allclose(pv1, pv2, rtol=1e-5)

    def test_reuses_prefit_pca_basis_without_recomputing(self, synthetic_edpsamples):
        pe1 = P.Parameterized_EDPSamples(synthetic_edpsamples, style="PCA_1D",
                                          hyper_params={"retaining_threshold": 0.999})
        pca = pe1.Parameterization.hyper_params["PCA"]
        mean = pe1.Parameterization.hyper_params["PCA_mean"]

        pe2 = P.Parameterized_EDPSamples(synthetic_edpsamples, style="PCA_1D",
                                          hyper_params={"PCA": pca, "PCA_mean": mean})
        assert np.array_equal(pe2.Parameterization.hyper_params["PCA"], pca)
        pv1 = pe1.EDPSamples["param_vec"].to_numpy()
        pv2 = pe2.EDPSamples["param_vec"].to_numpy()
        assert np.allclose(pv1, pv2, rtol=1e-8)

    def test_pca_style_without_threshold_or_pca_raises(self, synthetic_edpsamples):
        with pytest.raises(ValueError):
            P.Parameterized_EDPSamples(synthetic_edpsamples, style="PCA_1D", hyper_params={})


# ===========================================================================
# Diagnostic plots: smoke tests only (headless backend via conftest.py)
# ===========================================================================

class TestDiagnosticPlots:
    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_plot_reconstruction_profile(self, parameterized_by_style, style):
        import matplotlib.pyplot as plt
        axes = parameterized_by_style[style].plot_reconstruction_profile(sample_idx=0, geo_idx=0)
        assert axes is not None
        plt.close("all")

    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_plot_reconstruction_error_statistics(self, parameterized_by_style, style):
        import matplotlib.pyplot as plt
        axes = parameterized_by_style[style].plot_reconstruction_error_statistics()
        assert axes is not None
        plt.close("all")

    @pytest.mark.parametrize("style", PCA_STYLES)
    def test_plot_pca_spectrum(self, parameterized_by_style, style):
        import matplotlib.pyplot as plt
        axes = parameterized_by_style[style].plot_pca_spectrum()
        assert axes is not None
        plt.close("all")

    def test_plot_pca_spectrum_without_diagnostics_raises(self, rng):
        PCA = rng.normal(size=(10, 3))
        ep = P.EDP_Parameterization(style="PCA_1D", hyper_params={"PCA": PCA})
        with pytest.raises(ValueError):
            ep.plot_pca_spectrum()
