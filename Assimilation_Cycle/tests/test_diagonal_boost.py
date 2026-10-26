# -*- coding: utf-8 -*-
"""
Tests for diagonal_boost.py (user-proposed 2026-09-27).

Builds a small synthetic multi-point EDPSamples directly (via the ``edps=``
constructor path, which skips IRI2020 entirely -- see
edp_samples.py's __init__: passing ``edps``/``feature_edps`` explicitly
bypasses ``get_IRI2020_EDP``), so these tests need no IRI2020 executable
and no real data on disk. The synthetic ensemble is built with a single
shared low-rank ("climatological") mode across all members, points, and
heights -- the exact pathology diagonal boosting targets -- so the tests
can directly check that boosting adds genuinely new, less-correlated
spread rather than just inflating along the same existing mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Assimilation_Cycle.diagonal_boost import apply_diagonal_boost


def _make_synthetic_edp_samples(n_height=6, n_members=400, seed=0):
    """Builds a real EDPSamples via the ``Regional`` geo_type (the same
    code path used for every real grid in this project, so
    ``geolocation`` comes from the module's own real, well-exercised grid
    generator), then overwrites ``EDPs`` with a hand-built ensemble that
    has a single shared low-rank ("climatological") mode across every
    point/height -- the exact pathology diagonal boosting targets."""
    import edp_samples as E

    rng = np.random.default_rng(seed)
    altitude = np.linspace(100.0, 600.0, n_height)
    sampling_parameters = pd.DataFrame({"f107": rng.uniform(100, 150, n_members)})

    edp = E.EDPSamples(
        DateTime="2024-06-15T00:00:00",
        geo_type="Regional",
        altitude=altitude,
        sampling_parameters=sampling_parameters,
        Lon=19.0, Lat=69.6, radius=5.0, dLat=2.0,
    )
    n_geo = edp.geolocation.shape[0]
    assert n_geo >= 3, f"need several geo points to test horizontal smoothing, got {n_geo}"

    shared_mode = rng.standard_normal(n_members)
    base_density = 1e11
    edps = base_density * (1.0 + 0.1 * shared_mode)[None, None, :] * np.ones((n_height, n_geo, 1))
    edps = edps.astype(np.float64)

    edp = edp.copy(deep=True)
    var_name = edp.VAR_EDPS
    edp[var_name] = (edp[var_name].dims, edps.astype(edp[var_name].dtype))
    return edp


class TestNoOp:
    def test_amplitude_none_is_noop(self):
        edp = _make_synthetic_edp_samples()
        result = apply_diagonal_boost(edp, None)
        assert result is edp

    def test_amplitude_zero_is_noop(self):
        edp = _make_synthetic_edp_samples()
        result = apply_diagonal_boost(edp, 0.0)
        assert result is edp


class TestStatistics:
    def test_preserves_original_unchanged(self):
        edp = _make_synthetic_edp_samples()
        original = edp.edps.copy()
        apply_diagonal_boost(edp, 0.5, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(edp.edps, original)

    def test_preserves_per_point_mean(self):
        edp = _make_synthetic_edp_samples(n_members=2000)
        boosted = apply_diagonal_boost(edp, 0.5, rng=np.random.default_rng(1))
        mean_before = edp.edps.mean(axis=2)
        mean_after = boosted.edps.mean(axis=2)
        # relative tolerance: finite-ensemble mean of injected noise isn't
        # exactly zero, but should be tiny relative to the signal itself
        np.testing.assert_allclose(mean_after, mean_before, rtol=1e-2)

    def test_increases_per_point_variance_by_expected_amount(self):
        # Explicit taper_start_km disables the altitude taper (default
        # on since 2026-09-29) -- this test is about the core
        # amplitude-scaling mechanism, verified taper-free; TestAltitudeTaper
        # below verifies the taper itself precisely.
        edp = _make_synthetic_edp_samples(n_members=3000)
        amplitude = 0.4
        boosted = apply_diagonal_boost(edp, amplitude, rng=np.random.default_rng(2), taper_start_km=1e6, taper_end_km=2e6)
        std_before = edp.edps.std(axis=2)
        std_after = boosted.edps.std(axis=2)
        expected = np.sqrt(std_before ** 2 + (amplitude * std_before) ** 2)
        np.testing.assert_allclose(std_after, expected, rtol=0.15)

    def test_zero_amplitude_via_config_path_unaffected(self):
        # amplitude too small to matter numerically should still run and
        # not raise, exercising the full smoothing pipeline at n_geo>1.
        edp = _make_synthetic_edp_samples()
        boosted = apply_diagonal_boost(edp, 1e-6, rng=np.random.default_rng(3))
        assert boosted.edps.shape == edp.edps.shape


class TestDecorrelation:
    def test_reduces_cross_point_correlation(self):
        """The synthetic ensemble's points are perfectly correlated (single
        shared mode) before boosting; a large boost with a short
        horizontal smoothing scale should measurably reduce correlation
        between distant points, since the injected spread is
        point-specific rather than shared."""
        edp = _make_synthetic_edp_samples(n_members=3000)
        h = 0
        corr_before = np.corrcoef(edp.edps[h, 0, :], edp.edps[h, -1, :])[0, 1]
        assert corr_before > 0.999   # by construction, single shared mode

        boosted = apply_diagonal_boost(
            edp, amplitude=2.0, horizontal_scale_km=20.0, vertical_scale_km=10.0,
            rng=np.random.default_rng(4),
        )
        corr_after = np.corrcoef(boosted.edps[h, 0, :], boosted.edps[h, -1, :])[0, 1]
        assert corr_after < corr_before - 0.1

    def test_larger_horizontal_scale_keeps_more_correlation(self):
        """A horizontal smoothing scale much larger than the point spacing
        should leave nearby points more correlated than a short scale,
        since the injected field itself is closer to a shared mode."""
        edp = _make_synthetic_edp_samples(n_members=3000)
        h = 0
        boosted_short = apply_diagonal_boost(
            edp, amplitude=2.0, horizontal_scale_km=5.0, vertical_scale_km=10.0,
            rng=np.random.default_rng(5),
        )
        boosted_long = apply_diagonal_boost(
            edp, amplitude=2.0, horizontal_scale_km=2000.0, vertical_scale_km=10.0,
            rng=np.random.default_rng(5),
        )
        corr_short = np.corrcoef(boosted_short.edps[h, 0, :], boosted_short.edps[h, 1, :])[0, 1]
        corr_long = np.corrcoef(boosted_long.edps[h, 0, :], boosted_long.edps[h, 1, :])[0, 1]
        assert corr_long > corr_short


class TestAltitudeTaper:
    """Added 2026-09-29: user noticed analysis EDPs got visibly wavy at
    high altitude (with iri_spread_kwargs widened, Section 22) without a
    corresponding TEC-residual improvement -- TEC has little sensitivity
    to density that high, so the EnKF can't meaningfully constrain
    whatever a flat-amplitude boost injects there. These tests verify the
    altitude taper precisely, not just "runs without error" -- the
    existing fixture's altitude grid is 100-600km (6 points at 100km
    steps), so custom (not default) taper_start/end_km below are chosen
    to land squarely within that range: below start (full amplitude), in
    the linear transition, and at/above end (floor)."""

    def test_below_taper_start_gets_full_amplitude(self):
        edp = _make_synthetic_edp_samples(n_members=3000)
        amplitude = 0.4
        boosted = apply_diagonal_boost(
            edp, amplitude, rng=np.random.default_rng(2),
            taper_start_km=200.0, taper_end_km=400.0, taper_floor=0.1,
        )
        std_before = edp.edps.std(axis=2)
        std_after = boosted.edps.std(axis=2)
        expected_full = np.sqrt(std_before ** 2 + (amplitude * std_before) ** 2)
        # altitude[0]=100km, altitude[1]=200km -- both at/below taper_start_km
        np.testing.assert_allclose(std_after[:2], expected_full[:2], rtol=0.15)

    def test_at_and_above_taper_end_gets_floor_amplitude(self):
        edp = _make_synthetic_edp_samples(n_members=3000)
        amplitude = 0.4
        floor = 0.1
        boosted = apply_diagonal_boost(
            edp, amplitude, rng=np.random.default_rng(2),
            taper_start_km=200.0, taper_end_km=400.0, taper_floor=floor,
        )
        std_before = edp.edps.std(axis=2)
        std_after = boosted.edps.std(axis=2)
        effective = amplitude * floor
        expected_floor = np.sqrt(std_before ** 2 + (effective * std_before) ** 2)
        # altitude[3]=400km (==end), [4]=500km, [5]=600km -- all at/above taper_end_km
        np.testing.assert_allclose(std_after[3:], expected_floor[3:], rtol=0.2)

    def test_amplitude_decreases_monotonically_through_the_transition(self):
        """Injected variance (relative to the untapered baseline) should
        be highest at the lowest altitude and lowest at the highest --
        checked via the actual std increase, not by inspecting internals."""
        edp = _make_synthetic_edp_samples(n_members=3000)
        boosted = apply_diagonal_boost(
            edp, amplitude=0.6, rng=np.random.default_rng(3),
            taper_start_km=100.0, taper_end_km=600.0, taper_floor=0.05,
        )
        std_before = edp.edps.std(axis=2)
        std_after = boosted.edps.std(axis=2)
        added_variance = std_after.mean(axis=1) ** 2 - std_before.mean(axis=1) ** 2
        assert np.all(np.diff(added_variance) <= 1e-6)   # non-increasing with altitude

    def test_taper_floor_zero_means_no_boost_above_taper_end(self):
        edp = _make_synthetic_edp_samples(n_members=3000)
        boosted = apply_diagonal_boost(
            edp, amplitude=0.6, rng=np.random.default_rng(3),
            taper_start_km=200.0, taper_end_km=400.0, taper_floor=0.0,
        )
        std_before = edp.edps.std(axis=2)
        std_after = boosted.edps.std(axis=2)
        # altitude[3:] (400/500/600km) should be essentially unperturbed
        np.testing.assert_allclose(std_after[3:], std_before[3:], rtol=0.05)

    def test_default_taper_matches_confirmed_400_700_01(self):
        edp = _make_synthetic_edp_samples(n_members=3000)
        amplitude = 0.4
        boosted_default = apply_diagonal_boost(edp, amplitude, rng=np.random.default_rng(2))
        boosted_explicit = apply_diagonal_boost(
            edp, amplitude, rng=np.random.default_rng(2),
            taper_start_km=400.0, taper_end_km=700.0, taper_floor=0.1,
        )
        np.testing.assert_array_equal(boosted_default.edps, boosted_explicit.edps)

    def test_end_km_must_exceed_start_km(self):
        edp = _make_synthetic_edp_samples()
        with pytest.raises(ValueError, match="taper"):
            apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(1),
                                  taper_start_km=500.0, taper_end_km=400.0)


class TestPhysicalValidity:
    def test_stays_non_negative_even_with_large_amplitude(self):
        edp = _make_synthetic_edp_samples()
        boosted = apply_diagonal_boost(edp, amplitude=50.0, rng=np.random.default_rng(6))
        assert np.all(boosted.edps >= 1.0)


class TestReproducibility:
    def test_same_seed_reproduces_exactly(self):
        edp = _make_synthetic_edp_samples()
        r1 = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42))
        r2 = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1.edps, r2.edps)

    def test_different_seed_differs(self):
        edp = _make_synthetic_edp_samples()
        r1 = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42))
        r2 = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(43))
        assert not np.allclose(r1.edps, r2.edps)


class TestReturnType:
    def test_returns_edp_samples_with_metadata_preserved(self):
        edp = _make_synthetic_edp_samples()
        boosted = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(7))
        assert type(boosted) is type(edp)
        np.testing.assert_array_equal(boosted.altitude, edp.altitude)
        np.testing.assert_array_equal(boosted.geolocation, edp.geolocation)
        assert boosted.edps.shape == edp.edps.shape


class TestLogSpace:
    """log_space=True perturbs log10(density) instead of density directly
    -- added after a real-data sweep found linear-space boosting causes
    an 8-12 order-of-magnitude discontinuity (via the positivity clip)
    for log10-based styles (density_10ex/PCA_*_10ex). See module
    docstring and Assimilation_Cycle_Integration_Plan.md Section 17."""

    def test_stays_positive_even_at_extreme_amplitude_no_clip_needed(self):
        """The whole point of log_space: 10**(anything) is always > 0, so
        there should never be a value sitting exactly at the linear-space
        clip floor the way linear-space boosting produces."""
        edp = _make_synthetic_edp_samples()
        boosted = apply_diagonal_boost(edp, amplitude=50.0, rng=np.random.default_rng(6), log_space=True)
        assert np.all(boosted.edps > 0)
        assert not np.any(boosted.edps == 1.0)   # the linear-space floor value

    def test_preserves_geometric_mean_not_arithmetic_mean(self):
        """Zero-mean noise in log10 space preserves the per-point
        geometric mean (== 10**(mean of log10 values)), not the
        arithmetic mean -- Jensen's inequality means the arithmetic mean
        shifts slightly, so only the geometric-mean/log-mean check should
        be exact here."""
        edp = _make_synthetic_edp_samples(n_members=2000)
        boosted = apply_diagonal_boost(edp, 0.5, rng=np.random.default_rng(1), log_space=True)
        log_mean_before = np.log10(edp.edps).mean(axis=2)
        log_mean_after = np.log10(boosted.edps).mean(axis=2)
        np.testing.assert_allclose(log_mean_after, log_mean_before, rtol=1e-2)

    def test_increases_log_space_variance_by_expected_amount(self):
        # taper_start_km=1e6 disables the altitude taper -- see the note
        # on the linear-space equivalent test above.
        edp = _make_synthetic_edp_samples(n_members=3000)
        amplitude = 0.4
        boosted = apply_diagonal_boost(edp, amplitude, rng=np.random.default_rng(2), log_space=True, taper_start_km=1e6, taper_end_km=2e6)
        log_std_before = np.log10(edp.edps).std(axis=2)
        log_std_after = np.log10(boosted.edps).std(axis=2)
        expected = np.sqrt(log_std_before ** 2 + (amplitude * log_std_before) ** 2)
        np.testing.assert_allclose(log_std_after, expected, rtol=0.15)

    def test_reduces_cross_point_correlation_in_log_space(self):
        edp = _make_synthetic_edp_samples(n_members=3000)
        h = 0
        log_before = np.log10(edp.edps)
        corr_before = np.corrcoef(log_before[h, 0, :], log_before[h, -1, :])[0, 1]
        assert corr_before > 0.999

        boosted = apply_diagonal_boost(
            edp, amplitude=2.0, horizontal_scale_km=20.0, vertical_scale_km=10.0,
            rng=np.random.default_rng(4), log_space=True,
        )
        log_after = np.log10(boosted.edps)
        corr_after = np.corrcoef(log_after[h, 0, :], log_after[h, -1, :])[0, 1]
        assert corr_after < corr_before - 0.1

    def test_reproducible_with_seed(self):
        edp = _make_synthetic_edp_samples()
        r1 = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42), log_space=True)
        r2 = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42), log_space=True)
        np.testing.assert_array_equal(r1.edps, r2.edps)

    def test_differs_from_linear_space_result(self):
        edp = _make_synthetic_edp_samples()
        linear = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42), log_space=False)
        log = apply_diagonal_boost(edp, 0.3, rng=np.random.default_rng(42), log_space=True)
        assert not np.allclose(linear.edps, log.edps)
