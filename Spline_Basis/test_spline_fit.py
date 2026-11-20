# -*- coding: utf-8 -*-
"""Phase 2 tests for spline_fit.py (ratio-mode constrained batched fits)."""
from __future__ import annotations

import numpy as np
import pytest

from spline_basis import ConstrainedSplineBasis
from spline_fit import (BatchedSplineFit, _from_fit_space, concatenate_targets, prepare_targets,
                        residual_metrics, topside_scale_height)

H0, H1 = 90.0, 900.0
FIT_SPACES = [("linear", None), ("pow", 2.0), ("log10", None)]


def chapman_like(h, nm, hm, H_top):
    """Synthetic positive EDP with an exponential topside of scale height H_top above ~700 km."""
    z = (h - hm) / 60.0
    ne = nm * np.exp(1 - z - np.exp(-z))
    top = h > 700
    ne_700 = nm * np.exp(1 - (700 - hm) / 60.0 - np.exp(-(700 - hm) / 60.0))
    ne[top] = ne_700 * np.exp(-(h[top] - 700) / H_top)
    return ne + 1e8


def synthetic_collection(rng, n_col=4, n_mem=30, h=None):
    h = np.arange(H0, H1 + 1, 10.0) if h is None else h
    ne = np.empty((h.size, n_col, n_mem))
    for c in range(n_col):
        for j in range(n_mem):
            ne[:, c, j] = chapman_like(h, 10 ** rng.uniform(11.3, 12.1), rng.uniform(230, 330),
                                       rng.uniform(150, 250))
    return h, ne


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def grid_uniform(n):
    return np.linspace(H0, H1, n)


class TestScaleHeight:
    def test_exact_for_exponential(self):
        h = np.arange(800.0, 901.0, 10.0)
        mean = np.vstack([5e10 * np.exp(-(h - 800) / H) for H in (180.0, 300.0)])
        np.testing.assert_allclose(topside_scale_height(mean, h, 3), [180.0, 300.0])


class TestMatchesSingleColumnBasis:
    @pytest.mark.parametrize("fit_space,alpha", FIT_SPACES)
    def test_batched_equals_constrained_basis_lstsq(self, rng, fit_space, alpha):
        h, ne = synthetic_collection(rng, n_col=3, n_mem=5)
        t = prepare_targets(ne, h, h0=H0, h1=H1, fit_space=fit_space, alpha=alpha)
        grid = np.array([90, 110, 160, 230, 300, 400, 550, 700, 900.0])
        res = BatchedSplineFit(t).fit(grid)
        for c in range(t.n_col):
            cb = ConstrainedSplineBasis(grid, 3, lower_value=t.lower_value[c],
                                        upper_scale_height=None if np.isinf(t.H_eff[c]) else t.H_eff[c],
                                        upper_slope_offset=t.s0[c])
            w = np.sqrt(t.alt_weight[c])[:, None]
            A = cb.design_matrix(t.h)
            a, *_ = np.linalg.lstsq(w * A, w * (t.y[c] - cb.offset(t.h)[:, None]), rcond=None)
            np.testing.assert_allclose(res.coeffs[c], a, rtol=1e-8, atol=1e-10)


class TestBoundaryConditionsInDensity:
    @pytest.mark.parametrize("fit_space,alpha", FIT_SPACES)
    def test_ne_zero_at_h0_and_flat_at_h1(self, rng, fit_space, alpha):
        h, ne = synthetic_collection(rng, n_col=2, n_mem=4)
        t = prepare_targets(ne, h, h0=H0, h1=H1, fit_space=fit_space, alpha=alpha)
        grid = np.array([90, 120, 180, 250, 320, 420, 560, 720, 900.0])
        res = BatchedSplineFit(t).fit(grid)
        for c in range(t.n_col):
            H = t.scale_height[c]
            cb = ConstrainedSplineBasis(grid, 3, lower_value=t.lower_value[c],
                                        upper_scale_height=None if np.isinf(t.H_eff[c]) else t.H_eff[c],
                                        upper_slope_offset=t.s0[c])
            a = res.coeffs[c].T
            y0 = cb.evaluate(a, [H0])[:, 0]
            y1, dy1 = cb.evaluate(a, [H1])[:, 0], cb.evaluate(a, [H1], deriv=1)[:, 0]
            # mean near h1 is exactly exponential with scale height H in this setup
            m1 = t.mean[c, -1]
            dm1 = -m1 / H
            ne_h0 = _from_fit_space(y0, 1.0, fit_space, alpha)
            if fit_space == "log10":
                assert np.all(ne_h0 * np.interp(H0, t.h, t.mean[c]) <= 10 ** t.minlog10Density * 1.0001)
            else:
                np.testing.assert_allclose(ne_h0, 0.0, atol=1e-12)
            g = _from_fit_space(y1, 1.0, fit_space, alpha)
            eps = 1e-6
            gp = (_from_fit_space(y1 + eps, 1.0, fit_space, alpha)
                  - _from_fit_space(y1 - eps, 1.0, fit_space, alpha)) / (2 * eps)
            dne = dm1 * g + m1 * gp * dy1
            np.testing.assert_allclose(dne / (m1 / H), 0.0, atol=1e-6)


class TestExactRecovery:
    def test_profiles_in_the_spline_space_fit_exactly(self, rng):
        h = np.arange(H0, H1 + 1, 10.0)
        grid = np.array([90, 130, 200, 280, 380, 500, 650, 900.0])
        H = 250.0
        cb = ConstrainedSplineBasis(grid, 3, lower_value=0.0, upper_scale_height=H)
        mean = 1e11 * np.exp(-(h - H0) / H)
        r = cb.evaluate(rng.uniform(0.5, 1.5, (6, cb.n_free)), h)       # (6, n_h)
        ne = (mean[None, :] * r).T[:, None, :]                          # (n_h, 1, 6)
        t = prepare_targets(ne, h, h0=H0, h1=H1, mean=mean[:, None])
        np.testing.assert_allclose(t.scale_height, H)
        res = BatchedSplineFit(t).fit(grid)
        assert res.objective < 1e-20


class TestGradient:
    @pytest.mark.parametrize("fit_space,alpha", FIT_SPACES)
    @pytest.mark.parametrize("representation", ["ratio", "density"])
    def test_matches_finite_difference(self, rng, fit_space, alpha, representation):
        h, ne = synthetic_collection(rng, n_col=3, n_mem=6, h=np.arange(H0, H1 + 1, 5.0))
        t = prepare_targets(ne, h, h0=H0, h1=H1, fit_space=fit_space, alpha=alpha,
                            representation=representation, altitude_weight_beta=0.5,
                            profile_normalization=(representation == "density"))
        grid = np.array([90, 112.3, 171.9, 243.4, 318.8, 427.1, 566.6, 731.2, 900.0])
        fitter = BatchedSplineFit(t)
        res = fitter.fit(grid, gradient=True)
        eps = 1e-3
        fd = []
        for j in range(1, grid.size - 1):
            gp, gm = grid.copy(), grid.copy()
            gp[j] += eps
            gm[j] -= eps
            fd.append((fitter.fit(gp).objective - fitter.fit(gm).objective) / (2 * eps))
        fd = np.array(fd)
        np.testing.assert_allclose(res.gradient, fd, rtol=2e-4, atol=1e-6 * np.abs(fd).max())


class TestMomentObjective:
    @pytest.mark.parametrize("fit_space,alpha", FIT_SPACES)
    @pytest.mark.parametrize("representation", ["ratio", "density"])
    def test_equals_full_fit(self, rng, fit_space, alpha, representation):
        h, ne = synthetic_collection(rng, n_col=3, n_mem=7)
        t = prepare_targets(ne, h, h0=H0, h1=H1, fit_space=fit_space, alpha=alpha,
                            representation=representation, altitude_weight_beta=0.7,
                            profile_normalization=True)
        grid = np.array([90, 108.0, 166.0, 251.0, 333.0, 444.0, 590.0, 742.0, 900.0])
        f = BatchedSplineFit(t)
        full = f.fit(grid, gradient=True)
        R, g = f.objective(grid)
        assert R == pytest.approx(full.objective, rel=1e-9)
        np.testing.assert_allclose(g, full.gradient, rtol=1e-7, atol=1e-10 * np.abs(g).max())


class TestPooledMoments:
    def test_subset_members_keeps_full_objective(self, rng):
        h, ne = synthetic_collection(rng, n_col=3, n_mem=12)
        t = prepare_targets(ne, h, h0=H0, h1=H1)
        grid = np.array([90, 130, 200, 280, 380, 520, 700, 900.0])
        R_full, g_full = BatchedSplineFit(t).objective(grid)
        ts = t.subset_members(4)
        assert ts.n_mem == 4
        R_sub, g_sub = BatchedSplineFit(ts).objective(grid)
        assert R_sub == pytest.approx(R_full, rel=1e-12)
        np.testing.assert_allclose(g_sub, g_full, rtol=1e-10)
        both = concatenate_targets([ts, t.subset_members(4, np.random.default_rng(9))])
        R_both, _ = BatchedSplineFit(both).objective(grid)
        assert R_both == pytest.approx(2 * R_full, rel=1e-12)


class TestMetricsAndChecks:
    def test_metrics_keys_and_h0_excluded(self, rng):
        h, ne = synthetic_collection(rng, n_col=2, n_mem=5)
        t = prepare_targets(ne, h, h0=H0, h1=H1)
        assert np.all(t.alt_weight[:, 0] == 0.0)
        res = BatchedSplineFit(t).fit(grid_uniform(16))
        m = residual_metrics(t, res.y_fit)
        assert set(m["bands"]) == {"E", "F1", "F2", "topside"}
        assert "h0_sample_rms_ratio" in m and m["h0_sample_rms_ratio"] > 0.5
        assert m["all"]["rms_ratio"] < 0.2

    def test_rank_deficiency_detected(self, rng):
        h, ne = synthetic_collection(rng, n_col=1, n_mem=3, h=np.linspace(H0, H1, 17))
        t = prepare_targets(ne, h, h0=H0, h1=H1)
        with pytest.raises(np.linalg.LinAlgError):
            BatchedSplineFit(t).fit(np.array([90, 92, 94, 96, 98, 100, 500, 900.0]))

    def test_coverage_required(self, rng):
        h, ne = synthetic_collection(rng, n_col=1, n_mem=2)
        with pytest.raises(ValueError):
            prepare_targets(ne, h, h0=80.0, h1=H1)
