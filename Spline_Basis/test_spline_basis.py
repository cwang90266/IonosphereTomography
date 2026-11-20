# -*- coding: utf-8 -*-
"""
Phase 1 tests for spline_basis.py (plan: Documentation/Spline_Basis_Knot_Optimization_Plan.md).

Run with:  /opt/anaconda3/bin/python -m pytest Spline_Basis/ -q
"""
from __future__ import annotations

import mpmath
import numpy as np
import pytest
from scipy.integrate import quad

from spline_basis import (ConstrainedSplineBasis, SplineBasis, clamped_knot_vector,
                          divided_difference, reference_basis_P, upper_tie_factor)

DEGREES = [1, 2, 3, 5]


def random_grid(rng, n, h0=90.0, h1=900.0, min_gap=5.0):
    """Random strictly increasing grid with gaps >= min_gap."""
    gaps = rng.dirichlet(np.ones(n - 1)) * (h1 - h0 - (n - 1) * min_gap) + min_gap
    return np.concatenate([[h0], h0 + np.cumsum(gaps)])


def interior_points(grid, rng, n_per=3):
    """Random points strictly inside each grid interval (away from knots)."""
    pts = []
    for a, b in zip(grid[:-1], grid[1:]):
        pts.extend(a + (b - a) * rng.uniform(0.05, 0.95, n_per))
    return np.array(pts)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Divided differences
# ---------------------------------------------------------------------------
class TestDividedDifference:
    def test_polynomial_leading_coefficient(self):
        # [z_0..z_r] of a degree-r polynomial is its leading coefficient.
        f = lambda x: 3 * x ** 4 - 2 * x + 1
        assert divided_difference(f, [0.0, 0.3, 1.1, 2.0, 5.0]) == pytest.approx(3.0)

    def test_confluent_matches_derivative_over_factorial(self):
        f, df = np.exp, (lambda x, j: np.exp(x))
        assert divided_difference(f, [0.7, 0.7, 0.7], df) == pytest.approx(np.exp(0.7) / 2)

    def test_confluent_is_limit_of_distinct(self):
        f = np.sin
        df = lambda x, j: [np.sin, np.cos, lambda y: -np.sin(y), lambda y: -np.cos(y)][j](x)
        exact = divided_difference(f, [0.2, 0.5, 0.5, 1.0], df)
        near = divided_difference(f, [0.2, 0.5, 0.5 + 1e-5, 1.0])
        assert exact == pytest.approx(near, rel=1e-4)

    def test_repeated_points_without_derivative_raise(self):
        with pytest.raises(ValueError):
            divided_difference(np.exp, [0.0, 0.0])


# ---------------------------------------------------------------------------
# SplineBasis against the user's divided-difference definition
# ---------------------------------------------------------------------------
class TestBasisMatchesDefinition:
    @pytest.mark.parametrize("m", DEGREES)
    def test_P_normalization_equals_divided_difference(self, rng, m):
        grid = random_grid(rng, 9)
        basis = SplineBasis(grid, m, normalization="P")
        h = interior_points(grid, rng)
        B = basis.design_matrix(h)
        mpmath.mp.dps = 50
        knots = [mpmath.mpf(float(v)) for v in basis.knots]
        for k in range(basis.n_basis):
            ref = np.array([float(reference_basis_P(knots, k, m, mpmath.mpf(float(t)))) for t in h])
            np.testing.assert_allclose(B[:, k], ref, rtol=1e-9, atol=1e-14 * np.abs(ref).max())

    def test_N_is_width_times_P(self, rng):
        grid = random_grid(rng, 7)
        h = interior_points(grid, rng)
        N = SplineBasis(grid, 3, "N").design_matrix(h)
        P = SplineBasis(grid, 3, "P").design_matrix(h)
        M = SplineBasis(grid, 3, "M").design_matrix(h)
        w = SplineBasis(grid, 3).support_widths()
        np.testing.assert_allclose(N, P * w, atol=1e-13)
        np.testing.assert_allclose(M, 4 * P, atol=1e-13)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------
class TestBasisProperties:
    @pytest.mark.parametrize("m", DEGREES)
    def test_count_and_partition_of_unity(self, rng, m):
        grid = random_grid(rng, 11)
        basis = SplineBasis(grid, m)
        assert basis.n_basis == 11 + m - 1
        h = np.linspace(grid[0], grid[-1], 501)          # includes both ends
        B = basis.design_matrix(h)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(B >= -1e-14)

    @pytest.mark.parametrize("m", DEGREES)
    def test_M_has_unit_integral(self, rng, m):
        grid = random_grid(rng, 6)
        basis = SplineBasis(grid, m, "M")
        np.testing.assert_allclose(basis.integrals(), 1.0)
        for k in range(basis.n_basis):
            num = sum(quad(lambda x: basis.design_matrix([x])[0, k], a, b)[0]
                      for a, b in zip(grid[:-1], grid[1:]))
            assert num == pytest.approx(1.0, rel=1e-8)

    @pytest.mark.parametrize("m", DEGREES)
    def test_support(self, rng, m):
        grid = random_grid(rng, 10)
        basis = SplineBasis(grid, m)
        t = basis.knots
        h = interior_points(grid, rng, 5)
        B = basis.design_matrix(h)
        for k in range(basis.n_basis):
            outside = (h < t[k]) | (h > t[k + m + 1])
            assert np.all(B[outside, k] == 0.0)

    @pytest.mark.parametrize("m", DEGREES)
    def test_reproduces_polynomials_up_to_degree(self, rng, m):
        grid = random_grid(rng, 8)
        basis = SplineBasis(grid, m)
        h = np.linspace(grid[0], grid[-1], 400)
        B = basis.design_matrix(h)
        s = (h - 500.0) / 400.0
        for p in range(m + 1):
            c, *_ = np.linalg.lstsq(B, s ** p, rcond=None)
            np.testing.assert_allclose(B @ c, s ** p, atol=1e-10)

    @pytest.mark.parametrize("m", DEGREES)
    def test_derivative_matches_finite_difference(self, rng, m):
        grid = random_grid(rng, 8)
        basis = SplineBasis(grid, m)
        h = interior_points(grid, rng)
        eps = 1e-5
        fd = (basis.design_matrix(h + eps) - basis.design_matrix(h - eps)) / (2 * eps)
        np.testing.assert_allclose(basis.design_matrix(h, deriv=1), fd, atol=1e-7)

    def test_rejects_bad_grids_and_out_of_range(self):
        with pytest.raises(ValueError):
            SplineBasis([100.0, 100.0, 200.0])
        with pytest.raises(ValueError):
            SplineBasis([100.0])
        with pytest.raises(ValueError):
            SplineBasis([100.0, 200.0], degree=0)
        with pytest.raises(ValueError):
            SplineBasis([100.0, 200.0]).design_matrix([250.0])

    def test_clamped_knot_vector(self):
        t = clamped_knot_vector(np.array([0.0, 1.0, 3.0]), 3)
        np.testing.assert_array_equal(t, [0, 0, 0, 0, 1, 3, 3, 3, 3])


# ---------------------------------------------------------------------------
# Analytic grid-point Jacobian
# ---------------------------------------------------------------------------
def _fd_knot_jacobian(make, grid, h, eps=1e-4):
    out = []
    for j in range(1, grid.size - 1):
        gp, gm = grid.copy(), grid.copy()
        gp[j] += eps
        gm[j] -= eps
        out.append((make(gp).design_matrix(h) - make(gm).design_matrix(h)) / (2 * eps))
    return np.array(out)


class TestKnotJacobian:
    @pytest.mark.parametrize("m", DEGREES)
    @pytest.mark.parametrize("norm", ["N", "P", "M"])
    def test_matches_finite_difference(self, rng, m, norm):
        grid = random_grid(rng, 8, min_gap=20.0)
        h = interior_points(grid, rng)
        h = h[np.min(np.abs(h[:, None] - grid[None, :]), axis=1) > 1e-2]  # away from knots
        basis = SplineBasis(grid, m, norm)
        J = basis.knot_jacobian(h)
        fd = _fd_knot_jacobian(lambda g: SplineBasis(g, m, norm), grid, h)
        scale = np.abs(fd).max()
        np.testing.assert_allclose(J, fd, atol=1e-6 * scale)


# ---------------------------------------------------------------------------
# Constrained basis (boundary conditions)
# ---------------------------------------------------------------------------
class TestConstrainedBasis:
    @pytest.mark.parametrize("m", DEGREES)
    def test_free_count(self, rng, m):
        grid = random_grid(rng, 12)
        cb = ConstrainedSplineBasis(grid, m)
        assert cb.n_free == 12 + m - 3
        if m == 3:
            assert cb.n_free == 12

    @pytest.mark.parametrize("m", DEGREES)
    @pytest.mark.parametrize("lower", [0.0, 7.5])
    def test_lower_value_and_zero_slope_hold_exactly(self, rng, m, lower):
        grid = random_grid(rng, 10)
        cb = ConstrainedSplineBasis(grid, m, lower_value=lower, upper_bc="slope")
        a = rng.normal(size=(5, cb.n_free))
        y0 = cb.evaluate(a, [grid[0]])[:, 0]
        dy1 = cb.evaluate(a, [grid[-1]], deriv=1)[:, 0]
        np.testing.assert_allclose(y0, lower, atol=1e-12)
        np.testing.assert_allclose(dy1, 0.0, atol=1e-10)

    @pytest.mark.parametrize("m", DEGREES)
    def test_ratio_mode_gives_zero_density_slope(self, rng, m):
        # Ne = mean * r with mean = A exp(-(h - h1)/H): dNe/dh(h1) = 0 exactly.
        grid = random_grid(rng, 10)
        H = 300.0
        cb = ConstrainedSplineBasis(grid, m, lower_value=0.0, upper_scale_height=H)
        a = rng.uniform(0.5, 1.5, size=(4, cb.n_free))
        h1 = grid[-1]
        r, dr = cb.evaluate(a, [h1])[:, 0], cb.evaluate(a, [h1], deriv=1)[:, 0]
        mean, dmean = 1.0, -1.0 / H
        np.testing.assert_allclose(dmean * r + mean * dr, 0.0, atol=1e-12)
        np.testing.assert_allclose(cb.evaluate(a, [grid[0]])[:, 0], 0.0, atol=1e-12)

    def test_free_boundaries_recover_full_basis(self, rng):
        grid = random_grid(rng, 7)
        cb = ConstrainedSplineBasis(grid, 3, lower_value=None, upper_bc=None)
        assert cb.n_free == cb.basis.n_basis
        h = interior_points(grid, rng)
        np.testing.assert_allclose(cb.design_matrix(h), cb.basis.design_matrix(h))

    def test_reproduces_any_constrained_spline(self, rng):
        grid = random_grid(rng, 15)
        cb = ConstrainedSplineBasis(grid, 3, lower_value=0.0, upper_scale_height=280.0)
        h = np.linspace(grid[0], grid[-1], 300)
        a_true = rng.normal(size=cb.n_free)
        y = cb.evaluate(a_true, h)
        a_fit, *_ = np.linalg.lstsq(cb.design_matrix(h), y - cb.offset(h), rcond=None)
        np.testing.assert_allclose(a_fit, a_true, atol=1e-10)

    @pytest.mark.parametrize("m", DEGREES)
    @pytest.mark.parametrize("H,s0", [(None, 0.002), (-400.0, 0.0), (350.0, -0.001)])
    def test_general_upper_slope_condition(self, rng, m, H, s0):
        grid = random_grid(rng, 9)
        cb = ConstrainedSplineBasis(grid, m, lower_value=1.0, upper_scale_height=H,
                                    upper_slope_offset=s0)
        a = rng.normal(size=(3, cb.n_free))
        y1 = cb.evaluate(a, [grid[-1]])[:, 0]
        dy1 = cb.evaluate(a, [grid[-1]], deriv=1)[:, 0]
        expected = (0.0 if H is None else y1 / H) + s0
        np.testing.assert_allclose(dy1, expected, atol=1e-12)

    def test_degenerate_upper_tie_raises(self):
        with pytest.raises(ValueError):
            upper_tie_factor(np.array([0.0, 100.0, 1000.0]), 3, scale_height=200.0)

    @pytest.mark.parametrize("H,s0", [(None, 0.0), (300.0, 0.0), (None, 0.0015)])
    @pytest.mark.parametrize("lower", [0.0, 3.0])
    def test_knot_jacobian_matches_finite_difference(self, rng, H, s0, lower):
        grid = random_grid(rng, 8, min_gap=20.0)
        h = interior_points(grid, rng)
        h = h[np.min(np.abs(h[:, None] - grid[None, :]), axis=1) > 1e-2]
        make = lambda g: ConstrainedSplineBasis(g, 3, lower_value=lower, upper_scale_height=H,
                                                upper_slope_offset=s0)
        dBt, doff = make(grid).knot_jacobian(h)
        fd = _fd_knot_jacobian(make, grid, h)
        np.testing.assert_allclose(dBt, fd, atol=1e-6 * np.abs(fd).max())
        eps = 1e-4
        fd_off = []
        for j in range(1, grid.size - 1):
            gp, gm = grid.copy(), grid.copy()
            gp[j] += eps
            gm[j] -= eps
            fd_off.append((make(gp).offset(h) - make(gm).offset(h)) / (2 * eps))
        np.testing.assert_allclose(doff, np.array(fd_off), atol=1e-6 * max(1.0, lower))
