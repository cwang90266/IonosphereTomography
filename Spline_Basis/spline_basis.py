# -*- coding: utf-8 -*-
"""
Polynomial spline basis on a vertical grid, for parameterizing electron
density profiles (EDPs) -- Phase 1 of
Documentation/Spline_Basis_Knot_Optimization_Plan.md.

Definition (the user's)
-----------------------
For a grid h0 = x_0 < x_1 < ... < x_{n-1} = h1 (``n`` points including both
ends) and polynomial degree ``m`` (cubic: m = 3),

    P_k(t) = [x_k, ..., x_{k+m+1}] p_t,      p_t(x) = (x - t)_+^m ,

the divided difference of the truncated power in its first argument. This is
the Curry-Schoenberg B-spline up to normalization:

    M_k = (m+1) P_k                      (unit integral)
    N_k = (t_{k+m+1} - t_k) P_k          (partition of unity; DEFAULT here)

Boundary ("extra points"): the grid end points are repeated m more times
(clamped knot vector), giving K = n + m - 1 basis functions spanning the full
spline space on [h0, h1]. Repeated points make the divided differences
confluent (derivatives of p_t); the reference implementation
``divided_difference`` handles that.

Numerics: the divided-difference formula cancels catastrophically for close
knots, so it is only the *reference* (tests, mpmath precision). Production
evaluation uses the equivalent stable Cox-de Boor recursion via
``scipy.interpolate.BSpline``.

Grid-point derivatives: differentiating a divided difference w.r.t. one of its
(simple) points repeats that point,

    d/dx_q [.., x_q, ..] f = [.., x_q, x_q, ..] f ,

and [y_0..y_{m+2}](. - t)_+^m = -1/(m+1) d/dt [y_0..y_{m+2}](. - t)_+^{m+1},
i.e. the derivative of a degree-(m+1) B-spline on the augmented knot vector.
``SplineBasis.knot_jacobian`` uses exactly this, so it is analytic.

Constrained basis (boundary conditions, see plan sections 1.2 / 1.3a)
--------------------------------------------------------------------
``ConstrainedSplineBasis`` imposes, through the coefficients c (y = sum c_k N_k):

* lower:  y(h0) = lower_value          ->  c_0 = lower_value (fixed)
* upper:  y'(h1) = y(h1) / H + s0      ->  c_{K-2} = gamma * c_{K-1} - Delta s0 / m,
                                            gamma = 1 - Delta / (m H),
                                            Delta = h1 - x_{n-2}
  H = inf (``upper_scale_height=None``) and s0 = 0 give y'(h1) = 0. In ratio
  mode (Ne = mean * g(y)) dNe/dh(h1) = 0 becomes y' = g(y) / (g'(y) H) with
  H the mean's topside scale height: H_eff = H (linear r), alpha*H (pow),
  or H = inf with s0 = 1/(ln10 H) (log10 r). See spline_fit.py.

so y = lower_value * N_0 + B_tilde @ a with K - 2 = n + m - 3 free
coefficients a (= n for cubic).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import BSpline

__all__ = [
    "truncated_power", "truncated_power_derivative", "divided_difference",
    "reference_basis_P", "clamped_knot_vector", "SplineBasis",
    "ConstrainedSplineBasis", "upper_tie_factor",
]

_NORMALIZATIONS = ("N", "M", "P")


# ===========================================================================
# Reference implementation: divided differences of the truncated power
# ===========================================================================

def truncated_power(x, t, m: int):
    """p_t(x) = (x - t)^m if x > t else 0 (strict, as in the definition)."""
    return (x - t) ** m if x > t else 0 * (x - t)


def truncated_power_derivative(x, t, m: int, order: int):
    """d^order/dx^order of (x - t)_+^m (strict convention at x = t)."""
    if order > m:
        return 0 * (x - t)
    if not x > t:
        return 0 * (x - t)
    coef = math.factorial(m) // math.factorial(m - order)
    return coef * (x - t) ** (m - order)


def divided_difference(f: Callable, points: Sequence, fderiv: Callable | None = None):
    """
    Divided difference [z_0, ..., z_r] f, including the confluent case.

    ``points`` may contain repeated values; then ``fderiv(z, j)`` must return
    the j-th derivative of f at z, and the table uses f^{(j)}(z)/j! for a
    run of j+1 equal points. Works with any numeric type supporting the
    arithmetic (Python float, ``mpmath.mpf``), so it can be evaluated in
    extended precision for tests.
    """
    z = sorted(points)
    r = len(z) - 1
    col = [f(zi) for zi in z]           # order-0 column
    for k in range(1, r + 1):
        new = []
        for i in range(r - k + 1):
            if z[i + k] == z[i]:
                if fderiv is None:
                    raise ValueError("Repeated points need fderiv for confluent divided differences.")
                new.append(fderiv(z[i], k) / math.factorial(k))
            else:
                new.append((col[i + 1] - col[i]) / (z[i + k] - z[i]))
        col = new
    return col[0]


def reference_basis_P(knots: Sequence, k: int, m: int, t):
    """
    P_k(t) = [knots_k, ..., knots_{k+m+1}] (. - t)_+^m by divided differences.

    Reference only (cancellation for close knots); pass ``mpmath.mpf`` knots
    and t for extended precision. At t equal to a knot the strict convention
    of p_t applies, which differs from the right-continuous Cox-de Boor value
    only at a clamped left end; evaluate at non-knot t.
    """
    pts = list(knots[k:k + m + 2])
    return divided_difference(
        lambda x: truncated_power(x, t, m), pts,
        lambda x, j: truncated_power_derivative(x, t, m, j))


# ===========================================================================
# Knot vector
# ===========================================================================

def clamped_knot_vector(grid: np.ndarray, degree: int) -> np.ndarray:
    """Grid with both end points repeated ``degree`` extra times (multiplicity degree+1)."""
    grid = np.asarray(grid, dtype=float)
    return np.concatenate([np.full(degree, grid[0]), grid, np.full(degree, grid[-1])])


def _validate_grid(grid: np.ndarray, degree: int) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be 1-D with at least 2 points (h0 and h1).")
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid must be finite.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing.")
    if int(degree) != degree or degree < 1:
        raise ValueError("degree must be an integer >= 1.")
    return grid


# ===========================================================================
# Unconstrained basis
# ===========================================================================

class SplineBasis:
    """
    Degree-``degree`` B-spline basis on a clamped knot vector built from
    ``grid`` (n points, h0 = grid[0], h1 = grid[-1]); K = n + degree - 1
    functions.

    normalization: 'N' (default, partition of unity), 'M' (unit integral,
    = (m+1) P_k) or 'P' (the divided-difference definition itself).
    """

    def __init__(self, grid: Sequence[float], degree: int = 3, normalization: str = "N"):
        if normalization not in _NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {_NORMALIZATIONS}.")
        self.degree = int(degree)
        self.grid = _validate_grid(grid, self.degree)
        self.normalization = normalization
        self.knots = clamped_knot_vector(self.grid, self.degree)
        self.n_basis = self.grid.size + self.degree - 1
        self._spline = BSpline(self.knots, np.eye(self.n_basis), self.degree, extrapolate=False)

    # -- basic properties ---------------------------------------------------
    @property
    def h0(self) -> float:
        return float(self.grid[0])

    @property
    def h1(self) -> float:
        return float(self.grid[-1])

    @property
    def n_points(self) -> int:
        return int(self.grid.size)

    def support_widths(self) -> np.ndarray:
        """t_{k+m+1} - t_k for each basis function (the N <-> P factor)."""
        m = self.degree
        return self.knots[m + 1:m + 1 + self.n_basis] - self.knots[:self.n_basis]

    def _scale(self) -> np.ndarray:
        """Per-column factor converting N_k into the requested normalization."""
        if self.normalization == "N":
            return np.ones(self.n_basis)
        w = self.support_widths()
        return 1.0 / w if self.normalization == "P" else (self.degree + 1) / w

    def _check_h(self, h) -> np.ndarray:
        h = np.atleast_1d(np.asarray(h, dtype=float))
        tol = 1e-9 * max(1.0, abs(self.h1 - self.h0))
        if np.any(h < self.h0 - tol) or np.any(h > self.h1 + tol):
            raise ValueError(f"altitudes must lie in [h0, h1] = [{self.h0}, {self.h1}].")
        return np.clip(h, self.h0, self.h1)

    # -- evaluation ---------------------------------------------------------
    def design_matrix(self, h, deriv: int = 0) -> np.ndarray:
        """B[i, k] = d^deriv/dh^deriv B_k(h_i), shape (len(h), K)."""
        h = self._check_h(h)
        B = self._spline(h, nu=deriv) if deriv else self._spline(h)
        B = np.nan_to_num(B, nan=0.0)
        return B * self._scale()[None, :]

    def integrals(self) -> np.ndarray:
        """Integral of each basis function over [h0, h1] (analytic)."""
        return self.support_widths() / (self.degree + 1) * self._scale()

    def evaluate(self, coeffs: np.ndarray, h, deriv: int = 0) -> np.ndarray:
        """sum_k coeffs[..., k] B_k(h); coeffs (..., K) -> (..., len(h))."""
        return np.asarray(coeffs) @ self.design_matrix(h, deriv).T

    # -- grid-point derivatives ----------------------------------------------
    def knot_jacobian(self, h) -> np.ndarray:
        """
        dB_k(h_i)/dx_j for the interior grid points x_1..x_{n-2}.

        Returns shape (n - 2, len(h), K), in the basis' normalization.
        Analytic, from the repeated-point divided-difference identity (see
        the module docstring).
        """
        h = self._check_h(h)
        m, K, t = self.degree, self.n_basis, self.knots
        n_int = self.n_points - 2
        J = np.zeros((n_int, h.size, K))
        for j in range(n_int):
            q = j + 1 + m                      # knot index of interior grid point j+1
            xq = t[q]
            for k in range(max(0, q - m - 1), min(K - 1, q) + 1):
                width = t[k + m + 1] - t[k]
                aug = np.sort(np.append(t[k:k + m + 2], xq))
                # Q(t) = [aug](. - t)_+^{m+1} = N_aug / (aug[-1] - aug[0]), degree m+1
                elem = BSpline.basis_element(aug, extrapolate=False)
                dQ = np.nan_to_num(elem.derivative(1)(h), nan=0.0) / (aug[-1] - aug[0])
                dP = -dQ / (m + 1)                                   # d P_k / d x_q
                elem_k = BSpline.basis_element(t[k:k + m + 2], extrapolate=False)
                P = np.nan_to_num(elem_k(h), nan=0.0) / width        # P_k
                dwidth = (1.0 if q == k + m + 1 else 0.0) - (1.0 if q == k else 0.0)
                dN = width * dP + dwidth * P
                if self.normalization == "N":
                    J[j, :, k] = dN
                elif self.normalization == "P":
                    J[j, :, k] = dP
                else:
                    J[j, :, k] = (m + 1) * dP
        return J

    def with_grid(self, grid: Sequence[float]) -> "SplineBasis":
        return SplineBasis(grid, self.degree, self.normalization)

    def __repr__(self) -> str:
        return (f"SplineBasis(n_points={self.n_points}, degree={self.degree}, "
                f"h=[{self.h0:g}, {self.h1:g}], n_basis={self.n_basis}, "
                f"normalization='{self.normalization}')")


# ===========================================================================
# Boundary-condition constrained basis
# ===========================================================================

def upper_tie_factor(grid: np.ndarray, degree: int, scale_height: float | None) -> float:
    """
    gamma in c_{K-2} = gamma * c_{K-1}, imposing y'(h1) = y(h1)/H.

    scale_height None (H = inf) -> gamma = 1, i.e. y'(h1) = 0. Raises if
    Delta = h1 - x_{n-2} >= degree * H (gamma <= 0 would flip the tie).
    """
    if scale_height is None:
        return 1.0
    if scale_height == 0 or not np.isfinite(scale_height):
        raise ValueError("scale_height must be finite and non-zero (None for H = inf).")
    delta = float(grid[-1] - grid[-2])
    gamma = 1.0 - delta / (degree * scale_height)
    if gamma <= 0:
        raise ValueError(
            f"Upper interval {delta:g} km >= degree * H = {degree * scale_height:g} km: "
            "the dNe/dh(h1)=0 tie is degenerate. Use a finer top interval.")
    return gamma


class ConstrainedSplineBasis:
    """
    ``SplineBasis`` (normalization 'N') with the plan's boundary conditions
    folded into the basis:

        y(h) = offset(h) + design_matrix(h) @ a

    lower_value : float or None
        y(h0) value (c_0 fixed). None leaves y(h0) free.
    upper_bc : 'slope' or None
        'slope' imposes y'(h1) = y(h1)/upper_scale_height (or y'(h1)=0 when
        upper_scale_height is None). None leaves the top free.
    upper_scale_height : float or None
        H in km (any sign; None = infinity). In ratio mode use the column
        mean's topside scale height so the condition equals dNe/dh(h1) = 0.
    upper_slope_offset : float
        s0 in y'(h1) = y(h1)/H + s0 (default 0).
    """

    def __init__(self, grid: Sequence[float], degree: int = 3,
                 lower_value: float | None = 0.0, upper_bc: str | None = "slope",
                 upper_scale_height: float | None = None, upper_slope_offset: float = 0.0):
        self.basis = SplineBasis(grid, degree, normalization="N")
        if upper_bc not in ("slope", None):
            raise ValueError("upper_bc must be 'slope' or None.")
        K = self.basis.n_basis
        n_fixed = (lower_value is not None) + (upper_bc is not None)
        if K - n_fixed < 1:
            raise ValueError("Too few grid points for the requested boundary conditions.")
        self.lower_value = None if lower_value is None else float(lower_value)
        self.upper_bc = upper_bc
        self.upper_scale_height = upper_scale_height
        self.upper_slope_offset = float(upper_slope_offset)
        self.gamma = (upper_tie_factor(self.basis.grid, degree, upper_scale_height)
                      if upper_bc is not None else None)
        self.T, self.c_offset = self._build_transform()

    # -- the linear map a -> c ---------------------------------------------
    def _build_transform(self) -> tuple[np.ndarray, np.ndarray]:
        K = self.basis.n_basis
        first = 1 if self.lower_value is not None else 0
        last = K - 2 if self.upper_bc is not None else K - 1   # last *independent* c index
        n_free = last - first + 1
        T = np.zeros((K, n_free))
        T[first:last + 1, :] = np.eye(n_free)
        if self.upper_bc is not None:
            # c_{K-1} tied: c_{K-2} = gamma c_{K-1}. Free coefficient is c_{K-1};
            # reorder so the last free coefficient multiplies (gamma N_{K-2} + N_{K-1}).
            T[K - 2, n_free - 1] = self.gamma
            T[K - 1, n_free - 1] = 1.0
        c_off = np.zeros(K)
        if self.lower_value is not None:
            c_off[0] = self.lower_value
        if self.upper_bc is not None and self.upper_slope_offset != 0.0:
            delta = self.basis.grid[-1] - self.basis.grid[-2]
            c_off[K - 2] += -delta * self.upper_slope_offset / self.degree
        return T, c_off

    @property
    def n_free(self) -> int:
        return self.T.shape[1]

    @property
    def grid(self) -> np.ndarray:
        return self.basis.grid

    @property
    def degree(self) -> int:
        return self.basis.degree

    def coefficients(self, a: np.ndarray) -> np.ndarray:
        """Full B-spline coefficients c = T a + c_offset; a (..., n_free) -> (..., K)."""
        return np.asarray(a) @ self.T.T + self.c_offset

    def design_matrix(self, h, deriv: int = 0) -> np.ndarray:
        """B_tilde = B T, shape (len(h), n_free)."""
        return self.basis.design_matrix(h, deriv) @ self.T

    def offset(self, h, deriv: int = 0) -> np.ndarray:
        """lower_value * N_0^{(deriv)}(h) (zeros if no lower condition)."""
        return self.basis.design_matrix(h, deriv) @ self.c_offset

    def evaluate(self, a: np.ndarray, h, deriv: int = 0) -> np.ndarray:
        return self.basis.evaluate(self.coefficients(a), h, deriv)

    def integrals(self) -> np.ndarray:
        """Integrals of the constrained columns (offset integral not included)."""
        return self.basis.integrals() @ self.T

    def knot_jacobian(self, h) -> tuple[np.ndarray, np.ndarray]:
        """
        (dB_tilde/dx_j, d offset/dx_j) for interior grid points.

        Shapes (n-2, len(h), n_free) and (n-2, len(h)). Includes the
        dependence of gamma on Delta = h1 - x_{n-2} (d gamma/d x_{n-2} = 1/(mH)).
        """
        dB = self.basis.knot_jacobian(h)                       # (n-2, nh, K)
        dBt = dB @ self.T
        doff = dB @ self.c_offset
        if self.upper_bc is not None and self.basis.n_points > 2:
            K = self.basis.n_basis
            NK2 = self.basis.design_matrix(h)[:, K - 2]
            if self.upper_scale_height is not None:
                dBt[-1, :, -1] += NK2 / (self.degree * self.upper_scale_height)
            if self.upper_slope_offset != 0.0:          # d(-Delta s0/m)/dx_{n-2} = s0/m
                doff[-1] += NK2 * self.upper_slope_offset / self.degree
        return dBt, doff

    def with_grid(self, grid: Sequence[float]) -> "ConstrainedSplineBasis":
        return ConstrainedSplineBasis(grid, self.degree, self.lower_value, self.upper_bc,
                                      self.upper_scale_height, self.upper_slope_offset)

    def __repr__(self) -> str:
        return (f"ConstrainedSplineBasis(n_points={self.basis.n_points}, degree={self.degree}, "
                f"n_free={self.n_free}, lower_value={self.lower_value}, upper_bc={self.upper_bc!r}, "
                f"H={self.upper_scale_height})")
