# -*- coding: utf-8 -*-
"""
Constrained, batched spline fits of EDP collections -- Phase 2 of
Documentation/Spline_Basis_Knot_Optimization_Plan.md.

Representation
--------------
* ``representation='ratio'`` (default): Ne = mean(h) * g(y(h)), where ``mean``
  is the reference mean EDP of the profile's *column* (geo vertex; mean over
  that vertex's ensemble members) and y = sum_k a_k B_tilde_k is the spline.
* ``representation='density'``: Ne = g(y(h)) (mean == 1).

Fit space g (configuration ``fit_space``):

    linear : g(y) = y              y = q
    pow    : g(y) = |y|^alpha      y = q^(1/alpha)
    log10  : g(y) = 10^y           y = log10(max(Ne, 10^minlog10Density)) - log10(mean)

with q = Ne / mean.

Boundary conditions (user decisions, plan sections 1.3a / 5a)
--------------------------------------------------------------
* Ne(h0) = 0           -> y(h0) = g^{-1}(0): 0 (linear, pow) or the log10
                           floor minus log10 mean(h0) (log10).
* dNe/dh(h1) = 0       -> y'(h1) = g(y)/(g'(y) H) with H = -mean/mean' at h1:
                           linear: y' = y/H, pow: y' = y/(alpha H),
                           log10: y' = 1/(ln10 H). Density mode: y'(h1) = 0.

Both are linear (affine) in the coefficients, so every column is an ordinary
weighted least-squares problem. All columns share the unconstrained B-spline
design matrix; only the tied last column and the offset differ per column
(through gamma_c, the lower value and the slope offset), which lets the fit
and its grid gradient be computed in a few batched array operations.

Objective
---------
    R(grid) = sum_c sum_j pw_cj || sqrt(w_c) * (y_cj - off_c - A_c a_cj) ||^2

with a_cj the least-squares coefficients (variable projection). Because the
weighted residual is orthogonal to range(A_c), the exact gradient w.r.t. the
interior grid points is

    dR/dx_q = -2 sum_c sum_j pw_cj rw_cj^T sqrt(w_c) (dA_c/dx_q a_cj + doff_c/dx_q).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from spline_basis import SplineBasis

__all__ = [
    "Representation", "FitSpace", "ProfileTargets", "BatchedSplineFit", "FitResult",
    "prepare_targets", "concatenate_targets", "compute_moments", "topside_scale_height",
    "load_edp_samples", "targets_from_files", "residual_metrics", "DEFAULT_BANDS",
]

Representation = Literal["ratio", "density"]
FitSpace = Literal["linear", "log10", "pow"]
DEFAULT_BANDS = {"E": (0.0, 150.0), "F1": (150.0, 220.0), "F2": (220.0, 450.0),
                 "topside": (450.0, np.inf)}


# ===========================================================================
# Fit-space transforms
# ===========================================================================

def _to_fit_space(ne, mean, fit_space, alpha, minlog10Density):
    if fit_space == "log10":
        floor = 10.0 ** minlog10Density
        return np.log10(np.maximum(ne, floor)) - np.log10(mean)
    q = ne / mean
    if fit_space == "linear":
        return q
    if fit_space == "pow":
        return np.clip(q, 0.0, None) ** (1.0 / alpha)
    raise ValueError(f"unknown fit_space {fit_space!r}")


def _from_fit_space(y, mean, fit_space, alpha):
    if fit_space == "linear":
        return mean * y
    if fit_space == "pow":
        return mean * np.abs(y) ** alpha
    if fit_space == "log10":
        return mean * 10.0 ** y
    raise ValueError(f"unknown fit_space {fit_space!r}")


def topside_scale_height(mean: np.ndarray, h: np.ndarray, n_top: int = 3) -> np.ndarray:
    """
    H = -mean/mean' at the top, from a log-linear fit over the top ``n_top``
    samples of each column. mean (n_col, n_h) -> H (n_col,); inf if flat.
    """
    if n_top < 2:
        raise ValueError("n_top must be >= 2.")
    hh = h[-n_top:]
    lm = np.log(mean[:, -n_top:])
    hc = hh - hh.mean()
    slope = (lm - lm.mean(axis=1, keepdims=True)) @ hc / (hc @ hc)
    with np.errstate(divide="ignore"):
        H = np.where(slope == 0.0, np.inf, -1.0 / slope)
    return H


# ===========================================================================
# Targets
# ===========================================================================

@dataclass
class ProfileTargets:
    """Transformed profiles plus the per-column boundary-condition data."""
    h: np.ndarray                      # (n_h,) sample altitudes inside [h0, h1]
    h0: float
    h1: float
    y: np.ndarray                      # (n_col, n_h, n_mem) transformed targets
    ne: np.ndarray                     # (n_col, n_h, n_mem) original Ne
    mean: np.ndarray                   # (n_col, n_h) reference mean (1 in density mode)
    lower_value: np.ndarray | None     # (n_col,) y(h0); None = free
    upper_bc: bool                     # impose the dNe/dh(h1) = 0 condition
    H_eff: np.ndarray                  # (n_col,) effective scale height (inf allowed)
    s0: np.ndarray                     # (n_col,) slope offset
    alt_weight: np.ndarray             # (n_col, n_h)
    profile_weight: np.ndarray         # (n_col, n_mem)
    representation: str
    fit_space: str
    alpha: float | None
    minlog10Density: float
    scale_height: np.ndarray           # (n_col,) H of the mean (diagnostic)
    column_source: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # (file, geo)
    # Optional precomputed member moments (S0, S1, S2) over MORE members than
    # y/ne hold: then the objective uses all members, while y/ne are a subset
    # kept for metrics (see compute_moments / targets_from_files).
    moments: tuple | None = None

    @property
    def n_col(self) -> int:
        return self.y.shape[0]

    @property
    def n_mem(self) -> int:
        return self.y.shape[2]

    def subset_columns(self, idx) -> "ProfileTargets":
        idx = np.asarray(idx)
        kw = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k in ("y", "ne", "mean", "H_eff", "s0", "alt_weight", "profile_weight",
                  "scale_height", "column_source"):
            kw[k] = getattr(self, k)[idx]
        if self.lower_value is not None:
            kw["lower_value"] = self.lower_value[idx]
        if self.moments is not None:
            kw["moments"] = tuple(m[idx] for m in self.moments)
        return ProfileTargets(**kw)

    def subset_members(self, n: int, rng: np.random.Generator | None = None) -> "ProfileTargets":
        """Keep ``n`` members for y/ne (metrics); moments (computed first if absent) keep all."""
        if n >= self.n_mem:
            return self
        rng = np.random.default_rng(0) if rng is None else rng
        idx = np.sort(rng.choice(self.n_mem, n, replace=False))
        kw = {k: getattr(self, k) for k in self.__dataclass_fields__}
        kw["moments"] = self.moments if self.moments is not None else compute_moments(self)
        kw["y"], kw["ne"] = self.y[:, :, idx], self.ne[:, :, idx]
        kw["profile_weight"] = self.profile_weight[:, idx]
        return ProfileTargets(**kw)


def prepare_targets(ne: np.ndarray, h: np.ndarray, *, h0: float, h1: float,
                    representation: Representation = "ratio", fit_space: FitSpace = "linear",
                    alpha: float | None = None, minlog10Density: float = 4.0,
                    mean: np.ndarray | None = None, lower_bc: bool = True, upper_bc: bool = True,
                    n_top_H: int = 3, altitude_weight_beta: float = 0.0,
                    profile_normalization: bool = False,
                    quadrature_weights: bool = True) -> ProfileTargets:
    """
    Build fit targets from Ne of shape (n_h, n_col, n_mem) (EDPSamples
    layout: height, geo, sample).

    ``mean`` (n_h, n_col) is the reference mean; default = mean over the
    given members. Pass the full-ensemble mean when ``ne`` is a member
    subsample. Only samples with h0 <= h <= h1 are used; the data must cover
    [h0, h1]. The sample at h = h0 (if any) gets zero weight when the lower
    condition is imposed, since its residual is fixed by the condition.

    quadrature_weights: weight each sample by its trapezoid interval width
    (normalized by the median sample spacing), so the objective approximates
    the integral of the squared error over altitude and non-uniformly sampled
    profiles (e.g. 5 km below 200 km, 10 km above) do not over-weight the
    densely sampled region.
    """
    h = np.asarray(h, dtype=float)
    ne = np.asarray(ne, dtype=float)
    if ne.ndim != 3 or ne.shape[0] != h.size:
        raise ValueError("ne must have shape (n_h, n_col, n_mem) matching h.")
    if fit_space == "pow" and not (alpha and alpha > 0):
        raise ValueError("fit_space 'pow' needs alpha > 0.")
    if representation not in ("ratio", "density"):
        raise ValueError("representation must be 'ratio' or 'density'.")
    tol = 1e-6
    if h.min() > h0 + tol or h.max() < h1 - tol:
        raise ValueError(f"profile altitudes [{h.min():g}, {h.max():g}] do not cover [h0, h1] = [{h0:g}, {h1:g}].")
    if mean is None:
        mean = ne.mean(axis=2)
    keep = (h >= h0 - tol) & (h <= h1 + tol)
    h, ne, mean = h[keep], ne[keep], np.asarray(mean, float)[keep]
    ne = np.moveaxis(ne, 1, 0)                      # (n_col, n_h, n_mem)
    mean = mean.T                                   # (n_col, n_h)
    if representation == "ratio":
        floor = 10.0 ** minlog10Density
        if np.any(mean <= 0):
            mean = np.maximum(mean, floor)
        H = topside_scale_height(mean, h, n_top_H)
    else:
        mean = np.ones_like(mean)
        H = np.full(mean.shape[0], np.inf)

    y = _to_fit_space(ne, mean[:, :, None], fit_space, alpha, minlog10Density)
    n_col = y.shape[0]

    # lower value: y(h0) = g^{-1}(0)
    lower = None
    if lower_bc:
        if fit_space == "log10":
            logmean_h0 = np.array([np.interp(h0, h, np.log10(mc)) for mc in mean])
            lower = minlog10Density - logmean_h0
        else:
            lower = np.zeros(n_col)

    # upper: y'(h1) = y/H_eff + s0
    H_eff = np.full(n_col, np.inf)
    s0 = np.zeros(n_col)
    if upper_bc and representation == "ratio":
        if fit_space == "linear":
            H_eff = H.copy()
        elif fit_space == "pow":
            H_eff = alpha * H
        else:
            s0 = np.where(np.isfinite(H), 1.0 / (np.log(10.0) * H), 0.0)

    alt_w = np.ones((n_col, h.size))
    if altitude_weight_beta and representation == "ratio":
        alt_w = (mean / mean.max(axis=1, keepdims=True)) ** altitude_weight_beta
    if quadrature_weights and h.size > 1:
        dh = np.diff(h)
        q = np.zeros(h.size)
        q[:-1] += 0.5 * dh
        q[1:] += 0.5 * dh
        alt_w = alt_w * (q / np.median(dh))[None, :]
    if lower_bc:
        alt_w[:, np.abs(h - h0) < tol] = 0.0

    if profile_normalization:
        scale = np.abs(y).max(axis=1)                  # (n_col, n_mem)
        pw = 1.0 / np.maximum(scale, 1e-300) ** 2
    else:
        pw = np.ones((n_col, y.shape[2]))

    return ProfileTargets(h=h, h0=float(h0), h1=float(h1), y=y, ne=ne, mean=mean,
                          lower_value=lower, upper_bc=bool(upper_bc), H_eff=H_eff, s0=s0,
                          alt_weight=alt_w, profile_weight=pw, representation=representation,
                          fit_space=fit_space, alpha=alpha, minlog10Density=minlog10Density,
                          scale_height=H,
                          column_source=np.column_stack([np.zeros(n_col, int), np.arange(n_col)]))


def compute_moments(t: ProfileTargets) -> tuple:
    """Per-column weighted member moments S0 (n_col,), S1 (n_col, n_h), S2 (n_col, n_h, n_h)."""
    pw = t.profile_weight[:, None, :]
    S0 = t.profile_weight.sum(axis=1)
    S1 = (pw * t.y).sum(axis=2)
    S2 = (pw * t.y) @ t.y.transpose(0, 2, 1)
    return S0, S1, S2


def concatenate_targets(parts: Sequence[ProfileTargets]) -> ProfileTargets:
    """Pool several target sets (same h, settings, member count) along columns."""
    p0 = parts[0]
    for p in parts[1:]:
        if not np.allclose(p.h, p0.h) or p.n_mem != p0.n_mem:
            raise ValueError("all parts need identical altitudes and member counts.")
    cat = lambda k: np.concatenate([getattr(p, k) for p in parts], axis=0)
    src = np.concatenate([np.column_stack([np.full(p.n_col, i), p.column_source[:, 1]])
                          for i, p in enumerate(parts)])
    return ProfileTargets(
        h=p0.h, h0=p0.h0, h1=p0.h1, y=cat("y"), ne=cat("ne"), mean=cat("mean"),
        lower_value=None if p0.lower_value is None else cat("lower_value"),
        upper_bc=p0.upper_bc, H_eff=cat("H_eff"), s0=cat("s0"), alt_weight=cat("alt_weight"),
        profile_weight=cat("profile_weight"), representation=p0.representation,
        fit_space=p0.fit_space, alpha=p0.alpha, minlog10Density=p0.minlog10Density,
        scale_height=cat("scale_height"), column_source=src,
        moments=None if any(p.moments is None for p in parts) else tuple(
            np.concatenate([p.moments[i] for p in parts], axis=0) for i in range(3)))


# ===========================================================================
# Batched constrained fit and its grid gradient
# ===========================================================================

@dataclass
class FitResult:
    grid: np.ndarray
    coeffs: np.ndarray            # (n_col, n_free, n_mem)
    y_fit: np.ndarray             # (n_col, n_h, n_mem)
    objective: float              # R
    per_profile: np.ndarray       # (n_col, n_mem) weighted squared residual
    gradient: np.ndarray | None   # dR/dx for interior grid points


class BatchedSplineFit:
    """Fits every column of ``targets`` with the constrained spline on ``grid``."""

    def __init__(self, targets: ProfileTargets, degree: int = 3):
        self.t = targets
        self.degree = int(degree)
        self._moments = None

    # -- member moments (objective() depends on the data only through these) --
    def _get_moments(self):
        if self._moments is None:
            self._moments = self.t.moments if self.t.moments is not None else compute_moments(self.t)
        return self._moments

    # -- per-column constraint data -----------------------------------------
    def _column_terms(self, grid):
        m, t = self.degree, self.t
        delta = grid[-1] - grid[-2]
        with np.errstate(divide="ignore"):
            inv_H = np.where(np.isfinite(t.H_eff), 1.0 / t.H_eff, 0.0)
        gamma = 1.0 - delta * inv_H / m
        if t.upper_bc and np.any(gamma <= 0):
            raise ValueError(
                f"top interval {delta:g} km >= degree * H_eff for some columns; "
                "the dNe/dh(h1) = 0 tie is degenerate.")
        coff2 = -delta * t.s0 / m
        return gamma, inv_H / m, coff2, t.s0 / m       # gamma, dgamma, coff2, dcoff2

    def _assemble(self, B, gamma, coff2):
        """Per-column design (n_col, n_h, n_free) and offset (n_col, n_h)."""
        t, K = self.t, B.shape[1]
        first = 1 if t.lower_value is not None else 0
        if t.upper_bc:
            shared = B[:, first:K - 2]
            last = gamma[:, None] * B[None, :, K - 2] + B[None, :, K - 1]
            A = np.concatenate([np.broadcast_to(shared, (t.n_col,) + shared.shape),
                                last[:, :, None]], axis=2)
        else:
            A = np.broadcast_to(B[:, first:], (t.n_col, B.shape[0], K - first))
        off = np.zeros((t.n_col, B.shape[0]))
        if t.lower_value is not None:
            off += t.lower_value[:, None] * B[None, :, 0]
        if t.upper_bc:
            off += coff2[:, None] * B[None, :, K - 2]
        return A, off, first

    def fit(self, grid: Sequence[float], gradient: bool = False) -> FitResult:
        grid = np.asarray(grid, dtype=float)
        if abs(grid[0] - self.t.h0) > 1e-9 or abs(grid[-1] - self.t.h1) > 1e-9:
            raise ValueError("grid must start at h0 and end at h1 of the targets.")
        t = self.t
        basis = SplineBasis(grid, self.degree)
        B = basis.design_matrix(t.h)
        gamma, dgamma, coff2, dcoff2 = self._column_terms(grid)
        A, off, first = self._assemble(B, gamma, coff2)

        sw = np.sqrt(t.alt_weight)                                   # (n_col, n_h)
        Aw = sw[:, :, None] * A
        bw = sw[:, :, None] * (t.y - off[:, :, None])
        Q, R = self._qr(Aw)
        a = np.linalg.solve(R, Q.transpose(0, 2, 1) @ bw)             # (n_col, n_free, n_mem)
        y_fit = A @ a + off[:, :, None]
        rw = bw - Aw @ a
        per_profile = t.profile_weight * np.sum(rw ** 2, axis=1)
        R_tot = float(per_profile.sum())

        grad = None
        if gradient and grid.size > 2:
            Z = sw[:, :, None] * rw * t.profile_weight[:, None, :]    # (n_col, n_h, n_mem)
            G = Z @ a.transpose(0, 2, 1)                              # (n_col, n_h, n_free)
            S = Z.sum(axis=2)                                         # (n_col, n_h)
            grad = self._gradient(basis, B, G, S, gamma, dgamma, first)
        return FitResult(grid=grid, coeffs=a, y_fit=y_fit, objective=R_tot,
                         per_profile=per_profile, gradient=grad)

    @staticmethod
    def _qr(Aw):
        Q, R = np.linalg.qr(Aw)
        diag = np.abs(np.diagonal(R, axis1=1, axis2=2))
        if np.any(diag <= 1e-10 * diag.max()):
            raise np.linalg.LinAlgError(
                "rank-deficient spline fit: some basis function has (almost) no weighted "
                "samples in its support (Schoenberg-Whitney). Increase delta_min or the "
                "profile resolution.")
        return Q, R

    def objective(self, grid: Sequence[float], gradient: bool = True):
        """
        (R, dR/dx) from the per-column member moments, without forming
        coefficients or fitted profiles: cost independent of the member count.

        With D = diag(sqrt(w)), o the offset and M = sum_j pw_j (y_j - o)(y_j - o)^T,
        Mb = D M D, Aw = D A = QR:   R_c = tr(Mb) - tr(Q^T Mb Q),
        G_c = D (I - QQ^T) Mb Q R^{-T},  S_c = D (I - QQ^T) D (S1 - S0 o).
        """
        grid = np.asarray(grid, dtype=float)
        t = self.t
        S0, S1, S2 = self._get_moments()
        basis = SplineBasis(grid, self.degree)
        B = basis.design_matrix(t.h)
        gamma, dgamma, coff2, _ = self._column_terms(grid)
        A, off, first = self._assemble(B, gamma, coff2)
        sw = np.sqrt(t.alt_weight)
        Aw = sw[:, :, None] * A
        Q, R = self._qr(Aw)
        S1o = S1[:, :, None] * off[:, None, :]
        M = S2 - S1o - S1o.transpose(0, 2, 1) + S0[:, None, None] * off[:, :, None] * off[:, None, :]
        Mb = sw[:, :, None] * M * sw[:, None, :]
        X = Mb @ Q                                                    # (n_col, n_h, n_free)
        QtX = Q.transpose(0, 2, 1) @ X
        R_tot = float(np.trace(Mb, axis1=1, axis2=2).sum() - np.trace(QtX, axis1=1, axis2=2).sum())
        if not gradient or grid.size <= 2:
            return R_tot, None
        PX = X - Q @ QtX                                              # (I - P) Mb Q
        G = np.linalg.solve(R, (sw[:, :, None] * PX).transpose(0, 2, 1)).transpose(0, 2, 1)
        v = sw * (S1 - S0[:, None] * off)                             # D (S1 - S0 o)
        Pv = v - (Q @ (Q.transpose(0, 2, 1) @ v[:, :, None]))[:, :, 0]
        S = sw * Pv
        return R_tot, self._gradient(basis, B, G, S, gamma, dgamma, first)

    def _gradient(self, basis, B, G, S, gamma, dgamma, first):
        t, K = self.t, B.shape[1]
        dB = basis.knot_jacobian(t.h)                                 # (n_int, n_h, K)
        dcoff2 = t.s0 / self.degree
        n_free = G.shape[2]
        if t.upper_bc:
            n_shared = n_free - 1
            Gl = G[:, :, -1]
            g = np.einsum("jhk,hk->j", dB[:, :, first:first + n_shared], G[:, :, :n_shared].sum(0))
            g += dB[:, :, K - 2] @ (gamma[:, None] * Gl).sum(0) + dB[:, :, K - 1] @ Gl.sum(0)
        else:
            g = np.einsum("jhk,hk->j", dB[:, :, first:], G.sum(0))
        if t.lower_value is not None:
            g += dB[:, :, 0] @ (t.lower_value[:, None] * S).sum(0)
        if t.upper_bc:
            coff2_cols = -(basis.grid[-1] - basis.grid[-2]) * t.s0 / self.degree
            g += dB[:, :, K - 2] @ (coff2_cols[:, None] * S).sum(0)
            g[-1] += B[:, K - 2] @ ((dgamma[:, None] * G[:, :, -1]).sum(0)
                                    + (dcoff2[:, None] * S).sum(0))
        return -2.0 * g


# ===========================================================================
# Diagnostics
# ===========================================================================

def residual_metrics(targets: ProfileTargets, y_fit: np.ndarray,
                     bands: dict | None = None) -> dict:
    """
    Error summary of a fit, by altitude band, in ratio, relative-Ne, and
    log10-Ne terms, plus NmF2 / hmF2 / vertical-TEC errors (on the sample grid).
    The h0 sample (pinned by the lower condition) is excluded from the band
    statistics and reported separately.
    """
    t = targets
    bands = DEFAULT_BANDS if bands is None else bands
    mean = t.mean[:, :, None]
    ne_fit = _from_fit_space(y_fit, mean, t.fit_space, t.alpha)
    floor = 10.0 ** t.minlog10Density
    ratio_err = (ne_fit - t.ne) / mean
    rel_err = (ne_fit - t.ne) / np.maximum(t.ne, floor)
    log_err = np.log10(np.maximum(ne_fit, floor)) - np.log10(np.maximum(t.ne, floor))
    at_h0 = np.abs(t.h - t.h0) < 1e-6
    out = {"bands": {}}
    for name, (lo, hi) in bands.items():
        sel = (t.h >= lo) & (t.h < hi) & ~at_h0
        if not sel.any():
            continue
        out["bands"][name] = {
            "rms_ratio": float(np.sqrt(np.mean(ratio_err[:, sel] ** 2))),
            "rms_rel_ne": float(np.sqrt(np.mean(rel_err[:, sel] ** 2))),
            "rms_log10_ne": float(np.sqrt(np.mean(log_err[:, sel] ** 2))),
        }
    sel = ~at_h0
    out["all"] = {
        "rms_ratio": float(np.sqrt(np.mean(ratio_err[:, sel] ** 2))),
        "rms_rel_ne": float(np.sqrt(np.mean(rel_err[:, sel] ** 2))),
        "rms_log10_ne": float(np.sqrt(np.mean(log_err[:, sel] ** 2))),
    }
    if at_h0.any():
        out["h0_sample_rms_ratio"] = float(np.sqrt(np.mean(ratio_err[:, at_h0] ** 2)))
    ipk_true = t.ne.argmax(axis=1)
    ipk_fit = ne_fit.argmax(axis=1)
    nm_true = t.ne.max(axis=1)
    nm_fit = np.take_along_axis(ne_fit, ipk_true[:, None, :], axis=1)[:, 0, :]
    out["NmF2_rel_rms"] = float(np.sqrt(np.mean(((nm_fit - nm_true) / nm_true) ** 2)))
    out["hmF2_abs_mean_km"] = float(np.mean(np.abs(t.h[ipk_fit] - t.h[ipk_true])))
    tec_true = np.trapezoid(t.ne, t.h, axis=1)
    tec_fit = np.trapezoid(ne_fit, t.h, axis=1)
    out["vTEC_rel_rms"] = float(np.sqrt(np.mean(((tec_fit - tec_true) / tec_true) ** 2)))
    return out


# ===========================================================================
# Loading EDPSamples files (e.g. the IRI2020 prior edp_samples.nc)
# ===========================================================================

def load_edp_samples(path: str | Path, members_per_column: int | None = None,
                     geo_indices: Sequence[int] | None = None,
                     rng: np.random.Generator | None = None):
    """
    Read ``EDPs`` (height, geo_vertex, sample) and ``altitude`` from an
    EDPSamples NetCDF file.

    Returns (h, ne_subsample (n_h, n_geo, n_sub), full_mean (n_h, n_geo)); the
    mean is always over *all* members (the run's background mean), the member
    subsample (without replacement, ``rng``) only reduces the training size.
    """
    import xarray as xr
    with xr.open_dataset(path) as ds:
        da = ds["EDPs"].transpose("height", "geo_vertex", "sample")
        if geo_indices is not None:
            da = da.isel(geo_vertex=np.asarray(geo_indices))
        ne = da.values.astype(float)
        h = ds["altitude"].values.astype(float)
    mean = ne.mean(axis=2)
    if members_per_column is not None and members_per_column < ne.shape[2]:
        rng = np.random.default_rng() if rng is None else rng
        idx = np.sort(rng.choice(ne.shape[2], members_per_column, replace=False))
        ne = ne[:, :, idx]
    return h, ne, mean


def targets_from_files(paths: Sequence[str | Path], *, members_per_column: int | None = None,
                       geo_stride: int = 1, rng: np.random.Generator | None = None,
                       metric_members: int | None = None,
                       **prepare_kwargs) -> ProfileTargets:
    """
    Load several edp_samples files and pool their columns into one target set.

    metric_members: if given, each file's moments are computed over all loaded
    members and only ``metric_members`` members are kept in y/ne (for
    metrics), so pooling many files stays light in memory while the
    objective still uses every loaded member.
    """
    parts = []
    for p in paths:
        with __import__("xarray").open_dataset(p) as ds:
            n_geo = ds.sizes["geo_vertex"]
        geo = np.arange(0, n_geo, geo_stride)
        h, ne, mean = load_edp_samples(p, members_per_column, geo, rng)
        part = prepare_targets(ne, h, mean=mean, **prepare_kwargs)
        if metric_members is not None:
            part = part.subset_members(metric_members, rng)
        parts.append(part)
    return parts[0] if len(parts) == 1 else concatenate_targets(parts)
