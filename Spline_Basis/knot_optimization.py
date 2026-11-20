# -*- coding: utf-8 -*-
"""
Optimal placement of the vertical spline grid -- Phases 3/4 of
Documentation/Spline_Basis_Knot_Optimization_Plan.md.

Offline tool (not part of the assimilation cycle): given a collection of
IRI2020 profiles, find the n-point grid h0 = x_0 < ... < x_{n-1} = h1 that
minimizes the residual of the constrained spline fit (spline_fit.py), and
write a grid file that runs load.

Parameterization of the grid (keeps the spacing threshold exact):

    gap_i = delta_i + L_free * softmax(theta)_i ,   i = 1..n-1
    delta_1 = delta_min_first_km, delta_i = delta_min_km (i >= 2)
    L_free = (h1 - h0) - sum_i delta_i

Altitude-dependent threshold: ``delta_min_km`` may be a list of
[altitude_km, delta_km] breakpoints (piecewise linear, constant beyond the
ends). The same gap map is then applied in the stretched coordinate

    s(h) = integral_{h0}^{h} dh' / delta(h')

with unit floors (first floor = s(h0 + delta_min_first_km)), and mapped back
by h = s^{-1}(s). Every gap then satisfies integral_gap dh/delta >= 1, i.e.
it is at least the local threshold (exactly delta where delta is constant
over the gap; a delta-weighted harmonic mean where it varies). The chain
rule uses dx/ds = delta(x).

The coefficients are eliminated (variable projection) and the exact gradient
dR/dx from spline_fit.BatchedSplineFit is chained through the gap map; the
optimizer is L-BFGS on theta. Several starts (uniform, curvature-density
equidistribution, random perturbations of it) are refined and the best on
the training set is kept.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from spline_basis import ConstrainedSplineBasis, SplineBasis
from spline_fit import BatchedSplineFit, ProfileTargets, residual_metrics, targets_from_files

__all__ = [
    "KnotOptimizationConfig", "KnotResult", "GapMap", "WarpedGapMap", "SpacingProfile", "optimize_knots", "optimize_from_config",
    "residual_vs_n", "density_initial_grid", "save_grid_file", "load_grid_file",
]

GRID_FILE_VERSION = 1


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class KnotOptimizationConfig:
    """
    User configuration. Fields without defaults are required.

    delta_min_km is the lower threshold for the spacing between adjacent grid
    points, enforced exactly. A number, or a list of [altitude_km, delta_km]
    breakpoints for an altitude-dependent threshold (piecewise linear,
    constant beyond the ends). delta_min_first_km (default = the threshold
    at h0) overrides it for the first interval [h0, x_1].
    """
    h0_km: float
    h1_km: float
    n_points: int
    delta_min_km: float | list                      # km, or [[altitude_km, delta_km], ...]
    fit_space: str                                  # 'linear' | 'log10' | 'pow'
    training_files: list
    output_path: str
    delta_min_first_km: float | None = None
    degree: int = 3
    representation: str = "ratio"                  # 'ratio' | 'density'
    alpha: float | None = None                      # for fit_space 'pow'
    minlog10Density: float = 4.0
    lower_bc: bool = True                           # Ne(h0) = 0
    upper_bc: bool = True                           # dNe/dh(h1) = 0
    n_top_H: int = 3
    quadrature_weights: bool = True
    altitude_weight_beta: float = 0.0
    profile_normalization: bool | None = None       # None: auto (True for density linear/pow)
    validation_files: list = field(default_factory=list)
    members_per_column: int | None = 200
    metric_members: int | None = 20                  # members per file kept for metrics (moments use all)
    geo_stride: int = 1
    init: list = field(default_factory=lambda: ["uniform", "density"])
    n_random: int = 4
    random_jitter: float = 0.3
    seed: int = 0
    max_iter: int = 300
    n_points_list: list | None = None               # for residual_vs_n

    def __post_init__(self):
        prof = self.spacing_profile()                  # validates the threshold spec
        if self.delta_min_first_km is None:
            self.delta_min_first_km = float(prof.delta(self.h0_km))
        if not self.h1_km > self.h0_km:
            raise ValueError("h1_km must be greater than h0_km.")
        if int(self.n_points) != self.n_points or self.n_points < 2:
            raise ValueError("n_points must be an integer >= 2 (includes h0 and h1).")
        if self.delta_min_first_km <= 0:
            raise ValueError("spacing thresholds must be positive.")
        if self.fit_space not in ("linear", "log10", "pow"):
            raise ValueError("fit_space must be 'linear', 'log10' or 'pow'.")
        if self.fit_space == "pow" and not (self.alpha and self.alpha > 0):
            raise ValueError("fit_space 'pow' requires alpha > 0.")
        if self.representation not in ("ratio", "density"):
            raise ValueError("representation must be 'ratio' or 'density'.")
        if self.fit_space == "log10" and self.lower_bc:
            warnings.warn("fit_space 'log10' with Ne(h0)=0: the lower value is the log10 floor "
                          "(about -6 in log ratio), which makes the spline overshoot near h0. "
                          "Consider lower_bc=False for log10.", stacklevel=2)
        for n in [self.n_points] + list(self.n_points_list or []):
            self.check_feasible(int(n))
        if self.profile_normalization is None:
            self.profile_normalization = (self.representation == "density"
                                          and self.fit_space in ("linear", "pow"))

    def spacing_profile(self) -> "SpacingProfile":
        d = self.delta_min_km
        if isinstance(d, (int, float)):
            return SpacingProfile([self.h0_km], [float(d)])
        arr = np.asarray(d, float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("delta_min_km must be a number or a list of [altitude_km, delta_km] pairs.")
        return SpacingProfile(arr[:, 0], arr[:, 1])

    def gap_map(self, n: int) -> "WarpedGapMap":
        return WarpedGapMap(self.h0_km, self.h1_km, n, self.spacing_profile(), self.delta_min_first_km)

    def floors(self, n: int) -> np.ndarray:
        """Per-gap floors in km (constant threshold only; see gap_map for the general case)."""
        if not isinstance(self.delta_min_km, (int, float)):
            raise ValueError("floors() needs a constant delta_min_km; use gap_map().")
        d = np.full(n - 1, float(self.delta_min_km))
        d[0] = self.delta_min_first_km
        return d

    def check_feasible(self, n: int) -> None:
        gm = self.gap_map(n)                 # raises if infeasible
        if gm.inner.L_free < 0.05 * (gm.inner.h1 - gm.inner.h0):
            warnings.warn("spacing threshold leaves under 5% of the (stretched) span free: "
                          "the grid is nearly forced.", stacklevel=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "KnotOptimizationConfig":
        return cls(**json.loads(Path(path).read_text()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def target_kwargs(self) -> dict:
        return dict(h0=self.h0_km, h1=self.h1_km, representation=self.representation,
                    fit_space=self.fit_space, alpha=self.alpha, minlog10Density=self.minlog10Density,
                    lower_bc=self.lower_bc, upper_bc=self.upper_bc, n_top_H=self.n_top_H,
                    altitude_weight_beta=self.altitude_weight_beta,
                    profile_normalization=self.profile_normalization,
                    quadrature_weights=self.quadrature_weights)


# ===========================================================================
# Gap parameterization
# ===========================================================================

class GapMap:
    """theta (n-1,) <-> grid (n,), with gap_i >= floors_i exactly."""

    def __init__(self, h0: float, h1: float, floors: np.ndarray):
        self.h0, self.h1 = float(h0), float(h1)
        self.floors = np.asarray(floors, float)
        self.L_free = (self.h1 - self.h0) - self.floors.sum()
        if self.L_free < 0:
            raise ValueError("infeasible spacing thresholds.")

    def _softmax(self, theta):
        e = np.exp(theta - theta.max())
        return e / e.sum()

    def grid(self, theta: np.ndarray) -> np.ndarray:
        gaps = self.floors + self.L_free * self._softmax(np.asarray(theta, float))
        g = np.concatenate([[self.h0], self.h0 + np.cumsum(gaps)])
        g[-1] = self.h1
        return g

    def theta(self, grid: np.ndarray, min_share: float = 1e-4) -> np.ndarray:
        """Inverse map; gaps below their floor are lifted (min_share of L_free)."""
        gaps = np.diff(np.asarray(grid, float))
        s = np.maximum(gaps - self.floors, min_share * max(self.L_free, 1e-12))
        s /= s.sum()
        return np.log(s)

    def chain(self, theta: np.ndarray, dR_dx: np.ndarray) -> np.ndarray:
        """dR/dtheta from dR/dx at the n-2 interior points."""
        s = self._softmax(np.asarray(theta, float))
        dR_dgap = np.cumsum(np.append(dR_dx, 0.0)[::-1])[::-1]       # gap_i moves x_j for j >= i
        return self.L_free * s * (dR_dgap - s @ dR_dgap)


class SpacingProfile:
    """
    delta(h): piecewise linear through (altitudes, deltas), constant outside.
    Provides the stretched coordinate s(h) = int_{h_ref}^{h} dh/delta and its
    inverse analytically (log/exp on linear segments).
    """

    def __init__(self, altitudes, deltas):
        a = np.atleast_1d(np.asarray(altitudes, float))
        d = np.atleast_1d(np.asarray(deltas, float))
        if a.size != d.size or a.size < 1:
            raise ValueError("altitudes and deltas must have the same non-zero length.")
        if np.any(np.diff(a) <= 0):
            raise ValueError("spacing-profile altitudes must be strictly increasing.")
        if np.any(d <= 0) or not np.all(np.isfinite(d)):
            raise ValueError("spacing thresholds must be positive.")
        # pad with constant end segments (clipped segment lookup extrapolates them
        # linearly); kept short so s stays O(span/delta) and differences stay precise
        self.a = np.concatenate([[a[0] - 1.0], a, [a[-1] + 1.0]])
        self.d = np.concatenate([[d[0]], d, [d[-1]]])
        seg_s = [self._seg_int(k, self.a[k + 1]) for k in range(self.a.size - 1)]
        self.s_nodes = np.concatenate([[0.0], np.cumsum(seg_s)])     # s at self.a

    def _slope(self, k):
        return (self.d[k + 1] - self.d[k]) / (self.a[k + 1] - self.a[k])

    def _seg_int(self, k, h):
        b = self._slope(k)
        if abs(b) < 1e-12:
            return (h - self.a[k]) / self.d[k]
        return np.log((self.d[k] + b * (h - self.a[k])) / self.d[k]) / b

    def _seg(self, h):
        return np.clip(np.searchsorted(self.a, h, side="right") - 1, 0, self.a.size - 2)

    def delta(self, h):
        h = np.asarray(h, float)
        k = self._seg(h)
        return self.d[k] + (self.d[k + 1] - self.d[k]) / (self.a[k + 1] - self.a[k]) * (h - self.a[k])

    def s(self, h):
        h = np.asarray(h, float)
        k = self._seg(h)
        out = np.empty_like(h, dtype=float)
        for kk in np.unique(k):
            m = k == kk
            out[m] = self.s_nodes[kk] + self._seg_int(kk, h[m])
        return out

    def h_of_s(self, s):
        s = np.asarray(s, float)
        k = np.clip(np.searchsorted(self.s_nodes, s, side="right") - 1, 0, self.a.size - 2)
        out = np.empty_like(s, dtype=float)
        for kk in np.unique(k):
            m = k == kk
            ds = s[m] - self.s_nodes[kk]
            b = self._slope(kk)
            if abs(b) < 1e-12:
                out[m] = self.a[kk] + ds * self.d[kk]
            else:
                out[m] = self.a[kk] + self.d[kk] * (np.exp(b * ds) - 1.0) / b
        return out

    def is_constant(self) -> bool:
        return bool(np.all(self.d == self.d[0]))


class WarpedGapMap:
    """GapMap in the stretched coordinate s(h) of a SpacingProfile (see module docstring)."""

    def __init__(self, h0: float, h1: float, n: int, profile: SpacingProfile,
                 delta_first: float | None = None):
        self.h0, self.h1, self.profile = float(h0), float(h1), profile
        s0, s1 = (float(v) for v in profile.s(np.array([h0, h1])))
        self._s0 = s0
        floors = np.ones(n - 1)
        if delta_first is not None:
            floors[0] = float(profile.s(np.array([h0 + delta_first]))[0]) - s0
        try:
            self.inner = GapMap(0.0, s1 - s0, floors)
        except ValueError:
            raise ValueError(f"infeasible spacing: {n} points do not fit in [{h0:g}, {h1:g}] km "
                             "with the requested minimum spacing.") from None

    def grid(self, theta):
        g = self.profile.h_of_s(self.inner.grid(theta) + self._s0)
        g[0], g[-1] = self.h0, self.h1
        return g

    def theta(self, grid, min_share: float = 1e-4):
        return self.inner.theta(self.profile.s(np.asarray(grid, float)) - self._s0, min_share)

    def chain(self, theta, dR_dx):
        x = self.grid(theta)[1:-1]
        return self.inner.chain(theta, np.asarray(dR_dx) * self.profile.delta(x))

    def gaps_at_threshold(self, grid, tol: float = 0.01) -> list:
        sg = np.diff(self.profile.s(np.asarray(grid, float)))
        return np.flatnonzero(sg <= self.inner.floors * (1 + tol) + 1e-12).tolist()


# ===========================================================================
# Initial grids
# ===========================================================================

def density_initial_grid(targets: ProfileTargets, n: int, degree: int,
                         spacing: float | None = None) -> np.ndarray:
    """
    Equidistribute the asymptotically optimal knot density
    rho ~ |y^{(m+1)}|^{2/(2m+3)} (rms over profiles), with y^{(m+1)} estimated
    from jumps of y^{(m)} of a fit on a dense uniform grid.
    """
    t = targets
    if spacing is None:
        spacing = 2.0 * float(np.median(np.diff(t.h)))
    n_dense = int(np.clip(round((t.h1 - t.h0) / spacing) + 1, degree + 3, 400))
    dense = np.linspace(t.h0, t.h1, n_dense)
    fitter = BatchedSplineFit(t, degree)
    c = fitter.fit(dense).coeffs                                      # (n_col, n_free, n_mem)
    # y^{(m)} is piecewise constant: evaluate it at interval midpoints
    mid = 0.5 * (dense[:-1] + dense[1:])
    B = SplineBasis(dense, degree).design_matrix(mid, deriv=degree)
    gamma, _, coff2, _ = fitter._column_terms(dense)
    A, off, _ = fitter._assemble(B, gamma, coff2)
    ym = A @ c + off[:, :, None]                                      # (n_col, n_mid, n_mem)
    jumps = np.abs(np.diff(ym, axis=1)) / spacing                     # ~ |y^{(m+1)}| at interior knots
    w = t.profile_weight[:, None, :]
    rms = np.sqrt((w * jumps ** 2).sum(axis=(0, 2)) / w.sum(axis=(0, 2)))
    rho = rms ** (2.0 / (2 * degree + 3))
    rho = np.maximum(rho, 1e-3 * rho.max())
    hk = dense[1:-1]
    # piecewise-linear density on [h0, h1]
    hh = np.concatenate([[t.h0], hk, [t.h1]])
    rr = np.concatenate([[rho[0]], rho, [rho[-1]]])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rr[1:] + rr[:-1]) * np.diff(hh))])
    targets_q = np.linspace(0.0, cum[-1], n)
    g = np.interp(targets_q, cum, hh)
    g[0], g[-1] = t.h0, t.h1
    return g


# ===========================================================================
# Optimization
# ===========================================================================

@dataclass
class KnotResult:
    grid: np.ndarray
    objective: float
    initial_grids: dict
    start_results: dict              # name -> (objective, grid)
    history: list                    # objective per iteration of the best start
    train_metrics: dict
    validation_metrics: dict | None = None
    validation_objective: float | None = None
    runtime_s: float = 0.0


def _optimize_one(fitter: BatchedSplineFit, gap: GapMap, grid0: np.ndarray, max_iter: int):
    theta0 = gap.theta(grid0)
    R0, _ = fitter.objective(gap.grid(theta0), gradient=False)
    scale = 1.0 / max(R0, 1e-300)
    last = {}

    def fun(theta):
        try:
            R, dR = fitter.objective(gap.grid(theta), gradient=True)
        except (ValueError, np.linalg.LinAlgError):
            # infeasible trial step (degenerate dNe/dh(h1)=0 tie or rank-deficient
            # fit): a large value makes the line search backtrack.
            return 1e6, np.zeros_like(theta)
        last["R"] = R
        grad = gap.chain(theta, dR) if dR is not None else np.zeros_like(theta)
        return R * scale, grad * scale

    history = [R0]
    out = minimize(fun, theta0, jac=True, method="L-BFGS-B",
                   callback=lambda th: history.append(last["R"]),
                   options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-10})
    grid = gap.grid(out.x)
    return grid, fitter.objective(grid, gradient=False)[0], history


def optimize_knots(train: ProfileTargets, n_points: int, *, degree: int = 3,
                   floors: np.ndarray | None = None, gap_map=None,
                   init: Sequence[str] = ("uniform", "density"),
                   n_random: int = 4, random_jitter: float = 0.3,
                   rng: np.random.Generator | None = None, max_iter: int = 300,
                   validation: ProfileTargets | None = None, verbose: bool = False) -> KnotResult:
    t_start = time.time()
    rng = np.random.default_rng(0) if rng is None else rng
    if gap_map is None:
        if floors is None:
            raise ValueError("give floors or gap_map.")
        gap_map = GapMap(train.h0, train.h1, floors)
    gap = gap_map
    fitter = BatchedSplineFit(train, degree)
    hm = 0.5 * (train.h[1:] + train.h[:-1])
    local_dh = np.diff(train.h)
    thr = gap.profile.delta(hm) if isinstance(gap, WarpedGapMap) else np.full(hm.size, np.min(gap.floors))
    if np.any(thr < local_dh - 1e-9):
        i = int(np.argmax(local_dh - thr))
        warnings.warn(f"spacing threshold {thr[i]:g} km at {hm[i]:g} km is below the profile sample "
                      f"spacing {local_dh[i]:g} km there; fits may become rank deficient.", stacklevel=2)

    starts = {}
    if "uniform" in init:
        starts["uniform"] = np.linspace(train.h0, train.h1, n_points)
    if "density" in init or n_random:
        dens = density_initial_grid(train, n_points, degree)
        if "density" in init:
            starts["density"] = dens
        base = gap.theta(dens)
        for i in range(n_random):
            starts[f"random{i}"] = gap.grid(base + random_jitter * rng.normal(size=base.size))
    if not starts:
        raise ValueError("no initial grids selected.")

    results, best = {}, None
    for name, g0 in starts.items():
        g0 = gap.grid(gap.theta(g0))                     # project onto the feasible set
        if n_points <= 2:
            grid, R, hist = g0, fitter.objective(g0, gradient=False)[0], []
        else:
            grid, R, hist = _optimize_one(fitter, gap, g0, max_iter)
        results[name] = (R, grid)
        if verbose:
            print(f"  start {name:8s}: R0={hist[0] if hist else R:.6g} -> R={R:.6g}  ({len(hist)} it)")
        if best is None or R < best[0]:
            best = (R, grid, hist)

    R, grid, hist = best
    train_fit = fitter.fit(grid)
    out = KnotResult(grid=grid, objective=R, initial_grids=starts, start_results=results,
                     history=hist, train_metrics=residual_metrics(train, train_fit.y_fit))
    if validation is not None:
        vfitter = BatchedSplineFit(validation, degree)
        out.validation_metrics = residual_metrics(validation, vfitter.fit(grid).y_fit)
        out.validation_objective = vfitter.objective(grid, gradient=False)[0]
    out.runtime_s = time.time() - t_start
    return out


def _load_targets(cfg: KnotOptimizationConfig, files, rng):
    if not files:
        return None
    return targets_from_files(files, members_per_column=cfg.members_per_column,
                              geo_stride=cfg.geo_stride, rng=rng, metric_members=cfg.metric_members,
                              **cfg.target_kwargs())


def optimize_from_config(cfg: KnotOptimizationConfig, n_points: int | None = None,
                         train: ProfileTargets | None = None,
                         validation: ProfileTargets | None = None,
                         save: bool = True, verbose: bool = False) -> KnotResult:
    """Load the collections named in ``cfg``, optimize, and (optionally) write the grid file."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_points if n_points is None else int(n_points)
    train = _load_targets(cfg, cfg.training_files, rng) if train is None else train
    if validation is None and cfg.validation_files:
        validation = _load_targets(cfg, cfg.validation_files, rng)
    res = optimize_knots(train, n, degree=cfg.degree, gap_map=cfg.gap_map(n), init=cfg.init,
                         n_random=cfg.n_random, random_jitter=cfg.random_jitter, rng=rng,
                         max_iter=cfg.max_iter, validation=validation, verbose=verbose)
    if save:
        save_grid_file(cfg.output_path, res, cfg, train)
    return res


def residual_vs_n(cfg: KnotOptimizationConfig, n_list: Sequence[int] | None = None,
                  verbose: bool = False) -> list[dict]:
    """Optimized residual (train and validation) for each n -- the model-selection curve."""
    n_list = cfg.n_points_list if n_list is None else n_list
    rng = np.random.default_rng(cfg.seed)
    train = _load_targets(cfg, cfg.training_files, rng)
    validation = _load_targets(cfg, cfg.validation_files, rng)
    rows = []
    for n in n_list:
        cfg.check_feasible(int(n))
        r = optimize_from_config(cfg, n, train=train, validation=validation, save=False, verbose=verbose)
        g_uni = np.linspace(cfg.h0_km, cfg.h1_km, n)
        ftr = BatchedSplineFit(train, cfg.degree)
        uni = ftr.fit(g_uni)
        # R values from objective(): all members (moments); rms metrics: metric-member subset
        row = {"n": int(n), "grid": r.grid.tolist(), "train_R": r.objective,
               "train_R_uniform": ftr.objective(g_uni, gradient=False)[0],
               "train_rms_ratio": r.train_metrics["all"]["rms_ratio"],
               "train_rms_ratio_uniform": residual_metrics(train, uni.y_fit)["all"]["rms_ratio"]}
        if validation is not None:
            fva = BatchedSplineFit(validation, cfg.degree)
            vu = fva.fit(g_uni)
            row.update(val_R=r.validation_objective,
                       val_R_uniform=fva.objective(g_uni, gradient=False)[0],
                       val_rms_ratio=r.validation_metrics["all"]["rms_ratio"],
                       val_rms_ratio_uniform=residual_metrics(validation, vu.y_fit)["all"]["rms_ratio"])
        rows.append(row)
        if verbose:
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != "grid"})
    return rows


# ===========================================================================
# Grid file (the only thing regular runs load)
# ===========================================================================

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_grid_file(path: str | Path, result: KnotResult, cfg: KnotOptimizationConfig,
                   train: ProfileTargets | None = None) -> None:
    gm = cfg.gap_map(result.grid.size)
    payload = {
        "grid_file_version": GRID_FILE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid_km": result.grid,
        "degree": cfg.degree,
        "representation": cfg.representation,
        "fit_space": cfg.fit_space,
        "alpha": cfg.alpha,
        "minlog10Density": cfg.minlog10Density,
        "boundary_conditions": {"lower": "Ne(h0)=0" if cfg.lower_bc else "free",
                                "upper": "dNe/dh(h1)=0" if cfg.upper_bc else "free",
                                "reference_mean": "column" if cfg.representation == "ratio" else None,
                                "n_top_H": cfg.n_top_H},
        "spacing": {"delta_min_km": cfg.delta_min_km, "delta_min_first_km": cfg.delta_min_first_km,
                    "threshold_at_grid_km": gm.profile.delta(result.grid),
                    "gaps_at_threshold": gm.gaps_at_threshold(result.grid)},
        "train_objective": result.objective,
        "train_metrics": result.train_metrics,
        "validation_objective": result.validation_objective,
        "validation_metrics": result.validation_metrics,
        "start_results": {k: {"objective": v[0], "grid_km": v[1]} for k, v in result.start_results.items()},
        "training_summary": None if train is None else {
            "n_columns": train.n_col, "members_per_column": train.n_mem,
            "altitude_samples_km": [float(train.h[0]), float(train.h[-1]), float(np.median(np.diff(train.h)))]},
        "config": asdict(cfg),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2))


def load_grid_file(path: str | Path) -> dict:
    """Read a grid file written by ``save_grid_file`` (grid_km as an array)."""
    d = json.loads(Path(path).read_text())
    if d.get("grid_file_version") != GRID_FILE_VERSION:
        raise ValueError(f"unsupported grid file version {d.get('grid_file_version')}")
    d["grid_km"] = np.asarray(d["grid_km"], float)
    return d


def basis_from_grid_file(path: str | Path, upper_scale_height: float | None = None,
                         lower_value: float | None = 0.0,
                         upper_slope_offset: float = 0.0) -> ConstrainedSplineBasis:
    """
    Constrained basis for one column from a grid file. In ratio mode pass the
    column's effective scale height / slope offset (from its background mean),
    exactly as spline_fit.prepare_targets derives them.
    """
    d = load_grid_file(path)
    return ConstrainedSplineBasis(
        d["grid_km"], d["degree"],
        lower_value=lower_value if d["boundary_conditions"]["lower"] != "free" else None,
        upper_bc="slope" if d["boundary_conditions"]["upper"] != "free" else None,
        upper_scale_height=upper_scale_height, upper_slope_offset=upper_slope_offset)
