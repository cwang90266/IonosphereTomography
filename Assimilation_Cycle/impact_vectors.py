#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-RO impact vectors (``Impact_Vector_Plan.md``): for each RO/IGS entry
that enters an assimilation batch, how much of the analysis EDP change at
an arbitrary query point's vertical column is attributable to that
entry's own observations.

**Scope confirmed with the user (plan Section 10, Decisions 1/2/3/4)**:
this computes the decomposition of a ``centering="mean"`` *diagnostic*
pass (an approximate answer to "which RO mattered most," not a
reconstruction of whatever centering the production run actually used --
see the plan's Finding 4 for why ``centering="mean"`` is needed for an
exact decomposition), as an EDP-vs-altitude profile per RO (not reduced
to scalar ionospheric parameters), on the model's own altitude grid by
default, scoped to one batch at a time.

**Math** (plan Section 3): for GN iteration/step ``k`` (``k=0`` only for
the ``raw``/linear style), query point ``point``:

    A^k(point) = compose_observation_jacobian(F_EDP(point), jac_array_k, ...)
    G^k(point) = A^k(point) @ K^k
    contrib_ℓ^k(point) = G^k(point)[:, idx_ℓ] @ r^k[idx_ℓ]
    I_ℓ(point) = Σ_k w_k · contrib_ℓ^k(point)

where ``idx_ℓ`` is RO ``ℓ``'s row-range into this batch's observations
(``CycleBatch.entry_ray_ranges``, already tracked), and ``w_k`` is the
decay weight from the plan's Finding 3 (``w_k = alpha_k · Π_{m>k}
(1-alpha_m)``) -- NOT knowable until every iteration's accepted ``alpha``
is known, so :class:`ImpactVectorAccumulator` accumulates the *unweighted*
``contrib_ℓ^k`` per iteration during the run (Section 6's ``on_iteration``
hook) and applies the decay weights only in :meth:`ImpactVectorAccumulator.finalize`,
once ``AnalysisDiagnostics.alpha_history`` is available.

``F_EDP(point)`` -- the interpolation-weight operator reading off a
density value at one query point, for every altitude in
``QueryPoint.altitudes`` -- is built once per query point and reused
across every iteration (never once per RO), following the same
mesh-lookup-is-altitude-independent shortcut
``output._interpolate_field_at_latlon`` already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from Ensemble_Kalman_Engine.jacobian_utils import compose_observation_jacobian


@dataclass(frozen=True)
class QueryPoint:
    """One vertical-column query point (plan Decision 6: generic, not
    ISR-specific -- an ISR station is just one way to build one, via
    :func:`query_point_from_isr_station`)."""

    name: str
    lat: float
    lon: float
    altitudes: np.ndarray


def query_point_from_isr_station(station, edp_samples) -> QueryPoint:
    """Convenience constructor: an ISR station's own (name, lat, lon),
    altitudes defaulted to the model grid (plan Decision 3), not the ISR
    instrument's native (sparser) gates."""
    return QueryPoint(
        name=station.name, lat=float(station.lat), lon=float(station.lon),
        altitudes=np.asarray(edp_samples.altitude, dtype=float),
    )


def _mesh_weights_at_latlon(edp_samples, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """``(vertex_indices, weights)``, each length 3 (barycentric) -- or a
    single repeated vertex with weight 1 if the point falls outside every
    mesh triangle (nearest-neighbour fallback, matching
    ``output._interpolate_field_at_latlon``'s own fallback exactly)."""
    positions = np.array([[lat, lon, float(edp_samples.altitude[0])]])
    _, _, idx_mesh, weight_mesh = edp_samples.interp(positions, coordinate="lla")
    tri_idx = int(idx_mesh[0])

    if tri_idx == -1:
        from scipy.spatial import cKDTree
        tree = cKDTree(edp_samples.geolocation)
        _, nearest = tree.query([[lon, lat]])
        return np.array([int(nearest[0])] * 3), np.array([1.0, 0.0, 0.0])

    verts = np.asarray(edp_samples.mesh[tri_idx])
    return verts, np.asarray(weight_mesh[0], dtype=float)


def build_f_edp(edp_samples, lat: float, lon: float, altitudes: np.ndarray) -> np.ndarray:
    """``F_EDP(point)``: shape ``(len(altitudes), n_height*n_geo)``, the
    same ``(n_rows, n_height*n_geo)`` convention
    ``EDPSamples.get_observation_operator`` uses for ``H_grid`` -- so
    :func:`Ensemble_Kalman_Engine.jacobian_utils.compose_observation_jacobian`
    (written generically against that shape, not against ray geometry
    specifically) accepts it as a drop-in left operand in place of
    ``H_grid`` (plan Section 2, Finding 5 / Section 11 step 1 -- confirmed
    by direct reading, no change needed to that function).

    One mesh lookup for the whole column (mesh location is
    altitude-independent), then one altitude-interpolation call for the
    (possibly many) requested altitudes -- mirrors
    ``output._interpolate_field_at_latlon``'s existing shortcut, extended
    from "whole native altitude axis, no altitude interpolation needed"
    to "arbitrary altitudes" via ``interp_heights`` directly (needed since
    an explicitly-user-supplied ``QueryPoint`` need not sit exactly on the
    model grid, even though the default -- Decision 3 -- does)."""
    import edp_samples as _edp_samples_module  # same module this EDPSamples instance's class lives in

    n_height = len(edp_samples.altitude)
    n_geo = edp_samples.geolocation.shape[0]
    altitudes = np.asarray(altitudes, dtype=float)
    n_alt = altitudes.shape[0]

    verts, w_mesh = _mesh_weights_at_latlon(edp_samples, lat, lon)
    idx_alt, weight_alt = _edp_samples_module.interp_heights(
        np.asarray(edp_samples.altitude, dtype=float), altitudes,
    )

    F = np.zeros((n_alt, n_height * n_geo), dtype=float)
    rows = np.arange(n_alt)
    for k in range(3):
        F[rows, idx_alt * n_geo + verts[k]] += weight_alt[:, 0] * w_mesh[k]
        F[rows, (idx_alt + 1) * n_geo + verts[k]] += weight_alt[:, 1] * w_mesh[k]
    return F


class ImpactVectorAccumulator:
    """Accumulates, for one batch of one style's ``centering="mean"``
    diagnostic pass, each RO/IGS entry's (unweighted, per-iteration)
    contribution to the analysis change at every configured
    :class:`QueryPoint` -- via :meth:`on_iteration`, the hook added to
    ``AnalysisEngine.analyze`` (plan Section 6) -- then combines them with
    the actual accepted ``alpha_history`` in :meth:`finalize` (plan
    Finding 3's decay weights, not knowable until every iteration's
    ``alpha`` is known).

    Usage::

        acc = ImpactVectorAccumulator(edp_samples, obs_operator, query_points, batch.entry_ray_ranges)
        _, diag = driver.assimilate_one_cycle(
            ensemble_forecast, obs_operator, y_obs, batch.R, on_iteration=acc.on_iteration,
        )
        impacts = acc.finalize(diag.alpha_history)   # {(entry.label, point.name): (n_altitudes,) array}
    """

    def __init__(self, edp_samples, obs_operator, query_points: list[QueryPoint],
                 entry_ray_ranges: list[tuple[Any, slice]]):
        self.obs_operator = obs_operator
        self.query_points = list(query_points)
        self.entry_ray_ranges = list(entry_ray_ranges)
        self._F = {
            qp.name: build_f_edp(edp_samples, qp.lat, qp.lon, qp.altitudes)
            for qp in self.query_points
        }
        self._raw_contribs: dict[tuple[str, str], list[np.ndarray]] = {}
        self._n_iterations = 0

    def on_iteration(self, k: int, x_before: np.ndarray, K: np.ndarray, r: np.ndarray) -> None:
        """Matches ``AnalysisEngine.analyze``'s ``on_iteration`` hook
        signature exactly. Projects through every query point immediately
        and keeps only the tiny resulting ``(n_altitudes,)`` arrays --
        never holds ``K``/the Jacobian past this call (plan Section 6's
        efficiency requirement)."""
        jac_kind, jac_array = self.obs_operator.get_jacobian_array(x_before)
        for qp in self.query_points:
            F = self._F[qp.name]
            A = compose_observation_jacobian(
                F, jac_array, self.obs_operator.param_shape,
                self.obs_operator.n_height, self.obs_operator.n_geo, jac_kind=jac_kind,
            )
            G = A @ K   # (n_altitudes, n_obs) -- cheap, computed once per point, not once per RO
            for entry, ray_slice in self.entry_ray_ranges:
                contrib = G[:, ray_slice] @ r[ray_slice]   # (n_altitudes,)
                key = (entry.label, qp.name)
                self._raw_contribs.setdefault(key, []).append(contrib)
        self._n_iterations += 1

    def finalize(self, alpha_history: list[float]) -> dict[tuple[str, str], np.ndarray]:
        """Combines every iteration's unweighted contribution with the
        decay weights implied by the actual accepted ``alpha_history``
        (plan Finding 3). ``alpha_history`` empty with exactly one
        iteration recorded is the ``raw``/linear style's single, undamped
        step (``w_0=1``, no damping concept applies there at all) --
        every other case needs ``len(alpha_history) == n_iterations``
        exactly, one ``alpha`` per accepted step."""
        J = self._n_iterations
        if J == 0:
            return {key: np.array([]) for key in self._raw_contribs}

        if len(alpha_history) == 0 and J == 1:
            weights = [1.0]
        elif len(alpha_history) == J:
            weights = []
            for k in range(J):
                decay = 1.0
                for m in range(k + 1, J):
                    decay *= (1.0 - alpha_history[m])
                weights.append(alpha_history[k] * decay)
        else:
            raise ValueError(
                f"ImpactVectorAccumulator.finalize: {len(alpha_history)} alphas for "
                f"{J} recorded iterations -- expected equal counts (or 0 alphas with "
                "exactly 1 iteration, the undamped linear-style case)."
            )

        result: dict[tuple[str, str], np.ndarray] = {}
        for key, contribs in self._raw_contribs.items():
            total = np.zeros_like(contribs[0])
            for w, c in zip(weights, contribs):
                total = total + w * c
            result[key] = total
        return result


# ===========================================================================
# Persistence (user request, 2026-11-06): the diagnostic pass that
# produces ``impacts_by_ro`` is a real extra Gauss-Newton solve (hours,
# at full scale) -- a plotting-only change (a new ranking metric,
# recolored heatmap, relabeled axis, ...) should never need to rerun
# that pass just to see the new picture. ``save_impact_vectors`` persists
# everything one query point's worth of plots needs (the raw per-entry
# profiles plus, when available, the ISR-alignment plots' own inputs);
# ``load_impact_vectors``/``replot_impact_vectors_from_saved`` read it
# back and regenerate every plot from the saved numbers alone, with no
# pipeline involved. ``replot_impact_vectors_tree`` does this for every
# saved file under an existing run's output directory in one call.
# ===========================================================================


@dataclass(frozen=True)
class SavedImpactVectors:
    impacts_by_ro: dict[str, np.ndarray]
    altitude: np.ndarray
    forecast_profile: np.ndarray | None
    isr_density: np.ndarray | None
    isr_altitude: np.ndarray | None
    style: str | None
    batch_index: int | None
    query_point_name: str | None


def save_impact_vectors(
    path: str | Path, impacts_by_ro: dict[str, np.ndarray], altitude: np.ndarray, *,
    style: str | None = None, batch_index: int | None = None, query_point_name: str | None = None,
    forecast_profile: np.ndarray | None = None,
    isr_density: np.ndarray | None = None, isr_altitude: np.ndarray | None = None,
) -> Path:
    """Saves one query point's worth of one batch's impact-vector data
    (matching the convention ``ensemble_io.save_ensemble_netcdf`` already
    uses in this package: ``xarray``-to-netCDF, metadata as attrs).
    ``forecast_profile``/``isr_density``/``isr_altitude`` are optional --
    only present for an ISR-station query point -- and omitted from the
    file entirely (not written as empty/NaN) when not supplied, so
    :func:`load_impact_vectors` can tell "no ISR data for this point"
    apart from "ISR data that happens to be all zero"."""
    labels = list(impacts_by_ro.keys())
    n_alt = len(np.asarray(altitude))
    M = (np.stack([np.asarray(impacts_by_ro[label], dtype=float) for label in labels], axis=0)
         if labels else np.zeros((0, n_alt)))

    data_vars: dict[str, Any] = {"impact": (("entry", "altitude"), M)}
    coords: dict[str, Any] = {"entry": labels, "altitude": np.asarray(altitude, dtype=float)}
    if forecast_profile is not None:
        data_vars["forecast_profile"] = (("altitude",), np.asarray(forecast_profile, dtype=float))
    if isr_density is not None and isr_altitude is not None:
        data_vars["isr_density"] = (("isr_altitude",), np.asarray(isr_density, dtype=float))
        coords["isr_altitude"] = np.asarray(isr_altitude, dtype=float)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    if style is not None:
        ds.attrs["style"] = style
    if batch_index is not None:
        ds.attrs["batch_index"] = int(batch_index)
    if query_point_name is not None:
        ds.attrs["query_point_name"] = query_point_name

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def load_impact_vectors(path: str | Path) -> SavedImpactVectors:
    """Loads a file :func:`save_impact_vectors` wrote. ``forecast_profile``/
    ``isr_density``/``isr_altitude`` come back ``None`` when the file was
    saved without them (a non-ISR query point), not as zero-filled
    arrays."""
    with xr.open_dataset(path) as ds:
        ds.load()
        labels = [str(x) for x in ds["entry"].values]
        M = np.asarray(ds["impact"].values)
        impacts_by_ro = {label: M[i] for i, label in enumerate(labels)}
        altitude = np.asarray(ds["altitude"].values)
        forecast_profile = np.asarray(ds["forecast_profile"].values) if "forecast_profile" in ds else None
        isr_density = np.asarray(ds["isr_density"].values) if "isr_density" in ds else None
        isr_altitude = np.asarray(ds["isr_altitude"].values) if "isr_altitude" in ds else None
        style = ds.attrs.get("style")
        batch_index = int(ds.attrs["batch_index"]) if "batch_index" in ds.attrs else None
        query_point_name = ds.attrs.get("query_point_name")

    return SavedImpactVectors(
        impacts_by_ro=impacts_by_ro, altitude=altitude, forecast_profile=forecast_profile,
        isr_density=isr_density, isr_altitude=isr_altitude,
        style=style, batch_index=batch_index, query_point_name=query_point_name,
    )


def _save_fig(fig_or_ax, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def replot_impact_vectors_from_saved(
    path: str | Path, out_dir: str | Path, *, file_prefix: str | None = None, title_suffix: str | None = None,
) -> list[Path]:
    """Regenerates every plot ``package_run.py``'s per-batch impact-vector
    helper would have produced, straight from a file :func:`save_impact_vectors`
    wrote -- no diagnostic Gauss-Newton pass, no ensemble, no pipeline at
    all, just the saved numbers replotted. This is the actual point of
    persisting them: a plotting-only change should cost seconds, not the
    hours the original diagnostic pass took at full scale."""
    saved = load_impact_vectors(path)
    out_dir = Path(out_dir)
    prefix = file_prefix if file_prefix is not None else "impact_vectors"
    name = saved.query_point_name or "query point"
    suffix = f" ({title_suffix})" if title_suffix else ""
    written: list[Path] = []

    written.append(_save_fig(
        plot_impact_vector_heatmap(saved.impacts_by_ro, saved.altitude, title=f"{name} impact vectors{suffix}"),
        out_dir / f"{prefix}_heatmap.png",
    ))
    written.append(_save_fig(
        plot_impact_vector_ranking(saved.impacts_by_ro, title=f"{name} impact ranking{suffix}"),
        out_dir / f"{prefix}_ranking.png",
    ))
    written.append(_save_fig(
        plot_impact_vector_ranking_relative_to_norm(
            saved.impacts_by_ro, title=f"{name} impact ranking, relative to total change{suffix}",
        ),
        out_dir / f"{prefix}_ranking_relative_norm.png",
    ))
    written.append(_save_fig(
        plot_impact_vector_ranking_relative_to_local_change(
            saved.impacts_by_ro, saved.altitude,
            title=f"{name} impact ranking, relative to local change{suffix}",
        ),
        out_dir / f"{prefix}_ranking_relative_local.png",
    ))

    if saved.forecast_profile is not None and saved.isr_density is not None and saved.isr_altitude is not None:
        written.append(_save_fig(
            plot_impact_vector_isr_alignment(
                saved.impacts_by_ro, saved.forecast_profile, saved.isr_density, saved.isr_altitude, saved.altitude,
                title=f"{name} ISR-alignment ranking{suffix}",
            ),
            out_dir / f"{prefix}_isr_alignment.png",
        ))
        written.append(_save_fig(
            plot_impact_vector_isr_aligned_heatmap(
                saved.impacts_by_ro, saved.forecast_profile, saved.isr_density, saved.isr_altitude, saved.altitude,
                title=f"{name} ISR-aligned heatmap{suffix}",
            ),
            out_dir / f"{prefix}_isr_aligned_heatmap.png",
        ))

    return written


def replot_impact_vectors_tree(root: str | Path) -> list[Path]:
    """Convenience batch entry point: finds every ``*_data.nc`` file
    :func:`save_impact_vectors` wrote under ``root`` (e.g. one of the
    ``*_frac65_v2`` run directories) and regenerates its plots in place,
    without touching the assimilation pipeline at all."""
    root = Path(root)
    written: list[Path] = []
    for data_path in sorted(root.rglob("*_data.nc")):
        saved = load_impact_vectors(data_path)
        file_prefix = data_path.name[: -len("_data.nc")]
        title_bits = [b for b in (saved.style, f"batch {saved.batch_index}" if saved.batch_index is not None else None) if b]
        title_suffix = ", ".join(title_bits) if title_bits else None
        written.extend(replot_impact_vectors_from_saved(
            data_path, data_path.parent, file_prefix=file_prefix, title_suffix=title_suffix,
        ))
    return written


# ===========================================================================
# Output / visualization (plan Section 8). ``impacts_by_ro`` throughout is
# ``{ro_label: (n_altitude,) impact profile}`` -- one query point's worth
# of ``ImpactVectorAccumulator.finalize()`` output, already sliced down to
# that point (its keys there are ``(ro_label, point_name)`` pairs).
# ===========================================================================


def _total_abs_impact(profile: np.ndarray) -> float:
    return float(np.abs(profile).sum())


def _ranked_labels(impacts_by_ro: dict[str, np.ndarray]) -> list[str]:
    return sorted(impacts_by_ro, key=lambda label: -_total_abs_impact(impacts_by_ro[label]))


def _altitude_bin_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges for a ``pcolormesh`` x-axis from altitude *centers* --
    exact (not approximated to a uniform grid), so this works whether or
    not the model grid's spacing is uniform."""
    centers = np.asarray(centers, dtype=float)
    if centers.shape[0] == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def plot_impact_vector_heatmap(
    impacts_by_ro: dict[str, np.ndarray], altitude: np.ndarray, *,
    ax=None, figsize=(8, 6), cmap: str = "RdBu_r", title: str | None = None,
):
    """RO x altitude heatmap of the signed impact (plan Section 8) --
    ROs ordered by total ``|impact|`` (largest first), diverging colormap
    with symmetric ``vmin``/``vmax`` around zero (same convention as the
    analysis-minus-forecast diff plots, 2026-10-01) so "no effect" is
    always the colormap's center regardless of this batch's actual
    impact range."""
    labels = _ranked_labels(impacts_by_ro)
    M = np.stack([np.asarray(impacts_by_ro[label], dtype=float) for label in labels], axis=0)
    abs_max = float(np.abs(M).max()) or 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    alt_edges = _altitude_bin_edges(altitude)
    ro_edges = np.arange(len(labels) + 1, dtype=float)
    mesh = ax.pcolormesh(alt_edges, ro_edges, M, cmap=cmap, vmin=-abs_max, vmax=abs_max, shading="flat")

    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()   # largest-impact RO at the top
    ax.set_xlabel("Altitude (km)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Impact on Ne (m$^{-3}$)")
    ax.set_title(title or "Impact vector by RO and altitude")
    return ax


def plot_impact_vector_ranking(
    impacts_by_ro: dict[str, np.ndarray], *, ax=None, figsize=(6, 6), title: str | None = None,
):
    """Bar chart of total ``|impact|`` per RO (integrated over altitude)
    -- directly answers the source note's own motivating question, "which
    RO played the most critical role" (plan Section 8)."""
    labels = _ranked_labels(impacts_by_ro)
    totals = [_total_abs_impact(impacts_by_ro[label]) for label in labels]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, totals, color="C0")
    ax.invert_yaxis()   # largest-impact RO at the top
    ax.set_xlabel("Total |impact| on Ne (m$^{-3}$, integrated over altitude)")
    ax.set_title(title or "RO impact ranking")
    return ax


def _total_change_profile(impacts_by_ro: dict[str, np.ndarray]) -> np.ndarray:
    """The additive reconstruction of the actual (diagnostic-pass)
    analysis change at this query point: summing every RO/IGS entry's
    own contribution recovers the total change, since the decomposition
    in :class:`ImpactVectorAccumulator` is additive by construction
    (plan Section 3)."""
    profiles = [np.asarray(p, dtype=float) for p in impacts_by_ro.values()]
    return np.sum(profiles, axis=0) if profiles else np.array([])


def plot_impact_vector_ranking_relative_to_norm(
    impacts_by_ro: dict[str, np.ndarray], *, ax=None, figsize=(6, 6), title: str | None = None,
):
    """Same ranking as :func:`plot_impact_vector_ranking`, but as a
    fraction (%) of ``||total change||_2`` instead of raw m^-3 --
    unitless, so magnitudes become comparable across query points,
    styles, and batches regardless of how large the overall analysis
    correction happened to be at this particular point."""
    total = _total_change_profile(impacts_by_ro)
    total_norm = float(np.linalg.norm(total)) or 1.0
    labels = sorted(impacts_by_ro, key=lambda label: -float(np.linalg.norm(impacts_by_ro[label])))
    fractions = [100.0 * float(np.linalg.norm(impacts_by_ro[label])) / total_norm for label in labels]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, fractions, color="C0")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\|\mathrm{impact}\|_2 \,/\, \|\mathrm{total\ EDP\ change}\|_2$ (%)")
    ax.set_title(title or "RO impact ranking, relative to total change norm")
    return ax


def plot_impact_vector_ranking_relative_to_local_change(
    impacts_by_ro: dict[str, np.ndarray], altitude: np.ndarray, *,
    floor_frac: float = 0.05, ax=None, figsize=(6, 6), title: str | None = None,
):
    """A second, altitude-resolved relative metric: at every altitude
    ``h`` where the total change is resolvable (``|total(h)| >=
    floor_frac * max|total|`` -- guards against dividing by near-zero
    where the analysis barely moved at all), the local fraction
    ``contrib_ell(h) / total(h)`` is computed, then altitude-weighted
    (trapezoidal) averaged over that span into one number per RO/IGS
    entry. Unlike :func:`plot_impact_vector_ranking_relative_to_norm`
    (one global denominator), this rewards an entry that dominates the
    change within even a narrow altitude band, not only one with the
    largest overall-column impact."""
    altitude = np.asarray(altitude, dtype=float)
    total = _total_change_profile(impacts_by_ro)
    floor = floor_frac * (float(np.abs(total).max()) or 1.0)
    mask = np.abs(total) >= floor
    weight = mask.astype(float)
    denom = float(np.trapezoid(weight, altitude))

    def _score(label: str) -> float:
        if denom == 0.0:
            return 0.0
        contrib = np.asarray(impacts_by_ro[label], dtype=float)
        ratio = np.zeros_like(total)
        ratio[mask] = contrib[mask] / total[mask]
        return float(np.trapezoid(np.abs(ratio) * weight, altitude) / denom)

    labels = sorted(impacts_by_ro, key=lambda label: -_score(label))
    fractions = [100.0 * _score(label) for label in labels]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, fractions, color="C0")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Altitude-mean |local impact / local total change| (%),\n"
        f"over altitudes where |total change| >= {floor_frac:.0%} of its column max"
    )
    ax.set_title(title or "RO impact ranking, relative to local change")
    return ax


def plot_impact_vector_ranking_cross_style(
    impacts_by_ro_by_style: dict[str, dict[str, np.ndarray]], *,
    ax=None, figsize=(8, 6), title: str | None = None,
):
    """Same ranking as :func:`plot_impact_vector_ranking`, one bar-group
    per RO with one bar per style -- matches this project's established
    per-style + cross-style convention (``cross_style_data``,
    ``isr_comparison.py``'s own per-style/cross-style pairs). A style
    missing a given RO (e.g. it wasn't part of that style's batch, or
    only this style's run computed impact vectors) contributes a
    zero-height bar rather than erroring."""
    from .output import _color_for_style

    styles = list(impacts_by_ro_by_style.keys())
    all_labels = {label for impacts in impacts_by_ro_by_style.values() for label in impacts}
    rank_key = lambda label: -max(   # noqa: E731
        _total_abs_impact(impacts_by_ro_by_style[style][label])
        for style in styles if label in impacts_by_ro_by_style[style]
    )
    labels = sorted(all_labels, key=rank_key)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    n_styles = max(len(styles), 1)
    bar_h = 0.8 / n_styles
    y = np.arange(len(labels), dtype=float)
    for i, style in enumerate(styles):
        totals = [
            _total_abs_impact(impacts_by_ro_by_style[style][label]) if label in impacts_by_ro_by_style[style] else 0.0
            for label in labels
        ]
        offset = (i - (n_styles - 1) / 2.0) * bar_h
        ax.barh(y + offset, totals, height=bar_h, color=_color_for_style(style), label=style)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Total |impact| on Ne (m$^{-3}$, integrated over altitude)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title or "RO impact ranking (all styles)")
    return ax


# ===========================================================================
# Ranking/visualization by direction relative to ISR (plan Section 8,
# Decision 5): not just "how big was this RO's impact" but "did it push
# the analysis toward or away from the real ISR-measured profile" --
# directly answers the source note's own background question ("whether
# these changes improve agreement to EDP measured by ISR").
# ===========================================================================


def isr_correction_direction(
    forecast_profile: np.ndarray, isr_density: np.ndarray, isr_altitude: np.ndarray, model_altitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(d, mask)``: ``d[h] = M(h) - F(h)`` (the correction needed to
    reach the ISR-measured profile from the forecast), on
    ``model_altitude``'s grid, with ``M`` obtained by interpolating
    ``isr_density`` (native ISR gates) *onto* the model grid -- the
    reverse direction from ``isr_comparison.py``'s own interpolation
    (which goes model -> ISR gates, to avoid ever displaying an
    extrapolated ISR value on its own). That's fine here specifically
    because ``d`` is only ever used inside a dot product or masked
    display, restricted to ``mask`` -- no extrapolated value of ``M`` is
    ever shown or summed on its own. ``mask`` is ``True`` where
    ``model_altitude`` falls within the ISR instrument's own native
    altitude coverage (``isr_altitude.min()..max()``); ``d`` is ``0`` (not
    ``nan``) outside ``mask`` so a careless caller that forgets to apply
    the mask gets a zero contribution there, not a `nan`-poisoned sum."""
    from .isr_comparison import _interp_log10

    model_altitude = np.asarray(model_altitude, dtype=float)
    M_on_model = _interp_log10(model_altitude, np.asarray(isr_altitude, dtype=float), np.asarray(isr_density, dtype=float))
    mask = np.isfinite(M_on_model)
    d = np.where(mask, M_on_model - np.asarray(forecast_profile, dtype=float), 0.0)
    return d, mask


def rank_by_isr_alignment(
    impacts_by_ro: dict[str, np.ndarray], forecast_profile: np.ndarray,
    isr_density: np.ndarray, isr_altitude: np.ndarray, model_altitude: np.ndarray,
) -> dict[str, dict[str, float]]:
    """For each RO, the signed projection of its impact profile onto the
    ISR-correction direction (:func:`isr_correction_direction`), masked
    to the ISR instrument's own altitude coverage -- positive means that
    RO pushed the analysis *toward* the ISR-measured profile over the
    covered band, negative means *away* from it. Also reports ``cosine``,
    the same projection normalized by ``||impact||_2 * ||d||_2`` (both
    norms over the same masked band) -- bounded to ``[-1, 1]`` regardless
    of either vector's scale, so it is comparable across ROs, query
    points, styles and batches the way the raw ``projection`` (whose
    scale depends on both the entry's own magnitude and ISR's) is not.
    Returns ``{ro_label: {"projection": signed scalar, "cosine": signed
    scalar in [-1, 1], "total_abs_impact": plain magnitude}}`` -- all
    three reported together so "large but unhelpful", "small but
    unhelpful", and "well-aligned but small" stay distinguishable (plan
    Section 8)."""
    d, mask = isr_correction_direction(forecast_profile, isr_density, isr_altitude, model_altitude)
    d_norm = float(np.linalg.norm(d)) or 1.0
    result: dict[str, dict[str, float]] = {}
    for label, profile in impacts_by_ro.items():
        profile = np.asarray(profile, dtype=float)
        masked_profile = np.where(mask, profile, 0.0)
        projection = float(np.sum(masked_profile * d))
        profile_norm = float(np.linalg.norm(masked_profile)) or 1.0
        cosine = projection / (profile_norm * d_norm)
        result[label] = {
            "projection": projection, "cosine": cosine, "total_abs_impact": _total_abs_impact(profile),
        }
    return result


def plot_impact_vector_isr_alignment(
    impacts_by_ro: dict[str, np.ndarray], forecast_profile: np.ndarray,
    isr_density: np.ndarray, isr_altitude: np.ndarray, model_altitude: np.ndarray, *,
    ax=None, figsize=(6, 6), title: str | None = None,
):
    """Bar chart of :func:`rank_by_isr_alignment`'s ``cosine`` (projection
    normalized by ``||impact|| * ||d||``, bounded to ``[-1, 1]``) per RO,
    ranked most-positive (most helpful toward ISR truth) to most-negative
    (most harmful) -- diverging color by sign. Using ``cosine`` rather
    than the raw ``projection`` keeps the bar lengths directly comparable
    across ROs regardless of how large any one entry's own impact
    happens to be (plan Section 8, 2026-11-06 follow-up)."""
    ranked = rank_by_isr_alignment(impacts_by_ro, forecast_profile, isr_density, isr_altitude, model_altitude)
    labels = sorted(ranked, key=lambda label: -ranked[label]["cosine"])
    cosines = [ranked[label]["cosine"] for label in labels]
    colors = ["#c0392b" if c >= 0 else "#2980b9" for c in cosines]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, cosines, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine similarity: impact vs. (ISR - forecast) direction\n(+1 = fully toward ISR, -1 = fully away)")
    ax.set_title(title or "RO impact, ranked by alignment with ISR")
    return ax


def plot_impact_vector_isr_aligned_heatmap(
    impacts_by_ro: dict[str, np.ndarray], forecast_profile: np.ndarray,
    isr_density: np.ndarray, isr_altitude: np.ndarray, model_altitude: np.ndarray, *,
    ax=None, figsize=(8, 6), cmap: str = "RdBu_r", title: str | None = None,
):
    """Same ``(RO x altitude)`` grid as :func:`plot_impact_vector_heatmap`,
    but recolored by alignment with ISR instead of raw impact sign (plan
    Section 8, confirmed with the user 2026-11-05): per-cell value
    ``I_ℓ(h) * sign(d_masked(h))`` -- keeps the impact's own units/
    magnitude (m^-3), only flipping sign locally wherever the ISR-
    correction direction is negative there, so one consistent color
    always means "toward ISR" and the other always means "away from ISR",
    at every altitude, for every RO. Altitudes outside the ISR
    instrument's native coverage are left blank (not drawn), never
    silently extrapolated."""
    d, mask = isr_correction_direction(forecast_profile, isr_density, isr_altitude, model_altitude)
    sign_d = np.where(mask, np.sign(d), 0.0)

    ranked = rank_by_isr_alignment(impacts_by_ro, forecast_profile, isr_density, isr_altitude, model_altitude)
    labels = sorted(impacts_by_ro, key=lambda label: -ranked[label]["cosine"])

    aligned = {}
    for label in labels:
        profile = np.asarray(impacts_by_ro[label], dtype=float)
        row = profile * sign_d
        row = np.where(mask, row, np.nan)   # blank outside ISR coverage, not zero
        aligned[label] = row

    M = np.stack([aligned[label] for label in labels], axis=0)
    finite = M[np.isfinite(M)]
    abs_max = float(np.abs(finite).max()) if finite.size else 1.0
    abs_max = abs_max or 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    alt_edges = _altitude_bin_edges(model_altitude)
    ro_edges = np.arange(len(labels) + 1, dtype=float)
    M_masked = np.ma.masked_invalid(M)
    mesh = ax.pcolormesh(alt_edges, ro_edges, M_masked, cmap=cmap, vmin=-abs_max, vmax=abs_max, shading="flat")

    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Altitude (km)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Impact, aligned with ISR direction (m$^{-3}$)")
    ax.set_title(title or "Impact vector by RO and altitude, aligned with ISR")
    return ax
