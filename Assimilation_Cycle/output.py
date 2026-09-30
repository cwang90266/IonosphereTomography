#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output and diagnostics (plan Section 4.7/4.14): decode the final ensemble,
save it in ``EDPSamples``'s own netCDF schema, and plot it -- horizontal
(reusing ``EDPSamples.plot_horizontal_field`` directly, which already
accepts a raw ``(n_geo,)`` array), vertical (new -- forecast vs. analysis
ensemble mean +/- spread by altitude, since nothing in ``EDPSamples``/
``Parameterization`` plots that comparison), and the three ``metrics.py``
plots (RMSE reduction, rank histogram, effective-rank time series).

All figures use a headless (``Agg``) matplotlib backend, matching the
convention already used by every other test/plotting module in this
repository (``EDPSamples``/``Parameterization``/``observation_preparation``
conftest.py files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver

from .cycle_driver import CycleResult

# Per-geo_type attrs EDPSamples.__init__ needs to regenerate the same
# geolocation/mesh -- matches the attrs it stores for each branch
# (see edp_samples.py EDPSamples.__init__'s final `match geo_type` block).
_GEO_TYPE_ATTRS: dict[str, tuple[str, ...]] = {
    "Point": ("Lon", "Lat"),
    "Rectangle": ("minLon", "maxLon", "dLon", "minLat", "maxLat", "dLat"),
    "Polar": ("minLat", "dLat"),
    "Global": ("equal_spaced", "dLat", "dLon"),
    "Regional": ("Lon", "Lat", "radius", "dLat"),
}


def decode_ensemble(driver: GeneralEnKFDriver, ensemble: EnsembleState, obs_operator) -> np.ndarray:
    """``(n_height, n_geo, n_members)`` physical electron density (plan
    Section 4.7) -- thin wrapper around ``GeneralEnKFDriver.decode_to_edps``."""
    return driver.decode_to_edps(ensemble, obs_operator)


def save_decoded_field_netcdf(edp_samples_template, decoded_density: np.ndarray, path: str | Path):
    """
    Save a decoded ``(n_height, n_geo, n_members)`` density field as an
    ``EDPSamples``-schema netCDF (plan Section 4.7), reusing
    ``edp_samples_template``'s own grid (geolocation/mesh/altitude,
    regenerated via the same ``geo_type``-specific attrs it was built
    with, not copied by reference -- so this is a real, independent
    ``EDPSamples`` object, round-trippable through ``fromNetCDF`` and
    ``plot_horizontal_field``/``plot_geolocation`` like any other).

    ``feature_edps`` (foF2/hmF2/... -- normally IRI2020's own Fortran
    output, not derivable in Python from an arbitrary density field) is
    not recomputed for the assimilated field; the saved file is zero-filled
    there with ``attrs['feature_edps_computed'] = 0`` so this is explicit
    rather than silently misleading.
    """
    E = type(edp_samples_template)
    attrs = edp_samples_template.attrs
    geo_type = attrs["geo_type"]
    if geo_type not in _GEO_TYPE_ATTRS:
        raise ValueError(
            f"save_decoded_field_netcdf: geo_type={geo_type!r} not supported "
            f"(supported: {sorted(_GEO_TYPE_ATTRS)})"
        )
    geo_kwargs = {k: attrs[k] for k in _GEO_TYPE_ATTRS[geo_type]}

    n_height, n_geo, n_members = decoded_density.shape
    sampling_parameters = _import_pandas().DataFrame({"member": np.arange(n_members)})
    feature_edps = np.zeros((len(E.FEATURE_LABEL), n_geo, n_members))

    field = E(
        DateTime=attrs["DateTime"], geo_type=geo_type,
        altitude=np.asarray(edp_samples_template.altitude),
        sampling_parameters=sampling_parameters,
        edps=decoded_density, feature_edps=feature_edps,
        **geo_kwargs,
    )
    field.attrs["feature_edps_computed"] = 0

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field.saveNetCDF(path)
    return field


def _import_pandas():
    import pandas as pd
    return pd


def plot_horizontal(edp_samples_template, density_slice: np.ndarray, *, target_alt: float,
                     ax=None, **kwargs):
    """Thin wrapper around ``EDPSamples.plot_horizontal_field`` (plan
    Section 4.14 -- horizontal reuses the existing method as-is), for a
    caller-computed ``(n_geo,)`` array (e.g. an ensemble mean at one
    altitude) rather than a sample already stored in ``edp_samples_template``."""
    label = kwargs.pop("scalar_label", f"Electron Density (m$^{{-3}}$) [Alt: {target_alt:.1f} km]")
    return edp_samples_template.plot_horizontal_field(scalar=density_slice, ax=ax, scalar_label=label, **kwargs)


def plot_vertical_profile(
    altitude: np.ndarray,
    forecast_density: np.ndarray,
    analysis_density: np.ndarray,
    geo_idx: int = 0,
    ax=None,
    figsize=(6, 7),
):
    """
    New (plan Section 4.14): forecast ensemble mean +/- spread vs. analysis
    ensemble mean +/- spread, by altitude, at one horizontal grid point --
    distinct from ``Parameterized_EDPSamples.plot_reconstruction_profile``
    (parameterization reconstruction error, not forecast-vs-analysis).

    Parameters
    ----------
    altitude : ndarray, shape (n_height,)
    forecast_density, analysis_density : ndarray, shape (n_height, n_geo, n_members)
    geo_idx : int
        Horizontal grid point to plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    f = forecast_density[:, geo_idx, :]
    a = analysis_density[:, geo_idx, :]
    f_mean, f_std = f.mean(axis=1), f.std(axis=1)
    a_mean, a_std = a.mean(axis=1), a.std(axis=1)

    ax.plot(f_mean, altitude, color="C0", label="forecast mean")
    ax.fill_betweenx(altitude, f_mean - f_std, f_mean + f_std, color="C0", alpha=0.2, label="forecast +/-1 std")
    ax.plot(a_mean, altitude, color="C1", label="analysis mean")
    ax.fill_betweenx(altitude, a_mean - a_std, a_mean + a_std, color="C1", alpha=0.2, label="analysis +/-1 std")

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"Vertical profile (geo index {geo_idx})")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_rmse_reduction(cycle_result: CycleResult, ax=None, figsize=(7, 4)):
    """RMSE-reduction bar/line chart by batch (plan Section 4.14)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    batch_idx = [o.batch_index for o in cycle_result.batch_outcomes]
    forecast = [o.rmse_reduction.rmse_forecast for o in cycle_result.batch_outcomes]
    analysis = [o.rmse_reduction.rmse_analysis for o in cycle_result.batch_outcomes]

    ax.plot(batch_idx, forecast, "o-", color="C0", label="forecast RMSE")
    ax.plot(batch_idx, analysis, "o-", color="C1", label="analysis RMSE")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("RMSE (TECU)")
    ax.set_title("RMSE reduction by batch")
    ax.legend(loc="best")
    return ax


def plot_rank_histogram(cycle_result: CycleResult, ax=None, figsize=(6, 4)):
    """Rank histogram pooled across all batches in the cycle (plan Section 4.13/4.14)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pooled = None
    for outcome in cycle_result.batch_outcomes:
        pooled = outcome.rank_histogram.copy() if pooled is None else pooled + outcome.rank_histogram

    ranks = np.arange(len(pooled))
    ax.bar(ranks, pooled, color="C2")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.set_title("Rank histogram (pooled across batches)")
    return ax


def plot_effective_rank_series(cycle_result: CycleResult, ax=None, figsize=(7, 4)):
    """Effective ensemble rank (forecast/analysis) across batches (plan
    Section 4.13/4.14) -- monitors rank collapse/degeneracy over the cycle."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    batch_idx = [o.batch_index for o in cycle_result.batch_outcomes]
    forecast_rank = [o.effective_rank_forecast["rank"] for o in cycle_result.batch_outcomes]
    analysis_rank = [o.effective_rank_analysis["rank"] for o in cycle_result.batch_outcomes]

    ax.plot(batch_idx, forecast_rank, "o-", color="C0", label="forecast rank")
    ax.plot(batch_idx, analysis_rank, "o-", color="C1", label="analysis rank")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Effective ensemble rank")
    ax.set_title("Effective ensemble rank by batch")
    ax.legend(loc="best")
    return ax


# ===========================================================================
# Final_Packaging.docx visualizations (2026-09-25). Functions above return a
# single ``ax`` (one natural panel); the ones below need several panels at
# once (one per driving index / altitude / style metric), so they take
# ``fig=None`` and return ``fig`` instead -- documented per-function.
# ===========================================================================

_IRI_INPUT_COLUMNS = ("hour", "f107", "ap", "ig12", "rz12")


def plot_iri_input_distributions(sampling_parameters, fig=None, figsize=(11, 6)):
    """Histograms of the drawn IRI2020 driving indices (plan/docx step 3)
    -- one panel per column of ``iri_selection.sampling_parameters_for_cycle``'s
    return value. A column that's all-NaN (no spread requested for that
    index, Section 8.1's `pd.isna` convention) gets an empty panel
    labelled accordingly rather than a matplotlib error on an all-NaN
    histogram."""
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    else:
        axes = np.asarray(fig.axes).reshape(2, 3) if fig.axes else fig.subplots(2, 3)
    axes = np.asarray(axes).ravel()

    for ax, col in zip(axes, _IRI_INPUT_COLUMNS):
        values = sampling_parameters[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "no spread requested", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(finite, bins=min(30, max(5, finite.size // 5)), color="C0", edgecolor="black", linewidth=0.3)
        ax.set_title(col)
        ax.set_ylabel("count")
    for ax in axes[len(_IRI_INPUT_COLUMNS):]:
        ax.axis("off")

    fig.suptitle(f"IRI2020 input distributions (n={len(sampling_parameters)})")
    fig.tight_layout()
    return fig


def plot_observation_operator_sum(
    edp_samples, entry, altitudes=(100, 200, 300, 400, 500, 600, 700, 800),
    num_segments: int = 1000, fig=None, figsize=(14, 7),
):
    """Sum of the observation operator's rows over every ray in one RO/IGS
    ``entry``, shown on the horizontal grid at several altitudes (plan/docx
    step 5) -- "where on the grid does this occultation/arc actually have
    sensitivity, and at what height." One panel per altitude in
    ``altitudes`` (nearest available grid level used)."""
    H_entry = edp_samples.get_observation_operator(entry.to_operator_dict(), num_segments=num_segments)
    n_height = len(edp_samples.altitude)
    n_geo = edp_samples.geolocation.shape[0]
    summed = np.asarray(H_entry).sum(axis=0).reshape(n_height, n_geo)

    n_alt = len(altitudes)
    ncols = min(4, n_alt)
    nrows = int(np.ceil(n_alt / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                  subplot_kw={"projection": _cartopy_projection(edp_samples)})
    axes = np.asarray(fig.axes).ravel()

    for ax, target_alt in zip(axes, altitudes):
        alt_idx = int(np.argmin(np.abs(edp_samples.altitude - target_alt)))
        actual_alt = float(edp_samples.altitude[alt_idx])
        plot_horizontal(edp_samples, summed[alt_idx, :], target_alt=actual_alt, ax=ax,
                         scalar_label=f"sum(H) [{actual_alt:.0f} km]")
    for ax in axes[n_alt:]:
        ax.axis("off")

    fig.suptitle(f"Observation operator sum: {entry.label or entry.obs_type} ({entry.n_rays} rays)")
    fig.tight_layout()
    return fig


def _cartopy_projection(edp_samples):
    """Delegates to ``edp_samples.select_map_projection()`` -- the single
    source of truth for the projection/auto-polar decision (also used by
    ``EDPSamples._horizontal_map_axes`` internally) -- rather than a
    separately hardcoded choice, which previously left this module's
    multi-panel figures (built via ``plt.subplots(...,
    subplot_kw={"projection": ...})``, needing one projection fixed
    before any panel's axes exists) stuck on Plate Carree even for a
    high-latitude grid that should auto-switch to polar (real bug, found
    2026-09-28 from a user report after the auto-polar switch was already
    live for every *other* horizontal plot)."""
    proj, _, _, _ = edp_samples.select_map_projection()
    return proj


def plot_tec_profile_comparison(entry, y_forecast: np.ndarray, y_analysis: np.ndarray,
                                 y_measured: np.ndarray, ax=None, figsize=(6, 6)):
    """Forecast vs. analysis vs. measured TEC for one RO occultation (plan/
    docx step 7), by tangent altitude -- ``y_forecast``/``y_analysis``/
    ``y_measured`` are already sliced to this entry's rays (e.g. via
    ``CycleBatch.entry_ray_ranges``), all shape ``(entry.n_rays,)``. RO-
    specific: uses ``entry.tangent_alt_km`` as the y-axis, so this isn't
    meaningful for an IGS arc (no tangent point)."""
    if entry.tangent_alt_km is None:
        raise ValueError(
            "plot_tec_profile_comparison: entry has no tangent_alt_km -- "
            "this plot is RO-specific (occultations have a tangent-altitude "
            "profile; IGS arcs don't)."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    order = np.argsort(entry.tangent_alt_km)
    alt = np.asarray(entry.tangent_alt_km)[order]

    ax.plot(np.asarray(y_measured)[order], alt, "o-", color="k", label="measured", markersize=3)
    ax.plot(np.asarray(y_forecast)[order], alt, "o-", color="C0", label="forecast", markersize=3)
    ax.plot(np.asarray(y_analysis)[order], alt, "o-", color="C1", label="analysis", markersize=3)

    ax.set_xlabel("TEC (TECU)")
    ax.set_ylabel("Tangent altitude (km)")
    ax.set_title(f"TEC profile: {entry.label or entry.obs_type}")
    ax.legend(loc="best", fontsize=8)
    return ax


def _interpolate_field_at_latlon(edp_samples, field: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """Horizontally interpolate a ``(n_height, n_geo, ...)`` field at
    ``(lat, lon)`` using ``EDPSamples.interp``'s own mesh-locate machinery
    (the same barycentric interpolation ``get_observation_operator`` uses
    internally) -- returns shape ``(n_height, ...)``. Falls back to the
    nearest grid vertex if the point lies outside every mesh triangle
    (same fallback ``get_observation_operator`` uses for out-of-mesh ray
    points)."""
    positions = np.array([[lat, lon, float(edp_samples.altitude[0])]])
    _, _, idx_mesh, weight_mesh = edp_samples.interp(positions, coordinate="lla")
    tri_idx = int(idx_mesh[0])

    if tri_idx == -1:
        from scipy.spatial import cKDTree
        tree = cKDTree(edp_samples.geolocation)
        _, nearest = tree.query([[lon, lat]])
        return field[:, int(nearest[0]), ...]

    verts = edp_samples.mesh[tri_idx]
    w = weight_mesh[0]
    return w[0] * field[:, verts[0], ...] + w[1] * field[:, verts[1], ...] + w[2] * field[:, verts[2], ...]


def resolve_edp_query_point_and_profiles(
    edp_samples, entry, decoded_forecast: np.ndarray, decoded_analysis: np.ndarray,
    *, lat: float | None = None, lon: float | None = None,
    tangent_alt_window: tuple[float, float] = (250.0, 350.0),
):
    """Shared by :func:`plot_edp_profile_comparison` and the cross-style
    comparison plots (``package_run.py`` needs the same lat/lon-derivation
    and horizontal-interpolation logic to collect one style's profile at
    a time, before any of them are plotted): resolves the query
    ``(lat, lon)`` (derived from ``entry`` if not given -- see
    :func:`plot_edp_profile_comparison`'s docstring for the derivation
    rule) and returns ``(lat, lon, forecast_profile, analysis_profile)``,
    where the profiles are ``(n_height, n_members)`` -- the raw
    interpolated ensemble, not yet reduced to mean/std, so a caller
    comparing multiple styles can combine them however it needs to.

    Raises ``ValueError`` under the same conditions
    ``plot_edp_profile_comparison`` always has (no ``tangent_alt_km``, or
    no ray in ``tangent_alt_window``, when ``lat``/``lon`` aren't given
    explicitly).
    """
    if lat is None or lon is None:
        if entry.tangent_alt_km is None:
            raise ValueError("resolve_edp_query_point_and_profiles: entry has no tangent_alt_km; pass lat/lon explicitly.")
        window_mask = (
            (np.asarray(entry.tangent_alt_km) >= tangent_alt_window[0])
            & (np.asarray(entry.tangent_alt_km) <= tangent_alt_window[1])
        )
        if not window_mask.any():
            raise ValueError(
                f"resolve_edp_query_point_and_profiles: no rays with tangent_alt_km in "
                f"{tangent_alt_window} for entry {entry.label!r}; pass lat/lon explicitly."
            )
        candidate_idx = np.where(window_mask)[0]
        best = candidate_idx[np.argmax(np.asarray(entry.tec)[candidate_idx])]

        from edp_samples import EDPSamples as _E, _ecef_to_geodetic
        tangent_xyz, _, _ = _E.rayTangent(
            np.asarray(entry.rec_ecef_km)[:, best:best + 1],
            np.asarray(entry.gnss_ecef_km)[:, best:best + 1],
            units="km",
        )
        lat_arr, lon_arr, _ = _ecef_to_geodetic(tangent_xyz.T * 1000.0)
        lat, lon = float(lat_arr[0]), float(lon_arr[0])

    forecast_profile = _interpolate_field_at_latlon(edp_samples, decoded_forecast, lat, lon)
    analysis_profile = _interpolate_field_at_latlon(edp_samples, decoded_analysis, lat, lon)
    return lat, lon, forecast_profile, analysis_profile


def plot_edp_profile_comparison(
    edp_samples, entry, decoded_forecast: np.ndarray, decoded_analysis: np.ndarray,
    *, lat: float | None = None, lon: float | None = None,
    tangent_alt_window: tuple[float, float] = (250.0, 350.0),
    ax=None, figsize=(6, 7),
):
    """Forecast/analysis EDP vertical profile at a query location, overlaid
    with the Abel-retrieved profile (plan/docx step 8). If ``lat``/``lon``
    are omitted, derived from ``entry`` itself: the tangent point of the
    ray with maximum TEC among those whose tangent altitude falls in
    ``tangent_alt_window`` (default 250-350km, straddling the typical F2
    peak). Passing ``lat``/``lon`` explicitly instead makes this the same
    function the docx's planned future external-validation use reuses, at
    an arbitrary specified location rather than one derived from an RO.

    ``decoded_forecast``/``decoded_analysis`` : ``(n_height, n_geo, n_members)``.
    """
    lat, lon, forecast_profile, analysis_profile = resolve_edp_query_point_and_profiles(
        edp_samples, entry, decoded_forecast, decoded_analysis,
        lat=lat, lon=lon, tangent_alt_window=tangent_alt_window,
    )
    f_mean, f_std = forecast_profile.mean(axis=-1), forecast_profile.std(axis=-1)
    a_mean, a_std = analysis_profile.mean(axis=-1), analysis_profile.std(axis=-1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    altitude = np.asarray(edp_samples.altitude)
    ax.plot(f_mean, altitude, color="C0", label="forecast mean")
    ax.fill_betweenx(altitude, f_mean - f_std, f_mean + f_std, color="C0", alpha=0.2)
    ax.plot(a_mean, altitude, color="C1", label="analysis mean")
    ax.fill_betweenx(altitude, a_mean - a_std, a_mean + a_std, color="C1", alpha=0.2)

    if entry.abel and "Ne" in entry.abel and "alt_km" in entry.abel:
        ax.plot(entry.abel["Ne"], entry.abel["alt_km"], color="k", linestyle="--", label="Abel-retrieved")

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"EDP profile at ({lat:.2f}, {lon:.2f})" + (f" -- {entry.label}" if entry.label else ""))
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_tec_edp_profile_comparison(
    entry, y_forecast: np.ndarray, y_analysis: np.ndarray, y_measured: np.ndarray,
    edp_samples, decoded_forecast: np.ndarray, decoded_analysis: np.ndarray,
    *, lat: float | None = None, lon: float | None = None,
    tangent_alt_window: tuple[float, float] = (250.0, 350.0),
    figsize=(12, 6),
):
    """One two-panel figure per RO (2026-09-29 user request, replaces
    separately saving :func:`plot_tec_profile_comparison` and
    :func:`plot_edp_profile_comparison`): TEC comparison (left panel) and
    EDP comparison (right panel) for the same occultation, same style.
    Arguments are exactly the union of both functions' own (see their
    docstrings). If no ray falls in ``tangent_alt_window`` (the EDP
    panel's own possible failure mode), the EDP panel gets a placeholder
    message instead of failing the whole figure -- the TEC panel is
    always independently plottable."""
    fig, (ax_tec, ax_edp) = plt.subplots(1, 2, figsize=figsize)

    # Both plot_*_profile_comparison functions bake the entry label into
    # their own per-axes title (sensible for standalone use); strip that
    # redundant suffix here since the figure-level suptitle below already
    # carries it -- otherwise the two titles visually collide with the
    # suptitle (found visually reviewing the first version of this plot).
    plot_tec_profile_comparison(entry, y_forecast, y_analysis, y_measured, ax=ax_tec)
    ax_tec.set_title(ax_tec.get_title().split(":")[0])
    try:
        plot_edp_profile_comparison(
            edp_samples, entry, decoded_forecast, decoded_analysis,
            lat=lat, lon=lon, tangent_alt_window=tangent_alt_window, ax=ax_edp,
        )
        ax_edp.set_title(ax_edp.get_title().split(" -- ")[0])
    except ValueError:
        ax_edp.text(0.5, 0.5, f"no ray with tangent_alt_km in {tangent_alt_window}",
                     ha="center", va="center", transform=ax_edp.transAxes)
        ax_edp.set_title("EDP profile")

    fig.suptitle(entry.label or entry.obs_type)
    fig.tight_layout()
    return fig


_STYLE_COLORS = {}


def _color_for_style(style: str) -> str:
    """Stable color per style across a run (cross-style comparison plots
    need the same style to always get the same color across every RO's
    figure, not whatever matplotlib's cycler happens to be on)."""
    if style not in _STYLE_COLORS:
        _STYLE_COLORS[style] = f"C{len(_STYLE_COLORS) % 10}"
    return _STYLE_COLORS[style]


def plot_cross_style_tec_edp_comparison(
    entry, tec_by_style: dict, edp_by_style: dict, altitude: np.ndarray, figsize=(12, 6),
):
    """One two-panel figure per RO (2026-09-29 user request), comparing
    *every* style at once rather than one style's own forecast/analysis --
    complements (does not replace) :func:`plot_tec_edp_profile_comparison`'s
    per-style plots. Same style always gets the same color across every
    RO's figure (:func:`_color_for_style`); forecast is solid, analysis is
    dashed, so the encoding stays legible even with several styles
    overlaid in one panel.

    Parameters
    ----------
    entry : ObservationEntry
        Used for ``tangent_alt_km`` (TEC panel y-axis) and ``abel``/
        ``label`` (EDP panel overlay/title).
    tec_by_style : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        style -> ``(y_forecast, y_analysis, y_measured)``, each already
        sliced to this entry's rays. ``y_measured`` is plotted once (from
        whichever style is listed first -- it's the same real data
        regardless of style).
    edp_by_style : dict[str, tuple[np.ndarray, np.ndarray] | None]
        style -> ``(forecast_profile, analysis_profile)`` (each
        ``(n_height, n_members)``, mean/std taken here) from
        :func:`resolve_edp_query_point_and_profiles`, or ``None`` for a
        style that had no ray in the EDP query window -- plotted as a
        per-style placeholder note rather than dropped silently.
    altitude : np.ndarray, shape (n_height,)
    """
    fig, (ax_tec, ax_edp) = plt.subplots(1, 2, figsize=figsize)

    order = np.argsort(entry.tangent_alt_km) if entry.tangent_alt_km is not None else None
    y_measured = None
    for style, (y_forecast, y_analysis, y_meas) in tec_by_style.items():
        color = _color_for_style(style)
        if y_measured is None:
            y_measured = y_meas
        alt = np.asarray(entry.tangent_alt_km)[order] if order is not None else np.arange(len(y_forecast))
        ax_tec.plot(np.asarray(y_forecast)[order] if order is not None else y_forecast, alt,
                    color=color, linestyle="-", marker="o", markersize=2, label=f"{style} forecast")
        ax_tec.plot(np.asarray(y_analysis)[order] if order is not None else y_analysis, alt,
                    color=color, linestyle="--", marker="o", markersize=2, label=f"{style} analysis")
    if y_measured is not None:
        alt = np.asarray(entry.tangent_alt_km)[order] if order is not None else np.arange(len(y_measured))
        ax_tec.plot(np.asarray(y_measured)[order] if order is not None else y_measured, alt,
                    color="k", linestyle="-", marker="o", markersize=3, label="measured", zorder=5)
    ax_tec.set_xlabel("TEC (TECU)")
    ax_tec.set_ylabel("Tangent altitude (km)")
    ax_tec.set_title("TEC profile (all styles)")
    ax_tec.legend(loc="best", fontsize=7)

    missing_styles = []
    for style, profiles in edp_by_style.items():
        color = _color_for_style(style)
        if profiles is None:
            missing_styles.append(style)
            continue
        forecast_profile, analysis_profile = profiles
        f_mean = forecast_profile.mean(axis=-1)
        a_mean = analysis_profile.mean(axis=-1)
        ax_edp.plot(f_mean, altitude, color=color, linestyle="-", label=f"{style} forecast")
        ax_edp.plot(a_mean, altitude, color=color, linestyle="--", label=f"{style} analysis")
    if entry.abel and "Ne" in entry.abel and "alt_km" in entry.abel:
        ax_edp.plot(entry.abel["Ne"], entry.abel["alt_km"], color="k", linestyle=":", label="Abel-retrieved")
    ax_edp.set_xlabel("Electron density (m$^{-3}$)")
    ax_edp.set_ylabel("Altitude (km)")
    title = "EDP profile (all styles)"
    if missing_styles:
        title += f"\n(no window ray: {', '.join(missing_styles)})"
    ax_edp.set_title(title)
    if any(p is not None for p in edp_by_style.values()):
        ax_edp.legend(loc="best", fontsize=7)

    fig.suptitle(entry.label or entry.obs_type)
    fig.tight_layout()
    return fig


def plot_style_comparison_summary(results_by_style: dict, fig=None, figsize=(11, 8)):
    """Cross-style performance summary (plan/docx step 9): final RMSE,
    state dimension, wall-clock time, and convergence rate, one bar chart
    each. ``results_by_style`` maps style name -> ``(CycleResult, n_state,
    wall_time_s)``."""
    styles = list(results_by_style.keys())
    final_rmse = [
        results_by_style[s][0].batch_outcomes[-1].rmse_reduction.rmse_analysis
        if results_by_style[s][0].batch_outcomes else float("nan")
        for s in styles
    ]
    n_state = [results_by_style[s][1] for s in styles]
    wall_time = [results_by_style[s][2] for s in styles]
    convergence_rate = [
        (sum(o.diagnostics.converged for o in results_by_style[s][0].batch_outcomes)
         / max(len(results_by_style[s][0].batch_outcomes), 1))
        for s in styles
    ]

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = np.asarray(fig.axes).ravel()

    for ax, values, title, ylabel in zip(
        axes,
        (final_rmse, n_state, wall_time, convergence_rate),
        ("Final analysis RMSE", "State dimension", "Wall-clock time", "Convergence rate"),
        ("RMSE (TECU)", "n_state", "seconds", "fraction of batches converged"),
    ):
        ax.bar(styles, values, color="C0")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Cross-style comparison")
    fig.tight_layout()
    return fig


def plot_observations_geolocation(edp_samples, entries, ax=None, figsize=(10, 8)):
    """
    RO tangent-point tracks (colored by tangent altitude) + IGS pierce
    points, drawn on the *same* map projection/extent as the horizontal
    grid plots (``edp_samples.plot_geolocation()``) -- replaces
    ``observation_preparation.diagnostics.plot_geolocation``'s
    Orthographic/Robinson view for this package's own artifact, since
    that view has no ``set_extent`` call and shows the whole visible
    hemisphere/globe, leaving the actual ROI content (this project's real
    data all sits within ~2000km of one point) squeezed into a small part
    of an otherwise-empty frame (2026-09-27 user report: "all the
    interesting items are plotted near the top of the plot"). Reuses
    ``edp_samples.plot_geolocation()`` itself (same grid vertices/mesh
    drawing, same projection/framing logic) and overlays observation
    geometry on top of it, rather than duplicating that framing logic.

    Parameters
    ----------
    edp_samples : EDPSamples
        Whatever grid this cycle used -- its own projection/extent is
        reused as-is.
    entries : list[ObservationEntry]
        Real RO+IGS entries (e.g. from ``package_run.py``'s
        ``_collect_unique_entries``).
    """
    import cartopy.crs as ccrs
    from TEC_model.podTc_file_processing import rayTangent, ECEFtolla

    ax = edp_samples.plot_geolocation(ax=ax, figsize=figsize)

    ro_entries = [e for e in entries if e.obs_type == "RO"]
    igs_entries = [e for e in entries if e.obs_type == "IGS"]

    if ro_entries:
        lat_parts, lon_parts, alt_parts = [], [], []
        for e in ro_entries:
            tangent_xyz, _, _ = rayTangent(
                np.asarray(e.rec_ecef_km, dtype=float), np.asarray(e.gnss_ecef_km, dtype=float), units="km",
            )
            lat, lon, _ = ECEFtolla(tangent_xyz)
            lat = np.asarray(lat, dtype=float).ravel()
            lon = np.asarray(lon, dtype=float).ravel()
            alt = (np.asarray(e.tangent_alt_km, dtype=float).ravel()
                   if e.tangent_alt_km is not None else np.full(lat.shape, np.nan))
            valid = np.isfinite(lat) & np.isfinite(lon)
            if np.any(valid):
                ax.plot(lon[valid], lat[valid], transform=ccrs.Geodetic(),
                        color="red", linewidth=1.0, alpha=0.6, zorder=3)
            lat_parts.append(lat)
            lon_parts.append(lon)
            alt_parts.append(alt)

        cat_lat, cat_lon, cat_alt = np.concatenate(lat_parts), np.concatenate(lon_parts), np.concatenate(alt_parts)
        valid = np.isfinite(cat_lat) & np.isfinite(cat_lon) & np.isfinite(cat_alt)
        if np.any(valid):
            sc = ax.scatter(cat_lon[valid], cat_lat[valid], transform=ccrs.PlateCarree(),
                             c=cat_alt[valid], s=10, zorder=4, label=f"RO tangent points (n={len(ro_entries)})")
            cb = ax.figure.colorbar(sc, ax=ax, pad=0.04, shrink=0.7)
            cb.set_label("RO tangent altitude (km)")

    if igs_entries:
        ipp_lat = np.concatenate([
            np.asarray(e.pierce_lat, dtype=float) if e.pierce_lat is not None else np.array([np.nan])
            for e in igs_entries
        ])
        ipp_lon = np.concatenate([
            np.asarray(e.pierce_lon, dtype=float) if e.pierce_lon is not None else np.array([np.nan])
            for e in igs_entries
        ])
        valid = np.isfinite(ipp_lat) & np.isfinite(ipp_lon)
        if np.any(valid):
            ax.scatter(ipp_lon[valid], ipp_lat[valid], transform=ccrs.PlateCarree(),
                       s=18, color="tab:blue", marker="^", zorder=5,
                       label=f"IGS pierce points (n={len(igs_entries)})")

    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Geolocation of prepared observations (n={len(entries)})")
    return ax


def plot_observation_operator_sum_combined(
    edp_samples, entries, altitudes=(100, 200, 300, 400, 500, 600, 700, 800),
    num_segments: int = 1000, fig=None, figsize=(14, 7), label: str = "all RO",
):
    """Same as ``plot_observation_operator_sum``, but sums the observation
    operator's rows over *every ray from every entry* combined (e.g. all
    RO occultations at once), rather than one entry at a time -- "where,
    cumulatively, does this whole RO dataset have sensitivity, and at what
    height" (2026-09-27 user request, complements the existing per-RO
    plots rather than replacing them)."""
    rec = np.concatenate([np.asarray(e.rec_ecef_km, dtype=np.float64) for e in entries], axis=1)
    gnss = np.concatenate([np.asarray(e.gnss_ecef_km, dtype=np.float64) for e in entries], axis=1)
    H_combined = edp_samples.get_observation_operator(
        {"rec_ecef_km": rec, "gnss_ecef_km": gnss}, num_segments=num_segments,
    )
    n_height = len(edp_samples.altitude)
    n_geo = edp_samples.geolocation.shape[0]
    summed = np.asarray(H_combined).sum(axis=0).reshape(n_height, n_geo)

    n_alt = len(altitudes)
    ncols = min(4, n_alt)
    nrows = int(np.ceil(n_alt / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                  subplot_kw={"projection": _cartopy_projection(edp_samples)})
    axes = np.asarray(fig.axes).ravel()

    for ax, target_alt in zip(axes, altitudes):
        alt_idx = int(np.argmin(np.abs(edp_samples.altitude - target_alt)))
        actual_alt = float(edp_samples.altitude[alt_idx])
        plot_horizontal(edp_samples, summed[alt_idx, :], target_alt=actual_alt, ax=ax,
                         scalar_label=f"sum(H) [{actual_alt:.0f} km]")
    for ax in axes[n_alt:]:
        ax.axis("off")

    total_rays = sum(e.n_rays for e in entries)
    fig.suptitle(f"Observation operator sum: {label} ({len(entries)} entries, {total_rays} rays)")
    fig.tight_layout()
    return fig


def plot_igs_tec_scatter(batches, result: CycleResult, style_label: str | None = None, figsize=(12, 6)):
    """Two-panel scatter (2026-09-27 user request): forecast TEC vs.
    measured TEC, and analysis TEC vs. measured TEC, for IGS rays only,
    pooled across every batch -- the IGS-side counterpart to the existing
    per-RO TEC/EDP profile comparison plots, which don't cover IGS. One
    call per style (pass ``style_label`` for the title).

    Parameters
    ----------
    batches : list[CycleBatch]
        Same batches (same order) passed to ``run_cycle`` for this style
        -- supplies each batch's ``obs_type`` to select IGS rays, since
        ``BatchOutcome`` itself doesn't carry that back-reference.
    result : CycleResult
        This style's own cycle result (``batch_outcomes`` zipped 1:1 with
        ``batches``).
    """
    forecast_parts, analysis_parts, measured_parts = [], [], []
    for batch, outcome in zip(batches, result.batch_outcomes):
        if batch.obs_type is None or outcome.y_forecast is None or outcome.y_analysis is None:
            continue
        igs_mask = np.asarray(batch.obs_type) == "IGS"
        if not np.any(igs_mask):
            continue
        forecast_parts.append(outcome.y_forecast[igs_mask])
        analysis_parts.append(outcome.y_analysis[igs_mask])
        measured_parts.append(outcome.y_obs_used[igs_mask])

    fig, (ax_f, ax_a) = plt.subplots(1, 2, figsize=figsize)
    if not measured_parts:
        for ax in (ax_f, ax_a):
            ax.text(0.5, 0.5, "no IGS observations in this cycle", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle(f"IGS forecast/analysis vs. measured TEC" + (f" -- {style_label}" if style_label else ""))
        fig.tight_layout()
        return fig

    measured = np.concatenate(measured_parts)
    forecast = np.concatenate(forecast_parts)
    analysis = np.concatenate(analysis_parts)

    for ax, predicted, panel_label in ((ax_f, forecast, "Forecast"), (ax_a, analysis, "Analysis")):
        ax.scatter(measured, predicted, s=18, alpha=0.6, color="C0")
        lo = float(min(measured.min(), predicted.min()))
        hi = float(max(measured.max(), predicted.max()))
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="1:1")
        ax.set_xlabel("Measured TEC (TECU)")
        ax.set_ylabel(f"{panel_label} TEC (TECU)")
        ax.set_title(f"{panel_label} vs. measured (n={len(measured)})")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("IGS forecast/analysis vs. measured TEC" + (f" -- {style_label}" if style_label else ""))
    fig.tight_layout()
    return fig
