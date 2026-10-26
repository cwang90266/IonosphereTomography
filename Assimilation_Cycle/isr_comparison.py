#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISR-vs-analysis comparison tooling (``ISR_Integration_Plan.md`` Section
4.3): compares the assimilation's forecast/analysis EDP at an ISR
station's location against real ISR-measured profiles.

Reads a preprocessed ISR netCDF file (e.g. ``TROISR2025_nonan.nc``)
directly, rather than ``validation_preparation.isr_profile_validation``'s
raw-Madrigal-file path -- that file is already time-sorted, site-fixed,
and gap-filled, so a direct nearest-time lookup is simpler and faster than
rescanning a directory of raw files (see the plan's Section 4.3 for the
full rationale).

**Station-agnostic by design (plan Section 2):** station identity
(name/lat/lon) is read from the ISR file's own global attres
(``station_name``/``station_latitude``/``station_longitude``), never
hardcoded -- this module works unmodified for a future non-Tromsø ISR
station's region as long as its file carries the same attrs
``TROISR2025_nonan.nc`` does.

**No dependency on the IRI2020 topside extension**: the comparison
functions below interpolate the *model's* forecast/analysis profile (on
its own fine, full-range altitude grid) down onto the ISR instrument's
native altitude gates -- not the other way around -- so no extrapolation
of real ISR measurement above its native ceiling is ever implied by this
module. (The separate ISR-PCA parameterization work, plan Section 4.1,
does need the IRI2020 extension to build a basis on the production grid;
this comparison tooling does not.)

Two complementary ways to compare against ISR profiles (plan Section
4.3), both per-style and cross-style:
  - "Plot A" (nearest point): the single ISR profile nearest each batch's
    midpoint, overlaid with that batch's forecast/analysis mean +/- std.
  - "Plot B" (cycle-wide range): the *range* of every ISR profile across
    the cycle's entire observation window, overlaid with every batch's
    analysis profile -- shows whether the sequence of analyses tracks
    within real observed ISR variability over the whole period.
A pooled RMSE-by-altitude-gate table is the quantitative companion to
Plot A (Plot B is deliberately qualitative -- a range isn't a point
estimate to score against).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from .cycle_driver import CycleBatch
from .output import resolve_edp_query_point_and_profiles, _color_for_style


@dataclass(frozen=True)
class IsrStation:
    name: str
    lat: float
    lon: float


@dataclass
class IsrDataset:
    """One ISR site's full preprocessed record -- ``density`` on the
    instrument's own native (generally irregular) altitude gates, not
    resampled onto any model grid (see module docstring)."""

    station: IsrStation
    time: pd.DatetimeIndex          # (n_time,) ascending
    altitude: np.ndarray             # (n_alt,) native gates, km, ascending
    density: np.ndarray              # (n_alt, n_time) m^-3


def load_isr_dataset(path: str | Path) -> IsrDataset:
    """Read a preprocessed ISR netCDF (``TROISR2025_nonan.nc``'s schema:
    ``altitude``/``Ne``/``time_utc`` data variables, ``station_name``/
    ``station_latitude``/``station_longitude`` global attrs). Station
    identity comes from the file's own attrs -- never hardcoded -- so
    this works unmodified for any future ISR station's file with the same
    schema."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        altitude = np.asarray(ds["altitude"].values, dtype=float)
        density = np.asarray(ds["Ne"].values, dtype=float)
        time = pd.to_datetime(np.asarray(ds["time_utc"].values))
        station = IsrStation(
            name=str(ds.attrs.get("station_name", "ISR station")),
            lat=float(ds.attrs["station_latitude"]),
            lon=float(ds.attrs["station_longitude"]),
        )

    order = np.argsort(np.asarray(time))
    alt_order = np.argsort(altitude)
    return IsrDataset(
        station=station,
        time=pd.DatetimeIndex(time)[order],
        altitude=altitude[alt_order],
        density=density[alt_order, :][:, order],
    )


def nearest_isr_profile(isr: IsrDataset, target_time) -> tuple[pd.Timestamp, np.ndarray] | None:
    """Nearest-in-time ISR profile to ``target_time`` -- returns
    ``(timestamp, density)`` on ``isr.altitude``'s native gates, or
    ``None`` if ``isr`` has no profiles at all."""
    if len(isr.time) == 0:
        return None
    target = pd.Timestamp(target_time)
    deltas = np.abs((isr.time - target).to_numpy())
    idx = int(np.argmin(deltas))
    return isr.time[idx], isr.density[:, idx]


def isr_profiles_in_window(isr: IsrDataset, start_time, end_time) -> np.ndarray:
    """All ISR profiles (native gates) whose timestamp falls in
    ``[start_time, end_time]`` -- shape ``(n_alt, n_in_window)``, possibly
    empty in the second axis if nothing falls in the window."""
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    mask = (isr.time >= start) & (isr.time <= end)
    return isr.density[:, np.asarray(mask)]


def batch_midpoint(batch: CycleBatch) -> pd.Timestamp | None:
    if batch.start_time is None or batch.end_time is None:
        return None
    t0, t1 = pd.Timestamp(batch.start_time), pd.Timestamp(batch.end_time)
    return t0 + (t1 - t0) / 2


def _interp_log10(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``np.interp`` in log10 space (matches this codebase's standing
    log-space convention for density interpolation); points in ``x_new``
    outside ``[x.min(), x.max()]`` come back ``nan`` -- no extrapolation,
    see module docstring."""
    log_y = np.log10(np.clip(y, 1.0, None))
    log_interp = np.interp(x_new, x, log_y, left=np.nan, right=np.nan)
    return 10.0 ** log_interp


# ===========================================================================
# Per-batch collection (one style at a time; package_run.py's on_batch
# callback pattern -- see cross_style_data in package_run.py for the
# existing analogous mechanism).
# ===========================================================================


@dataclass
class BatchIsrMatch:
    """Everything one batch needs for the ISR comparison plots/metrics:
    the nearest ISR profile to the batch's midpoint, and the model's
    forecast/analysis ensemble profile interpolated to the ISR station's
    (lat, lon)."""

    batch_index: int
    isr_time: pd.Timestamp | None
    isr_density: np.ndarray | None     # (n_isr_alt,) native gates, or None (no ISR coverage near this batch)
    forecast_profile: np.ndarray       # (n_height, n_members), model altitude grid
    analysis_profile: np.ndarray       # (n_height, n_members), model altitude grid


class IsrComparisonCollector:
    """Accumulates one style's :class:`BatchIsrMatch` list across a cycle,
    via ``run_batch_loop``'s ``on_batch`` callback -- same wiring pattern
    ``package_run.py`` already uses for its ``cross_style_data`` dict.

    Usage::

        collector = IsrComparisonCollector(edp_samples, isr)
        run_cycle(cfg, edp_samples, parameterization, batches, ensemble_prior,
                  on_batch=collector.on_batch)
        # collector.matches now has one BatchIsrMatch per batch
    """

    def __init__(self, edp_samples, isr: IsrDataset):
        self.edp_samples = edp_samples
        self.isr = isr
        self.matches: list[BatchIsrMatch] = []

    def on_batch(self, batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome):
        decoded_forecast = obs_operator.decode(ensemble_forecast.to_param_shape())
        decoded_analysis = obs_operator.decode(ensemble_analysis.to_param_shape())
        _, _, forecast_profile, analysis_profile = resolve_edp_query_point_and_profiles(
            self.edp_samples, None, decoded_forecast, decoded_analysis,
            lat=self.isr.station.lat, lon=self.isr.station.lon,
        )

        mid = batch_midpoint(batch)
        nearest = nearest_isr_profile(self.isr, mid) if mid is not None else None

        self.matches.append(BatchIsrMatch(
            batch_index=batch.batch_index,
            isr_time=nearest[0] if nearest is not None else None,
            isr_density=nearest[1] if nearest is not None else None,
            forecast_profile=forecast_profile,
            analysis_profile=analysis_profile,
        ))


# ===========================================================================
# Plot type A: per-batch nearest-point overlay.
# ===========================================================================


def plot_isr_nearest_point_comparison(
    match: BatchIsrMatch, model_altitude: np.ndarray, isr_altitude: np.ndarray,
    station_name: str = "ISR", style: str | None = None, ax=None, figsize=(6, 7),
):
    """One style's forecast/analysis mean +/- std (model altitude grid)
    overlaid with the nearest real ISR profile (native altitude gates),
    for one batch. ``style``, if given, is shown in the title -- the
    per-style plots otherwise carry no on-figure indication of which
    style they're from (only their file path does)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    f_mean, f_std = match.forecast_profile.mean(axis=-1), match.forecast_profile.std(axis=-1)
    a_mean, a_std = match.analysis_profile.mean(axis=-1), match.analysis_profile.std(axis=-1)

    ax.plot(f_mean, model_altitude, color="C0", label="forecast mean")
    ax.fill_betweenx(model_altitude, f_mean - f_std, f_mean + f_std, color="C0", alpha=0.2)
    ax.plot(a_mean, model_altitude, color="C1", label="analysis mean")
    ax.fill_betweenx(model_altitude, a_mean - a_std, a_mean + a_std, color="C1", alpha=0.2)

    if match.isr_density is not None:
        ax.plot(match.isr_density, isr_altitude, "o-", color="k", markersize=3,
                 label=f"{station_name} ISR ({match.isr_time:%Y-%m-%d %H:%M})")
    else:
        ax.text(0.5, 0.02, "no ISR profile near this batch", ha="center", va="bottom",
                 transform=ax.transAxes, fontsize=8)

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    title = f"Batch {match.batch_index}: EDP vs. nearest ISR profile"
    if style:
        title = f"{style} -- {title}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_isr_nearest_point_comparison_cross_style(
    matches_by_style: dict[str, BatchIsrMatch], model_altitude: np.ndarray, isr_altitude: np.ndarray,
    station_name: str = "ISR", ax=None, figsize=(7, 8),
):
    """Cross-style version of :func:`plot_isr_nearest_point_comparison`:
    every style's analysis mean overlaid (forecast dropped from this view
    to keep it legible -- the per-style plot already covers forecast),
    for one batch. Same ISR profile shown once (identical real data
    regardless of style)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    isr_density, isr_time, batch_index = None, None, None
    for style, match in matches_by_style.items():
        color = _color_for_style(style)
        a_mean = match.analysis_profile.mean(axis=-1)
        ax.plot(a_mean, model_altitude, color=color, label=f"{style} analysis")
        if isr_density is None and match.isr_density is not None:
            isr_density, isr_time = match.isr_density, match.isr_time
        batch_index = match.batch_index

    if isr_density is not None:
        ax.plot(isr_density, isr_altitude, "o-", color="k", markersize=3,
                 label=f"{station_name} ISR ({isr_time:%Y-%m-%d %H:%M})", zorder=5)
    else:
        ax.text(0.5, 0.02, "no ISR profile near this batch", ha="center", va="bottom",
                 transform=ax.transAxes, fontsize=8)

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"Batch {batch_index}: analysis EDP vs. nearest ISR profile (all styles)")
    ax.legend(loc="best", fontsize=7)
    return ax


# ===========================================================================
# Plot type B: cycle-wide ISR range.
# ===========================================================================


def plot_isr_range_comparison(
    matches: list[BatchIsrMatch], isr: IsrDataset, cycle_start, cycle_end,
    model_altitude: np.ndarray, percentile: float = 0.0, style: str | None = None,
    ax=None, figsize=(7, 8),
):
    """The range of every real ISR profile across the cycle's full
    ``[cycle_start, cycle_end]`` observation window (shaded band, native
    ISR altitude gates), overlaid with every batch's analysis mean
    profile (one line per batch, model altitude grid) -- shows whether
    the sequence of analyses tracks within real observed ISR variability
    over the whole period (one style's own batches; see the cross-style
    variant for all styles at once).

    Parameters
    ----------
    percentile : float
        ``0.0`` (default): shade the literal min-max range. A value in
        ``(0, 50)`` instead shades the ``[percentile, 100-percentile]``
        band -- less sensitive to a single outlier scan, at the cost of
        no longer being a literal bound.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    window = isr_profiles_in_window(isr, cycle_start, cycle_end)
    if window.shape[1] == 0:
        ax.text(0.5, 0.5, "no ISR coverage in this cycle's window", ha="center", va="center",
                 transform=ax.transAxes)
    else:
        if percentile > 0:
            lo = np.nanpercentile(window, percentile, axis=1)
            hi = np.nanpercentile(window, 100 - percentile, axis=1)
        else:
            lo = np.nanmin(window, axis=1)
            hi = np.nanmax(window, axis=1)
        ax.fill_betweenx(isr.altitude, lo, hi, color="k", alpha=0.15,
                          label=f"ISR range (n={window.shape[1]})")

    cmap = plt.get_cmap("viridis")
    n = max(len(matches), 1)
    for i, match in enumerate(sorted(matches, key=lambda m: m.batch_index)):
        a_mean = match.analysis_profile.mean(axis=-1)
        ax.plot(a_mean, model_altitude, color=cmap(i / max(n - 1, 1)), alpha=0.85,
                 label=f"batch {match.batch_index} analysis")

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    title_prefix = f"{style} -- " if style else ""
    ax.set_title(f"{title_prefix}Analysis EDP vs. ISR range over the cycle\n{isr.station.name}, "
                 f"{pd.Timestamp(cycle_start)} - {pd.Timestamp(cycle_end)}")
    ax.legend(loc="best", fontsize=6, ncol=2)
    return ax


def plot_isr_range_comparison_cross_style(
    matches_by_style: dict[str, list[BatchIsrMatch]], isr: IsrDataset, cycle_start, cycle_end,
    model_altitude: np.ndarray, percentile: float = 0.0, ax=None, figsize=(7, 8),
):
    """Cross-style version of :func:`plot_isr_range_comparison`: all
    styles' per-batch analysis lines over the same ISR range band,
    color-coded by style (not by batch) -- the most direct visual answer
    to "which style's analysis best tracks the real ISR-observed range
    over the whole cycle"."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    window = isr_profiles_in_window(isr, cycle_start, cycle_end)
    if window.shape[1] == 0:
        ax.text(0.5, 0.5, "no ISR coverage in this cycle's window", ha="center", va="center",
                 transform=ax.transAxes)
    else:
        if percentile > 0:
            lo = np.nanpercentile(window, percentile, axis=1)
            hi = np.nanpercentile(window, 100 - percentile, axis=1)
        else:
            lo = np.nanmin(window, axis=1)
            hi = np.nanmax(window, axis=1)
        ax.fill_betweenx(isr.altitude, lo, hi, color="k", alpha=0.15,
                          label=f"ISR range (n={window.shape[1]})")

    for style, matches in matches_by_style.items():
        color = _color_for_style(style)
        for j, match in enumerate(sorted(matches, key=lambda m: m.batch_index)):
            a_mean = match.analysis_profile.mean(axis=-1)
            ax.plot(a_mean, model_altitude, color=color, alpha=0.7,
                     label=f"{style} analysis" if j == 0 else None)

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"Analysis EDP vs. ISR range over the cycle (all styles)\n{isr.station.name}, "
                 f"{pd.Timestamp(cycle_start)} - {pd.Timestamp(cycle_end)}")
    ax.legend(loc="best", fontsize=7)
    return ax


# ===========================================================================
# Pooled RMSE by altitude gate -- the quantitative companion to Plot A.
# ===========================================================================


def pooled_isr_rmse_by_altitude(
    matches: list[BatchIsrMatch], model_altitude: np.ndarray, isr_altitude: np.ndarray,
) -> pd.DataFrame:
    """RMSE (forecast and analysis, vs. the matched nearest ISR profile),
    pooled across every batch with ISR coverage, one row per native ISR
    altitude gate (exact -- every profile in a given ISR file shares the
    same fixed per-gate altitude, so no re-binning is needed, unlike
    ``validation_preparation.isr_profile_validation``'s raw-file path,
    which has no such guarantee)."""
    sq_forecast = np.zeros_like(isr_altitude)
    sq_analysis = np.zeros_like(isr_altitude)
    n_pairs = np.zeros_like(isr_altitude, dtype=int)

    for match in matches:
        if match.isr_density is None:
            continue
        f_mean = match.forecast_profile.mean(axis=-1)
        a_mean = match.analysis_profile.mean(axis=-1)
        f_at_isr = _interp_log10(isr_altitude, model_altitude, f_mean)
        a_at_isr = _interp_log10(isr_altitude, model_altitude, a_mean)

        valid = np.isfinite(f_at_isr) & np.isfinite(a_at_isr) & np.isfinite(match.isr_density)
        sq_forecast[valid] += (f_at_isr[valid] - match.isr_density[valid]) ** 2
        sq_analysis[valid] += (a_at_isr[valid] - match.isr_density[valid]) ** 2
        n_pairs[valid] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        rmse_forecast = np.sqrt(sq_forecast / np.where(n_pairs > 0, n_pairs, 1))
        rmse_analysis = np.sqrt(sq_analysis / np.where(n_pairs > 0, n_pairs, 1))
    rmse_forecast = np.where(n_pairs > 0, rmse_forecast, np.nan)
    rmse_analysis = np.where(n_pairs > 0, rmse_analysis, np.nan)

    return pd.DataFrame({
        "altitude_km": isr_altitude,
        "rmse_forecast_m3": rmse_forecast,
        "rmse_analysis_m3": rmse_analysis,
        "n_pairs": n_pairs,
    })


def pooled_isr_rmse_by_altitude_cross_style(
    matches_by_style: dict[str, list[BatchIsrMatch]], model_altitude: np.ndarray, isr_altitude: np.ndarray,
) -> pd.DataFrame:
    """Cross-style table: one :func:`pooled_isr_rmse_by_altitude` per
    style, concatenated with a ``style`` column for side-by-side
    comparison."""
    frames = []
    for style, matches in matches_by_style.items():
        frame = pooled_isr_rmse_by_altitude(matches, model_altitude, isr_altitude)
        frame.insert(0, "style", style)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_pooled_isr_rmse(df: pd.DataFrame, style: str | None = None, ax=None, figsize=(6, 7)):
    """Line plot of :func:`pooled_isr_rmse_by_altitude`'s output (one
    style)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(df["rmse_forecast_m3"], df["altitude_km"], "o-", color="C0", markersize=3, label="forecast RMSE")
    ax.plot(df["rmse_analysis_m3"], df["altitude_km"], "o-", color="C1", markersize=3, label="analysis RMSE")
    ax.set_xlabel("RMSE vs. ISR (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    title_prefix = f"{style} -- " if style else ""
    ax.set_title(f"{title_prefix}Pooled RMSE vs. ISR, by altitude gate")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_pooled_isr_rmse_cross_style(df: pd.DataFrame, ax=None, figsize=(7, 8)):
    """Cross-style line plot of :func:`pooled_isr_rmse_by_altitude_cross_style`'s
    output -- analysis RMSE only (one line per style), the direct
    side-by-side answer to "which style matches ISR best, at which
    altitude"."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    for style, group in df.groupby("style"):
        color = _color_for_style(style)
        ax.plot(group["rmse_analysis_m3"], group["altitude_km"], "o-", color=color, markersize=3, label=style)
    ax.set_xlabel("Analysis RMSE vs. ISR (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Pooled analysis RMSE vs. ISR, by altitude gate (all styles)")
    ax.legend(loc="best", fontsize=8)
    return ax
