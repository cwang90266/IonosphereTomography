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

**Richer ISR files + ionosonde cross-check (2026-11-19 user request):**
some preprocessed ISR files (e.g. a newer ``MILISR2025_nonan.nc``) carry,
beyond the hourly-median ``Ne``, the per-(altitude, hour) ``Ne_min``/
``Ne_max``/``profiles_used_per_gate`` of the raw profiles that went into
that median. ``load_isr_dataset`` reads these when present
(``IsrDataset.density_min``/``density_max``/``n_profiles``, ``None``
otherwise -- every function below degrades to its original behavior for
an older file lacking them). The min/max bound the true value; treating
the ``n`` raw profiles as independent samples symmetric about the truth,
all ``n`` landing on the same side of it has probability ``2 * 0.5**n``,
so the min/max bound it with confidence ``1 - 2**(1-n)``
(:func:`_confidence`) -- the comparison plots shade this band with
opacity proportional to that confidence (:func:`_draw_isr_confidence_band`).

A companion ionosonde file (:class:`IonosondeDataset`,
:func:`load_ionosonde_dataset`) gives independent ``nmF2``/``hmF2``
time series -- an independent calibration check on the ISR profile's
F2-peak density/height, plotted as red ``(nmF2, hmF2)`` dots wherever a
record falls inside the window being compared
(:func:`ionosonde_points_in_window`).

**Post-processing without a pipeline rerun** (bottom of this module):
``GenericObservationOperator.decode()`` needs only a parameterization and
the ensemble's own param-shape state -- no ray geometry/``H_grid`` at all
-- and ``package_run.py`` already unconditionally saves every batch's
analysis ensemble plus each style's fitted PCA basis. So an already
completed run's forecast/analysis density at the ISR station is fully
reconstructible from files already on disk (batch 0's forecast is the
deterministic, RNG-free ``EnsembleState.from_parameterized_edp_samples``;
batch i>0's forecast is batch i-1's saved analysis), letting
:func:`replot_isr_comparison_tree` regenerate every ISR-comparison plot
under an existing run tree against a newer ISR/ionosonde file, with no
EnKF solve at all -- same "save once, replot forever" philosophy as
``impact_vectors.save_impact_vectors``/``replot_impact_vectors_from_saved``.
"""

from __future__ import annotations

import json
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
import matplotlib.colors

from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.observation_operator import GenericObservationOperator

from .cycle_driver import CycleBatch
from .ensemble_io import load_ensemble_netcdf
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
    resampled onto any model grid (see module docstring).

    ``density_min``/``density_max``/``n_profiles`` are the per-(altitude,
    hour) extrema/count of the raw profiles behind each hourly median --
    ``None`` when the source file doesn't carry them (older files, e.g.
    ``TROISR2025_nonan.nc``), in which case every confidence-shading code
    path below falls back to its pre-2026-11-19 behavior."""

    station: IsrStation
    time: pd.DatetimeIndex          # (n_time,) ascending
    altitude: np.ndarray             # (n_alt,) native gates, km, ascending
    density: np.ndarray              # (n_alt, n_time) m^-3
    density_min: np.ndarray | None = None    # (n_alt, n_time) m^-3, or None
    density_max: np.ndarray | None = None    # (n_alt, n_time) m^-3, or None
    n_profiles: np.ndarray | None = None     # (n_alt, n_time) int, or None


def load_isr_dataset(path: str | Path) -> IsrDataset:
    """Read a preprocessed ISR netCDF (``TROISR2025_nonan.nc``'s schema:
    ``altitude``/``Ne``/``time_utc`` data variables, ``station_name``/
    ``station_latitude``/``station_longitude`` global attrs). Station
    identity comes from the file's own attrs -- never hardcoded -- so
    this works unmodified for any future ISR station's file with the same
    schema.

    Also reads ``Ne_min``/``Ne_max``/``profiles_used_per_gate`` when
    present (module docstring) -- absent from older files, in which case
    ``IsrDataset.density_min``/``density_max``/``n_profiles`` come back
    ``None``."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        altitude = np.asarray(ds["altitude"].values, dtype=float)
        density = np.asarray(ds["Ne"].values, dtype=float)
        time = pd.to_datetime(np.asarray(ds["time_utc"].values))
        density_min = np.asarray(ds["Ne_min"].values, dtype=float) if "Ne_min" in ds else None
        density_max = np.asarray(ds["Ne_max"].values, dtype=float) if "Ne_max" in ds else None
        n_profiles = (
            np.asarray(ds["profiles_used_per_gate"].values, dtype=float)
            if "profiles_used_per_gate" in ds else None
        )
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
        density_min=density_min[alt_order, :][:, order] if density_min is not None else None,
        density_max=density_max[alt_order, :][:, order] if density_max is not None else None,
        n_profiles=n_profiles[alt_order, :][:, order] if n_profiles is not None else None,
    )


@dataclass
class IonosondeDataset:
    """One ionosonde station's NmF2/hmF2 time series -- an independent
    (different-instrument) calibration check on the ISR profile's F2-peak
    density/height (module docstring). ``nmF2``/``hmF2`` may contain
    ``nan`` at timestamps with no valid measurement."""

    station: IsrStation
    time: pd.DatetimeIndex   # (n_time,) ascending
    nmF2: np.ndarray          # (n_time,) m^-3, may contain nan
    hmF2: np.ndarray          # (n_time,) km, may contain nan


def load_ionosonde_dataset(path: str | Path) -> IonosondeDataset:
    """Read a preprocessed ionosonde netCDF (``MILIONO_Aug2530_2025.nc``'s
    schema: ``nmF2``/``hmF2``/``time_utc`` data variables, same
    ``station_name``/``station_latitude``/``station_longitude`` global
    attrs convention as :func:`load_isr_dataset`)."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        nmF2 = np.asarray(ds["nmF2"].values, dtype=float)
        hmF2 = np.asarray(ds["hmF2"].values, dtype=float)
        time = pd.to_datetime(np.asarray(ds["time_utc"].values))
        station = IsrStation(
            name=str(ds.attrs.get("station_name", "ionosonde station")),
            lat=float(ds.attrs["station_latitude"]),
            lon=float(ds.attrs["station_longitude"]),
        )

    order = np.argsort(np.asarray(time))
    return IonosondeDataset(
        station=station,
        time=pd.DatetimeIndex(time)[order],
        nmF2=nmF2[order],
        hmF2=hmF2[order],
    )


def ionosonde_points_in_window(
    iono: IonosondeDataset | None, start_time, end_time,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite ``(nmF2, hmF2)`` pairs whose timestamp falls in
    ``[start_time, end_time]`` -- ``iono=None`` or no matching finite
    records returns two empty arrays, never ``None``, so callers can
    unconditionally check ``len(...)``."""
    if iono is None:
        return np.array([]), np.array([])
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    in_window = (iono.time >= start) & (iono.time <= end)
    finite = np.isfinite(iono.nmF2) & np.isfinite(iono.hmF2)
    mask = np.asarray(in_window) & finite
    return iono.nmF2[mask], iono.hmF2[mask]


def _confidence(n) -> np.ndarray:
    """Confidence that ``[min, max]`` of ``n`` raw samples bounds the true
    (symmetric-about-it) value: all ``n`` landing on the same side of the
    truth has probability ``2 * 0.5**n``, so the bound holds with
    confidence ``1 - 2**(1-n)`` (module docstring). Clipped to ``[0, 1]``
    (``n<=1`` gives ``<=0``, i.e. no information from a single sample)."""
    n = np.asarray(n, dtype=float)
    return np.clip(1.0 - 2.0 ** (1.0 - n), 0.0, 1.0)


def _draw_isr_confidence_band(
    ax, altitude: np.ndarray, lo: np.ndarray, hi: np.ndarray, n: np.ndarray | None = None,
    max_alpha: float = 0.35, color: str = "0.35", label: str | None = None,
):
    """Per-altitude-gate shaded band from ``lo`` to ``hi``; opacity
    proportional to :func:`_confidence(n)` when ``n`` is given (new-schema
    ISR file), else a flat ``max_alpha`` (old-schema fallback -- today's
    literal-range behavior, unchanged). Drawn at ``zorder=0`` so it sits
    behind every other element already on ``ax``. No-op if ``lo``/``hi``
    is ``None``."""
    if lo is None or hi is None:
        return
    altitude = np.asarray(altitude, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    valid = np.isfinite(altitude) & np.isfinite(lo) & np.isfinite(hi)
    if n is not None:
        conf = _confidence(n)
        valid = valid & np.isfinite(conf)
        alphas = max_alpha * conf[valid]
    else:
        alphas = np.full(int(np.count_nonzero(valid)), max_alpha)
    if not np.any(valid):
        return
    # Matplotlib's bar/barh `alpha=` only accepts one scalar for the whole
    # call (confirmed against the installed 3.10.6: a per-element list
    # raises `TypeError: alpha must be numeric or None`) -- per-bar opacity
    # has to be baked into each bar's own RGBA `color` instead.
    base_rgb = matplotlib.colors.to_rgb(color)
    colors = [(*base_rgb, float(a)) for a in alphas]
    alt_sorted = np.sort(altitude[np.isfinite(altitude)])
    height = float(np.median(np.diff(alt_sorted))) if len(alt_sorted) > 1 else 1.0
    bars = ax.barh(altitude[valid], hi[valid] - lo[valid], left=lo[valid], height=height,
                    color=colors, linewidth=0, zorder=0)
    if label is not None and len(bars):
        bars[0].set_label(label)


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


def nearest_ionosonde_point(
    iono: IonosondeDataset | None, target_time,
) -> tuple[pd.Timestamp, float, float] | None:
    """Nearest-in-time ``(time, nmF2, hmF2)`` with both finite, or ``None``
    if ``iono`` is ``None`` or has no finite point at all -- the ionosonde
    analog of :func:`nearest_isr_profile`, for picking one representative
    value (e.g. for a tabulated row) instead of
    :func:`ionosonde_points_in_window`'s whole-window array."""
    if iono is None:
        return None
    finite = np.isfinite(iono.nmF2) & np.isfinite(iono.hmF2)
    if not np.any(finite):
        return None
    target = pd.Timestamp(target_time)
    idx_finite = np.where(finite)[0]
    deltas = np.abs((iono.time[idx_finite] - target).to_numpy())
    idx = idx_finite[int(np.argmin(deltas))]
    return iono.time[idx], float(iono.nmF2[idx]), float(iono.hmF2[idx])


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
    (lat, lon).

    ``isr_density_min``/``isr_density_max``/``isr_n_profiles`` (same
    native gates as ``isr_density``, same matched time) and
    ``ionosonde_nmF2``/``ionosonde_hmF2`` (finite ionosonde records inside
    this batch's own ``[start_time, end_time]``) are ``None``/empty when
    the richer ISR file or an ionosonde file wasn't supplied -- see module
    docstring."""

    batch_index: int
    isr_time: pd.Timestamp | None
    isr_density: np.ndarray | None     # (n_isr_alt,) native gates, or None (no ISR coverage near this batch)
    forecast_profile: np.ndarray       # (n_height, n_members), model altitude grid
    analysis_profile: np.ndarray       # (n_height, n_members), model altitude grid
    isr_density_min: np.ndarray | None = None    # (n_isr_alt,) or None
    isr_density_max: np.ndarray | None = None    # (n_isr_alt,) or None
    isr_n_profiles: np.ndarray | None = None      # (n_isr_alt,) or None
    ionosonde_nmF2: np.ndarray | None = None      # (n_points,) m^-3, within this batch's window
    ionosonde_hmF2: np.ndarray | None = None      # (n_points,) km, within this batch's window


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

    def __init__(self, edp_samples, isr: IsrDataset, ionosonde: IonosondeDataset | None = None):
        self.edp_samples = edp_samples
        self.isr = isr
        self.ionosonde = ionosonde
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
        isr_time = nearest[0] if nearest is not None else None
        isr_density = nearest[1] if nearest is not None else None

        isr_density_min = isr_density_max = isr_n_profiles = None
        if isr_time is not None and self.isr.density_min is not None:
            t_idx = int(np.argmin(np.abs((self.isr.time - isr_time).to_numpy())))
            isr_density_min = self.isr.density_min[:, t_idx]
            isr_density_max = self.isr.density_max[:, t_idx]
            if self.isr.n_profiles is not None:
                isr_n_profiles = self.isr.n_profiles[:, t_idx]

        ionosonde_nmF2, ionosonde_hmF2 = (
            ionosonde_points_in_window(self.ionosonde, batch.start_time, batch.end_time)
            if self.ionosonde is not None and batch.start_time is not None and batch.end_time is not None
            else (None, None)
        )

        self.matches.append(BatchIsrMatch(
            batch_index=batch.batch_index,
            isr_time=isr_time,
            isr_density=isr_density,
            forecast_profile=forecast_profile,
            analysis_profile=analysis_profile,
            isr_density_min=isr_density_min,
            isr_density_max=isr_density_max,
            isr_n_profiles=isr_n_profiles,
            ionosonde_nmF2=ionosonde_nmF2,
            ionosonde_hmF2=ionosonde_hmF2,
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

    _draw_isr_confidence_band(
        ax, isr_altitude, match.isr_density_min, match.isr_density_max, match.isr_n_profiles,
    )

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

    if match.ionosonde_nmF2 is not None and len(match.ionosonde_nmF2):
        ax.scatter(match.ionosonde_nmF2, match.ionosonde_hmF2, color="red", marker="o",
                    s=28, zorder=6, label="ionosonde NmF2 @ hmF2")

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
    isr_density_min = isr_density_max = isr_n_profiles = None
    ionosonde_nmF2 = ionosonde_hmF2 = None
    for style, match in matches_by_style.items():
        color = _color_for_style(style)
        a_mean = match.analysis_profile.mean(axis=-1)
        ax.plot(a_mean, model_altitude, color=color, label=f"{style} analysis")
        if isr_density is None and match.isr_density is not None:
            isr_density, isr_time = match.isr_density, match.isr_time
            isr_density_min, isr_density_max = match.isr_density_min, match.isr_density_max
            isr_n_profiles = match.isr_n_profiles
        if (ionosonde_nmF2 is None or not len(ionosonde_nmF2)) and match.ionosonde_nmF2 is not None:
            ionosonde_nmF2, ionosonde_hmF2 = match.ionosonde_nmF2, match.ionosonde_hmF2
        batch_index = match.batch_index

    _draw_isr_confidence_band(ax, isr_altitude, isr_density_min, isr_density_max, isr_n_profiles)

    if isr_density is not None:
        ax.plot(isr_density, isr_altitude, "o-", color="k", markersize=3,
                 label=f"{station_name} ISR ({isr_time:%Y-%m-%d %H:%M})", zorder=5)
    else:
        ax.text(0.5, 0.02, "no ISR profile near this batch", ha="center", va="bottom",
                 transform=ax.transAxes, fontsize=8)

    if ionosonde_nmF2 is not None and len(ionosonde_nmF2):
        ax.scatter(ionosonde_nmF2, ionosonde_hmF2, color="red", marker="o",
                    s=28, zorder=6, label="ionosonde NmF2 @ hmF2")

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"Batch {batch_index}: analysis EDP vs. nearest ISR profile (all styles)")
    ax.legend(loc="best", fontsize=7)
    return ax


# ===========================================================================
# Plot type B: cycle-wide ISR range.
# ===========================================================================


def _isr_window_bounds_and_confidence(
    isr: IsrDataset, start_time, end_time, percentile: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Per-altitude-gate ``(lo, hi, n)`` for the band
    :func:`plot_isr_range_comparison`/its cross-style variant shade.

    When ``isr.density_min``/``density_max``/``n_profiles`` are available
    (richer file), pools them across every hour in the window: ``lo`` is
    the min of every hour's ``Ne_min``, ``hi`` the max of every hour's
    ``Ne_max``, ``n`` the sum of every hour's raw-profile count -- the
    literal bound a reader would get by pooling every raw profile in the
    window, with confidence drawn from the total sample size. Falls back
    to the literal min/max of the hourly-median ``Ne`` itself (``n=None``,
    today's pre-2026-11-19 behavior) when the richer fields aren't there.
    ``percentile`` only applies to that fallback (kept for backward
    compatibility with existing callers; the richer path's ``lo``/``hi``
    are already literal bounds, not a percentile of medians)."""
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    mask = np.asarray((isr.time >= start) & (isr.time <= end))
    if not mask.any():
        return None, None, None

    if isr.density_min is not None and isr.density_max is not None:
        lo = np.nanmin(isr.density_min[:, mask], axis=1)
        hi = np.nanmax(isr.density_max[:, mask], axis=1)
        n = (
            np.nansum(isr.n_profiles[:, mask], axis=1)
            if isr.n_profiles is not None else None
        )
        return lo, hi, n

    window = isr.density[:, mask]
    if percentile > 0:
        lo = np.nanpercentile(window, percentile, axis=1)
        hi = np.nanpercentile(window, 100 - percentile, axis=1)
    else:
        lo = np.nanmin(window, axis=1)
        hi = np.nanmax(window, axis=1)
    return lo, hi, None


def plot_isr_range_comparison(
    matches: list[BatchIsrMatch], isr: IsrDataset, cycle_start, cycle_end,
    model_altitude: np.ndarray, percentile: float = 0.0, style: str | None = None,
    ax=None, figsize=(7, 8), ionosonde: IonosondeDataset | None = None,
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
    lo, hi, n_profiles = _isr_window_bounds_and_confidence(isr, cycle_start, cycle_end, percentile)
    if lo is None:
        ax.text(0.5, 0.5, "no ISR coverage in this cycle's window", ha="center", va="center",
                 transform=ax.transAxes)
    else:
        _draw_isr_confidence_band(ax, isr.altitude, lo, hi, n_profiles,
                                   max_alpha=0.35 if n_profiles is not None else 0.15,
                                   label=f"ISR range (n={window.shape[1]} hours)")

    ionosonde_nmF2, ionosonde_hmF2 = ionosonde_points_in_window(ionosonde, cycle_start, cycle_end)
    if len(ionosonde_nmF2):
        ax.scatter(ionosonde_nmF2, ionosonde_hmF2, color="red", marker="o",
                    s=20, zorder=6, label="ionosonde NmF2 @ hmF2")

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
    ionosonde: IonosondeDataset | None = None,
):
    """Cross-style version of :func:`plot_isr_range_comparison`: all
    styles' per-batch analysis lines over the same ISR range band,
    color-coded by style (not by batch) -- the most direct visual answer
    to "which style's analysis best tracks the real ISR-observed range
    over the whole cycle"."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    window = isr_profiles_in_window(isr, cycle_start, cycle_end)
    lo, hi, n_profiles = _isr_window_bounds_and_confidence(isr, cycle_start, cycle_end, percentile)
    if lo is None:
        ax.text(0.5, 0.5, "no ISR coverage in this cycle's window", ha="center", va="center",
                 transform=ax.transAxes)
    else:
        _draw_isr_confidence_band(ax, isr.altitude, lo, hi, n_profiles,
                                   max_alpha=0.35 if n_profiles is not None else 0.15,
                                   label=f"ISR range (n={window.shape[1]} hours)")

    ionosonde_nmF2, ionosonde_hmF2 = ionosonde_points_in_window(ionosonde, cycle_start, cycle_end)
    if len(ionosonde_nmF2):
        ax.scatter(ionosonde_nmF2, ionosonde_hmF2, color="red", marker="o",
                    s=20, zorder=6, label="ionosonde NmF2 @ hmF2")

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


# ===========================================================================
# Post-processing: regenerate ISR-comparison plots for an already-completed
# run, against a newer ISR/ionosonde file, with no EnKF rerun (module
# docstring). Mirrors impact_vectors.py's own
# save/load/replot_..._from_saved/replot_..._tree precedent.
# ===========================================================================


@dataclass(frozen=True)
class _ReconstructedStyleRun:
    matches: list[BatchIsrMatch]
    model_altitude: np.ndarray
    cycle_start: pd.Timestamp | None
    cycle_end: pd.Timestamp | None


def _build_batch_isr_matches_from_saved(
    style_output_prefix: str | Path, isr: IsrDataset, lat: float, lon: float,
    ionosonde: IonosondeDataset | None = None,
) -> _ReconstructedStyleRun:
    """Rebuilds one style's full :class:`BatchIsrMatch` list for an
    already-completed run, from files ``package_run.py`` already saves --
    no EnKF rerun (module docstring).

    ``style_output_prefix`` is a run's per-style output path prefix, e.g.
    ``".../mil_allstyles_no_boost_20250826_0000_0100_PCA_1D_pow"`` -- its
    sibling ``<prefix>_parameterized.nc`` and
    ``<prefix>_ensembles/ensemble_batch_*.nc`` are read. ``lat``/``lon``
    is the query point (the ISR station's, for this module's use).

    Exploits ``GenericObservationOperator.decode()`` needing no ray
    geometry/``H_grid`` (confirmed by reading ``observation_operator.py``):
    a decode-only operator is built directly from the loaded
    parameterization, with a dummy zero-row ``H_grid`` that decode() never
    touches. Batch 0's forecast is the literal, RNG-free
    ``EnsembleState.from_parameterized_edp_samples`` reconstruction of the
    original IRI ensemble (exactly what ``package_run.py`` built it from
    live); batch i>0's forecast is simply batch i-1's saved analysis
    ensemble (plan Section 8 decision 3: no inter-batch inflation, the
    analysis ensemble *is* the next batch's prior)."""
    from Parameterization import Parameterized_EDPSamples   # deferred: needs its dir on sys.path

    prefix = Path(style_output_prefix)
    parameterized_path = prefix.parent / f"{prefix.name}_parameterized.nc"
    ensembles_dir = prefix.parent / f"{prefix.name}_ensembles"
    batch_paths = sorted(ensembles_dir.glob("ensemble_batch_*.nc"))
    if not batch_paths:
        raise FileNotFoundError(f"no saved ensemble_batch_*.nc files under {ensembles_dir}")

    pes = Parameterized_EDPSamples.fromNetCDF(parameterized_path)
    edp_samples = pes.EDPSamples
    n_height, n_geo = edp_samples.edps.shape[0], edp_samples.edps.shape[1]
    altitude = np.asarray(edp_samples.altitude)

    forecast_ensemble = EnsembleState.from_parameterized_edp_samples(pes)
    operator = GenericObservationOperator(
        parameterization=pes.Parameterization,
        H_grid=np.zeros((0, n_height * n_geo)),
        param_shape=forecast_ensemble.param_shape,
        n_height=n_height, n_geo=n_geo, altitude=altitude,
    )

    matches: list[BatchIsrMatch] = []
    cycle_start = cycle_end = None
    for i, batch_path in enumerate(batch_paths):
        analysis_ensemble, meta = load_ensemble_netcdf(batch_path)
        batch_index = int(meta.get("batch_index", i))
        start_time = meta.get("batch_start_time")
        end_time = meta.get("batch_end_time")
        if cycle_start is None and "cycle_start_time" in meta:
            cycle_start, cycle_end = meta["cycle_start_time"], meta["cycle_end_time"]

        decoded_forecast = operator.decode(forecast_ensemble.to_param_shape())
        decoded_analysis = operator.decode(analysis_ensemble.to_param_shape())
        _, _, forecast_profile, analysis_profile = resolve_edp_query_point_and_profiles(
            edp_samples, None, decoded_forecast, decoded_analysis, lat=lat, lon=lon,
        )

        mid = (
            pd.Timestamp(start_time) + (pd.Timestamp(end_time) - pd.Timestamp(start_time)) / 2
            if start_time is not None and end_time is not None else None
        )
        nearest = nearest_isr_profile(isr, mid) if mid is not None else None
        isr_time = nearest[0] if nearest is not None else None
        isr_density = nearest[1] if nearest is not None else None

        isr_density_min = isr_density_max = isr_n_profiles = None
        if isr_time is not None and isr.density_min is not None:
            t_idx = int(np.argmin(np.abs((isr.time - isr_time).to_numpy())))
            isr_density_min = isr.density_min[:, t_idx]
            isr_density_max = isr.density_max[:, t_idx]
            if isr.n_profiles is not None:
                isr_n_profiles = isr.n_profiles[:, t_idx]

        ionosonde_nmF2, ionosonde_hmF2 = (
            ionosonde_points_in_window(ionosonde, start_time, end_time)
            if ionosonde is not None and start_time is not None and end_time is not None
            else (None, None)
        )

        matches.append(BatchIsrMatch(
            batch_index=batch_index, isr_time=isr_time, isr_density=isr_density,
            forecast_profile=forecast_profile, analysis_profile=analysis_profile,
            isr_density_min=isr_density_min, isr_density_max=isr_density_max,
            isr_n_profiles=isr_n_profiles,
            ionosonde_nmF2=ionosonde_nmF2, ionosonde_hmF2=ionosonde_hmF2,
        ))

        forecast_ensemble = analysis_ensemble   # next batch's prior (see docstring)

    return _ReconstructedStyleRun(
        matches=matches, model_altitude=altitude,
        cycle_start=pd.Timestamp(cycle_start) if cycle_start is not None else None,
        cycle_end=pd.Timestamp(cycle_end) if cycle_end is not None else None,
    )


def _save_fig(fig_or_ax, path: Path, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(path)


def _write_isr_comparison_plots(
    recon: _ReconstructedStyleRun, isr: IsrDataset, out_dir: Path, *,
    style: str | None, ionosonde: IonosondeDataset | None, isr_range_percentile: float,
) -> list[Path]:
    """Writes exactly what ``package_run.py`` writes under one style's
    ``{label}_isr_comparison/{style}/`` directory, from an already-built
    :class:`_ReconstructedStyleRun`."""
    written: list[Path] = []
    for match in recon.matches:
        _save_fig(
            plot_isr_nearest_point_comparison(
                match, recon.model_altitude, isr.altitude, station_name=isr.station.name, style=style,
            ),
            out_dir / "nearest_point" / f"batch{match.batch_index:04d}.png", written,
        )
    if recon.cycle_start is not None and recon.cycle_end is not None:
        _save_fig(
            plot_isr_range_comparison(
                recon.matches, isr, recon.cycle_start, recon.cycle_end, recon.model_altitude,
                percentile=isr_range_percentile, style=style, ionosonde=ionosonde,
            ),
            out_dir / "range_comparison.png", written,
        )
    rmse_df = pooled_isr_rmse_by_altitude(recon.matches, recon.model_altitude, isr.altitude)
    out_dir.mkdir(parents=True, exist_ok=True)
    rmse_df.to_csv(out_dir / "pooled_rmse.csv", index=False)
    _save_fig(plot_pooled_isr_rmse(rmse_df, style=style), out_dir / "pooled_rmse.png", written)
    return written


def _write_isr_comparison_cross_style_plots(
    matches_by_style: dict[str, list[BatchIsrMatch]], isr: IsrDataset, model_altitude: np.ndarray,
    out_dir: Path, *, ionosonde: IonosondeDataset | None, cycle_start, cycle_end,
    isr_range_percentile: float,
) -> list[Path]:
    """Writes exactly what ``package_run.py`` writes under
    ``{label}_isr_comparison_cross_style/``, from already-built per-style
    match lists."""
    written: list[Path] = []
    batch_indices = sorted({m.batch_index for matches in matches_by_style.values() for m in matches})
    for batch_index in batch_indices:
        matches_for_batch = {
            s: next((m for m in matches if m.batch_index == batch_index), None)
            for s, matches in matches_by_style.items()
        }
        matches_for_batch = {s: m for s, m in matches_for_batch.items() if m is not None}
        if not matches_for_batch:
            continue
        _save_fig(
            plot_isr_nearest_point_comparison_cross_style(
                matches_for_batch, model_altitude, isr.altitude, station_name=isr.station.name,
            ),
            out_dir / "nearest_point" / f"batch{batch_index:04d}.png", written,
        )
    if cycle_start is not None and cycle_end is not None:
        _save_fig(
            plot_isr_range_comparison_cross_style(
                matches_by_style, isr, cycle_start, cycle_end, model_altitude,
                percentile=isr_range_percentile, ionosonde=ionosonde,
            ),
            out_dir / "range_comparison.png", written,
        )
    rmse_cross_df = pooled_isr_rmse_by_altitude_cross_style(matches_by_style, model_altitude, isr.altitude)
    out_dir.mkdir(parents=True, exist_ok=True)
    rmse_cross_df.to_csv(out_dir / "pooled_rmse.csv", index=False)
    _save_fig(plot_pooled_isr_rmse_cross_style(rmse_cross_df), out_dir / "pooled_rmse.png", written)
    return written


def replot_isr_comparison_for_style(
    style_output_prefix: str | Path, isr: IsrDataset, out_dir: str | Path, *,
    style: str | None = None, ionosonde: IonosondeDataset | None = None,
    isr_range_percentile: float = 0.0,
) -> list[Path]:
    """Regenerates one style's ISR-comparison plots for an already
    completed run, straight from its saved parameterization + per-batch
    analysis ensembles (module docstring) -- no EnKF rerun. See
    :func:`replot_isr_comparison_tree` for the batch entry point that
    walks a whole run tree (including the cross-style plots) at once."""
    recon = _build_batch_isr_matches_from_saved(
        style_output_prefix, isr, isr.station.lat, isr.station.lon, ionosonde=ionosonde,
    )
    return _write_isr_comparison_plots(
        recon, isr, Path(out_dir), style=style, ionosonde=ionosonde,
        isr_range_percentile=isr_range_percentile,
    )


def replot_isr_comparison_tree(
    root: str | Path, isr_file_path: str | Path, *,
    ionosonde_file_path: str | Path | None = None, isr_range_percentile: float = 0.0,
    verbose: bool = True,
) -> list[Path]:
    """Regenerates every ISR-comparison plot (per-style and cross-style)
    under an existing run tree against a newer ISR/ionosonde file, in
    place, with no pipeline rerun at all (module docstring) -- the
    ``replot_impact_vectors_tree``-style batch entry point for this
    module. Finds every ``*_edp_samples.nc`` under ``root`` (one per hour
    x boost-condition folder in a multi-hour run, e.g. a Millstone Hill
    run's Google-Drive output tree) and, for each of its sibling
    ``*_<style>_parameterized.nc`` files, rewrites that style's
    ``{label}_isr_comparison/{style}/`` plots plus the shared
    ``{label}_isr_comparison_cross_style/`` plots -- overwriting whatever
    was there before (safe: these are derived PNG/CSV outputs only, no
    other file is touched).

    Station identity (and hence which ``(lat, lon)`` is queried) comes
    entirely from ``isr_file_path``'s own attrs -- nothing here is
    Millstone-Hill-specific."""
    root = Path(root)
    isr = load_isr_dataset(isr_file_path)
    ionosonde = load_ionosonde_dataset(ionosonde_file_path) if ionosonde_file_path is not None else None

    edp_paths = sorted(root.rglob("*_edp_samples.nc"))
    if verbose:
        print(f"[isr_comparison] found {len(edp_paths)} hour/condition folders under {root}", flush=True)

    written: list[Path] = []
    for i, edp_path in enumerate(edp_paths, start=1):
        run_dir = edp_path.parent
        label = edp_path.name[: -len("_edp_samples.nc")]
        if verbose:
            print(f"[isr_comparison] ({i}/{len(edp_paths)}) {run_dir}", flush=True)

        matches_by_style: dict[str, list[BatchIsrMatch]] = {}
        model_altitude: np.ndarray | None = None
        cycle_start = cycle_end = None

        for parameterized_path in sorted(run_dir.glob(f"{label}_*_parameterized.nc")):
            style = parameterized_path.name[len(label) + 1: -len("_parameterized.nc")]
            style_prefix = run_dir / f"{label}_{style}"
            if not (run_dir / f"{label}_{style}_ensembles").is_dir():
                continue
            if verbose:
                print(f"[isr_comparison]   style={style} ...", flush=True)

            recon = _build_batch_isr_matches_from_saved(
                style_prefix, isr, isr.station.lat, isr.station.lon, ionosonde=ionosonde,
            )
            isr_dir = run_dir / f"{label}_isr_comparison" / style
            written.extend(_write_isr_comparison_plots(
                recon, isr, isr_dir, style=style, ionosonde=ionosonde,
                isr_range_percentile=isr_range_percentile,
            ))

            matches_by_style[style] = recon.matches
            model_altitude = recon.model_altitude
            if cycle_start is None:
                cycle_start, cycle_end = recon.cycle_start, recon.cycle_end

        if len(matches_by_style) > 1 and model_altitude is not None:
            cross_dir = run_dir / f"{label}_isr_comparison_cross_style"
            written.extend(_write_isr_comparison_cross_style_plots(
                matches_by_style, isr, model_altitude, cross_dir, ionosonde=ionosonde,
                cycle_start=cycle_start, cycle_end=cycle_end,
                isr_range_percentile=isr_range_percentile,
            ))

    if verbose:
        print(f"[isr_comparison] done -- wrote {len(written)} files under {root}", flush=True)
    return written
