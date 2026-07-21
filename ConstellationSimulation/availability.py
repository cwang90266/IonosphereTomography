"""
availability.py

Bridge between the geometric constellation simulation (occultation.py) and
the existing regional-availability analysis in demo_occultation_availability.py.

Simulated occultation events (timestamp, TX id, RX id, ECEF tangent point)
are translated into the exact pandas.DataFrame schema that
demo_occultation_availability.py's read_day_occultations() produces
(filename, spacecraft, tecmax_time, lat, lon, tecmax, dist_<site>_km,
nearest_site, nearest_km, in_roi), so the same plotting entry point
(plot_occultation_availability) can be reused unmodified on simulated data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyproj

# ── ECEF (EPSG:4978) -> geodetic lon/lat/height (EPSG:4979) ──────────────
# Exact transformer scheme used by demo_occultation_availability.py
# (_ECEF_TO_GEODETIC / _leo_track_points). Built once and reused.
_ECEF_TO_GEODETIC = pyproj.Transformer.from_crs(
    "EPSG:4978", "EPSG:4979", always_xy=True)


def tangent_point_to_geodetic(r_t_km: np.ndarray) -> "tuple[float, float]":
    """Convert an ECEF tangent-point position vector r_t = [x_t, y_t, z_t]^T
    (km) to geodetic (lat_deg, lon_deg), via the EPSG:4978 -> EPSG:4979
    pyproj transformer (km -> m at the EPSG:4978 boundary, height dropped).
    """
    x_t, y_t, z_t = np.asarray(r_t_km, dtype=float)
    lon, lat, _h = _ECEF_TO_GEODETIC.transform(
        x_t * 1000.0, y_t * 1000.0, z_t * 1000.0)
    return float(lat), float(lon)


# ──────────────────────────────────────────────────────────────────────────
# demo_occultation_availability.py import (repo root is one directory above
# ConstellationSimulation/, same pattern as occultation.py's
# _import_test_param_iono()).
# ──────────────────────────────────────────────────────────────────────────

def _import_demo_occultation_availability():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import demo_occultation_availability as doa
    return doa


# ──────────────────────────────────────────────────────────────────────────
# Simulated occultation event -> analysis-ready DataFrame
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class OccultationEvent:
    """One simulated occultation event: TEC-max-equivalent timestamp, the
    TX/RX satellite identifiers, and the SLTA tangent point in ECEF (km)."""

    timestamp: datetime
    tx_id: str
    rx_id: str
    r_t_ecef_km: np.ndarray


def events_from_occultation_links(
        link_history: Sequence[Sequence[dict]], dt_s: float,
        gap_tol: float = 1.5,
) -> List[OccultationEvent]:
    """Build one OccultationEvent per continuous occultation pass (NOT one
    per epoch): link_history is the per-epoch list-of-lists scan_availability()
    produces. scan_availability() reports SLTA validity independently at each
    time step, so a single physical occultation spanning many consecutive
    valid epochs must collapse to one event -- otherwise every step of the
    same rising/setting pass is double-counted as its own "occultation".

    Consecutive per-(TX,RX) links are grouped into passes via
    occultation.segment_occultation_passes(dt_s, gap_tol) -- the same
    segmentation build_occultation_arcs() uses -- so a pair that occults more
    than once during the run (separate orbital passes) still yields one event
    per pass, not one event for the whole run. Each pass's representative
    link is the epoch of minimum tangent altitude (deepest SLTA penetration /
    closest approach), analogous to how real RO products characterise an
    occultation by its point of closest approach.

    occultation.py link dicts carry r_t in whatever frame the constellation
    is propagated in (pseudo-inertial ECI for Walker-generated satellites,
    see propagator.py); this converts r_t to ECEF at the representative
    link's own epoch via propagator.eci_to_ecef(), since geodetic lat/lon is
    only meaningful in an Earth-fixed frame.
    """
    from propagator import eci_to_ecef
    from occultation import segment_occultation_passes

    segments = segment_occultation_passes(link_history, dt_s, gap_tol)

    events: List[OccultationEvent] = []
    for link_list in segments:
        rep = min(link_list, key=lambda link: link["h_t"])
        r_t_ecef_km, _ = eci_to_ecef(rep["r_t"], np.zeros(3), rep["epoch"])
        events.append(OccultationEvent(
            timestamp=rep["epoch"],
            tx_id=rep["tx"].name,
            rx_id=rep["rx"].name,
            r_t_ecef_km=r_t_ecef_km,
        ))
    return events


TecmaxSpec = Union[float, Sequence[float], Callable[[OccultationEvent], float]]


def _resolve_tecmax(events: Sequence[OccultationEvent], tecmax: TecmaxSpec) -> np.ndarray:
    """Resolve the mock tecmax column: a constant, a per-event sequence, or
    a callable(event) -> float."""
    if callable(tecmax):
        return np.array([float(tecmax(ev)) for ev in events])
    if np.isscalar(tecmax):
        return np.full(len(events), float(tecmax))
    arr = np.asarray(tecmax, dtype=float)
    if arr.shape != (len(events),):
        raise ValueError("tecmax sequence must have one value per event")
    return arr


def build_occultation_dataframe(
        events: Sequence[OccultationEvent],
        tecmax: TecmaxSpec = 50.0,
) -> pd.DataFrame:
    """Translate simulated occultation events into the DataFrame schema
    demo_occultation_availability.py's plotting functions expect:

        filename, spacecraft, tecmax_time, lat, lon, tecmax,
        dist_<site>_km (one per ISR site), nearest_site, nearest_km, in_roi

    Parameters
    ----------
    events : sequence of OccultationEvent (see events_from_occultation_links).
    tecmax : mock peak TEC value -- a constant applied to every event, a
        per-event sequence, or a callable(event) -> float (e.g. a peak
        pulled from a test_param_iono.py forward-modeled TEC profile).
    """
    if not events:
        return pd.DataFrame()

    doa = _import_demo_occultation_availability()
    tecmax_vals = _resolve_tecmax(events, tecmax)

    rows = []
    for ev, tmax in zip(events, tecmax_vals):
        lat, lon = tangent_point_to_geodetic(ev.r_t_ecef_km)
        ts = pd.Timestamp(ev.timestamp)
        rows.append({
            "filename":    f"sim_{ev.tx_id}_{ev.rx_id}_{ts.strftime('%Y%m%dT%H%M%S%f')}",
            "spacecraft":  ev.rx_id,
            "tecmax_time": ts,
            "lat":         lat,
            "lon":         lon,
            "tecmax":      float(tmax),
        })

    df = pd.DataFrame(rows).sort_values("tecmax_time").reset_index(drop=True)
    _attach_station_distances(df, doa)
    return df


def _attach_station_distances(df: pd.DataFrame, doa) -> pd.DataFrame:
    """Compute proximity to the active ISR ground stations (ESR/TRO), in
    place on *df*. Mirrors demo_occultation_availability.read_day_occultations
    exactly, importing ISR_SITES/INSTRUMENTS/ISR_ROI_MAX_KM/_haversine_km
    from *doa* (demo_occultation_availability.py) itself rather than from
    demo_isr_da_comparison, so the LON_SHIFT_DEG-shifted INSTRUMENTS
    coordinates (if LON_SHIFT_DEG != 0) are respected automatically.
    """
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()

    dist_cols = []
    for site in doa.ISR_SITES:
        inst = doa.INSTRUMENTS[site]
        col = f"dist_{site}_km"
        df[col] = doa._haversine_km(inst["lat"], inst["lon"], lat, lon)
        dist_cols.append(col)

    dist_mat = df[dist_cols].to_numpy()
    nearest_idx = np.argmin(dist_mat, axis=1)
    df["nearest_site"] = [doa.ISR_SITES[i] for i in nearest_idx]
    df["nearest_km"]   = dist_mat[np.arange(len(df)), nearest_idx]
    df["in_roi"]       = df["nearest_km"] <= doa.ISR_ROI_MAX_KM
    return df


# ──────────────────────────────────────────────────────────────────────────
# Plotting hook
# ──────────────────────────────────────────────────────────────────────────

def compute_rx_ground_tracks(
        rx_constellation,
        start_time: datetime,
        duration_hours: float = 2.0,
        step_s: float = 60.0,
        mass_kg: Optional[float] = None,
        area_m2: Optional[float] = None,
        cr: Optional[float] = None,
) -> pd.DataFrame:
    """Propagate an independent copy of *rx_constellation* forward
    *duration_hours* (default 2 h) at *step_s* resolution and record each
    satellite's geodetic ground-track position at every step.

    This is fully decoupled from occultation.py's scan_availability() (which
    hides its own propagation loop and does not expose per-epoch RX
    positions) -- a deep copy of the RX constellation is propagated
    separately here purely for visualization, so it never disturbs the
    "real" simulation state.

    Parameters
    ----------
    rx_constellation : RXConstellation
        The constellation to snapshot (a deep copy is propagated; the
        original is left untouched). Its satellites' .epoch is overwritten
        with *start_time* before propagating, so the ground track always
        starts at the requested time regardless of the epoch the
        satellites were originally built with.
    start_time : datetime
        UTC epoch the ground track starts at.
    duration_hours : float
        Total ground-track length in hours (default 2.0).
    step_s : float
        Propagation step size in seconds (default 60.0).
    mass_kg, area_m2, cr : float, optional
        Forwarded to propagator.propagate_constellation (defaults to that
        module's DEFAULT_MASS_KG / DEFAULT_AREA_M2 / DEFAULT_CR if omitted).

    Returns
    -------
    pd.DataFrame with columns: name, constellation, utc_time, lat, lon
    """
    import copy

    from propagator import (
        DEFAULT_AREA_M2, DEFAULT_CR, DEFAULT_MASS_KG,
        eci_to_ecef, propagate_constellation,
    )

    mass_kg = DEFAULT_MASS_KG if mass_kg is None else mass_kg
    area_m2 = DEFAULT_AREA_M2 if area_m2 is None else area_m2
    cr = DEFAULT_CR if cr is None else cr

    rx = copy.deepcopy(rx_constellation)
    for sat in rx.satellites:
        sat.epoch = start_time

    n_steps = max(int(round(duration_hours * 3600.0 / step_s)), 1)
    rows: List[dict] = []

    def _snapshot() -> None:
        for sat in rx.satellites:
            r_ecef_km, _v_ecef_km_s = eci_to_ecef(
                np.asarray(sat.r_eci_m, dtype=float) * 1e-3,
                np.asarray(sat.v_eci_m_s, dtype=float) * 1e-3,
                sat.epoch,
            )
            lat, lon = tangent_point_to_geodetic(r_ecef_km)
            rows.append({
                "name": sat.name, "constellation": sat.constellation,
                "utc_time": pd.Timestamp(sat.epoch), "lat": lat, "lon": lon,
            })

    _snapshot()
    for _ in range(n_steps):
        propagate_constellation(rx, dt_s=step_s, n_steps=1,
                                 mass_kg=mass_kg, area_m2=area_m2, cr=cr)
        _snapshot()

    return pd.DataFrame(rows)


def _split_track_segments(lat: np.ndarray, lon: np.ndarray,
                           jump_deg: float = 100.0):
    """Yield (lat_segment, lon_segment) sub-arrays of a single satellite's
    ground track, split wherever consecutive longitude samples jump by more
    than *jump_deg* (antimeridian wraparound / near-pole passes), so each
    segment can be plotted separately without a spurious straight line
    cutting across the map.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size == 0:
        return
    breaks = np.where(np.abs(np.diff(lon)) > jump_deg)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [lat.size]))
    for s, e in zip(starts, ends):
        if e - s >= 1:
            yield lat[s:e], lon[s:e]


def _draw_ground_tracks(ax, tracks_df: pd.DataFrame) -> None:
    """Draw color-coded per-satellite ground tracks (from
    compute_rx_ground_tracks) onto a cartopy GeoAxes (any global
    projection, e.g. Orthographic), with a start-point marker and legend.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="0.97", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
    ax.set_global()

    names = sorted(tracks_df["name"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(names)}

    for name in names:
        sub = tracks_df.loc[tracks_df["name"] == name].sort_values("utc_time")
        lat = sub["lat"].to_numpy(dtype=float)
        lon = sub["lon"].to_numpy(dtype=float)
        color = colors[name]
        first = True
        for lat_seg, lon_seg in _split_track_segments(lat, lon):
            ax.plot(lon_seg, lat_seg, transform=ccrs.Geodetic(),
                    color=color, linewidth=1.4,
                    label=name if first else None)
            first = False
        ax.plot(lon[0], lat[0], marker="o", markersize=5,
                color=color, transform=ccrs.PlateCarree(), zorder=5)

    ax.legend(loc="lower left", fontsize=7, framealpha=0.85,
               bbox_to_anchor=(0.0, -0.05))


def _append_ground_track_panel(fig, tracks_df: pd.DataFrame,
                                extra_width_in: float = 5.0,
                                clon: float = 0.0, clat: float = 30.0):
    """Post-process an already-built Figure (from
    demo_occultation_availability.plot_occultation_availability) to append
    a new Orthographic ground-track panel on the right, without touching
    the plotting function that built the original axes.

    Widens the figure canvas by *extra_width_in* inches and rescales every
    existing axes' position to make room, then adds the new panel in the
    freed-up strip.
    """
    import cartopy.crs as ccrs

    w0, h0 = fig.get_size_inches()
    w1 = w0 + extra_width_in
    shrink = w0 / w1

    axes = list(fig.axes)
    positions = [ax.get_position() for ax in axes]

    fig.set_size_inches(w1, h0, forward=True)
    for ax, pos in zip(axes, positions):
        ax.set_position([pos.x0 * shrink, pos.y0, pos.width * shrink, pos.height])

    panel_x0 = 1.0 - extra_width_in / w1 + 0.02
    panel_w = extra_width_in / w1 - 0.06
    ax_track = fig.add_axes(
        [panel_x0, 0.08, panel_w, 0.84],
        projection=ccrs.Orthographic(clon, clat),
    )
    _draw_ground_tracks(ax_track, tracks_df)
    duration_h = (
        (tracks_df["utc_time"].max() - tracks_df["utc_time"].min()).total_seconds()
        / 3600.0
    ) if len(tracks_df) else 0.0
    ax_track.set_title(f"RX ground tracks ({duration_h:.1f} h)", fontsize=10)

    return ax_track


def plot_simulated_availability(
        df: pd.DataFrame,
        day: pd.Timestamp,
        window_hours: float = 1.0,
        roi_thresholds_km: Optional[Sequence[float]] = None,
        alt_min: Optional[float] = None,
        save_path: Optional[Path] = None,
        ground_tracks: Optional[pd.DataFrame] = None,
):
    """Call demo_occultation_availability.plot_occultation_availability() on
    a simulated-occultation DataFrame (see build_occultation_dataframe).

    If *ground_tracks* is given (a DataFrame from compute_rx_ground_tracks),
    an extra Orthographic ground-track panel is appended to the returned
    figure via _append_ground_track_panel before saving. This
    post-processes the Figure object only -- it does not modify
    demo_occultation_availability.plot_occultation_availability() itself,
    which is also used (unchanged) by the real-data pipelines.
    """
    doa = _import_demo_occultation_availability()
    # When appending a ground-track panel we do our own save (after
    # appending), so suppress plot_occultation_availability's internal save.
    inner_save_path = None if ground_tracks is not None else save_path
    kwargs = dict(window_hours=window_hours, alt_min=alt_min, save_path=inner_save_path)
    if roi_thresholds_km is not None:
        kwargs["roi_thresholds_km"] = roi_thresholds_km
    fig = doa.plot_occultation_availability(df, day, **kwargs)

    if ground_tracks is not None and fig is not None and not ground_tracks.empty:
        _append_ground_track_panel(fig, ground_tracks)
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved figure → {save_path}")

    return fig


def plot_global_occultation_density(
        df: pd.DataFrame,
        day: pd.Timestamp,
        radii_km: Sequence[float] = (500.0, 1500.0, 2500.0),
        grid_deg: float = 2.0,
        window_hours: float = 1.0,
        bin_minutes: float = 5.0,
        cmap: str = "viridis",
        save_path: Optional[Path] = None,
        ground_tracks: Optional[pd.DataFrame] = None,
        track_clon: float = 0.0,
        track_clat: float = 30.0,
):
    """Three global Mollweide-projection maps (one per entry of *radii_km*)
    of simulated occultation availability: at every cell of a *grid_deg*-
    spaced lat/lon grid spanning the whole planet, this finds the PEAK
    number of occultation tangent points within that cell's radius that
    occur inside any rolling *window_hours*-wide time window over the day
    -- i.e. the same "peak in-ROI occultations / 1 h window" quantity
    generate_output() already prints for the fixed ESR/TRO ROI
    (doa.rolling_window_count), just made spatially continuous instead of
    evaluated at one fixed site/radius. A flat whole-day total count would
    hide bursty regions (e.g. a radius that only ever sees 1 occultation
    at a time, spread evenly through the day, looks identical to one that
    sees a cluster of 10 simultaneously) -- the rolling-window peak is what
    actually matters for e.g. simultaneous-RO capacity planning.

    "Within R km" uses the same great-circle distance (doa._haversine_km,
    imported from demo_isr_da_comparison via demo_occultation_availability)
    already used for the regional dist_<site>_km / nearest_km / in_roi
    columns. This is unrelated to the fixed-site ISR ROI rings panel inside
    plot_simulated_availability -- there is no "site" here, every point on
    Earth is its own query point.

    Implementation note: rather than looping doa.rolling_window_count per
    grid cell (91x181 cells x a python-level sliding window would be very
    slow), every event is first digitized into fixed *bin_minutes*-wide
    time bins spanning the day (one-hot membership matrix B, shape
    (n_events, n_bins)). For each radius, the (n_cells, n_events) boolean
    "event within R km of this cell" mask is matrix-multiplied by B to get
    per-cell per-bin counts in one BLAS call, then a cumulative-sum sliding
    window turns those into rolling *window_hours* sums, and the max over
    time gives the per-cell peak.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_occultation_dataframe (must have "lat"/"lon"/
        "tecmax_time").
    day : pd.Timestamp
        The UTC calendar day the rolling time-window grid spans (events
        are located relative to this day's midnight, same convention as
        generate_output()'s doa.rolling_window_count call).
    radii_km : sequence of float
        One panel is drawn per radius (default: the same 500/1500/2500 km
        thresholds used by the regional ROI-rings panel).
    grid_deg : float
        Query-grid spacing in degrees (default 2.0 -> 91x181 grid).
    window_hours : float
        Width of the rolling time window (default 1.0, matching the
        regional "peak / 1 h window" metric).
    bin_minutes : float
        Time-bin width used to discretize the rolling window (default 5.0
        minutes) -- the granularity of the peak-window search, not the
        window width itself.
    cmap : str
        Matplotlib colormap name for the density heatmap.
    save_path : Path, optional
        If given, the figure is saved here (dpi=150, bbox_inches="tight").
    ground_tracks : pd.DataFrame, optional
        Output of compute_rx_ground_tracks. If given, an extra Orthographic
        panel with color-coded per-satellite ground tracks is appended
        after the density panels (kept on its own projection rather than
        Mollweide, since Mollweide's Geodetic line rendering is visually
        much less clean for near-polar tracks).
    track_clon, track_clat : float
        Center longitude/latitude (deg) of the ground-track panel's
        Orthographic projection.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    doa = _import_demo_occultation_availability()

    if df.empty:
        raise ValueError("plot_global_occultation_density: df is empty -- nothing to plot")

    lat_deg = df["lat"].to_numpy(dtype=float)
    lon_deg = df["lon"].to_numpy(dtype=float)
    n_events = len(df)

    grid_lat = np.arange(-90.0, 90.0 + 1e-9, grid_deg)
    grid_lon = np.arange(-180.0, 180.0 + 1e-9, grid_deg)
    glon, glat = np.meshgrid(grid_lon, grid_lat)  # each (M, N)
    n_cells = glat.size

    # Great-circle distance from every grid cell (M, N) to every event (K),
    # broadcast into one (M, N, K) matrix and reused for every threshold
    # below rather than recomputed per radius.
    dist_flat = doa._haversine_km(
        glat[:, :, None], glon[:, :, None],
        lat_deg[None, None, :], lon_deg[None, None, :],
    ).reshape(n_cells, n_events)

    # Digitize events into fixed time bins spanning *day*, and build a
    # one-hot event->bin membership matrix (independent of grid_deg, so
    # this stays small regardless of grid resolution).
    day0 = pd.Timestamp(day).normalize()
    n_bins = max(int(round(24 * 60 / bin_minutes)), 1)
    # Series-level subtraction (rather than raw .to_numpy() on a tz-aware
    # column, which yields an object array of Timestamps that numpy can't
    # subtract a datetime64 from) correctly handles the tz-aware
    # tecmax_time column produced by build_occultation_dataframe.
    t_event_min = (
        (pd.to_datetime(df["tecmax_time"]) - day0).dt.total_seconds() / 60.0
    ).to_numpy()
    bin_idx = np.clip((t_event_min // bin_minutes).astype(int), 0, n_bins - 1)
    B = np.zeros((n_events, n_bins), dtype=np.float32)
    B[np.arange(n_events), bin_idx] = 1.0

    w = max(int(round(window_hours * 60.0 / bin_minutes)), 1)

    has_tracks = ground_tracks is not None and not ground_tracks.empty
    n_panels = len(radii_km) + (1 if has_tracks else 0)

    fig = plt.figure(figsize=(6.0 * n_panels, 6.0))
    axes = [
        fig.add_subplot(1, n_panels, i + 1, projection=ccrs.Mollweide())
        for i in range(len(radii_km))
    ]

    for ax, r in zip(axes, radii_km):
        spatial_mask = (dist_flat <= r).astype(np.float32)      # (n_cells, n_events)
        counts_per_bin = spatial_mask @ B                        # (n_cells, n_bins)
        csum = np.cumsum(
            np.pad(counts_per_bin, ((0, 0), (1, 0))), axis=1)    # (n_cells, n_bins+1)
        rolling = csum[:, w:] - csum[:, :-w]                     # (n_cells, n_bins-w+1)
        peak = rolling.max(axis=1).reshape(glat.shape)           # (M, N)

        mesh = ax.pcolormesh(
            glon, glat, peak, transform=ccrs.PlateCarree(),
            cmap=cmap, shading="auto",
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="dimgray", alpha=0.5)
        ax.set_global()
        ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85)
        cb.set_label(f"peak occultations / {window_hours:g} h window within {int(r)} km")
        ax.set_title(f"≤ {int(r)} km  (peak={int(peak.max())})")

    if has_tracks:
        ax_track = fig.add_subplot(
            1, n_panels, n_panels,
            projection=ccrs.Orthographic(track_clon, track_clat),
        )
        _draw_ground_tracks(ax_track, ground_tracks)
        duration_h = (
            (ground_tracks["utc_time"].max() - ground_tracks["utc_time"].min())
            .total_seconds() / 3600.0
        )
        ax_track.set_title(f"RX ground tracks ({duration_h:.1f} h)")

    fig.suptitle(f"Global occultation availability — {pd.Timestamp(day).date()} "
                 f"({n_events} events, {window_hours:g} h rolling window)", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
