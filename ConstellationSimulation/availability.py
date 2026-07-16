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
from typing import Callable, Iterable, List, Optional, Sequence, Union

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
        links: Union[Iterable[dict], Iterable[Iterable[dict]]],
) -> List[OccultationEvent]:
    """Build OccultationEvent records from occultation.py link dicts (either
    a flat list, e.g. find_valid_links()'s return value, or the nested
    per-epoch list-of-lists returned by scan_availability()).

    occultation.py link dicts carry r_t in whatever frame the constellation
    is propagated in (pseudo-inertial ECI for Walker-generated satellites,
    see propagator.py); this converts r_t to ECEF at the link's own epoch
    via propagator.eci_to_ecef() before storing it, since geodetic lat/lon
    is only meaningful in an Earth-fixed frame.
    """
    from propagator import eci_to_ecef

    flat: List[dict] = []
    for item in links:
        if isinstance(item, dict):
            flat.append(item)
        else:
            flat.extend(item)

    events: List[OccultationEvent] = []
    for link in flat:
        r_t_ecef_km, _ = eci_to_ecef(link["r_t"], np.zeros(3), link["epoch"])
        events.append(OccultationEvent(
            timestamp=link["epoch"],
            tx_id=link["tx"].name,
            rx_id=link["rx"].name,
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

def plot_simulated_availability(
        df: pd.DataFrame,
        day: pd.Timestamp,
        window_hours: float = 1.0,
        roi_thresholds_km: Optional[Sequence[float]] = None,
        alt_min: Optional[float] = None,
        save_path: Optional[Path] = None,
):
    """Call demo_occultation_availability.plot_occultation_availability() on
    a simulated-occultation DataFrame (see build_occultation_dataframe).
    """
    doa = _import_demo_occultation_availability()
    kwargs = dict(window_hours=window_hours, alt_min=alt_min, save_path=save_path)
    if roi_thresholds_km is not None:
        kwargs["roi_thresholds_km"] = roi_thresholds_km
    return doa.plot_occultation_availability(df, day, **kwargs)
