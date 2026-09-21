from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .roi_tools import (
    circular_roi_points, geodesic_circle_latlon,
    DEFAULT_FIBONACCI_SPACING_DEG, DEFAULT_FIBONACCI_SPACING_KM,
)

from TEC_model.igs_tec_pipeline import (
    igs_obs_to_clean_entry,
    process_igs_station,
)


_IGS_ARC_PER_EPOCH_FIELDS = (
    "tec",
    "tangent_km",
    "ipp_lat",
    "ipp_lon",
    "arc_time_sec",
    "time_s",
    "time_utc_h",
    "elev_deg",
)


def _normalize_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _haversine_km(lat0: float, lon0: float, lat, lon) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r_earth_km = 6371.0

    p0 = np.deg2rad(float(lat0))
    p1 = np.deg2rad(lat)
    dphi = p1 - p0
    dlambda = np.deg2rad(lon - float(lon0))

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(p0) * np.cos(p1) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r_earth_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _dates_touched(start_time, end_time, explicit_date=None) -> list[pd.Timestamp]:
    """
    Return midnight timestamps for the dates whose IGS files may be needed.

    If no time window is supplied, explicit_date is required.
    """
    start = _normalize_timestamp(start_time)
    end = _normalize_timestamp(end_time)

    if start is None and end is None:
        if explicit_date is None:
            raise ValueError("date is required when start_time/end_time are not supplied.")
        return [pd.Timestamp(explicit_date).normalize()]

    if start is None or end is None:
        raise ValueError("start_time and end_time must be supplied together.")

    if end <= start:
        raise ValueError("end_time must be later than start_time.")

    # end is exclusive, so subtract a microsecond for exact-midnight end times.
    last = (end - pd.Timedelta(microseconds=1)).normalize()
    first = start.normalize()

    days = []
    cur = first
    while cur <= last:
        days.append(cur)
        cur += pd.Timedelta(days=1)
    return days


def filter_igs_by_time(
    entries: list[dict],
    start_time=None,
    end_time=None,
) -> list[dict]:
    """Keep IGS arcs whose entry['date'] falls in [start_time, end_time)."""
    start = _normalize_timestamp(start_time)
    end = _normalize_timestamp(end_time)

    if start is None and end is None:
        return list(entries)
    if start is None or end is None:
        raise ValueError("start_time and end_time must be supplied together.")

    out = []
    for entry in entries:
        dt = entry.get("date")
        if dt is None or pd.isna(dt):
            continue
        ts = _normalize_timestamp(dt)
        if start <= ts < end:
            out.append(entry)
    return out


def filter_igs_by_roi(
    entries: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> list[dict]:
    """
    Keep arcs whose representative IPP/tangent point is inside the ROI.

    This is optional because the current main code historically selected IGS
    coverage primarily by station list rather than applying an additional
    spatial gate after arc generation.
    """
    out = []
    for entry in entries:
        lat = float(entry.get("lat_tecmax_tangent", np.nan))
        lon = float(entry.get("lon_tecmax_tangent", np.nan))
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        dist = float(_haversine_km(center_lat, center_lon, lat, lon))
        if dist <= float(radius_km):
            out.append(entry)
    return out


def collapse_igs_arc_to_central_epoch(arc: dict) -> dict:
    """
    Preserve the current static-ionosphere experiment:
    reduce one multi-epoch IGS arc to its central epoch.

    LEO/GNSS stay shape (3, 1), and all aligned per-epoch fields are reduced
    to length 1.
    """
    out = deepcopy(arc)

    tec = np.asarray(out.get("tec", []))
    n = len(tec)
    if n <= 1:
        return out

    idx = n // 2

    for key in ("LEO", "GNSS"):
        arr = out.get(key)
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == n:
                out[key] = arr[:, idx:idx + 1]

    for key in _IGS_ARC_PER_EPOCH_FIELDS:
        arr = out.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 1 and len(arr) == n:
            out[key] = arr[idx:idx + 1]

    return out

def filter_igs_epochs_by_roi(
    entries: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> list[dict]:

    out = []

    for entry in entries:

        ipp_lat = np.asarray(entry.get("ipp_lat", []), dtype=float)
        ipp_lon = np.asarray(entry.get("ipp_lon", []), dtype=float)

        if len(ipp_lat) == 0 or len(ipp_lon) == 0:
            continue

        # Distance of EVERY IPP epoch from ROI center
        dist_km = _haversine_km(
            center_lat,
            center_lon,
            ipp_lat,
            ipp_lon,
        )

        keep = (
            np.isfinite(ipp_lat)
            & np.isfinite(ipp_lon)
            & np.isfinite(dist_km)
            & (dist_km <= radius_km)
        )

        if not np.any(keep):
            continue

        new_entry = dict(entry)

        n = len(entry["tec"])

        # Array fields aligned epoch-by-epoch
        per_epoch_fields = (
            "tec",
            "tangent_km",
            "elev_deg",
            "ipp_lat",
            "ipp_lon",
            "arc_time_sec",
            "time_s",
            "time_utc_h",
        )

        for key in per_epoch_fields:
            value = entry.get(key)

            if (
                value is not None
                and hasattr(value, "__len__")
                and len(value) == n
            ):
                new_entry[key] = np.asarray(value)[keep]

        # LEO/GNSS use shape (3, N)
        new_entry["LEO"] = entry["LEO"][:, keep]
        new_entry["GNSS"] = entry["GNSS"][:, keep]

        # Useful diagnostic
        new_entry["ipp_distance_km"] = np.asarray(dist_km)[keep]

        out.append(new_entry)

    return out

def prepare_igs_observations(
    *,
    stations: Iterable[str],
    cache_dir: str | Path,
    date=None,
    start_time=None,
    end_time=None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    rinex_version: int = 3,
    use_iri: bool = False,
    min_valid_epochs: int = 50,
    max_rays_per_arc: int = 200,
    epoch_mode: str = "center",
    return_report: bool = False,
    output_dir: str | Path | None = None,
    output_prefix: str = "igs_observations",
):
    """
    Prepare IGS observations for tomography, with no KF/EKF/voxel dependency.

    The low-level IGS pipeline remains in TEC_model.igs_tec_pipeline and still
    owns RINEX parsing, dual-frequency selection, elevation cutoff,
    cycle-slip/time-gap arc splitting, carrier-phase leveling, DCB correction,
    and raw TEC plausibility filtering.

    This function adds the higher-level preparation that was scattered across
    the main/demo code:
      1. run each requested station/day,
      2. convert raw pipeline arcs with igs_obs_to_clean_entry(),
      3. keep [start_time, end_time),
      4. optionally apply a representative-point ROI filter,
      5. optionally collapse each arc to its central epoch.

    Parameters
    ----------
    epoch_mode : {"center", "all"}
        "center" reproduces the current static-ionosphere experiment.
        "all" keeps the post-downsampling epochs in each arc.
    """
    stations = list(stations)
    cache_dir = Path(cache_dir)

    if epoch_mode not in {"center", "all"}:
        raise ValueError("epoch_mode must be 'center' or 'all'.")

    roi_requested = (
        center_lat is not None
        or center_lon is not None
        or radius_km is not None
    )
    if roi_requested and (
        center_lat is None or center_lon is None or radius_km is None
    ):
        raise ValueError(
            "center_lat, center_lon, and radius_km must either all be set or all be None."
        )

    days = _dates_touched(start_time, end_time, explicit_date=date)

    raw_arc_count = 0
    clean_before_time = []
    station_reports = []

    for day in days:
        pydate = day.to_pydatetime()
        for station in stations:
            try:
                raw_obs = process_igs_station(
                    station=station,
                    date=pydate,
                    rinex_version=rinex_version,
                    cache_dir=cache_dir,
                    use_iri=use_iri,
                    max_rays=max_rays_per_arc,

                    tlim=(
                        pd.Timestamp(start_time),
                        pd.Timestamp(end_time),
                    ) if start_time is not None and end_time is not None else None,
                )
            except Exception as exc:
                station_reports.append({
                    "station": station,
                    "date": day.strftime("%Y-%m-%d"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_raw_arcs": 0,
                    "n_clean_arcs": 0,
                })
                continue

            raw_arc_count += len(raw_obs)
            n_clean_station = 0

            for obs in raw_obs:
                entry = igs_obs_to_clean_entry(
                    obs,
                    max_rays=max_rays_per_arc,
                    min_valid=min_valid_epochs,
                )
                if entry is not None:
                    # Preserve the ground receiver identifier through later
                    # epoch filtering/collapse.  Strip only a trailing numeric
                    # monument suffix (e.g. TRO1 -> TRO).
                    station_id = str(obs.get("station_id", station)).strip().upper()
                    entry["rx_id"] = re.sub(r"\d+$", "", station_id) or station_id
                    clean_before_time.append(entry)
                    n_clean_station += 1

            station_reports.append({
                "station": station,
                "date": day.strftime("%Y-%m-%d"),
                "status": "ok",
                "n_raw_arcs": int(len(raw_obs)),
                "n_clean_arcs": int(n_clean_station),
            })

    after_time = filter_igs_by_time(
        clean_before_time,
        start_time=start_time,
        end_time=end_time,
    )

    if roi_requested:
        after_roi = filter_igs_epochs_by_roi(
            after_time,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            radius_km=float(radius_km),
        )
    else:
        after_roi = list(after_time)

    if epoch_mode == "center":
        observations = [
            collapse_igs_arc_to_central_epoch(x)
            for x in after_roi
        ]
    else:
        observations = after_roi

    report = {
        "source": "IGS_ground",
        "stations": stations,
        "dates_loaded": [x.strftime("%Y-%m-%d") for x in days],
        "n_raw_pipeline_arcs": int(raw_arc_count),
        "n_clean_before_time_filter": int(len(clean_before_time)),
        "n_after_time_filter": int(len(after_time)),
        "n_after_roi_filter": int(len(after_roi)),
        "n_final_observations": int(len(observations)),
        "n_final_rays_total": int(sum(len(x.get("tec", [])) for x in observations)),
        "min_valid_epochs": int(min_valid_epochs),
        "max_rays_per_arc": int(max_rays_per_arc),
        "epoch_mode": epoch_mode,
        "roi_filter_applied": bool(roi_requested),
        "roi_radius_km": float(radius_km) if radius_km is not None else None,
        "roi_fibonacci_spacing_deg": float(DEFAULT_FIBONACCI_SPACING_DEG),
        "roi_fibonacci_spacing_km": float(DEFAULT_FIBONACCI_SPACING_KM),
        "station_runs": station_reports,
    }

    if output_dir is not None:
        export_igs_outputs(
            observations, report, output_dir, output_prefix,
            center_lat=center_lat, center_lon=center_lon, radius_km=radius_km,
            start_time=start_time, end_time=end_time,
        )

    return (observations, report) if return_report else observations


def _bearing_and_distance_km(lat0, lon0, lat, lon):
    """Return initial great-circle bearing [rad] and distance [km] from center."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    phi0 = np.deg2rad(float(lat0))
    phi = np.deg2rad(lat)
    dlon = np.deg2rad(lon - float(lon0))

    y = np.sin(dlon) * np.cos(phi)
    x = np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlon)
    bearing = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    distance = _haversine_km(lat0, lon0, lat, lon)
    return bearing, distance




def _format_filename_number(value) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _time_window_csv_tag(start_time, end_time) -> str:
    start = _normalize_timestamp(start_time)
    end = _normalize_timestamp(end_time)
    if start is None or end is None:
        return "timewindow_unknown"
    minutes = (end - start).total_seconds() / 60.0
    if np.isclose(minutes % 60.0, 0.0):
        duration = f"{_format_filename_number(minutes / 60.0)}hr"
    else:
        duration = f"{_format_filename_number(minutes)}min"
    if start.date() == end.date():
        return f"timewindow{duration}_{start:%Y%m%d}_{start:%H%M}{end:%H%M}"
    return f"timewindow{duration}_{start:%Y%m%d_%H%M}_{end:%Y%m%d_%H%M}"


def _igs_csv_filename(center_lat, center_lon, radius_km, start_time, end_time) -> str:
    if center_lat is None or center_lon is None or radius_km is None:
        roi = "roi_unspecified"
    else:
        roi = (
            f"lat{_format_filename_number(center_lat)}"
            f"lon{_format_filename_number(center_lon)}_"
            f"radius{_format_filename_number(radius_km)}km"
        )
    return f"igs_{roi}_{_time_window_csv_tag(start_time, end_time)}.csv"


def export_igs_outputs(
    observations,
    report,
    output_dir,
    prefix="igs_observations",
    center_lat=None,
    center_lon=None,
    radius_km=None,
    start_time=None,
    end_time=None,
):
    """Write IGS CSV and a geographic IPP/ROI diagnostic figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for e in observations:
        t = pd.Timestamp(e.get("date"))
        tec = np.asarray(e.get("tec", []), float).ravel()
        lat = np.asarray(e.get("ipp_lat", []), float).ravel()
        lon = np.asarray(e.get("ipp_lon", []), float).ravel()
        rx_xyz = np.asarray(e.get("LEO", np.empty((3, 0))), dtype=float)
        gnss_xyz = np.asarray(e.get("GNSS", np.empty((3, 0))), dtype=float)

        rx_id = str(e.get("rx_id") or e.get("leo_id") or "").strip().upper()
        rx_id = re.sub(r"\d+$", "", rx_id) or rx_id

        for i, v in enumerate(tec):
            rows.append({
                "accepted": True,
                "reject_reason": "",
                "TEC": v,
                "rx_id": rx_id,
                "PRN": e.get("prn_id", e.get("prn")),
                "year": t.year if t is not pd.NaT else np.nan,
                "date": t.date().isoformat() if t is not pd.NaT else "",
                "time": t.time().isoformat() if t is not pd.NaT else "",
                "IPP_lat": lat[i] if i < len(lat) else np.nan,
                "IPP_lon": lon[i] if i < len(lon) else np.nan,
                "Rx_x": rx_xyz[0, i] if rx_xyz.ndim == 2 and rx_xyz.shape[0] >= 3 and i < rx_xyz.shape[1] else np.nan,
                "Rx_y": rx_xyz[1, i] if rx_xyz.ndim == 2 and rx_xyz.shape[0] >= 3 and i < rx_xyz.shape[1] else np.nan,
                "Rx_z": rx_xyz[2, i] if rx_xyz.ndim == 2 and rx_xyz.shape[0] >= 3 and i < rx_xyz.shape[1] else np.nan,
                "GNSS_x": gnss_xyz[0, i] if gnss_xyz.ndim == 2 and gnss_xyz.shape[0] >= 3 and i < gnss_xyz.shape[1] else np.nan,
                "GNSS_y": gnss_xyz[1, i] if gnss_xyz.ndim == 2 and gnss_xyz.shape[0] >= 3 and i < gnss_xyz.shape[1] else np.nan,
                "GNSS_z": gnss_xyz[2, i] if gnss_xyz.ndim == 2 and gnss_xyz.shape[0] >= 3 and i < gnss_xyz.shape[1] else np.nan,
            })

    csv_path = out / _igs_csv_filename(
        center_lat, center_lon, radius_km, start_time, end_time
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(12, 5))
    if center_lat is not None and center_lon is not None:
        map_crs = ccrs.Orthographic(
            central_longitude=float(center_lon),
            central_latitude=float(center_lat),
        )
    else:
        map_crs = ccrs.Robinson()

    ax_map = fig.add_subplot(121, projection=map_crs)
    ax_tec = fig.add_subplot(122)

    ax_map.set_title("IGS IPP locations + common 5° Fibonacci voxels")
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax_map.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.65, zorder=1)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.55, zorder=1)
    ax_map.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        color="gray",
        alpha=0.35,
        linestyle="--",
    )

    tec_points = []
    labels = []
    for i, e in enumerate(observations):
        lat = np.asarray(e.get("ipp_lat", []), dtype=float).ravel()
        lon = np.asarray(e.get("ipp_lon", []), dtype=float).ravel()
        tec = np.asarray(e.get("tec", []), dtype=float).ravel()
        valid = np.isfinite(lat) & np.isfinite(lon)

        if np.any(valid):
            ax_map.scatter(
                lon[valid], lat[valid],
                transform=ccrs.PlateCarree(),
                s=24, zorder=4,
            )

        for v in tec[np.isfinite(tec)]:
            tec_points.append(float(v))
            labels.append(i)

    if center_lat is not None and center_lon is not None:
        ax_map.scatter(
            [float(center_lon)], [float(center_lat)],
            transform=ccrs.PlateCarree(),
            marker="*", s=95, c="k", zorder=7,
            label=f"Center ({float(center_lat):.1f}°, {float(center_lon):.1f}°)",
        )

        if radius_km is not None and np.isfinite(float(radius_km)):
            # Reuse the project's existing circular/Fibonacci ROI generator.
            fib_lat, fib_lon = circular_roi_points(
                float(center_lat), float(center_lon), float(radius_km),
                spacing_deg=DEFAULT_FIBONACCI_SPACING_DEG,
            )
            if len(fib_lat):
                ax_map.scatter(
                    fib_lon, fib_lat,
                    transform=ccrs.PlateCarree(),
                    s=28, color="k", alpha=0.90, zorder=2,
                    label="Voxels",
                )

            # Draw the exact requested-radius edge for readability. The region
            # itself above comes from the existing Fibonacci ROI code.
            roi_lat, roi_lon = geodesic_circle_latlon(
                float(center_lat), float(center_lon), float(radius_km)
            )
            ax_map.plot(
                roi_lon, roi_lat,
                transform=ccrs.Geodetic(),
                color="green", linewidth=2.0, zorder=6,
                label=f"ROI = {float(radius_km):.0f} km",
            )

    ax_map.legend(loc="lower left", fontsize=8)

    if tec_points:
        ax_tec.scatter(labels, tec_points, s=28)
    ax_tec.set(xlabel="IGS arc number", ylabel="TEC (TECU)", title="TEC")
    ax_tec.grid(True, alpha=0.25)

    fig.suptitle(Path(prefix).name)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

