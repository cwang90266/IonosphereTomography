from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd

from .roi import DEFAULT_FIBONACCI_SPACING_DEG, DEFAULT_FIBONACCI_SPACING_KM
from .roi_selection import haversine_km, los_within_roi, build_roi_dict
from .time_filter import normalize_timestamp, format_filename_number, time_window_tag
from .schema import ObservationEntry
from .netcdf_io import write_observations
from .diagnostics import plot_geolocation

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
    # ipp_distance_km: attached by filter_igs_epochs_by_roi/_by_full_los,
    # not by igs_obs_to_clean_entry -- found missing here via real-data
    # testing (Plan Section 10 step 5): without it, epoch_mode="center"
    # collapsed every other per-epoch field to length 1 but left this one
    # at its original (pre-collapse) length, an internally inconsistent
    # entry that then broke netcdf_io's scalar/array classification.
    "ipp_distance_km",
)


def _dates_touched(start_time, end_time, explicit_date=None) -> list[pd.Timestamp]:
    """
    Return midnight timestamps for the dates whose IGS files may be needed.

    If no time window is supplied, explicit_date is required.
    """
    start = normalize_timestamp(start_time)
    end = normalize_timestamp(end_time)

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
    start = normalize_timestamp(start_time)
    end = normalize_timestamp(end_time)

    if start is None and end is None:
        return list(entries)
    if start is None or end is None:
        raise ValueError("start_time and end_time must be supplied together.")

    out = []
    for entry in entries:
        dt = entry.get("date")
        if dt is None or pd.isna(dt):
            continue
        ts = normalize_timestamp(dt)
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
        dist = float(haversine_km(center_lat, center_lon, lat, lon))
        if dist <= float(radius_km):
            out.append(entry)
    return out


def collapse_igs_arc_to_central_epoch(arc: dict) -> dict:
    """
    Preserve the current static-ionosphere experiment:
    reduce one multi-epoch IGS arc to its central epoch.

    rec_ecef_km/gnss_ecef_km stay shape (3, 1), and all aligned per-epoch
    fields are reduced to length 1.
    """
    out = deepcopy(arc)

    tec = np.asarray(out.get("tec", []))
    n = len(tec)
    if n <= 1:
        return out

    idx = n // 2

    for key in ("rec_ecef_km", "gnss_ecef_km"):
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
        dist_km = haversine_km(
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

        # rec_ecef_km/gnss_ecef_km use shape (3, N)
        new_entry["rec_ecef_km"] = entry["rec_ecef_km"][:, keep]
        new_entry["gnss_ecef_km"] = entry["gnss_ecef_km"][:, keep]

        # Useful diagnostic
        new_entry["ipp_distance_km"] = np.asarray(dist_km)[keep]

        out.append(new_entry)

    return out


def filter_igs_epochs_by_full_los(
    entries: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
    alt_limit_km: float,
    fraction_required: float = 1.0,
    num_segments: int = 200,
) -> list[dict]:
    """
    Per-epoch full-LOS ROI containment (Plan Section 5/10 step 3), the
    IGS analogue of filter_igs_epochs_by_roi: instead of gating each epoch
    on distance from a single fixed-height pierce point, requires at least
    fraction_required of that epoch's own receiver-to-GNSS LOS below
    alt_limit_km to be within radius_km (see roi_selection.los_within_roi).
    """
    out = []

    for entry in entries:
        rec = np.asarray(entry.get("rec_ecef_km", np.empty((3, 0))), dtype=float)
        gnss = np.asarray(entry.get("gnss_ecef_km", np.empty((3, 0))), dtype=float)

        if rec.shape[1] == 0:
            continue

        keep = los_within_roi(
            rec, gnss, center_lat, center_lon, radius_km, alt_limit_km,
            num_segments=num_segments, fraction_required=fraction_required,
        )

        if not np.any(keep):
            continue

        new_entry = dict(entry)

        n = len(entry["tec"])

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

        new_entry["rec_ecef_km"] = rec[:, keep]
        new_entry["gnss_ecef_km"] = gnss[:, keep]

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
    roi_mode: str = "full_los",
    alt_limit_km: float | None = None,
    fraction_required: float = 1.0,
    los_num_segments: int = 200,
    rinex_version: int = 3,
    use_iri: bool = False,
    local_obs_by_station: dict[str, str | Path] | None = None,
    local_nav: str | Path | None = None,
    local_dcb: str | Path | None = None,
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

    local_obs_by_station, local_nav, local_dcb : optional
        Pass-throughs to ``process_igs_station``'s own ``local_obs``/
        ``local_nav``/``local_dcb`` (Plan Section 10 step 5, added while
        verifying against real data): use an already-downloaded RINEX
        observation/navigation/DCB file instead of fetching from CDDIS.
        ``local_obs_by_station`` is a ``{station: path}`` dict since
        observation files are per-station; nav/DCB are shared across
        stations for the same day, so those two are single paths.

    This function adds the higher-level preparation that was scattered across
    the main/demo code:
      1. run each requested station/day,
      2. convert raw pipeline arcs with igs_obs_to_clean_entry(),
      3. keep [start_time, end_time),
      4. optionally apply a per-epoch ROI filter (see roi_mode),
      5. optionally collapse each arc to its central epoch.

    Parameters
    ----------
    epoch_mode : {"center", "all"}
        "center" reproduces the current static-ionosphere experiment.
        "all" keeps the post-downsampling epochs in each arc.
    roi_mode : {"full_los", "pierce_point"}, default "full_los"
        "pierce_point" (today's original behavior) keeps an epoch if its
        fixed-height ionospheric pierce point is within radius_km.
        "full_los" (Plan Section 5, objective 3; the new default per
        Section 8.1) instead requires at least fraction_required of that
        epoch's own receiver-to-GNSS LOS below alt_limit_km to be within
        radius_km (see roi_selection.los_within_roi) -- unlike RO, this
        needs no separate scan-stage pre-filter, since IGS epochs are
        already fully parsed (with real ECEF geometry) before any ROI
        filtering runs.
    alt_limit_km : float, required when roi_mode="full_los"
        No auto-default (Plan Section 8.1): receiver altitude varies
        across epochs in a batch, so there's no single value to derive it
        from without a first pass over the batch.
    fraction_required : float, default 1.0
        Minimum fraction of each epoch's sub-alt_limit_km LOS that must be
        within radius_km (Plan Section 8.2). 1.0 = every such point.
    """
    stations = list(stations)
    cache_dir = Path(cache_dir)

    if epoch_mode not in {"center", "all"}:
        raise ValueError("epoch_mode must be 'center' or 'all'.")
    if roi_mode not in {"full_los", "pierce_point"}:
        raise ValueError(f"roi_mode must be 'full_los' or 'pierce_point', got {roi_mode!r}.")

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
    if roi_requested and roi_mode == "full_los" and alt_limit_km is None:
        raise ValueError(
            "alt_limit_km is required when roi_mode='full_los' (Plan Section 8.1) "
            "-- e.g. the LEO's own altitude, or the model's configured top altitude."
        )

    days = _dates_touched(start_time, end_time, explicit_date=date)

    raw_arc_count = 0
    clean_before_time = []
    station_reports = []

    for day in days:
        pydate = day.to_pydatetime()
        for station in stations:
            try:
                # process_igs_station() has no tlim parameter (only the
                # inner IGSTECPipeline class does, and it's never forwarded)
                # -- found via real-data testing (Plan Section 10 step 5),
                # this call used to raise TypeError every time and get
                # silently swallowed by the except below, so
                # prepare_igs_observations never actually returned any
                # arcs. Not passing tlim costs a little efficiency (arcs
                # outside the window get generated before being discarded)
                # but not correctness: filter_igs_by_time already
                # re-applies [start_time, end_time) below on every arc
                # process_igs_station returns. TEC_model itself is left
                # unmodified (Section 9's non-goal) since this is fixable
                # entirely on this side of the boundary.
                raw_obs = process_igs_station(
                    station=station,
                    date=pydate,
                    rinex_version=rinex_version,
                    cache_dir=cache_dir,
                    use_iri=use_iri,
                    max_rays=max_rays_per_arc,
                    local_obs=(
                        str(local_obs_by_station[station])
                        if local_obs_by_station and station in local_obs_by_station
                        else None
                    ),
                    local_nav=str(local_nav) if local_nav is not None else None,
                    local_dcb=str(local_dcb) if local_dcb is not None else None,
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

                    # rec_ecef_km/gnss_ecef_km: observation_preparation's
                    # schema names (plan Section 6.1/8a) -- this is the
                    # boundary where TEC_model's own 'LEO'/'GNSS' convention
                    # (igs_obs_to_clean_entry's return, out of scope per
                    # Section 8a) gets translated into it.
                    entry["rec_ecef_km"] = entry.pop("LEO")
                    entry["gnss_ecef_km"] = entry.pop("GNSS")

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

    if roi_requested and roi_mode == "full_los":
        after_roi = filter_igs_epochs_by_full_los(
            after_time,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            radius_km=float(radius_km),
            alt_limit_km=float(alt_limit_km),
            fraction_required=float(fraction_required),
            num_segments=los_num_segments,
        )
    elif roi_requested:
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
        "roi_mode": roi_mode if roi_requested else None,
        "roi_radius_km": float(radius_km) if radius_km is not None else None,
        "alt_limit_km": alt_limit_km,
        "fraction_required": float(fraction_required),
        "roi_fibonacci_spacing_deg": float(DEFAULT_FIBONACCI_SPACING_DEG),
        "roi_fibonacci_spacing_km": float(DEFAULT_FIBONACCI_SPACING_KM),
        "station_runs": station_reports,
    }

    if output_dir is not None:
        export_igs_outputs(
            observations, report, output_dir, output_prefix,
            center_lat=center_lat, center_lon=center_lon, radius_km=radius_km,
            start_time=start_time, end_time=end_time,
            roi_mode=roi_mode, alt_limit_km=alt_limit_km, fraction_required=fraction_required,
        )

    return (observations, report) if return_report else observations


def _igs_nc_filename(center_lat, center_lon, radius_km, start_time, end_time) -> str:
    if center_lat is None or center_lon is None or radius_km is None:
        roi = "roi_unspecified"
    else:
        roi = (
            f"lat{format_filename_number(center_lat)}"
            f"lon{format_filename_number(center_lon)}_"
            f"radius{format_filename_number(radius_km)}km"
        )
    return f"igs_{roi}_{time_window_tag(start_time, end_time)}.nc"


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
    roi_mode="pierce_point",
    alt_limit_km=None,
    fraction_required=None,
):
    """Write one netCDF file (Plan Section 6.2) and a combined geolocation
    map. Replaces the old per-epoch CSV outright (Section 8.7). The old
    inline map + "TEC vs. arc number" panel (Section 7.1/10 step 4) is
    replaced by ``diagnostics.plot_geolocation``, shared with RO -- see
    ``diagnostics.plot_tec_comparison`` for a far more informative TEC
    diagnostic than the old arc-number scatter."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = [ObservationEntry.from_dict(e, obs_type="IGS") for e in observations]
    roi = build_roi_dict(
        center_lat, center_lon, radius_km, roi_mode,
        alt_limit_km=alt_limit_km, fraction_required=fraction_required,
    )

    plot_geolocation(entries, roi=roi, output_path=out / f"{prefix}_geolocation.png")

    nc_path = out / _igs_nc_filename(
        center_lat, center_lon, radius_km, start_time, end_time
    )
    write_observations(entries, nc_path, roi=roi)
    report["output_nc_path"] = str(nc_path)

    return nc_path

