from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .roi_tools import (
    circular_roi_points,
    geodesic_circle_latlon,
    DEFAULT_FIBONACCI_SPACING_DEG,
    DEFAULT_FIBONACCI_SPACING_KM,
)

from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent, ECEFtolla
from Abel_Inverter.lei_abel_inverter import run_abel_inversion


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


def _normalize_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _read_ro_metadata(file_path: Path) -> dict[str, Any] | None:
    """Read only the netCDF attributes needed for spatial/time selection."""
    try:
        with netCDF4.Dataset(file_path, "r") as nc:
            lat = float(nc.getncattr("lat_tecmax_tangent"))
            lon = float(nc.getncattr("lon_tecmax_tangent"))

            year = int(nc.getncattr("year"))
            month = int(nc.getncattr("month"))
            day = int(nc.getncattr("day"))
            hour = int(nc.getncattr("hour"))
            minute = int(nc.getncattr("minute"))
            second = int(float(nc.getncattr("second")))

            date = pd.Timestamp(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
            )

            return {
                "filename": file_path.name,
                "full_path": str(file_path),
                "date": date,
                "lat": lat,
                "lon": lon,
                "leo_id": str(nc.getncattr("leo_id")) if "leo_id" in nc.ncattrs() else "",
                "conid": str(nc.getncattr("conid")) if "conid" in nc.ncattrs() else "",
                "prn_id": str(nc.getncattr("prn_id")) if "prn_id" in nc.ncattrs() else "",
            }
    except Exception:
        return None


def scan_ro_metadata(
    podtc_dir: str | Path,
    file_pattern: str = "*.0001_nc",
) -> pd.DataFrame:
    """Build a lightweight metadata table without parsing full RO files."""
    podtc_dir = Path(podtc_dir)
    rows = []

    for file_path in sorted(podtc_dir.glob(file_pattern)):
        row = _read_ro_metadata(file_path)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "filename", "full_path", "date", "lat", "lon",
                "leo_id", "conid", "prn_id",
            ]
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def filter_ro_metadata(
    metadata: pd.DataFrame,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    start_time=None,
    end_time=None,
) -> pd.DataFrame:
    """
    Filter RO occultations by TEC-max tangent-point location and peak time.

    Time convention is [start_time, end_time), matching the current main code.
    """
    if metadata.empty:
        return metadata.copy()

    keep = np.ones(len(metadata), dtype=bool)

    if center_lat is not None or center_lon is not None or radius_km is not None:
        if center_lat is None or center_lon is None or radius_km is None:
            raise ValueError(
                "center_lat, center_lon, and radius_km must either all be set or all be None."
            )
        dist = _haversine_km(
            center_lat,
            center_lon,
            metadata["lat"].to_numpy(),
            metadata["lon"].to_numpy(),
        )
        keep &= dist <= float(radius_km)

    start = _normalize_timestamp(start_time)
    end = _normalize_timestamp(end_time)
    dates = pd.to_datetime(metadata["date"])

    if start is not None:
        keep &= dates.to_numpy() >= np.datetime64(start)
    if end is not None:
        keep &= dates.to_numpy() < np.datetime64(end)

    return metadata.loc[keep].sort_values("date").reset_index(drop=True)


def select_ro_occultations(
    metadata: pd.DataFrame,
    max_occultations: int | None,
    selection_key: str,
) -> pd.DataFrame:
    """
    Apply the project's existing occultation-count selection logic.

    None means use every selected occultation.
    """
    if max_occultations is None or len(metadata) <= max_occultations:
        return metadata.reset_index(drop=True)

    from test_param_iono import select_arcs_by_count_bin

    dummy_arcs = list(range(len(metadata)))
    _, selection_meta = select_arcs_by_count_bin(
        dummy_arcs,
        max_occultations,
        selection_key,
    )
    idx = selection_meta["selected_indices"]
    return metadata.iloc[idx].reset_index(drop=True)


def _build_clean_ro_entry(
    data: dict,
    label: str,
    min_valid_rays: int,
    max_rays_per_occultation: int,
) -> tuple[dict | None, dict]:
    """
    Reproduce the RO clean-list preparation formerly inside process_group().

    parse_podTc2_nc_file() has already applied its file-level geometry QC.
    This function preserves the second-stage rules:
      finite TEC, TEC > 0, >= min_valid_rays, uniform stride decimation.
    """
    _, _, tang_raw = rayTangent(data["LEO"], data["GNSS"], units="km")

    # rayTangent(..., units="km") returns geodetic altitude in metres from
    # pyproj, so the existing process_group() multiplies by 1e-3 here.
    tang_km = np.asarray(tang_raw, dtype=float) * 1e-3

    meas_tec = np.asarray(
        data.get("TEC_podTc2", data.get("TEC", np.zeros_like(tang_km))),
        dtype=float,
    )

    valid = np.isfinite(meas_tec) & (meas_tec > 0)
    n_valid = int(valid.sum())

    info = {
        "label": label,
        "n_valid_before_downsample": n_valid,
        "accepted": False,
        "reject_reason": None,
        "n_output_rays": 0,
        "stride": None,
    }

    if n_valid < int(min_valid_rays):
        info["reject_reason"] = "too_few_valid_rays"
        return None, info

    if n_valid > int(max_rays_per_occultation):
        stride = int(math.ceil(n_valid / int(max_rays_per_occultation)))
        dec_idx = np.where(valid)[0][::stride]
        mask = np.zeros(len(meas_tec), dtype=bool)
        mask[dec_idx] = True
    else:
        stride = 1
        mask = valid

    leo_id = str(data.get("leo_id", "??")).strip()
    con_id = str(data.get("conid", "?")).strip()
    prn_num = str(data.get("prn_id", "??")).strip()
    full_prn = f"{con_id}{prn_num}"

    entry = {
        "tec": np.asarray(meas_tec[mask], dtype=np.float64).flatten(),
        "tangent_km": np.asarray(tang_km[mask], dtype=np.float64).flatten(),
        "LEO": np.asarray(data["LEO"][:, mask], dtype=np.float64),
        "GNSS": np.asarray(data["GNSS"][:, mask], dtype=np.float64),
        "tec_type": "absolute",
        "leo_id": leo_id,
        "prn_id": full_prn,
        "label": label,
        "obs_source": "RO_podTc2",
        "date": pd.Timestamp(data.get("date")) if data.get("date") is not None else pd.NaT,
        "lat_tecmax_tangent": float(data.get("lat_tecmax_tangent", np.nan)),
        "lon_tecmax_tangent": float(data.get("lon_tecmax_tangent", np.nan)),
        "occ_type": data.get("occ_type"),
    }

    # Optional SNR arrays are diagnostic only.  Some podTc2 parser versions
    # leave SNR at the original raw-file length even after TEC/LEO/GNSS have
    # been geometry-QC masked.  Never index a mismatched diagnostic array with
    # the post-QC TEC mask.  Keep aligned SNR when available; otherwise return
    # NaNs of the correct output-ray length and record the mismatch in the
    # per-file report instead of rejecting an otherwise valid occultation.
    n_parser_rays = len(meas_tec)
    n_out = int(mask.sum())

    for src_key, out_key in (("caL1_SNR", "snr_l1"), ("pL2_SNR", "snr_l2")):
        if src_key not in data:
            continue

        arr = np.asarray(data[src_key])
        arr = np.ma.filled(arr, np.nan) if np.ma.isMaskedArray(arr) else arr
        arr = np.asarray(arr, dtype=np.float64).squeeze()

        if arr.ndim == 1 and len(arr) == n_parser_rays:
            entry[out_key] = np.asarray(arr[mask], dtype=np.float64).flatten()
        else:
            entry[out_key] = np.full(n_out, np.nan, dtype=np.float64)
            info[f"{src_key}_alignment"] = (
                f"skipped_unaligned: shape={arr.shape}, parser_rays={n_parser_rays}"
            )

    info["accepted"] = True
    info["n_output_rays"] = int(mask.sum())
    info["stride"] = stride
    return entry, info


def prepare_ro_observations(
    podtc_dir: str | Path,
    *,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    start_time=None,
    end_time=None,
    max_occultations: int | None = None,
    selection_key: str | None = None,
    min_valid_rays: int = 50,
    max_rays_per_occultation: int = 200,
    file_pattern: str = "*.0001_nc",
    include_abel: bool = True,
    return_report: bool = False,
    output_dir: str | Path | None = None,
    output_prefix: str = "ro_observations",
):
    """
    Prepare RO observations for tomography, with no KF/EKF/voxel dependency.

    Processing order
    ----------------
    1. Scan lightweight netCDF metadata.
    2. Filter by ROI and [start_time, end_time).
    3. Optionally subsample the number of occultations using the project's
       select_arcs_by_count_bin() routine.
    4. Parse each selected podTc2 file. The parser performs the existing
       file-level geometry QC.
    5. Keep finite positive TEC only.
    6. Reject occultations with fewer than min_valid_rays.
    7. Uniformly decimate to max_rays_per_occultation.
    8. If include_abel=True, run Lei-Abel inversion on the FULL parsed RO arc
       (not on the downsampled tomography rays) and attach the result as
       entry["abel"].

    Returns
    -------
    list[dict]
        Clean observation entries ready for observation-operator construction.

    If return_report=True:
        (observations, report)
    """
    podtc_dir = Path(podtc_dir)
    if not podtc_dir.is_dir():
        raise FileNotFoundError(f"RO directory does not exist: {podtc_dir}")

    metadata_all = scan_ro_metadata(podtc_dir, file_pattern=file_pattern)
    metadata_selected = filter_ro_metadata(
        metadata_all,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        start_time=start_time,
        end_time=end_time,
    )

    if selection_key is None:
        if start_time is not None and end_time is not None:
            selection_key = (
                f"{_normalize_timestamp(start_time)}__{_normalize_timestamp(end_time)}"
            )
        else:
            selection_key = str(podtc_dir)

    metadata_selected = select_ro_occultations(
        metadata_selected,
        max_occultations=max_occultations,
        selection_key=selection_key,
    )

    observations = []
    per_file = []
    n_parser_reject = 0
    n_abel_success = 0
    n_abel_failed = 0

    for row in metadata_selected.itertuples(index=False):
        label = Path(row.full_path).name
        try:
            data = parse_podTc2_nc_file(row.full_path)
        except Exception as exc:
            n_parser_reject += 1
            per_file.append({
                "label": label,
                "accepted": False,
                "reject_reason": f"parser_exception: {type(exc).__name__}",
                "n_valid_before_downsample": 0,
                "n_output_rays": 0,
                "stride": None,
            })
            continue

        if data is None:
            n_parser_reject += 1
            per_file.append({
                "label": label,
                "accepted": False,
                "reject_reason": "parser_qc_reject",
                "n_valid_before_downsample": 0,
                "n_output_rays": 0,
                "stride": None,
            })
            continue

        entry, info = _build_clean_ro_entry(
            data,
            label=label,
            min_valid_rays=min_valid_rays,
            max_rays_per_occultation=max_rays_per_occultation,
        )
        if entry is not None and include_abel:
            # IMPORTANT: Abel must use the full parsed podTc2 arc.  The clean
            # entry has already been uniformly decimated for tomography, but
            # ``data`` still contains the complete parser output.
            try:
                abel = run_abel_inversion(data)
                if abel is None or len(abel.get("Ne", [])) == 0:
                    abel = None
            except Exception as exc:
                abel = None
                info["abel_error"] = f"{type(exc).__name__}: {exc}"

            entry["abel"] = abel
            if abel is None:
                n_abel_failed += 1
                info["abel_status"] = "failed"
            else:
                n_abel_success += 1
                info["abel_status"] = "ok"
                info["n_abel_levels"] = int(len(abel.get("Ne", [])))
        elif entry is not None:
            entry["abel"] = None
            info["abel_status"] = "disabled"

        per_file.append(info)
        if entry is not None:
            observations.append(entry)

    report = {
        "source": "RO_podTc2",
        "podtc_dir": str(podtc_dir),
        "n_files_scanned": int(len(metadata_all)),
        "n_files_after_roi_time_filter": int(
            len(filter_ro_metadata(
                metadata_all,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                start_time=start_time,
                end_time=end_time,
            ))
        ),
        "n_files_after_occultation_selection": int(len(metadata_selected)),
        "n_parser_reject": int(n_parser_reject),
        "n_clean_observations": int(len(observations)),
        "n_output_rays_total": int(sum(len(x["tec"]) for x in observations)),
        "include_abel": bool(include_abel),
        "n_abel_success": int(n_abel_success),
        "n_abel_failed": int(n_abel_failed),
        "min_valid_rays": int(min_valid_rays),
        "max_rays_per_occultation": int(max_rays_per_occultation),
        "max_occultations": max_occultations,
        "per_file": per_file,
        "metadata_selected": metadata_selected.to_dict("records"),
    }

    if output_dir is not None:
        export_ro_outputs(
            observations, report, output_dir, output_prefix,
            center_lat=center_lat, center_lon=center_lon, radius_km=radius_km,
            start_time=start_time, end_time=end_time,
        )

    return (observations, report) if return_report else observations


def _format_filename_number(value) -> str:
    """Compact, filesystem-safe numeric token used in exported CSV names."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _time_window_csv_tag(start_time, end_time) -> str:
    """Return e.g. ``timewindow1hr_20251118_10001100``."""
    start = _normalize_timestamp(start_time)
    end = _normalize_timestamp(end_time)
    if start is None or end is None:
        return "timewindow_unknown"

    minutes = (end - start).total_seconds() / 60.0
    if np.isclose(minutes % 60.0, 0.0):
        hours = minutes / 60.0
        duration = f"{_format_filename_number(hours)}hr"
    else:
        duration = f"{_format_filename_number(minutes)}min"

    if start.date() == end.date():
        return f"timewindow{duration}_{start:%Y%m%d}_{start:%H%M}{end:%H%M}"
    return f"timewindow{duration}_{start:%Y%m%d_%H%M}_{end:%Y%m%d_%H%M}"


def _ro_csv_filename(center_lat, center_lon, radius_km, start_time, end_time) -> str:
    if center_lat is None or center_lon is None or radius_km is None:
        roi = "roi_unspecified"
    else:
        roi = (
            f"lat{_format_filename_number(center_lat)}"
            f"lon{_format_filename_number(center_lon)}_"
            f"radius{_format_filename_number(radius_km)}km"
        )
    return f"ro_{roi}_{_time_window_csv_tag(start_time, end_time)}.csv"


def export_ro_outputs(
    observations,
    report,
    output_dir,
    prefix="ro_observations",
    center_lat=None,
    center_lon=None,
    radius_km=None,
    start_time=None,
    end_time=None,
):
    """Write per-ray RO CSV and one three-panel figure per accepted occultation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted = {x.get("label") for x in observations}
    rows = []
    abel_rows = []

    # Keep one row for selected files rejected by parser/clean QC.
    for meta in report.get("metadata_selected", []):
        label = Path(str(meta.get("filename", ""))).name
        if label not in accepted:
            reject_reason = ";".join(
                str(item.get("reject_reason", ""))
                for item in report.get("per_file", [])
                if item.get("label") == label
            )
            # Do not export the redundant metadata ``year`` field.
            clean_meta = {k: v for k, v in meta.items() if k != "year"}
            rows.append({
                "accepted": False,
                "reject_reason": reject_reason,
                "filename": label,
                **clean_meta,
            })

    for e in observations:
        n = len(e["tec"])
        ab = e.get("abel") or {}
        an = np.asarray(ab.get("Ne", []), float).ravel()
        aa = np.asarray(ab.get("alt_km", ab.get("alt", [])), float).ravel()
        at = np.asarray(ab.get("TEC_cal", []), float).ravel()
        af = np.asarray(ab.get("TEC_forward", []), float).ravel()
        t = pd.Timestamp(e.get("date"))

        # IMPORTANT: Abel profile length is generally NOT the same as the
        # tomography-clean ray count because Abel runs on the full parsed arc
        # while the observation entry can be downsampled to <= max_rays.
        # Store the complete Abel profile in a companion profile table instead
        # of index-pairing/truncating it onto the ray table.
        n_ab = max(len(an), len(aa), len(at), len(af))
        for j in range(n_ab):
            abel_rows.append({
                "filename": e.get("label"),
                "profile_index": j,
                "Abel_Ne": an[j] if j < len(an) else np.nan,
                "Abel_alt_km": aa[j] if j < len(aa) else np.nan,
                "Abel_TEC_cal_TECU": at[j] if j < len(at) else np.nan,
                "Abel_TEC_forward_TECU": af[j] if j < len(af) else np.nan,
            })

        for i in range(n):
            rows.append({
                "accepted": True,
                "reject_reason": "",
                "filename": e.get("label"),
                "LEO": e.get("leo_id"),
                "PRN": e.get("prn_id"),
                # ``year`` intentionally omitted: it duplicates ``date``.
                "date": t.date().isoformat() if t is not pd.NaT else "",
                "time": t.time().isoformat() if t is not pd.NaT else "",
                "setting_rising": e.get("occ_type"),
                "TEC": e["tec"][i],
                "tangent_height_km": e["tangent_km"][i],
                "LEO_x": e["LEO"][0, i],
                "LEO_y": e["LEO"][1, i],
                "LEO_z": e["LEO"][2, i],
                "GNSS_x": e["GNSS"][0, i],
                "GNSS_y": e["GNSS"][1, i],
                "GNSS_z": e["GNSS"][2, i],
            })

        _plot_ro_event(
            e,
            out / f"{prefix}_{e.get('label', 'event')}.png",
            center_lat,
            center_lon,
            radius_km,
        )

    csv_path = out / _ro_csv_filename(
        center_lat, center_lon, radius_km, start_time, end_time
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Full Abel profiles: one row per Abel altitude level, never truncated to
    # the downsampled ray table length.
    abel_csv_path = csv_path.with_name(csv_path.stem + "_abel_profiles.csv")
    pd.DataFrame(abel_rows).to_csv(abel_csv_path, index=False)
    report["abel_profiles_csv"] = str(abel_csv_path)

    return csv_path

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



def _plot_ro_event(e, path, center_lat, center_lon, radius_km):
    """Plot RO tangent track over a geographic map, plus TEC and Abel Ne."""
    fig = plt.figure(figsize=(15, 5))

    if center_lat is not None and center_lon is not None:
        map_crs = ccrs.Orthographic(
            central_longitude=float(center_lon),
            central_latitude=float(center_lat),
        )
    else:
        map_crs = ccrs.Robinson()

    ax_map = fig.add_subplot(131, projection=map_crs)
    ax_map.set_title("RO tangent track")
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

    # Recompute tangent-point ECEF from the SAME downsampled LEO/GNSS geometry
    # stored in the clean observation, then convert to geodetic coordinates.
    tangent_xyz, _, _ = rayTangent(
        np.asarray(e["LEO"], dtype=float),
        np.asarray(e["GNSS"], dtype=float),
        units="km",
    )
    tangent_lat, tangent_lon, _ = ECEFtolla(tangent_xyz)
    tangent_lat = np.asarray(tangent_lat, dtype=float).ravel()
    tangent_lon = np.asarray(tangent_lon, dtype=float).ravel()
    tangent_alt = np.asarray(e.get("tangent_km", []), dtype=float).ravel()

    valid = (
        np.isfinite(tangent_lat)
        & np.isfinite(tangent_lon)
        & np.isfinite(tangent_alt)
    )

    if np.any(valid):
        lat_v = tangent_lat[valid]
        lon_v = tangent_lon[valid]
        alt_v = tangent_alt[valid]

        ax_map.plot(
            lon_v, lat_v,
            transform=ccrs.Geodetic(),
            color="red",
            linewidth=2.2,
            zorder=4,
            label="RO tangent track",
        )
        sc = ax_map.scatter(
            lon_v, lat_v,
            transform=ccrs.PlateCarree(),
            c=alt_v,
            s=16,
            zorder=5,
        )
        cb = fig.colorbar(sc, ax=ax_map, pad=0.04, shrink=0.78)
        cb.set_label("Tangent altitude (km)")

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
                float(center_lat),
                float(center_lon),
                float(radius_km),
                spacing_deg=DEFAULT_FIBONACCI_SPACING_DEG,
            )
            if len(fib_lat):
                ax_map.scatter(
                    fib_lon, fib_lat,
                    transform=ccrs.PlateCarree(),
                    s=28, color="k", alpha=0.90, zorder=2,
                    label="voxels",
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

    ax_tec = fig.add_subplot(132)
    ax_tec.plot(e["tec"], e["tangent_km"])
    ax_tec.set(xlabel="TEC (TECU)", ylabel="Tangent height (km)")
    ax_tec.grid(True, alpha=0.25)

    ax_abel = fig.add_subplot(133)
    ab = e.get("abel") or {}
    ax_abel.plot(ab.get("Ne", []), ab.get("alt_km", ab.get("alt", [])))
    ax_abel.set(xlabel="Abel Ne (m$^{-3}$)", ylabel="Altitude (km)")
    ax_abel.grid(True, alpha=0.25)

    fig.suptitle(Path(path).name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

