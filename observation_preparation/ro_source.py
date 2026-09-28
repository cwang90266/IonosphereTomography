from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import zlib

import netCDF4
import numpy as np
import pandas as pd

from .roi_selection import haversine_km, los_within_roi, build_roi_dict
from .time_filter import normalize_timestamp, format_filename_number, time_window_tag
from .schema import ObservationEntry
from .netcdf_io import write_observations
from .diagnostics import plot_geolocation

from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from Abel_Inverter.lei_abel_inverter import run_abel_inversion


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
    apply_geo_filter: bool = True,
) -> pd.DataFrame:
    """
    Filter RO occultations by TEC-max tangent-point location and peak time.

    Time convention is [start_time, end_time), matching the current main code.

    apply_geo_filter : bool, default True
        Set False for ``roi_mode="full_los"`` (Plan Section 5.3/10 step 3):
        the TEC-max tangent point being within ``radius_km`` is neither
        necessary nor sufficient for the *entire* LOS below
        ``alt_limit_km`` to be inside the ROI, so this cheap scan-stage
        pre-filter would silently reject occultations that full_los would
        otherwise accept (and vice versa). full_los's real decision is made
        per-ray, post-parse, in ``_build_clean_ro_entry`` -- this flag just
        keeps the (still valid, still cheap) time filter below active
        while skipping the geo one.
    """
    if metadata.empty:
        return metadata.copy()

    keep = np.ones(len(metadata), dtype=bool)

    if apply_geo_filter and (
        center_lat is not None or center_lon is not None or radius_km is not None
    ):
        if center_lat is None or center_lon is None or radius_km is None:
            raise ValueError(
                "center_lat, center_lon, and radius_km must either all be set or all be None."
            )
        dist = haversine_km(
            center_lat,
            center_lon,
            metadata["lat"].to_numpy(),
            metadata["lon"].to_numpy(),
        )
        keep &= dist <= float(radius_km)

    start = normalize_timestamp(start_time)
    end = normalize_timestamp(end_time)
    dates = pd.to_datetime(metadata["date"])

    if start is not None:
        keep &= dates.to_numpy() >= np.datetime64(start)
    if end is not None:
        keep &= dates.to_numpy() < np.datetime64(end)

    return metadata.loc[keep].sort_values("date").reset_index(drop=True)


def select_arcs_by_count_bin(
    arc_list: list,
    bin_count: int | None,
    window_key: str,
) -> tuple[list, dict]:
    """
    Randomly subsample *arc_list* down to *bin_count* arcs.

    Vendored from Austin_Demo_Code/test_param_iono.py (Plan Section 8.6):
    that module is outside the surviving module set (Plan Section 8a) and
    this selection logic is genuinely observation_preparation's own concern,
    not a leftover cross-module dependency.

    bin_count=None means "use all available arcs" (no subsampling -- the
    OCC_COUNT_BINS convention for the densest bin).  If arc_list already has
    fewer than bin_count arcs there is nothing to subsample, so the full list
    is returned unchanged.  Otherwise the arcs are subsampled to bin_count
    entries that are both:

      * reproducible -- the RNG is seeded from a stable zlib.crc32 of
        window_key (NOT the process-salted built-in hash()), so re-running the
        same window reproduces the same subsets across runs / restarts, which
        checkpoint-resume and the DA cache rely on; and
      * nested -- every bin_count takes the first bin_count entries of a single
        window-level random permutation, so a smaller bin is always a subset
        of every larger bin (bin=5 ⊂ bin=15 ⊂ … ⊂ all).  The occultation-count
        sweep therefore *adds* measurements between bins instead of drawing an
        unrelated random set each time, isolating measurement density as the
        only variable.

    Returns
    -------
    selected : list
        The chosen arcs (or all of arc_list, per the rules above).
    meta : dict
        "requested_count"  : bin_count as passed in.
        "actual_count"     : len(selected).
        "selected_indices" : indices into arc_list that were kept, ascending.
    """
    n = len(arc_list)

    if bin_count is None or bin_count >= n:
        selected_indices = list(range(n))
        selected = list(arc_list)
    else:
        # Reproducible seed: Python's built-in hash() is salted per process
        # (PYTHONHASHSEED), so it draws a different subset every run and
        # desyncs the DA cache.  zlib.crc32 is a stable, process-independent
        # hash of the window key, so the same window always seeds identically.
        seed = zlib.crc32(str(window_key).encode("utf-8"))
        rng = np.random.default_rng(seed)
        # NESTED subsets: draw ONE reproducible random permutation of all n
        # arcs (the seed depends only on window_key, NOT bin_count) and take
        # its first bin_count entries.  Because every bin_count reuses the
        # same permutation, a smaller bin is always a subset of a larger one
        # (bin=5 ⊂ bin=15 ⊂ … ⊂ all), so the occultation-count sweep *adds*
        # measurements rather than swapping to an unrelated random draw --
        # removing the "which arcs happened to be picked" confound from the
        # count-sensitivity study while keeping the selection random.
        perm = rng.permutation(n)
        selected_indices = sorted(int(i) for i in perm[:bin_count])
        selected = [arc_list[i] for i in selected_indices]

    meta = dict(
        requested_count=bin_count,
        actual_count=len(selected),
        selected_indices=selected_indices,
    )
    return selected, meta


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
    *,
    roi_mode: str = "tangent_point",
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
    alt_limit_km: float | None = None,
    fraction_required: float = 1.0,
    los_num_segments: int = 200,
) -> tuple[dict | None, dict]:
    """
    Reproduce the RO clean-list preparation formerly inside process_group().

    parse_podTc2_nc_file() has already applied its file-level geometry QC.
    This function preserves the second-stage rules:
      finite TEC, TEC > 0, >= min_valid_rays, uniform stride decimation.

    roi_mode="full_los" (Plan Section 5/10 step 3) additionally requires
    each individual ray's own LOS, below alt_limit_km, to be within
    radius_km of (center_lat, center_lon) -- applied here (per-ray, on the
    full pre-decimation geometry) rather than in filter_ro_metadata's
    lightweight scan-stage pre-filter, since it needs the full parsed
    LEO/GNSS geometry that only exists after parse_podTc2_nc_file().
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

    if roi_mode == "full_los":
        valid &= los_within_roi(
            data["LEO"], data["GNSS"],
            center_lat, center_lon, radius_km, alt_limit_km,
            num_segments=los_num_segments, fraction_required=fraction_required,
        )

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
        # rec_ecef_km/gnss_ecef_km: observation_preparation's schema names
        # (plan Section 6.1/8a) -- data['LEO']/data['GNSS'] above is
        # TEC_model's own raw parser output and keeps its own convention;
        # this is the boundary where the rename happens.
        "rec_ecef_km": np.asarray(data["LEO"][:, mask], dtype=np.float64),
        "gnss_ecef_km": np.asarray(data["GNSS"][:, mask], dtype=np.float64),
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
    roi_mode: str = "full_los",
    alt_limit_km: float | None = None,
    fraction_required: float = 1.0,
    los_num_segments: int = 200,
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
    2. Filter by [start_time, end_time), and by ROI (roi_mode="tangent_point"
       only -- see roi_mode below).
    3. Optionally subsample the number of occultations using the project's
       select_arcs_by_count_bin() routine.
    4. Parse each selected podTc2 file. The parser performs the existing
       file-level geometry QC.
    5. Keep finite positive TEC only, and (roi_mode="full_los") rays whose
       own LOS satisfies the full-LOS ROI containment check.
    6. Reject occultations with fewer than min_valid_rays.
    7. Uniformly decimate to max_rays_per_occultation.
    8. If include_abel=True, run Lei-Abel inversion on the FULL parsed RO arc
       (not on the downsampled tomography rays) and attach the result as
       entry["abel"].

    Parameters
    ----------
    roi_mode : {"full_los", "tangent_point"}, default "full_los"
        "tangent_point" (today's original behavior) keeps an occultation
        if its TEC-max tangent point is within radius_km -- a cheap,
        scan-stage-only check. "full_los" (Plan Section 5, objective 3;
        the new default per Section 8.1) instead requires, per ray, that
        at least fraction_required of that ray's own LOS below
        alt_limit_km stay within radius_km -- a stricter, per-ray check
        made after parsing (see _build_clean_ro_entry), since it needs
        the full receiver/transmitter geometry. Because that geometry
        isn't available at the cheap scan stage, "full_los" parses every
        file in [start_time, end_time) regardless of location -- a known,
        accepted cost (Plan Section 10 step 3), not optimized here.
    alt_limit_km : float, required when roi_mode="full_los"
        No auto-default (Plan Section 8.1): receiver altitude varies
        across LEOs in a batch, so there's no single value to derive it
        from without a first pass over the batch.
    fraction_required : float, default 1.0
        Minimum fraction of each ray's sub-alt_limit_km LOS that must be
        within radius_km (Plan Section 8.2). 1.0 = every such point.

    Returns
    -------
    list[dict]
        Clean observation entries ready for observation-operator construction.

    If return_report=True:
        (observations, report)
    """
    if roi_mode not in {"full_los", "tangent_point"}:
        raise ValueError(f"roi_mode must be 'full_los' or 'tangent_point', got {roi_mode!r}.")
    if roi_mode == "full_los" and alt_limit_km is None:
        raise ValueError(
            "alt_limit_km is required when roi_mode='full_los' (Plan Section 8.1) "
            "-- e.g. the LEO's own altitude, or the model's configured top altitude."
        )

    podtc_dir = Path(podtc_dir)
    if not podtc_dir.is_dir():
        raise FileNotFoundError(f"RO directory does not exist: {podtc_dir}")

    apply_geo_filter = roi_mode != "full_los"

    metadata_all = scan_ro_metadata(podtc_dir, file_pattern=file_pattern)
    metadata_selected = filter_ro_metadata(
        metadata_all,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        start_time=start_time,
        end_time=end_time,
        apply_geo_filter=apply_geo_filter,
    )

    if selection_key is None:
        if start_time is not None and end_time is not None:
            selection_key = (
                f"{normalize_timestamp(start_time)}__{normalize_timestamp(end_time)}"
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
            roi_mode=roi_mode,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            alt_limit_km=alt_limit_km,
            fraction_required=fraction_required,
            los_num_segments=los_num_segments,
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
                apply_geo_filter=apply_geo_filter,
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
        "roi_mode": roi_mode,
        "alt_limit_km": alt_limit_km,
        "fraction_required": float(fraction_required),
        "per_file": per_file,
        "metadata_selected": metadata_selected.to_dict("records"),
    }

    if output_dir is not None:
        export_ro_outputs(
            observations, report, output_dir, output_prefix,
            center_lat=center_lat, center_lon=center_lon, radius_km=radius_km,
            start_time=start_time, end_time=end_time,
            roi_mode=roi_mode, alt_limit_km=alt_limit_km, fraction_required=fraction_required,
        )

    return (observations, report) if return_report else observations


def _ro_nc_filename(center_lat, center_lon, radius_km, start_time, end_time) -> str:
    if center_lat is None or center_lon is None or radius_km is None:
        roi = "roi_unspecified"
    else:
        roi = (
            f"lat{format_filename_number(center_lat)}"
            f"lon{format_filename_number(center_lon)}_"
            f"radius{format_filename_number(radius_km)}km"
        )
    return f"ro_{roi}_{time_window_tag(start_time, end_time)}.nc"


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
    roi_mode="tangent_point",
    alt_limit_km=None,
    fraction_required=None,
):
    """Write one netCDF file (Plan Section 6.2) and one combined
    geolocation map. Replaces the old per-ray CSV + companion Abel-profile
    CSV outright (Section 8.7) -- the Abel profiles now live in the same
    netCDF file as a properly dimensioned ``abel_level`` variable instead
    of being joined to the ray table only by filename string. Also
    replaces the old one-PNG-per-occultation map (Section 7.1/10 step 4)
    with a single combined plot via ``diagnostics.plot_geolocation``; call
    ``diagnostics.plot_obs_detail`` yourself for the per-occultation
    TEC/Abel panels the old PNGs also carried."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = [ObservationEntry.from_dict(e, obs_type="RO") for e in observations]
    roi = build_roi_dict(
        center_lat, center_lon, radius_km, roi_mode,
        alt_limit_km=alt_limit_km, fraction_required=fraction_required,
    )

    plot_geolocation(entries, roi=roi, output_path=out / f"{prefix}_geolocation.png")

    nc_path = out / _ro_nc_filename(
        center_lat, center_lon, radius_km, start_time, end_time
    )
    write_observations(entries, nc_path, roi=roi)
    report["output_nc_path"] = str(nc_path)

    return nc_path

