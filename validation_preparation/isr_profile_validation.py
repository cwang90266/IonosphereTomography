"""Validate one model Ne profile against real ISR observations.

This module is intentionally self-contained for the validation-preparation
refactor.  It does not import any ``demo_*`` module or plotting/main script.
The instrument definitions and ALT_GRID are copied from
``demo_isr_initial_conditions.py`` so downstream validation code no longer
needs to call that demo module.
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISR_DATA_DIR = PROJECT_ROOT / "Data" / "ISR_Data"

# Copied from demo_isr_initial_conditions.py during rearrangement.
ALT_GRID = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
INSTRUMENTS = {
    "ESR": {
        "lat": 78.09, "lon": 16.02,
        "lat_bounds": (60.0, 88.0),
        "lon_bounds": (-20.0, 60.0),
        "label": "EISCAT Svalbard Radar",
    },
    "TRO": {
        "lat": 69.583, "lon": 19.21,
        "lat_bounds": (55.0, 80.0),
        "lon_bounds": (-5.0, 45.0),
        "label": "EISCAT Tromsø UHF Radar",
    },
    "JRO": {
        "lat": -11.95, "lon": -76.87,
        "lat_bounds": (-25.0, 15.0),
        "lon_bounds": (-105.0, -45.0),
        "label": "Jicamarca IS Radar",
    },
}

# Project value previously imported from plotIonosphereTomography.py.
ISR_MIN_VALID_GATES = 5
ISR_SITE_MATCH_DEG = 0.5


def _utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("Invalid timestamp.")
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _resolve_window(time_window, time, doy, year, half_minutes):
    """Use exact supplied RO bounds, or construct a centered window."""
    if time_window is not None:
        lo = _utc(time_window["lo"])
        hi = _utc(time_window["hi"])
        centre = _utc(time_window["t_centre"])
    else:
        if any(v is None for v in (time, doy, year)):
            raise ValueError("Supply time_window, or time/doy/year.")
        day = pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")
        day += pd.Timedelta(days=int(doy) - 1)
        if isinstance(time, (int, float, np.number)):
            centre = day + pd.Timedelta(hours=float(time))
        elif isinstance(time, str) and "-" not in time:
            fields = [float(v) for v in time.split(":")]
            if len(fields) not in (2, 3):
                raise ValueError("Use HH:MM[:SS] for clock time.")
            centre = day + pd.Timedelta(
                hours=fields[0], minutes=fields[1],
                seconds=fields[2] if len(fields) == 3 else 0,
            )
        else:
            centre = _utc(time)
        delta = pd.Timedelta(minutes=float(half_minutes))
        lo, hi = centre - delta, centre + delta
    if not lo <= centre <= hi or lo == hi:
        raise ValueError("Require lo <= t_centre <= hi and lo < hi.")
    print(f"[Validation window] {lo} through {hi} (inclusive)")
    return lo, hi, centre


def _nearest_site(lat, lon, sites, max_distance_km=None):
    distances = {}
    for name, site in sites.items():
        p1, p2 = np.radians([lat, site["lat"]])
        dlon = np.radians(site["lon"] - lon)
        a = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
        distances[name] = 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    name = min(distances, key=distances.get)
    distance = float(distances[name])
    if max_distance_km is not None and distance > max_distance_km:
        raise ValueError(f"Nearest station {name} is {distance:.1f} km away.")
    print(f"[Station] {name}; input-to-station distance {distance:.1f} km")
    return name, sites[name], distance


def _profile(altitude, density):
    alt = np.ma.asarray(altitude, dtype=float).filled(np.nan).ravel()
    ne = np.ma.asarray(density, dtype=float).filled(np.nan).ravel()
    if alt.size != ne.size:
        raise ValueError("Altitude and Ne arrays must have equal lengths.")
    ne = np.where(np.isfinite(ne) & (ne > 0), ne, np.nan)
    valid_alt = np.isfinite(alt)
    frame = pd.DataFrame({"alt": alt[valid_alt], "ne": ne[valid_alt]})
    frame = frame.groupby("alt", sort=True, as_index=False)["ne"].mean()
    return frame["alt"].to_numpy(), frame["ne"].to_numpy()


def _availability(label, times):
    times = sorted(times)
    if not times:
        print(f"[{label} availability] No records")
        return
    print(f"[{label} availability] {times[0]} through {times[-1]} "
          f"({len(times)} records; coverage may contain gaps)")


def observation_files(root):
    """Return ISR observation files below *root* without relying on demo code."""
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    patterns = ("*.nc", "*.nc4", "*.h5", "*.hdf5")
    files = set()
    for pattern in patterns:
        files.update(root.rglob(pattern))
    return sorted(files)


def _variable(ds, names, required=True):
    for name in names:
        if name in ds.variables:
            return np.ma.asarray(ds.variables[name][:]).filled(np.nan).squeeze()
    if required:
        raise KeyError(f"None of variables {names} found in {ds.filepath()}.")
    return None


def _scalar_from_dataset(ds, names):
    """Read first finite scalar from variable or global attribute candidates."""
    for name in names:
        if name in ds.variables:
            arr = np.ma.asarray(ds.variables[name][:]).filled(np.nan).astype(float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(arr[0])
        if hasattr(ds, name):
            try:
                value = float(getattr(ds, name))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return np.nan


def _kindat_from_dataset(ds, path):
    for name in ("kindat", "kindat_code", "data_type"):
        if name in ds.variables:
            arr = np.ma.asarray(ds.variables[name][:]).filled(np.nan).ravel()
            if arr.size:
                try:
                    return str(int(float(arr[0])))
                except Exception:
                    return str(arr[0])
        if hasattr(ds, name):
            return str(getattr(ds, name)).strip()
    match = re.search(r"(?:MAD|kindat)?(\d{4})", path.name, re.IGNORECASE)
    return match.group(1) if match else ""


def _as_time_profiles(values, n_time):
    values = np.ma.asarray(values, dtype=float).filled(np.nan)
    if values.ndim == 1:
        return np.repeat(values[np.newaxis, :], n_time, axis=0)
    axes = [axis for axis, size in enumerate(values.shape) if size == n_time]
    if not axes:
        raise ValueError(f"Cannot identify time dimension {n_time} in shape {values.shape}.")
    values = np.moveaxis(values, axes[-1], 0)
    return values.reshape(n_time, -1)


def _timestamps_to_utc(ds, raw):
    raw = np.ma.asarray(raw).filled(np.nan).ravel()
    # Unix seconds are what the current Madrigal NetCDFs use.
    variable = None
    for name in ("timestamps", "timestamp", "time"):
        if name in ds.variables:
            variable = ds.variables[name]
            break
    units = str(getattr(variable, "units", "")).strip() if variable is not None else ""
    if "since" in units.lower():
        calendar = str(getattr(variable, "calendar", "standard"))
        out = []
        for value in raw:
            if not np.isfinite(value):
                out.append(None)
                continue
            dt = netCDF4.num2date(value, units=units, calendar=calendar)
            out.append(_utc(dt.isoformat()))
        return out
    out = []
    for value in raw:
        if not np.isfinite(value):
            out.append(None)
        else:
            out.append(pd.to_datetime(float(value), unit="s", utc=True))
    return out


def read_isr_file(path, default_site=None):
    """Read one ISR NetCDF/HDF5 file into profile dictionaries.

    Expected science fields are the same ones used by the previous validation
    reader: time/timestamps, gdalt/altitude/alt and ne/Ne/electron_density.
    ``dne`` is read when present but is not subtracted.
    """
    path = Path(path)
    records = []
    with netCDF4.Dataset(path) as ds:
        raw_time = _variable(ds, ("timestamps", "timestamp", "time"))
        times = _timestamps_to_utc(ds, raw_time)
        n_time = len(times)
        altitude = _as_time_profiles(_variable(ds, ("gdalt", "altitude", "alt")), n_time)
        ne = _as_time_profiles(_variable(ds, ("ne", "Ne", "electron_density")), n_time)
        lat = _scalar_from_dataset(ds, ("gdlat", "latitude", "lat", "site_lat", "station_lat"))
        lon = _scalar_from_dataset(ds, ("glon", "longitude", "lon", "site_lon", "station_lon"))
        kindat = _kindat_from_dataset(ds, path)

    if default_site is not None:
        if not np.isfinite(lat):
            lat = float(default_site["lat"])
        if not np.isfinite(lon):
            lon = float(default_site["lon"])

    for i, timestamp in enumerate(times):
        if timestamp is None:
            continue
        alt_i, ne_i = _profile(altitude[i], ne[i])
        valid = np.isfinite(alt_i) & np.isfinite(ne_i) & (ne_i > 0)
        if valid.sum() < 2:
            continue
        records.append({
            "time": _utc(timestamp),
            "lat": float(lat) if np.isfinite(lat) else np.nan,
            "lon": float(lon) if np.isfinite(lon) else np.nan,
            "alt_km": alt_i[valid],
            "ne_m3": ne_i[valid],
            "kindat": kindat,
            "source_file": str(path),
        })
    return records


def load_isr_observations(root, site=None):
    """Load every readable ISR profile from raw observation files."""
    default_site = INSTRUMENTS.get(site) if isinstance(site, str) else site
    files = observation_files(root)
    if not files:
        raise FileNotFoundError(f"No ISR NetCDF/HDF5 files found below {root}.")
    records, failures = [], []
    for path in files:
        try:
            records.extend(read_isr_file(path, default_site=default_site))
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    records.sort(key=lambda r: r["time"])
    if not records:
        detail = "; ".join(failures[:5])
        raise LookupError(f"No readable ISR profiles below {root}. {detail}")
    if failures:
        print(f"[ISR] Skipped {len(failures)} unreadable file(s); first: {failures[0]}")
    return records


def _window_tag(lo, hi):
    lo, hi = _utc(lo), _utc(hi)
    return f"{lo:%Y%m%d}_{lo:%H%M}{hi:%H%M}"


def _export_isr_raw_csv(matched, site_name, site, lo, hi, output_dir):
    rows = []
    for profile_index, (timestamp, profile) in enumerate(matched):
        alt, ne = _profile(profile["alt_km"], profile["ne_m3"])
        for gate_index, (alt_km, ne_m3) in enumerate(zip(alt, ne)):
            if not (np.isfinite(alt_km) and np.isfinite(ne_m3)):
                continue
            rows.append({
                "isr_id": site_name,
                "isr_lat": float(site["lat"]),
                "isr_lon": float(site["lon"]),
                "profile_index": profile_index,
                "datetime_utc": timestamp,
                "gate_index": gate_index,
                "alt_km": float(alt_km),
                "Ne_m3": float(ne_m3),
                "kindat": str(profile.get("kindat", "")),
                "source_file": profile.get("source_file", ""),
            })
    path = Path(output_dir) / f"isr_{site_name}_{_window_tag(lo, hi)}_raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _export_isr_rmse_csv(height_table, site_name, site, lo, hi, output_dir):
    frame = height_table.copy()
    frame.insert(0, "isr_lon", float(site["lon"]))
    frame.insert(0, "isr_lat", float(site["lat"]))
    frame.insert(0, "isr_id", site_name)
    if "n_scans" in frame.columns:
        frame = frame.rename(columns={"n_scans": "n_observations"})
    path = Path(output_dir) / f"isr_{site_name}_{_window_tag(lo, hi)}_rmse.csv"
    frame.to_csv(path, index=False)
    return path


def validate_isr_profile(
    input_ne_m3,
    input_alt_km,
    lat,
    lon,
    time=None,
    doy=None,
    year=None,
    *,
    time_window=None,
    time_window_minutes=15.0,
    height_bin_km=10.0,
    output_path="isr_validation.png",
    isr_data_dir=None,
    edps=None,
    sites=None,
    min_ne_m3=1e7,
    min_valid_gates=ISR_MIN_VALID_GATES,
    max_site_distance_km=None,
):
    """Compare an input Ne column with all real ISR scans in the time window."""
    if not np.isfinite(height_bin_km) or height_bin_km <= 0:
        raise ValueError("height_bin_km must be positive.")
    if min_valid_gates < 1:
        raise ValueError("min_valid_gates must be >= 1.")

    sites = {k: INSTRUMENTS[k] for k in ("ESR", "TRO")} if sites is None else sites
    lo, hi, centre = _resolve_window(time_window, time, doy, year, time_window_minutes)
    name, site, distance = _nearest_site(lat, lon, sites, max_site_distance_km)

    if edps is None:
        root = ISR_DATA_DIR if isr_data_dir is None else Path(isr_data_dir).expanduser()
        print(f"[ISR] Reading raw observations from {root}")
        edps = load_isr_observations(root, site=name)

    records = []
    for profile in edps:
        kindat = str(profile.get("kindat", "")).strip()
        # Preserve the main product selection when kindat is available.  Some
        # raw files do not expose kindat as a variable/attribute; those are kept.
        if kindat and kindat != "6400":
            continue
        plat = float(profile.get("lat", np.nan))
        plon = float(profile.get("lon", np.nan))
        if np.isfinite(plat) and abs(plat - float(site["lat"])) > ISR_SITE_MATCH_DEG:
            continue
        if np.isfinite(plon) and abs(plon - float(site["lon"])) > ISR_SITE_MATCH_DEG:
            continue
        records.append((_utc(profile["time"]), profile))

    records.sort(key=lambda item: item[0])
    _availability(f"ISR {name}", [t for t, _ in records])
    matched = [(t, p) for t, p in records if lo <= t <= hi]
    if not matched:
        raise LookupError(f"No {name} ISR scans in the selected window.")

    model_alt, model_ne = _profile(input_alt_km, input_ne_m3)
    if np.isfinite(model_ne).sum() < 2:
        raise ValueError("Input requires at least two valid Ne samples.")

    per_time, squared_profiles, observed_on_grid, raw_profiles = [], [], [], []
    for timestamp, profile in matched:
        obs_alt, obs_ne = _profile(profile["alt_km"], profile["ne_m3"])
        if obs_alt.size < 2:
            per_time.append({"time_utc": timestamp, "rmse_ne_m3": np.nan,
                             "n_valid_gates": 0, "used": False})
            continue

        model_at_isr = np.interp(obs_alt, model_alt, model_ne, left=np.nan, right=np.nan)
        valid = np.isfinite(obs_ne) & (obs_ne > min_ne_m3) & np.isfinite(model_at_isr)
        used = int(valid.sum()) >= min_valid_gates
        rmse = float(np.sqrt(np.mean((model_at_isr[valid]-obs_ne[valid])**2))) if used else np.nan
        per_time.append({"time_utc": timestamp, "rmse_ne_m3": rmse,
                         "n_valid_gates": int(valid.sum()), "used": used})
        if not used:
            continue

        obs_at_model = np.interp(model_alt, obs_alt, obs_ne, left=np.nan, right=np.nan)
        good = np.isfinite(obs_at_model) & (obs_at_model > min_ne_m3) & np.isfinite(model_ne)
        squared = np.full(model_alt.shape, np.nan)
        squared[good] = (model_ne[good]-obs_at_model[good])**2
        squared_profiles.append(squared)
        observed_on_grid.append(obs_at_model)
        raw_profiles.append((timestamp, obs_alt, obs_ne))

    if not squared_profiles:
        raise LookupError("No ISR scans passed the overlap/valid-gate checks.")

    squared_profiles = np.asarray(squared_profiles)
    bin_ids = np.floor(model_alt / height_bin_km).astype(int)
    rows = []
    for bin_id in range(bin_ids.min(), bin_ids.max()+1):
        block = squared_profiles[:, bin_ids == bin_id]
        finite = np.isfinite(block)
        values = block[finite]
        rows.append({
            "alt_lower_km": bin_id*height_bin_km,
            "alt_upper_km": (bin_id+1)*height_bin_km,
            "alt_center_km": (bin_id+0.5)*height_bin_km,
            "rmse_ne_m3": float(np.sqrt(values.mean())) if values.size else np.nan,
            "n_pairs": int(values.size),
            "n_scans": int(np.any(finite, axis=1).sum()),
        })

    height_table = pd.DataFrame(rows)
    time_table = pd.DataFrame(per_time)
    if not np.isfinite(height_table["rmse_ne_m3"]).any():
        raise LookupError("No valid ISR/model pairs at input voxel heights.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 15), sharey=True)
    axes[0].plot(height_table["rmse_ne_m3"], height_table["alt_center_km"], "k.-")
    axes[0].set_xlabel(r"Ne RMSE (m$^{-3}$)")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title(f"Full-profile Ne RMSE\n{height_bin_km:g}-km bins; {len(raw_profiles)} ISR scans")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for i, (_, obs_alt, obs_ne) in enumerate(raw_profiles):
        axes[1].plot(obs_ne, obs_alt, color="tab:blue", alpha=0.35,
                     label="Real ISR scans" if i == 0 else None)
    axes[1].plot(model_ne, model_alt, color="red", linewidth=2.5, label="Input Ne")
    axes[1].set_xscale("log")
    axes[1].set_xlim(left=1e7,right=1e13)
    axes[1].set_xlabel(r"Ne (m$^{-3}$)")
    axes[1].set_title(f"{name}: input and measured ISR profiles")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{name} ISR validation\n{lo} — {hi}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    raw_csv_path = _export_isr_raw_csv(matched, name, site, lo, hi, output.parent)
    rmse_csv_path = _export_isr_rmse_csv(height_table, name, site, lo, hi, output.parent)
    print(f"[ISR] {len(raw_profiles)}/{len(matched)} scans used; saved {output}")
    print(f"[ISR] Raw CSV: {raw_csv_path}")
    print(f"[ISR] RMSE CSV: {rmse_csv_path}")

    return {
        "site": name,
        "site_distance_km": distance,
        "target_time": centre,
        "window_start": lo,
        "window_end": hi,
        "by_height": height_table,
        "by_time": time_table,
        "input_alt_km": model_alt,
        "isr_ne_on_input_altitude_m3": np.asarray(observed_on_grid),
        "isr_times": [t for t, _, _ in raw_profiles],
        "figure_path": str(output),
        "raw_csv_path": str(raw_csv_path),
        "rmse_csv_path": str(rmse_csv_path),
    }
