#!/usr/bin/env python3
"""Create daily TRO/SVL adaptive-Ne panels with stacked RO counts.

The first two panels reuse the processing functions and defaults in the supplied
Tromso and Svalbard adaptive ISR scripts.  The third panel counts podTc2 RO
occultation files whose TEC-maximum tangent point is inside a circular ROI.
Only days with TRO data, SVL data, and at least one in-ROI occultation from each
requested satellite are plotted by default.

Time convention
---------------
Processing and availability matching use UTC days [00:00, 24:00). Plot labels
are converted to Europe/Oslo local civil time, matching the ISR scripts.

Example
-------
python plot_tro_svl_ro_daily.py \
  --tro-dir /path/to/TRO \
  --svl-dir /path/to/SVL \
  --ro-dir /path/to/podTc2 \
  --year 2025 \
  --output-dir /path/to/daily_TRO_SVL_RO \
  --tro-processor adaptive_tromso_isr_native_alt_interplowhighpoint.py \
  --svl-processor adaptive_svalbard_isr_native_alt_interplowhighpoint.py

  python plot_tro_svl_ro_daily.py   --tro-dir "/home/pin/Desktop/tomography_project/Data/ISR_Data/TRO/"   --svl-dir "/home/pin/Desktop/tomography_project/Data/ISR_Data/SVL/"   --ro-dir "/home/pin/Desktop/tomography_project/piq_data/podTc2/"   --output-dir "/home/pin/Desktop/IonosphereTomography-pin_dev/ISR_PCA/figure/ISRNe_ROcount/"   --tro-processor "/home/pin/Desktop/IonosphereTomography-pin_dev/ISR_PCA/adaptive_tromso_isr_native_alt_interplowhighpoint.py"   --svl-processor "/home/pin/Desktop/IonosphereTomography-pin_dev/ISR_PCA/adaptive_svalbard_isr_native_alt_interplowhighpoint.py"   --year 2025   --recursive-isr
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import netCDF4
import numpy as np


DEFAULT_TRO_LAT = 69.5864
DEFAULT_TRO_LON = 19.2272
SATELLITES = ("GN-4", "GN-5", "Y-8")
SATELLITE_COLORS = {"GN-4": "blue", "GN-5": "green", "Y-8": "red"}
ISR_DATE_RE = re.compile(r"MAD6400_(\d{4}-\d{2}-\d{2})_", re.IGNORECASE)
RO_DAY_DIR_RE = re.compile(r"^(\d{4})\.(\d{3})$")


@dataclass(frozen=True)
class RoEvent:
    path: Path
    timestamp: float
    latitude: float
    longitude: float
    satellite: str


@dataclass
class StationData:
    label: str
    module: ModuleType
    args: argparse.Namespace
    by_day: dict[str, list]
    common_grid: np.ndarray
    adaptive_is_moving: bool


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import processor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def locate_processor(explicit: Path | None, station: str) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = Path(__file__).resolve().parent
    patterns = [
        f"adaptive_{station}_isr_native_alt_interplowhighpoint.py",
        f"adaptive_{station}_isr_native_alt_interplowhighpoint*.py",
        f"adaptive_{station}_isr*.py",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find the {station} processor beside this script; "
        f"provide --{station[:3]}-processor explicitly"
    )


def processor_namespace(module: ModuleType, input_path: Path, year: int) -> argparse.Namespace:
    parser = module.parser_builder()
    args = parser.parse_args([str(input_path), "--year", str(year)])
    args.output_interval_minutes = 4
    return args


def list_netcdf_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.rglob("*.nc") if recursive else path.glob("*.nc"))


def discover_isr_files(path: Path, year: int, recursive: bool) -> dict[str, list[Path]]:
    """Index ISR files by UTC date using MAD6400_YYYY-MM-DD filenames only."""
    by_day: dict[str, list[Path]] = {}
    for file_path in list_netcdf_files(path, recursive):
        match = ISR_DATE_RE.search(file_path.name)
        if match is None:
            continue
        file_date = date.fromisoformat(match.group(1))
        if file_date.year == year:
            by_day.setdefault(file_date.strftime("%Y%m%d"), []).append(file_path)
    return by_day


def discover_ro_files(ro_dir: Path, pattern: str, year: int) -> dict[str, list[Path]]:
    """Index RO files from YYYY.DDD directory names without opening NetCDF files."""
    if not ro_dir.is_dir():
        raise FileNotFoundError(ro_dir)
    by_day: dict[str, list[Path]] = {}
    for folder in ro_dir.rglob(f"{year}.*"):
        if not folder.is_dir():
            continue
        match = RO_DAY_DIR_RE.fullmatch(folder.name)
        if match is None or int(match.group(1)) != year:
            continue
        try:
            folder_date = datetime.strptime(folder.name, "%Y.%j").date()
        except ValueError:
            continue
        files = sorted(folder.glob(pattern))
        if files:
            by_day.setdefault(folder_date.strftime("%Y%m%d"), []).extend(files)
    return by_day


def days_with_neighbors(day_keys: set[str]) -> set[str]:
    """Include adjacent UTC dates needed by centered moving-median windows."""
    expanded: set[str] = set()
    for day_key in day_keys:
        value = datetime.strptime(day_key, "%Y%m%d").date()
        for offset in (-1, 0, 1):
            expanded.add((value + timedelta(days=offset)).strftime("%Y%m%d"))
    return expanded


def read_station_profiles(
    label: str,
    module: ModuleType,
    args: argparse.Namespace,
    recursive: bool,
    files: list[Path] | None = None,
) -> dict[str, list]:
    files = list_netcdf_files(args.input_path, recursive) if files is None else sorted(files)
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {label}: {args.input_path}")
    by_day: dict[str, list] = {}
    failures = 0
    read_parameters = inspect.signature(module.read_profiles).parameters
    for index, path in enumerate(files, 1):
        try:
            positional = [
                path, args.year, args.alt_min, args.alt_max,
                args.ne_floor, args.ne_ceiling,
                args.chunk_profiles, args.min_profile_top_km,
            ]
            if "drop_highest_gate" in read_parameters:
                positional.append(getattr(args, "drop_highest_gate", False))
            result = module.read_profiles(*positional)
            for day_key, profiles in result.items():
                by_day.setdefault(day_key, []).extend(profiles)
        except Exception as exc:
            failures += 1
            print(f"{label} SKIP {path.name}: {type(exc).__name__}: {exc}")
        if index % 25 == 0 or index == len(files):
            print(f"{label}: scanned {index}/{len(files)} file(s)", flush=True)
    if not by_day:
        raise RuntimeError(f"No usable {label} profiles for {args.year}")
    print(f"{label}: {sum(map(len, by_day.values()))} profiles on {len(by_day)} UTC day(s); "
          f"{failures} failed file(s)")
    return by_day


def apply_svalbard_geometry_filter(
    module: ModuleType,
    args: argparse.Namespace,
    by_day: dict[str, list],
) -> dict[str, list]:
    if not hasattr(module, "retain_dominant_geometry"):
        return by_day
    profiles = [profile for group in by_day.values() for profile in group]
    retained, geometry = module.retain_dominant_geometry(
        profiles,
        args.gate_count_tolerance,
        args.profile_bottom_tolerance_km,
        args.profile_top_tolerance_km,
    )
    allowed = {id(profile) for profile in retained}
    filtered = {
        day_key: [profile for profile in group if id(profile) in allowed]
        for day_key, group in by_day.items()
    }
    filtered = {day_key: group for day_key, group in filtered.items() if group}
    if not filtered:
        raise RuntimeError("No Svalbard profiles matched the dominant geometry")
    if geometry is not None:
        print(
            f"SVL dominant geometry: {geometry[0]} gates, "
            f"{geometry[1]:.1f}-{geometry[2]:.1f} km; "
            f"retained {len(retained)}/{len(profiles)} profiles"
        )
    return filtered


def processor_uses_moving_adaptive(module: ModuleType) -> bool:
    source = inspect.getsource(module.process_day).replace(" ", "").replace("\n", "")
    return "medium,_=moving_median_curves(" in source


def build_station_data(
    label: str,
    processor_path: Path,
    input_path: Path,
    year: int,
    recursive: bool,
    files: list[Path] | None = None,
) -> StationData:
    module = load_module(processor_path, f"_{label.lower()}_adaptive_processor")
    args = processor_namespace(module, input_path, year)
    by_day = read_station_profiles(label, module, args, recursive, files)
    if label == "SVL":
        by_day = apply_svalbard_geometry_filter(module, args, by_day)
    profiles = [profile for group in by_day.values() for profile in group]
    grid = module.common_daily_altitude(
        profiles,
        args.altitude_rounding_km,
        args.max_daily_gates,
        args.min_gate_fraction,
    )
    if grid.size < 2:
        raise RuntimeError(f"Could not construct the {label} altitude grid")
    print(f"{label}: shared grid has {grid.size} gates, {grid.min():.1f}-{grid.max():.1f} km")
    return StationData(
        label=label,
        module=module,
        args=args,
        by_day=by_day,
        common_grid=grid,
        adaptive_is_moving=processor_uses_moving_adaptive(module),
    )


def call_moving_median(
    station: StationData,
    profiles: list,
    day_start: datetime,
    window_minutes: int,
    max_profiles: int,
):
    function = station.module.moving_median_curves
    parameters = inspect.signature(function).parameters
    values = [profiles, day_start, station.args, station.common_grid, window_minutes]
    if "max_profiles" in parameters:
        values.append(max_profiles)
    return function(*values)


def neighboring_profiles(station: StationData, day_key: str) -> list:
    day_start = datetime.strptime(day_key, "%Y%m%d").replace(tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    half_seconds = max(station.args.topside_minutes, station.args.adaptive_minutes) * 30
    previous_key = (day_start - timedelta(days=1)).strftime("%Y%m%d")
    next_key = day_end.strftime("%Y%m%d")
    previous = [
        profile for profile in station.by_day.get(previous_key, [])
        if profile.timestamp >= day_start.timestamp() - half_seconds
    ]
    current = station.by_day[day_key]
    following = [
        profile for profile in station.by_day.get(next_key, [])
        if profile.timestamp < day_end.timestamp() + half_seconds
    ]
    return previous + current + following


def compute_daily_adaptive_ne(station: StationData, day_key: str):
    module, args = station.module, station.args
    day_start = datetime.strptime(day_key, "%Y%m%d").replace(tzinfo=timezone.utc)
    profiles = station.by_day[day_key]
    moving_source = neighboring_profiles(station, day_key)
    n_output = 1440 // args.output_interval_minutes
    grids = module.choose_output_grids(
        profiles, day_start, args, n_output, station.common_grid
    )
    short, _ = module.median_curves(
        profiles, day_start, args.short_minutes, args,
        args.max_short_profiles, station.common_grid,
    )
    if station.adaptive_is_moving:
        medium, _ = call_moving_median(
            station, moving_source, day_start,
            args.adaptive_minutes, args.max_adaptive_profiles,
        )
    else:
        medium, _ = module.median_curves(
            profiles, day_start, args.adaptive_minutes, args,
            args.max_adaptive_profiles, station.common_grid,
        )
    topside, _ = call_moving_median(
        station, moving_source, day_start,
        args.topside_minutes, args.max_topside_profiles,
    )
    decision_groups = module.groups_by_interval(
        profiles, day_start, args.decision_time_minutes
    )
    decision_grids = [station.common_grid.copy() for _ in decision_groups]
    decision_mean, _ = module.deviation_decisions(
        profiles, short, day_start, decision_grids, args
    )
    altitude, ne, _, _, _ = module.construct_output(
        profiles, grids, short, medium, topside,
        decision_mean, decision_grids, day_start, args,
    )
    times = (
        day_start.timestamp()
        + (np.arange(n_output) + 0.5) * args.output_interval_minutes * 60
    )
    return times, altitude, ne


def haversine_km(lat0: float, lon0: float, lat: float, lon: float) -> float:
    earth_radius_km = 6371.0
    phi0, phi1 = np.deg2rad([lat0, lat])
    dphi = phi1 - phi0
    dlambda = np.deg2rad(lon - lon0)
    value = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi0) * np.cos(phi1) * np.sin(dlambda / 2.0) ** 2
    )
    return float(2.0 * earth_radius_km * np.arctan2(np.sqrt(value), np.sqrt(1.0 - value)))


def satellite_from_file(path: Path, leo_id: str) -> str | None:
    name = path.name.upper().replace("-", "").replace("_", "")
    if "GN04" in name or "GN4" in name:
        return "GN-4"
    if "GN05" in name or "GN5" in name:
        return "GN-5"
    # The Y-8 receiver is identified as YM08 in the podTc2 filenames.
    # Retain Y08/Y8 as compatibility aliases for older or renamed files.
    if "YM08" in name or "Y08" in name or re.search(r"PODTC2Y8(?:\.|\d)", name):
        return "Y-8"
    normalized = str(leo_id).strip().upper().replace("-", "")
    fallback = {"GN04": "GN-4", "GN4": "GN-4", "4": "GN-4",
                "GN05": "GN-5", "GN5": "GN-5", "5": "GN-5",
                "YM08": "Y-8", "YM8": "Y-8",
                "Y08": "Y-8", "Y8": "Y-8", "8": "Y-8"}
    return fallback.get(normalized)


def read_ro_event(path: Path) -> RoEvent | None:
    try:
        with netCDF4.Dataset(path, "r") as dataset:
            latitude = float(dataset.getncattr("lat_tecmax_tangent"))
            longitude = float(dataset.getncattr("lon_tecmax_tangent"))
            event_time = datetime(
                int(dataset.getncattr("year")),
                int(dataset.getncattr("month")),
                int(dataset.getncattr("day")),
                int(dataset.getncattr("hour")),
                int(dataset.getncattr("minute")),
                int(float(dataset.getncattr("second"))),
                tzinfo=timezone.utc,
            )
            leo_id = str(dataset.getncattr("leo_id")) if "leo_id" in dataset.ncattrs() else ""
        satellite = satellite_from_file(path, leo_id)
        if satellite is None or not np.isfinite(latitude) or not np.isfinite(longitude):
            return None
        return RoEvent(path, event_time.timestamp(), latitude, longitude, satellite)
    except Exception:
        return None


def scan_ro_events(
    ro_dir: Path,
    pattern: str,
    year: int,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    files_by_day: dict[str, list[Path]] | None = None,
) -> tuple[dict[str, list[RoEvent]], int, int]:
    if not ro_dir.is_dir():
        raise FileNotFoundError(ro_dir)
    files = (sorted(ro_dir.rglob(pattern)) if files_by_day is None else
             sorted(path for group in files_by_day.values() for path in group))
    by_day: dict[str, list[RoEvent]] = {}
    unreadable = 0
    outside = 0
    for index, path in enumerate(files, 1):
        event = read_ro_event(path)
        if event is None:
            unreadable += 1
            continue
        event_time = datetime.fromtimestamp(event.timestamp, timezone.utc)
        if event_time.year != year:
            continue
        if haversine_km(center_lat, center_lon, event.latitude, event.longitude) > radius_km:
            outside += 1
            continue
        by_day.setdefault(event_time.strftime("%Y%m%d"), []).append(event)
        if index % 1000 == 0:
            print(f"RO: scanned {index}/{len(files)} file(s)", flush=True)
    print(
        f"RO: found {sum(map(len, by_day.values()))} in-ROI occultations on "
        f"{len(by_day)} UTC day(s); {outside} outside ROI; {unreadable} unreadable/unsupported"
    )
    return by_day, unreadable, outside


def bin_ro_counts(events: list[RoEvent], day_start: datetime, bin_minutes: int):
    n_bins = 1440 // bin_minutes
    counts = {satellite: np.zeros(n_bins, dtype=int) for satellite in SATELLITES}
    start_seconds = day_start.timestamp()
    for event in events:
        index = int((event.timestamp - start_seconds) // (bin_minutes * 60))
        if 0 <= index < n_bins and event.satellite in counts:
            counts[event.satellite][index] += 1
    return counts


def date_in_requested_range(day_key: str, start_date: date | None, end_date: date | None) -> bool:
    value = datetime.strptime(day_key, "%Y%m%d").date()
    if start_date is not None and value < start_date:
        return False
    if end_date is not None and value > end_date:
        return False
    return True


def time_edges_for_centers(times: np.ndarray, width_minutes: int) -> np.ndarray:
    centers = np.asarray(times, dtype=float)
    half = width_minutes * 30.0
    return np.r_[centers[0] - half, centers + half]


def epoch_to_matplotlib(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if hasattr(mdates, "epoch2num"):
        return mdates.epoch2num(values)
    return values / 86400.0 + mdates.date2num(datetime(1970, 1, 1))


def plot_ne_panel(
    axis: plt.Axes,
    station: StationData,
    times: np.ndarray,
    altitude: np.ndarray,
    ne: np.ndarray,
    day_start: datetime,
    day_end: datetime,
    local_timezone: ZoneInfo,
    altitude_min: float,
    altitude_max: float,
    ne_min: float,
    ne_max: float,
):
    zplot, shown = station.module.interpolate_altitude_for_plot(
        ne, altitude, log_scale=True
    )
    xedges = time_edges_for_centers(times, station.args.output_interval_minutes)
    if zplot.size:
        yedges = np.r_[zplot[0] - 0.5, zplot + 0.5]
        mesh = axis.pcolormesh(
            epoch_to_matplotlib(xedges), yedges, np.ma.masked_invalid(shown),
            shading="flat", cmap="jet", norm=LogNorm(vmin=ne_min, vmax=ne_max),
            rasterized=True,
        )
    else:
        mesh = axis.scatter([], [], c=[], cmap="jet", norm=LogNorm(vmin=ne_min, vmax=ne_max))
    axis.set_xlim(day_start.astimezone(local_timezone), day_end.astimezone(local_timezone))
    axis.set_ylim(altitude_min, altitude_max)
    axis.set_title(f"{station.label} adaptive median $N_e$")
    axis.set_xlabel("Local time")
    axis.set_ylabel("Altitude (km)")
    axis.grid(False)
    return mesh


def format_time_axis(axis: plt.Axes, display_timezone):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9, tz=display_timezone)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator, tz=display_timezone)
    )


def plot_daily_figure(
    output_path: Path,
    day_key: str,
    tro: StationData,
    svl: StationData,
    tro_values,
    svl_values,
    ro_counts: dict[str, np.ndarray],
    args: argparse.Namespace,
):
    local_timezone = ZoneInfo(args.local_timezone)
    day_start = datetime.strptime(day_key, "%Y%m%d").replace(tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.axes_title_size,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.tick_font_size,
        "ytick.labelsize": args.tick_font_size,
        "legend.fontsize": args.legend_font_size,
    })
    figure, axes = plt.subplots(3,1, figsize=(25, 15), constrained_layout=True)
    mesh = plot_ne_panel(
        axes[0], tro, *tro_values, day_start, day_end, local_timezone,
        args.plot_alt_min, args.plot_alt_max, args.plot_ne_min, args.plot_ne_max,
    )
    plot_ne_panel(
        axes[1], svl, *svl_values, day_start, day_end, local_timezone,
        args.plot_alt_min, args.plot_alt_max, args.plot_ne_min, args.plot_ne_max,
    )
    for axis in axes[:2]:
        format_time_axis(axis, local_timezone)
    colorbar = figure.colorbar(mesh, ax=axes[:2].tolist(), pad=0.015, extend="both")
    colorbar.set_label(r"Electron density $N_e$ (m$^{-3}$)")

    centers = day_start.timestamp() + (
        np.arange(1440 // args.ro_bin_minutes) + 0.5
    ) * args.ro_bin_minutes * 60
    x = epoch_to_matplotlib(centers)
    width = 0.92 * args.ro_bin_minutes / 1440.0
    bottom = np.zeros_like(centers, dtype=float)
    for satellite in SATELLITES:
        values = ro_counts[satellite]
        axes[2].bar(
            x, values, width=width, bottom=bottom,
            color=SATELLITE_COLORS[satellite], label=satellite,
            align="center", edgecolor="none",
        )
        bottom += values
    # RO bins are defined and displayed in UTC so they can be compared
    # directly with UTC RO-selection windows and exported observation tables.
    axes[2].set_xlim(day_start, day_end)
    axes[2].set_ylim(bottom=0)
    axes[2].set_title(f"RO occultations inside {args.roi_radius_km:g}-km ROI")
    axes[2].set_xlabel("UT")
    axes[2].set_ylabel(f"RO count per {args.ro_bin_minutes}-minute bin")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, axis="y", alpha=0.25)
    format_time_axis(axes[2], timezone.utc)

    local_start = day_start.astimezone(local_timezone)
    local_end = day_end.astimezone(local_timezone)
    figure.suptitle(
        f"UTC day {day_start:%Y-%m-%d} | LT {local_start:%Y-%m-%d %H:%M} to "
        f"{local_end:%Y-%m-%d %H:%M} | ROI center "
        f"({args.roi_center_lat:.4f}°, {args.roi_center_lon:.4f}°)",
        fontsize=args.suptitle_font_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def parser_builder() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tro-dir", type=Path, required=True)
    parser.add_argument("--svl-dir", type=Path, required=True)
    parser.add_argument("--ro-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tro-processor", type=Path)
    parser.add_argument("--svl-processor", type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--recursive-isr", action="store_true")
    parser.add_argument("--ro-pattern", default="podTc2*_*_nc")
    parser.add_argument("--roi-center-lat", type=float, default=DEFAULT_TRO_LAT)
    parser.add_argument("--roi-center-lon", type=float, default=DEFAULT_TRO_LON)
    parser.add_argument("--roi-radius-km", type=float, default=2500.0)
    parser.add_argument("--ro-bin-minutes", type=int, default=30)
    parser.add_argument("--local-timezone", default="Europe/Oslo")
    parser.add_argument("--plot-alt-min", type=float, default=80.0)
    parser.add_argument("--plot-alt-max", type=float, default=650.0)
    parser.add_argument("--plot-ne-min", type=float, default=1e10)
    parser.add_argument("--plot-ne-max", type=float, default=1e12)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--font-size", type=float, default=16.0)
    parser.add_argument("--tick-font-size", type=float, default=14.0)
    parser.add_argument("--axes-title-size", type=float, default=18.0)
    parser.add_argument("--legend-font-size", type=float, default=14.0)
    parser.add_argument("--suptitle-font-size", type=float, default=20.0)
    parser.add_argument(
        "--allow-missing-satellites", action="store_true",
        help="Plot days even when one or more of GN-4, GN-5, and Y-8 have zero in-ROI events",
    )
    return parser


def validate_args(args: argparse.Namespace):
    if args.roi_radius_km <= 0:
        raise ValueError("--roi-radius-km must be positive")
    if args.ro_bin_minutes <= 0 or 1440 % args.ro_bin_minutes:
        raise ValueError("--ro-bin-minutes must divide 1440")
    if args.plot_alt_min >= args.plot_alt_max:
        raise ValueError("Invalid altitude plot limits")
    if not 0 < args.plot_ne_min < args.plot_ne_max:
        raise ValueError("Invalid Ne color limits")
    if min(
        args.font_size, args.tick_font_size, args.axes_title_size,
        args.legend_font_size, args.suptitle_font_size,
    ) <= 0:
        raise ValueError("All font-size options must be positive")
    if args.start_date and args.start_date.year != args.year:
        raise ValueError("--start-date must be in --year")
    if args.end_date and args.end_date.year != args.year:
        raise ValueError("--end-date must be in --year")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date is after --end-date")
    ZoneInfo(args.local_timezone)


def main():
    args = parser_builder().parse_args()
    validate_args(args)
    tro_processor = locate_processor(args.tro_processor, "tromso")
    svl_processor = locate_processor(args.svl_processor, "svalbard")
    print(f"TRO processor: {tro_processor}")
    print(f"SVL processor: {svl_processor}")

    # Fast availability pass: infer ISR dates from filenames and RO dates from
    # YYYY.DDD folder names.  No NetCDF file is opened during this stage.
    tro_file_index = discover_isr_files(args.tro_dir, args.year, args.recursive_isr)
    svl_file_index = discover_isr_files(args.svl_dir, args.year, args.recursive_isr)
    ro_file_index = discover_ro_files(args.ro_dir, args.ro_pattern, args.year)
    candidate_days = set(tro_file_index) & set(svl_file_index) & set(ro_file_index)
    candidate_days = {
        key for key in candidate_days
        if date_in_requested_range(key, args.start_date, args.end_date)
    }
    print(
        f"Filename/folder availability: TRO {len(tro_file_index)} day(s), "
        f"SVL {len(svl_file_index)} day(s), RO {len(ro_file_index)} day(s); "
        f"{len(candidate_days)} common candidate day(s)"
    )
    if not candidate_days:
        raise RuntimeError("No dates have TRO, SVL, and RO files in common")

    # Adjacent ISR days are also loaded so the centered moving windows remain
    # correct at 00:00 and 24:00 on each candidate day.
    station_load_days = days_with_neighbors(candidate_days)
    tro_files = [
        path for key in station_load_days for path in tro_file_index.get(key, [])
    ]
    svl_files = [
        path for key in station_load_days for path in svl_file_index.get(key, [])
    ]
    candidate_ro_files = {key: ro_file_index[key] for key in candidate_days}

    tro = build_station_data(
        "TRO", tro_processor, args.tro_dir, args.year, args.recursive_isr, tro_files
    )
    svl = build_station_data(
        "SVL", svl_processor, args.svl_dir, args.year, args.recursive_isr, svl_files
    )
    ro_by_day, _, _ = scan_ro_events(
        args.ro_dir, args.ro_pattern, args.year,
        args.roi_center_lat, args.roi_center_lon, args.roi_radius_km,
        candidate_ro_files,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"TRO_SVL_RO_availability_{args.year}.csv"
    plotted = 0
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "utc_date", "tro_profiles", "svl_profiles",
            "GN-4_in_roi", "GN-5_in_roi", "Y-8_in_roi", "plotted", "reason",
        ])
        all_relevant_days = sorted(candidate_days)
        for day_key in all_relevant_days:
            if not date_in_requested_range(day_key, args.start_date, args.end_date):
                continue
            events = ro_by_day.get(day_key, [])
            totals = {
                satellite: sum(event.satellite == satellite for event in events)
                for satellite in SATELLITES
            }
            missing_sources = []
            if day_key not in tro.by_day:
                missing_sources.append("TRO")
            if day_key not in svl.by_day:
                missing_sources.append("SVL")
            if not events:
                missing_sources.append("RO")
            missing_satellites = [sat for sat in SATELLITES if totals[sat] == 0]
            should_plot = not missing_sources and (
                args.allow_missing_satellites or not missing_satellites
            )
            reason = ""
            if missing_sources:
                reason = "missing " + "/".join(missing_sources)
            elif missing_satellites and not args.allow_missing_satellites:
                reason = "missing " + "/".join(missing_satellites)
            if should_plot:
                print(f"{day_key}: processing daily comparison", flush=True)
                day_start = datetime.strptime(day_key, "%Y%m%d").replace(tzinfo=timezone.utc)
                ro_counts = bin_ro_counts(events, day_start, args.ro_bin_minutes)
                tro_values = compute_daily_adaptive_ne(tro, day_key)
                svl_values = compute_daily_adaptive_ne(svl, day_key)
                output = args.output_dir / f"TRO_SVL_RO_{day_key}.png"
                plot_daily_figure(
                    output, day_key, tro, svl, tro_values, svl_values,
                    ro_counts, args,
                )
                plotted += 1
                reason = "plotted"
                print(f"  saved {output}")
            writer.writerow([
                datetime.strptime(day_key, "%Y%m%d").date().isoformat(),
                len(tro.by_day.get(day_key, [])),
                len(svl.by_day.get(day_key, [])),
                totals["GN-4"], totals["GN-5"], totals["Y-8"],
                int(should_plot), reason,
            ])
    print(f"Saved {plotted} daily figure(s)")
    print(f"Availability summary: {summary_path}")


if __name__ == "__main__":
    main()
