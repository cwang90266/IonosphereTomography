#!/usr/bin/env python3
"""Create daily Tromso UHF ISR raw, median, uncertainty, and profile plots.

The UTC day is detected automatically from the first timestamp unless --date
is supplied.  All day-level panels span exactly 24 hours.  Four-minute median
profiles use the nearest native altitude gate and one adjacent gate on either
side from every one-minute profile in the interval.  Individual raw-versus-
median profile figures are written for every interval that produces a valid
median profile.

exxammple command:

python batch_tromso_ne_median.py \                
  "/Users/User2/Desktop/PlanetiQ/ISR_PCA/data/TRO/MAD6400_2025-12-16_beata_60@uhfa.nc" --date 2025-12-16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import netCDF4
import numpy as np


TROMSO_TIMEZONE = ZoneInfo("Europe/Oslo")
DEFAULT_OUTPUT_DIR = Path("/home/pin/Desktop/IonosphereTomography-pin_dev/ISR_PCA/figure/")
DEFAULT_SPACING_TABLE = "80:120:3,120:200:10,200:300:15,300:450:25,450:700:30"
REQUIRED_VARIABLES = {"timestamps", "gdalt", "ne"}


@dataclass(frozen=True)
class AltitudeBand:
    lower_km: float
    upper_km: float
    spacing_km: float


def parse_spacing_table(value: str) -> list[AltitudeBand]:
    bands: list[AltitudeBand] = []
    try:
        for item in value.split(","):
            lower, upper, spacing = (float(part) for part in item.split(":"))
            if lower >= upper or spacing <= 0:
                raise ValueError
            bands.append(AltitudeBand(lower, upper, spacing))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use lower:upper:spacing entries separated by commas"
        ) from exc
    if not bands:
        raise argparse.ArgumentTypeError("The spacing table cannot be empty")
    bands.sort(key=lambda band: band.lower_km)
    for previous, current in zip(bands[:-1], bands[1:]):
        if current.lower_km < previous.upper_km:
            raise argparse.ArgumentTypeError("Altitude bands cannot overlap")
    return bands


def output_altitudes(bands: list[AltitudeBand]) -> np.ndarray:
    pieces = [
        np.arange(band.lower_km, band.upper_km, band.spacing_km, dtype=float)
        for band in bands
    ]
    values = np.unique(np.concatenate(pieces))
    if values.size < 2:
        raise ValueError("The spacing table produced fewer than two altitudes")
    return values


def as_float_array(variable: netCDF4.Variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=float).squeeze()


def orient_range_time(
    variable: netCDF4.Variable, n_times: int, variable_name: str
) -> np.ndarray:
    values = as_float_array(variable)
    if values.ndim != 2:
        raise ValueError(f"{variable_name} must be 2-D; got {values.shape}")
    dimensions = tuple(variable.dimensions)
    if dimensions == ("range", "timestamps"):
        return values
    if dimensions == ("timestamps", "range"):
        return values.T
    if values.shape[1] == n_times and values.shape[0] != n_times:
        return values
    if values.shape[0] == n_times and values.shape[1] != n_times:
        return values.T
    raise ValueError(
        f"Cannot identify time axis for {variable_name}: "
        f"dimensions={dimensions}, shape={values.shape}, timestamps={n_times}"
    )


def detect_day_start(timestamps: np.ndarray, requested_date: str | None) -> datetime:
    if requested_date:
        selected_date = date.fromisoformat(requested_date)
        return datetime.combine(selected_date, datetime.min.time(), timezone.utc)
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size == 0:
        raise ValueError("The file contains no finite timestamps")
    first = datetime.fromtimestamp(float(np.min(finite)), timezone.utc)
    return first.replace(hour=0, minute=0, second=0, microsecond=0)


def time_edges(start: datetime, end: datetime, interval_minutes: float) -> list[datetime]:
    step = timedelta(minutes=interval_minutes)
    edges = [start]
    while edges[-1] < end:
        edges.append(min(edges[-1] + step, end))
    return edges


def altitude_edges(centers: np.ndarray) -> np.ndarray:
    middle = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], middle, [last]))


def adjacent_values(
    profile_altitude: np.ndarray,
    profile_ne: np.ndarray,
    target_altitude: float,
    adjacent_points: int,
    ne_floor: float,
    ne_ceiling: float,
) -> np.ndarray:
    valid_z = np.isfinite(profile_altitude)
    z = profile_altitude[valid_z]
    ne = profile_ne[valid_z]
    if z.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(z)
    z = z[order]
    ne = ne[order]
    if target_altitude < z[0] or target_altitude > z[-1]:
        return np.array([], dtype=float)
    center = int(np.argmin(np.abs(z - target_altitude)))
    lower = max(0, center - adjacent_points)
    upper = min(z.size, center + adjacent_points + 1)
    selected = ne[lower:upper]
    good = np.isfinite(selected) & (selected > ne_floor) & (selected <= ne_ceiling)
    return selected[good]


def nearest_gate_profile(
    profile_altitude: np.ndarray,
    profile_ne: np.ndarray,
    targets: np.ndarray,
    ne_floor: float,
    ne_ceiling: float,
) -> np.ndarray:
    """Map one raw profile to target altitudes without smoothing."""
    result = np.full(targets.shape, np.nan)
    valid_z = np.isfinite(profile_altitude)
    z = profile_altitude[valid_z]
    ne = profile_ne[valid_z]
    if z.size == 0:
        return result
    order = np.argsort(z)
    z = z[order]
    ne = ne[order]
    for index, target in enumerate(targets):
        if target < z[0] or target > z[-1]:
            continue
        nearest = int(np.argmin(np.abs(z - target)))
        value = ne[nearest]
        if np.isfinite(value) and ne_floor < value <= ne_ceiling:
            result[index] = value
    return result


def process_median_bins(
    timestamps: np.ndarray,
    altitude: np.ndarray,
    electron_density: np.ndarray,
    edges: list[datetime],
    targets: np.ndarray,
    adjacent_points: int,
    min_profiles: int,
    max_profiles: int,
    min_samples: int,
    ne_floor: float,
    ne_ceiling: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    n_bins = len(edges) - 1
    median_ne = np.full((targets.size, n_bins), np.nan)
    sample_count = np.zeros((targets.size, n_bins), dtype=int)
    indices_by_bin: list[np.ndarray] = []

    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        indices = np.flatnonzero(
            np.isfinite(timestamps)
            & (timestamps >= left.timestamp())
            & (timestamps < right.timestamp())
        )
        if indices.size > max_profiles:
            center_time = 0.5 * (left.timestamp() + right.timestamp())
            closest = np.argsort(np.abs(timestamps[indices] - center_time))[
                :max_profiles
            ]
            indices = np.sort(indices[closest])
        indices_by_bin.append(indices)
        if indices.size < min_profiles:
            continue
        for altitude_index, target in enumerate(targets):
            pieces = [
                adjacent_values(
                    altitude[:, record],
                    electron_density[:, record],
                    float(target),
                    adjacent_points,
                    ne_floor,
                    ne_ceiling,
                )
                for record in indices
            ]
            pieces = [piece for piece in pieces if piece.size]
            if not pieces:
                continue
            pooled = np.concatenate(pieces)
            sample_count[altitude_index, bin_index] = pooled.size
            if pooled.size >= min_samples:
                median_ne[altitude_index, bin_index] = np.median(pooled)
    return median_ne, sample_count, indices_by_bin


def build_one_minute_raw_grid(
    timestamps: np.ndarray,
    altitude: np.ndarray,
    electron_density: np.ndarray,
    day_start: datetime,
    day_end: datetime,
    targets: np.ndarray,
    ne_floor: float,
    ne_ceiling: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    n_minutes = int((day_end - day_start).total_seconds() // 60)
    records_by_minute: list[list[int]] = [[] for _ in range(n_minutes)]
    in_day = np.flatnonzero(
        np.isfinite(timestamps)
        & (timestamps >= day_start.timestamp())
        & (timestamps < day_end.timestamp())
    )
    for record in in_day:
        minute = int((timestamps[record] - day_start.timestamp()) // 60)
        if 0 <= minute < n_minutes:
            records_by_minute[minute].append(int(record))

    raw_grid = np.full((targets.size, n_minutes), np.nan)
    stored_indices: list[np.ndarray] = []
    for minute, records in enumerate(records_by_minute):
        indices = np.asarray(records, dtype=int)
        stored_indices.append(indices)
        if not records:
            continue
        mapped = np.column_stack(
            [
                nearest_gate_profile(
                    altitude[:, record],
                    electron_density[:, record],
                    targets,
                    ne_floor,
                    ne_ceiling,
                )
                for record in records
            ]
        )
        if mapped.shape[1] == 1:
            raw_grid[:, minute] = mapped[:, 0]
        else:
            for altitude_index in range(mapped.shape[0]):
                values = mapped[altitude_index]
                values = values[np.isfinite(values)]
                if values.size:
                    raw_grid[altitude_index, minute] = np.median(values)
    return raw_grid, stored_indices


def calculate_uncertainty(
    raw_grid: np.ndarray,
    median_ne: np.ndarray,
    interval_minutes: int,
) -> np.ndarray:
    uncertainty = np.full(raw_grid.shape, np.nan)
    for minute in range(raw_grid.shape[1]):
        bin_index = minute // interval_minutes
        if bin_index >= median_ne.shape[1]:
            continue
        raw = raw_grid[:, minute]
        background = median_ne[:, bin_index]
        valid = np.isfinite(raw) & np.isfinite(background) & (background > 0)
        uncertainty[valid, minute] = (
            100.0 * np.abs(raw[valid] - background[valid]) / background[valid]
        )
    return uncertainty


def format_time_axis(axis: plt.Axes, maximum_ticks: int = 10) -> None:
    locator = mdates.AutoDateLocator(
        minticks=5, maxticks=maximum_ticks, tz=TROMSO_TIMEZONE
    )
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator, tz=TROMSO_TIMEZONE)
    )


def plot_density_panel(
    values: np.ndarray,
    targets: np.ndarray,
    edges: list[datetime],
    title: str,
    colorbar_label: str,
    output: Path,
    vmin: float,
    vmax: float,
) -> None:
    local_edges = [edge.astimezone(TROMSO_TIMEZONE) for edge in edges]
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    colormap = plt.colormaps["jet"].copy()
    colormap.set_bad("white")
    mesh = axis.pcolormesh(
        mdates.date2num(local_edges),
        altitude_edges(targets),
        np.ma.masked_invalid(values),
        cmap=colormap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="flat",
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02, extend="both")
    colorbar.set_label(colorbar_label)
    format_time_axis(axis)
    axis.set_xlim(local_edges[0], local_edges[-1])
    axis.set_ylim(targets[0], targets[-1])
    axis.set_xlabel("Local time (LT)")
    axis.set_ylabel("Geodetic altitude (km)")
    axis.set_title(
        title
        + "\n"
        + f"LT: {local_edges[0]:%Y-%m-%d %H:%M:%S} to "
        + f"{local_edges[-1]:%Y-%m-%d %H:%M:%S}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_uncertainty_panel(
    uncertainty: np.ndarray,
    targets: np.ndarray,
    edges: list[datetime],
    uncertainty_max: float,
    output: Path,
) -> None:
    local_edges = [edge.astimezone(TROMSO_TIMEZONE) for edge in edges]
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    colormap = plt.colormaps["jet"].copy()
    colormap.set_bad("white")
    mesh = axis.pcolormesh(
        mdates.date2num(local_edges),
        altitude_edges(targets),
        np.ma.masked_invalid(uncertainty),
        cmap=colormap,
        norm=Normalize(vmin=0.0, vmax=uncertainty_max),
        shading="flat",
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02, extend="max")
    colorbar.set_label(
        r"Relative deviation $100|N_{e,raw}-N_{e,median}|/N_{e,median}$ (%)"
    )
    format_time_axis(axis)
    axis.set_xlim(local_edges[0], local_edges[-1])
    axis.set_ylim(targets[0], targets[-1])
    axis.set_xlabel("Local time (LT)")
    axis.set_ylabel("Geodetic altitude (km)")
    axis.set_title(
        "Tromsø UHF ISR one-minute relative deviation from 4-minute median"
        + "\n"
        + f"LT: {local_edges[0]:%Y-%m-%d %H:%M:%S} to "
        + f"{local_edges[-1]:%Y-%m-%d %H:%M:%S}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_interval_profile(
    altitude: np.ndarray,
    electron_density: np.ndarray,
    indices: np.ndarray,
    median_profile: np.ndarray,
    targets: np.ndarray,
    bin_start: datetime,
    bin_end: datetime,
    ne_floor: float,
    ne_ceiling: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 9), constrained_layout=True)
    raw_count = 0
    for record in indices:
        z = altitude[:, record]
        ne = electron_density[:, record]
        valid = (
            np.isfinite(z)
            & np.isfinite(ne)
            & (ne > ne_floor)
            & (ne <= ne_ceiling)
            & (z >= targets[0])
            & (z <= targets[-1])
        )
        if not np.any(valid):
            continue
        order = np.argsort(z[valid])
        axis.plot(
            ne[valid][order],
            z[valid][order],
            "--*",
            color="0.65",
            alpha=0.70,
            linewidth=2,
            markersize=6,
            label="Raw one-minute profiles" if raw_count == 0 else None,
            zorder=1,
        )
        raw_count += 1

    valid_median = (
        np.isfinite(median_profile)
        & (median_profile >= 1e9)
        & (median_profile <= 1e12)
    )
    axis.plot(
        median_profile[valid_median],
        targets[valid_median],
        "--*",
        color="blue",
        linewidth=2,
        markersize=7,
        label="Median profile",
        zorder=3,
    )
    local_start = bin_start.astimezone(TROMSO_TIMEZONE)
    local_end = bin_end.astimezone(TROMSO_TIMEZONE)
    axis.set_xscale("log")
    axis.set_xlim(1e9, 1e12)
    axis.set_ylim(targets[0], targets[-1])
    axis.set_xlabel(r"Electron density $N_e$ (m$^{-3}$)")
    axis.set_ylabel("Geodetic altitude (km)")
    axis.set_title(
        "Tromsø UHF ISR: raw and median $N_e$ profile\n"
        f"LT: {local_start:%Y-%m-%d %H:%M:%S} to {local_end:%H:%M:%S}"
    )
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("netcdf_file", type=Path)
    parser.add_argument(
        "--date",
        help="UTC day as YYYY-MM-DD; default is the date of the first timestamp",
    )
    parser.add_argument("--interval-minutes", type=int, default=4)
    parser.add_argument(
        "--spacing-table",
        type=parse_spacing_table,
        default=parse_spacing_table(DEFAULT_SPACING_TABLE),
        help="Altitude output spacing; default " + DEFAULT_SPACING_TABLE,
    )
    parser.add_argument("--adjacent-points", type=int, default=1)
    parser.add_argument("--min-profiles-per-bin", type=int, default=3)
    parser.add_argument(
        "--max-profiles-per-bin",
        type=int,
        default=4,
        help="Use at most this many profiles, chosen nearest the interval center",
    )
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--ne-floor", type=float, default=1e7)
    parser.add_argument("--ne-ceiling", type=float, default=1e13)
    parser.add_argument("--plot-ne-min", type=float, default=1e10)
    parser.add_argument("--plot-ne-max", type=float, default=1e12)
    parser.add_argument("--uncertainty-max", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-profile-plots",
        action="store_true",
        help="Create only the three day-level pcolor panels",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.netcdf_file.is_file():
        raise FileNotFoundError(args.netcdf_file)
    if args.interval_minutes <= 0 or 1440 % args.interval_minutes:
        raise ValueError("--interval-minutes must divide evenly into 1440")
    if args.adjacent_points < 0:
        raise ValueError("--adjacent-points cannot be negative")
    if args.min_profiles_per_bin < 1 or args.min_samples < 1:
        raise ValueError("Minimum profile/sample counts must be positive")
    if args.max_profiles_per_bin < args.min_profiles_per_bin:
        raise ValueError(
            "--max-profiles-per-bin cannot be smaller than --min-profiles-per-bin"
        )
    if not 0 < args.plot_ne_min < args.plot_ne_max:
        raise ValueError("Check --plot-ne-min and --plot-ne-max")
    if args.uncertainty_max <= 0:
        raise ValueError("--uncertainty-max must be positive")

    targets = output_altitudes(args.spacing_table)
    with netCDF4.Dataset(args.netcdf_file) as dataset:
        missing = REQUIRED_VARIABLES - set(dataset.variables)
        if missing:
            raise ValueError("Missing variables: " + ", ".join(sorted(missing)))
        timestamps = np.atleast_1d(as_float_array(dataset.variables["timestamps"]))
        altitude = orient_range_time(dataset.variables["gdalt"], timestamps.size, "gdalt")
        electron_density = orient_range_time(
            dataset.variables["ne"], timestamps.size, "ne"
        )
        day_start = detect_day_start(timestamps, args.date)
        day_end = day_start + timedelta(days=1)
        median_edges = time_edges(
            day_start, day_end, float(args.interval_minutes)
        )
        minute_edges = time_edges(day_start, day_end, 1.0)
        median_ne, sample_count, indices_by_bin = process_median_bins(
            timestamps,
            altitude,
            electron_density,
            median_edges,
            targets,
            args.adjacent_points,
            args.min_profiles_per_bin,
            args.max_profiles_per_bin,
            args.min_samples,
            args.ne_floor,
            args.ne_ceiling,
        )
        if not np.any(np.isfinite(median_ne)):
            raise ValueError("No valid 4-minute median profiles were produced")
        raw_ne, _ = build_one_minute_raw_grid(
            timestamps,
            altitude,
            electron_density,
            day_start,
            day_end,
            targets,
            args.ne_floor,
            args.ne_ceiling,
        )
        uncertainty = calculate_uncertainty(
            raw_ne, median_ne, args.interval_minutes
        )

        day_text = f"{day_start:%Y%m%d_000000}_1440min"
        median_output = args.output_dir / f"tromso_ne_median_pcolor_{day_text}.png"
        raw_output = args.output_dir / f"tromso_ne_raw_profile_{day_text}.png"
        uncertainty_output = (
            args.output_dir / f"tromso_ne_uncertainty_{day_text}.png"
        )
        plot_density_panel(
            median_ne,
            targets,
            median_edges,
            f"Tromsø UHF ISR median $N_e$ ({args.interval_minutes}-min intervals)",
            r"Median electron density $N_e$ (m$^{-3}$)",
            median_output,
            args.plot_ne_min,
            args.plot_ne_max,
        )
        plot_density_panel(
            raw_ne,
            targets,
            minute_edges,
            "Tromsø UHF ISR one-minute raw $N_e$ (nearest-gate mapping)",
            r"Raw electron density $N_e$ (m$^{-3}$)",
            raw_output,
            args.plot_ne_min,
            args.plot_ne_max,
        )
        plot_uncertainty_panel(
            uncertainty,
            targets,
            minute_edges,
            args.uncertainty_max,
            uncertainty_output,
        )

        profile_directory = (
            args.output_dir / f"TRO_medianprf_{day_start:%Y-%m-%d}"
        )
        profile_count = 0
        if not args.skip_profile_plots:
            for bin_index, indices in enumerate(indices_by_bin):
                if indices.size == 0 or not np.any(np.isfinite(median_ne[:, bin_index])):
                    continue
                interval_start = median_edges[bin_index]
                profile_output = profile_directory / (
                    f"tromso_ne_median_profile_{interval_start:%Y%m%d_%H%M%S}.png"
                )
                plot_interval_profile(
                    altitude,
                    electron_density,
                    indices,
                    median_ne[:, bin_index],
                    targets,
                    interval_start,
                    median_edges[bin_index + 1],
                    args.ne_floor,
                    args.ne_ceiling,
                    profile_output,
                )
                profile_count += 1

    finite_times = timestamps[
        np.isfinite(timestamps)
        & (timestamps >= day_start.timestamp())
        & (timestamps < day_end.timestamp())
    ]
    availability_start = datetime.fromtimestamp(float(np.min(finite_times)), timezone.utc)
    availability_end = datetime.fromtimestamp(float(np.max(finite_times)), timezone.utc)
    valid_bins = np.count_nonzero(np.any(np.isfinite(median_ne), axis=0))
    produced_counts = sample_count[np.isfinite(median_ne)]
    print(
        f"Detected data availability: {availability_start:%Y-%m-%d %H:%M:%S} "
        f"to {availability_end:%Y-%m-%d %H:%M:%S} UTC"
    )
    print(f"UTC day plotted: {day_start:%Y-%m-%d} (24 hours)")
    print(f"One-minute records in day: {finite_times.size}")
    print(f"Valid {args.interval_minutes}-minute median intervals: {valid_bins}")
    if produced_counts.size:
        print(
            "Contributing samples per produced median: "
            f"{produced_counts.min()} to {produced_counts.max()}"
        )
    print(f"Saved: {median_output}")
    print(f"Saved: {raw_output}")
    print(f"Saved: {uncertainty_output}")
    if args.skip_profile_plots:
        print("Individual interval profile plots skipped by request")
    else:
        print(f"Saved {profile_count} interval profiles in: {profile_directory}")


if __name__ == "__main__":
    main()
