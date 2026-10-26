#!/usr/bin/env python3
"""Plot a discrete histogram of RO occultation counts per UTC hour.

Each RO NetCDF file counts as one occultation when:
  1. its satellite is GN04, GN05, or YM08; and
  2. its TEC-maximum tangent point is inside the circular ROI.

Every UTC hour in the inclusive requested date range is retained. Therefore,
hours containing no qualifying RO files contribute to the x=0 histogram bar.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import netCDF4
import numpy as np


DEFAULT_TRO_LAT = 69.5864
DEFAULT_TRO_LON = 19.2272
SATELLITES = ("GN-4", "GN-5", "Y-8")
RO_DAY_DIRECTORY = re.compile(r"^(\d{4})\.(\d{3})$")


@dataclass(frozen=True)
class RoEvent:
    timestamp: datetime
    latitude: float
    longitude: float
    satellite: str


def satellite_from_file(path: Path, leo_id: str = "") -> str | None:
    name = path.name.upper().replace("-", "").replace("_", "")
    if "GN04" in name or "GN4" in name:
        return "GN-4"
    if "GN05" in name or "GN5" in name:
        return "GN-5"
    if "YM08" in name or "YM8" in name or "Y08" in name or "Y8" in name:
        return "Y-8"

    normalized = str(leo_id).strip().upper().replace("-", "")
    return {
        "GN04": "GN-4", "GN4": "GN-4", "4": "GN-4",
        "GN05": "GN-5", "GN5": "GN-5", "5": "GN-5",
        "YM08": "Y-8", "YM8": "Y-8", "Y08": "Y-8", "Y8": "Y-8", "8": "Y-8",
    }.get(normalized)


def haversine_km(lat0: float, lon0: float, lat: float, lon: float) -> float:
    radius_km = 6371.0
    phi0, phi1 = np.deg2rad([lat0, lat])
    dphi = phi1 - phi0
    dlon = np.deg2rad(lon - lon0)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi0) * np.cos(phi1) * np.sin(dlon / 2.0) ** 2
    )
    a = float(np.clip(a, 0.0, 1.0))
    return float(2.0 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def read_ro_event(path: Path) -> RoEvent | None:
    try:
        with netCDF4.Dataset(path, "r") as dataset:
            latitude = float(dataset.getncattr("lat_tecmax_tangent"))
            longitude = float(dataset.getncattr("lon_tecmax_tangent"))
            timestamp = datetime(
                int(dataset.getncattr("year")),
                int(dataset.getncattr("month")),
                int(dataset.getncattr("day")),
                int(dataset.getncattr("hour")),
                int(dataset.getncattr("minute")),
                int(float(dataset.getncattr("second"))),
                tzinfo=timezone.utc,
            )
            leo_id = (
                str(dataset.getncattr("leo_id"))
                if "leo_id" in dataset.ncattrs() else ""
            )
        satellite = satellite_from_file(path, leo_id)
        if satellite is None or not np.isfinite(latitude) or not np.isfinite(longitude):
            return None
        return RoEvent(timestamp, latitude, longitude, satellite)
    except Exception:
        return None


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def find_daily_folders(ro_dir: Path, requested_dates: list[date]) -> dict[date, list[Path]]:
    """Locate YYYY.DDD folders first, before opening any RO files."""
    wanted = {value.strftime("%Y.%j"): value for value in requested_dates}
    found = {value: [] for value in requested_dates}

    # Fast path for the normal layout: RO_ROOT/YYYY.DDD/files.
    for folder_name, value in wanted.items():
        direct = ro_dir / folder_name
        if direct.is_dir():
            found[value].append(direct)

    missing_names = {
        folder_name for folder_name, value in wanted.items() if not found[value]
    }
    if missing_names:
        # Compatibility with an additional directory layer under RO_ROOT.
        for folder in ro_dir.rglob("*"):
            if not folder.is_dir() or folder.name not in missing_names:
                continue
            match = RO_DAY_DIRECTORY.fullmatch(folder.name)
            if match is not None:
                found[wanted[folder.name]].append(folder)
    return found


def count_hourly_ro(args: argparse.Namespace):
    requested_dates = list(iter_dates(args.start_date, args.end_date))
    start = datetime.combine(args.start_date, datetime.min.time(), timezone.utc)
    end = datetime.combine(
        args.end_date + timedelta(days=1), datetime.min.time(), timezone.utc
    )
    n_hours = len(requested_dates) * 24
    total = np.zeros(n_hours, dtype=int)
    by_satellite = {
        satellite: np.zeros(n_hours, dtype=int) for satellite in SATELLITES
    }

    folders_by_date = find_daily_folders(args.ro_dir, requested_dates)
    examined = inside = outside = unreadable = 0
    seen_paths: set[Path] = set()

    for requested_date in requested_dates:
        folders = folders_by_date[requested_date]
        if not folders:
            print(f"WARNING: no RO folder for {requested_date:%Y-%m-%d} "
                  f"({requested_date:%Y.%j}); its 24 hours remain zero")
            continue
        for folder in folders:
            for path in sorted(folder.glob(args.ro_pattern)):
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                examined += 1
                event = read_ro_event(path)
                if event is None:
                    unreadable += 1
                    continue
                if not (start <= event.timestamp < end):
                    continue
                distance = haversine_km(
                    args.roi_center_lat, args.roi_center_lon,
                    event.latitude, event.longitude,
                )
                if distance > args.roi_radius_km:
                    outside += 1
                    continue
                hour_index = int((event.timestamp - start).total_seconds() // 3600)
                total[hour_index] += 1
                by_satellite[event.satellite][hour_index] += 1
                inside += 1

    hour_starts = [start + timedelta(hours=index) for index in range(n_hours)]
    return hour_starts, total, by_satellite, folders_by_date, {
        "examined": examined,
        "inside": inside,
        "outside": outside,
        "unreadable": unreadable,
    }


def save_hourly_csv(
    path: Path,
    hour_starts: list[datetime],
    total: np.ndarray,
    by_satellite: dict[str, np.ndarray],
    local_timezone: ZoneInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "utc_start", "utc_end", "local_start", "local_end",
            "local_hour_start", "total_ro", "GN-4", "GN-5", "Y-8",
        ])
        for index, hour_start in enumerate(hour_starts):
            local_start = hour_start.astimezone(local_timezone)
            local_end = (hour_start + timedelta(hours=1)).astimezone(local_timezone)
            writer.writerow([
                hour_start.strftime("%Y-%m-%dT%H:%M:%S"),
                (hour_start + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                local_start.strftime("%Y-%m-%dT%H:%M:%S"),
                local_end.strftime("%Y-%m-%dT%H:%M:%S"),
                local_start.strftime("%H:%M"),
                int(total[index]),
                int(by_satellite["GN-4"][index]),
                int(by_satellite["GN-5"][index]),
                int(by_satellite["Y-8"][index]),
            ])


def plot_histogram(
    path: Path,
    args: argparse.Namespace,
    hour_starts: list[datetime],
    hourly_total: np.ndarray,
) -> None:
    maximum = int(hourly_total.max()) if hourly_total.size else 0
    x_values = np.arange(maximum + 1, dtype=int)
    frequencies = np.bincount(hourly_total, minlength=maximum + 1)

    figure, (axis, time_axis) = plt.subplots(
        2, 1, figsize=(13, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )
    bars = axis.bar(
        x_values, frequencies, width=0.82,
        color="steelblue", edgecolor="black", linewidth=0.8,
    )
    axis.set_xticks(x_values)
    axis.set_xlim(-0.6, maximum + 0.6)
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("Number of RO occultations in one UTC hour", fontsize=args.font_size)
    axis.set_ylabel("Number of one-hour intervals", fontsize=args.font_size)
    axis.set_title(
        f"Hourly RO availability inside {args.roi_radius_km:g}-km ROI\n"
        f"{args.start_date.isoformat()} through {args.end_date.isoformat()} UTC",
        fontsize=args.title_font_size,
    )
    axis.tick_params(axis="both", labelsize=args.tick_font_size)
    axis.grid(True, axis="y", alpha=0.25)

    for bar, frequency in zip(bars, frequencies):
        if frequency:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(), str(int(frequency)),
                ha="center", va="bottom", fontsize=args.annotation_font_size,
            )

    axis.text(
        0.99, 0.97,
        f"ROI center: {args.roi_center_lat:.4f}°, {args.roi_center_lon:.4f}° | "
        f"Hours: {hourly_total.size} | RO files in ROI: {int(hourly_total.sum())}",
        transform=axis.transAxes, ha="right", va="top",
        fontsize=args.annotation_font_size,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    local_timezone = ZoneInfo(args.local_timezone)
    local_datetimes = [value.astimezone(local_timezone) for value in hour_starts]
    local_hours = np.asarray([
        value.hour + value.minute / 60.0 for value in local_datetimes
    ])
    time_axis.scatter(
        hourly_total, local_hours, color="black",
        s=42, alpha=0.75, edgecolors="none",
    )
    time_axis.set_xticks(x_values)
    time_axis.set_xlim(-0.6, maximum + 0.6)
    time_axis.set_ylim(-0.5, 24.5)
    time_axis.set_yticks(np.arange(0, 25, 2))
    time_axis.set_xlabel("Number of RO occultations in one UTC hour", fontsize=args.font_size)
    time_axis.set_ylabel(f"Local hour ({args.local_timezone})", fontsize=args.font_size)
    time_axis.set_title(
        "Local-time occurrence of each hourly-count bin",
        fontsize=args.title_font_size,
    )
    time_axis.tick_params(axis="both", labelsize=args.tick_font_size)
    time_axis.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def parser_builder() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ro-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--ro-pattern", default="podTc2*_*_nc")
    parser.add_argument("--roi-center-lat", type=float, default=DEFAULT_TRO_LAT)
    parser.add_argument("--roi-center-lon", type=float, default=DEFAULT_TRO_LON)
    parser.add_argument("--roi-radius-km", type=float, default=2000.0)
    parser.add_argument("--local-timezone", default="Europe/Oslo")
    parser.add_argument("--font-size", type=float, default=16.0)
    parser.add_argument("--title-font-size", type=float, default=18.0)
    parser.add_argument("--tick-font-size", type=float, default=14.0)
    parser.add_argument("--annotation-font-size", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.ro_dir.is_dir():
        raise FileNotFoundError(args.ro_dir)
    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")
    if args.roi_radius_km <= 0:
        raise ValueError("--roi-radius-km must be positive")
    if min(
        args.font_size, args.title_font_size,
        args.tick_font_size, args.annotation_font_size,
    ) <= 0:
        raise ValueError("Font sizes must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    ZoneInfo(args.local_timezone)


def main() -> None:
    args = parser_builder().parse_args()
    validate_args(args)
    hour_starts, total, by_satellite, folders, stats = count_hourly_ro(args)
    plot_histogram(args.output_figure, args, hour_starts, total)
    if args.output_csv is not None:
        save_hourly_csv(
            args.output_csv, hour_starts, total, by_satellite,
            ZoneInfo(args.local_timezone),
        )

    found_days = sum(bool(day_folders) for day_folders in folders.values())
    print(f"Requested UTC days: {len(folders)}; daily folders found: {found_days}")
    print(f"Hourly intervals represented: {len(total)}")
    print(f"RO files examined: {stats['examined']}")
    print(f"RO files inside ROI: {stats['inside']}")
    print(f"RO files outside ROI: {stats['outside']}")
    print(f"Unreadable/unsupported files: {stats['unreadable']}")
    print(f"Hours with zero RO: {int(np.count_nonzero(total == 0))}")
    print("Inside ROI by satellite: " + ", ".join(
        f"{satellite}={int(values.sum())}"
        for satellite, values in by_satellite.items()
    ))
    print(f"Saved figure: {args.output_figure}")
    if args.output_csv is not None:
        print(f"Saved hourly counts: {args.output_csv}")


if __name__ == "__main__":
    main()
