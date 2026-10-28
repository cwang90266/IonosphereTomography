#!/usr/bin/env python3
"""Plot hourly accepted RO counts using the tomography RO precheck.

Acceptance is identical to the current tomography RO preparation:
  - radius_km = 4000
  - grid_radius_deg = 36
  - roi_mode = "full_los"
  - alt_limit_km = None (mean LEO altitude derived per occultation)
  - fraction_required = 0.65
  - min_valid_rays = 50
  - max_rays_per_occultation = 200
  - include_abel = True

Every UTC hour in the requested date range is retained, including zero-count hours.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

REPO = Path("/home/pin/Desktop/IonosphereTomography-pin_dev")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle import observation_stream


DEFAULT_TRO_LAT = 69.5864
DEFAULT_TRO_LON = 19.2272
SATELLITES = ("GN-4", "GN-5", "Y-8")

ROI_RADIUS_KM = 3000.0
GRID_RADIUS_DEG = 27.0
MAX_TEC_PER_BATCH = 100_000
OBS_SIGMA_TECU = 3.0

RO_KWARGS_TEMPLATE = {
    "roi_mode": "full_los",
    "alt_limit_km": None,
    "fraction_required": 0.65,
    "min_valid_rays": 50,
    "max_rays_per_occultation": 200,
    "include_abel": True,
}


def ro_day_dir(ro_root: Path, day: date) -> Path:
    return ro_root / f"{day.year}.{day.timetuple().tm_yday:03d}"


def unique_ro_entries(batches) -> list:
    seen = set()
    entries = []
    for batch in batches:
        for entry, _sl in batch.entry_ray_ranges:
            if entry.obs_type != "RO":
                continue
            key = id(entry)
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries


def satellite_from_entry(entry) -> str | None:
    text = " ".join(
        str(v) for v in (
            getattr(entry, "label", ""),
            getattr(entry, "rec_id", ""),
            getattr(entry, "obs_source", ""),
        )
        if v is not None
    ).upper().replace("-", "").replace("_", "")

    if "GN04" in text or "GN4" in text:
        return "GN-4"
    if "GN05" in text or "GN5" in text:
        return "GN-5"
    if "YM08" in text or "YM8" in text or "Y08" in text:
        return "Y-8"
    return None


def make_precheck_cfg(start, end, ro_dir, center_lat, center_lon):
    ro_kwargs = dict(RO_KWARGS_TEMPLATE)
    ro_kwargs["podtc_dir"] = str(ro_dir)
    return CycleConfig(
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=ROI_RADIUS_KM,
        grid_radius_deg=GRID_RADIUS_DEG,
        grid_margin_km=0.0,
        max_tec_per_batch=MAX_TEC_PER_BATCH,
        obs_sigma=OBS_SIGMA_TECU,
        obs_sources="RO",
        ro_kwargs=ro_kwargs,
    )


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def count_hourly_ro(args: argparse.Namespace):
    requested_dates = list(iter_dates(args.start_date, args.end_date))
    start_all = datetime.combine(
        args.start_date, datetime.min.time(), timezone.utc
    )
    n_hours = len(requested_dates) * 24

    total = np.zeros(n_hours, dtype=int)
    by_satellite = {
        satellite: np.zeros(n_hours, dtype=int) for satellite in SATELLITES
    }

    stats = {
        "processed_hours": 0,
        "missing_hour_folders": 0,
        "accepted_ro": 0,
        "accepted_tec_rays": 0,
        "failed_hours": 0,
    }

    for hour_index in range(n_hours):
        start = start_all + timedelta(hours=hour_index)
        end = start + timedelta(hours=1)
        folder = ro_day_dir(args.ro_dir, start.date())

        if not folder.is_dir():
            stats["missing_hour_folders"] += 1
            print(
                f"WARNING: no RO folder for {start:%Y-%m-%d} "
                f"({start:%Y.%j}); {start:%H}:00 remains zero"
            )
            continue

        cfg = make_precheck_cfg(
            start, end, folder,
            args.roi_center_lat, args.roi_center_lon,
        )

        try:
            batches = observation_stream.assemble(cfg)
        except Exception as exc:
            stats["failed_hours"] += 1
            print(f"WARNING: {start:%Y-%m-%d %H}:00 failed: {exc}")
            continue

        entries = unique_ro_entries(batches)
        n_ro = len(entries)
        n_tec = int(sum(len(batch.y_obs) for batch in batches))

        total[hour_index] = n_ro
        stats["processed_hours"] += 1
        stats["accepted_ro"] += n_ro
        stats["accepted_tec_rays"] += n_tec

        for entry in entries:
            satellite = satellite_from_entry(entry)
            if satellite in by_satellite:
                by_satellite[satellite][hour_index] += 1

        print(
            f"{start:%Y-%m-%d %H}:00 UTC | "
            f"accepted RO={n_ro:2d} | TEC rays={n_tec:4d}"
        )

    hour_starts = [
        start_all + timedelta(hours=index) for index in range(n_hours)
    ]
    return hour_starts, total, by_satellite, stats


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
    axis.set_xlabel("Number of accepted RO occultations in one UTC hour", fontsize=args.font_size)
    axis.set_ylabel("Number of one-hour intervals", fontsize=args.font_size)
    axis.set_title(
        f"Hourly RO availability inside {ROI_RADIUS_KM:g}-km ROI "
        f"(65% full-LOS criterion)\n"
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
        f"Grid radius: {GRID_RADIUS_DEG:g}° | Hours: {hourly_total.size} | "
        f"Accepted RO: {int(hourly_total.sum())}",
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
    time_axis.set_xlabel("Number of accepted RO occultations in one UTC hour", fontsize=args.font_size)
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
    parser.add_argument("--roi-center-lat", type=float, default=DEFAULT_TRO_LAT)
    parser.add_argument("--roi-center-lon", type=float, default=DEFAULT_TRO_LON)
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

    print()
    print("RO acceptance settings:")
    print(f"  ROI radius             : {ROI_RADIUS_KM:g} km")
    print(f"  Grid radius            : {GRID_RADIUS_DEG:g} deg")
    print(f"  ROI mode               : {RO_KWARGS_TEMPLATE['roi_mode']}")
    print(f"  LOS fraction required  : {RO_KWARGS_TEMPLATE['fraction_required']}")
    print("  Altitude limit         : auto mean LEO altitude per occultation")
    print(f"  Minimum valid rays     : {RO_KWARGS_TEMPLATE['min_valid_rays']}")
    print(f"  Max rays / occultation : {RO_KWARGS_TEMPLATE['max_rays_per_occultation']}")
    print()
    print(f"Hourly intervals represented: {len(total)}")
    print(f"Hours processed successfully: {stats['processed_hours']}")
    print(f"Hours missing RO folder:      {stats['missing_hour_folders']}")
    print(f"Hours failed:                 {stats['failed_hours']}")
    print(f"Accepted RO occultations:     {stats['accepted_ro']}")
    print(f"Accepted TEC rays:            {stats['accepted_tec_rays']}")
    print(f"Hours with zero accepted RO:  {int(np.count_nonzero(total == 0))}")
    print("Accepted RO by satellite: " + ", ".join(
        f"{satellite}={int(values.sum())}"
        for satellite, values in by_satellite.items()
    ))
    print(f"Saved figure: {args.output_figure}")
    if args.output_csv is not None:
        print(f"Saved hourly counts: {args.output_csv}")


if __name__ == "__main__":
    main()
