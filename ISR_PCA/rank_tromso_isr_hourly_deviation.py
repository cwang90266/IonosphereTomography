#!/usr/bin/env python3
"""Rank Tromso ISR time windows by relative Ne deviation from a short-term median.

The score follows the uncertainty plot definition used by batch_tromso_ne_median:
100 * abs(raw Ne - median Ne) / median Ne. Only 100--500 km is scored.
Input netCDF files are grouped by UTC day; candidate windows start at :00 or :30
by default. A window is ranked only when it has enough valid minutes.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np

REQUIRED_VARIABLES = {"timestamps", "gdalt", "ne"}


def float_values(variable: netCDF4.Variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=float).squeeze()


def range_by_time(variable: netCDF4.Variable, n_times: int, name: str) -> np.ndarray:
    values = float_values(variable)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got {values.shape}")
    dims = tuple(variable.dimensions)
    if dims == ("timestamps", "range"):
        values = values.T
    elif dims != ("range", "timestamps"):
        if values.shape[1] == n_times and values.shape[0] != n_times:
            pass
        elif values.shape[0] == n_times and values.shape[1] != n_times:
            values = values.T
        else:
            raise ValueError(
                f"Cannot identify time axis for {name}: dimensions={dims}, "
                f"shape={values.shape}, timestamps={n_times}"
            )
    if values.shape[1] != n_times:
        raise ValueError(f"{name} time dimension does not match timestamps")
    return values


def profile_on_altitude_grid(
    altitude_km: np.ndarray,
    electron_density: np.ndarray,
    targets_km: np.ndarray,
    ne_floor: float,
    ne_ceiling: float,
) -> np.ndarray:
    """Bin actual native gates without reusing one gate at multiple altitudes."""
    result = np.full(targets_km.shape, np.nan)
    if targets_km.size < 2:
        raise ValueError("At least two altitude-bin centers are required")
    spacing = float(np.median(np.diff(targets_km)))
    valid = (
        np.isfinite(altitude_km)
        & np.isfinite(electron_density)
        & (altitude_km >= targets_km[0])
        & (altitude_km <= targets_km[-1])
        & (electron_density > ne_floor)
        & (electron_density <= ne_ceiling)
    )
    z = altitude_km[valid]
    ne = electron_density[valid]
    if z.size == 0:
        return result

    # Each observed gate enters exactly one 10-km bin. This avoids nearest-gate
    # interpolation filling many bins from a single sparse measurement.
    edges = np.concatenate((
        [targets_km[0] - spacing / 2],
        (targets_km[:-1] + targets_km[1:]) / 2,
        [targets_km[-1] + spacing / 2],
    ))
    indices = np.searchsorted(edges, z, side="right") - 1
    in_grid = (indices >= 0) & (indices < targets_km.size)
    for index in np.unique(indices[in_grid]):
        result[index] = np.median(ne[in_grid & (indices == index)])
    return result


def load_daily_profiles(
    files: list[Path], targets_km: np.ndarray, ne_floor: float, ne_ceiling: float, year: int
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Read all files and group mapped profiles by UTC date."""
    daily: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for file_number, path in enumerate(files, start=1):
        print(f"[{file_number}/{len(files)}] Reading {path.name}")
        with netCDF4.Dataset(path) as ds:
            missing = REQUIRED_VARIABLES - set(ds.variables)
            if missing:
                raise ValueError(f"{path}: missing variables {sorted(missing)}")
            timestamps = np.atleast_1d(float_values(ds.variables["timestamps"]))
            alt = range_by_time(ds.variables["gdalt"], timestamps.size, "gdalt")
            ne = range_by_time(ds.variables["ne"], timestamps.size, "ne")
            if alt.shape != ne.shape:
                raise ValueError(f"{path}: gdalt and ne shapes differ")

            for col, timestamp in enumerate(timestamps):
                if not np.isfinite(timestamp):
                    continue
                try:
                    dt = datetime.fromtimestamp(float(timestamp), timezone.utc)
                except (OverflowError, OSError, ValueError):
                    continue
                if dt.year != year:
                    continue
                mapped = profile_on_altitude_grid(
                    alt[:, col], ne[:, col], targets_km, ne_floor, ne_ceiling
                )
                if np.any(np.isfinite(mapped)):
                    daily[dt.strftime("%Y-%m-%d")].append((float(timestamp), mapped))
    return daily


def make_minute_grids(
    profiles: list[tuple[float, np.ndarray]],
    day: datetime,
    n_altitudes: int,
    reference_bin_minutes: int,
    min_profiles_per_reference_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minute-level raw Ne and its reference-bin median Ne."""
    n_minutes = 24 * 60
    raw_samples: list[list[list[float]]] = [
        [[] for _ in range(n_minutes)] for _ in range(n_altitudes)
    ]
    for timestamp, mapped in profiles:
        minute = int((timestamp - day.timestamp()) // 60)
        if not 0 <= minute < n_minutes:
            continue
        for alt_i, value in enumerate(mapped):
            if np.isfinite(value):
                raw_samples[alt_i][minute].append(float(value))

    raw = np.full((n_altitudes, n_minutes), np.nan)
    for alt_i in range(n_altitudes):
        for minute in range(n_minutes):
            values = raw_samples[alt_i][minute]
            if values:
                raw[alt_i, minute] = np.median(values)

    n_bins = (n_minutes + reference_bin_minutes - 1) // reference_bin_minutes
    background = np.full((n_altitudes, n_bins), np.nan)
    for bin_i in range(n_bins):
        start = bin_i * reference_bin_minutes
        stop = min(start + reference_bin_minutes, n_minutes)
        # Count distinct minutes containing data, matching the profile-count rule.
        for alt_i in range(n_altitudes):
            values = raw[alt_i, start:stop]
            valid = values[np.isfinite(values)]
            if valid.size >= min_profiles_per_reference_bin:
                background[alt_i, bin_i] = np.median(valid)
    return raw, background


def rank_day_windows(
    day_text: str,
    profiles: list[tuple[float, np.ndarray]],
    targets_km: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    day = datetime.strptime(day_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    raw, background = make_minute_grids(
        profiles,
        day,
        targets_km.size,
        args.reference_bin_minutes,
        args.min_profiles_per_reference_bin,
    )
    bin_ids = np.arange(24 * 60) // args.reference_bin_minutes
    background_per_minute = background[:, bin_ids]
    valid = (
        np.isfinite(raw)
        & np.isfinite(background_per_minute)
        & (background_per_minute > 0)
    )
    deviation = np.full(raw.shape, np.nan)
    deviation[valid] = (
        100.0 * np.abs(raw[valid] - background_per_minute[valid])
        / background_per_minute[valid]
    )

    # One robust score per minute: median deviation across supported altitudes.
    minute_score = np.full(24 * 60, np.nan)
    minute_alt_coverage = valid.mean(axis=0)
    for minute in range(24 * 60):
        values = deviation[:, minute]
        values = values[np.isfinite(values)]
        if values.size and minute_alt_coverage[minute] >= args.min_altitude_coverage:
            minute_score[minute] = float(np.median(values))

    rows: list[dict[str, object]] = []
    profile_counts_by_minute = np.zeros(24 * 60, dtype=int)
    for timestamp, _ in profiles:
        minute = int((timestamp - day.timestamp()) // 60)
        if 0 <= minute < 24 * 60:
            profile_counts_by_minute[minute] += 1
    profile_count_prefix = np.concatenate(([0], np.cumsum(profile_counts_by_minute)))
    duration = args.window_minutes
    last_start = 24 * 60 - duration
    for start in range(0, last_start + 1):
        if start % 60 not in args.start_minute_offsets:
            continue
        end = start + duration
        scores = minute_score[start:end]
        scores = scores[np.isfinite(scores)]
        if scores.size < duration - args.max_missing_minutes:
            continue
        n_window_profiles = int(profile_count_prefix[end] - profile_count_prefix[start])
        start_dt = day + timedelta(minutes=start)
        end_dt = day + timedelta(minutes=end)
        rows.append({
            "utc_date": day_text,
            "start_utc": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_utc": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": duration,
            "valid_minutes": int(scores.size),
            "missing_minutes": int(duration - scores.size),
            "median_altitude_coverage": round(
                float(np.nanmedian(minute_alt_coverage[start:end])), 4
            ),
            "mean_abs_relative_deviation_pct": round(float(np.mean(scores)), 4),
            "median_abs_relative_deviation_pct": round(float(np.median(scores)), 4),
            "p90_minute_deviation_pct": round(float(np.percentile(scores, 90)), 4),
            "profiles_in_window": n_window_profiles,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing 2015 .nc files")
    parser.add_argument("--recursive", action="store_true", help="Search input_dir recursively")
    parser.add_argument("--output-csv", type=Path, default=Path("tromso_isr_hourly_ranking_2015.csv"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--monthly-minimum", type=int, default=1, help="At least this many windows from each month with qualifying candidates")
    parser.add_argument("--year", type=int, default=2015, help="UTC year to rank (default: 2015)")
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument(
        "--start-minute-offsets",
        type=lambda value: [int(part) for part in value.split(",")],
        default=[0, 30],
        help="Candidate start minutes within each hour (default: 0,30)",
    )
    parser.add_argument("--max-missing-minutes", type=int, default=5)
    parser.add_argument("--min-altitude-km", type=float, default=100.0)
    parser.add_argument("--max-altitude-km", type=float, default=500.0)
    parser.add_argument("--altitude-step-km", type=float, default=10.0)
    parser.add_argument(
        "--min-altitude-coverage",
        type=float,
        default=0.05,
        help="Minimum fraction of distinct 100-500 km altitude bins with a usable deviation per minute (default 0.05)",
    )
    parser.add_argument("--reference-bin-minutes", type=int, default=4)
    parser.add_argument(
        "--min-profiles-per-reference-bin",
        type=int,
        default=1,
        help="Minimum observed minute profiles in each reference bin (default 1)",
    )
    parser.add_argument("--ne-floor", type=float, default=1e7)
    parser.add_argument("--ne-ceiling", type=float, default=1e13)
    args = parser.parse_args()
    if args.window_minutes < 1 or args.window_minutes > 1440:
        parser.error("--window-minutes must be between 1 and 1440")
    if (not args.start_minute_offsets
            or any(offset < 0 or offset > 59 for offset in args.start_minute_offsets)
            or len(set(args.start_minute_offsets)) != len(args.start_minute_offsets)):
        parser.error("--start-minute-offsets must be unique values from 0 to 59")
    if not 0 <= args.max_missing_minutes < args.window_minutes:
        parser.error("--max-missing-minutes must be >=0 and < window length")
    if args.reference_bin_minutes < 1:
        parser.error("--reference-bin-minutes must be positive")
    if args.min_profiles_per_reference_bin < 1:
        parser.error("--min-profiles-per-reference-bin must be positive")
    if not 0 <= args.min_altitude_coverage <= 1:
        parser.error("--min-altitude-coverage must be in [0, 1]")
    if args.altitude_step_km <= 0 or args.min_altitude_km >= args.max_altitude_km:
        parser.error("Check altitude limits and --altitude-step-km")
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    if args.monthly_minimum < 0:
        parser.error("--monthly-minimum cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    pattern = "**/*.nc" if args.recursive else "*.nc"
    files = sorted(args.input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {args.input_dir}")

    targets = np.arange(
        args.min_altitude_km,
        args.max_altitude_km + args.altitude_step_km * 0.5,
        args.altitude_step_km,
    )
    daily = load_daily_profiles(files, targets, args.ne_floor, args.ne_ceiling, args.year)
    all_rows: list[dict[str, object]] = []
    required_minutes = args.window_minutes - args.max_missing_minutes
    for day_text in sorted(daily):
        rows = rank_day_windows(day_text, daily[day_text], targets, args)
        all_rows.extend(rows)
        valid_minute_count = sum(
            1 for row in rows if int(row["valid_minutes"]) >= required_minutes
        )
        print(f"{day_text}: {valid_minute_count} qualifying windows")

    sort_key = lambda row: (
        float(row["mean_abs_relative_deviation_pct"]),
        -int(row["valid_minutes"]),
        str(row["start_utc"]),
    )
    all_rows.sort(key=sort_key)

    # Reserve the best qualifying candidates in each month first, then fill the
    # remaining slots by global score. Months without valid windows are reported.
    selected: list[dict[str, object]] = []
    if args.monthly_minimum:
        by_month: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in all_rows:
            by_month[str(row["utc_date"])[:7]].append(row)
        required_slots = len(by_month) * args.monthly_minimum
        if required_slots > args.top_n:
            raise ValueError(
                f"top-n={args.top_n} is too small for {len(by_month)} represented "
                f"months at monthly-minimum={args.monthly_minimum}"
            )
        for month in sorted(by_month):
            selected.extend(by_month[month][: args.monthly_minimum])

    selected_keys = {str(row["start_utc"]) for row in selected}
    for row in all_rows:
        if len(selected) >= args.top_n:
            break
        if str(row["start_utc"]) not in selected_keys:
            selected.append(row)
            selected_keys.add(str(row["start_utc"]))
    selected.sort(key=sort_key)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "utc_date", "start_utc", "end_utc", "duration_minutes",
        "valid_minutes", "missing_minutes", "median_altitude_coverage",
        "mean_abs_relative_deviation_pct", "median_abs_relative_deviation_pct",
        "p90_minute_deviation_pct", "profiles_in_window",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            writer.writerow({"rank": rank, **row})

    months_with_data = sorted({day_text[:7] for day_text in daily})
    months_selected = sorted({str(row["utc_date"])[:7] for row in selected})
    months_without_selection = sorted(set(months_with_data) - set(months_selected))
    print(f"Qualifying windows: {len(all_rows)}")
    print(f"Months represented in selected windows: {', '.join(months_selected) or 'none'}")
    if months_without_selection:
        print(
            "Months with input profiles but no selected qualifying window: "
            + ", ".join(months_without_selection)
        )
    print(f"Saved top {len(selected)} to {args.output_csv}")
    if not selected:
        required = args.window_minutes - args.max_missing_minutes
        print(f"No windows had at least {required} usable minutes.")
        print("A minute is usable only if its 100-500 km deviation has sufficient altitude coverage and a reference median.")
        print("Try --min-altitude-coverage 0.01 or --min-profiles-per-reference-bin 1 if appropriate for your files.")


if __name__ == "__main__":
    main()
