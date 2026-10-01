#!/usr/bin/env python3
"""Create Tromso ISR raw/median profile plots for several averaging durations.

For each UTC-aligned interval, plots all native 100–1000 km Ne profiles in
gray and overlays a median profile. Also creates an hourly composite: the
hour's raw profiles are gray and medians from the beginning of the hour to
5/10/15/30/60 minutes are drawn in their designated colors.

Example:
  python tromso_ne_median_interval_sweep_2025.py \
    /home/pin/Desktop/tomography_project/Data/ISR_Data/TRO/ --year 2025
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

LOCAL_TZ = ZoneInfo("Europe/Oslo")
DURATIONS = (5, 10, 15, 30, 60)
COLORS = {5: "tab:blue", 10: "tab:green", 15: "tab:red", 30: "tab:purple", 60: "black"}


@dataclass
class Profile:
    timestamp: float
    altitude_km: np.ndarray
    ne: np.ndarray


def float_data(variable):
    value = variable[:]
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    return np.asarray(value, dtype=float).squeeze()


def orient_profiles(variable, ntime, name):
    a = float_data(variable)
    if a.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {a.shape}")
    dims = tuple(variable.dimensions)
    if dims == ("timestamps", "range"):
        return a
    if dims == ("range", "timestamps"):
        return a.T
    if a.shape[0] == ntime and a.shape[1] != ntime:
        return a
    if a.shape[1] == ntime and a.shape[0] != ntime:
        return a.T
    raise ValueError(f"Cannot identify time axis for {name}: {dims}, {a.shape}")


def load_file(path, alt_min, alt_max, ne_min, ne_max):
    profiles = []
    with netCDF4.Dataset(path) as ds:
        missing = {"timestamps", "gdalt", "ne"} - set(ds.variables)
        if missing:
            raise ValueError(f"missing variables: {', '.join(sorted(missing))}")
        times = float_data(ds.variables["timestamps"]).reshape(-1)
        alt = orient_profiles(ds.variables["gdalt"], len(times), "gdalt")
        ne = orient_profiles(ds.variables["ne"], len(times), "ne")
        if alt.shape != ne.shape or alt.shape[0] != len(times):
            raise ValueError(f"shape mismatch: times={times.shape}, alt={alt.shape}, ne={ne.shape}")
        units = getattr(ds.variables["timestamps"], "units", "")
        if " since " in units:
            times = netCDF4.num2date(times, units, only_use_cftime_datetimes=False,
                                     only_use_python_datetimes=True)
            times = np.array([v.replace(tzinfo=timezone.utc).timestamp() if v.tzinfo is None
                              else v.timestamp() for v in times], dtype=float)
        # Read the actual altitude per profile. Do not require the altitude grid
        # to be constant across time and do not fill gaps between native gates.
        for i, stamp in enumerate(times):
            if not np.isfinite(stamp):
                continue
            z = alt[i]
            density = ne[i]
            good = (np.isfinite(z) & np.isfinite(density) & (z >= alt_min) &
                    (z <= alt_max) & (density > ne_min) & (density <= ne_max))
            if np.count_nonzero(good) < 2:
                continue
            z, density = z[good], density[good]
            order = np.argsort(z)
            profiles.append(Profile(float(stamp), z[order], density[order]))
    return profiles


def median_profile(profiles, target_altitudes):
    """Interpolate each native profile only within its altitude span, then median."""
    rows = []
    for p in profiles:
        # Collapse duplicate altitude gates before interpolation.
        z, inverse = np.unique(p.altitude_km, return_inverse=True)
        if z.size < 2:
            continue
        density = np.array([np.median(p.ne[inverse == j]) for j in range(z.size)])
        row = np.full(target_altitudes.shape, np.nan)
        inside = (target_altitudes >= z[0]) & (target_altitudes <= z[-1])
        row[inside] = np.interp(target_altitudes[inside], z, density)
        rows.append(row)
    if not rows:
        return np.full(target_altitudes.shape, np.nan)
    values = np.asarray(rows)
    result = np.full(target_altitudes.shape, np.nan)
    # Avoid warnings at altitudes that no profile in this interval reaches.
    for j in range(target_altitudes.size):
        finite = values[:, j][np.isfinite(values[:, j])]
        if finite.size:
            result[j] = np.median(finite)
    return result


def plot_raw(ax, profiles):
    for p in profiles:
        ax.plot(p.ne, p.altitude_km, "--*", color="0.60", alpha=0.42,
                lw=1.0, ms=4.0, zorder=1)


def finish_axes(ax, title, ne_min, ne_max, alt_min, alt_max):
    ax.set_xscale("log")
    ax.set_xlim(ne_min, ne_max)
    ax.set_ylim(alt_min, alt_max)
    ax.set_xlabel("Electron density Ne (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(axis="both", labelsize=11)


def aligned_start(stamp, interval_minutes):
    dt = datetime.fromtimestamp(stamp, timezone.utc)
    minute = (dt.minute // interval_minutes) * interval_minutes
    return dt.replace(minute=minute, second=0, microsecond=0).timestamp()


def format_time(stamp):
    utc = datetime.fromtimestamp(stamp, timezone.utc)
    local = utc.astimezone(LOCAL_TZ)
    return f"{utc:%Y-%m-%d %H:%M} UTC / {local:%H:%M} LT"


def make_altitude_grid(alt_min, alt_max):
    """Use coarser median markers aloft, like the established Tromso plots."""
    bands = ((80, 120, 3), (120, 200, 10), (200, 300, 15),
             (300, 450, 25), (450, 700, 25))
    values = [np.arange(max(lo, alt_min), min(hi, alt_max) + 1e-9, step)
              for lo, hi, step in bands if min(hi, alt_max) >= max(lo, alt_min)]
    if not values:
        return np.linspace(alt_min, alt_max, 50)
    result = np.unique(np.concatenate(values))
    if result[0] > alt_min:
        result = np.insert(result, 0, alt_min)
    if result[-1] < alt_max:
        result = np.append(result, alt_max)
    return result


def save_interval_plot(subset, start, duration, output_dir, altitudes, args):
    if not subset:
        return False
    median = median_profile(subset, altitudes)
    fig, ax = plt.subplots(figsize=(7.0, 9.0))
    plot_raw(ax, subset)
    ax.plot(median, altitudes, "--*", color=COLORS[duration], lw=2.2, ms=6.0,
            label=f"{duration}-min median ({len(subset)} profiles)", zorder=3)
    finish_axes(ax, f"Tromsø ISR Ne profiles — {format_time(start)} to {duration} min",
                args.ne_min, args.ne_max, args.alt_min, args.alt_max)
    ax.legend(loc="best")
    fig.tight_layout()
    stamp = datetime.fromtimestamp(start, timezone.utc).strftime("%Y%m%d_%H%M")
    fig.savefig(output_dir / f"tromso_ne_{duration}min_{stamp}UTC.png", dpi=args.dpi)
    plt.close(fig)
    return True


def save_hourly_plot(hour_profiles, hour_start, output_dir, altitudes, args):
    if not hour_profiles:
        return False
    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    plot_raw(ax, hour_profiles)
    handles = []
    labels = []
    for duration in DURATIONS:
        end = hour_start + duration * 60
        subset = [p for p in hour_profiles if p.timestamp < end]
        if not subset:
            continue
        med = median_profile(subset, altitudes)
        line, = ax.plot(med, altitudes, "--*", color=COLORS[duration], lw=2.2, ms=5.5,
                        label=f"00–{duration:02d} min ({len(subset)} profiles)", zorder=3+duration)
        handles.append(line)
        labels.append(line.get_label())
    finish_axes(ax, f"Tromsø ISR Ne — {format_time(hour_start)} to 60 min\n"
                f"Gray: all {len(hour_profiles)} raw profiles in the hour",
                args.ne_min, args.ne_max, args.alt_min, args.alt_max)
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    stamp = datetime.fromtimestamp(hour_start, timezone.utc).strftime("%Y%m%d_%H")
    fig.savefig(output_dir / f"tromso_ne_cprtime_{stamp}00UTC.png", dpi=args.dpi)
    plt.close(fig)
    return True


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path, help="Directory containing ISR NetCDF files")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--output-root", type=Path,
                   default=Path("/home/pin/Desktop/IonosphereTomography-pin_dev/ISR_PCA/figure"))
    p.add_argument("--alt-min", type=float, default=80.0)
    p.add_argument("--alt-max", type=float, default=700.0)
    p.add_argument("--ne-min", type=float, default=1e9)
    p.add_argument("--ne-max", type=float, default=1e12)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    return p


def main():
    args = make_parser().parse_args()
    if args.alt_min >= args.alt_max or args.ne_min <= 0 or args.ne_min >= args.ne_max:
        raise SystemExit("Invalid altitude or Ne limits")
    files = sorted(args.input_dir.rglob("*.nc") if args.recursive else args.input_dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"No .nc files found in {args.input_dir}")
    profiles = []
    errors = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Reading {path.name}", flush=True)
        try:
            loaded = load_file(path, args.alt_min, args.alt_max, args.ne_min, args.ne_max)
            profiles.extend(loaded)
            print(f"    usable profiles: {len(loaded)}", flush=True)
        except Exception as exc:
            errors.append((path, str(exc)))
            print(f"    SKIP: {type(exc).__name__}: {exc}", flush=True)
    year_start = datetime(args.year, 1, 1, tzinfo=timezone.utc).timestamp()
    year_end = datetime(args.year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
    profiles = sorted((p for p in profiles if year_start <= p.timestamp < year_end), key=lambda p: p.timestamp)
    if not profiles:
        raise SystemExit(f"No usable profiles found in {args.year}")
    altitudes = make_altitude_grid(args.alt_min, args.alt_max)
    args.output_root.mkdir(parents=True, exist_ok=True)
    # interval_counts = {}
    # for duration in DURATIONS:
    #     out = args.output_root / f"{duration}min"
    #     out.mkdir(parents=True, exist_ok=True)
    #     groups = {}
    #     for profile in profiles:
    #         start = aligned_start(profile.timestamp, duration)
    #         groups.setdefault(start, []).append(profile)
    #     count = sum(save_interval_plot(group, start, duration, out, altitudes, args)
    #                 for start, group in sorted(groups.items()))
    #     interval_counts[duration] = count
    cpr = args.output_root / "cprtime"
    cpr.mkdir(parents=True, exist_ok=True)
    hour_groups = {}
    for profile in profiles:
        start = aligned_start(profile.timestamp, 60)
        hour_groups.setdefault(start, []).append(profile)
    hourly_count = sum(save_hourly_plot(group, start, cpr, altitudes, args)
                       for start, group in sorted(hour_groups.items()))
    print("\nFinished")
    print(f"Year {args.year} usable profiles: {len(profiles)}")
    # for duration, count in interval_counts.items():
    #     print(f"{duration}min plots: {count} -> {args.output_root / f'{duration}min'}")
    print(f"Hourly cprtime plots: {hourly_count} -> {cpr}")
    if errors:
        print(f"Skipped files due to read/format errors: {len(errors)}")
        for path, message in errors:
            print(f"  {path}: {message}")


if __name__ == "__main__":
    main()
