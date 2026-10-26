#!/usr/bin/env python3
"""Plot Jicamarca ISR Ne profiles in a user-selected UTC interval and a keogram.

Examples:
    python Jic_ne_analysis.py jro20251109_050302.hdf5 --start '2025-11-09 05:00:00' --end '2025-11-09 06:00:00'
    python Jic_ne_analysis.py jro20251109_050302.hdf5 --start '2025-11-09 05:00:00' --end '2025-11-09 06:00:00' --outputdir ./figures
    python Jic_ne_analysis.py jro20251109_050302.hdf5 --list-available

Local time is mean local solar time (LT = UT + longitude/15 hours), not
civil clock time. The longitude is read from the Madrigal instrument metadata.
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext

UTC = timezone.utc


def parse_utc(value):
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date/time '{value}'. Use 'YYYY-MM-DD HH:MM:SS' (UT)."
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def read_metadata(file):
    dataset = file.get('/Metadata/Experiment Parameters')
    if dataset is None:
        return {}

    def decode(s):
        return s.decode('utf-8', errors='replace').strip() if isinstance(s, bytes) else str(s).strip()

    return {decode(row['name']).lower(): decode(row['value']) for row in dataset[:]}


def load_profiles(path):
    with h5py.File(path, 'r') as f:
        root = f['/Data/Array Layout']
        timestamp = np.asarray(root['timestamps'][:], dtype=float)
        altitude = np.asarray(root['gdalt'][:], dtype=float)
        density = np.asarray(root['2D Parameters/ne'][:], dtype=float)
        if density.shape == (len(altitude), len(timestamp)):
            density = density.T
        elif density.shape != (len(timestamp), len(altitude)):
            raise ValueError(
                f'Unexpected Ne shape {density.shape}; expected '
                f'({len(altitude)}, {len(timestamp)}) or its transpose.'
            )

        valid = np.isfinite(density) & (density > 0) & np.isfinite(altitude)[None, :]
        has_profile = np.any(valid, axis=1)

        meta = read_metadata(f)
        lat = float(meta.get('instrument latitude', 'nan'))
        lon = float(meta.get('instrument longitude', 'nan'))
        if not np.isfinite(lat) or not np.isfinite(lon):
            one_d = root['1D Parameters']
            if not np.isfinite(lat):
                lat = float(np.nanmedian(one_d['gdlatr'][:]))
            if not np.isfinite(lon):
                lon = float(np.nanmedian(one_d['gdlonr'][:]))

    lon = ((lon + 180) % 360) - 180
    order = np.argsort(timestamp)
    return timestamp[order], altitude, density[order], has_profile[order], lat, lon


def fmt(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def as_datetime(t):
    return datetime.fromtimestamp(float(t), tz=UTC)


def solar_time(ut, longitude):
    return ut + timedelta(hours=longitude / 15.0)


def coordinate_label(value, pos_hemisphere, neg_hemisphere):
    hemisphere = neg_hemisphere if value < 0 else pos_hemisphere
    return f'{abs(value):.2f}° {hemisphere}'


def floor_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def ceil_to_hour(dt):
    floored = floor_to_hour(dt)
    return floored if dt == floored else floored + timedelta(hours=1)


def make_altitude_edges(altitude_km):
    altitude_km = np.asarray(altitude_km, dtype=float)
    if altitude_km.size == 1:
        return np.array([altitude_km[0] - 0.5, altitude_km[0] + 0.5], dtype=float)
    deltas = np.diff(altitude_km)
    edges = np.empty(altitude_km.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (altitude_km[:-1] + altitude_km[1:])
    edges[0] = altitude_km[0] - 0.5 * deltas[0]
    edges[-1] = altitude_km[-1] + 0.5 * deltas[-1]
    return edges


def print_available(timestamps, has_profile, longitude):
    available = timestamps[has_profile]
    if len(available) == 0:
        print('No valid positive Ne measurements exist in this file.')
        return
    print(f'Available Ne profiles: {len(available)}')
    print(f'Overall UT range: {fmt(as_datetime(available[0]))} to {fmt(as_datetime(available[-1]))}')
    print('Note: ranges below describe discrete profile timestamps, not continuous coverage.')
    if len(available) == 1:
        cadence = None
    else:
        differences = np.diff(available)
        cadence = float(np.median(differences[differences > 0]))
    print(f'Typical profile cadence: {cadence / 60:.1f} min' if cadence else 'Single valid profile')
    gap_cutoff = 1.5 * cadence if cadence else float('inf')
    breaks = np.flatnonzero(np.diff(available) > gap_cutoff) + 1
    runs = np.split(available, breaks)
    print('Available observation runs (UT | local solar time):')
    for i, run in enumerate(runs, 1):
        first, last = as_datetime(run[0]), as_datetime(run[-1])
        first_lt, last_lt = solar_time(first, longitude), solar_time(last, longitude)
        print(f'  {i:2d}. UT {fmt(first)} to {fmt(last)} | '
              f'LT {fmt(first_lt)} to {fmt(last_lt)} | {len(run)} profiles')
    print('\nExact available profile timestamps (UT | LT):')
    for stamp in available:
        ut = as_datetime(stamp)
        print(f'  {fmt(ut)} UT | {fmt(solar_time(ut, longitude))} LT')


def build_output_paths(args):
    default_base = f'Jic_Ne_{args.start:%Y%m%d_%H%M%S}_{args.end:%Y%m%d_%H%M%S}'
    if args.output is not None:
        base_output = (args.outputdir / args.output.name) if args.outputdir else args.output
        stem = base_output.stem
        suffix = base_output.suffix if base_output.suffix else '.png'
        parent = base_output.parent
        profiles_path = parent / f'{stem}_profiles{suffix}'
        keogram_path = parent / f'{stem}_keogram{suffix}'
    else:
        parent = args.outputdir or args.file.parent
        profiles_path = parent / f'{default_base}_profiles.png'
        keogram_path = parent / f'{default_base}_keogram.png'
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    keogram_path.parent.mkdir(parents=True, exist_ok=True)
    return profiles_path, keogram_path


def plot_profiles(indices, ts, alt, ne, args, lat, lon, output_path):
    plt.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 27,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 19,
        'figure.dpi': 150,
    })

    fig, ax = plt.subplots(figsize=(12, 20))
    cmap = plt.get_cmap('turbo')
    colors = cmap(np.linspace(0.05, 0.95, len(indices)))
    for color, idx in zip(colors, indices):
        valid = np.isfinite(ne[idx]) & (ne[idx] > 0) & np.isfinite(alt)
        dt = as_datetime(ts[idx])
        lt = solar_time(dt, lon)
        ax.plot(ne[idx, valid], alt[valid], '--o', lw=2.0, ms=5.5, color=color, label=f'{lt:%H:%M:%S} LT')

    start_lt = solar_time(args.start, lon)
    end_lt = solar_time(args.end, lon)
    lat_label = coordinate_label(lat, 'N', 'S')
    lon_label = coordinate_label(lon, 'E', 'W')
    ax.set_title(
        f'Jicamarca ISR (Lat: {lat_label}; Lon: {lon_label}) Ne on {args.start:%Y-%m-%d}\n'
        f'UT {args.start:%H:%M:%S}-{args.end:%H:%M:%S}; '
        f'LT {start_lt:%H:%M:%S}-{end_lt:%H:%M:%S}',
        pad=17,
    )
    ax.set_xscale('log')
    ax.set_xlim(1e9, 1e12)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xlabel(r'Electron density $N_e$ (m$^{-3}$)', labelpad=10)
    ax.set_ylabel('Altitude (km)', labelpad=10)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', axis='x', alpha=0.15)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    ax.legend(
        title='Profile local solar time',
        title_fontsize=19,
        loc='upper right',
        framealpha=0.9,
        facecolor='white',
        edgecolor='0.7',
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_keogram(ts, alt, ne, good, args, lat, lon, output_path):
    """24-hour LT pcolor plot, retaining date information across midnight."""
    indices = np.flatnonzero(good)
    y_order = np.argsort(alt)
    y_edges = make_altitude_edges(alt[y_order])

    plt.rcParams.update({
        'font.size': 25, 'axes.titlesize': 30, 'axes.labelsize': 28,
        'xtick.labelsize': 24, 'ytick.labelsize': 24, 'figure.dpi': 150,
    })
    fig, ax = plt.subplots(figsize=(20, 12))
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')
    norm = LogNorm(vmin=1e9, vmax=1e12)

    good_times = np.sort(ts[np.any(np.isfinite(ne) & (ne > 0), axis=1)])
    dt = np.diff(good_times)
    dt = dt[dt > 0]
    cadence = float(np.median(dt)) if len(dt) else 60.0
    # Use nearest observations to limit each cell width, avoiding fake coverage.
    selected_times = np.sort(ts[good])
    local_dates_by_hour = {}
    for idx in indices:
        ut = float(ts[idx])
        loc = solar_time(as_datetime(ut), lon)
        local_hour = (loc.hour + loc.minute / 60 + loc.second / 3600
                      + loc.microsecond / 3.6e9)
        local_dates_by_hour.setdefault(loc.date(), []).append(local_hour)
        pos = np.searchsorted(selected_times, ut)
        gap_before = ut - selected_times[pos - 1] if pos else cadence
        gap_after = selected_times[pos + 1] - ut if pos + 1 < len(selected_times) else cadence
        half_hour = 0.5 * min(cadence, gap_before, gap_after) / 3600
        if half_hour <= 0:
            half_hour = 30 / 3600

        row = ne[idx, y_order].astype(float)
        row[~np.isfinite(row) | (row <= 0)] = np.nan
        if not np.any(np.isfinite(row)):
            continue
        # Wrap samples near midnight to opposite ends of the 0-24 h axis.
        for shift in (-24, 0, 24):
            left = max(0., local_hour - half_hour + shift)
            right = min(24., local_hour + half_hour + shift)
            if right > left:
                ax.pcolormesh([left, right], y_edges, row[:, None],
                              cmap=cmap, norm=norm, shading='flat', rasterized=True)

    lat_label = coordinate_label(lat, 'N', 'S')
    lon_label = coordinate_label(lon, 'E', 'W')
    local_dates = sorted(local_dates_by_hour)
    if len(local_dates) == 1:
        date_title = local_dates[0].strftime('%Y-%m-%d')
    else:
        date_title = f"{local_dates[0]:%Y-%m-%d} to {local_dates[-1]:%Y-%m-%d}"
    ax.set_title(f'Jicamarca ISR (Lat: {lat_label}; Lon: {lon_label}) Ne on {date_title}', pad=20)
    ax.set_xlabel('Local Time (hour)', labelpad=75)
    ax.set_ylabel('Altitude (km)', labelpad=12)
    ax.set_xlim(0, 24)
    tick_hours = np.arange(0, 25, 2)
    ax.set_xticks(tick_hours)
    ax.set_xticklabels([str(h) for h in tick_hours])

    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    # Local date captions below the axis, near the sampled sections. This avoids
    # implying that both sides of midnight are the same calendar day.
    for local_date, hours in sorted(local_dates_by_hour.items()):
        center = float(np.median(hours))
        if center < 1.5:
            center = 1.2
        elif center > 22.5:
            center = 22.8
        ax.text(center, -0.085, local_date.strftime('%m-%d'),
                ha='center', va='top', fontsize=22, fontweight='bold',
                transform=ax.get_xaxis_transform(), clip_on=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'Electron density $N_e$ (m$^{-3}$)', fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=Path, help='Jicamarca Madrigal .hdf5 file')
    parser.add_argument('--start', type=parse_utc, help='Start UT, e.g. "2025-11-09 05:00:00"')
    parser.add_argument('--end', type=parse_utc, help='End UT, e.g. "2025-11-09 06:00:00"')
    parser.add_argument('--list-available', action='store_true', help='Print all Ne observation times')
    parser.add_argument('--output', type=Path, help='Optional base output PNG filename/path')
    parser.add_argument('--outputdir', type=Path, help='Directory for saved PNG plots (defaults beside input file)')
    parser.add_argument('--ymin', type=float, help='Minimum altitude, km')
    parser.add_argument('--ymax', type=float, help='Maximum altitude, km')
    args = parser.parse_args()

    if not args.list_available and (args.start is None or args.end is None):
        parser.error('--start and --end are required unless --list-available is specified.')
    if args.start and args.end and args.end <= args.start:
        parser.error('--end must be later than --start.')

    ts, alt, ne, good, lat, lon = load_profiles(args.file)
    if args.list_available:
        print_available(ts, good, lon)
        return

    selected = good & (ts >= args.start.timestamp()) & (ts <= args.end.timestamp())
    indices = np.flatnonzero(selected)
    profiles_path, keogram_path = build_output_paths(args)
    if len(indices):
        plot_profiles(indices, ts, alt, ne, args, lat, lon, profiles_path)
        print(f'Plotted {len(indices)} Ne profiles; saved {profiles_path}')
    else:
        print(f'No Ne profiles found for requested window: {fmt(args.start)} to {fmt(args.end)} UT.\n')
        print_available(ts, good, lon)
    if np.any(good):
        plot_keogram(ts, alt, ne, good, args, lat, lon, keogram_path)
        print(f'Created full-file 24-hour LT keogram using {np.count_nonzero(good)} profiles; saved {keogram_path}')


if __name__ == '__main__':
    main()
