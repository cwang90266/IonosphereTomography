"""Horizontally uniform ray operator for each hour of an hourly tomography run.

For every hour folder this script rebuilds the ray operator H_grid (rays x
grid cells, the same one the EnKF used) and collapses it over the horizontal
grid:

    U = H_grid.reshape(n_rays, n_height, n_geo).sum(axis=2)

so that ``U @ ne`` is each ray's TEC (TECU, ne in m^-3) if the profile
``ne(z)`` held at every grid point. U is each ray's path length through each
model layer, in TEC units. Only ray geometry and the grid are needed, not the
ensembles, so this is much faster than extract_run_diagnostics.py.

Output: ``<out_dir>/<hour folder>.npz`` with ``U``, ``alt``, observed ``y``,
tangent altitude ``tang``, occultation index ``occ``, receiver id ``rec`` and
the tangent-point latitude/longitude of each ray's occultation
(``occ_lat``/``occ_lon``, at the TEC maximum).

Example::

    python Run_Scoring/extract_uniform_operator.py \\
        --output-root "/path/to/MIL_aug2529_2025_ROI2000kmRadius_Op3Boost" \\
        --out-dir /path/to/uniform_mil2k \\
        --center-lat 42.6195 --center-lon -71.4917 --radius-km 2000 --grid-radius-deg 18
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_run_diagnostics import _single, find_hour_dirs, parse_hour_dir  # noqa: E402  (sets sys.path)


def extract_hour(hour_dir: Path, out_dir: Path, args) -> str:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TQDM_DISABLE", "1")

    import netCDF4
    from Assimilation_Cycle.cycle_config import CycleConfig
    from Assimilation_Cycle import observation_stream
    import edp_samples as E

    start, end = parse_hour_dir(hour_dir.name)
    obs_path = hour_dir / "accepted_ro_observations.nc"
    cfg = CycleConfig(
        start_time=start.isoformat(), end_time=end.isoformat(),
        center_lat=args.center_lat, center_lon=args.center_lon,
        radius_km=args.radius_km, grid_radius_deg=args.grid_radius_deg, grid_margin_km=0.0,
        max_tec_per_batch=args.max_tec_per_batch, obs_sigma=5.0,
        obs_sources="RO", ro_observations_path=str(obs_path),
    )
    batches = observation_stream.assemble(cfg)

    with netCDF4.Dataset(obs_path) as ds:
        rec_ids = np.array([str(x) for x in ds["rec_id"][:]])
        labels = np.array([str(x) for x in ds["label"][:]])
        tlat = np.asarray(ds["extra_lat_tecmax_tangent"][:].filled(np.nan), dtype=float)
        tlon = np.asarray(ds["extra_lon_tecmax_tangent"][:].filled(np.nan), dtype=float)
    by_label = {lab: i for i, lab in enumerate(labels)}

    y = np.concatenate([b.y_obs for b in batches])
    tang = np.full(y.shape, np.nan)
    occ = np.zeros(y.shape, dtype=int)
    rec = np.full(y.shape, "", dtype=object)
    olat = np.full(y.shape, np.nan)
    olon = np.full(y.shape, np.nan)
    offset, k_occ = 0, 0
    for b in batches:
        for entry, sl in b.entry_ray_ranges:
            idx = slice(offset + sl.start, offset + sl.stop)
            if entry.tangent_alt_km is not None:
                tang[idx] = entry.tangent_alt_km
            occ[idx] = k_occ
            i = by_label.get(str(entry.label))
            if i is not None:
                rec[idx], olat[idx], olon[idx] = rec_ids[i], tlat[i], tlon[i]
            k_occ += 1
        offset += len(b.y_obs)

    cond_dir = sorted(p for p in hour_dir.iterdir() if p.is_dir() and list(p.glob("*_edp_samples.nc")))[0]
    edp = E.EDPSamples.fromNetCDF(_single(str(cond_dir / "*_edp_samples.nc")))
    alt = np.asarray(edp.altitude, dtype=float)
    n_h, n_g = len(alt), edp.geolocation.shape[0]
    U = np.vstack([
        np.asarray(edp.get_observation_operator(b.podTc2_data, num_segments=args.num_segments))
        .reshape(-1, n_h, n_g).sum(axis=2)
        for b in batches
    ])

    np.savez_compressed(
        out_dir / f"{hour_dir.name}.npz", U=U, alt=alt, y=y, tang=tang, occ=occ,
        rec=rec.astype(str), occ_lat=olat, occ_lon=olon,
        window_start=str(start), window_end=str(end),
    )
    return f"{hour_dir.name}: {len(y)} rays, {k_occ} RO, rec matched {np.mean(rec != ''):.2f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True, type=Path, help="folder holding the hour folders")
    p.add_argument("--out-dir", required=True, type=Path, help="where <hour>.npz files are written")
    p.add_argument("--hours", nargs="*", help="hour folder names to process (default: all)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    # Must match the run configuration (defaults = run_hourly_all4styles_boost_compare.py).
    p.add_argument("--center-lat", type=float, default=69.6)
    p.add_argument("--center-lon", type=float, default=19.2)
    p.add_argument("--radius-km", type=float, default=2000.0)
    p.add_argument("--grid-radius-deg", type=float, default=18.0)
    p.add_argument("--max-tec-per-batch", type=int, default=100_000)
    p.add_argument("--num-segments", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hour_dirs = find_hour_dirs(args.output_root)
    if args.hours:
        hour_dirs = [h for h in hour_dirs if h.name in set(args.hours)]
    # Only hours the run assimilated have a condition folder with EDP samples.
    hour_dirs = [h for h in hour_dirs if any(p.is_dir() and list(p.glob("*_edp_samples.nc")) for p in h.iterdir())]
    todo = [h for h in hour_dirs if args.overwrite or not (args.out_dir / f"{h.name}.npz").exists()]
    print(f"{len(hour_dirs)} hour folders, {len(todo)} to process, {args.workers} workers", flush=True)

    if args.workers <= 1:
        for h in todo:
            print(extract_hour(h, args.out_dir, args), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_hour, h, args.out_dir, args): h for h in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # keep going; report the failing hour
                print(f"{futures[fut].name}: FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
