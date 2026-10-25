#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild forecast/analysis diagnostics for hourly assimilation runs from the
files a run leaves behind, so runs can be scored after the fact.

Input layout (one folder per hour, as written by
``run_hourly_all4styles_boost_compare.py``)::

    <output_root>/<YYYY-MM-DD>_<HHMM><HHMM>/
        accepted_ro_observations.nc
        dual_isr_errors.csv                       (optional, used for QC)
        <condition>/                              (e.g. no_boost, boosted)
            *_edp_samples.nc
            *_<style>_parameterized.nc
            *_<style>_ensembles/ensemble_batch_NNNN.nc

For every hour this script writes ``<out_dir>/<hour folder>.npz`` holding

  - ray-level TEC: observed ``y``, tangent altitude ``tang``, occultation
    index ``occ``, and per run (``<condition>|<style>``) the forecast and
    analysis ensemble-mean TEC ``Yf``/``Ya`` plus the part of it coming from
    grid levels below ``--e-top-km`` (``YEf``/``YEa``);
  - ISR-site profiles on the model altitude grid for every ISR station:
    forecast/analysis ensemble mean, std, and std of log10 Ne
    (``<run>|<station>|fmean`` ... ``alogstd``);
  - ``HG_<station>``: H_grid @ G^T, shape (n_rays, n_height) -- the TEC
    sensitivity of each ray to a unit Ne change at each level of the ISR
    column (G = the barycentric interpolation of the grid onto the site);
  - ``qc``: max relative difference between the rebuilt analysis error and
    ``dual_isr_errors.csv`` for each run/station (should be ~1e-12).

The forecast is the prior ensemble decoded from ``*_parameterized.nc``; the
analysis is the last saved ``ensemble_batch_*.nc`` (the final analysis of the
hour). All rays of all batches are scored against the final analysis.

Example::

    python Run_Scoring/extract_run_diagnostics.py \\
        --output-root "/path/to/tomography_output" \\
        --out-dir /path/to/scoring/extract \\
        --isr TRO=/path/TROISR2025_nonan.nc --isr SVL=/path/SVLISR2025_nonan.nc \\
        --workers 3

Memory: each worker holds one decoded ensemble (n_height x n_geo x members,
~200 MB for the standard 82 x 161 x 2000 grid) plus the parameterization;
3-4 workers fit in 16 GB. Reading from a Google Drive mount is I/O bound
(~1.5 GB per hour folder). Existing outputs are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HOUR_DIR_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})(\d{2})$")


def parse_hour_dir(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """'2025-10-16_14001500' -> (2025-10-16 14:00, 2025-10-16 15:00)."""
    m = HOUR_DIR_PATTERN.match(name)
    if m is None:
        return None
    day, h0, m0, h1, m1 = m.groups()
    start = pd.Timestamp(f"{day} {h0}:{m0}")
    end = pd.Timestamp(f"{day} {h1}:{m1}")
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end


def find_hour_dirs(output_root: Path) -> list[Path]:
    return sorted(
        p for p in output_root.iterdir()
        if p.is_dir() and parse_hour_dir(p.name) is not None
        and (p / "accepted_ro_observations.nc").is_file()
    )


def find_runs(cond_dir: Path) -> list[str]:
    """Styles that have both a parameterization and a saved analysis."""
    styles = []
    for f in sorted(cond_dir.glob("*_parameterized.nc")):
        m = re.search(r"_\d{8}_\d{4}_\d{4}_(.+)_parameterized\.nc$", f.name)
        if m is None:
            continue
        style = m.group(1)
        if list(cond_dir.glob(f"*_{style}_ensembles/ensemble_batch_*.nc")):
            styles.append(style)
    return styles


def _single(pattern: str) -> str:
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one file for {pattern!r}, found {len(hits)}")
    return hits[0]


def extract_hour(hour_dir: Path, out_dir: Path, isr_files: dict[str, str], args) -> str:
    """Rebuild one hour folder; returns a one-line status."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TQDM_DISABLE", "1")

    from Assimilation_Cycle.cycle_config import CycleConfig
    from Assimilation_Cycle import observation_stream, output
    from Assimilation_Cycle.ensemble_io import load_ensemble_netcdf
    from Assimilation_Cycle.isr_comparison import load_isr_dataset
    from Parameterization import Parameterized_EDPSamples
    from Ensemble_Kalman_Engine import EnsembleState
    from Ensemble_Kalman_Engine.observation_operator import GenericObservationOperator
    import edp_samples as E

    out_path = out_dir / f"{hour_dir.name}.npz"
    start, end = parse_hour_dir(hour_dir.name)
    isrs = {code: load_isr_dataset(path) for code, path in isr_files.items()}

    cfg = CycleConfig(
        start_time=start.isoformat(), end_time=end.isoformat(),
        center_lat=args.center_lat, center_lon=args.center_lon,
        radius_km=args.radius_km, grid_radius_deg=args.grid_radius_deg, grid_margin_km=0.0,
        max_tec_per_batch=args.max_tec_per_batch, obs_sigma=args.obs_sigma,
        obs_sources="RO", ro_observations_path=str(hour_dir / "accepted_ro_observations.nc"),
    )
    batches = observation_stream.assemble(cfg)

    y = np.concatenate([b.y_obs for b in batches])
    tang = np.full(y.shape, np.nan)
    occ = np.zeros(y.shape, dtype=int)
    offset, k_occ = 0, 0
    for b in batches:
        for entry, sl in b.entry_ray_ranges:
            idx = slice(offset + sl.start, offset + sl.stop)
            if entry.tangent_alt_km is not None:
                tang[idx] = entry.tangent_alt_km
            occ[idx] = k_occ
            k_occ += 1
        offset += len(b.y_obs)

    res = {
        "y": y, "tang": tang, "occ": occ, "n_batches": len(batches),
        "window_start": str(start), "window_end": str(end),
        "batch_start": str(batches[0].start_time), "batch_end": str(batches[-1].end_time),
        "obs_sigma": args.obs_sigma, "e_top_km": args.e_top_km,
        "stations": np.array(list(isrs)),
        "station_lat": np.array([isrs[c].station.lat for c in isrs]),
        "station_lon": np.array([isrs[c].station.lon for c in isrs]),
    }

    dual_path = hour_dir / "dual_isr_errors.csv"
    dual = pd.read_csv(dual_path) if dual_path.is_file() and dual_path.stat().st_size > 5 else None

    conditions = sorted(
        p.name for p in hour_dir.iterdir()
        if p.is_dir() and list(p.glob("*_edp_samples.nc"))
    )
    if args.conditions:
        conditions = [c for c in conditions if c in args.conditions]

    H = None
    runs, qc = [], []
    for cond in conditions:
        cond_dir = hour_dir / cond
        edp = E.EDPSamples.fromNetCDF(_single(str(cond_dir / "*_edp_samples.nc")))
        alt = np.asarray(edp.altitude, dtype=float)
        n_h, n_g = len(alt), edp.geolocation.shape[0]

        if H is None:
            # Ray geometry and grid are the same for every condition/style of
            # an hour, so H_grid is built once (the slow step, ~1 min).
            H = np.vstack([
                edp.get_observation_operator(b.podTc2_data, num_segments=args.num_segments)
                for b in batches
            ])
            res["alt"] = alt
            e_mask = np.repeat(alt < args.e_top_km, n_g)
            for code, isr in isrs.items():
                positions = np.array([[isr.station.lat, isr.station.lon, float(alt[0])]])
                _, _, idx_mesh, weight_mesh = edp.interp(positions, coordinate="lla")
                tri = int(idx_mesh[0])
                if tri == -1:
                    from scipy.spatial import cKDTree
                    _, nearest = cKDTree(edp.geolocation).query([[isr.station.lon, isr.station.lat]])
                    verts, w = [int(nearest[0])], [1.0]
                else:
                    verts, w = edp.mesh[tri], weight_mesh[0]
                G = np.zeros((n_h, n_h * n_g))
                for z in range(n_h):
                    for v, wv in zip(verts, w):
                        G[z, z * n_g + v] += wv
                res[f"HG_{code}"] = H @ G.T

        styles = find_runs(cond_dir)
        if args.styles:
            styles = [s for s in styles if s in args.styles]

        for style in styles:
            pes = Parameterized_EDPSamples.fromNetCDF(_single(str(cond_dir / f"*_{style}_parameterized.nc")))
            prior = EnsembleState.from_parameterized_edp_samples(pes)
            final_batch = sorted(glob.glob(str(cond_dir / f"*_{style}_ensembles" / "ensemble_batch_*.nc")))[-1]
            analysis, _ = load_ensemble_netcdf(final_batch)
            op = GenericObservationOperator(
                parameterization=pes.Parameterization, H_grid=H,
                param_shape=prior.param_shape, n_height=n_h, n_geo=n_g, altitude=alt,
            )
            key = f"{cond}|{style}"
            runs.append(key)
            for tag, ens in (("f", prior), ("a", analysis)):
                density = op.decode(ens.to_param_shape())          # (n_h, n_g, members)
                mean_flat = density.mean(axis=-1).reshape(-1)
                res[f"{key}|Y{tag}"] = H @ mean_flat
                res[f"{key}|YE{tag}"] = H[:, e_mask] @ mean_flat[e_mask]
                for code, isr in isrs.items():
                    prof = output._interpolate_field_at_latlon(edp, density, isr.station.lat, isr.station.lon)
                    res[f"{key}|{code}|{tag}mean"] = prof.mean(axis=-1)
                    res[f"{key}|{code}|{tag}std"] = prof.std(axis=-1)
                    res[f"{key}|{code}|{tag}logstd"] = np.log10(np.clip(prof, 1.0, None)).std(axis=-1)
                del density

            if dual is not None:
                for code, isr in isrs.items():
                    d = dual[(dual["style"] == style) & (dual["condition"] == cond) & (dual["station"] == code)]
                    if d.empty:
                        continue
                    d = d.sort_values("altitude_km")
                    mid = start + (end - start) / 2
                    j = int(np.argmin(np.abs(isr.time - mid)))
                    log_a = np.log10(np.clip(res[f"{key}|{code}|amean"], 1.0, None))
                    model = 10.0 ** np.interp(isr.altitude, alt, log_a, left=np.nan, right=np.nan)
                    err = np.abs(model - isr.density[:, j])
                    stored = d["analysis_error_m3"].to_numpy()
                    qc.append((cond, style, code, float(np.nanmax(np.abs(err - stored) / stored))))

    res["runs"] = np.array(runs)
    res["qc"] = np.array([f"{c}|{s}|{st}|{v:.3e}" for c, s, st, v in qc])
    np.savez_compressed(out_path, **res)
    worst = max((v for *_, v in qc), default=None)
    return f"{hour_dir.name}: {len(runs)} runs, QC max rel diff = " + (f"{worst:.2e}" if worst is not None else "n/a (no dual_isr_errors)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True, type=Path, help="folder holding the hour folders")
    p.add_argument("--out-dir", required=True, type=Path, help="where <hour>.npz files are written")
    p.add_argument("--isr", action="append", required=True, metavar="CODE=PATH",
                   help="ISR station file, e.g. TRO=/path/TROISR2025_nonan.nc (repeatable); "
                        "CODE must match the 'station' column of dual_isr_errors.csv for QC")
    p.add_argument("--hours", nargs="*", help="hour folder names to process (default: all)")
    p.add_argument("--conditions", nargs="*", help="limit to these condition subfolders")
    p.add_argument("--styles", nargs="*", help="limit to these styles")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    # Must match the run configuration (defaults = run_hourly_all4styles_boost_compare.py).
    p.add_argument("--center-lat", type=float, default=69.6)
    p.add_argument("--center-lon", type=float, default=19.2)
    p.add_argument("--radius-km", type=float, default=2000.0)
    p.add_argument("--grid-radius-deg", type=float, default=18.0)
    p.add_argument("--max-tec-per-batch", type=int, default=100_000)
    p.add_argument("--obs-sigma", type=float, default=3.0, help="TEC observation error (TECU)")
    p.add_argument("--num-segments", type=int, default=1000)
    p.add_argument("--e-top-km", type=float, default=150.0, help="top of the E region for the E-share of model TEC")
    return p.parse_args()


def main():
    args = parse_args()
    isr_files = dict(item.split("=", 1) for item in args.isr)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hour_dirs = find_hour_dirs(args.output_root)
    if args.hours:
        hour_dirs = [h for h in hour_dirs if h.name in set(args.hours)]
    todo = [h for h in hour_dirs if args.overwrite or not (args.out_dir / f"{h.name}.npz").exists()]
    print(f"{len(hour_dirs)} hour folders, {len(todo)} to process, {args.workers} workers", flush=True)

    if args.workers <= 1:
        for h in todo:
            print(extract_hour(h, args.out_dir, isr_files, args), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_hour, h, args.out_dir, isr_files, args): h for h in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # keep going; report the failing hour
                print(f"{futures[fut].name}: FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
