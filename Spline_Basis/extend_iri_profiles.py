# -*- coding: utf-8 -*-
"""
Regenerate the IRI2020 prior members of existing ``edp_samples.nc`` files on a
new altitude grid (e.g. 60-900 km instead of 90-900 km), for the grid
optimization training set.

Each edp_samples.nc stores its members' IRI driving inputs
(``sampling_parameters``: hour, f107, ap, ig12, rz12), its ``geolocation`` and
``DateTime``, so the same members can be re-evaluated exactly with
``EDPSamples.get_IRI2020_EDP`` (verified: < 1e-4 relative difference at the
original heights, float32 output). Only a seeded random subset of members is
regenerated (IRI costs ~3 ms per profile).

IRI's "no value" flag (-1, e.g. below ~65 km by day and ~75 km at night) and
any negative value are set to 0, consistent with Ne(h0) = 0.

Requires IRI2020_PATH (``source Driver_Scripts/init_iri2020_env.sh``).

Usage
-----
    python Spline_Basis/extend_iri_profiles.py OUT_DIR FILE [FILE ...] \
        --h0 60 --h1 900 --dh 10 --members 200 --seed 0 --workers 8

    # non-uniform heights: 5 km from 60 to 200 km, 10 km from 200 to 900 km
    python Spline_Basis/extend_iri_profiles.py OUT_DIR FILE ... --alt-spec 60:200:5,200:900:10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "EDPSamples") not in sys.path:
    sys.path.insert(0, str(_REPO / "EDPSamples"))


def _run_chunk(args):
    dt, alt, geo, sp_values, columns = args
    from edp_samples import get_IRI2020_EDP
    edps, _ = get_IRI2020_EDP(dt, alt, geo, pd.DataFrame(sp_values, columns=columns))
    return np.asarray(edps, dtype=np.float32)


def parse_alt_spec(spec: str) -> np.ndarray:
    """'60:200:5,200:900:10' -> 60, 65, ..., 200, 210, ..., 900 (shared end points once)."""
    pts = []
    for seg in spec.split(","):
        a, b, d = (float(v) for v in seg.split(":"))
        pts.append(np.arange(a, b + 0.5 * d, d))
    alt = np.unique(np.round(np.concatenate(pts), 6))
    return alt


def output_name(src: str | Path, suffix: str = "") -> str:
    """'<run>/<hour>/edp_samples.nc' -> '<run>__<hour><suffix>.nc'."""
    p = Path(src)
    return f"{p.parent.parent.name}__{p.parent.name}{suffix}.nc"


def extend_file(src: str | Path, out_dir: str | Path, altitude: np.ndarray, members: int,
                seed: int, workers: int, chunk: int = 25, overwrite: bool = False,
                suffix: str = "") -> Path:
    out = Path(out_dir) / output_name(src, suffix)
    if out.exists() and not overwrite:
        return out
    with xr.open_dataset(src) as ds:
        geo = ds["geolocation"].values
        sp = ds["sampling_parameters"].values
        columns = [str(c) for c in ds["param"].values]
        dt = ds.attrs["DateTime"]
        attrs = {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float))}
        geo_dims = ds["geolocation"].dims
    n_all = sp.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_all, min(members, n_all), replace=False))
    chunks = [idx[i:i + chunk] for i in range(0, idx.size, chunk)]
    tasks = [(dt, altitude, geo, sp[c], columns) for c in chunks]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_run_chunk, tasks))
    else:
        parts = [_run_chunk(t) for t in tasks]
    edps = np.concatenate(parts, axis=2)
    edps = np.where(edps < 0, 0.0, edps).astype(np.float32)
    ds_out = xr.Dataset(
        {"EDPs": (("height", "geo_vertex", "sample"), edps),
         "geolocation": (geo_dims, geo),
         "sampling_parameters": (("sample", "param"), sp[idx]),
         "member_index": (("sample",), idx)},
        coords={"altitude": ("height", altitude), "param": columns},
        attrs={**attrs, "source_file": str(src), "member_seed": seed, "n_members_source": n_all,
               "negative_or_flag_values_set_to": 0.0,
               "generated_by": "Spline_Basis/extend_iri_profiles.py"})
    out.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out)
    print(f"{out.name}: {edps.shape} in {time.time() - t0:.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--h0", type=float, default=60.0)
    ap.add_argument("--h1", type=float, default=900.0)
    ap.add_argument("--dh", type=float, default=10.0)
    ap.add_argument("--alt-spec", default=None,
                    help="comma-separated start:stop:step segments (km), overrides --h0/--h1/--dh")
    ap.add_argument("--suffix", default="", help="appended to output file names")
    ap.add_argument("--members", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not os.environ.get("IRI2020_PATH"):
        sys.exit("IRI2020_PATH is not set: source Driver_Scripts/init_iri2020_env.sh first.")
    alt = parse_alt_spec(a.alt_spec) if a.alt_spec else np.arange(a.h0, a.h1 + 0.5 * a.dh, a.dh)
    for f in a.files:
        extend_file(f, a.out_dir, alt, a.members, a.seed, a.workers, overwrite=a.overwrite,
                    suffix=a.suffix)


if __name__ == "__main__":
    main()
