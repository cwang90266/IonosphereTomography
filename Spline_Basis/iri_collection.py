# -*- coding: utf-8 -*-
"""
Build/cache the IRI2020 training collection for knot optimization, via the
*exact* ensemble-generation code the main assimilation package uses for
real runs (``Assimilation_Cycle.ensemble_init.load_or_build_edp_samples``)
-- not a second, parallel implementation of "how to build an EDPSamples
ensemble" (plan Documentation/Spline_Basis_Knot_Optimization_Plan.md §4,
named but not built until now; this closes that gap alongside
``Driver_Scripts/optimize_vertical_grid.py``).

Each condition (one training/validation hour) becomes one full-resolution
``edp_samples.nc`` (``n_ensemble`` members, full horizontal grid, no
subsampling -- subsampling for the optimizer itself, if any, is a
``KnotOptimizationConfig`` concern, applied later when loading these files
for fitting, not here).

Deferred imports: ``Assimilation_Cycle`` (and transitively ``edp_samples``/
``IRI_Sample_inputs``) are only importable once their directories are on
``sys.path`` -- same convention this whole repo already uses elsewhere.
Note this module is *never* imported by the production runtime path
(``Parameterization.py``'s ``SPLINE`` style only ever imports
``spline_basis``/``spline_fit``/``knot_optimization``, not this file), so
this module importing ``Assimilation_Cycle`` back does not create a real
circular import -- the two import graphs never meet in the same call
chain.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["IriCollectionCondition", "conditions_from_run_hours", "build_collection"]

_HOUR_FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})(\d{2})$")


@dataclass(frozen=True)
class IriCollectionCondition:
    """One training/validation hour: a real IRI2020 ensemble draw at a
    given time and region, built through the main package's own code."""

    label: str              # e.g. "MIL_2025-08-26_0000" -- used as the output file's basename
    time: str                # ISO datetime, the cycle's center time
    center_lat: float
    center_lon: float
    grid_radius_deg: float


def conditions_from_run_hours(
    run_dir: str | Path, center_lat: float, center_lon: float, grid_radius_deg: float,
    label_prefix: str,
) -> list[IriCollectionCondition]:
    """Parses ``YYYY-MM-DD_HHMMHHMM`` subfolder names directly under
    ``run_dir`` (``package_run.py``'s per-hour output convention) into one
    condition per hour -- the start hour, minute fixed at ``:00`` -- so the
    exact hour list used by an existing real run (e.g. "all 34 MIL hourly
    folders") can be replicated for a fresh, extended-altitude collection
    without hand-transcribing it. Raises if no matching subfolder is found
    (likely a wrong ``run_dir`), so a typo fails loudly rather than
    silently producing an empty collection."""
    run_dir = Path(run_dir)
    conditions = []
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = _HOUR_FOLDER_RE.match(entry.name)
        if not m:
            continue
        date_str, start_hour = m.group(1), m.group(2)
        conditions.append(IriCollectionCondition(
            label=f"{label_prefix}_{date_str}_{start_hour}00",
            time=f"{date_str}T{start_hour}:00:00",
            center_lat=center_lat, center_lon=center_lon, grid_radius_deg=grid_radius_deg,
        ))
    if not conditions:
        raise ValueError(
            f"conditions_from_run_hours: no 'YYYY-MM-DD_HHMMHHMM' subfolders found under {run_dir} "
            "-- check the path."
        )
    return conditions


def build_collection(
    conditions: list[IriCollectionCondition], altitude_grid: np.ndarray, output_dir: str | Path,
    *, n_ensemble: int = 2000, horizontal_resolution_deg: float = 2.5,
    iri_spread_kwargs: dict | None = None, iri_rng_seed: int | None = None,
    skip_existing: bool = True, verbose: bool = True,
) -> list[Path]:
    """Builds one full-resolution ``edp_samples.nc`` per condition, via
    ``Assimilation_Cycle.ensemble_init.load_or_build_edp_samples`` -- the
    same call a real cycle makes, so this collection's statistics (driving
    -index spread, horizontal grid construction, the NaN-replacement guard)
    can never silently diverge from production. Returns the list of
    written (or already-cached) paths, same order as ``conditions``.

    ``skip_existing=True`` (default) makes this resumable: a condition
    whose output file already exists is left alone and just returned,
    without rerunning IRI2020 -- real IRI2020 at the full ensemble size is
    the expensive part of this whole collection-building step."""
    from Assimilation_Cycle.cycle_config import CycleConfig, default_iri_spread_kwargs
    from Assimilation_Cycle.ensemble_init import load_or_build_edp_samples

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spread = iri_spread_kwargs if iri_spread_kwargs is not None else default_iri_spread_kwargs()

    paths: list[Path] = []
    for i, cond in enumerate(conditions, start=1):
        out_path = output_dir / f"{cond.label}_edp_samples.nc"
        if skip_existing and out_path.exists():
            if verbose:
                print(f"[iri_collection] ({i}/{len(conditions)}) {cond.label}: cached, skipping", flush=True)
            paths.append(out_path)
            continue
        if verbose:
            print(f"[iri_collection] ({i}/{len(conditions)}) {cond.label}: building "
                  f"(n_ensemble={n_ensemble}, lat={cond.center_lat}, lon={cond.center_lon}) ...", flush=True)

        cfg = CycleConfig(
            start_time=cond.time, end_time=cond.time,
            center_lat=cond.center_lat, center_lon=cond.center_lon,
            radius_km=cond.grid_radius_deg,
            altitude_grid=altitude_grid, horizontal_resolution_deg=horizontal_resolution_deg,
            grid_radius_deg=cond.grid_radius_deg,
            n_ensemble=n_ensemble, iri_rng_seed=iri_rng_seed, iri_spread_kwargs=spread,
            edp_samples_output_path=out_path, output_dir=output_dir / cond.label,
        )
        try:
            load_or_build_edp_samples(cfg)
        except Exception:
            warnings.warn(f"iri_collection.build_collection: failed on condition {cond.label!r} "
                          f"(time={cond.time}) -- see the raised traceback.", stacklevel=2)
            raise
        paths.append(out_path)

    return paths
