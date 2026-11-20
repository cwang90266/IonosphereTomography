#!/usr/bin/env python3
"""Tabulates NmF2/hmF2 (F2-layer peak density/height) for every complete
one-hour x condition x style run found under a local results tree, from
already-saved run artifacts -- no EnKF/pipeline rerun (2026-11-20 user
request: a single hour is not statistically meaningful on its own, so
comparing NmF2/hmF2 across many runs is inherently a post-processing
task). One row per (run, hour, condition, style, batch), with both the
run's identifying/configuration parameters and its forecast/analysis
density (decoded at the ISR station, same as ``isr_comparison.py``) and,
where available, the real ISR and ionosonde NmF2/hmF2 for the same window.

Reuses, rather than reimplements:
  - ``Assimilation_Cycle.isr_comparison._build_batch_isr_matches_from_saved``
    for forecast/analysis ensemble reconstruction + ISR/ionosonde window
    matching (no pipeline rerun -- see that module's docstring).
  - ``Run_Scoring.score_runs.retrieve_peak`` for the NmF2/hmF2 retrieval
    itself (3-point log10 parabola refinement) -- the same function
    ``score_runs.py`` already uses for the model's and ISR's own peaks.
  - The same tree-walking shape as
    ``isr_comparison.replot_isr_comparison_tree``: ``rglob`` for every
    ``*_edp_samples.nc`` under ``--results-root`` (works whether that root
    holds one run tree or many side by side), then per hour folder, a
    style only counts if it has both a ``*_parameterized.nc`` and a
    non-empty ``*_ensembles/ensemble_batch_*.nc`` -- same gate
    ``Run_Scoring/extract_run_diagnostics.py`` uses to skip incomplete
    runs rather than raising.

Station matching is by each hour's own ``*_edp_samples.nc`` ``Lat``/``Lon``
attrs against each provided ISR/ionosonde file's own station attrs (same
pattern as
``Postprocessing/regenerate_all_local_mil_isr_comparison_plots.py``'s
``_edp_center``) -- not by parsing the run directory name's site prefix.
A run with no matching ISR station still gets a row (forecast/analysis
NmF2/hmF2 at the run's own ROI center), with ISR/ionosonde columns blank.

Configuration columns come from three independent, non-exclusive sources
(each column filled if its source has it, blank otherwise): the hour's
own ``*_edp_samples.nc`` attrs (``center_lat``/``center_lon``/
``roi_radius_km`` -- always present, most authoritative); the run tree
root's own ``each_run_setting.csv`` (written once per multi-hour run by
``run_hourly_all4styles_boost_compare.py``, keyed by the run directory's
own name); and, as a last-resort fallback, parsing known tokens out of the
run directory's own name (real run directories on disk use more than one
naming scheme -- see ``_parse_run_label_tokens``'s docstring). Anything
left over in the name that isn't a recognized token is kept verbatim in
``extra_tags`` rather than silently dropped.

Usage::

    python3 Postprocessing/tabulate_nmf2_hmf2.py \\
        --results-root /local/path/to/tomography_output \\
        --isr MIL=/local/path/MILISR2025_nonan.nc \\
        --isr TRO=/local/path/TROISR2025_nonan.nc \\
        --ionosonde MIL=/local/path/MILIONO_Aug2530_2025.nc \\
        --out /local/path/nmf2_hmf2_table.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs",
           REPO / "Spline_Basis", REPO / "Run_Scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import xarray as xr

from Assimilation_Cycle.isr_comparison import (
    IonosondeDataset, IsrDataset, IsrStation,
    _build_batch_isr_matches_from_saved, _interp_log10,
    load_ionosonde_dataset, load_isr_dataset, nearest_ionosonde_point,
)
from score_runs import retrieve_peak

HOUR_DIR_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})(\d{2})$")
KNOWN_CONDITIONS = {"no_boost", "boosted"}


def parse_hour_dir(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    m = HOUR_DIR_PATTERN.match(name)
    if m is None:
        return None
    day, h0, m0, h1, m1 = m.groups()
    start = pd.Timestamp(f"{day} {h0}:{m0}")
    end = pd.Timestamp(f"{day} {h1}:{m1}")
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end


def _edp_attrs(path: Path) -> dict:
    """``(center_lat, center_lon, grid_radius_deg)`` from a
    `*_edp_samples.nc`'s own ``Lat``/``Lon``/``radius`` attrs (same attrs
    used throughout this session's other local-disk tools) -- note
    ``radius`` here is the *grid* extent in degrees, a different quantity
    from the physical ROI radius in km used for RO observation selection
    (``roi_radius_km``, only available from `each_run_setting.csv` or the
    run label's own ``R<n>k`` token -- see `parse_run_label_tokens`).
    Missing attrs come back ``None``."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        return dict(
            center_lat=float(ds.attrs["Lat"]) if "Lat" in ds.attrs else None,
            center_lon=float(ds.attrs["Lon"]) if "Lon" in ds.attrs else None,
            grid_radius_deg=float(ds.attrs["radius"]) if "radius" in ds.attrs else None,
        )


def _empty_isr_dataset(lat: float, lon: float) -> IsrDataset:
    """A station-less stand-in with zero profiles -- `nearest_isr_profile`
    (called internally by `_build_batch_isr_matches_from_saved`) already
    handles zero profiles by returning `None`, so this lets every run get
    a row (forecast/analysis decoded at `(lat, lon)`) even when no real
    ISR station was provided/matched for it."""
    return IsrDataset(
        station=IsrStation(name="none", lat=lat, lon=lon),
        time=pd.DatetimeIndex([]), altitude=np.array([]), density=np.empty((0, 0)),
    )


def _decode_boost_tag(tag: str) -> float | None:
    """Inverse of `_boost_tag` (run_hourly_all4styles_boost_compare.py,
    `origin/server` branch): 'Op3' -> 0.3 (letter-O prefix marks the
    '0.'-leading case for this token specifically); otherwise 'p'->'.',
    'm'->'-' (e.g. '1p2' -> 1.2)."""
    try:
        if tag.startswith("Op"):
            return float("0." + tag[2:])
        return float(tag.replace("p", ".").replace("m", "-"))
    except ValueError:
        return None


def _decode_compact_number_tag(tag: str) -> float | None:
    """Inverse of `_compact_number_tag` (same branch/file): '0p65' -> 0.65
    (digit-zero prefix, NOT the letter 'O' `_boost_tag` uses -- the two
    tokens are genuinely different encodings for the same '< 1' case);
    otherwise 'p'->'.', 'm'->'-'."""
    try:
        if tag.startswith("0p"):
            return float("0." + tag[2:])
        return float(tag.replace("p", ".").replace("m", "-"))
    except ValueError:
        return None


_ELABORATE_SCHEME_RE = re.compile(
    r"R(?P<roi>\d+p?\d*)k_B(?P<boost>Op\d+|\d+p?\d*)_Obs(?P<obs>\d+p?\d*)_"
    r"IRIScale(?P<iri>\d+p?\d*)_LOSfrac(?P<los>0p\d+|\d+p?\d*)_PCAdisScale(?P<pca>\d+p?\d*)"
)
_SIMPLE_SCHEME_RE = re.compile(r"ROI(?P<roi_km>\d+(?:\.\d+)?)kmRadius_(?P<boost>Op\d+|\d+p?\d*)Boost")
_KNOWN_TOKEN_SPANS_RE = re.compile(
    r"R\d+p?\d*k|B(?:Op\d+|\d+p?\d*)|Obs\d+p?\d*|IRIScale\d+p?\d*|LOSfrac(?:0p\d+|\d+p?\d*)|"
    r"PCAdisScale\d+p?\d*|ROI\d+(?:\.\d+)?kmRadius|(?:Op\d+|\d+p?\d*)Boost"
)


def parse_run_label_tokens(label: str) -> dict:
    """Best-effort config extraction from a run directory's own name --
    fallback only, used when `each_run_setting.csv` doesn't have (or
    doesn't cover) this run. Real run directories on disk use more than
    one naming scheme (confirmed this session): an elaborate one
    (`..._R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_...`, documented
    on the `server`/`develop` branches' `run_hourly_all4styles_boost_compare.py`)
    and a simpler one (`..._ROI2000kmRadius_Op3Boost...`, this branch's
    own). Each of the two is tried independently via `re.search` (not a
    strict split), so a label matching neither just leaves those columns
    blank rather than raising. Anything in the label not covered by either
    scheme's matched spans is kept verbatim in `extra_tags`."""
    out = dict(
        roi_radius_km=None, boost_amplitude=None, obs_sigma_tecu=None,
        iri_downsampling_factor=None, ro_los_fraction_required=None,
        pca_discard_scale=None, extra_tags="",
    )
    m = _ELABORATE_SCHEME_RE.search(label)
    if m:
        out["roi_radius_km"] = _decode_compact_number_tag(m["roi"]) * 1000.0 if m["roi"] else None
        out["boost_amplitude"] = _decode_boost_tag(m["boost"])
        out["obs_sigma_tecu"] = _decode_compact_number_tag(m["obs"])
        out["iri_downsampling_factor"] = _decode_compact_number_tag(m["iri"])
        out["ro_los_fraction_required"] = _decode_compact_number_tag(m["los"])
        out["pca_discard_scale"] = _decode_compact_number_tag(m["pca"])
    else:
        m2 = _SIMPLE_SCHEME_RE.search(label)
        if m2:
            out["roi_radius_km"] = float(m2["roi_km"])
            out["boost_amplitude"] = _decode_boost_tag(m2["boost"])

    remainder = _KNOWN_TOKEN_SPANS_RE.sub("", label)
    leftover = [t for t in remainder.split("_") if t and not t.isdigit()]
    out["extra_tags"] = ";".join(leftover)
    return out


def load_each_run_setting(results_root: Path) -> dict[str, dict]:
    """``{label_of_each_run: row_dict}`` from ``results_root/each_run_setting.csv``
    (`run_hourly_all4styles_boost_compare.py`'s own schema), or ``{}`` if
    that file doesn't exist here -- the fallback tiers (Section 4 of the
    plan) still apply per-column in that case."""
    path = results_root / "each_run_setting.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if "label_of_each_run" not in df.columns:
        return {}
    return {str(row["label_of_each_run"]): row.to_dict() for _, row in df.iterrows()}


def config_from_run_setting(row: dict | None) -> dict:
    if row is None:
        return dict(grid_radius_deg=None, roi_radius_km=None, n_ensemble=None,
                    obs_sigma_tecu=None, boost_amplitude=None, ro_los_fraction_required=None)
    def _get(col):
        v = row.get(col)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return dict(
        grid_radius_deg=_get("grid_radius_deg"), roi_radius_km=_get("roi_radius_km"),
        n_ensemble=_get("n_ensemble"), obs_sigma_tecu=_get("obs_sigma_tecu"),
        boost_amplitude=_get("boost_ge_700_km"), ro_los_fraction_required=_get("fraction_required"),
    )


def _retrieve_nmf2_hmf2_triplet(
    altitude: np.ndarray, forecast_mean: np.ndarray, analysis_mean: np.ndarray,
    isr_density: np.ndarray | None, isr_altitude: np.ndarray | None,
    e_top_km: float, f_top_km: float,
) -> dict:
    """Forecast/analysis NmF2/hmF2 are retrieved on the ISR's own native
    gates (log10-interpolated there first) when an ISR profile matched
    this batch, exactly mirroring `score_runs.py`'s own convention so the
    three are retrieved on the same altitude samples; otherwise retrieved
    directly on the model's own altitude grid."""
    if isr_density is not None and isr_altitude is not None and len(isr_altitude):
        f_on_isr = _interp_log10(isr_altitude, altitude, forecast_mean)
        a_on_isr = _interp_log10(isr_altitude, altitude, analysis_mean)
        NmF2_f, hmF2_f = retrieve_peak(isr_altitude, f_on_isr, e_top_km, f_top_km)
        NmF2_a, hmF2_a = retrieve_peak(isr_altitude, a_on_isr, e_top_km, f_top_km)
        NmF2_i, hmF2_i = retrieve_peak(isr_altitude, isr_density, e_top_km, f_top_km)
    else:
        NmF2_f, hmF2_f = retrieve_peak(altitude, forecast_mean, e_top_km, f_top_km)
        NmF2_a, hmF2_a = retrieve_peak(altitude, analysis_mean, e_top_km, f_top_km)
        NmF2_i, hmF2_i = np.nan, np.nan
    return dict(NmF2_forecast=NmF2_f, hmF2_forecast=hmF2_f,
                NmF2_analysis=NmF2_a, hmF2_analysis=hmF2_a,
                NmF2_isr=NmF2_i, hmF2_isr=hmF2_i)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--isr", action="append", default=[], metavar="CODE=PATH")
    parser.add_argument("--ionosonde", action="append", default=[], metavar="CODE=PATH")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--e-top-km", type=float, default=150.0)
    parser.add_argument("--f-top-km", type=float, default=700.0)
    parser.add_argument("--center-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--hours", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--styles", nargs="*", default=None)
    args = parser.parse_args()

    isr_by_code = {code: load_isr_dataset(path) for code, path in
                   (item.split("=", 1) for item in args.isr)}
    iono_by_code = {code: load_ionosonde_dataset(path) for code, path in
                    (item.split("=", 1) for item in args.ionosonde)}
    each_run_setting = load_each_run_setting(args.results_root)

    edp_paths = sorted(args.results_root.rglob("*_edp_samples.nc"))
    print(f"[tabulate] found {len(edp_paths)} hour/condition folders under {args.results_root}", flush=True)

    rows: list[dict] = []
    for i, edp_path in enumerate(edp_paths, start=1):
        run_dir = edp_path.parent
        label = edp_path.name[: -len("_edp_samples.nc")]

        if run_dir.name in KNOWN_CONDITIONS:
            condition, hour_dir = run_dir.name, run_dir.parent
        else:
            condition, hour_dir = "", run_dir

        if args.conditions is not None and condition not in args.conditions:
            continue
        if args.hours is not None and hour_dir.name not in args.hours:
            continue

        # The run tree directory is always the hour folder's own parent --
        # true whether --results-root is that single tree itself or a
        # common parent holding many trees side by side.
        run_label = hour_dir.parent.name

        window = parse_hour_dir(hour_dir.name)
        window_start = window[0] if window else None
        window_end = window[1] if window else None

        edp_attrs = _edp_attrs(edp_path)
        center_lat, center_lon = edp_attrs["center_lat"], edp_attrs["center_lon"]

        matched_isr_code, matched_isr = None, None
        if center_lat is not None:
            for code, ds in isr_by_code.items():
                if (abs(ds.station.lat - center_lat) <= args.center_tolerance_deg
                        and abs(ds.station.lon - center_lon) <= args.center_tolerance_deg):
                    matched_isr_code, matched_isr = code, ds
                    break
        matched_iono_code, matched_iono = None, None
        if center_lat is not None:
            for code, ds in iono_by_code.items():
                if (abs(ds.station.lat - center_lat) <= args.center_tolerance_deg
                        and abs(ds.station.lon - center_lon) <= args.center_tolerance_deg):
                    matched_iono_code, matched_iono = code, ds
                    break
        station_code = matched_isr_code or matched_iono_code or ""

        if matched_isr is not None:
            query_lat, query_lon = matched_isr.station.lat, matched_isr.station.lon
            isr_ds = matched_isr
        elif center_lat is not None:
            query_lat, query_lon = center_lat, center_lon
            isr_ds = _empty_isr_dataset(center_lat, center_lon)
        else:
            print(f"[tabulate]   skip {edp_path} (no center_lat/lon attrs, no ISR match possible)", flush=True)
            continue

        run_setting_row = each_run_setting.get(run_label)
        run_setting_cfg = config_from_run_setting(run_setting_row)
        fallback_cfg = parse_run_label_tokens(run_label)

        def _pick(primary, fallback):
            return primary if primary is not None else fallback

        config_cols = dict(
            center_lat=center_lat, center_lon=center_lon,
            # roi_radius_km (RO-selection radius, km) and grid_radius_deg
            # (EDP grid extent, degrees) are different quantities -- see
            # _edp_attrs's docstring. each_run_setting.csv's own
            # roi_radius_km is preferred over the run label's rounded
            # "R2k"-style token when both are available.
            roi_radius_km=_pick(run_setting_cfg["roi_radius_km"], fallback_cfg["roi_radius_km"]),
            grid_radius_deg=_pick(edp_attrs["grid_radius_deg"], run_setting_cfg["grid_radius_deg"]),
            n_ensemble=run_setting_cfg["n_ensemble"],
            obs_sigma_tecu=_pick(run_setting_cfg["obs_sigma_tecu"], fallback_cfg["obs_sigma_tecu"]),
            boost_amplitude=_pick(run_setting_cfg["boost_amplitude"], fallback_cfg["boost_amplitude"]),
            iri_downsampling_factor=fallback_cfg["iri_downsampling_factor"],
            ro_los_fraction_required=_pick(run_setting_cfg["ro_los_fraction_required"],
                                            fallback_cfg["ro_los_fraction_required"]),
            pca_discard_scale=fallback_cfg["pca_discard_scale"],
            extra_tags=fallback_cfg["extra_tags"],
        )

        for parameterized_path in sorted(run_dir.glob(f"{label}_*_parameterized.nc")):
            style = parameterized_path.name[len(label) + 1: -len("_parameterized.nc")]
            if args.styles is not None and style not in args.styles:
                continue
            ensembles_dir = run_dir / f"{label}_{style}_ensembles"
            if not list(ensembles_dir.glob("ensemble_batch_*.nc")):
                print(f"[tabulate]   skip {run_dir}/{label}_{style} (no saved ensemble batches)", flush=True)
                continue

            try:
                recon = _build_batch_isr_matches_from_saved(
                    run_dir / f"{label}_{style}", isr_ds, query_lat, query_lon, ionosonde=matched_iono,
                )
            except FileNotFoundError as exc:
                print(f"[tabulate]   skip {run_dir}/{label}_{style}: {exc}", flush=True)
                continue

            for match in recon.matches:
                forecast_mean = np.nanmean(match.forecast_profile, axis=-1)
                analysis_mean = np.nanmean(match.analysis_profile, axis=-1)
                peaks = _retrieve_nmf2_hmf2_triplet(
                    recon.model_altitude, forecast_mean, analysis_mean,
                    match.isr_density, isr_ds.altitude if matched_isr is not None else None,
                    args.e_top_km, args.f_top_km,
                )

                NmF2_iono = hmF2_iono = np.nan
                iono_time_delta_min = np.nan
                if match.isr_time is not None:
                    point = nearest_ionosonde_point(matched_iono, match.isr_time)
                    if point is not None:
                        iono_time, NmF2_iono, hmF2_iono = point
                        iono_time_delta_min = abs((iono_time - match.isr_time).total_seconds()) / 60.0
                elif match.ionosonde_nmF2 is not None and len(match.ionosonde_nmF2):
                    NmF2_iono = float(np.nanmean(match.ionosonde_nmF2))
                    hmF2_iono = float(np.nanmean(match.ionosonde_hmF2))

                rows.append(dict(
                    run_label=run_label, run_dir=str(run_dir), hour_label=hour_dir.name,
                    window_start_utc=window_start, window_end_utc=window_end,
                    batch_index=match.batch_index, condition=condition, style=style,
                    station_code=station_code,
                    **config_cols,
                    **peaks,
                    NmF2_ionosonde=NmF2_iono, hmF2_ionosonde=hmF2_iono,
                    isr_time_delta_min=(
                        np.nan if match.isr_time is None else
                        abs((match.isr_time - (window_start + (window_end - window_start) / 2)).total_seconds()) / 60.0
                        if window_start is not None else np.nan
                    ),
                    ionosonde_time_delta_min=iono_time_delta_min,
                ))

        if i % 25 == 0 or i == len(edp_paths):
            print(f"[tabulate] ({i}/{len(edp_paths)}) processed, {len(rows)} rows so far", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
