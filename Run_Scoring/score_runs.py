#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score hourly assimilation runs from the ``<hour>.npz`` files written by
``extract_run_diagnostics.py``, against ISR profiles.

Writes to ``--out-dir``:

  hourly_metrics.csv        one row per hour x run x station (all metrics below)
  summary_tec_fit.csv       TEC misfit (TECU RMS) by ray class, per run
  summary_isr_relative.csv  E/F relative error vs ISR, forecast and analysis, per run x station
  summary_peaks.csv         NmF2/hmF2 and NmE/hmE errors + E-peak detection counts
  summary_joint_success.csv TEC-fit x ISR-agreement classification
  summary_predictors.csv    Spearman rho of skill against RO count, overlap, spread reduction

Metric definitions
------------------
TEC fit (per run-hour): RMS of (ensemble-mean model TEC - observed TEC) over
  all rays and by tangent-altitude class: E rays (< --e-top-km), F bottomside
  rays (E top .. --f-split-km) and topside rays (>= --f-split-km). ``Eshare``
  = median fraction of model TEC from grid levels below the E top, on E rays.

ISR matching: the ISR profile nearest the hour midpoint, and only if it lies
  inside the hour (the rule of run_hourly_all4styles_boost_compare.py). Model
  profiles are interpolated in log10 Ne to the native ISR gates, no
  extrapolation.

Relative error: mean over ISR gates of |model - ISR| / ISR, separately for
  the E region [--e-bottom-km, --e-top-km) and the F region
  [--e-top-km, --f-top-km).

Peak parameters (one retrieval for ISR, forecast and analysis, all on the
  ISR gates): the gate of maximum Ne in the layer's search range, refined by a
  3-point parabola in log10 Ne. A maximum on the first or last gate of the
  range means "no peak". NmF2/hmF2 search >= E top; NmE/hmE search the E
  region and additionally require NmE >= --e-valley-ratio x the minimum Ne
  between the peak and --e-valley-top-km (a real valley above the E peak).

Joint success: TEC RMS <= --tec-ok-sigma x obs sigma, AND F relative error
  below the forecast's, AND E relative error not above the forecast's by
  more than --e-tolerance (fraction).

Overlap: sum over rays and region levels of H_grid @ G^T (G = grid -> ISR
  column interpolation), scaled to TECU per 1e11 m^-3; the F-region sum takes
  every model level above the E top. Spread reduction: 1 - mean over region
  levels (capped at the top ISR gate) of (analysis / forecast std of log10 Ne).

Example::

    python Run_Scoring/score_runs.py --npz-dir /path/to/scoring/extract \\
        --out-dir /path/to/scoring/results \\
        --isr TRO=/path/TROISR2025_nonan.nc --isr SVL=/path/SVLISR2025_nonan.nc
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_isr(path: str):
    """(altitude, Ne[alt, time], time) sorted by altitude and time."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        alt = np.asarray(ds["altitude"].values, dtype=float)
        ne = np.asarray(ds["Ne"].values, dtype=float)
        t = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds["time_utc"].values)))
    ao, to = np.argsort(alt), np.argsort(np.asarray(t))
    return alt[ao], ne[ao][:, to], t[to]


def nearest_in_window(times: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int | None:
    mid = start + (end - start) / 2
    j = int(np.argmin(np.abs(times - mid)))
    return j if start <= times[j] <= end else None


def retrieve_peak(alt, ne, lo, hi, *, valley_top=None, valley_ratio=None):
    """(Nm, hm) of the layer peak in [lo, hi] on the given gates, or (nan, nan)."""
    ok = (alt >= lo) & (alt <= hi) & np.isfinite(ne) & (ne > 0)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return np.nan, np.nan
    k = idx[np.argmax(ne[idx])]
    if k == idx[0] or k == idx[-1]:
        return np.nan, np.nan
    if valley_top is not None:
        above = np.where((alt > alt[k]) & (alt <= valley_top) & np.isfinite(ne))[0]
        if len(above) == 0 or ne[k] < valley_ratio * np.nanmin(ne[above]):
            return np.nan, np.nan
    x, yv = alt[k - 1:k + 2], np.log10(ne[k - 1:k + 2])
    c = np.polyfit(x, yv, 2)
    if c[0] >= 0:
        return ne[k], alt[k]
    h = float(np.clip(-c[1] / (2 * c[0]), x[0], x[2]))
    return 10.0 ** np.polyval(c, h), h


def hourly_metrics(npz_path: Path, isr: dict, a) -> list[dict]:
    z = np.load(npz_path, allow_pickle=False)
    hour = npz_path.stem
    if "window_start" in z:
        start, end = pd.Timestamp(str(z["window_start"])), pd.Timestamp(str(z["window_end"]))
    else:  # npz written before the window was stored: take it from the hour-folder name
        start = pd.Timestamp(f"{hour[:10]} {hour[11:13]}:{hour[13:15]}")
        end = start + pd.Timedelta(hours=1)
    runs = z["runs"] if "runs" in z else sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|Yf")})
    y, tang, alt = z["y"], z["tang"], z["alt"]
    n_ro = int(z["occ"].max() + 1)
    ray_classes = {
        "all": np.ones_like(tang, dtype=bool),
        "Eray": tang < a.e_top_km,
        "Fbot": (tang >= a.e_top_km) & (tang < a.f_split_km),
        "Ftop": tang >= a.f_split_km,
    }
    regions = {"E": (a.e_bottom_km, a.e_top_km), "F": (a.e_top_km, a.f_top_km)}

    rows = []
    for run in runs:
        cond, style = str(run).split("|")
        base = dict(hour=hour, window_start=start, condition=cond, style=style,
                    n_ro=n_ro, n_rays=len(y))
        for tag in "fa":
            resid = z[f"{run}|Y{tag}"] - y
            for name, m in ray_classes.items():
                base[f"tec_rms_{tag}_{name}"] = float(np.sqrt(np.mean(resid[m] ** 2))) if m.any() else np.nan
            share = z[f"{run}|YE{tag}"] / z[f"{run}|Y{tag}"]
            base[f"Eshare_{tag}"] = float(np.median(share[ray_classes["Eray"]])) if ray_classes["Eray"].any() else np.nan
        for name, m in ray_classes.items():
            base[f"n_rays_{name}"] = int(m.sum())

        for code, (ia, ine, it) in isr.items():
            if f"HG_{code}" not in z:
                continue
            r = dict(base, station=code)
            hg = z[f"HG_{code}"]
            r["n_rays_crossing_column"] = int((hg.sum(axis=1) > 0).sum())
            for reg, (lo, hi) in regions.items():
                lev = (alt >= lo) & (alt < hi)
                # F-region overlap counts the whole column above the E top:
                # topside Ne above f_top still contributes to the rays' TEC.
                lev_ov = lev if reg == "E" else alt >= lo
                r[f"overlap_{reg}"] = float(hg[:, lev_ov].sum() * 1e11)
                lev_isr = lev & (alt <= ia.max())
                r[f"spread_reduction_{reg}"] = float(
                    1 - np.mean(z[f"{run}|{code}|alogstd"][lev_isr] / z[f"{run}|{code}|flogstd"][lev_isr])
                )
            j = nearest_in_window(it, start, end)
            if j is not None:
                prof_isr = ine[:, j]
                r["isr_time"] = it[j]
                profs = {"i": prof_isr}
                for tag in "fa":
                    log_m = np.log10(np.clip(z[f"{run}|{code}|{tag}mean"], 1.0, None))
                    profs[tag] = 10.0 ** np.interp(ia, alt, log_m, left=np.nan, right=np.nan)
                    rel = np.abs(profs[tag] - prof_isr) / prof_isr
                    for reg, (lo, hi) in regions.items():
                        m = (ia >= lo) & (ia < hi) & np.isfinite(rel)
                        r[f"rel_{tag}_{reg}"] = float(np.mean(rel[m])) if m.any() else np.nan
                for tag, p in profs.items():
                    r[f"NmF2_{tag}"], r[f"hmF2_{tag}"] = retrieve_peak(ia, p, a.e_top_km, a.f_top_km)
                    r[f"NmE_{tag}"], r[f"hmE_{tag}"] = retrieve_peak(
                        ia, p, a.e_bottom_km, a.e_top_km,
                        valley_top=a.e_valley_top_km, valley_ratio=a.e_valley_ratio,
                    )
            rows.append(r)
    return rows


def _improved(df, a_col, f_col):
    return (df[a_col] < df[f_col]).where(df[a_col].notna() & df[f_col].notna())


def summaries(d: pd.DataFrame, a) -> dict[str, pd.DataFrame]:
    keys = ["style", "condition"]
    out = {}

    # TEC fit is a property of the run-hour, not the station.
    runs = d.drop_duplicates(["hour", "condition", "style"])
    tec_ok = a.tec_ok_sigma * a.obs_sigma
    g = runs.groupby(keys)
    tec = g[[c for c in runs.columns if c.startswith("tec_rms_")] + ["Eshare_f", "Eshare_a"]].median()
    tec["n_hours"] = g.size()
    tec[f"hours_within_{tec_ok:g}_TECU"] = g["tec_rms_a_all"].apply(lambda x: int((x <= tec_ok).sum()))
    tec[f"hours_below_{a.overfit_sigma * a.obs_sigma:g}_TECU"] = g["tec_rms_a_all"].apply(
        lambda x: int((x < a.overfit_sigma * a.obs_sigma).sum()))
    out["summary_tec_fit"] = tec.reset_index()

    s = d.dropna(subset=["rel_a_F"]).copy()
    gs = s.groupby(["station"] + keys)
    rel = gs[["rel_f_E", "rel_a_E", "rel_f_F", "rel_a_F"]].median() * 100
    rel.columns = [c + "_pct" for c in rel.columns]
    rel["n_hours"] = gs.size()
    for reg in "EF":
        s[f"imp_{reg}"] = _improved(s, f"rel_a_{reg}", f"rel_f_{reg}")
        rel[f"hours_improved_{reg}"] = s.groupby(["station"] + keys)[f"imp_{reg}"].sum().astype(int)
    out["summary_isr_relative"] = rel.reset_index()

    for tag in "fa":
        s[f"dNmF2_{tag}_pct"] = 100 * (s[f"NmF2_{tag}"] / s["NmF2_i"] - 1)
        s[f"dhmF2_{tag}_km"] = s[f"hmF2_{tag}"] - s["hmF2_i"]
        s[f"dNmE_{tag}_pct"] = 100 * (s[f"NmE_{tag}"] / s["NmE_i"] - 1)
        s[f"dhmE_{tag}_km"] = s[f"hmE_{tag}"] - s["hmE_i"]
    gs = s.groupby(["station"] + keys)
    pk = pd.DataFrame({
        "hours_with_isr_F2_peak": gs["NmF2_i"].count(),
        "NmF2_abs_err_f_pct": gs["dNmF2_f_pct"].apply(lambda x: x.abs().median()),
        "NmF2_abs_err_a_pct": gs["dNmF2_a_pct"].apply(lambda x: x.abs().median()),
        "NmF2_bias_a_pct": gs["dNmF2_a_pct"].median(),
        "hmF2_abs_err_f_km": gs["dhmF2_f_km"].apply(lambda x: x.abs().median()),
        "hmF2_abs_err_a_km": gs["dhmF2_a_km"].apply(lambda x: x.abs().median()),
        "hmF2_bias_a_km": gs["dhmF2_a_km"].median(),
        "hours_NmF2_improved": gs.apply(lambda x: int((x["dNmF2_a_pct"].abs() < x["dNmF2_f_pct"].abs()).sum())),
        "hours_hmF2_improved": gs.apply(lambda x: int((x["dhmF2_a_km"].abs() < x["dhmF2_f_km"].abs()).sum())),
        "hours_with_isr_E_peak": gs["NmE_i"].count(),
        "hours_with_forecast_E_peak": gs["NmE_f"].count(),
        "hours_with_analysis_E_peak": gs["NmE_a"].count(),
        "hours_E_peak_in_isr_and_analysis": gs["dNmE_a_pct"].count(),
        "NmE_abs_err_a_pct": gs["dNmE_a_pct"].apply(lambda x: x.abs().median()),
        "hmE_abs_err_a_km": gs["dhmE_a_km"].apply(lambda x: x.abs().median()),
    })
    out["summary_peaks"] = pk.reset_index()

    s["tec_ok"] = s["tec_rms_a_all"] <= tec_ok
    s["isr_ok"] = (s["rel_a_F"] < s["rel_f_F"]) & (s["rel_a_E"] <= s["rel_f_E"] + a.e_tolerance)
    gs = s.groupby(["station"] + keys)
    out["summary_joint_success"] = pd.DataFrame({
        "n_hours": gs.size(),
        "success": gs.apply(lambda x: int((x.tec_ok & x.isr_ok).sum())),
        "fits_tec_misses_isr": gs.apply(lambda x: int((x.tec_ok & ~x.isr_ok).sum())),
        "misses_tec_agrees_isr": gs.apply(lambda x: int((~x.tec_ok & x.isr_ok).sum())),
        "misses_both": gs.apply(lambda x: int((~x.tec_ok & ~x.isr_ok).sum())),
    }).reset_index()

    rows = []
    for (st, sty, cond), grp in s.groupby(["station"] + keys):
        for reg in "EF":
            gain = np.log(grp[f"rel_a_{reg}"] / grp[f"rel_f_{reg}"])
            r = dict(station=st, style=sty, condition=cond, region=reg, n_hours=int(gain.notna().sum()))
            for name, col in [("n_ro", "n_ro"), ("n_rays", "n_rays"), ("overlap", f"overlap_{reg}"),
                              ("rays_crossing_column", "n_rays_crossing_column"),
                              ("spread_reduction", f"spread_reduction_{reg}")]:
                r[f"rho_{name}"] = gain.corr(grp[col], method="spearman")
            rows.append(r)
    out["summary_predictors"] = pd.DataFrame(rows)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--isr", action="append", required=True, metavar="CODE=PATH",
                   help="ISR station file; CODE must match the station codes used at extraction")
    p.add_argument("--obs-sigma", type=float, default=3.0, help="TEC observation error (TECU)")
    p.add_argument("--tec-ok-sigma", type=float, default=1.5, help="TEC fit OK if RMS <= this x obs sigma")
    p.add_argument("--overfit-sigma", type=float, default=2 / 3, help="flag RMS below this x obs sigma")
    p.add_argument("--e-tolerance", type=float, default=0.10, help="allowed E relative-error increase (fraction)")
    p.add_argument("--e-bottom-km", type=float, default=90.0)
    p.add_argument("--e-top-km", type=float, default=150.0)
    p.add_argument("--f-top-km", type=float, default=700.0)
    p.add_argument("--f-split-km", type=float, default=400.0, help="bottomside/topside split for ray classes")
    p.add_argument("--e-valley-top-km", type=float, default=200.0)
    p.add_argument("--e-valley-ratio", type=float, default=1.1)
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    a = parse_args()
    isr = {code: load_isr(path) for code, path in (item.split("=", 1) for item in a.isr)}
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(a.npz_dir.glob("*.npz")):
        rows.extend(hourly_metrics(f, isr, a))
    d = pd.DataFrame(rows)
    d.to_csv(a.out_dir / "hourly_metrics.csv", index=False)
    print(f"hourly_metrics.csv: {len(d)} rows ({d['hour'].nunique()} hours)")

    for name, table in summaries(d, a).items():
        table.to_csv(a.out_dir / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(table)} rows")


if __name__ == "__main__":
    main()
