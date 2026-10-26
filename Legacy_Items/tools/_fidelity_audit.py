#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
Phase-2b fidelity audit: how faithfully does the 8-parameter state round-trip
the IRI Ne(h) profile?

For a given group we call the same Fortran IRI batch the production EKF uses
(_get_iri_edp_and_features_batch) to obtain, per grid point:
    ne_iri   : the reference IRI electron-density profile
    feat     : the 13 IRI scalar features
then build the state two ways --
    direct : _state_from_iri_direct(ne_iri, feat, alt)   [production path]
    fit    : _fit_iri_params(ne_iri, alt)                [fallback path]
reconstruct Ne(h) from each state via _ne_profile_ensemble (what the EKF
assimilates against) and score the reconstruction vs the reference.

Metrics per column: log10-RMSE over the whole profile, and separately over
the topside (h>hmF2), the F-region bottomside (hmF2>h>150), and the E-region
(h<150). A column is "failing" if whole-profile log10-RMSE > --fail-thresh
(default 0.10 ~ 26% mean multiplicative error). Reports the failing fraction,
the distribution, and the worst columns with a per-region breakdown so we can
see WHICH part of the profile the parametric model cannot represent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo_isr_da_comparison as D
from demo_compare_kf_enkf import (
    _state_from_iri_direct, _fit_iri_params,
    _get_iri_edp_and_features_batch, _solar_sampling_df,
)
from Ionosphere_Tomography_Inverter.observation_operator import _ne_profile_ensemble
from Ionosphere_Tomography_Inverter.ionospheric_state import (
    N_STATE, PARAM_NAMES, I_LOG_NMF2, I_LOG_NME,
)


def _reconstruct(state_log: np.ndarray, alt_grid: np.ndarray) -> np.ndarray:
    """state (log convention) -> Ne(h) via the EKF's own profile model."""
    lin = state_log.astype(float).copy()
    lin[I_LOG_NMF2] = 10.0 ** lin[I_LOG_NMF2]
    lin[I_LOG_NME] = 10.0 ** lin[I_LOG_NME]
    return _ne_profile_ensemble(alt_grid, lin[:, None])[:, 0]


def _region_logrmse(ne_ref, ne_fit, alts, hmf2):
    ref = np.maximum(np.asarray(ne_ref), 1.0)
    fit = np.maximum(np.asarray(ne_fit), 1.0)
    d = np.log10(fit) - np.log10(ref)
    valid = np.isfinite(ne_ref) & (ne_ref > 1.0)   # only score real IRI points

    def rms(mask):
        m = mask & valid
        return float(np.sqrt(np.mean(d[m] ** 2))) if m.sum() >= 2 else np.nan

    whole = rms(np.ones_like(alts, dtype=bool))
    top = rms(alts > hmf2)
    fbot = rms((alts <= hmf2) & (alts > 150.0))
    ereg = rms(alts <= 150.0)
    return whole, top, fbot, ereg


def _audit_one(group, obs_mode, bin_count, fail_thresh):
    import pickle
    path = D._cache_path(group, bin_count, obs_mode, "gridded_kf")
    with open(path, "rb") as fh:
        res_kf = pickle.load(fh)

    alt_grid = D.ALT_GRID
    eds_occ = res_kf["eds_occ"]
    verts = eds_occ.geolocation
    lats = verts[:, 1].astype(float)
    lons = verts[:, 0].astype(float)
    n_geo = len(lats)
    t_centre = D._parse_time_window(res_kf.get("time_window", group))
    sdf = _solar_sampling_df(t_centre)

    print(f"[audit] {group} {obs_mode}: IRI batch for {n_geo} grid points "
          f"@ {t_centre} ...")
    ne_all, feat_all = _get_iri_edp_and_features_batch(
        t_centre, lats, lons, alt_grid, sdf)

    rows = {"direct": [], "fit": []}
    for g in range(n_geo):
        ne_ref = ne_all[:, g]
        hmf2 = float(feat_all[1, g])   # _FEAT_HMF2 = 1
        for name, builder in (
            ("direct", lambda: _state_from_iri_direct(ne_ref, feat_all[:, g], alt_grid)),
            ("fit",    lambda: _fit_iri_params(ne_ref, alt_grid)),
        ):
            try:
                st = builder()
                ne_fit = _reconstruct(st, alt_grid)
                w, t, fb, e = _region_logrmse(ne_ref, ne_fit, alt_grid, hmf2)
            except Exception as exc:  # noqa: BLE001
                w = t = fb = e = np.nan
                st = np.full(N_STATE, np.nan)
            rows[name].append((g, w, t, fb, e, st))
    return rows, n_geo


def _summarize(name, rows, fail_thresh):
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    whole = arr[:, 0]
    good = np.isfinite(whole)
    n = int(good.sum())
    fail = whole > fail_thresh
    nfail = int((fail & good).sum())
    print(f"\n===== {name.upper()} path =====")
    print(f"  columns scored : {n}")
    print(f"  whole-profile log10-RMSE: "
          f"median={np.nanmedian(whole):.4f}  mean={np.nanmean(whole):.4f}  "
          f"p90={np.nanpercentile(whole[good],90):.4f}  max={np.nanmax(whole):.4f}")
    print(f"  FAILING (>{fail_thresh:.2f}): {nfail}/{n} = {100*nfail/max(n,1):.1f}%")
    for lbl, col in (("topside", 1), ("F-bottom", 2), ("E-region", 3)):
        c = arr[:, col]
        print(f"    {lbl:>9} log10-RMSE: median={np.nanmedian(c):.4f}  "
              f"mean={np.nanmean(c):.4f}  max={np.nanmax(c):.4f}")
    order = np.argsort(-np.where(np.isfinite(whole), whole, -1))
    print(f"  worst {min(6,n)} columns (col: whole | top / Fbot / Ereg):")
    for i in order[:6]:
        g, w, t, fb, e, st = rows[i]
        print(f"    g={g:>4}: {w:.3f} | {t:.3f} / {fb:.3f} / {e:.3f}")
    return whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--obs-mode", default="ro_only", choices=["ro_only", "ro_igs"])
    ap.add_argument("--bin", default="5")
    ap.add_argument("--fail-thresh", type=float, default=0.10)
    args = ap.parse_args()

    bin_count = D._parse_bin_count_value(args.bin)
    rows, n_geo = _audit_one(args.group, args.obs_mode, bin_count, args.fail_thresh)
    print("\n" + "=" * 64)
    print(f"FIDELITY AUDIT  {args.group}  {args.obs_mode}  ({n_geo} grid pts)")
    print("  score = log10-RMSE of reconstructed Ne(h) vs reference IRI Ne(h)")
    print("=" * 64)
    for name in ("direct", "fit"):
        _summarize(name, rows[name], args.fail_thresh)
    print()


if __name__ == "__main__":
    main()
