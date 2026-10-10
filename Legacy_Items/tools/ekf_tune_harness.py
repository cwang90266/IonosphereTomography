#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
Fast single-group parametric-EKF tuning + identifiability harness.

Phase-0 tool for the ISR-DA tuning effort. Instead of re-running the multi-hour
demo_isr_da_comparison pipeline, this loads ONE group's already-cached
gridded-KF result from Data/DA_Cache (which carries the eds_occ mesh, clean_list
arcs and IRI prior EDP that _run_parametric_ekf needs as input) and re-runs only
the parametric EKF across a small sweep of hyperparameters, scoring each combo
against co-located ISR truth via the pipeline's own compute_isr_metrics().

It also prints the "1a" identifiability read-out: per-parameter posterior
variance reduction 1 - diag(post_P)/diag(prior_P), aggregated over grid points.
Parameters whose variance barely shrinks are the ones TEC observations cannot
constrain (expected: hmF2 and the shape params under TEC-only obs) -- this is
the quantitative core of "why does data inclusion help unevenly".

Usage
-----
  tools/ekf_tune_harness.py --group 2025-08-27_0912 --obs-mode ro_only \
      --alpha 0.5 0.35 0.2 --prior-scale 1.0 0.5 --sigma-obs 10

Each of --alpha/--sigma-obs/--max-rays/--tol/--max-iter/--n-members/
--prior-scale accepts one or more values; the harness runs the full Cartesian
product. Omitted knobs use the demo's current EKF_* defaults. Only ro_only and
ro_igs are supported (igs_only uses a separate run_info_window adapter path that
is not reconstructable from the cached gridded_kf pickle -- follow-up work).
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

# Headless plotting: _run_parametric_ekf writes a small convergence figure.
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo_isr_da_comparison as D
from Ionosphere_Tomography_Inverter.ionospheric_state import N_STATE, PARAM_NAMES


def _parse_free_spec(spec: str):
    """Turn a --free-sets token into (free_params, param_stages) for EKF_Param.

    "all"                    -> (None, None)               all params free
    "log10(NmF2)"            -> (["log10(NmF2)"], None)    freeze the rest
    "log10(NmF2),hmF2"       -> (["log10(NmF2)","hmF2"], None)
    "log10(NmF2);all"        -> (None, [["log10(NmF2)"], None])   staged
    """
    if spec is None or spec.strip().lower() == "all":
        return None, None
    if ";" in spec:
        stages = []
        for sub in spec.split(";"):
            sub = sub.strip()
            stages.append(None if sub.lower() == "all"
                          else [p.strip() for p in sub.split(",") if p.strip()])
        return None, stages
    return [p.strip() for p in spec.split(",") if p.strip()], None


def _load_kf_cache(group_key: str, bin_count, obs_mode: str) -> dict:
    path = D._cache_path(group_key, bin_count, obs_mode, "gridded_kf")
    if not path.exists():
        raise SystemExit(
            f"[harness] no cached gridded_kf result at {path}\n"
            f"          (run the pipeline for this group/obs_mode first, or pick "
            f"another --group/--obs-mode/--bin)")
    print(f"[harness] loading KF cache {path.name} "
          f"({path.stat().st_size / 1e6:.1f} MB) ...")
    with open(path, "rb") as fh:
        res_kf = pickle.load(fh)
    status = res_kf.get("status", "?") if isinstance(res_kf, dict) else "?"
    if not isinstance(res_kf, dict) or status != "Success":
        raise SystemExit(
            f"[harness] cached KF result is unusable (status={status!r}); "
            f"re-run the pipeline with --force for this group/obs_mode.")
    return res_kf


def _identifiability(res_ekf: dict) -> np.ndarray:
    """Per-parameter posterior variance reduction, averaged over grid points.

    prior_P/post_P are (N_STATE*n_geo, N_STATE*n_geo) in param-major/geo-minor
    C-order, so the diagonal reshapes to (N_STATE, n_geo).
    Returns a length-N_STATE array of 1 - mean(post_var)/mean(prior_var).
    """
    prior_P = np.asarray(res_ekf["prior_P"])
    post_P = np.asarray(res_ekf["post_P"])
    n_state = prior_P.shape[0]
    n_geo = n_state // N_STATE
    prior_var = np.diag(prior_P).reshape(N_STATE, n_geo)
    post_var = np.diag(post_P).reshape(N_STATE, n_geo)
    pv = prior_var.mean(axis=1)
    qv = post_var.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        red = 1.0 - np.where(pv > 0, qv / pv, np.nan)
    return red


def _score(rows: list[dict]) -> dict:
    """Reduce compute_isr_metrics rows (one per ISR instrument) to scalars.

    foF2/hmF2 errors are averaged as absolute values across instruments; the
    prior columns are the assimilation-free baseline for the same window.
    """
    if not rows:
        return {}
    def _absmean(key):
        vals = [abs(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan
    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan
    return dict(
        n_isr           = len(rows),
        prior_foF2_err  = _absmean("prior_foF2_err_mhz"),
        post_foF2_err   = _absmean("post_foF2_err_mhz"),
        prior_hmF2_err  = _mean("prior_hmF2_err_km"),
        post_hmF2_err   = _mean("post_hmF2_err_km"),
        prior_bp_mae    = _mean("prior_below_peak_mae_ne"),
        post_bp_mae     = _mean("post_below_peak_mae_ne"),
        post_tec_rmse   = _mean("post_tec_rmse"),
        ekf_converged   = rows[0].get("ekf_converged"),
        ekf_n_iter      = rows[0].get("ekf_n_iterations"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True,
                    help='group_key, e.g. "2025-08-27_0912"')
    ap.add_argument("--obs-mode", default="ro_only", choices=["ro_only", "ro_igs"],
                    help="observation mode (igs_only not supported; see module docstring)")
    ap.add_argument("--bin", default="all",
                    help='occultation bin: "all" or an int (default all)')
    ap.add_argument("--alpha", type=float, nargs="+", default=None)
    ap.add_argument("--sigma-obs", type=float, nargs="+", default=None)
    ap.add_argument("--max-rays", type=int, nargs="+", default=None)
    ap.add_argument("--tol", type=float, nargs="+", default=None)
    ap.add_argument("--max-iter", type=int, nargs="+", default=None)
    ap.add_argument("--tec-rmse-tol", type=float, nargs="+", default=None,
                    help="TEC-innovation RMSE gate (TECU) added to the early-stop "
                         "test: converge only when ||dP||/||P||<tol AND RMSE<this. "
                         "Omit / None = step-norm only (legacy).")
    ap.add_argument("--n-members", type=int, nargs="+", default=None)
    ap.add_argument("--prior-scale", type=float, nargs="+", default=None)
    ap.add_argument("--adapt-alpha", action="store_true",
                    help="Phase-2a: residual-monitoring adaptive step scale "
                         "(accept+accelerate on descent, reject+shrink on rise).")
    ap.add_argument("--alpha-max", type=float, default=1.0,
                    help="ceiling for the adaptive step scale (default 1.0).")
    ap.add_argument("--step-clip", type=float, default=None,
                    help="Phase-2a per-element trust region: clip each dP to "
                         "±step_clip·prior_std. None = off.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="global np.random seed reset before each combo so the "
                         "prior ensemble draw is identical across free-sets "
                         "(reproducible sweep). Default 12345.")
    ap.add_argument(
        "--free-sets", nargs="+", default=None,
        help='parameter-freedom specs (one config each; outer sweep dimension). '
             '"all" = all 8 params free (default/legacy). A comma list such as '
             '"log10(NmF2)" frees ONLY those and freezes the rest at the prior '
             '(Phase-2c: TEC only observes NmF2). Use ";" to STAGE, e.g. '
             '"log10(NmF2);all" fits NmF2 first then relaxes all params. '
             'Example: --free-sets all "log10(NmF2)" "log10(NmF2);all"')
    args = ap.parse_args()

    bin_count = D._parse_bin_count_value(args.bin)
    group_key = args.group
    date = D.pd.Timestamp(group_key.split("_")[0]).date()

    # Sweep grids default to the demo's current EKF_* config (single point).
    alphas   = args.alpha       or [D.EKF_ALPHA]
    sigmas   = args.sigma_obs   or [D.EKF_SIGMA_OBS]
    maxrays  = args.max_rays    or [D.EKF_MAX_RAYS]
    tols     = args.tol         or [D.EKF_TOL]
    maxiters = args.max_iter    or [D.EKF_MAX_ITER]
    nmembers = args.n_members   or [D.EKF_N_MEMBERS]
    pscales  = args.prior_scale or [D.EKF_PRIOR_SCALE]
    tectols  = args.tec_rmse_tol or [None]
    freesets = args.free_sets   or ["all"]

    res_kf = _load_kf_cache(group_key, bin_count, args.obs_mode)
    print("[harness] loading ISR EDPs ...")
    edps = D.load_edps()

    tmp_dir = tempfile.mkdtemp(prefix="ekf_tune_")
    combos = list(itertools.product(
        freesets, alphas, sigmas, maxrays, tols, maxiters, nmembers, pscales,
        tectols))
    print(f"[harness] group={group_key} obs_mode={args.obs_mode} bin={args.bin} "
          f"| {len(combos)} combo(s)\n")

    results = []
    ident_last = None
    for (fspec, alpha, sigma, mrays, tol, miter, nmem, pscale, ttol) in combos:
        free_params, param_stages = _parse_free_spec(fspec)
        tag = (f"free={fspec} alpha={alpha} sigma={sigma} rays={mrays} tol={tol} "
               f"iter={miter} nmem={nmem} pscale={pscale} tec_rmse_tol={ttol}")
        print(f"── running EKF | {tag}")
        # Reseed the global NumPy RNG before *every* combo so each free-set /
        # tuning point sees the identical prior ensemble draw
        # (generate_ensemble[_spatial] uses np.random.randn). This makes the
        # free-set comparison apples-to-apples and the whole sweep reproducible.
        np.random.seed(args.seed)
        try:
            res_ekf = D._run_parametric_ekf(
                res_kf=res_kf, alt_grid=D.ALT_GRID, save_dir=tmp_dir,
                group_key=f"{group_key}_{args.obs_mode}",
                n_members=nmem, sigma_obs=sigma, max_update_rays=mrays,
                alpha=alpha, tol=tol, max_iter=miter, tec_rmse_tol=ttol,
                adapt_alpha=args.adapt_alpha, alpha_max=args.alpha_max,
                step_clip=args.step_clip,
                prior_scale=pscale,
                free_params=free_params, param_stages=param_stages,
            )
        except Exception as exc:  # noqa: BLE001 -- surface, keep sweeping
            print(f"   [FAILED] {exc!r}")
            results.append((fspec, alpha, sigma, mrays, tol, miter, nmem, pscale, ttol, {}))
            continue

        filter_results = {args.obs_mode: {"parametric_ekf": res_ekf}}
        day_info = dict(date=date, group_key=group_key, bin_count=bin_count,
                        bin_label=args.bin,
                        n_ro_occultations=res_kf.get("n_occultations", np.nan),
                        n_igs_arcs=np.nan)
        rows = D.compute_isr_metrics(day_info, filter_results, edps)
        score = _score(rows)
        results.append((fspec, alpha, sigma, mrays, tol, miter, nmem, pscale, ttol, score))
        ident_last = _identifiability(res_ekf)
        if not rows:
            print("   [warn] no co-located ISR truth in this window -- "
                  "profile metrics unavailable (try another --group).")

    # ── Sweep results table ─────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("SWEEP RESULTS  (foF2/hmF2 vs ISR; prior = assimilation-free baseline)")
    print("=" * 108)
    hdr = (f"{'free':>18} {'alpha':>6} {'sigma':>6} {'tectol':>7} {'pscale':>7} "
           f"{'conv':>5} {'nit':>4} | {'foF2 MHz':>16} {'hmF2 km':>16} "
           f"{'belowpk MAE':>13} {'TECrmse':>9}")
    print(hdr)
    print(f"{'':>18} {'':>6} {'':>6} {'':>7} {'':>7} {'':>5} {'':>4} | "
          f"{'prior->post':>16} {'prior->post':>16}")
    print("-" * 124)
    for (fs, a, s, mr, tl, mi, nm, ps, tt, sc) in results:
        fs_lbl = (fs if fs else "all")[:18]
        tt_lbl = "-" if tt is None else f"{tt:g}"
        if not sc:
            print(f"{fs_lbl:>18} {a:>6} {s:>6} {tt_lbl:>7} {ps:>7}  FAILED / no result")
            continue
        conv = {True: "yes", False: "no"}.get(sc.get("ekf_converged"), "?")
        print(f"{fs_lbl:>18} {a:>6} {s:>6} {tt_lbl:>7} {ps:>7} {conv:>5} "
              f"{str(sc.get('ekf_n_iter','?')):>4} | "
              f"{sc['prior_foF2_err']:>6.3f}->{sc['post_foF2_err']:<8.3f} "
              f"{sc['prior_hmF2_err']:>6.1f}->{sc['post_hmF2_err']:<8.1f} "
              f"{sc['post_bp_mae']:>13.3e} {sc['post_tec_rmse']:>9.3f}")

    # ── Identifiability read-out (from the last successful combo) ────────────
    if ident_last is not None:
        print("\n" + "=" * 60)
        print("IDENTIFIABILITY  (posterior variance reduction per parameter)")
        print("  1 - post_var/prior_var, averaged over grid points; ~0 => the")
        print("  observations do not constrain this parameter.")
        print("=" * 60)
        for name, red in zip(PARAM_NAMES, ident_last):
            bar = "#" * int(max(0.0, min(1.0, red)) * 40)
            print(f"  {name:>12}  {red:6.3f}  {bar}")
    print()


if __name__ == "__main__":
    main()
