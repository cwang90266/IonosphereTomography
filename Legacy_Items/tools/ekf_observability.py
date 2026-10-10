#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
Phase-1 parametric-EKF observability + prior round-trip diagnostic.

This is the rigorous, geometry-based answer to "why does data inclusion help
unevenly": it measures which of the 8 IRI parameters the TEC observation
geometry can actually constrain, WITHOUT the ensemble-covariance overconfidence
that inflates the filter's own posterior-variance-reduction read-out
(tools/ekf_tune_harness.py's IDENTIFIABILITY block).

Two analyses are produced from a single group's cached gridded-KF result:

  1a. OBSERVABILITY (Fisher information / Cramer-Rao).  Using the analytical
      sTEC Jacobian J = dy/dP evaluated at the *true IRI prior* (exposed by
      EKF_Param(..., return_diagnostics=True)) and the prior parameter
      variances, we form the dimensionless normalised Jacobian

          J~[:,j] = J[:,j] * prior_std_j / sigma_obs          (obs-sigma per prior-sigma)

      and the Fisher information  F~ = J~^T J~.  Then:
        * raw per-param SNR^2  = diag(F~)          (signal a 1-sigma prior
          excursion of param j puts into the data, in noise units, summed over
          rays; << 1 => unobservable in isolation);
        * CRB variance reduction = 1 - diag((I + F~)^-1)     (single-step,
          correlation-aware, the clean analogue of the filter read-out);
        * the singular spectrum of J~ (observable modes) and how many effective
          DOF the geometry resolves (sum s^2/(1+s^2)).
      For contrast we also print the filter's single-step ENSEMBLE reduction
      1 - diag(post_P)/diag(prior_P); the gap between it and the CRB value is
      the ensemble/low-rank overconfidence.

  1b. PRIOR ROUND-TRIP LOSS.  The EKF prior EDP is a lossy round trip
      (raw IRI EDP -> fit 8 params -> reconstruct EDP), unlike the gridded KF
      whose prior is the raw IRI EDP.  We compare foF2/hmF2/profile-RMS of the
      raw IRI prior (res_kf['prior_edp_3d']) against the parametric-fit prior
      (res_ekf['prior_edp_3d']) column by column -- isolating how much of the
      "EKF prior is worse" is baked in before any assimilation.

Usage
-----
  tools/ekf_observability.py --group 2025-08-27_1227 --bin 5 --obs-mode ro_only

Only ro_only / ro_igs are supported (same cache-reconstruction constraint as
ekf_tune_harness.py).  No co-located ISR truth is required -- this runs on any
group that has a cached gridded_kf pickle.
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
from Ionosphere_Tomography_Inverter.ionospheric_state import N_STATE, PARAM_NAMES


def _load_kf_cache(group_key: str, bin_count, obs_mode: str) -> dict:
    path = D._cache_path(group_key, bin_count, obs_mode, "gridded_kf")
    if not path.exists():
        raise SystemExit(f"[obs] no cached gridded_kf result at {path}")
    print(f"[obs] loading KF cache {path.name} "
          f"({path.stat().st_size / 1e6:.1f} MB) ...")
    import pickle
    with open(path, "rb") as fh:
        res_kf = pickle.load(fh)
    if not isinstance(res_kf, dict) or res_kf.get("status") != "Success":
        raise SystemExit("[obs] cached KF result unusable; re-run pipeline --force.")
    return res_kf


def _observability(res_ekf: dict) -> dict:
    """Compute the Fisher-information observability read-out from the prior Jacobian."""
    J       = np.asarray(res_ekf["prior_jacobian"], dtype=float)   # (n_obs, n_state)
    sigma   = float(res_ekf["prior_obs_sigma"])
    prior_P = np.asarray(res_ekf["prior_P"], dtype=float)          # (n_state, n_state)
    post_P  = np.asarray(res_ekf["post_P"], dtype=float)
    n_state = J.shape[1]
    n_geo   = n_state // N_STATE

    prior_var = np.clip(np.diag(prior_P), 0.0, None)
    prior_std = np.sqrt(prior_var)                                 # (n_state,)

    # Dimensionless normalised Jacobian: obs-sigma of signal per prior-sigma of param.
    J_tilde = J * (prior_std[np.newaxis, :] / sigma)              # (n_obs, n_state)
    F_tilde = J_tilde.T @ J_tilde                                 # (n_state, n_state)

    # (1) Raw per-param SNR^2 (diagonal of Fisher info; ignores correlations).
    snr2 = np.diag(F_tilde)

    # (2) Correlation-aware single-step CRB reduction: prior = I in normalised
    #     coords, so P_post~ = (I + F~)^-1; reduction = 1 - diag(P_post~).
    try:
        import scipy.linalg as _la
        P_post_tilde = _la.solve(
            np.eye(n_state) + F_tilde, np.eye(n_state), assume_a="pos")
    except Exception:
        P_post_tilde = np.linalg.pinv(np.eye(n_state) + F_tilde)
    crb_red = 1.0 - np.clip(np.diag(P_post_tilde), 0.0, None)

    # (3) Filter's own single-step ENSEMBLE reduction (the confounded one).
    with np.errstate(divide="ignore", invalid="ignore"):
        ens_red = 1.0 - np.where(prior_var > 0,
                                 np.diag(post_P) / prior_var, np.nan)

    # Aggregate per-parameter over grid points (param-major/geo-minor C-order).
    def _per_param(vec):
        return np.asarray(vec).reshape(N_STATE, n_geo).mean(axis=1)

    # (4) Singular spectrum of the normalised Jacobian -> observable modes.
    s = np.linalg.svd(J_tilde, compute_uv=False)                  # (min(n_obs,n_state),)
    resolved_dof = float(np.sum(s ** 2 / (1.0 + s ** 2)))         # trace of resolution

    # Which params load the leading observable modes (|V| aggregated over geo).
    #   J~ = U S V^T ; right singular vectors V (n_state, k)
    _, _, Vt = np.linalg.svd(J_tilde, full_matrices=False)
    n_modes = min(4, Vt.shape[0])
    mode_loadings = []
    for m in range(n_modes):
        v = Vt[m].reshape(N_STATE, n_geo)
        load = np.sqrt((v ** 2).sum(axis=1))                     # per-param energy
        mode_loadings.append(load / (load.sum() + 1e-30))

    return dict(
        n_obs        = J.shape[0],
        n_state      = n_state,
        n_geo        = n_geo,
        snr2         = _per_param(snr2),
        crb_red      = _per_param(crb_red),
        ens_red      = _per_param(ens_red),
        sing_vals    = s,
        resolved_dof = resolved_dof,
        mode_loadings= mode_loadings,
    )


def _roundtrip_loss(res_kf: dict, res_ekf: dict) -> dict:
    """foF2/hmF2/profile loss from the IRI -> 8-param -> EDP round trip (pre-assimilation)."""
    raw  = np.asarray(res_kf["prior_edp_3d"], dtype=float)    # (n_alt, n_geo) raw IRI
    fit  = np.asarray(res_ekf["prior_edp_3d"], dtype=float)   # (n_alt, n_geo) parametric fit
    alt  = np.asarray(D.ALT_GRID, dtype=float)
    if raw.shape != fit.shape:
        return dict(error=f"shape mismatch raw{raw.shape} vs fit{fit.shape}")

    n_geo = raw.shape[1]
    dfof2, dhmf2, prof_rms, prof_scale = [], [], [], []
    ne_to_mhz = 8.978e-6
    for g in range(n_geo):
        rc, fc = raw[:, g], fit[:, g]
        rnm, rhm = D.extract_robust_f2_peak(rc, alt)
        fnm, fhm = D.extract_robust_f2_peak(fc, alt)
        if np.isfinite(rnm) and np.isfinite(fnm) and rnm > 0 and fnm > 0:
            dfof2.append(ne_to_mhz * (np.sqrt(fnm) - np.sqrt(rnm)))
        if np.isfinite(rhm) and np.isfinite(fhm):
            dhmf2.append(fhm - rhm)

        m = np.isfinite(rc) & np.isfinite(fc) & (rc > 1e8)
        if m.sum() > 5:
            prof_rms.append(np.sqrt(np.mean((fc[m] - rc[m]) ** 2)))
            prof_scale.append(np.sqrt(np.mean(rc[m] ** 2)))

    dhmf2 = np.asarray(dhmf2)
    return dict(
        n_geo       = n_geo,
        dfoF2_mean  = float(np.mean(np.abs(dfof2)))   if dfof2 else np.nan,
        dfoF2_max   = float(np.max(np.abs(dfof2)))    if dfof2 else np.nan,
        dhmF2_mean  = float(np.mean(dhmf2))           if dhmf2.size else np.nan,
        dhmF2_med   = float(np.median(dhmf2))         if dhmf2.size else np.nan,
        dhmF2_absmn = float(np.mean(np.abs(dhmf2)))   if dhmf2.size else np.nan,
        dhmF2_max   = float(np.max(np.abs(dhmf2)))    if dhmf2.size else np.nan,
        dhmF2_frac50= (float(np.mean(np.abs(dhmf2) > 50.0)) if dhmf2.size else np.nan),
        prof_rms    = float(np.mean(prof_rms))        if prof_rms else np.nan,
        prof_rel    = (float(np.mean(prof_rms) / (np.mean(prof_scale) + 1e-30))
                       if prof_rms else np.nan),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--obs-mode", default="ro_only", choices=["ro_only", "ro_igs"])
    ap.add_argument("--bin", default="all")
    ap.add_argument("--sigma-obs", type=float, default=None,
                    help="override obs sigma (default: demo EKF_SIGMA_OBS)")
    ap.add_argument("--n-members", type=int, default=None)
    args = ap.parse_args()

    bin_count = D._parse_bin_count_value(args.bin)
    res_kf = _load_kf_cache(args.group, bin_count, args.obs_mode)

    import tempfile
    tmp = tempfile.mkdtemp(prefix="ekf_obs_")
    sigma = args.sigma_obs if args.sigma_obs is not None else D.EKF_SIGMA_OBS
    nmem  = args.n_members if args.n_members is not None else D.EKF_N_MEMBERS

    print(f"[obs] running EKF prior linearisation "
          f"(max_iter=1, sigma_obs={sigma}, n_members={nmem}) ...\n")
    res_ekf = D._run_parametric_ekf(
        res_kf=res_kf, alt_grid=D.ALT_GRID, save_dir=tmp,
        group_key=f"{args.group}_{args.obs_mode}",
        n_members=nmem, sigma_obs=sigma, max_update_rays=D.EKF_MAX_RAYS,
        alpha=D.EKF_ALPHA, tol=D.EKF_TOL, max_iter=1,
        prior_scale=D.EKF_PRIOR_SCALE, return_diagnostics=True,
    )
    if res_ekf.get("prior_jacobian") is None:
        raise SystemExit("[obs] prior_jacobian missing -- check return_diagnostics wiring.")

    obs = _observability(res_ekf)
    rt  = _roundtrip_loss(res_kf, res_ekf)

    # ── 1a. Observability read-out ───────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"OBSERVABILITY  (group={args.group} bin={args.bin} obs_mode={args.obs_mode})")
    print(f"  {obs['n_obs']} update rays | {obs['n_geo']} grid pts | "
          f"n_state={obs['n_state']} | resolved DOF ~ {obs['resolved_dof']:.2f} "
          f"of {obs['n_state']} state dims (~{obs['resolved_dof']/obs['n_geo']:.2f}/grid pt)")
    print("  SNR^2  = diag(Fisher) : signal a 1sigma prior excursion puts in the")
    print("          data (noise units); <1 => unobservable in isolation.")
    print("  CRB    = 1 - diag((I+F)^-1)  : clean single-step variance reduction.")
    print("  ENS    = 1 - post_var/prior_var (filter's ensemble read-out; inflated).")
    print("=" * 74)
    print(f"  {'param':>12} {'SNR^2':>10} {'sqrt':>7}  {'CRB red':>8}  {'ENS red':>8}")
    print("-" * 74)
    for i, name in enumerate(PARAM_NAMES):
        snr2 = obs["snr2"][i]
        bar = "#" * int(max(0.0, min(1.0, obs["crb_red"][i])) * 24)
        print(f"  {name:>12} {snr2:>10.3g} {np.sqrt(max(snr2,0)):>7.2f}  "
              f"{obs['crb_red'][i]:>8.3f}  {obs['ens_red'][i]:>8.3f}  {bar}")

    print("\n  Singular spectrum of normalised Jacobian (observable modes):")
    sv = obs["sing_vals"]
    top = sv[:8]
    print("    " + "  ".join(f"{v:.2f}" for v in top) +
          (f"   (+{len(sv)-8} more)" if len(sv) > 8 else ""))
    print(f"    modes with s>1 (signal>noise): {int(np.sum(sv > 1.0))}")
    print("\n  Leading observable modes -- dominant parameter loadings:")
    for m, load in enumerate(obs["mode_loadings"]):
        order = np.argsort(load)[::-1][:3]
        parts = ", ".join(f"{PARAM_NAMES[j]}={load[j]:.2f}" for j in order)
        print(f"    mode {m} (s={sv[m]:.2f}): {parts}")

    # ── 1b. Prior round-trip loss ────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("PRIOR ROUND-TRIP LOSS  (raw IRI EDP  ->  8-param fit  ->  reconstructed EDP)")
    print("  How much foF2/hmF2/shape is lost by the parametric fit BEFORE any")
    print("  assimilation.  The gridded KF keeps the raw IRI prior; the EKF does not.")
    print("=" * 74)
    if "error" in rt:
        print(f"  [skip] {rt['error']}")
    else:
        print(f"  grid points compared : {rt['n_geo']}")
        print(f"  |dfoF2|  mean / max  : {rt['dfoF2_mean']:.3f} / {rt['dfoF2_max']:.3f} MHz")
        print(f"  dhmF2    mean/median : {rt['dhmF2_mean']:+.2f} / {rt['dhmF2_med']:+.2f} km "
              f"(signed)  |mean| {rt['dhmF2_absmn']:.2f}, max {rt['dhmF2_max']:.1f}")
        print(f"  dhmF2    |.|>50km    : {100*rt['dhmF2_frac50']:.0f}% of columns")
        print(f"  profile RMS diff     : {rt['prof_rms']:.3e} e/m^3  "
              f"({100*rt['prof_rel']:.1f}% of raw profile RMS)")
    print()


if __name__ == "__main__":
    main()
