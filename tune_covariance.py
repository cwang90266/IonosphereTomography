#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
tune_covariance.py — Covariance parameter sweep using RO cross-validation.

Sweeps over a grid of:
  • GAUSSIAN_COV_SIGMA  — all (sigma_h_km, sigma_latlon_km) combinations
    from {30,60,90,120} km vertical × {100,250,500,1000,1500} km horizontal
  • PRIOR_EDP_NOISE_SIGMA — lognormal noise injected into the EDP ensemble

For each combination the script:
  1. Splits orbit-1 RO occultations 50/50 into assimilation and withheld sets.
  2. Runs the joint KF assimilation on the assimilation half only.
  3. Evaluates the posterior EDP by computing predicted TEC for the withheld
     half and computing RMSE against the measured TEC (cross-validation).
  4. Saves every diagnostic figure with a tag encoding the parameter set.
  5. Appends one row to a results CSV.

After the sweep a summary plot ranks all configurations by cross-validation
TEC RMSE and TEC RMSE improvement (prior → posterior).

Run from the project root:
    python tune_covariance.py
"""

import sys
import os
import copy
import itertools
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

import datetime

import matplotlib.patches as _mpatch
from matplotlib.colors import Normalize as _MplNorm
from scipy.stats import gaussian_kde as _gaussian_kde

import demo_group as _demo_group
import demo_verification as _dv
from demo_verification import (
    filter_to_verif_region,
    assign_orbit_groups,
    VERIF_LAT_MIN, VERIF_LAT_MAX, VERIF_LON_MIN, VERIF_LON_MAX,
    ISR_LAT, ISR_LON_W, WINDOW_MINUTES,
)
from demo_group import (
    scan_metadata,
    process_group,
    CONSTELLATION_CONFIG,
    _CONST_FALLBACK_CMAP,
    _save_stats_csv,
)
from collections import defaultdict
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from demo import build_daily_global_edps, extract_robust_f2_peak
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
)


# ─────────────────────────────────────────────────────────────────────────────
# §0  User-configurable settings
# ─────────────────────────────────────────────────────────────────────────────

DOY        = 154
YYYY       = 2025
base_path  = (
    f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
    f"{YYYY}.{DOY}/"
)
alt_grid   = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
TYPE       = "log"
save_dir   = "./Figures/CovTuning/"
num_workers = 12
kf_config  = {"measurement_err": 1.0, "relaxation": 0.99}

# ── Parameter grid ─────────────────────────────────────────────────────────
# All combinations of vertical × horizontal Gaussian smoothing lengths,
# plus None for raw ensemble covariance (no smoothing).
_VERT_KM   = [30, 60, 90, 120]
_HORIZ_KM  = [100, 250, 500, 1000, 1500]
GAUSSIAN_SIGMA_GRID: list = [None] + list(itertools.product(_VERT_KM, _HORIZ_KM))

# Lognormal noise added to the EDP ensemble before covariance computation.
NOISE_SIGMA_GRID: list = [0.0, 0.0001, 0.001, 0.01, 0.5, 0.1]

NUM_RAY_SEGMENTS = 50   # passed to get_observation_operator_batch


# ─────────────────────────────────────────────────────────────────────────────
# §1  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _param_tag(gauss: tuple | None, noise: float) -> str:
    g_tag = "noSmooth" if gauss is None else f"g{gauss[0]}h{gauss[1]}ll"
    n_tag = f"n{noise:.4f}".rstrip("0").rstrip(".")
    return f"{g_tag}_{n_tag}"


def _param_label(gauss: tuple | None, noise: float) -> str:
    g_lbl = "No smoothing" if gauss is None else f"σ_h={gauss[0]} km, σ_ll={gauss[1]} km"
    n_lbl = f"noise σ={noise}"
    return f"{g_lbl} | {n_lbl}"


def _apply_params(gauss: tuple | None, noise: float) -> None:
    _demo_group.GAUSSIAN_COV_SIGMA    = gauss
    _demo_group.PRIOR_EDP_NOISE_SIGMA = noise if noise > 0 else None
    _demo_group._NOISE_SUFFIX         = (
        f"_noise{noise}" if (noise and noise > 0) else ""
    )


# ─────────────────────────────────────────────────────────────────────────────
# §1b  Arc-innovation diagnostic helpers (shared with demo_compare_kf_enkf)
# ─────────────────────────────────────────────────────────────────────────────

def _tangent_latlon_single(
    gnss_pt_km: np.ndarray,
    leo_pt_km: np.ndarray,
) -> tuple:
    """Return (lat_deg, lon_deg) of the tangent point for one ray epoch."""
    d     = leo_pt_km - gnss_pt_km
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        r   = float(np.linalg.norm(leo_pt_km))
        lat = float(np.degrees(np.arcsin(np.clip(leo_pt_km[2] / r, -1, 1))))
        lon = float(np.degrees(np.arctan2(leo_pt_km[1], leo_pt_km[0])))
        return lat, lon
    t_tp = -float(np.dot(gnss_pt_km, d)) / denom
    tp   = gnss_pt_km + np.clip(t_tp, 0.0, 1.0) * d
    r    = float(np.linalg.norm(tp))
    lat  = float(np.degrees(np.arcsin(np.clip(tp[2] / r, -1, 1))))
    lon  = float(np.degrees(np.arctan2(tp[1], tp[0])))
    return lat, lon


def _arc_stats_from_tec_slices(
    tec_slices: list,
    clean_list: list,
    sat_ids:    list,
) -> dict:
    """
    Compute per-arc innovation statistics from KF tec_slices.

    Returns dict with keys:
        arc_labels, arc_prior_mean, arc_post_mean,
        arc_prior_rmse, arc_post_rmse,
        arc_lats, arc_lons, all_prior, all_post
    """
    arc_labels     = []
    arc_prior_mean = []
    arc_post_mean  = []
    arc_prior_rmse = []
    arc_post_rmse  = []
    arc_lats_out   = []
    arc_lons_out   = []
    all_prior_list = []
    all_post_list  = []

    for i, sl in enumerate(tec_slices):
        meas        = np.asarray(sl["measured"],  dtype=float)
        prior       = np.asarray(sl["prior_tec"], dtype=float)
        post        = np.asarray(sl["post_tec"],  dtype=float)
        resid_prior = meas - prior
        resid_post  = meas - post
        all_prior_list.append(resid_prior)
        all_post_list.append(resid_post)

        arc_prior_mean.append(float(np.nanmean(resid_prior)))
        arc_post_mean.append( float(np.nanmean(resid_post)))
        arc_prior_rmse.append(float(np.sqrt(np.nanmean(resid_prior ** 2))))
        arc_post_rmse.append( float(np.sqrt(np.nanmean(resid_post  ** 2))))

        if i < len(sat_ids) and sat_ids[i]:
            _, prn = sat_ids[i]
            arc_labels.append(str(prn))
        else:
            arc_labels.append(f"arc{i:02d}")

        cl        = clean_list[i]
        leo_ecef  = np.asarray(cl["LEO"],  dtype=float)
        gnss_ecef = np.asarray(cl["GNSS"], dtype=float)
        n_rays    = leo_ecef.shape[1]
        tp_lats_i, tp_lons_i = [], []
        for j in range(n_rays):
            lat_j, lon_j = _tangent_latlon_single(gnss_ecef[:, j], leo_ecef[:, j])
            tp_lats_i.append(lat_j)
            tp_lons_i.append(lon_j)
        arc_lats_out.append(float(np.mean(tp_lats_i)))
        arc_lons_out.append(float(np.mean(tp_lons_i)))

    return dict(
        arc_labels     = arc_labels,
        arc_prior_mean = np.array(arc_prior_mean),
        arc_post_mean  = np.array(arc_post_mean),
        arc_prior_rmse = np.array(arc_prior_rmse),
        arc_post_rmse  = np.array(arc_post_rmse),
        arc_lats       = np.array(arc_lats_out),
        arc_lons       = np.array(arc_lons_out),
        all_prior      = np.concatenate(all_prior_list),
        all_post       = np.concatenate(all_post_list),
    )


def _plot_arc_innovation_diagnostic(
    arc_labels:     list,
    arc_prior_mean: np.ndarray,
    arc_post_mean:  np.ndarray,
    arc_prior_rmse: np.ndarray,
    arc_post_rmse:  np.ndarray,
    arc_lats:       np.ndarray,
    arc_lons:       np.ndarray,
    all_prior:      np.ndarray,
    all_post_main:  np.ndarray,
    group_key:      str,
    save_dir:       str,
    filter_name:    str,
    prior_rmse:     float,
    post_rmse:      float,
    all_post_raw:   "np.ndarray | None" = None,
    post_raw_label: str = "Post (raw)",
) -> None:
    """
    Four-panel per-arc TEC residual diagnostic figure.

    Panel A  Signed mean residual bars (prior vs posterior, sorted by |prior|).
    Panel B  Prior vs posterior RMSE scatter coloured by ΔRMSE.
    Panel C  Geographic map: hollow ring = prior RMSE, filled dot = post RMSE,
             dot colour encodes ΔRMSE via diverging colorbar.
    Panel D  Residual KDE + histogram.
    """
    n_arcs   = len(arc_labels)
    imp_mean = np.abs(arc_post_mean) < np.abs(arc_prior_mean)
    sort_idx = np.argsort(np.abs(arc_prior_mean))[::-1]

    fig = plt.figure(figsize=(18, max(10, 0.38 * n_arcs + 2)))
    gs  = fig.add_gridspec(
        3, 2,
        width_ratios=[1.5, 1],
        height_ratios=[1, 1, 1],
        hspace=0.52, wspace=0.42,
    )
    ax_bar  = fig.add_subplot(gs[:, 0])
    ax_scat = fig.add_subplot(gs[0, 1])
    ax_map  = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, 1])

    # ── Panel A: signed mean residual bar chart ───────────────────────────────
    bh    = 0.28
    y_pos = np.arange(n_arcs, dtype=float)
    for k, si in enumerate(sort_idx):
        y   = y_pos[k]
        imp = bool(imp_mean[si])
        ax_bar.barh(y + bh, arc_prior_mean[si], height=bh * 1.85,
                    color="#2166ac", alpha=0.88,
                    label="Prior  mean(obs−model)" if k == 0 else "")
        bar_col = "#1a9641" if imp else "#d7191c"
        ax_bar.barh(y - bh, arc_post_mean[si], height=bh * 1.85,
                    color=bar_col, alpha=0.84,
                    label=("Post  ↓ improved"  if (k == 0 and imp)
                           else ("Post  ↑ degraded" if (k == 0 and not imp)
                                 else "")))
    ax_bar.axvline(0, color="k", lw=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [arc_labels[sort_idx[k]] for k in range(n_arcs)],
        fontsize=8, fontfamily="monospace",
    )
    ax_bar.set_xlabel("Mean residual  obs − model  (TECU)", fontsize=9)
    ax_bar.set_title(
        f"Per-occultation mean TEC error — {filter_name}  ·  group {group_key}\n"
        f"Global RMSE: Prior {prior_rmse:.2f} TECU  →  Post {post_rmse:.2f} TECU",
        fontsize=9, fontweight="bold",
    )
    handles = [
        _mpatch.Patch(color="#2166ac", alpha=0.88, label="Prior  mean(obs−model)"),
        _mpatch.Patch(color="#1a9641", alpha=0.84, label="Post  ↓ |bias| reduced"),
        _mpatch.Patch(color="#d7191c", alpha=0.84, label="Post  ↑ |bias| increased"),
    ]
    ax_bar.legend(handles=handles, fontsize=8, loc="lower right")
    ax_bar.grid(axis="x", lw=0.4, alpha=0.5)

    # ── Panel B: prior vs posterior RMSE scatter ──────────────────────────────
    delta_rmse = arc_post_rmse - arc_prior_rmse
    v_sc  = max(float(np.percentile(np.abs(delta_rmse), 95)), 2.0)
    sc    = ax_scat.scatter(arc_prior_rmse, arc_post_rmse,
                            c=delta_rmse, cmap="RdYlGn_r",
                            norm=_MplNorm(-v_sc, v_sc),
                            s=60, edgecolors="k", linewidths=0.4, zorder=4)
    lim   = max(float(np.concatenate([arc_prior_rmse, arc_post_rmse]).max()) * 1.08, 5.0)
    ax_scat.plot([0, lim], [0, lim], "--", color="0.5", lw=0.9, label="no change")
    ax_scat.set_xlim(0, lim); ax_scat.set_ylim(0, lim)
    ax_scat.set_xlabel("Prior RMSE (TECU)", fontsize=8)
    ax_scat.set_ylabel("Post RMSE (TECU)",  fontsize=8)
    ax_scat.set_title(f"{filter_name}  Prior → Posterior RMSE per arc", fontsize=8)
    ax_scat.legend(fontsize=7)
    cb_sc = fig.colorbar(sc, ax=ax_scat, fraction=0.05, pad=0.02)
    cb_sc.set_label("ΔRMSE  post−prior (TECU)", fontsize=7)
    for k in range(n_arcs):
        ax_scat.annotate(arc_labels[k],
                         (arc_prior_rmse[k], arc_post_rmse[k]),
                         fontsize=5, ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points", zorder=5)

    # ── Panel C: geographic map ───────────────────────────────────────────────
    _sz_scale      = 5.0
    sz_prior       = 20 + _sz_scale * arc_prior_rmse
    sz_post        = 20 + _sz_scale * arc_post_rmse
    delta_rmse_map = arc_post_rmse - arc_prior_rmse
    v_map          = max(float(np.percentile(np.abs(delta_rmse_map), 95)), 2.0)

    ax_map.scatter(arc_lons, arc_lats,
                   s=sz_prior, facecolors="none",
                   edgecolors="#555555", linewidths=1.6, zorder=3)
    sc_map = ax_map.scatter(arc_lons, arc_lats,
                             s=sz_post,
                             c=delta_rmse_map, cmap="RdYlGn_r",
                             norm=_MplNorm(-v_map, v_map),
                             alpha=0.82, edgecolors="k", linewidths=0.35, zorder=4)
    for k in range(n_arcs):
        ax_map.annotate(arc_labels[k], (arc_lons[k], arc_lats[k]),
                        fontsize=5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", zorder=5)
    cb_map = fig.colorbar(sc_map, ax=ax_map, fraction=0.05, pad=0.02)
    cb_map.set_label("ΔRMSE  post − prior (TECU)\n← improved   degraded →", fontsize=7)
    map_handles = [
        _mpatch.Patch(facecolor="none", edgecolor="#555555",
                      linewidth=1.6, label="Prior RMSE (ring size)"),
        _mpatch.Patch(facecolor="grey", edgecolor="k",
                      linewidth=0.5, label="Post RMSE (dot size, coloured by ΔRMSE)"),
    ]
    ax_map.legend(handles=map_handles, fontsize=6, loc="best")
    ax_map.set_xlabel("Longitude (°E)", fontsize=8)
    ax_map.set_ylabel("Latitude (°N)",  fontsize=8)
    ax_map.set_title(
        f"{filter_name}  Prior ○ vs Posterior ● RMSE per arc\n"
        f"Dot colour: ΔRMSE (green = improved, red = degraded)",
        fontsize=8,
    )
    ax_map.grid(lw=0.3, alpha=0.4)

    # ── Panel D: residual histograms ──────────────────────────────────────────
    all_arrs     = [all_prior, all_post_main]
    if all_post_raw is not None:
        all_arrs.append(all_post_raw)
    finite_vals  = np.concatenate([a[np.isfinite(a)] for a in all_arrs])
    lo           = np.percentile(finite_vals,  1) - 5
    hi           = np.percentile(finite_vals, 99) + 5
    bins         = np.linspace(lo, hi, 45)

    hist_series = [
        (all_prior,     "#2166ac",
         f"Prior      μ={np.nanmean(all_prior):+.1f}  σ={np.nanstd(all_prior):.1f}"),
        (all_post_main, "#1a9641",
         f"Post {filter_name}   μ={np.nanmean(all_post_main):+.1f}  σ={np.nanstd(all_post_main):.1f}"),
    ]
    if all_post_raw is not None:
        hist_series.append(
            (all_post_raw, "#fdae61",
             f"{post_raw_label}  μ={np.nanmean(all_post_raw):+.1f}  σ={np.nanstd(all_post_raw):.1f}")
        )
    for arr, col, lbl in hist_series:
        ax_hist.hist(arr[np.isfinite(arr)], bins=bins,
                     density=True, alpha=0.42, color=col, label=lbl)
        try:
            kde_fn = _gaussian_kde(arr[np.isfinite(arr)])
            x_k    = np.linspace(bins[0], bins[-1], 300)
            ax_hist.plot(x_k, kde_fn(x_k), color=col, lw=1.6)
        except Exception:
            pass
    ax_hist.axvline(0, color="k", lw=0.8, linestyle="--")
    ax_hist.set_xlabel("Residual  obs − model  (TECU)", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.set_title(f"{filter_name}  residual distribution (all samples)", fontsize=8)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(lw=0.3, alpha=0.4)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    safe_key = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    tag      = filter_name.lower().replace(" ", "_")
    out_path = os.path.join(save_dir, f"{tag}_arc_innovations_{safe_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [{filter_name}] Arc innovation diagnostic → {out_path}")


def _split_orbit_meta(
    meta: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a group's occultation metadata 50/50 into assimilation and
    withheld verification sets.  Shuffle deterministically so the same
    split is used across all parameter combinations.
    """
    idx = meta.index.tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    half = int(np.ceil(len(idx) * 0.90)) - 1
    # print(half)
    assim_idx = idx[:half] if half > 0 else idx
    verif_idx = idx[half:] if half < len(idx) else []
    return (
        meta.loc[assim_idx].copy().reset_index(drop=True),
        meta.loc[verif_idx].copy().reset_index(drop=True) if verif_idx else pd.DataFrame(columns=meta.columns),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Withheld-RO cross-validation TEC RMSE
# ─────────────────────────────────────────────────────────────────────────────

def _build_clean_list(
    verif_meta: pd.DataFrame,
    podtc_max_rays: int = 500,
) -> tuple[list[dict], list[tuple[str, str]], list[str]]:
    """
    Parse withheld RO files and build the same clean_list structure that
    process_group uses for assimilation.  Only absolute-TEC arcs are included.

    Returns
    -------
    clean_list  : list of ray dicts
    sat_ids     : list of (leo_id, prn_id) per arc
    file_labels : list of display labels per arc
    """
    clean_list  = []
    sat_ids     = []
    file_labels = []
    for _, row in verif_meta.iterrows():
        data = parse_podTc2_nc_file(row["full_path"])
        if data is None:
            continue
        _, _, tang_raw = rayTangent(data["LEO"], data["GNSS"], units="km")
        tang_km  = tang_raw * 1e-3
        meas_tec = data.get("TEC_podTc2", data.get("TEC", np.zeros_like(tang_km)))
        valid    = ~np.isnan(meas_tec) & (meas_tec > 0)
        n_valid  = int(valid.sum())
        if n_valid < 50:
            continue
        if n_valid > podtc_max_rays:
            stride   = int(np.ceil(n_valid / podtc_max_rays))
            dec_idx  = np.where(valid)[0][::stride]
            dec_mask = np.zeros(len(meas_tec), dtype=bool)
            dec_mask[dec_idx] = True
        else:
            dec_mask = valid

        # Satellite identifiers
        leo_id  = str(data.get("leo_id", "??")).strip()
        con_id  = str(data.get("conid",  "?")).strip()
        prn_num = str(data.get("prn_id", "??")).strip()
        full_prn = f"{con_id}{prn_num}"

        # Time label from TEC-max epoch
        _tec_vals = data.get("TEC_podTc2", data.get("TEC", None))
        if _tec_vals is not None and len(_tec_vals) > 0:
            _tmax_idx  = int(np.argmax(_tec_vals))
            _tmax_time = data["date"] + pd.to_timedelta(float(data["time"][_tmax_idx]), unit="s")
            time_str   = _tmax_time.strftime("%H:%M")
        else:
            time_str = row["date"].strftime("%H:%M") if hasattr(row["date"], "strftime") else ""

        clean_list.append({
            "tec":        np.asarray(meas_tec[dec_mask], dtype=np.float64).flatten(),
            "tangent_km": tang_km[dec_mask].flatten(),
            "LEO":        data["LEO"][:,  dec_mask],
            "GNSS":       data["GNSS"][:, dec_mask],
            "tec_type":   "absolute",
        })
        sat_ids.append((leo_id, full_prn))
        file_labels.append(f"{row['spacecraft']} {time_str}")

    return clean_list, sat_ids, file_labels


def _verif_tec_rmse(
    res:        dict,
    verif_meta: pd.DataFrame,
    alt_grid:   np.ndarray,
) -> dict:
    """
    Compute cross-validation TEC RMSE on the withheld occultations.

    Returns a dict with scalar metrics plus per-arc slices for plotting:
        verif_tec_rmse_prior  — RMSE of prior predicted TEC vs. withheld TEC
        verif_tec_rmse_post   — RMSE of posterior predicted TEC vs. withheld TEC
        n_verif_rays          — number of withheld rays used
        tec_slices            — list of per-arc dicts with measured/prior/post TEC
        sat_ids               — list of (leo_id, prn_id) per arc
        file_labels           — list of display labels per arc
    """
    out = {
        "verif_tec_rmse_prior": np.nan,
        "verif_tec_rmse_post":  np.nan,
        "n_verif_rays":         0,
        "tec_slices":           [],
        "sat_ids":              [],
        "file_labels":          [],
    }
    if res.get("status") != "Success" or verif_meta.empty:
        return out

    eds_occ = res.get("eds_occ")
    if eds_occ is None:
        return out

    verif_clean, sat_ids, file_labels = _build_clean_list(verif_meta)
    if not verif_clean:
        print("    [verif] No valid withheld arcs.")
        return out

    # Build a fresh inverter on the same mesh (geometry only — no KF update)
    try:
        inv_verif = Ionosphere_Tomography_Inverter(
            EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0,
            n_rel_arcs=0, topside_alpha=0.0,
            gaussian_cov_sigma=_demo_group.GAUSSIAN_COV_SIGMA,
        )
        H_blocks = inv_verif.get_observation_operator_batch(
            verif_clean, num_segments=NUM_RAY_SEGMENTS
        )
    except Exception as exc:
        print(f"    [verif] H-matrix build failed: {exc}")
        return out

    n_sv     = inv_verif.attrs["n_state_vars"]
    n_sv_aug = inv_verif.attrs["n_state_vars_aug"]

    prior_flat  = res["prior_edp_3d"].flatten()[:, None]
    x_top_prior = inv_verif.attrs["x_top_prior"]
    post_flat   = res["joint_post_edp_3d"].flatten()[:, None]
    x_top_post  = res.get("x_top_tecu_jnt", x_top_prior)

    all_obs = []
    all_prior_tec = []
    all_post_tec  = []
    tec_slices = []

    for cl, H_arc in zip(verif_clean, H_blocks):
        H_arc_f  = H_arc.astype(np.float64)
        obs_arc  = cl["tec"]
        pr_arc   = (
            H_arc_f[:, :n_sv] @ prior_flat
            + H_arc_f[:, n_sv:n_sv_aug] @ x_top_prior[:, None]
        ).flatten()
        po_arc   = (
            H_arc_f[:, :n_sv] @ post_flat
            + H_arc_f[:, n_sv:n_sv_aug] @ x_top_post[:, None]
        ).flatten()

        all_obs.append(obs_arc)
        all_prior_tec.append(pr_arc)
        all_post_tec.append(po_arc)
        tec_slices.append({
            "measured":   obs_arc,
            "prior_tec":  pr_arc,
            "post_tec":   po_arc,
            "tangent_km": cl["tangent_km"],
        })

    obs_all   = np.concatenate(all_obs)
    prior_all = np.concatenate(all_prior_tec)
    post_all  = np.concatenate(all_post_tec)
    n_rays    = len(obs_all)

    out["verif_tec_rmse_prior"] = float(np.sqrt(np.mean((prior_all - obs_all) ** 2)))
    out["verif_tec_rmse_post"]  = float(np.sqrt(np.mean((post_all  - obs_all) ** 2)))
    out["n_verif_rays"]         = n_rays
    out["tec_slices"]           = tec_slices
    out["sat_ids"]              = sat_ids
    out["file_labels"]          = file_labels

    print(f"    [verif] {n_rays} withheld rays  "
          f"prior RMSE={out['verif_tec_rmse_prior']:.3f}  "
          f"post RMSE={out['verif_tec_rmse_post']:.3f} TECU")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §3  Cross-validation all-TEC plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_verif_all_tec(
    cv:          dict,
    group_key:   str,
    param_label: str,
    save_dir:    str,
) -> str | None:
    """
    2×2 constellation-panel figure showing measured / prior / posterior TEC
    for every withheld (cross-validation) occultation, mirroring the
    `group_{key}_joint_all_tec.png` produced by demo_group._plot_group.
    """
    tec_slices  = cv.get("tec_slices", [])
    sat_ids     = cv.get("sat_ids",    [])
    file_labels = cv.get("file_labels",[])
    n_occ       = len(tec_slices)
    if n_occ == 0:
        return None

    os.makedirs(save_dir, exist_ok=True)

    # ── Constellation colour assignment ──────────────────────────────────────
    _CONST_POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}

    occ_const     = []
    const_counts  = defaultdict(int)
    for i in range(n_occ):
        prn   = sat_ids[i][1] if i < len(sat_ids) else ""
        const = prn[0].upper() if prn else "?"
        occ_const.append(const)
        const_counts[const] += 1

    const_counter = defaultdict(int)
    occ_colours   = []
    for const in occ_const:
        cfg       = CONSTELLATION_CONFIG.get(const, {})
        cmap_name = cfg.get("cmap", _CONST_FALLBACK_CMAP)
        cmap_obj  = mpl.colormaps.get_cmap(cmap_name)
        n_in      = const_counts[const]
        idx_in    = const_counter[const]
        t         = (0.40 + 0.50 * (idx_in / max(n_in - 1, 1))) if n_in > 1 else 0.70
        occ_colours.append(cmap_obj(t))
        const_counter[const] += 1

    # ── Figure ────────────────────────────────────────────────────────────────
    all_alts = np.concatenate([sl["tangent_km"] for sl in tec_slices])
    alt_ylim = (0, max(float(np.nanmax(all_alts)) + 50, 800.0))

    prior_rmse = cv.get("verif_tec_rmse_prior", float("nan"))
    post_rmse  = cv.get("verif_tec_rmse_post",  float("nan"))
    safe_key   = group_key.replace("/", "_").replace(" ", "_").replace(":", "")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Cross-Validation TEC Profiles (withheld RO)  |  {group_key}\n"
        f"{param_label}\n"
        f"{n_occ} withheld occ — "
        f"Prior RMSE {prior_rmse:.2f} → Post RMSE {post_rmse:.2f} TECU",
        fontsize=10,
    )
    gs_tec = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.50)

    ax_tec    = {}
    first_ax  = None
    for const, (row, col) in _CONST_POS.items():
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
        ax  = fig.add_subplot(
            gs_tec[row, col],
            sharey=first_ax if first_ax is not None else None,
        )
        ax.set_title(cfg["name"], fontsize=9,
                     color=cfg.get("title_color", "black"), fontweight="bold")
        ax.grid(True, alpha=0.3, ls=":")
        ax_tec[const] = ax
        if first_ax is None:
            first_ax = ax

    style_handles = [
        Line2D([0], [0], color="gray", lw=2.2,          label="Measured TEC"),
        Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Prior TEC"),
        Line2D([0], [0], color="gray", lw=1.5, ls=":",  label="KF Posterior"),
    ]
    const_legend   = defaultdict(list)
    style_placed   = False

    for i, (sl, col) in enumerate(zip(tec_slices, occ_colours)):
        const = occ_const[i]
        ax_a  = ax_tec.get(const) or ax_tec.get("G") or next(iter(ax_tec.values()))
        ax_a.plot(sl["measured"],  sl["tangent_km"], color=col, lw=2.2)
        ax_a.plot(sl["prior_tec"], sl["tangent_km"], color=col, lw=1.3, ls="--", alpha=0.6)
        ax_a.plot(sl["post_tec"],  sl["tangent_km"], color=col, lw=1.5, ls=":",  alpha=0.9)
        prn_code = sat_ids[i][1]     if i < len(sat_ids)     else f"Occ {i + 1}"
        time_str = file_labels[i].split()[-1] if i < len(file_labels) else ""
        lbl      = f"{prn_code}  ({time_str})" if time_str else prn_code
        const_legend[const].append(Line2D([0], [0], color=col, lw=2.2, label=lbl))

    for const, ax_a in ax_tec.items():
        entries = const_legend.get(const, [])
        if entries:
            leg_h = entries + (style_handles if not style_placed else [])
            ax_a.legend(handles=leg_h, fontsize=7, loc="upper right", framealpha=0.85)
            style_placed = True
        else:
            ax_a.text(0.5, 0.5, "No data", transform=ax_a.transAxes,
                      ha="center", va="center", color="lightgray",
                      fontsize=11, style="italic")
        ax_a.set_ylim(*alt_ylim)
        if const in ("G", "R"):
            ax_a.set_ylabel("Tangent Altitude (km)")
        else:
            ax_a.tick_params(labelleft=False)
        if const in ("R", "C"):
            ax_a.set_xlabel("TEC (TECU)")

    noise_sfx = _demo_group._NOISE_SUFFIX
    fname     = f"group_{safe_key}{noise_sfx}_joint_verif_all_tec.png"
    path      = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    [verif] Cross-validation TEC plot → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# §4  Summary sweep plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sweep_summary(df: pd.DataFrame, save_dir: str) -> None:
    """
    Four-panel summary across all parameter combinations:
      TL — Cross-validation TEC RMSE (posterior) for each run
      TR — Cross-validation TEC RMSE improvement (%) for each run
      BL — Assimilation TEC RMSE improvement (%) for each run
      BR — Cross-validation TEC RMSE heat-map: x=noise sigma, y=Gaussian tag
    """
    os.makedirs(save_dir, exist_ok=True)

    ok = df[df["status"] == "Success"].copy()
    if ok.empty:
        print("  [sweep summary] No successful runs to plot.")
        return

    noise_vals   = sorted(ok["noise_sigma"].unique())
    n_noise      = max(len(noise_vals), 1)
    cmap_noise   = cm.get_cmap("plasma", n_noise)
    noise_colors = {v: cmap_noise(i) for i, v in enumerate(noise_vals)}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Covariance Parameter Sweep — Orbit 1  |  RO Cross-Validation",
                 fontsize=14, y=1.01)
    ax_cv, ax_cv_imprv, ax_assim_imprv, ax_heat = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    x       = np.arange(len(ok))
    xlabels = ok["param_tag"].tolist()

    # ── TL: Cross-validation posterior RMSE ──────────────────────────────────
    bar_colors = [noise_colors.get(v, "gray") for v in ok["noise_sigma"]]
    ax_cv.bar(x, ok["verif_tec_rmse_post"], color=bar_colors, edgecolor="black", lw=0.5)
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=5)
    ax_cv.set_ylabel("Cross-validation TEC RMSE (TECU)", fontsize=9)
    ax_cv.set_title("Posterior Cross-Validation RMSE (withheld RO)", fontsize=10)
    ax_cv.grid(True, axis="y", alpha=0.3, ls=":")

    # ── TR: Cross-validation improvement % ───────────────────────────────────
    ok["cv_imprv_pct"] = (
        (ok["verif_tec_rmse_prior"] - ok["verif_tec_rmse_post"])
        / ok["verif_tec_rmse_prior"].replace(0, np.nan) * 100.0
    ).fillna(0.0)
    ax_cv_imprv.bar(x, ok["cv_imprv_pct"], color=bar_colors, edgecolor="black", lw=0.5)
    ax_cv_imprv.axhline(0, color="black", lw=0.8, ls="--")
    ax_cv_imprv.set_xticks(x)
    ax_cv_imprv.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=5)
    ax_cv_imprv.set_ylabel("CV TEC RMSE improvement (%)", fontsize=9)
    ax_cv_imprv.set_title("Cross-Validation RMSE Improvement (prior → posterior)", fontsize=10)
    ax_cv_imprv.grid(True, axis="y", alpha=0.3, ls=":")

    # Shared colour bar for noise sigma (TL + TR)
    sm = cm.ScalarMappable(
        cmap=cmap_noise,
        norm=mcolors.BoundaryNorm(
            boundaries=(
                [noise_vals[0] - 1e-9]
                + [(noise_vals[i] + noise_vals[i + 1]) / 2 for i in range(len(noise_vals) - 1)]
                + [noise_vals[-1] + 1e-9]
            ),
            ncolors=n_noise,
        ),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_cv, ax_cv_imprv], orientation="vertical",
                        fraction=0.02, pad=0.01, label="Noise σ")
    cbar.set_ticks(noise_vals)
    cbar.set_ticklabels([str(v) for v in noise_vals], fontsize=7)

    # ── BL: Assimilation TEC RMSE improvement ────────────────────────────────
    ok["assim_imprv_pct"] = (
        (ok["prior_tec_rmse"] - ok["post_tec_rmse"])
        / ok["prior_tec_rmse"].replace(0, np.nan) * 100.0
    ).fillna(0.0)
    ax_assim_imprv.bar(x, ok["assim_imprv_pct"], color=bar_colors, edgecolor="black", lw=0.5)
    ax_assim_imprv.axhline(0, color="black", lw=0.8, ls="--")
    ax_assim_imprv.set_xticks(x)
    ax_assim_imprv.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=5)
    ax_assim_imprv.set_ylabel("Assimilation TEC RMSE improvement (%)", fontsize=9)
    ax_assim_imprv.set_title("Assimilation RMSE Improvement (prior → posterior)", fontsize=10)
    ax_assim_imprv.grid(True, axis="y", alpha=0.3, ls=":")

    # ── BR: Cross-validation RMSE heat map ───────────────────────────────────
    gauss_labels = sorted(ok["gauss_tag"].unique(), key=lambda s: s)
    noise_labels = sorted(ok["noise_sigma"].unique())

    heat = np.full((len(gauss_labels), len(noise_labels)), np.nan)
    for _, row in ok.iterrows():
        gi = gauss_labels.index(row["gauss_tag"])
        ni = noise_labels.index(row["noise_sigma"])
        heat[gi, ni] = row["verif_tec_rmse_post"]

    abs_max = np.nanmax(heat) or 1.0
    im = ax_heat.imshow(heat, aspect="auto", cmap="viridis_r",
                        vmin=0, vmax=abs_max, origin="upper")
    ax_heat.set_xticks(np.arange(len(noise_labels)))
    ax_heat.set_xticklabels([str(v) for v in noise_labels], fontsize=7, rotation=45)
    ax_heat.set_yticks(np.arange(len(gauss_labels)))
    ax_heat.set_yticklabels(gauss_labels, fontsize=6)
    ax_heat.set_xlabel("Noise σ", fontsize=9)
    ax_heat.set_ylabel("Gaussian smoothing (vert×horiz km)", fontsize=9)
    ax_heat.set_title("Cross-Validation TEC RMSE Heat Map (TECU)", fontsize=10)
    for gi in range(len(gauss_labels)):
        for ni in range(len(noise_labels)):
            v = heat[gi, ni]
            if not np.isnan(v):
                ax_heat.text(ni, gi, f"{v:.2f}", ha="center", va="center",
                             fontsize=5, color="white")
    fig.colorbar(im, ax=ax_heat, label="CV TEC RMSE (TECU)",
                 fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(save_dir, "sweep_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sweep summary figure → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def tune_covariance_main() -> None:
    print("=" * 70)
    print("  tune_covariance.py — Covariance Parameter Sweep (RO cross-validation)")
    print("=" * 70)

    if not os.path.exists(base_path):
        print(f"ERROR: base_path not found: {base_path}")
        return

    # ── Step 1: Scan, filter, and group metadata ──────────────────────────────
    meta = scan_metadata(base_path)
    if meta.empty:
        print("No valid podTc2 files found.  Exiting.")
        return
    meta_verif = filter_to_verif_region(meta)
    if meta_verif.empty:
        print("No occultations in verification region.  Exiting.")
        return
    meta_verif = assign_orbit_groups(meta_verif)

    # ── Step 2: Select orbit 1 only ──────────────────────────────────────────
    sorted_gkeys = sorted(meta_verif["group_key"].unique())
    if not sorted_gkeys:
        print("No orbit groups found.  Exiting.")
        return
    orbit1_key  = sorted_gkeys[0]
    orbit1_meta = meta_verif[meta_verif["group_key"] == orbit1_key].copy()
    print(f"\nOrbit 1 group key  : {orbit1_key}")
    print(f"Total occultations : {len(orbit1_meta)}")

    # ── Step 3: 50/50 RO split ────────────────────────────────────────────────
    assim_meta, verif_meta = _split_orbit_meta(orbit1_meta)
    print(f"  Assimilation set : {len(assim_meta)} occultations")
    print(f"  Verification set : {len(verif_meta)} occultations (withheld)")

    # ── Step 4: Build global EDP prior cache ──────────────────────────────────
    batch_date = pd.Timestamp(
        datetime.date(YYYY, 1, 1) + datetime.timedelta(days=DOY - 1)
    )
    global_edp_data_dir = f"./Data/Global_EDPS_{DOY}_{TYPE}/"
    print(f"\nBuilding global EDP cache for {batch_date.date()} …")
    global_edp_cache = build_daily_global_edps(
        batch_date, alt_grid,
        dLat=5.0, dLon=5.0,
        num_workers=num_workers,
        data_dir=global_edp_data_dir,
    )
    print("Global EDP cache ready.\n")

    os.makedirs(save_dir, exist_ok=True)

    # ── Step 5: Parameter sweep ────────────────────────────────────────────────
    param_grid = list(itertools.product(GAUSSIAN_SIGMA_GRID, NOISE_SIGMA_GRID))
    n_total    = len(param_grid)
    print(f"Sweeping {len(GAUSSIAN_SIGMA_GRID)} Gaussian settings × "
          f"{len(NOISE_SIGMA_GRID)} noise levels = {n_total} combinations …\n")

    rows = []

    for run_idx, (gauss, noise) in enumerate(param_grid):
        tag      = _param_tag(gauss, noise)
        label    = _param_label(gauss, noise)
        run_save = os.path.join(save_dir, tag)
        os.makedirs(run_save, exist_ok=True)

        print(f"[{run_idx + 1:3d}/{n_total}] {label}")

        _apply_params(gauss, noise)

        try:
            res = process_group(
                group_key        = orbit1_key,
                group_meta       = assim_meta,
                alt_grid         = alt_grid,
                global_edp_cache = global_edp_cache,
                generate_plots   = True,
                save_dir         = run_save,
                run_sequential   = False,
                **kf_config,
            )
        except Exception as exc:
            print(f"    [ERROR] process_group failed: {exc}")
            res = {
                "status":               f"Error: {exc}",
                "prior_tec_rmse":       np.nan,
                "post_tec_rmse":        np.nan,
                "joint_post_tec_rmse":  np.nan,
                "time_window":          orbit1_key,
            }

        # Arc innovation diagnostic for the assimilation arcs
        if res.get("status") == "Success":
            try:
                _kf_stats = _arc_stats_from_tec_slices(
                    tec_slices = res.get("joint_tec_slices", res.get("tec_slices", [])),
                    clean_list = res.get("clean_list", []),
                    sat_ids    = res.get("sat_ids", []),
                )
                if _kf_stats["arc_labels"]:
                    _plot_arc_innovation_diagnostic(
                        arc_labels     = _kf_stats["arc_labels"],
                        arc_prior_mean = _kf_stats["arc_prior_mean"],
                        arc_post_mean  = _kf_stats["arc_post_mean"],
                        arc_prior_rmse = _kf_stats["arc_prior_rmse"],
                        arc_post_rmse  = _kf_stats["arc_post_rmse"],
                        arc_lats       = _kf_stats["arc_lats"],
                        arc_lons       = _kf_stats["arc_lons"],
                        all_prior      = _kf_stats["all_prior"],
                        all_post_main  = _kf_stats["all_post"],
                        group_key      = orbit1_key,
                        save_dir       = run_save,
                        filter_name    = "KF",
                        prior_rmse     = float(res.get("prior_tec_rmse",
                                               res.get("joint_post_tec_rmse", np.nan))),
                        post_rmse      = float(res.get("joint_post_tec_rmse",
                                               res.get("post_tec_rmse", np.nan))),
                    )
            except Exception as _exc:
                print(f"    [warn] Arc innovation plot failed: {_exc}")

        # Cross-validation metrics on withheld RO data
        cv = _verif_tec_rmse(res, verif_meta, alt_grid)

        # Cross-validation TEC plot (withheld occultations)
        _plot_verif_all_tec(cv, orbit1_key, label, run_save)

        gauss_tag = "noSmooth" if gauss is None else f"g{gauss[0]}h{gauss[1]}ll"

        row = {
            "run_idx":               run_idx + 1,
            "param_tag":             tag,
            "param_label":           label,
            "gauss_tag":             gauss_tag,
            "gauss_h_km":            gauss[0] if gauss else None,
            "gauss_ll_km":           gauss[1] if gauss else None,
            "noise_sigma":           noise,
            "status":                res.get("status", "Unknown"),
            "prior_tec_rmse":        res.get("prior_tec_rmse",      np.nan),
            "post_tec_rmse":         res.get("joint_post_tec_rmse", np.nan),
            "verif_tec_rmse_prior":  cv["verif_tec_rmse_prior"],
            "verif_tec_rmse_post":   cv["verif_tec_rmse_post"],
            "n_verif_rays":          cv["n_verif_rays"],
            "tec_rmse_imprv_pct":    np.nan,
            "cv_imprv_pct":          np.nan,
        }

        if not np.isnan(row["prior_tec_rmse"]) and row["prior_tec_rmse"] > 0:
            row["tec_rmse_imprv_pct"] = (
                (row["prior_tec_rmse"] - row["post_tec_rmse"])
                / row["prior_tec_rmse"] * 100.0
            )
        if not np.isnan(cv["verif_tec_rmse_prior"]) and cv["verif_tec_rmse_prior"] > 0:
            row["cv_imprv_pct"] = (
                (cv["verif_tec_rmse_prior"] - cv["verif_tec_rmse_post"])
                / cv["verif_tec_rmse_prior"] * 100.0
            )

        rows.append(row)
        print(f"    status={row['status']}  "
              f"assim prior={row['prior_tec_rmse']:.3f}  post={row['post_tec_rmse']:.3f}  "
              f"cv prior={row['verif_tec_rmse_prior']:.3f}  cv post={row['verif_tec_rmse_post']:.3f} TECU")

    # ── Step 6: Write CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    col_order = [
        "run_idx", "param_tag", "param_label",
        "gauss_tag", "gauss_h_km", "gauss_ll_km", "noise_sigma",
        "status",
        "prior_tec_rmse", "post_tec_rmse", "tec_rmse_imprv_pct",
        "verif_tec_rmse_prior", "verif_tec_rmse_post", "cv_imprv_pct",
        "n_verif_rays",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = os.path.join(save_dir, f"cov_sweep_{YYYY}_{DOY}.csv")
    df.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"\nResults CSV → {csv_path}")

    # ── Step 7: Print ranked summary ──────────────────────────────────────────
    ok = df[df["status"] == "Success"].copy()
    if not ok.empty:
        print("\n── Top 5 by cross-validation TEC RMSE (best first) ──")
        ranked_cv = ok.sort_values("verif_tec_rmse_post")
        for _, r in ranked_cv.head(5).iterrows():
            print(f"  [{r['run_idx']:3d}] {r['param_tag']:<32}  "
                  f"CV RMSE={r['verif_tec_rmse_post']:.3f}  "
                  f"CV imprv={r['cv_imprv_pct']:+.1f}%  "
                  f"assim imprv={r['tec_rmse_imprv_pct']:+.1f}%")

        print("\n── Top 5 by cross-validation RMSE improvement (best first) ──")
        ranked_imprv = ok.sort_values("cv_imprv_pct", ascending=False)
        for _, r in ranked_imprv.head(5).iterrows():
            print(f"  [{r['run_idx']:3d}] {r['param_tag']:<32}  "
                  f"CV imprv={r['cv_imprv_pct']:+.1f}%  "
                  f"CV RMSE={r['verif_tec_rmse_post']:.3f} TECU")

    # ── Step 8: Sweep summary figure ──────────────────────────────────────────
    _plot_sweep_summary(df, save_dir)

    print("\nParameter sweep complete.  All figures and CSV written.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tune_covariance_main()
