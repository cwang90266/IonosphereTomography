#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_verification1D.py — Millstone Hill ISR verification using 1-D spherically-
symmetric tomography.

Instead of building a full 3-D geographic mesh (as demo_verification.py does),
this script uses a single-point IRI prior at the Millstone Hill location and
assembles H matrices that integrate each occultation ray purely through altitude
shells (the n_geo=1 path in Ionosphere_Tomography_Inverter).  This is equivalent
to assuming spherical symmetry around the ISR site and inverting all occultations
in a time window jointly against a single vertical EDP.

Key differences from demo_verification.py:
  • Prior : single-point IRI at (ISR_LAT, ISR_LON_W) per UTC hour — no global mesh.
  • H matrix : altitude-only integration (n_geo=1); horizontal position is ignored.
  • Grouping : same orbit-based grouping as demo_verification.py.
  • All occultations in a window are assimilated regardless of distance from MH
    (spherical symmetry assumption).
  • ISR comparison plots are identical to demo_verification.py.

Run from the project root:
    python demo_verification1D.py
"""

import sys
import os
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
import time
import gc
import datetime

from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from EDPSamples.edp_samples import EDPSamples
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
)
from demo import extract_robust_f2_peak

# ── Re-use helpers from demo_group ───────────────────────────────────────────
import demo_group as _demo_group
from demo_group import (
    scan_metadata,
    GAUSSIAN_COV_SIGMA,
    _save_stats_csv,
    WINDOW_MINUTES,
)

try:
    from demo_group import load_conPhs
except ImportError:
    load_conPhs = None

# ── Re-use all ISR helpers from demo_verification (no duplication) ────────────
from demo_verification import (
    ISR_LAT, ISR_LON_W, ISR_LON,
    VERIF_LAT_MIN, VERIF_LAT_MAX, VERIF_LON_MIN, VERIF_LON_MAX,
    HALF_LAT, HALF_LON,
    filter_to_verif_region,
    assign_orbit_groups,
    load_isr_profiles,
    compute_isr_tec,
    millstone_vertex_idx,
    interp_to_isr_alt,
    plot_isr_profile_comparison,
    plot_isr_summary,
    plot_verif_region_map,
    _haversine_km,
)


# ─────────────────────────────────────────────────────────────────────────────
# §A  Build single-point IRI prior cache at Millstone Hill
# ─────────────────────────────────────────────────────────────────────────────

def _build_hourly_1d_edp(args: tuple) -> tuple:
    """
    Worker: generate and save one hour's single-point IRI at Millstone Hill.
    Returns (hr, nc_path).
    """
    import gc
    import numpy as np
    import pandas as pd
    from EDPSamples.edp_samples import EDPSamples

    (hr, alt_grid, f107_center, ap_center, ig12_center, rz12_center,
     n_mc, data_dir, date_tag, date_ymd) = args

    nc_path = os.path.join(data_dir, f"EDP1D_MH_{date_tag}_{hr:02d}h.nc")

    if os.path.exists(nc_path):
        try:
            EDPSamples.fromNetCDF(nc_path)
            return hr, nc_path
        except Exception:
            pass

    np.random.seed(42 + hr)
    mc_df = pd.DataFrame({
        "hour": np.full(n_mc, float(hr)),
        "f107": np.random.normal(loc=f107_center, scale=10, size=n_mc).clip(70, 250),
        "ap":   np.random.normal(loc=ap_center,   scale=5,  size=n_mc).clip(0,  400),
        "ig12": np.random.normal(loc=ig12_center, scale=10, size=n_mc).clip(50, 200),
        "rz12": np.random.normal(loc=rz12_center, scale=10, size=n_mc).clip(50, 200),
    })
    mc_df.iloc[0] = {
        "hour": float(hr), "f107": f107_center, "ap": ap_center,
        "ig12": ig12_center, "rz12": rz12_center,
    }

    eds = EDPSamples(
        DateTime=f"{date_ymd} {hr:02d}:00:00",
        geo_type="Point",
        altitude=alt_grid,
        sampling_parameters=mc_df,
        evaluate_iri=1,
        Lat=ISR_LAT,
        Lon=ISR_LON_W,
    )
    eds.saveNetCDF(nc_path)
    del eds
    gc.collect()

    return hr, nc_path


def build_1d_prior_cache(
    date: pd.Timestamp,
    alt_grid: np.ndarray,
    n_mc: int = 50,
    data_dir: str = "./Data/EDP1D_MH/",
    num_workers: int = 8,
) -> dict:
    """
    Build (or load from NetCDF) a single-point IRI prior at Millstone Hill for
    all 24 hours of `date`.  Returns dict keyed by hour (0-23) → EDPSamples
    with n_geo=1 at (ISR_LAT, ISR_LON_W).
    """
    from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs
    import multiprocessing as mp

    os.makedirs(data_dir, exist_ok=True)

    inp = IRI_Sample_Inputs(date.strftime("%Y-%m-%d 12:00:00"))
    f107_center = float(inp.apf107['f107'][inp.current_idx_f107])
    ap_center   = float(inp.apf107['iapda'][inp.current_idx_f107])
    ig12_center = float(inp.ig_rz['ig'][inp.current_idx_igrz])
    rz12_center = float(inp.ig_rz['rz'][inp.current_idx_igrz])

    print(f" -> 1-D MH IRI inputs for {date.date()}: "
          f"F10.7={f107_center:.1f}, Ap={ap_center:.0f}, "
          f"IG12={ig12_center:.1f}, Rz12={rz12_center:.1f}")

    date_tag = date.strftime("%Y%m%d")
    date_ymd = date.strftime("%Y-%m-%d")

    tasks = [
        (hr, alt_grid, f107_center, ap_center, ig12_center, rz12_center,
         n_mc, data_dir, date_tag, date_ymd)
        for hr in range(24)
    ]

    nc_paths = {}
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for i, (hr, nc_path) in enumerate(
            pool.imap_unordered(_build_hourly_1d_edp, tasks)
        ):
            nc_paths[hr] = nc_path
            print(f"    [Hour {hr:02d}] 1-D MH EDP ready ({i+1}/24)")

    hourly_grids = {hr: EDPSamples.fromNetCDF(nc_paths[hr]) for hr in range(24)}
    print(" -> All 24 hourly 1-D MH EDP priors loaded.")
    return hourly_grids


# ─────────────────────────────────────────────────────────────────────────────
# §B  Process a single orbit group with 1-D spherically symmetric KF
# ─────────────────────────────────────────────────────────────────────────────

def process_group_1d(
    group_key:        str,
    group_meta:       pd.DataFrame,
    alt_grid:         np.ndarray,
    prior_cache_1d:   dict,
    measurement_err:  float = 10.0,
    relaxation:       float = 0.99,
    save_dir:         str   = "./Figures/Verification1D/",
    conphs_base_dir:  str   = None,
    conphs_max_rays:  int   = 200,
    num_ray_segments: int   = 500,
) -> dict:
    """
    Assimilate all occultations in a time-window group against a single-point
    1-D IRI prior at Millstone Hill (spherical symmetry assumption).

    The H matrix for each ray is computed via the n_geo=1 path in
    Ionosphere_Tomography_Inverter, which integrates only through altitude bins
    and ignores horizontal position.  All occultations in the window contribute
    to updating the single vertical EDP regardless of their horizontal distance
    from the ISR site.

    Returns a dict matching the schema used by demo_group.process_group and
    demo_verification._process_verif_group, so that the same ISR comparison
    and incremental-KF plots work without modification.
    """
    t_start  = time.time()
    n_occ    = len(group_meta)
    win_key  = group_meta["time_window"].iloc[0]
    region   = group_meta["region"].iloc[0]

    print(f"\n{'─'*60}")
    print(f"  Group (1D) : {group_key}")
    print(f"  Window     : {win_key}  |  {n_occ} occultation(s)")
    print(f"{'─'*60}")

    result = {
        "group_key":             group_key,
        "region":                region,
        "time_window":           win_key,
        "n_occultations":        n_occ,
        "files":                 list(group_meta["filename"]),
        "lats":                  list(group_meta["lat"]),
        "lons":                  list(group_meta["lon"]),
        "status":                "Failed",
        "prior_tec_rmse":        np.nan,
        "post_tec_rmse":         np.nan,
        "joint_post_tec_rmse":   np.nan,
        "plot_path":             None,
        "joint_plot_path":       None,
        "comparison_plot_path":  None,
        "eds_occ":               None,
        "prior_edp_3d":          None,
        "post_edp_3d":           None,
        "joint_post_edp_3d":     None,
        "clean_list":            [],
        "sat_ids":               [],
    }

    try:
        # ── Parse occultations ────────────────────────────────────────────────
        parsed_list  = []
        file_labels  = []
        conphs_list  = []
        for _, row in group_meta.iterrows():
            data = parse_podTc2_nc_file(row["full_path"])
            if data is None:
                print(f"    [skip] {row['filename']} — parse returned None")
                continue
            parsed_list.append(data)
            file_labels.append(f"{row['spacecraft']} {row['date'].strftime('%H:%M')}")

            if conphs_base_dir is not None and load_conPhs is not None:
                cp = load_conPhs(row["full_path"], conPhs_base_dir=conphs_base_dir)
                conphs_list.append(cp)
                if cp is not None:
                    print(f"    [conPhs] Loaded for {row['filename']}")
            else:
                conphs_list.append(None)

        if not parsed_list:
            result["status"] = "No Valid Files"
            return result

        # ── Build clean ray lists ─────────────────────────────────────────────
        clean_list    = []
        clean_sat_ids = []
        for i, data in enumerate(parsed_list):
            _, _, tang_raw = rayTangent(data["LEO"], data["GNSS"], units="km")
            tang_km  = tang_raw * 1e-3
            meas_tec = data.get("TEC_podTc2", data.get("TEC", np.zeros_like(tang_km)))
            valid    = ~np.isnan(meas_tec) & (meas_tec > 0)

            leo_id   = str(data.get("leo_id", "??")).strip()
            con_id   = str(data.get("conid",  "?")).strip()
            prn_num  = str(data.get("prn_id", "??")).strip()

            if valid.sum() < 50:
                print(f"    [skip] {file_labels[i]} — only {valid.sum()} valid rays")
            else:
                clean_list.append({
                    "tec":        np.asarray(meas_tec[valid], dtype=np.float64).flatten(),
                    "tangent_km": tang_km[valid].flatten(),
                    "LEO":        data["LEO"][:,  valid],
                    "GNSS":       data["GNSS"][:, valid],
                    "tec_type":   "absolute",
                })
                clean_sat_ids.append((leo_id, f"{con_id}{prn_num}"))

            # Optionally add conPhs relative-TEC arc
            cp = conphs_list[i]
            if cp is not None:
                rel_tec  = cp.get("rel_TEC", np.array([]))
                cp_valid = np.isfinite(rel_tec) & (cp["tangent_alt_km"] > 0)
                n_cp     = int(cp_valid.sum())
                if n_cp >= 50:
                    if n_cp > conphs_max_rays:
                        stride  = int(np.ceil(n_cp / conphs_max_rays))
                        dec_idx = np.where(cp_valid)[0][::stride]
                        dec_mask = np.zeros(len(rel_tec), dtype=bool)
                        dec_mask[dec_idx] = True
                    else:
                        dec_mask = cp_valid
                    clean_list.append({
                        "tec":        np.asarray(rel_tec[dec_mask], dtype=np.float64).flatten(),
                        "tangent_km": cp["tangent_alt_km"][dec_mask].flatten(),
                        "LEO":        cp["LEO"][:,  dec_mask],
                        "GNSS":       cp["GNSS"][:, dec_mask],
                        "tec_type":   "relative",
                    })
                    clean_sat_ids.append((leo_id, f"{con_id}{prn_num}"))

        if not clean_list:
            result["status"] = "Insufficient Rays"
            return result

        result["clean_list"] = clean_list
        result["sat_ids"]    = clean_sat_ids

        # ── Select 1-D prior at group's median hour ───────────────────────────
        all_dates    = [d["date"] for d in parsed_list]
        median_ts    = pd.Timestamp(np.median([d.value for d in all_dates]))
        profile_hour = median_ts.hour
        eds_1d       = prior_cache_1d[profile_hour]   # n_geo=1 EDPSamples at MH

        result["eds_occ"] = eds_1d

        n_height = len(alt_grid)
        # n_geo=1 by construction; MH vertex is always index 0
        print(f"  1-D prior at hour {profile_hour:02d}  |  "
              f"{len(clean_list)} arc(s)  |  n_height={n_height}")

        # ── Build inverter and H matrices ─────────────────────────────────────
        n_rel_arcs = sum(1 for cl in clean_list if cl["tec_type"] == "relative")
        inverter = Ionosphere_Tomography_Inverter(
            EDPSam=eds_1d, meanscale=1, topside_prior_floor_tecu=1.0,
            n_rel_arcs=n_rel_arcs, topside_alpha=0.0,
            gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
        )

        H_blocks = inverter.get_observation_operator_batch(
            clean_list, num_segments=num_ray_segments
        )

        tec_obs   = [cl["tec"] for cl in clean_list]
        H_joint   = np.vstack(H_blocks).astype(np.float32)
        obs_joint = np.concatenate(tec_obs).astype(np.float64)
        print(f"  H shape: {H_joint.shape}  ({len(obs_joint)} total rays)")

        # ── Prior TEC ─────────────────────────────────────────────────────────
        prior_flat  = inverter.attrs["initial_edps_mean"]   # (n_sv, 1) for n_geo=1
        x_top_prior = inverter.attrs["x_top_prior"]         # (1,)
        _n_sv       = inverter.attrs["n_state_vars"]
        _n_sv_aug   = inverter.attrs["n_state_vars_aug"]

        prior_tec = (
            H_joint[:, :_n_sv] @ prior_flat
            + H_joint[:, _n_sv:_n_sv_aug] @ x_top_prior[:, None]
        ).flatten()

        prior_rmse = float(np.sqrt(np.nanmean((prior_tec - obs_joint) ** 2)))
        result["prior_tec_rmse"] = prior_rmse
        print(f"  Prior TEC RMSE  : {prior_rmse:.3f} TECU")

        # Store 1-D prior as shape (n_height, 1) for consistency with 3-D callers
        prior_edp_1d = prior_flat.reshape(n_height, 1).copy()
        result["prior_edp_3d"] = prior_edp_1d

        # ── Joint KF update (all arcs at once) ───────────────────────────────
        post_flat = inverter.assimilate(
            obs=obs_joint, obs_operator=H_joint,
            relaxation=relaxation, measurement_err=measurement_err,
        )

        post_flat_arr = np.asarray(post_flat).flatten()          # (n_sv,) grid EDP
        x_top_post    = inverter.x_top_tecu.flatten()             # (1,)  topside TECU
        post_tec = (
            H_joint[:, :_n_sv]          @ post_flat_arr[:, None]
            + H_joint[:, _n_sv:_n_sv_aug] @ x_top_post[:, None]
        ).flatten()
        post_rmse = float(np.sqrt(np.nanmean((post_tec - obs_joint) ** 2)))
        result["post_tec_rmse"]       = post_rmse
        result["joint_post_tec_rmse"] = post_rmse
        print(f"  Posterior TEC RMSE : {post_rmse:.3f} TECU  "
              f"(Δ = {prior_rmse - post_rmse:+.3f})")

        post_edp_1d = np.asarray(post_flat).reshape(n_height, 1)
        result["post_edp_3d"]       = post_edp_1d
        result["joint_post_edp_3d"] = post_edp_1d

        result["status"] = "Success"

    except Exception as exc:
        import traceback
        print(f"  [ERROR] {group_key}: {exc}")
        traceback.print_exc()

    result["elapsed_s"] = time.time() - t_start
    return result


# ─────────────────────────────────────────────────────────────────────────────
# §C  Incremental KF (1-D version): add occultations one-by-one by distance
# ─────────────────────────────────────────────────────────────────────────────

def _run_incremental_kf_1d(
    res_full:        dict,
    alt_grid:        np.ndarray,
    measurement_err: float = 10.0,
    relaxation:      float = 0.99,
) -> tuple:
    """
    Sort occultations by great-circle distance from Millstone Hill and run
    cumulative joint 1-D KF updates with 1, 2, 3, … N occultations.

    Returns (prior_edp_mh, steps) where steps is a list of dicts
    matching the schema expected by _plot_incremental_convergence in
    demo_verification.py.
    """
    clean_list = res_full["clean_list"]
    eds_1d     = res_full["eds_occ"]
    sat_ids    = res_full.get("sat_ids", [])
    n_occ      = len(clean_list)

    if n_occ == 0:
        return np.array([]), []

    n_height = len(alt_grid)

    # Compute tangent-point lat/lon for each arc (for distance sorting)
    def _tangent_latlon(cl):
        try:
            i_tm   = int(np.argmax(cl["tec"]))
            leo_pt = cl["LEO"][:, i_tm]
            gns_pt = cl["GNSS"][:, i_tm]
            d      = gns_pt - leo_pt
            denom  = float(np.dot(d, d))
            if denom == 0:
                raise ValueError
            t_tp = -float(np.dot(leo_pt, d)) / denom
            tp   = leo_pt + t_tp * d
            r    = float(np.linalg.norm(tp))
            lat  = float(np.degrees(np.arcsin(tp[2] / r)))
            lon  = float(np.degrees(np.arctan2(tp[1], tp[0])))
            return lat, lon
        except Exception:
            return ISR_LAT, ISR_LON_W

    occ_meta = []
    for i, cl in enumerate(clean_list):
        tec_lat, tec_lon = _tangent_latlon(cl)
        dist  = _haversine_km(ISR_LAT, ISR_LON_W, tec_lat, tec_lon)
        label = sat_ids[i][1] if i < len(sat_ids) else f"occ{i}"
        occ_meta.append({"idx": i, "dist_km": dist, "label": label})

    occ_meta.sort(key=lambda m: m["dist_km"])

    # Build H matrices once for all arcs
    inverter_ref = Ionosphere_Tomography_Inverter(
        EDPSam=eds_1d, meanscale=1, topside_prior_floor_tecu=1.0,
        topside_alpha=0.0, gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
    )
    H_all = inverter_ref.get_observation_operator_batch(clean_list, num_segments=500)
    prior_flat = inverter_ref.attrs["initial_edps_mean"]
    prior_edp_mh = prior_flat.reshape(n_height).copy()  # scalar since n_geo=1

    steps = []
    for step, meta in enumerate(occ_meta):
        subset_idxs = [m["idx"] for m in occ_meta[:step + 1]]

        H_sub  = np.vstack([H_all[i] for i in subset_idxs]).astype(np.float32)
        obs_sub = np.concatenate([clean_list[i]["tec"] for i in subset_idxs]).astype(np.float64)

        n_rel = sum(1 for i in subset_idxs
                    if clean_list[i].get("tec_type") == "relative")
        inv = Ionosphere_Tomography_Inverter(
            EDPSam=eds_1d, meanscale=1, topside_prior_floor_tecu=1.0,
            topside_alpha=0.0, gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
            n_rel_arcs=n_rel,
        )
        if n_rel > 0:
            H_sub = np.hstack([H_sub, np.zeros((H_sub.shape[0], n_rel), dtype=np.float32)])

        post_flat = inv.assimilate(
            obs=obs_sub, obs_operator=H_sub,
            relaxation=relaxation, measurement_err=measurement_err,
        )
        edp_mh = np.asarray(post_flat).reshape(n_height)

        steps.append({
            "n_occ":    step + 1,
            "dist_km":  meta["dist_km"],
            "edp_mh":   edp_mh.copy(),
            "label":    meta["label"],
            "orig_idx": meta["idx"],
        })
        print(f"    [incr 1D KF] step {step+1}/{n_occ}: +{meta['label']} "
              f"({meta['dist_km']:.0f} km)  NmF2 = "
              f"{float(np.max(edp_mh)):.2e} m⁻³")

    return prior_edp_mh, steps


# ─────────────────────────────────────────────────────────────────────────────
# §D  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def demo_verification1D_main() -> None:
    """
    1-D spherically-symmetric Millstone Hill ISR verification pipeline.

    Steps:
      1.  Scan podTc2 metadata for the day.
      2.  Filter to the 30°×60° region centred on Millstone Hill.
      3.  Group by orbital pass.
      4.  Build single-point IRI prior cache at MH (n_geo=1 per UTC hour).
      5.  Process each orbit group with a 1-D joint KF update.
      6.  Compare posterior to ISR truth using the same plots as demo_verification.
    """

    # ── User-configurable settings ─────────────────────────────────────────────
    DOY  = 154
    YYYY = 2025
    base_path = (
        f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
        f"{YYYY}.{DOY}/"
    )
    alt_grid    = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
    save_dir    = "./Figures/Verification1D/"
    num_workers = 12
    kf_config   = {"measurement_err": 1.0, "relaxation": 0.99}

    conphs_base_dir = None
    # conphs_base_dir = (
    #     f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/conPhs/"
    #     f"{YYYY}.{DOY}/"
    # )
    conphs_max_rays = 200

    isr_files = [
        "./DataFiles/EDPS/mlh250603m.002.nc",
    ]
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(base_path):
        print(f"ERROR: base_path not found: {base_path}")
        return

    print("=" * 65)
    print("  demo_verification1D.py — 1-D Spherically-Symmetric Verification")
    print("=" * 65)
    print(f"  ISR site : {ISR_LAT:.2f}°N, {ISR_LON:.2f}°E ({ISR_LON_W:.2f}°)")
    print(f"  Patch    : lat [{VERIF_LAT_MIN:.1f}, {VERIF_LAT_MAX:.1f}]  "
          f"lon [{VERIF_LON_MIN:.1f}, {VERIF_LON_MAX:.1f}]")
    print(f"  Prior    : single-point IRI at MH (n_geo=1, spherical symmetry)")

    # ── Step 1: Scan and filter metadata ──────────────────────────────────────
    meta = scan_metadata(base_path)
    if meta.empty:
        print("No valid podTc2 files found.  Exiting.")
        return

    meta_verif = filter_to_verif_region(meta)
    if meta_verif.empty:
        print("No occultations found in the verification region.  Exiting.")
        return

    meta_verif = assign_orbit_groups(meta_verif)

    # ── Step 2: Region map ────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    plot_verif_region_map(meta_verif, save_dir, doy=DOY, year=YYYY)

    # ── Step 3: Build 1-D IRI prior cache at MH ───────────────────────────────
    batch_date = pd.Timestamp(
        datetime.date(YYYY, 1, 1) + datetime.timedelta(days=DOY - 1)
    )
    data_dir_1d = f"./Data/EDP1D_MH_{DOY}/"
    print(f"\nBuilding 1-D MH IRI prior cache for {batch_date.date()} …")
    prior_cache_1d = build_1d_prior_cache(
        batch_date, alt_grid,
        data_dir=data_dir_1d,
        num_workers=num_workers,
    )
    print("1-D prior cache ready.\n")

    # ── Step 4: Load ISR truth ─────────────────────────────────────────────────
    isr_profiles: list[dict] = []
    if isr_files:
        isr_profiles = load_isr_profiles(isr_files)
    else:
        print("  [ISR] No ISR files configured — skipping ISR comparison plots.")

    # ── Step 5: Process each orbit group ─────────────────────────────────────
    groups     = meta_verif.groupby("group_key", sort=True)
    group_keys = list(groups.groups.keys())
    print(f"\nProcessing {len(group_keys)} time-window group(s) …")

    all_results: list[dict] = []
    for g_idx, gk in enumerate(group_keys):
        print(f"\n[{g_idx + 1}/{len(group_keys)}]", end="")
        gm  = groups.get_group(gk)
        if conphs_base_dir is not None:
            gk_proc = f"{gk}_conPhs"
        else:
            gk_proc = gk

        res = process_group_1d(
            group_key       = gk_proc,
            group_meta      = gm,
            alt_grid        = alt_grid,
            prior_cache_1d  = prior_cache_1d,
            save_dir        = save_dir,
            conphs_base_dir = conphs_base_dir,
            conphs_max_rays = conphs_max_rays,
            **kf_config,
        )
        all_results.append(res)

        # Per-window ISR comparison (only if ISR data available)
        if isr_profiles and res.get("status") == "Success":
            eds  = res.get("eds_occ")
            if eds is not None:
                # n_geo=1: MH vertex is always index 0
                pr_mh = np.asarray(res["prior_edp_3d"]).reshape(len(alt_grid))
                po_mh = np.asarray(res["joint_post_edp_3d"]).reshape(len(alt_grid))

                win = res["time_window"]
                try:
                    hhmm   = win.split("_")[-1]
                    h_mid  = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
                    half   = WINDOW_MINUTES / 120.0
                    isr_win = [p for p in isr_profiles
                               if abs(p["hour_utc"] - h_mid) < half]
                    if not isr_win:
                        isr_win = [min(isr_profiles,
                                       key=lambda p: min(abs(p["hour_utc"] - h_mid),
                                                         24 - abs(p["hour_utc"] - h_mid)))]
                except Exception:
                    isr_win = isr_profiles[:1]

                try:
                    plot_isr_profile_comparison(
                        isr_profiles    = isr_win,
                        prior_edp_at_mh = pr_mh,
                        post_edp_at_mh  = po_mh,
                        alt_grid        = alt_grid,
                        group_key       = gk_proc,
                        save_dir        = save_dir,
                    )
                except Exception as exc:
                    print(f"  [warn] ISR comparison plot failed: {exc}")

                # Incremental KF: add occultations one-by-one by distance to MH
                if res.get("clean_list") and len(res["clean_list"]) > 1:
                    print(f"  Running incremental 1-D KF ({len(res['clean_list'])} arcs) …")
                    try:
                        from demo_verification import _plot_incremental_convergence
                        prior_mh_incr, incr_steps = _run_incremental_kf_1d(
                            res_full        = res,
                            alt_grid        = alt_grid,
                            measurement_err = kf_config.get("measurement_err", 10.0),
                            relaxation      = kf_config.get("relaxation", 0.99),
                        )
                        _plot_incremental_convergence(
                            prior_edp_mh = prior_mh_incr,
                            steps        = incr_steps,
                            isr_profiles = isr_win,
                            alt_grid     = alt_grid,
                            group_key    = gk_proc,
                            save_dir     = save_dir,
                        )
                    except Exception as exc_incr:
                        print(f"  [warn] Incremental 1-D KF failed: {exc_incr}")

    # ── Step 6: Statistics CSV ────────────────────────────────────────────────
    stats_csv = _save_stats_csv(all_results, YYYY, DOY)
    print(f"\nStats CSV saved → {stats_csv}")

    # ── Step 7: Summary ISR plot ───────────────────────────────────────────────
    if isr_profiles:
        # Adapt result dicts: plot_isr_summary expects (n_height, n_geo) arrays and
        # a geolocation with >= 1 row.  With n_geo=1 everything is already shape
        # (n_height, 1), and millstone_vertex_idx returns 0 correctly.
        try:
            plot_isr_summary(all_results, isr_profiles, alt_grid, save_dir)
        except Exception as exc:
            print(f"  [warn] ISR summary plot failed: {exc}")

    # ── Step 8: Console statistics ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  1-D Verification complete.  Statistics:")
    print("=" * 65)
    success = [r for r in all_results if r["status"] == "Success"]
    print(f"  Groups processed   : {len(success)} / {len(all_results)}")
    if success:
        rmse_pr = np.nanmean([r["prior_tec_rmse"] for r in success])
        rmse_po = np.nanmean([r["post_tec_rmse"]  for r in success])
        imprv   = (rmse_pr - rmse_po) / rmse_pr * 100.0 if rmse_pr > 0 else 0.0
        print(f"  Mean prior  TEC RMSE : {rmse_pr:.3f} TECU")
        print(f"  Mean post   TEC RMSE : {rmse_po:.3f} TECU")
        print(f"  Mean TEC improvement : {imprv:.1f} %")

    if isr_profiles:
        nm_bias_list, hm_bias_list = [], []
        for res in success:
            _po = res.get("joint_post_edp_3d")
            if _po is None:
                _po = res.get("post_edp_3d")
            po_mh_arr = np.asarray(_po) if _po is not None else None
            if po_mh_arr is None:
                continue
            po_mh = po_mh_arr.reshape(len(alt_grid))
            nm_po, hm_po = extract_robust_f2_peak(po_mh, alt_grid)
            nm_isr = float(np.nanmean([p["nm_f2"] for p in isr_profiles]))
            hm_isr = float(np.nanmean([p["hm_f2"] for p in isr_profiles]))
            if not np.isnan(nm_po):
                nm_bias_list.append(nm_po - nm_isr)
            if not np.isnan(hm_po):
                hm_bias_list.append(hm_po - hm_isr)

        if nm_bias_list:
            print(f"  NmF2 bias (post−ISR): {np.nanmean(nm_bias_list):.3e} m⁻³")
        if hm_bias_list:
            print(f"  hmF2 bias (post−ISR): {np.nanmean(hm_bias_list):.1f} km")

    print("\nAll figures written.  Done.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_verification1D_main()
