#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_verification_single.py — Single-occultation verification for GN04 G23 at ~2.62 UTC.

Runs the same Millstone Hill ISR verification pipeline as demo_verification.py
but restricts processing to one specific occultation: GN04 × G23, DOY 154 2025,
TEC-max near 02:37 UTC (2.62 h).  Produces the joint KF diagnostic plot
(_plot_group) with the ISR truth overlay, identical to what demo_verification_main
generates for the full day.

Run from the project root:
    python demo_verification_single.py
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
import datetime

from demo import build_daily_global_edps

# Re-use helpers from demo_group and demo_verification
import demo_group as _demo_group
from demo_group import scan_metadata, process_group, WINDOW_MINUTES
from demo_verification import (
    load_isr_profiles,
    millstone_vertex_idx,
    plot_isr_profile_comparison,
    _process_verif_group,
    _patched_region_bounding_box,
    _patched_plot_group,
)
import demo_verification as _demo_verif

# ── Apply the same module-level patches as demo_verification ─────────────────
_demo_group.region_bounding_box = _patched_region_bounding_box
_demo_group._plot_group         = _patched_plot_group

# ── Use the full VERIF_MH bbox — skip subset_union_triangles trimming ─────────
# For a single occultation the union-triangle step shrinks the mesh to a narrow
# band around the one raypath.  Replacing subset_union_triangles with a no-op
# keeps the full eds_bbox mesh (the entire VERIF_MH 30°×60° box).
from EDPSamples.edp_samples import EDPSamples as _EDPSamples
_EDPSamples.subset_union_triangles = lambda self, *args, **kwargs: self

# ─────────────────────────────────────────────────────────────────────────────
# Target occultation
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SPACECRAFT = "GN04"
TARGET_PRN        = "G23"
TARGET_FILE       = (
    "podTc2_GN04.2025.154.02.14.0025.G23.01_0000.0001_nc"
)

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── User-configurable settings ─────────────────────────────────────────────
    DOY  = 154
    YYYY = 2025
    base_path = (
        f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
        f"{YYYY}.{DOY}/"
    )
    alt_grid    = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
    TYPE        = "log"
    save_dir    = "./Figures/Verification/Single/"
    num_workers = 12
    kf_config   = {"measurement_err": 1.0, "relaxation": 0.99}

    isr_files = [
        "./DataFiles/EDPS/mlh250603m.002.nc",
    ]
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(base_path):
        print(f"ERROR: base_path not found: {base_path}")
        sys.exit(1)

    target_path = os.path.join(base_path, TARGET_FILE)
    if not os.path.exists(target_path):
        print(f"ERROR: target file not found:\n  {target_path}")
        sys.exit(1)

    print("=" * 65)
    print("  demo_verification_single.py")
    print(f"  Target : {TARGET_SPACECRAFT} × {TARGET_PRN}  (~2.62 UTC)")
    print(f"  File   : {TARGET_FILE}")
    print("=" * 65)

    # ── Step 1: Build a single-row metadata DataFrame for the target file ─────
    meta_full = scan_metadata(base_path)

    # Filter to just the target file by filename
    meta_single = meta_full[meta_full["filename"] == TARGET_FILE].copy()
    if meta_single.empty:
        print(f"ERROR: '{TARGET_FILE}' not found in metadata scan.")
        print("  Check the filename or that the file exists in base_path.")
        sys.exit(1)

    # Force the region to VERIF_MH so the bbox patch applies
    meta_single["region"]     = "VERIF_MH"
    meta_single["group_key"]  = meta_single["time_window"] + "__VERIF_MH"

    print(f"\n  Single-file metadata row:")
    print(f"    spacecraft  : {meta_single['spacecraft'].iloc[0]}")
    print(f"    date        : {meta_single['date'].iloc[0]}")
    print(f"    lat/lon     : {meta_single['lat'].iloc[0]:.2f}°N  "
          f"{meta_single['lon'].iloc[0]:.2f}°E")
    print(f"    group_key   : {meta_single['group_key'].iloc[0]}")

    # ── Step 2: Build global EDP prior cache ──────────────────────────────────
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

    # ── Step 3: Load ISR truth ────────────────────────────────────────────────
    isr_profiles: list[dict] = []
    if isr_files:
        isr_profiles = load_isr_profiles(isr_files)
    else:
        print("  [ISR] No ISR files configured — skipping ISR overlay.")

    # Make ISR profiles available to the _plot_group patch
    _demo_verif._isr_profiles_for_patch = isr_profiles

    # ── Step 4: Run the joint KF for this single occultation ──────────────────
    os.makedirs(save_dir, exist_ok=True)

    gk = meta_single["group_key"].iloc[0]
    print(f"\nProcessing group: {gk}")

    res = _process_verif_group(
        group_key        = gk,
        group_meta       = meta_single,
        alt_grid         = alt_grid,
        global_edp_cache = global_edp_cache,
        generate_plots   = True,
        save_dir         = save_dir,
        **kf_config,
    )

    print(f"\n  Status : {res.get('status')}")

    # ── Step 5: ISR profile comparison plot ───────────────────────────────────
    if isr_profiles and res.get("status") == "Success":
        eds  = res.get("eds_occ")
        if eds is not None:
            verts  = eds.geolocation
            idx_mh = millstone_vertex_idx(verts)
            n_geo  = verts.shape[0]
            n_h    = len(alt_grid)
            pr_mh  = np.asarray(res["prior_edp_3d"]).reshape(n_h, n_geo)[:, idx_mh]
            po_mh  = np.asarray(res["joint_post_edp_3d"]).reshape(n_h, n_geo)[:, idx_mh]

            win = res["time_window"]
            try:
                hhmm   = win.split("_")[-1]
                h_mid  = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
                half   = WINDOW_MINUTES / 120.0
                isr_win = [p for p in isr_profiles if abs(p["hour_utc"] - h_mid) < half]
                if not isr_win:
                    isr_win = [min(
                        isr_profiles,
                        key=lambda p: min(
                            abs(p["hour_utc"] - h_mid),
                            24 - abs(p["hour_utc"] - h_mid),
                        ),
                    )]
            except Exception:
                isr_win = isr_profiles[:1]

            try:
                plot_isr_profile_comparison(
                    isr_profiles    = isr_win,
                    prior_edp_at_mh = pr_mh,
                    post_edp_at_mh  = po_mh,
                    alt_grid        = alt_grid,
                    group_key       = gk,
                    save_dir        = save_dir,
                )
            except Exception as exc:
                print(f"  [warn] ISR comparison plot failed: {exc}")

    print(f"\nDone.  Figures written to: {save_dir}")
