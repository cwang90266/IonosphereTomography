#!/usr/bin/env python3
"""Re-run of the ISR_Full_Scale_No_Boost/ISR_Full_Scale_Boosted pair
(2026-10-06), now that two real bugs/findings from reviewing that run's
results have been fixed in the codebase:
  - Parameterized_EDPSamples.reconstruction_error() no longer redundantly
    re-fits ANCHOR's Chapman model a second time (was already fixed
    before the original run, unaffected here).
  - CycleConfig.diagonal_boost_horizontal_scale_km default changed
    200 -> 500km, after a real 3-parameter sweep (scored against real
    ISR ground truth) found the *original* run's horizontal scale (200,
    the then-default) was the actual cause of a 1-3 order-of-magnitude
    regression against real ISR density, not the boost amplitude --
    amplitude=0.5 at the corrected h=500 default should instead be a
    genuine win on both the real RO+IGS TEC fit and the real ISR match.
This script intentionally changes nothing else (same amplitude=0.5, same
everything) so the only difference from the original pair is this one
fix, isolating its real effect at full production scale across all 4
styles. Output goes to new _v2 folders -- the original pair is kept
as the "before" comparison, not overwritten.

Full production-scale end-to-end run, mirroring
Assimilation_Cycle_Integration_Plan.md step22_taper_single_batch_boosted /
step23_no_boost_single_batch exactly (same real window/ROI/grid/ensemble
size/single-batch real RO+IGS data), but with the 4th default style
(PCA_1D_10ex_ISR) and the new ISR comparison tooling (per-style and
cross-style Plot A/B/pooled RMSE) added on top.

Config, confirmed against the real step22/step23 artifacts before this
run (saved ensemble netCDF attrs, edp_samples.nc attrs):
  - real window: 2025-11-18T10:00:00 - 11:00:00 UTC, Tromso ROI
    (69.6N, 19.2E, radius_km=2000 observation capture, grid_radius_deg=18)
  - n_ensemble=2000, altitude_grid=90-900km/10km (82 levels),
    horizontal_resolution_deg=2.5 (161 geo points)
  - default_iri_spread_kwargs() (matches step22/23's saved attrs exactly:
    hour=3, f107=30, ap=30, ig=12, rz=12)
  - single batch falls out naturally (38 real entries < default
    batch_size=200), matching step22/23's "single_batch" naming with no
    special override needed
  - obs_sigma=3.0: not recoverable from step22/23's saved metadata
    directly, so taken from this project's own established tuned value
    for this exact real dataset (Assimilation_Cycle_Integration_Plan.md's
    real-data convergence investigation)
  - diagonal boost: amplitude=0.5, log_space=True, default taper
    (400/700km, floor 0.1) -- the confirmed production recommendation;
    step22's name ("taper") confirms the taper feature was in use

The real RO+IGS observations are the same already-prepared real files
used throughout this project's later sessions (step12_all_styles), not
rebuilt from raw RINEX/podTc2 here -- same real geometry/data, just
avoiding the multi-minute IGS reprocessing cost a second time.

Runs "no boost" first (also builds+saves the real 2000-member base
ensemble), then "boosted" reusing that exact same base ensemble via
edp_samples_path -- so the only difference between the two conditions is
the boost itself, not an independent IRI2020 draw.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

from Assimilation_Cycle.cycle_config import CycleConfig, default_styles, default_hyper_params_by_style
from Assimilation_Cycle.package_run import run_package

STEP12 = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"
OUT_ROOT = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle")
OUT_NOBOOST = OUT_ROOT / "ISR_Full_Scale_No_Boost_v2"
OUT_BOOSTED = OUT_ROOT / "ISR_Full_Scale_Boosted_v2"


def main():
    cfg_base = CycleConfig(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
        grid_radius_deg=18.0, horizontal_resolution_deg=2.5,
        n_ensemble=2000,
        ro_observations_path=STEP12 / "ro_observations.nc",
        igs_observations_path=STEP12 / "igs_observations.nc",
        isr_file_path=ISR_FILE, isr_pca_basis_path=ISR_BASIS,
        obs_sigma=3.0,
        analysis_rng_seed=777,
        styles=default_styles(),
        hyper_params_by_style=default_hyper_params_by_style(),
    )

    print("=== Run 1/2: no boost (also builds+saves the real base ensemble) ===")
    t0 = time.time()
    cfg_noboost = replace(cfg_base, output_dir=OUT_NOBOOST, label="isr_full_no_boost_v2")
    result_noboost = run_package(cfg_noboost)
    print(f"no-boost run complete in {time.time() - t0:.1f}s")
    for style, (cycle_result, n_state, wall_time) in result_noboost.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU")

    base_edp_path = OUT_NOBOOST / "isr_full_no_boost_v2_edp_samples.nc"
    assert base_edp_path.exists(), f"expected base ensemble at {base_edp_path}"

    print("\n=== Run 2/2: boosted (amplitude=0.5, log_space=True, reusing the same base ensemble) ===")
    t0 = time.time()
    cfg_boosted = replace(
        cfg_base, output_dir=OUT_BOOSTED, label="isr_full_boosted_v2",
        edp_samples_path=base_edp_path,
        diagonal_boost_amplitude=0.5, diagonal_boost_log_space=True, diagonal_boost_rng_seed=888,
        # diagonal_boost_horizontal_scale_km intentionally NOT set here --
        # picks up CycleConfig's new default (500, changed from 200) automatically.
    )
    result_boosted = run_package(cfg_boosted)
    print(f"boosted run complete in {time.time() - t0:.1f}s")
    for style, (cycle_result, n_state, wall_time) in result_boosted.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU")

    print("\n=== Done. Outputs: ===")
    print(f"  no boost: {OUT_NOBOOST}")
    print(f"  boosted:  {OUT_BOOSTED}")


if __name__ == "__main__":
    main()
