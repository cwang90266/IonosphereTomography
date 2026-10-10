#!/usr/bin/env python3
"""Re-run of the ORIGINAL ISR_Full_Scale_No_Boost/ISR_Full_Scale_Boosted
pair (2026-10-06), now that a real bug in the IRI2020 Fortran source
(irifun.for) has been found and fixed, and the executable rebuilt
(2026-10-08). Overwrites the original two folders -- those results were
generated with the buggy Fortran code and are no longer valid.

Deliberately preserves that pair's ORIGINAL experimental design
(amplitude=0.5, diagonal_boost_horizontal_scale_km=200 explicitly pinned
below) rather than picking up CycleConfig's now-current default of 500 --
this is what makes it a faithful re-run of the ORIGINAL pair specifically
(the h=200 vs h=500 comparison IS the pair-vs-_v2 comparison; collapsing
both pairs onto h=500 would destroy that). The companion script for the
_v2 pair (h=500, CycleConfig's current default, no override needed)
re-runs separately.

Everything else identical to the other full-scale scripts this session:
real 2025-11-18 10:00-11:00 Tromso window, n_ensemble=2000, 82-level/
161-geo grid, single batch, real RO+IGS data from step12_all_styles,
obs_sigma=3.0, all 4 default styles, full ISR comparison tooling.
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
OUT_NOBOOST = OUT_ROOT / "ISR_Full_Scale_No_Boost"
OUT_BOOSTED = OUT_ROOT / "ISR_Full_Scale_Boosted"


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

    print("=== Run 1/2: no boost (also builds+saves the real base ensemble, FIXED IRI2020) ===")
    t0 = time.time()
    cfg_noboost = replace(cfg_base, output_dir=OUT_NOBOOST, label="isr_full_no_boost")
    result_noboost = run_package(cfg_noboost)
    print(f"no-boost run complete in {time.time() - t0:.1f}s")
    for style, (cycle_result, n_state, wall_time) in result_noboost.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU")

    base_edp_path = OUT_NOBOOST / "isr_full_no_boost_edp_samples.nc"
    assert base_edp_path.exists(), f"expected base ensemble at {base_edp_path}"

    print("\n=== Run 2/2: boosted (amplitude=0.5, log_space=True, horizontal_scale_km=200 PINNED "
          "to preserve the original pair's design, reusing the same base ensemble) ===")
    t0 = time.time()
    cfg_boosted = replace(
        cfg_base, output_dir=OUT_BOOSTED, label="isr_full_boosted",
        edp_samples_path=base_edp_path,
        diagonal_boost_amplitude=0.5, diagonal_boost_log_space=True, diagonal_boost_rng_seed=888,
        diagonal_boost_horizontal_scale_km=200.0,  # pinned -- see module docstring
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
