#!/usr/bin/env python3
"""Post-merge regression check (2026-11-05): re-runs
isr_full_scale_run_fullos_fix.py's No_Boost/Boosted frac65 pair
unchanged (identical config, seeds, RO/IGS/ISR inputs, styles) to
confirm the pipeline still produces consistent results after merging
the per-RO/IGS impact-vector feature (Impact_Vector_Plan.md) into
chunming_dev/develop -- that feature is off by default
(CycleConfig.compute_impact_vectors=False), so this run exercises no new
code paths, only confirms zero regression in the existing ones it
touched (analysis_engine.py's on_iteration hook, observation_operator.py's
get_jacobian_array/linearized refactor, driver.py's pass-through).

Output goes to new *_v2 folders -- the original frac65 pair is kept
untouched as the reference to compare against.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs", REPO / "Spline_Basis"):
    sys.path.insert(0, str(p))

from Assimilation_Cycle.cycle_config import CycleConfig, default_styles, default_hyper_params_by_style
from Assimilation_Cycle.package_run import run_package

RO_DATA_DIR = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/RO_Data")
STEP12 = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles")
STEP12_FIX = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_full_los_frac65_v2")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"
OUT_ROOT = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle")
OUT_NOBOOST = OUT_ROOT / "ISR_Full_Scale_No_Boost_frac65_v2"
OUT_BOOSTED = OUT_ROOT / "ISR_Full_Scale_Boosted_frac65_v2"


def main():
    STEP12_FIX.mkdir(parents=True, exist_ok=True)

    cfg_base = CycleConfig(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
        grid_radius_deg=18.0, horizontal_resolution_deg=2.5,
        n_ensemble=2000,
        ro_kwargs=dict(
            podtc_dir=RO_DATA_DIR,
            roi_mode="full_los",
            alt_limit_km=None,
            fraction_required=0.65,
        ),
        ro_observations_output_path=STEP12_FIX / "ro_observations.nc",
        igs_observations_path=STEP12 / "igs_observations.nc",
        isr_file_path=ISR_FILE, isr_pca_basis_path=ISR_BASIS,
        obs_sigma=3.0,
        analysis_rng_seed=777,
        iri_rng_seed=555,
        styles=default_styles(),
        hyper_params_by_style=default_hyper_params_by_style(),
    )

    print("=== Run 1/2: no boost (also builds+saves the real base ensemble) ===", flush=True)
    t0 = time.time()
    cfg_noboost = replace(cfg_base, output_dir=OUT_NOBOOST, label="isr_full_no_boost_frac65_v2")
    result_noboost = run_package(cfg_noboost)
    print(f"no-boost run complete in {time.time() - t0:.1f}s", flush=True)
    for style, (cycle_result, n_state, wall_time) in result_noboost.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU",
              flush=True)

    base_edp_path = OUT_NOBOOST / "isr_full_no_boost_frac65_v2_edp_samples.nc"
    assert base_edp_path.exists(), f"expected base ensemble at {base_edp_path}"

    print("\n=== Run 2/2: boosted (amplitude=0.5, log_space=True, reusing the same base ensemble) ===", flush=True)
    t0 = time.time()
    cfg_boosted = replace(
        cfg_base, output_dir=OUT_BOOSTED, label="isr_full_boosted_frac65_v2",
        edp_samples_path=base_edp_path,
        diagonal_boost_amplitude=0.5, diagonal_boost_log_space=True, diagonal_boost_rng_seed=888,
    )
    result_boosted = run_package(cfg_boosted)
    print(f"boosted run complete in {time.time() - t0:.1f}s", flush=True)
    for style, (cycle_result, n_state, wall_time) in result_boosted.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU",
              flush=True)

    print("\n=== Done. Outputs: ===", flush=True)
    print(f"  fresh RO (full_los): {STEP12_FIX / 'ro_observations.nc'}", flush=True)
    print(f"  no boost: {OUT_NOBOOST}", flush=True)
    print(f"  boosted:  {OUT_BOOSTED}", flush=True)


if __name__ == "__main__":
    main()
