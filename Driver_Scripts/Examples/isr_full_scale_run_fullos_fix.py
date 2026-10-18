#!/usr/bin/env python3
"""Re-run of isr_full_scale_run.py's No_Boost/Boosted pair (2026-10-16),
investigating an isolated high-value artifact found in
ISR_Full_Scale_Boosted_v2's observation-operator diagnostic plot for
podTc2_YM08.2025.322.10.06.0022.C22.01_0000.0001_nc (200-500km, near the
top-left edge of the ROI).

Three rounds of fixes, all confirmed by the user:

1. The RO observations step12_all_styles/ro_observations.nc used by the
   v2 pair were prepared with roi_mode="tangent_point" (confirmed by
   reading that file's own netCDF global attrs directly), not the
   stricter roi_mode="full_los" the user recalled using -- "tangent_point"
   only checks whether one ray's TEC-max tangent point falls within
   radius_km, so a ray can graze well outside the ROI along the rest of
   its line of sight and still pass. This script re-prepares RO from the
   same raw podTc2 files (Data/Tomography_data/RO_Data/, the same 15 real
   occultations -- confirmed by matching filenames, including this exact
   flagged file) with roi_mode="full_los" and alt_limit_km=None -- per a
   follow-up instruction, this auto-derives each occultation's own
   altitude threshold as its LEO receiver's mean geodetic altitude
   (prepare_ro_observations' own new default), instead of a hand-picked
   constant (600km), since receiver altitude varies across LEOs in a
   batch (observation_preparation/ro_source.py, _build_clean_ro_entry).

2. EDPSamples.get_observation_operator's nearest-neighbour fallback for
   ray segments outside the mesh had no distance cap at all (confirmed
   against real data: 22.8% of 27,108 segments outside the mesh for this
   run, 81% of those collapsing onto a single vertex, 145,858km of
   accumulated path length on one ~25km grid cell -- the direct cause of
   the artifact). Fixed in EDPSamples/edp_samples.py by adding
   max_extrap_m, mirroring the cap already implemented and documented in
   the sibling function forward_model_mesh_tec. Segments farther than
   this are dropped (zero weight) instead of extrapolated.

3. fraction_required=1.0 (round 1's choice, matching what the user
   recalled) turned out to systematically reject low-tangent-altitude
   (deep) occultation rays for a geometric reason unrelated to their
   actual ROI relevance: with alt_limit_km auto-derived near the LEO's
   own ~550km cruising altitude, the "below-threshold" test window
   extends from the tangent point all the way back up to the receiver's
   own position, and that far end can be thousands of km from the ROI
   regardless of how good the tangent-point geometry is (confirmed
   against real data: the deepest ray in the flagged occultation,
   tangent_alt=1.04km, scores only 61% containment at the current
   threshold despite its tangent point itself sitting at 1436km < the
   2000km ROI radius). Rather than force every occultation's tangent
   geometry through an unrealistically strict 100% bar, fraction_required
   is relaxed to 0.65 (user-directed, 2026-10-16) and
   get_observation_operator's/forward_model_mesh_tec's max_extrap_m is
   widened 500km -> 1000km (user-directed) as a complementary second line
   of defense, since more rays with a genuinely-outside-ROI tail segment
   will now pass the ray-level filter and rely on the per-segment H-matrix
   cap to drop that tail's contribution instead.

Only the RO ROI filter and the observation-operator cap differ from the
original v2 pair; the IGS observations, ISR files, grid, ensemble size,
obs_sigma, rng seeds, and boost parameters are all unchanged, to isolate
the real effect of both fixes together at full production scale. Output
goes to new _frac65 folders -- the earlier fraction_required=1.0 pair
(_fullos_fix) is kept as an intermediate comparison point, not overwritten.
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

RO_DATA_DIR = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/RO_Data")
STEP12 = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles")
STEP12_FIX = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_full_los_frac65")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"
OUT_ROOT = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle")
OUT_NOBOOST = OUT_ROOT / "ISR_Full_Scale_No_Boost_frac65"
OUT_BOOSTED = OUT_ROOT / "ISR_Full_Scale_Boosted_frac65"


def main():
    STEP12_FIX.mkdir(parents=True, exist_ok=True)

    cfg_base = CycleConfig(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
        grid_radius_deg=18.0, horizontal_resolution_deg=2.5,
        n_ensemble=2000,
        # RO: prepare fresh from raw podTc2 with the stricter full-LOS ROI
        # filter, instead of loading step12_all_styles' tangent_point file.
        ro_kwargs=dict(
            podtc_dir=RO_DATA_DIR,
            roi_mode="full_los",
            alt_limit_km=None,  # auto: each occultation's own LEO mean altitude
            fraction_required=0.65,  # relaxed from 1.0 -- see module docstring, round 3
        ),
        ro_observations_output_path=STEP12_FIX / "ro_observations.nc",
        # IGS: unchanged, reuse the same already-prepared real file.
        igs_observations_path=STEP12 / "igs_observations.nc",
        isr_file_path=ISR_FILE, isr_pca_basis_path=ISR_BASIS,
        obs_sigma=3.0,
        analysis_rng_seed=777,
        iri_rng_seed=555,  # 2026-10-18: fixes a real non-reproducibility gap --
        # the no-boost run rebuilds the IRI2020 ensemble fresh, and that draw
        # was previously unseeded, so two otherwise-identical runs could (and
        # did) produce materially different per-style results. See
        # CycleConfig.iri_rng_seed's docstring.
        styles=default_styles(),
        hyper_params_by_style=default_hyper_params_by_style(),
    )

    print("=== Run 1/2: no boost (also builds+saves the real base ensemble) ===")
    t0 = time.time()
    cfg_noboost = replace(cfg_base, output_dir=OUT_NOBOOST, label="isr_full_no_boost_frac65")
    result_noboost = run_package(cfg_noboost)
    print(f"no-boost run complete in {time.time() - t0:.1f}s")
    for style, (cycle_result, n_state, wall_time) in result_noboost.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU")

    base_edp_path = OUT_NOBOOST / "isr_full_no_boost_frac65_edp_samples.nc"
    assert base_edp_path.exists(), f"expected base ensemble at {base_edp_path}"

    print("\n=== Run 2/2: boosted (amplitude=0.5, log_space=True, reusing the same base ensemble) ===")
    t0 = time.time()
    cfg_boosted = replace(
        cfg_base, output_dir=OUT_BOOSTED, label="isr_full_boosted_frac65",
        edp_samples_path=base_edp_path,
        diagonal_boost_amplitude=0.5, diagonal_boost_log_space=True, diagonal_boost_rng_seed=888,
    )
    result_boosted = run_package(cfg_boosted)
    print(f"boosted run complete in {time.time() - t0:.1f}s")
    for style, (cycle_result, n_state, wall_time) in result_boosted.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, converged={outcome.diagnostics.converged}, "
              f"RMSE forecast/analysis={outcome.rmse_reduction.rmse_forecast:.3f}/{outcome.rmse_reduction.rmse_analysis:.3f} TECU")

    print("\n=== Done. Outputs: ===")
    print(f"  fresh RO (full_los): {STEP12_FIX / 'ro_observations.nc'}")
    print(f"  no boost: {OUT_NOBOOST}")
    print(f"  boosted:  {OUT_BOOSTED}")


if __name__ == "__main__":
    main()
