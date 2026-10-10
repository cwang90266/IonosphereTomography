#!/usr/bin/env python3
"""ISR_Integration_Plan.md Section 4.2: real-data style comparison across
the 4 default styles (ANCHOR, PCA_3D_10ex, PCA_1D_10ex, PCA_1D_10ex_ISR),
multi-seed (3 independent member subsamples + held-fixed analysis_rng_seed
per trial), against the real 2025-11-18 Tromso RO+IGS window, single-batch
(this project's own established best-or-tied-best/fastest real-data
finding -- also sidesteps the per-batch-vs-pooled RMSE distinction
entirely, since there is only one batch covering the whole real dataset).

Reuses the already-built real 2000-member/82-level/161-geo IRI2020
ensemble (step15_n2000_10km) by SUBSAMPLING members per trial rather than
rebuilding from scratch, and the real RO+IGS observation geometry already
prepared for an earlier real-data run (step12_all_styles) -- both
expensive, already-validated artifacts from this project's prior
sessions, not rebuilt here.
"""
import sys
import time
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import edp_samples as E
from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver
from Parameterization import Parameterized_EDPSamples

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle import observation_stream, isr_comparison as isrc
from Assimilation_Cycle.cycle_driver import run_batch_loop
from Assimilation_Cycle.isr_pca_basis import resolve_hyper_params_for_style

IRI_ENSEMBLE_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step15_n2000_10km/edp_samples_n2000_10km.nc"
RO_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles/ro_observations.nc"
IGS_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles/igs_observations.nc"
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"
OUT_DIR = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SUBSAMPLE = 150
SEEDS = (101, 102, 103)
STYLES = {
    "ANCHOR": None,
    "PCA_3D_10ex": {"retaining_threshold": 1 - 1e-4},
    "PCA_1D_10ex": {"retaining_threshold": 1 - 1e-5},
    "PCA_1D_10ex_ISR": None,  # filled in via isr_pca_basis_path below
}


def main():
    print(f"Loading real full ensemble: {IRI_ENSEMBLE_PATH}")
    full_edp = E.EDPSamples.fromNetCDF(IRI_ENSEMBLE_PATH)
    altitude_grid = np.asarray(full_edp.altitude, dtype=float)
    n_member_full = full_edp.edps.shape[-1]
    print(f"Full ensemble: {full_edp.edps.shape}")

    cfg_base = CycleConfig(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
        grid_radius_deg=18.0, horizontal_resolution_deg=2.5,
        altitude_grid=altitude_grid,
        batch_size=99999,  # force a single batch (this project's established best/fastest real-data finding)
        obs_sigma=3.0,     # this project's own tuned value for this exact real dataset (see memory)
        ro_observations_path=RO_PATH, igs_observations_path=IGS_PATH,
        isr_file_path=ISR_FILE, isr_pca_basis_path=ISR_BASIS,
    )
    print("Assembling real RO+IGS batches (shared across all styles/trials)...")
    batches = observation_stream.assemble(cfg_base)
    print(f"{len(batches)} batch(es), {sum(len(b.y_obs) for b in batches)} total rays")

    isr = isrc.load_isr_dataset(cfg_base.isr_file_path)

    rows = []
    for trial_seed in SEEDS:
        subsample_rng = np.random.default_rng(trial_seed)
        member_idx = subsample_rng.choice(n_member_full, size=N_SUBSAMPLE, replace=False)
        sub_edp = E.EDPSamples.from_xarray(full_edp.isel(sample=member_idx))

        for style, hyper_params in STYLES.items():
            hyper_params = resolve_hyper_params_for_style(cfg_base, style, hyper_params)
            t0 = time.time()
            pes = Parameterized_EDPSamples(sub_edp, style=style, hyper_params=hyper_params)
            ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)

            driver = GeneralEnKFDriver(style=style, hyper_params=hyper_params)
            driver.config.rng = np.random.default_rng(trial_seed)
            obs_operators = [
                driver.build_observation_operator(sub_edp, pes.Parameterization, ensemble_prior.param_shape,
                                                   b.podTc2_data)
                for b in batches
            ]
            result = run_batch_loop(ensemble_prior, driver, obs_operators, batches,
                                     rng=np.random.default_rng(trial_seed))
            wall_time_s = time.time() - t0

            outcome = result.batch_outcomes[0]
            collector = isrc.IsrComparisonCollector(sub_edp, isr)
            collector.on_batch(batches[0], obs_operators[0], ensemble_prior, result.final_ensemble, outcome)
            rmse_df = isrc.pooled_isr_rmse_by_altitude(collector.matches, altitude_grid, isr.altitude)
            isr_rmse_forecast = float(rmse_df["rmse_forecast_m3"].mean(skipna=True))
            isr_rmse_analysis = float(rmse_df["rmse_analysis_m3"].mean(skipna=True))

            row = dict(
                trial_seed=trial_seed, style=style, n_state=ensemble_prior.n_state,
                wall_time_s=wall_time_s, converged=outcome.diagnostics.converged,
                n_iterations=getattr(outcome.diagnostics, "n_iterations", None),
                rmse_forecast=outcome.rmse_reduction.rmse_forecast,
                rmse_analysis=outcome.rmse_reduction.rmse_analysis,
                isr_rmse_forecast_m3=isr_rmse_forecast, isr_rmse_analysis_m3=isr_rmse_analysis,
            )
            rows.append(row)
            print(f"[seed {trial_seed}] {style}: n_state={row['n_state']}, wall_time={wall_time_s:.1f}s, "
                  f"converged={row['converged']}, RMSE forecast/analysis={row['rmse_forecast']:.3f}/"
                  f"{row['rmse_analysis']:.3f} TECU, ISR RMSE forecast/analysis="
                  f"{isr_rmse_forecast:.3e}/{isr_rmse_analysis:.3e} m^-3")

            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(OUT_DIR / "style_comparison.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== Summary across seeds (mean +/- std) ===")
    summary = df.groupby("style").agg(
        n_state=("n_state", "first"),
        wall_time_s=("wall_time_s", "mean"),
        converged_frac=("converged", "mean"),
        rmse_forecast_mean=("rmse_forecast", "mean"),
        rmse_analysis_mean=("rmse_analysis", "mean"),
        rmse_analysis_std=("rmse_analysis", "std"),
        isr_rmse_analysis_mean=("isr_rmse_analysis_m3", "mean"),
        isr_rmse_analysis_std=("isr_rmse_analysis_m3", "std"),
    )
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "style_comparison_summary.csv")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
