#!/usr/bin/env python3
"""ISR_Integration_Plan.md Section 4.2: diagonal-boost amplitude resweep
specific to 'PCA_1D_10ex_ISR'. Reuses this project's own established
held-out methodology (Assimilation_Cycle_Integration_Plan.md Section 17):
fit the assimilation on a subset of real entries, score forecast/analysis
only against the held-out entries' real TEC -- never touched by the fit.

Always log_space=True (this is a log10-based style, and the project's own
hard-won finding is that linear-space boosting corrupts log10-fitting
styles via a positivity-clip discontinuity).

Architecturally different from PCA_3D_10ex/PCA_1D_10ex here: this style's
basis is PRE-FIT and FIXED (loaded from isr_pca_basis_path), not refit
from the (now-boosted) ensemble -- so unlike the other PCA_*_10ex styles,
boosting cannot inflate this style's state dimension. Whether the boost's
new variance is even representable in the fixed ISR basis's span (or
mostly projected away) is exactly the open question this sweep answers.
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
from Assimilation_Cycle import observation_stream
from Assimilation_Cycle.cycle_driver import run_batch_loop, CycleBatch
from Assimilation_Cycle.diagonal_boost import apply_diagonal_boost
from Assimilation_Cycle.isr_pca_basis import resolve_hyper_params_for_style

IRI_ENSEMBLE_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step15_n2000_10km/edp_samples_n2000_10km.nc"
RO_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles/ro_observations.nc"
IGS_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles/igs_observations.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"
OUT_DIR = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SUBSAMPLE = 150
AMPLITUDES = (0.0, 0.2, 0.5, 0.8, 1.2)
N_SPLITS = 3           # independent fit/score entry splits
ANALYSIS_SEED = 201    # fixed across every amplitude/split -- isolates the boost's effect
STYLE = "PCA_1D_10ex_ISR"
N_SCORE_ENTRIES = 8    # matches this project's own established fit=30/score=8-of-38 split


def split_entries_to_rays(entry_ray_ranges, n_total_rays, score_ids):
    """``score_ids``: a set of ``id(entry)`` ints, not entries themselves
    -- ``ObservationEntry``'s auto-generated ``__eq__``/``__hash__`` would
    compare numpy array fields (raises rather than a bool), the same
    reason ``package_run.py``'s ``_collect_unique_entries`` uses ``id()``
    instead of ``in``/set-membership on the entries directly."""
    score_mask = np.zeros(n_total_rays, dtype=bool)
    for entry, ray_slice in entry_ray_ranges:
        if id(entry) in score_ids:
            score_mask[ray_slice] = True
    return ~score_mask, score_mask


def main():
    full_edp = E.EDPSamples.fromNetCDF(IRI_ENSEMBLE_PATH)
    altitude_grid = np.asarray(full_edp.altitude, dtype=float)

    cfg_base = CycleConfig(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
        grid_radius_deg=18.0, horizontal_resolution_deg=2.5,
        altitude_grid=altitude_grid, batch_size=99999, obs_sigma=3.0,
        ro_observations_path=RO_PATH, igs_observations_path=IGS_PATH,
        isr_pca_basis_path=ISR_BASIS,
    )
    batches = observation_stream.assemble(cfg_base)
    batch = batches[0]
    entries = [e for e, _ in batch.entry_ray_ranges]
    n_total_rays = len(batch.y_obs)
    print(f"{len(entries)} entries, {n_total_rays} total rays")

    hyper_params = resolve_hyper_params_for_style(cfg_base, STYLE, None)

    rows = []
    for split in range(N_SPLITS):
        split_rng = np.random.default_rng(1000 + split)
        score_idx = split_rng.choice(len(entries), size=N_SCORE_ENTRIES, replace=False)
        score_ids = {id(entries[i]) for i in score_idx}
        fit_mask, score_mask = split_entries_to_rays(batch.entry_ray_ranges, n_total_rays, score_ids)
        print(f"split {split}: {fit_mask.sum()} fit rays, {score_mask.sum()} score rays "
              f"({len(score_ids)} held-out entries)")

        rec = np.asarray(batch.podTc2_data["rec_ecef_km"])
        gnss = np.asarray(batch.podTc2_data["gnss_ecef_km"])
        y_obs = np.asarray(batch.y_obs)
        fit_podTc2 = {"rec_ecef_km": rec[:, fit_mask], "gnss_ecef_km": gnss[:, fit_mask]}
        score_podTc2 = {"rec_ecef_km": rec[:, score_mask], "gnss_ecef_km": gnss[:, score_mask]}
        y_fit, y_score = y_obs[fit_mask], y_obs[score_mask]

        member_rng = np.random.default_rng(2000 + split)
        member_idx = member_rng.choice(full_edp.edps.shape[-1], size=N_SUBSAMPLE, replace=False)
        sub_edp_base = E.EDPSamples.from_xarray(full_edp.isel(sample=member_idx))

        for amplitude in AMPLITUDES:
            t0 = time.time()
            boost_rng = np.random.default_rng(3000 + split)  # same seed across amplitudes within a split
            # apply_diagonal_boost no-ops (returns edp_samples unchanged) for a falsy amplitude (0.0) --
            # no separate branch needed here.
            sub_edp = apply_diagonal_boost(sub_edp_base, amplitude, rng=boost_rng, log_space=True)

            pes = Parameterized_EDPSamples(sub_edp, style=STYLE, hyper_params=hyper_params)
            ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)

            driver = GeneralEnKFDriver(style=STYLE, hyper_params=hyper_params)
            driver.config.rng = np.random.default_rng(ANALYSIS_SEED)

            fit_op = driver.build_observation_operator(sub_edp, pes.Parameterization, ensemble_prior.param_shape, fit_podTc2)
            fit_batch = CycleBatch(podTc2_data=fit_podTc2, y_obs=y_fit,
                                    R=(cfg_base.obs_sigma ** 2) * np.eye(len(y_fit)), batch_index=0)
            result = run_batch_loop(ensemble_prior, driver, [fit_op], [fit_batch],
                                     rng=np.random.default_rng(ANALYSIS_SEED))
            outcome = result.batch_outcomes[0]

            score_op = driver.build_observation_operator(sub_edp, pes.Parameterization, ensemble_prior.param_shape, score_podTc2)
            y_score_forecast = score_op.forward_single(ensemble_prior.ensemble_mean())
            y_score_analysis = score_op.forward_single(result.final_ensemble.ensemble_mean())
            score_rmse_forecast = float(np.sqrt(np.mean((y_score_forecast - y_score) ** 2)))
            score_rmse_analysis = float(np.sqrt(np.mean((y_score_analysis - y_score) ** 2)))
            wall_time_s = time.time() - t0

            row = dict(
                split=split, amplitude=amplitude, n_state=ensemble_prior.n_state,
                wall_time_s=wall_time_s, converged=outcome.diagnostics.converged,
                fit_rmse_forecast=outcome.rmse_reduction.rmse_forecast,
                fit_rmse_analysis=outcome.rmse_reduction.rmse_analysis,
                heldout_rmse_forecast=score_rmse_forecast, heldout_rmse_analysis=score_rmse_analysis,
            )
            rows.append(row)
            print(f"  amplitude={amplitude}: n_state={row['n_state']}, converged={row['converged']}, "
                  f"fit RMSE f/a={row['fit_rmse_forecast']:.3f}/{row['fit_rmse_analysis']:.3f}, "
                  f"HELD-OUT RMSE f/a={score_rmse_forecast:.3f}/{score_rmse_analysis:.3f} TECU "
                  f"({wall_time_s:.1f}s)")
            pd.DataFrame(rows).to_csv(OUT_DIR / "diagonal_boost_resweep.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== Summary across splits (mean +/- std, held-out RMSE) ===")
    summary = df.groupby("amplitude").agg(
        n_state=("n_state", "mean"),
        converged_frac=("converged", "mean"),
        heldout_rmse_analysis_mean=("heldout_rmse_analysis", "mean"),
        heldout_rmse_analysis_std=("heldout_rmse_analysis", "std"),
    )
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "diagonal_boost_resweep_summary.csv")


if __name__ == "__main__":
    main()
