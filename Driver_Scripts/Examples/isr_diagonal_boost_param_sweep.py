#!/usr/bin/env python3
"""Diagonal-boost 3-parameter sweep (amplitude, vertical_scale_km,
horizontal_scale_km) scored directly against real ISR ground truth --
the user's follow-up after the full-scale run showed amplitude=0.5
(default taper/smoothing scales) is a real regression vs. ISR, for every
style, even though it looks like a big win on held-out TEC alone.

Unlike the Section 4.2 diagonal-boost resweep (which needed a fit/score
RO-entry split because it only had TEC to validate against), this sweep
scores against ISR -- a genuinely independent ground truth -- so the
full real single-batch RO+IGS dataset is used for the fit (no holdout
needed) and ISR is the only score.

Reuses the real cached base ensemble from the full-scale no-boost run
(ISR_Full_Scale_No_Boost/isr_full_no_boost_edp_samples.nc -- real
n=2000/82-level/161-geo climatology, 2025-11-18 Tromso window) and the
same real RO+IGS observations, applying apply_diagonal_boost fresh for
each (amplitude, vertical_scale_km, horizontal_scale_km) combination.

Style: PCA_3D_10ex only for the main grid (cheap -- no Chapman fit,
~2min/run at full n=2000 -- and the style whose boosting was previously
"most validated" in this project's history, so the most informative to
re-derive). ANCHOR is excluded (expensive, never converges, and the
full-scale run showed it does best WITHOUT boosting anyway -- boosting
isn't being proposed for it). The best combination found is spot-checked
against PCA_1D_10ex and PCA_1D_10ex_ISR afterward to check it
generalizes, not swept independently per style (would be the full-scale
run's cost x this grid's size -- intractable).
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
from Assimilation_Cycle.diagonal_boost import apply_diagonal_boost
from Assimilation_Cycle.isr_pca_basis import resolve_hyper_params_for_style

BASE_EDP_PATH = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_Full_Scale_No_Boost/isr_full_no_boost_edp_samples.nc")
STEP12 = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step12_all_styles")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"
OUT_DIR = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AMPLITUDES = (0.0, 0.1, 0.2, 0.3, 0.5)
VERTICAL_SCALES_KM = (10.0, 30.0, 100.0)
HORIZONTAL_SCALES_KM = (100.0, 200.0, 500.0)
ANALYSIS_SEED = 555
BOOST_SEED = 666
PRIMARY_STYLE = "PCA_3D_10ex"
SPOT_CHECK_STYLES = ("PCA_1D_10ex", "PCA_1D_10ex_ISR")

# Score band: the altitude range the full-scale run showed is most
# affected by boosting (E-region through F2-peak/lower-topside) -- the
# primary quantity used to rank combinations. Full per-altitude RMSE is
# still saved for every combination either way.
SCORE_ALT_MIN, SCORE_ALT_MAX = 95.0, 440.0


def run_one(edp_samples, batch, isr, style, hyper_params, analysis_seed):
    pes = Parameterized_EDPSamples(edp_samples, style=style, hyper_params=hyper_params)
    ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)

    driver = GeneralEnKFDriver(style=style, hyper_params=hyper_params)
    driver.config.rng = np.random.default_rng(analysis_seed)
    obs_op = driver.build_observation_operator(edp_samples, pes.Parameterization, ensemble_prior.param_shape,
                                                 batch.podTc2_data)
    result = run_batch_loop(ensemble_prior, driver, [obs_op], [batch], rng=np.random.default_rng(analysis_seed))
    outcome = result.batch_outcomes[0]

    collector = isrc.IsrComparisonCollector(edp_samples, isr)
    collector.on_batch(batch, obs_op, ensemble_prior, result.final_ensemble, outcome)
    rmse_df = isrc.pooled_isr_rmse_by_altitude(collector.matches, np.asarray(edp_samples.altitude), isr.altitude)

    band = rmse_df[(rmse_df["altitude_km"] >= SCORE_ALT_MIN) & (rmse_df["altitude_km"] <= SCORE_ALT_MAX)]
    band_rmse_analysis = float(np.sqrt(np.nanmean(band["rmse_analysis_m3"].to_numpy() ** 2)))
    full_rmse_analysis = float(np.sqrt(np.nanmean(rmse_df["rmse_analysis_m3"].to_numpy() ** 2)))

    return dict(
        n_state=ensemble_prior.n_state, converged=outcome.diagnostics.converged,
        rmse_forecast_tecu=outcome.rmse_reduction.rmse_forecast, rmse_analysis_tecu=outcome.rmse_reduction.rmse_analysis,
        isr_band_rmse_analysis=band_rmse_analysis, isr_full_rmse_analysis=full_rmse_analysis,
    ), rmse_df


def main():
    base_edp = E.EDPSamples.fromNetCDF(str(BASE_EDP_PATH))
    altitude_grid = np.asarray(base_edp.altitude)

    cfg = CycleConfig(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
        grid_radius_deg=18.0, horizontal_resolution_deg=2.5, altitude_grid=altitude_grid,
        ro_observations_path=STEP12 / "ro_observations.nc", igs_observations_path=STEP12 / "igs_observations.nc",
        isr_pca_basis_path=ISR_BASIS,
        obs_sigma=3.0,
    )
    batches = observation_stream.assemble(cfg)
    batch = batches[0]
    isr = isrc.load_isr_dataset(ISR_FILE)
    print(f"{len(batch.y_obs)} real rays, base ensemble {base_edp.edps.shape}")

    hyper_params = {"retaining_threshold": 1 - 1e-4}

    rows = []
    combos = [(0.0, VERTICAL_SCALES_KM[0], HORIZONTAL_SCALES_KM[0])]  # baseline, scale-independent
    combos += [(a, v, h) for a in AMPLITUDES if a > 0 for v in VERTICAL_SCALES_KM for h in HORIZONTAL_SCALES_KM]

    for amplitude, vscale, hscale in combos:
        t0 = time.time()
        boost_rng = np.random.default_rng(BOOST_SEED)
        edp = apply_diagonal_boost(base_edp, amplitude, vertical_scale_km=vscale, horizontal_scale_km=hscale,
                                    rng=boost_rng, log_space=True)
        metrics, rmse_df = run_one(edp, batch, isr, PRIMARY_STYLE, hyper_params, ANALYSIS_SEED)
        wall_time_s = time.time() - t0

        row = dict(amplitude=amplitude, vertical_scale_km=vscale, horizontal_scale_km=hscale,
                   wall_time_s=wall_time_s, **metrics)
        rows.append(row)
        print(f"amp={amplitude} v={vscale} h={hscale}: n_state={metrics['n_state']}, "
              f"converged={metrics['converged']}, RO+IGS RMSE f/a={metrics['rmse_forecast_tecu']:.3f}/"
              f"{metrics['rmse_analysis_tecu']:.3f} TECU, ISR band RMSE={metrics['isr_band_rmse_analysis']:.3e}, "
              f"ISR full RMSE={metrics['isr_full_rmse_analysis']:.3e} ({wall_time_s:.1f}s)")
        pd.DataFrame(rows).to_csv(OUT_DIR / "diagonal_boost_param_sweep.csv", index=False)

    df = pd.DataFrame(rows)
    best = df.loc[df["isr_band_rmse_analysis"].idxmin()]
    print(f"\n=== Best combination by ISR band RMSE ===\n{best}")

    # Spot-check the best combination (and no-boost) against two more styles.
    print("\n=== Spot-check: best combination vs. no-boost, other styles ===")
    spot_rows = []
    for style in SPOT_CHECK_STYLES:
        hp = resolve_hyper_params_for_style(cfg, style, {"retaining_threshold": 1 - 1e-5} if style == "PCA_1D_10ex" else None)
        for label, (amplitude, vscale, hscale) in (("no_boost", (0.0, 30.0, 200.0)), ("best", (best.amplitude, best.vertical_scale_km, best.horizontal_scale_km))):
            boost_rng = np.random.default_rng(BOOST_SEED)
            edp = apply_diagonal_boost(base_edp, amplitude, vertical_scale_km=vscale, horizontal_scale_km=hscale,
                                        rng=boost_rng, log_space=True)
            metrics, _ = run_one(edp, batch, isr, style, hp, ANALYSIS_SEED)
            spot_rows.append(dict(style=style, condition=label, **metrics))
            print(f"{style} [{label}]: RO+IGS RMSE analysis={metrics['rmse_analysis_tecu']:.3f} TECU, "
                  f"ISR band RMSE={metrics['isr_band_rmse_analysis']:.3e}")
    pd.DataFrame(spot_rows).to_csv(OUT_DIR / "diagonal_boost_param_sweep_spotcheck.csv", index=False)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
