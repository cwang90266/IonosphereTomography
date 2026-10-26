#!/usr/bin/env python3
"""ISR_Integration_Plan.md Section 4.2 basis-quality diagnostic:
cross-reconstruct held-out real IRI2020-ensemble profiles and held-out
real ISR profiles through both a climatology-fitted (IRI) and an
ISR-fitted PCA basis, at matched retaining_threshold, across multiple
random train/test splits -- tests the "ISR provides a wider range of
EDPs than the climatology" hypothesis directly, with no EnKF involved.

Both bases are fit the same way PCA_1D_10ex's own basis-fit works
(Parameterization.get_PCA on log10(density), reshaped to (n_height, -1))
-- same math, different population.
"""
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import xarray as xr
import edp_samples as E
from Parameterization import get_PCA, PCA2EDP_1D, EDP2PCA_1D
from Assimilation_Cycle import isr_pca_basis as ipb
from Assimilation_Cycle.cycle_config import default_altitude_grid

IRI_ENSEMBLE_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step15_n2000_10km/edp_samples_n2000_10km.nc"
EXTENDED_ISR_PATH = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/TROISR2025_nonan_extended.nc"
OUT_DIR = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RETAINING_THRESHOLD = 1 - 1e-4
TRAIN_FRACTION = 0.7
N_SPLITS = 5
DENSITY_FLOOR = 1.0e4


def recon_rmse(PCA, mean, test_log):
    coeff = EDP2PCA_1D(test_log, PCA, mean=mean, linear=True)
    recon = PCA2EDP_1D(coeff, PCA, mean=mean, linear=True)
    return float(np.sqrt(np.mean((test_log - recon) ** 2)))


def split_indices(n, frac_train, rng):
    idx = rng.permutation(n)
    n_train = int(round(n * frac_train))
    return idx[:n_train], idx[n_train:]


def main():
    altitude_grid = default_altitude_grid()

    print(f"Loading real IRI2020 ensemble: {IRI_ENSEMBLE_PATH}")
    edp = E.EDPSamples.fromNetCDF(IRI_ENSEMBLE_PATH)
    iri_density = np.asarray(edp.edps, dtype=float)  # (n_height, n_geo, n_member)
    assert np.allclose(np.asarray(edp.altitude, dtype=float), altitude_grid), (
        "IRI ensemble's altitude grid does not match default_altitude_grid() -- "
        "cross-reconstruction would be comparing different altitude axes."
    )
    iri_log = np.log10(np.maximum(iri_density, DENSITY_FLOOR))
    n_height, n_geo, n_member = iri_log.shape
    print(f"IRI ensemble: n_height={n_height}, n_geo={n_geo}, n_member={n_member}")

    print(f"Loading real extended ISR file: {EXTENDED_ISR_PATH}")
    with xr.open_dataset(EXTENDED_ISR_PATH, decode_timedelta=False) as ds:
        isr_altitude = np.asarray(ds["altitude"].values, dtype=float)
        isr_density = np.asarray(ds["Ne"].values, dtype=float)
    isr_log_full = ipb.resample_isr_profiles_to_grid(isr_altitude, isr_density, altitude_grid)
    valid = np.all(np.isfinite(isr_log_full), axis=0)
    isr_log_full = isr_log_full[:, valid]
    n_isr = isr_log_full.shape[1]
    print(f"ISR profiles resampled to the production grid: {n_isr}/{len(ds['time_utc'])} fully valid "
          f"(0-{altitude_grid.max():.0f}km fully inside the extended file's native+extended range).")

    rows = []
    for split in range(N_SPLITS):
        rng = np.random.default_rng(split)

        train_m, test_m = split_indices(n_member, TRAIN_FRACTION, rng)
        iri_train = iri_log[:, :, train_m].reshape(n_height, -1)
        iri_test = iri_log[:, :, test_m].reshape(n_height, -1)

        train_p, test_p = split_indices(n_isr, TRAIN_FRACTION, rng)
        isr_train = isr_log_full[:, train_p]
        isr_test = isr_log_full[:, test_p]

        PCA_iri, mean_iri, diag_iri = get_PCA(iri_train, RETAINING_THRESHOLD)
        PCA_isr, mean_isr, diag_isr = get_PCA(isr_train, RETAINING_THRESHOLD)

        row = dict(
            split=split,
            n_iri_components=diag_iri.n_retained,
            n_isr_components=diag_isr.n_retained,
            rmse_iri_basis_on_iri_test=recon_rmse(PCA_iri, mean_iri, iri_test),
            rmse_iri_basis_on_isr_test=recon_rmse(PCA_iri, mean_iri, isr_test),
            rmse_isr_basis_on_iri_test=recon_rmse(PCA_isr, mean_isr, iri_test),
            rmse_isr_basis_on_isr_test=recon_rmse(PCA_isr, mean_isr, isr_test),
        )
        rows.append(row)
        print(f"split {split}: n_iri_comp={row['n_iri_components']}, n_isr_comp={row['n_isr_components']}, "
              f"IRI-on-IRI={row['rmse_iri_basis_on_iri_test']:.4f}, IRI-on-ISR={row['rmse_iri_basis_on_isr_test']:.4f}, "
              f"ISR-on-IRI={row['rmse_isr_basis_on_iri_test']:.4f}, ISR-on-ISR={row['rmse_isr_basis_on_isr_test']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "basis_quality_cross_reconstruction.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Summary across splits (mean +/- std, log10-space RMSE) ===")
    for col in ("n_iri_components", "n_isr_components", "rmse_iri_basis_on_iri_test",
                "rmse_iri_basis_on_isr_test", "rmse_isr_basis_on_iri_test", "rmse_isr_basis_on_isr_test"):
        print(f"{col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
    print(f"\nSaved per-split table to {out_csv}")

    # -- Matched-component-count control --------------------------------
    # The primary comparison above fits both bases at the same
    # retaining_threshold, which gave the IRI basis 13 components and the
    # ISR basis 26 -- more components could trivially reconstruct
    # anything better, so this isolates "basis quality" from "basis size"
    # by truncating both to the SAME k (13 and 26, the two natural counts
    # found above), fit once at a generous near-full-rank threshold so
    # enough components exist to truncate from.
    print("\n=== Matched-component-count control (same k for both bases) ===")
    matched_rows = []
    for split in range(N_SPLITS):
        rng = np.random.default_rng(split)
        train_m, test_m = split_indices(n_member, TRAIN_FRACTION, rng)
        iri_train = iri_log[:, :, train_m].reshape(n_height, -1)
        iri_test = iri_log[:, :, test_m].reshape(n_height, -1)
        train_p, test_p = split_indices(n_isr, TRAIN_FRACTION, rng)
        isr_train = isr_log_full[:, train_p]
        isr_test = isr_log_full[:, test_p]

        PCA_iri_full, mean_iri, _ = get_PCA(iri_train, 0.9999999)
        PCA_isr_full, mean_isr, _ = get_PCA(isr_train, 0.9999999)

        for k in (13, 26):
            k_iri = min(k, PCA_iri_full.shape[1])
            k_isr = min(k, PCA_isr_full.shape[1])
            PCA_iri_k = PCA_iri_full[:, :k_iri]
            PCA_isr_k = PCA_isr_full[:, :k_isr]
            matched_rows.append(dict(
                split=split, k=k,
                rmse_iri_basis_on_iri_test=recon_rmse(PCA_iri_k, mean_iri, iri_test),
                rmse_iri_basis_on_isr_test=recon_rmse(PCA_iri_k, mean_iri, isr_test),
                rmse_isr_basis_on_iri_test=recon_rmse(PCA_isr_k, mean_isr, iri_test),
                rmse_isr_basis_on_isr_test=recon_rmse(PCA_isr_k, mean_isr, isr_test),
            ))

    df_k = pd.DataFrame(matched_rows)
    out_csv_k = OUT_DIR / "basis_quality_matched_k.csv"
    df_k.to_csv(out_csv_k, index=False)
    for k in (13, 26):
        sub = df_k[df_k["k"] == k]
        print(f"\nk={k}:")
        for col in ("rmse_iri_basis_on_iri_test", "rmse_iri_basis_on_isr_test",
                    "rmse_isr_basis_on_iri_test", "rmse_isr_basis_on_isr_test"):
            print(f"  {col}: {sub[col].mean():.4f} +/- {sub[col].std():.4f}")
    print(f"\nSaved matched-k table to {out_csv_k}")


if __name__ == "__main__":
    main()
