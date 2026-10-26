#!/usr/bin/env python3
"""Investigate why ANCHOR's analysis EDP spread (in the ISR comparison
plots) is much wider than the other styles', using the REAL saved
artifacts from the just-completed full-scale no-boost run -- no new
assimilation run needed."""
import sys
import json
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

import numpy as np
import edp_samples as E
from Parameterization import Parameterized_EDPSamples
from Assimilation_Cycle.ensemble_io import load_ensemble_netcdf
from Assimilation_Cycle.output import _interpolate_field_at_latlon

OUT = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_Full_Scale_No_Boost")
STATION_LAT, STATION_LON = 69.5864, 19.2272  # real TRO ISR site (from the file's own attrs)

for style in ("ANCHOR", "PCA_3D_10ex"):
    print(f"\n=== {style} ===")
    pes = Parameterized_EDPSamples.fromNetCDF(OUT / f"isr_full_no_boost_{style}_parameterized.nc")
    edp_samples = pes.EDPSamples
    altitude = np.asarray(edp_samples.altitude)

    forecast_param_vec = np.asarray(edp_samples["param_vec"].to_numpy())
    forecast_density = pes.Parameterization.get_density(forecast_param_vec, alt=altitude if pes.Parameterization.needs_altitude else None)

    analysis_ensemble, meta = load_ensemble_netcdf(OUT / f"isr_full_no_boost_{style}_ensembles" / "ensemble_batch_0000.nc")
    analysis_param_vec = analysis_ensemble.to_param_shape()
    analysis_density = pes.Parameterization.get_density(analysis_param_vec, alt=altitude if pes.Parameterization.needs_altitude else None)

    f_at_isr = _interpolate_field_at_latlon(edp_samples, forecast_density, STATION_LAT, STATION_LON)
    a_at_isr = _interpolate_field_at_latlon(edp_samples, analysis_density, STATION_LAT, STATION_LON)

    f_std, a_std = f_at_isr.std(axis=-1), a_at_isr.std(axis=-1)
    f_mean, a_mean = f_at_isr.mean(axis=-1), a_at_isr.mean(axis=-1)
    print("altitude_km  forecast_mean   forecast_std   forecast_relstd   analysis_mean   analysis_std   analysis_relstd")
    for i in range(0, len(altitude), 4):
        print(f"{altitude[i]:8.1f}  {f_mean[i]:.3e}  {f_std[i]:.3e}  {f_std[i]/max(f_mean[i],1):.3f}  "
              f"{a_mean[i]:.3e}  {a_std[i]:.3e}  {a_std[i]/max(a_mean[i],1):.3f}")

    if style == "ANCHOR":
        # Look directly at parameter-space spread at the nearest geo vertex
        # (ANCHOR's state is per-geo-point parameters, not a density field,
        # so "interpolate" doesn't directly apply the same way -- nearest
        # vertex is the natural diagnostic view here).
        from scipy.spatial import cKDTree
        tree = cKDTree(edp_samples.geolocation)
        _, nearest = tree.query([[STATION_LON, STATION_LAT]])
        g = int(nearest[0])
        from Parameterization import ANCHOR_PARAM_LABELS
        print(f"\nANCHOR param-space spread at nearest geo vertex (idx={g}):")
        print(f"{'param':10s}  {'forecast_mean':>14s}  {'forecast_std':>14s}  {'analysis_mean':>14s}  {'analysis_std':>14s}  {'std_ratio(A/F)':>14s}")
        for k, label in enumerate(ANCHOR_PARAM_LABELS):
            fm, fs = forecast_param_vec[k, g, :].mean(), forecast_param_vec[k, g, :].std()
            am, asd = analysis_param_vec[k, g, :].mean(), analysis_param_vec[k, g, :].std()
            ratio = asd / fs if fs > 0 else float("nan")
            print(f"{label:10s}  {fm:14.4f}  {fs:14.4f}  {am:14.4f}  {asd:14.4f}  {ratio:14.3f}")
        print(f"\nconverged: {meta.get('style')}, diagnostics not stored in ensemble file -- "
              f"recall from the run log: ANCHOR converged=False in both conditions.")
