#!/usr/bin/env python3
"""Build the real ISR-derived PCA basis from the real IRI2020-extended
TROISR2025_nonan file, on the project's standard production altitude grid
(90-900km/10km), and save it for reuse."""
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

import numpy as np
from Assimilation_Cycle import isr_pca_basis as ipb
from Assimilation_Cycle.cycle_config import default_altitude_grid

EXTENDED = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/TROISR2025_nonan_extended.nc"
OUT = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_1e-4.nc"

altitude_grid = default_altitude_grid()
basis = ipb.build_isr_pca_basis(EXTENDED, altitude_grid, retaining_threshold=1 - 1e-4)
ipb.save_isr_pca_basis(basis, OUT)

d = basis.diagnostics
print(f"n_profiles_used/total: {d.n_profiles_used}/{d.n_profiles_total}")
print(f"n_retained: {d.n_retained} (retaining_threshold={d.retaining_threshold})")
print(f"singular values (first 10): {d.singular_values[:10]}")
print(f"cumulative variance (first 10): {d.cumulative_variance_ratio[:10]}")
print(f"PCA shape: {basis.PCA.shape}, PCA_mean shape: {basis.PCA_mean.shape}")
print(f"Saved to {OUT}")
