#!/usr/bin/env python3
"""Smoke test: full 4-default-style package_run (ANCHOR, PCA_3D_10ex,
PCA_1D_10ex, PCA_1D_10ex_ISR) against real cached Tromso 2025-11-18 data,
with the real ISR comparison tooling AND the real ISR-derived PCA basis
wired in -- end-to-end verification of everything built this session."""
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

import xarray as xr
from Assimilation_Cycle import CycleConfig, run_package
from Assimilation_Cycle import isr_pca_basis as ipb
from Assimilation_Cycle.cycle_config import default_styles, default_hyper_params_by_style

CACHE = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step16_package_smoke")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
EXTENDED_ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/TROISR2025_nonan_extended.nc"
ISR_BASIS = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/isr_pca_basis_smoke_17lvl.nc"
OUT = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/isr_full_style_smoke_test"

# The cached step16_package_smoke edp_samples.nc was built on a coarse
# 17-level/50km test grid (not the 82-level/10km production default) --
# the ISR PCA basis must be built on the SAME altitude grid the cycle's
# edp_samples actually uses (a real pitfall: Parameterization.EDP2PCA_1D
# requires matching n_height, same class of gotcha as this project's
# documented 17-vs-82-level/radius_km-vs-grid_radius_deg mismatches).
with xr.open_dataset(CACHE / "smoke_test_edp_samples.nc", decode_timedelta=False) as _ds:
    smoke_altitude_grid = _ds["altitude"].values
basis = ipb.build_isr_pca_basis(EXTENDED_ISR_FILE, smoke_altitude_grid, retaining_threshold=1 - 1e-4)
ipb.save_isr_pca_basis(basis, ISR_BASIS)
print(f"Built ISR PCA basis matching the smoke-test grid: n_retained={basis.diagnostics.n_retained}")

cfg = CycleConfig(
    start_time="2025-11-18T10:00:00",
    end_time="2025-11-18T11:00:00",
    center_lat=69.6, center_lon=19.2, radius_km=2000.0,
    grid_radius_deg=18.0,
    altitude_grid=smoke_altitude_grid,
    edp_samples_path=CACHE / "smoke_test_edp_samples.nc",
    ro_observations_path=CACHE / "smoke_test_ro_observations.nc",
    igs_observations_path=CACHE / "smoke_test_igs_observations.nc",
    isr_file_path=ISR_FILE,
    isr_pca_basis_path=ISR_BASIS,
    styles=default_styles(),
    hyper_params_by_style=default_hyper_params_by_style(),
    output_dir=OUT,
    label="isr_full",
)

result = run_package(cfg)
print("Styles run:", list(result.results_by_style.keys()))
for style, (cycle_result, n_state, wall_time) in result.results_by_style.items():
    final_rmse = (cycle_result.batch_outcomes[-1].rmse_reduction.rmse_analysis
                  if cycle_result.batch_outcomes else float("nan"))
    print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, final_analysis_rmse={final_rmse:.3f}")
for style, matches in result.isr_matches_by_style.items():
    n_with_isr = sum(1 for m in matches if m.isr_density is not None)
    print(f"  ISR match -- {style}: {len(matches)} batches, {n_with_isr} with an ISR match")
print("Output dir:", result.output_dir)
