#!/usr/bin/env python3
"""One-off smoke test for the new ISR comparison wiring in package_run.py
-- reuses step16_package_smoke's cached edp_samples/ro/igs observations
(small n_ensemble, real Tromso 2025-11-18 10:00-11:00 data) so this runs
fast (no fresh IRI2020/IGS processing), and exercises the new
cfg.isr_file_path path end to end against the real TROISR2025_nonan.nc.
"""
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

from Assimilation_Cycle import CycleConfig, run_package

CACHE = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step16_package_smoke")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
OUT = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/isr_wiring_smoke_test"

cfg = CycleConfig(
    start_time="2025-11-18T10:00:00",
    end_time="2025-11-18T11:00:00",
    center_lat=69.6, center_lon=19.2, radius_km=2000.0,
    grid_radius_deg=18.0,
    edp_samples_path=CACHE / "smoke_test_edp_samples.nc",
    ro_observations_path=CACHE / "smoke_test_ro_observations.nc",
    igs_observations_path=CACHE / "smoke_test_igs_observations.nc",
    isr_file_path=ISR_FILE,
    styles=["ANCHOR", "PCA_3D_10ex"],
    hyper_params_by_style={"PCA_3D_10ex": {"retaining_threshold": 1 - 1e-4}},
    output_dir=OUT,
    label="isr_smoke",
)

result = run_package(cfg)
print("Styles run:", list(result.results_by_style.keys()))
for style, matches in result.isr_matches_by_style.items():
    n_with_isr = sum(1 for m in matches if m.isr_density is not None)
    print(f"{style}: {len(matches)} batches, {n_with_isr} with an ISR match")
print("Output dir:", result.output_dir)
