#!/usr/bin/env python3
"""One-off smoke test for the new impact-vector wiring in package_run.py
(Impact_Vector_Plan.md Section 11 step 8) -- reuses step16_package_smoke's
cached edp_samples/ro/igs observations (small n_ensemble, real Tromso
2025-11-18 10:00-11:00 data) so this runs fast (no fresh IRI2020/IGS
processing), and exercises cfg.compute_impact_vectors end to end: a
nonlinear style (ANCHOR) and a dedicated `raw`-style run (raw isn't a
default style, but the plan's test 6 explicitly wants a linear-decode
case too), both with the real TROISR2025_nonan.nc ISR station auto-added
as a query point via cfg.isr_file_path.
"""
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

from Assimilation_Cycle import CycleConfig, run_package

CACHE = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/step16_package_smoke")
ISR_FILE = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc"
OUT = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test/impact_vector_wiring_smoke_test"

cfg = CycleConfig(
    start_time="2025-11-18T10:00:00",
    end_time="2025-11-18T11:00:00",
    center_lat=69.6, center_lon=19.2, radius_km=2000.0,
    grid_radius_deg=18.0,
    edp_samples_path=CACHE / "smoke_test_edp_samples.nc",
    ro_observations_path=CACHE / "smoke_test_ro_observations.nc",
    igs_observations_path=CACHE / "smoke_test_igs_observations.nc",
    isr_file_path=ISR_FILE,
    styles=["ANCHOR", "raw"],
    hyper_params_by_style={},
    compute_impact_vectors=True,
    output_dir=OUT,
    label="iv_smoke",
)

result = run_package(cfg)
print("Styles run:", list(result.results_by_style.keys()))

out = Path(result.output_dir)
for style in result.results_by_style:
    iv_dir = out / f"iv_smoke_{style}" / "impact_vectors"
    if iv_dir.exists():
        for point_dir in sorted(iv_dir.iterdir()):
            pngs = sorted(point_dir.glob("*.png"))
            print(f"{style} / {point_dir.name}: {len(pngs)} plots -> {[p.name for p in pngs]}")
    else:
        print(f"{style}: NO impact_vectors directory produced")

cross_dir = out / "iv_smoke_impact_vectors_cross_style"
if cross_dir.exists():
    for point_dir in sorted(cross_dir.iterdir()):
        pngs = sorted(point_dir.glob("*.png"))
        print(f"cross-style / {point_dir.name}: {len(pngs)} plots -> {[p.name for p in pngs]}")
else:
    print("NO cross-style impact_vectors directory produced")

print("Output dir:", result.output_dir)
