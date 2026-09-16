#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone test for RO observation preparation + Abel inversion.

Run from the tomography project root:
    python -m observation_preparation.test_ro_preparation
"""

from pathlib import Path
import numpy as np
import pandas as pd

from observation_preparation.prepare_ro_observations import prepare_ro_observations


# ============================================================
# USER TEST SETTINGS
# ============================================================
PODTC_DIR = Path(
    "/home/pin/Desktop/tomography_project/piq_data/podTc2/2025.322/"
)

# Tromsø-centered example used in the recent RO tests.
CENTER_LAT = 69.6
CENTER_LON = 19.2
RADIUS_KM = 2000.0

START_TIME = pd.Timestamp("2025-11-18 10:00:00")
END_TIME = pd.Timestamp("2025-11-18 11:00:00")

MIN_VALID_RAYS = 50
MAX_RAYS_PER_OCCULTATION = 200

# Set to an integer if you want to limit the number of occultations.
MAX_OCCULTATIONS = None
OUTPUT_DIR = Path("Figures/preparation_test/RO")


# ============================================================
# RUN
# ============================================================
print("\n==========================================")
print(" TESTING RO OBSERVATION PREPARATION + ABEL")
print("==========================================")
print("directory :", PODTC_DIR)
print("center    :", CENTER_LAT, CENTER_LON)
print("ROI km    :", RADIUS_KM)
print("window    :", START_TIME, "->", END_TIME)

ro_obs, report = prepare_ro_observations(
    PODTC_DIR,
    center_lat=CENTER_LAT,
    center_lon=CENTER_LON,
    radius_km=RADIUS_KM,
    start_time=START_TIME,
    end_time=END_TIME,
    max_occultations=MAX_OCCULTATIONS,
    min_valid_rays=MIN_VALID_RAYS,
    max_rays_per_occultation=MAX_RAYS_PER_OCCULTATION,
    include_abel=True,
    return_report=True,
    output_dir=OUTPUT_DIR,
    output_prefix="ro_observations",
)


# ============================================================
# REPORT
# ============================================================
print("\n================ RO REPORT ================")
for key in (
    "source",
    "podtc_dir",
    "n_files_scanned",
    "n_files_after_roi_time_filter",
    "n_files_after_occultation_selection",
    "n_parser_reject",
    "n_clean_observations",
    "n_output_rays_total",
    "include_abel",
    "n_abel_success",
    "n_abel_failed",
    "min_valid_rays",
    "max_rays_per_occultation",
    "max_occultations",
):
    print(f"{key}: {report.get(key)}")

print("\n============== FILE RESULTS ===============")
for item in report.get("per_file", []):
    print(item)


# ============================================================
# OUTPUT CONTENT + ABEL CHECK
# ============================================================
print("\n================ OUTPUT ===================")
print("Number of final RO occultations:", len(ro_obs))
print("CSV/plots directory:", OUTPUT_DIR.resolve())
assert (OUTPUT_DIR / "ro_observations.csv").is_file()

abel_ok = 0
abel_failed = 0

for i, obs in enumerate(ro_obs):
    print(f"\n--- RO #{i} ---")
    print("label      :", obs.get("label"))
    print("LEO        :", obs.get("leo_id"))
    print("PRN        :", obs.get("prn_id"))
    print("date       :", obs.get("date"))
    print("source     :", obs.get("obs_source"))
    print("occ type   :", obs.get("occ_type"))
    print("TEC shape  :", np.shape(obs.get("tec")))
    print("tangent    :", np.shape(obs.get("tangent_km")))
    print("LEO shape  :", np.shape(obs.get("LEO")))
    print("GNSS shape :", np.shape(obs.get("GNSS")))

    tec = np.asarray(obs.get("tec", []), dtype=float)
    tan = np.asarray(obs.get("tangent_km", []), dtype=float)
    if tec.size:
        print("TEC range  :", float(np.nanmin(tec)), "->", float(np.nanmax(tec)), "TECU")
    if tan.size:
        print("tan alt km :", float(np.nanmin(tan)), "->", float(np.nanmax(tan)))

    # ---------------- Abel output ----------------
    abel = obs.get("abel")
    if abel is None:
        abel_failed += 1
        print("Abel       : FAILED / None")
        continue

    abel_ne = np.asarray(abel.get("Ne", []), dtype=float)
    abel_alt = np.asarray(abel.get("alt_km", []), dtype=float)

    print("Abel       : OK")
    print("Abel Ne    :", abel_ne.shape)
    print("Abel alt   :", abel_alt.shape)

    if abel_ne.size:
        print(
            "Abel Ne rng:",
            float(np.nanmin(abel_ne)),
            "->",
            float(np.nanmax(abel_ne)),
            "m^-3",
        )
    if abel_alt.size:
        print(
            "Abel alt rng:",
            float(np.nanmin(abel_alt)),
            "->",
            float(np.nanmax(abel_alt)),
            "km",
        )

    # Primary Abel Ne and altitude must be paired.
    assert abel_ne.ndim == 1, "Abel Ne must be 1-D"
    assert abel_alt.ndim == 1, "Abel altitude must be 1-D"
    assert len(abel_ne) == len(abel_alt), (
        f"Abel Ne/alt length mismatch for {obs.get('label')}: "
        f"{len(abel_ne)} vs {len(abel_alt)}"
    )
    assert len(abel_ne) > 0, f"Empty Abel profile for {obs.get('label')}"

    # The inverter sorts the primary profile upward in altitude.
    finite_alt = abel_alt[np.isfinite(abel_alt)]
    if len(finite_alt) > 1:
        assert np.all(np.diff(finite_alt) >= 0), (
            f"Abel altitude is not ascending for {obs.get('label')}"
        )

    abel_ok += 1


# ============================================================
# STRUCTURAL ASSERTIONS
# ============================================================
assert report["include_abel"] is True
assert report["n_clean_observations"] == len(ro_obs)
assert report["n_output_rays_total"] == sum(len(x["tec"]) for x in ro_obs)

for obs in ro_obs:
    n = len(obs["tec"])
    assert n >= MIN_VALID_RAYS
    assert n <= MAX_RAYS_PER_OCCULTATION
    assert len(obs["tangent_km"]) == n
    assert np.shape(obs["LEO"]) == (3, n)
    assert np.shape(obs["GNSS"]) == (3, n)
    assert np.all(np.isfinite(obs["tec"]))
    assert np.all(np.asarray(obs["tec"]) > 0)

assert report["n_abel_success"] == abel_ok
assert report["n_abel_failed"] == abel_failed

# If there are accepted RO occultations, require at least one successful Abel
# profile so the test does not print PASS when Abel is completely broken.
if len(ro_obs) > 0:
    assert abel_ok > 0, "RO observations were accepted, but every Abel inversion failed"

print("\n=============== ABEL SUMMARY ==============")
print("successful Abel profiles:", abel_ok)
print("failed Abel profiles    :", abel_failed)

print("\n==========================================")
print(" RO + ABEL PREPARATION TEST PASSED")
print("==========================================")
