import pandas as pd
from pathlib import Path
from observation_preparation.prepare_igs_observations import prepare_igs_observations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# COPY THESE FROM YOUR CURRENT MAIN CODE
# ============================================================

IGS_STATIONS_NORDIC = [
    "TRO1",
    "WUTH",
]

RINEX_CACHE = "/home/pin/Desktop/tomography_project/Data/RINEX_cache"

center_LAT = 69.6
center_LON = 19.2
ROI_RADIUS_KM = 2000.0
lo = pd.Timestamp("2025-11-18 10:00:00")
hi = pd.Timestamp("2025-11-18 11:00:00")
OUTPUT_DIR = "Figures/preparation_test/IGS"


print("\n==========================================")
print(" TESTING IGS OBSERVATION PREPARATION")
print("==========================================")

print("stations :", IGS_STATIONS_NORDIC)
print("cache    :", RINEX_CACHE)
print("window   :", lo, "->", hi)


igs_obs, report = prepare_igs_observations(
    stations=IGS_STATIONS_NORDIC,
    cache_dir=RINEX_CACHE,

    start_time=lo,
    end_time=hi,

    center_lat=center_LAT,
    center_lon=center_LON,
    radius_km=ROI_RADIUS_KM,

    rinex_version=3,
    use_iri=False,

    min_valid_epochs=50,
    max_rays_per_arc=200,

    epoch_mode="center",

    return_report=True,
    output_dir=OUTPUT_DIR,
    output_prefix="igs_observations",
)


# ============================================================
# REPORT
# ============================================================

print("\n=============== IGS REPORT ===============")

for key, value in report.items():
    if key != "station_runs":
        print(f"{key}: {value}")



# ============================================================
# OUTPUT CHECK
# ============================================================

print("\n=============== OUTPUT =================")

print("Number of final IGS arcs:", len(igs_obs))
print("CSV/plot directory:", Path(OUTPUT_DIR).resolve())
assert Path(OUTPUT_DIR, "igs_observations.csv").is_file()

for i, obs in enumerate(igs_obs):

    print(f"\n--- IGS #{i} ---")

    print("label      :", obs.get("label"))
    print("station    :", obs.get("leo_id"))
    print("PRN        :", obs.get("prn_id"))
    print("date       :", obs.get("date"))
    print("source     :", obs.get("obs_source"))

    print("TEC shape  :", obs["tec"].shape)
    print("tangent    :", obs["tangent_km"].shape)
    print("LEO shape  :", obs["LEO"].shape)
    print("GNSS shape :", obs["GNSS"].shape)

    print("TEC        :", obs["tec"])
    print("elevation  :", obs.get("elev_deg"))
    print("IPP lat    :", obs.get("ipp_lat"))
    print("IPP lon    :", obs.get("ipp_lon"))
    print("date       :", obs.get("date"))
    print("arc time s :", obs.get("arc_time_sec"))
    print("UTC hour   :", obs.get("time_utc_h"))

    # ========================================================
    # Assertions for epoch_mode="center"
    # ========================================================

    assert len(obs["tec"]) == 1

    assert len(obs["tangent_km"]) == 1

    assert obs["LEO"].shape == (3, 1)
    assert obs["GNSS"].shape == (3, 1)

    assert obs["tec"][0] > 0

    if obs.get("elev_deg") is not None:
        assert len(obs["elev_deg"]) == 1

    if obs.get("ipp_lat") is not None:
        assert len(obs["ipp_lat"]) == 1

    if obs.get("ipp_lon") is not None:
        assert len(obs["ipp_lon"]) == 1


print("\n==========================================")
print(" IGS PREPARATION TEST PASSED")
print("==========================================")
