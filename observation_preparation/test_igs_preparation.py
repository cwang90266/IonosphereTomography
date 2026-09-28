import pandas as pd

from observation_preparation import prepare_igs_observations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# COPY THESE FROM YOUR CURRENT MAIN CODE
# ============================================================

IGS_STATIONS_NORDIC = [
    "TRO1",
    "WUTH",
]

RINEX_CACHE = "/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/RINEX_Cache"

center_LAT = 69.6
center_LON = 19.2
ROI_RADIUS_KM = 2000.0

# prepare_igs_observations() now defaults to roi_mode="full_los" (Plan
# Section 8.1/10 step 3), which requires alt_limit_km and has different
# selection behavior than this script's originally-validated results.
# Pinned to "pierce_point" here to keep reproducing exactly that known
# behavior; switch to "full_los" (and set ALT_LIMIT_KM) to try the new
# selection mode against real data.
ROI_MODE = "pierce_point"
ALT_LIMIT_KM = None

# No CDDIS/Earthdata credentials (~/.netrc) are configured in this
# environment, so point directly at the already-cached RINEX/nav/DCB files
# (Plan Section 10 step 5) instead of letting process_igs_station() try to
# fetch/list from CDDIS over the network.
LOCAL_OBS_BY_STATION = {
    "TRO1": f"{RINEX_CACHE}/TRO100NOR_S_20253220000_01D_30S_MO.crx",
    "WUTH": f"{RINEX_CACHE}/WUTH00NOR_R_20253220000_01D_30S_MO.crx",
}
LOCAL_NAV = f"{RINEX_CACHE}/BRDC00IGS_R_20253220000_01D_MN.rnx"
LOCAL_DCB = f"{RINEX_CACHE}/CAS0OPSRAP_20253220000_01D_01D_DCB.BIA"

lo = pd.Timestamp("2025-11-18 10:00:00")
hi = pd.Timestamp("2025-11-18 11:00:00")
OUTPUT_DIR = "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test/IGS_step5"


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
    roi_mode=ROI_MODE,
    alt_limit_km=ALT_LIMIT_KM,

    local_obs_by_station=LOCAL_OBS_BY_STATION,
    local_nav=LOCAL_NAV,
    local_dcb=LOCAL_DCB,

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

for i, obs in enumerate(igs_obs):

    print(f"\n--- IGS #{i} ---")

    print("label      :", obs.get("label"))
    print("station    :", obs.get("leo_id"))
    print("PRN        :", obs.get("prn_id"))
    print("date       :", obs.get("date"))
    print("source     :", obs.get("obs_source"))

    print("TEC shape  :", obs["tec"].shape)
    print("tangent    :", obs["tangent_km"].shape)
    print("rec shape  :", obs["rec_ecef_km"].shape)
    print("GNSS shape :", obs["gnss_ecef_km"].shape)

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

    assert obs["rec_ecef_km"].shape == (3, 1)
    assert obs["gnss_ecef_km"].shape == (3, 1)

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