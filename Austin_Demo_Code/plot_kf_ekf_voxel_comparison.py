# plot_kf_ekf_voxel_comparison.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Input files
# ============================================================

Time_slot = "2025-11-22_1100"
observation_mode = "ro_only"

BASE_DIR = Path(
    "/home/pin/Desktop/tomography_project/"
    "Figures/ISR_DA_Comparison/OUTPUT/bin_all/"
    f"{Time_slot}/{observation_mode}/voxel_output"
)

BASE_DIR_IRINE = Path(
    "/home/pin/Desktop/tomography_project/"
    "Figures/ISR_DA_Comparison/OUTPUT/bin_all/"
    f"{Time_slot}/{observation_mode}/IRI_8param_profiles"
)

EKF_COLUMNS_FILE = BASE_DIR / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_EKF_columns.csv"
)

EKF_VOXELS_FILE = BASE_DIR / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_"
    "EKF_reconstructed_voxels.csv"
)

KF_VOXELS_FILE = BASE_DIR / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_KF_voxels.csv"
)

OBS_COUNT_FILE = BASE_DIR / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_"
    "KF_voxel_observation_counts.csv"
)

TEC_VOXEL_FILE = BASE_DIR / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_"
    "KF_EKF_prior_TEC_voxel_contributions.csv"
)

PRIOR_NE_COMPARE_FILE = BASE_DIR / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_"
    "KF_EKF_prior_Ne_voxel_comparison.csv"
)

IRI_8PARAM_FILE = BASE_DIR_IRINE / (
    f"{Time_slot}_{observation_mode}_parametric_ekf_"
    "IRI_vs_8param.csv"
)


OUTPUT_DIR = BASE_DIR / "comparison_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

ekf_columns = pd.read_csv(EKF_COLUMNS_FILE)
ekf_voxels = pd.read_csv(EKF_VOXELS_FILE)
kf_voxels = pd.read_csv(KF_VOXELS_FILE)
obs_count = pd.read_csv(OBS_COUNT_FILE)
tec_voxels = pd.read_csv(TEC_VOXEL_FILE)
prior_ne_compare = pd.read_csv(PRIOR_NE_COMPARE_FILE)
iri_8param = pd.read_csv(IRI_8PARAM_FILE)

print("TEC voxel contributions:", tec_voxels.shape)
print("EKF columns:", ekf_columns.shape)
print("EKF voxels :", ekf_voxels.shape)
print("KF voxels  :", kf_voxels.shape)
print("Observation counts:", obs_count.shape)

label_fontsize = 20
MODE = "RO"

# ============================================================
# Select one vertical column
# ============================================================

# Select by column index.
# Change this number to examine another horizontal location.
COLUMN_INDEX = 0

if COLUMN_INDEX >= len(ekf_columns):
    raise IndexError(
        f"COLUMN_INDEX={COLUMN_INDEX} exceeds "
        f"{len(ekf_columns) - 1}"
    )

column = ekf_columns.iloc[COLUMN_INDEX]

target_lat = column["lat"]
target_lon = column["lon"]

print(f"Selected column: {COLUMN_INDEX}")
print(f"Latitude : {target_lat:.4f}")
print(f"Longitude: {target_lon:.4f}")


def select_profile(df, lat, lon):
    """Select the vertical profile at the requested horizontal location."""

    mask = (
        np.isclose(df["lat"], lat)
        & np.isclose(df["lon"], lon)
    )

    profile = df.loc[mask].sort_values("alt_km").copy()

    if profile.empty:
        raise ValueError(
            f"No profile found at lat={lat}, lon={lon}"
        )

    return profile

def extract_hmf2(profile):
    """Return F2-peak altitude from one Ne profile."""
    ne = profile["prior_ne_m3"].to_numpy(dtype=float)
    alt = profile["alt_km"].to_numpy(dtype=float)

    valid = np.isfinite(ne) & np.isfinite(alt)
    if not valid.any():
        return np.nan

    idx = np.nanargmax(ne[valid])
    return alt[valid][idx]


def rmse_below_alt(profile_a, profile_b, max_alt):
    """RMSE between two profiles below a common altitude."""
    merged = profile_a.merge(
        profile_b,
        on=["lat", "lon", "alt_km"],
        suffixes=("_kf", "_ekf"),
    )

    valid = (
        (merged["alt_km"] <= max_alt)
        & np.isfinite(merged["prior_ne_m3_kf"])
        & np.isfinite(merged["prior_ne_m3_ekf"])
    )

    if not valid.any():
        return np.nan

    diff = (
        merged.loc[valid, "prior_ne_m3_kf"]
        - merged.loc[valid, "prior_ne_m3_ekf"]
    )

    rmse = float(np.sqrt(np.mean(diff**2)))

    return rmse


ekf_profile = select_profile(
    ekf_voxels,
    target_lat,
    target_lon,
)

kf_profile = select_profile(
    kf_voxels,
    target_lat,
    target_lon,
)

obs_profile = select_profile(
    obs_count,
    target_lat,
    target_lon,
)

kf_hmf2 = extract_hmf2(kf_profile)
ekf_hmf2 = extract_hmf2(ekf_profile)

# Common altitude range for a fair comparison
common_topside = 800.0 

prior_rmse_below = rmse_below_alt(
    kf_profile,
    ekf_profile,
    common_topside,
)

# Posterior RMSE
merged_profile = kf_profile.merge(
    ekf_profile,
    on=["lat", "lon", "alt_km"],
    suffixes=("_kf", "_ekf"),
)

valid_post = (
    (merged_profile["alt_km"] <= common_topside)
    & np.isfinite(merged_profile["post_ne_m3_kf"])
    & np.isfinite(merged_profile["post_ne_m3_ekf"])
)

post_rmse_below = np.sqrt(
    np.mean(
        (
            merged_profile.loc[valid_post, "post_ne_m3_kf"]
            - merged_profile.loc[valid_post, "post_ne_m3_ekf"]
        ) ** 2
    )
)

# ============================================================
# Plot 1:
# EKF 8 parameters and reconstructed Ne profile
# ============================================================

parameter_names = [
    "log10(NmF2)",
    "hmF2",
    "H0",
    "gamma",
    "B0",
    "B1",
    "log10(NmE)",
    "hmE",
]

prior_parameters = np.array(
    [column[f"prior_{name}"] for name in parameter_names]
)

post_parameters = np.array(
    [column[f"post_{name}"] for name in parameter_names]
)

fig = plt.figure(figsize=(16, 9))

ax_parameters = fig.add_subplot(1, 2, 1)
x = np.arange(len(parameter_names))
width = 0.38

ax_parameters.bar(
    x - width / 2,
    prior_parameters,
    width,
    label="Prior",
)

ax_parameters.bar(
    x + width / 2,
    post_parameters,
    width,
    label="Posterior",
)

ax_parameters.set_xticks(x)
ax_parameters.set_xticklabels(
    parameter_names,
    rotation=45,
    ha="right",
    fontsize=label_fontsize
)
ax_parameters.set_ylabel("Parameter value", fontsize=label_fontsize)
ax_parameters.tick_params(axis='y', labelsize=label_fontsize)
ax_parameters.set_title("EKF 8-Parameter State", fontsize=label_fontsize+5)
ax_parameters.grid(True, axis="y", alpha=0.3)
ax_parameters.legend()

ax_profile = fig.add_subplot(1, 2, 2)

ax_profile.plot(
    ekf_profile["prior_ne_m3"],
    ekf_profile["alt_km"],
    linewidth=2,
    label="EKF prior reconstructed Ne",
)

ax_profile.plot(
    ekf_profile["post_ne_m3"],
    ekf_profile["alt_km"],
    linewidth=2,
    label="EKF posterior reconstructed Ne",
)

ax_profile.set_xscale("log")
ax_profile.set_xlim(1e7, 1e13)
ax_profile.set_xlabel(r"Ne (m$^{-3}$)", fontsize=label_fontsize)
# make xtick font size larger
ax_profile.tick_params(axis='x', labelsize=label_fontsize)
ax_profile.tick_params(axis='y', labelsize=label_fontsize)
ax_profile.set_ylabel("Altitude (km)", fontsize=label_fontsize)
ax_profile.set_title("Ne Reconstructed From 8 Parameters", fontsize=label_fontsize+5)
ax_profile.grid(True, which="both", alpha=0.3)
ax_profile.legend()
# make legend font size larger
for ax in [ax_parameters, ax_profile]:
    ax.legend(fontsize=label_fontsize)

fig.suptitle(
    f"EKF Parameter-to-Ne Transition\n"
    f"Column {COLUMN_INDEX}: "
    f"lat={target_lat:.2f}°, lon={target_lon:.2f}°", fontsize=label_fontsize+5
)

fig.tight_layout()

plot1_path = OUTPUT_DIR / (
    f"column_{COLUMN_INDEX:03d}_"
    "EKF_parameters_to_Ne.png"
)

fig.savefig(plot1_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {plot1_path}")


# ============================================================
# Plot 2:
# KF versus EKF reconstructed Ne
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 8),
    sharey=True,
)

# Prior comparison
axes[0].plot(
    kf_profile["prior_ne_m3"],
    kf_profile["alt_km"],
    linewidth=2,
    label=f"KF prior",
)

axes[0].plot(
    ekf_profile["prior_ne_m3"],
    ekf_profile["alt_km"],
    linewidth=2,
    linestyle="--",
    label=(
    f"EKF prior\n"
    f"RMSE below {common_topside:.0f} km = "
    f"{prior_rmse_below:.2e} m$^{{-3}}$",
    )
)

axes[0].set_xscale("log")
axes[0].set_xlim(1e7, 1e13)
axes[0].set_xlabel(r"Ne(m$^{-3}$)", fontsize=label_fontsize)
axes[0].set_ylabel("Altitude (km)", fontsize=label_fontsize)
axes[0].tick_params(axis='x', labelsize=label_fontsize)
axes[0].tick_params(axis='y', labelsize=label_fontsize)
axes[0].set_title("Prior Ne Comparison", fontsize=label_fontsize+5)
axes[0].grid(True, which="both", alpha=0.3)
for ax in axes:
    ax.axhline(
        common_topside,
        linestyle=":",
        linewidth=1.5,
        label=f"Cutoff = {common_topside:.0f} km",
    )

axes[0].legend()
for ax in axes:
    ax.legend(fontsize=label_fontsize-5)

# Posterior comparison
axes[1].plot(
    kf_profile["post_ne_m3"],
    kf_profile["alt_km"],
    linewidth=2,
    label="KF posterior",
)

axes[1].plot(
    ekf_profile["post_ne_m3"],
    ekf_profile["alt_km"],
    linewidth=2,
    linestyle="--",
    label="EKF posterior",
)

axes[1].set_xscale("log")
axes[1].set_xlim(1e7, 1e13)
axes[1].set_xlabel(r"Ne (m$^{-3}$)", fontsize=label_fontsize)
axes[1].tick_params(axis='x', labelsize=label_fontsize)
axes[1].tick_params(axis='y', labelsize=label_fontsize)
axes[1].set_title("Posterior Ne Comparison", fontsize=label_fontsize+5)
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()
# legend fontsize larger
for ax in axes:
    ax.legend(fontsize=label_fontsize)
    # place legend in middle left of the plot
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))

fig.suptitle(
    f"KF and EKF Ne Comparison\n"
    f"Column {COLUMN_INDEX}: "
    f"lat={target_lat:.2f}°, lon={target_lon:.2f}°", fontsize=label_fontsize+5
)

fig.tight_layout()

plot2_path = OUTPUT_DIR / (
    f"column_{COLUMN_INDEX:03d}_"
    "KF_EKF_Ne_comparison.png"
)

fig.savefig(plot2_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {plot2_path}")

# ============================================================
# Plot 3:
# Number of observations crossing each altitude voxel
# ============================================================

fig, ax = plt.subplots(figsize=(8, 9))

# ax.step(
#     obs_profile["observation_count"],
#     obs_profile["alt_km"],
#     where="mid",
#     linewidth=2,
# )

ax.scatter(
    obs_profile["observation_count"],
    obs_profile["alt_km"],
    s=35,
)

ax.set_xlabel("Number of observation rays", fontsize=label_fontsize)
ax.set_ylabel("Altitude (km)", fontsize=label_fontsize)
ax.tick_params(axis='x', labelsize=label_fontsize)
ax.tick_params(axis='y', labelsize=label_fontsize)
ax.set_title(
    "Observation Count per KF Voxel\n"
    f"Column {COLUMN_INDEX}: "
    f"lat={target_lat:.2f}°, lon={target_lon:.2f}°", fontsize=label_fontsize+5
)

ax.set_xlim(left=0)
ax.grid(True, alpha=0.3)

fig.tight_layout()

plot3_path = OUTPUT_DIR / (
    f"column_{COLUMN_INDEX:03d}_"
    "voxel_observation_count.png"
)

fig.savefig(plot3_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {plot3_path}")

# ============================================================
# Plot 4:
# KF-EKF RMSE as a function of altitude across all columns
# ============================================================

all_voxels = kf_voxels.merge(
    ekf_voxels,
    on=["lat", "lon", "alt_km"],
    suffixes=("_kf", "_ekf"),
)

altitude_stats = []

for alt, group in all_voxels.groupby("alt_km"):
    prior_diff = (
        group["prior_ne_m3_kf"]
        - group["prior_ne_m3_ekf"]
    )

    post_diff = (
        group["post_ne_m3_kf"]
        - group["post_ne_m3_ekf"]
    )

    altitude_stats.append({
        "alt_km": alt,
        "prior_rmse": np.sqrt(np.nanmean(prior_diff**2)),
        "posterior_rmse": np.sqrt(np.nanmean(post_diff**2)),
        "prior_mae": np.nanmean(np.abs(prior_diff)),
        "posterior_mae": np.nanmean(np.abs(post_diff)),
    })

altitude_stats = pd.DataFrame(altitude_stats).sort_values("alt_km")

worst_prior = altitude_stats.loc[
    altitude_stats["prior_rmse"].idxmax()
]

worst_post = altitude_stats.loc[
    altitude_stats["posterior_rmse"].idxmax()
]

print(
    f"Worst prior altitude: {worst_prior['alt_km']:.1f} km, "
    f"RMSE={worst_prior['prior_rmse']:.3e} m^-3"
)

print(
    f"Worst posterior altitude: {worst_post['alt_km']:.1f} km, "
    f"RMSE={worst_post['posterior_rmse']:.3e} m^-3"
)

altitude_stats.to_csv(
    OUTPUT_DIR / "KF_EKF_RMSE_by_altitude.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(9, 8))

ax.plot(
    altitude_stats["prior_rmse"],
    altitude_stats["alt_km"],
    linewidth=2,
    label="Prior KF–EKF RMSE",
)

ax.plot(
    altitude_stats["posterior_rmse"],
    altitude_stats["alt_km"],
    linewidth=2,
    label="Posterior KF–EKF RMSE",
)

ax.set_xscale("log")
ax.set_xlabel(r"RMSE across columns (m$^{-3}$)")
ax.set_ylabel("Altitude (km)")
ax.set_title("KF–EKF Difference by Altitude")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "KF_EKF_RMSE_by_altitude.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# Plot 5:
# Columns with largest KF-EKF prior difference below hmF2
# ============================================================

column_results = []

for column_index, column in ekf_columns.iterrows():
    lat = column["lat"]
    lon = column["lon"]

    kf_prof = select_profile(kf_voxels, lat, lon)
    ekf_prof = select_profile(ekf_voxels, lat, lon)

    kf_peak = extract_hmf2(kf_prof)
    ekf_peak = extract_hmf2(ekf_prof)
    cutoff = np.nanmin([kf_peak, ekf_peak])

    rmse = rmse_below_alt(kf_prof, ekf_prof, cutoff)

    column_results.append({
        "column_index": column_index,
        "lat": lat,
        "lon": lon,
        "common_topside_km": cutoff,
        "prior_rmse_below_hmf2": rmse,
    })

column_results = pd.DataFrame(column_results).sort_values(
    "prior_rmse_below_hmf2",
    ascending=False,
)

column_results.to_csv(
    OUTPUT_DIR / "KF_EKF_RMSE_by_column.csv",
    index=False,
)

N_EXAMPLES = 3
worst_columns = column_results.head(N_EXAMPLES)

fig, axes = plt.subplots(
    1,
    N_EXAMPLES,
    figsize=(5 * N_EXAMPLES, 8),
    sharey=True,
)

if N_EXAMPLES == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, worst_columns.iterrows()):
    lat = row["lat"]
    lon = row["lon"]
    cutoff = row["common_topside_km"]

    kf_prof = select_profile(kf_voxels, lat, lon)
    ekf_prof = select_profile(ekf_voxels, lat, lon)

    ax.plot(
        kf_prof["prior_ne_m3"],
        kf_prof["alt_km"],
        linewidth=2,
        label="KF prior"
    )

    ax.plot(
        ekf_prof["prior_ne_m3"],
        ekf_prof["alt_km"],
        linewidth=2,
        linestyle="--",
        label="EKF prior"
    )

    # ax.axhline(
    #     cutoff + 150.0, # reflect the modification in diagnose_priorstates.py to extend the comparison up to ~450 km
    #     linestyle=":",
    #     linewidth=2,
    # )

    ax.set_xscale("log")
    ax.set_xlim(1e7, 1e13)
    ax.set_xlabel(r"Ne (m$^{-3}$)", fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.set_title(
        f"Column {int(row['column_index'])}\n"
        f"lat={lat:.1f}°, lon={lon:.1f}°\n"
        f"RMSE={row['prior_rmse_below_hmf2']:.2e}", fontsize=label_fontsize+5
    )
    ax.grid(True, which="both", alpha=0.3)

axes[0].set_ylabel("Altitude (km)", fontsize=label_fontsize)
axes[0].legend()

fig.suptitle("Largest KF–EKF Prior Differences Below hmF2", fontsize=label_fontsize+5)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "largest_KF_EKF_prior_RMSE_examples.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# Figure 6:
# KF-prior and EKF-prior RMSE versus ISR truth
# Two maps with the same color scale
# ============================================================

import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

TRO_LAT = 69.583
TRO_LON = 19.21

# Change this to your actual saved window_edps file
ISR_PKL_FILE = Path(
    "/home/pin/Desktop/tomography_project/Data/WINDOW_EDPS/"
    "2025-11-22_1035_binall_window_edps.pkl"
)

# Minimum valid ISR density
ISR_NE_MIN = 1e7


# ------------------------------------------------------------
# Load ISR profiles
# ------------------------------------------------------------

with open(ISR_PKL_FILE, "rb") as f:
    window_edps = pickle.load(f)

print(f"Loaded {len(window_edps)} ISR profiles from:")
print(ISR_PKL_FILE)


# ------------------------------------------------------------
# Build one median ISR truth profile
# ------------------------------------------------------------

# Use the KF/EKF model altitude grid
model_alt_grid = np.sort(
    kf_voxels["alt_km"].unique().astype(float)
)

isr_profiles_interp = []

for edp in window_edps:
    isr_alt_i = np.asarray(edp["alt_km"], dtype=float)
    isr_ne_i = np.asarray(edp["ne_m3"], dtype=float)

    valid_i = (
        np.isfinite(isr_alt_i)
        & np.isfinite(isr_ne_i)
        & (isr_ne_i > ISR_NE_MIN)
    )

    if valid_i.sum() < 5:
        continue

    order = np.argsort(isr_alt_i[valid_i])
    alt_valid = isr_alt_i[valid_i][order]
    ne_valid = isr_ne_i[valid_i][order]

    isr_on_model_grid = np.interp(
        model_alt_grid,
        alt_valid,
        ne_valid,
        left=np.nan,
        right=np.nan,
    )

    isr_profiles_interp.append(isr_on_model_grid)

if not isr_profiles_interp:
    raise ValueError("No valid ISR profiles were available for Figure 6.")

isr_profiles_interp = np.vstack(isr_profiles_interp)

# Median truth from all ISR scans in the window
isr_median_ne = np.nanmedian(
    isr_profiles_interp,
    axis=0,
)


# ------------------------------------------------------------
# Find ISR hmF2
# ------------------------------------------------------------

valid_isr_peak = (
    np.isfinite(isr_median_ne)
    & (isr_median_ne > ISR_NE_MIN)
)

if not valid_isr_peak.any():
    raise ValueError("Cannot determine ISR hmF2 from the median profile.")

isr_peak_idx = np.nanargmax(
    np.where(valid_isr_peak, isr_median_ne, np.nan)
)

isr_hmf2_km = model_alt_grid[isr_peak_idx]

print(f"Median ISR hmF2: {isr_hmf2_km:.1f} km")


# ------------------------------------------------------------
# Calculate prior RMSE for every grid column
# ------------------------------------------------------------

rmse_map_rows = []

for column_index, column in ekf_columns.iterrows():

    lat = float(column["lat"])
    lon = float(column["lon"])

    kf_prof = select_profile(
        kf_voxels,
        lat,
        lon,
    )

    ekf_prof = select_profile(
        ekf_voxels,
        lat,
        lon,
    )

    # Ensure profiles are aligned with the model altitude grid
    kf_prof = kf_prof.sort_values("alt_km")
    ekf_prof = ekf_prof.sort_values("alt_km")

    kf_alt = kf_prof["alt_km"].to_numpy(dtype=float)
    kf_ne = kf_prof["prior_ne_m3"].to_numpy(dtype=float)

    ekf_alt = ekf_prof["alt_km"].to_numpy(dtype=float)
    ekf_ne = ekf_prof["prior_ne_m3"].to_numpy(dtype=float)

    # Interpolate in case the exported altitude order differs
    kf_ne_grid = np.interp(
        model_alt_grid,
        kf_alt,
        kf_ne,
        left=np.nan,
        right=np.nan,
    )

    ekf_ne_grid = np.interp(
        model_alt_grid,
        ekf_alt,
        ekf_ne,
        left=np.nan,
        right=np.nan,
    )

    # Same mask for both filters
    valid = (
        (model_alt_grid <= isr_hmf2_km)
        & np.isfinite(isr_median_ne)
        & (isr_median_ne > ISR_NE_MIN)
        & np.isfinite(kf_ne_grid)
        & np.isfinite(ekf_ne_grid)
    )

    if valid.sum() < 3:
        kf_rmse = np.nan
        ekf_rmse = np.nan
    else:
        kf_rmse = float(np.sqrt(np.mean(
            (kf_ne_grid[valid] - isr_median_ne[valid]) ** 2
        )))

        ekf_rmse = float(np.sqrt(np.mean(
            (ekf_ne_grid[valid] - isr_median_ne[valid]) ** 2
        )))

    rmse_map_rows.append({
        "column_index": int(column_index),
        "lat": lat,
        "lon": lon,
        "kf_prior_rmse_vs_isr": kf_rmse,
        "ekf_prior_rmse_vs_isr": ekf_rmse,
        "isr_hmf2_km": isr_hmf2_km,
        "n_altitude_levels": int(valid.sum()),
    })

rmse_map_df = pd.DataFrame(rmse_map_rows)

rmse_csv_path = OUTPUT_DIR / (
    "KF_EKF_prior_RMSE_vs_ISR_by_column.csv"
)

rmse_map_df.to_csv(
    rmse_csv_path,
    index=False,
)

print(f"Saved: {rmse_csv_path}")


# ------------------------------------------------------------
# Shared color scale
# ------------------------------------------------------------

all_rmse = np.concatenate([
    rmse_map_df["kf_prior_rmse_vs_isr"].to_numpy(dtype=float),
    rmse_map_df["ekf_prior_rmse_vs_isr"].to_numpy(dtype=float),
])

all_rmse = all_rmse[
    np.isfinite(all_rmse)
    & (all_rmse >= 0)
]

if all_rmse.size == 0:
    raise ValueError("No finite RMSE values available for Figure 6.")

# Exact common limits for both figures
shared_vmin = float(np.nanmin(all_rmse))
shared_vmax = float(np.nanmax(all_rmse))

if shared_vmax <= shared_vmin:
    shared_vmax = shared_vmin + 1.0

shared_norm = Normalize(
    vmin=shared_vmin,
    vmax=shared_vmax,
)

print(
    f"Shared color scale: "
    f"{shared_vmin:.3e} to {shared_vmax:.3e} m^-3"
)


# ------------------------------------------------------------
# Map plotting function
# ------------------------------------------------------------

def plot_prior_rmse_map(
    rmse_column,
    title,
    output_filename,
):
    plot_df = rmse_map_df[
        np.isfinite(rmse_map_df[rmse_column])
    ].copy()

    projection = ccrs.Orthographic(
        central_longitude=TRO_LON,
        central_latitude=TRO_LAT,
    )

    fig = plt.figure(figsize=(12, 11))
    ax = plt.axes(projection=projection)

    ax.set_global()

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.90",
        zorder=0,
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        zorder=0,
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.9,
        zorder=2,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5,
        linestyle=":",
        zorder=2,
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        linestyle="--",
        alpha=0.5,
        zorder=1,
    )

    scatter = ax.scatter(
        plot_df["lon"],
        plot_df["lat"],
        c=plot_df[rmse_column],
        s=75,
        cmap="jet",
        norm=shared_norm,
        edgecolors="black",
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Tromsø ISR
    ax.scatter(
        TRO_LON,
        TRO_LAT,
        marker="*",
        s=400,
        color="red",
        edgecolors="black",
        linewidths=1.2,
        transform=ccrs.PlateCarree(),
        label="Tromsø ISR",
        zorder=10,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        orientation="vertical",
        shrink=0.72,
        pad=0.04, cmap="jet"
    )

    cbar.set_label(
        r"Prior Ne RMSE vs. ISR below ISR hmF2 (m$^{-3}$)",
        fontsize=label_fontsize,
    )

    cbar.ax.tick_params(
        labelsize=label_fontsize - 2
    )

    formatter = ScalarFormatter(
        useMathText=True
    )
    formatter.set_powerlimits((0, 0))
    cbar.formatter = formatter
    cbar.update_ticks()

    ax.set_title(
        title
        + f"\nISR hmF2 cutoff = {isr_hmf2_km:.1f} km",
        fontsize=label_fontsize + 4,
    )

    ax.legend(
        loc="lower left",
        fontsize=label_fontsize,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / output_filename

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# Produce the two maps
# ------------------------------------------------------------

plot_prior_rmse_map(
    rmse_column="kf_prior_rmse_vs_isr",
    title="KF Prior RMSE Versus Tromsø ISR",
    output_filename="Figure6a_KF_prior_RMSE_vs_ISR_map.png",
)

plot_prior_rmse_map(
    rmse_column="ekf_prior_rmse_vs_isr",
    title="EKF Prior RMSE Versus Tromsø ISR",
    output_filename="Figure6b_EKF_prior_RMSE_vs_ISR_map.png",
)

# ============================================================
# Plot 7:
# Geographic distribution of KF-EKF prior RMSE below hmF2
# ============================================================

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

TRO_LAT = 69.583
TRO_LON = 19.21

map_df = column_results.copy()
map_df = map_df[
    np.isfinite(map_df["lat"])
    & np.isfinite(map_df["lon"])
    & np.isfinite(map_df["prior_rmse_below_hmf2"])
    & (map_df["prior_rmse_below_hmf2"] > 0)
].copy()

if map_df.empty:
    print("Plot 7 skipped: no finite positive RMSE values.")
else:
    projection = ccrs.Orthographic(
        central_longitude=TRO_LON,
        central_latitude=TRO_LAT,
    )

    fig = plt.figure(figsize=(12, 11))
    ax = plt.axes(projection=projection)

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        alpha=0.5,
        linestyle="--",
    )

    rmse_values = map_df["prior_rmse_below_hmf2"].to_numpy()

    scatter = ax.scatter(
        map_df["lon"],
        map_df["lat"],
        c=rmse_values,
        s=70,
        cmap="jet",
        vmin=1e8, vmax=2e11,
        edgecolors="black",
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # Tromsø ISR location
    ax.scatter(
        TRO_LON,
        TRO_LAT,
        marker="*",
        s=350,
        color="red",
        edgecolors="black",
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=10,
        label="Tromsø ISR"
    )

    cbar = plt.colorbar(
        scatter,
        ax=ax,
        orientation="vertical",
        shrink=0.72,
        pad=0.04, cmap="jet"
    )

    
    cbar.set_label(
        r"KF–EKF prior RMSE below common hmF2 (m$^{-3}$)",
        fontsize=label_fontsize,
    )
    cbar.ax.tick_params(labelsize=label_fontsize - 2)

    ax.set_title(
        "Geographic Distribution of KF–EKF Prior Differences\n"
        "RMSE Below Common hmF2",
        fontsize=label_fontsize + 4,
    )

    ax.legend(
        loc="lower left",
        fontsize=label_fontsize,
    )

    fig.tight_layout()

    plot7_path = OUTPUT_DIR / (
        "KF_EKF_prior_RMSE_below_hmF2_global_map.png"
    )

    fig.savefig(
        plot7_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {plot7_path}")

# ============================================================
# Figures 8–10
# ============================================================

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import LogNorm, Normalize


TRO_LAT = 69.583
TRO_LON = 19.21

POLAR_PROJECTION = ccrs.Orthographic(
    central_longitude=TRO_LON,
    central_latitude=TRO_LAT,
)


ALTITUDE_BANDS = [
    (0.0,   100.0),
    (100.0, 200.0),
    (200.0, 300.0),
    (300.0, 400.0),
    (400.0, 500.0),
    (500.0, 600.0),
    (600.0, 700.0),
    (700.0, 800.1),      
]


def configure_polar_axis(ax):
    """Common polar-map formatting."""

    ax.set_global()

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.90",
        zorder=0,
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        zorder=0,
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.7,
        zorder=2,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.35,
        linestyle=":",
        zorder=2,
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.35,
        linestyle="--",
        alpha=0.45,
        zorder=1,
    )

    # Tromsø ISR location
    ax.scatter(
        TRO_LON,
        TRO_LAT,
        marker="*",
        s=180,
        color="red",
        edgecolors="black",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=20,
    )


# # ============================================================
# # Figure 8:
# # Polar map of KF–EKF prior TEC RMSE
# # ============================================================

# KF_TEC_COLUMN = "kf_prior_tec_mean_per_ray_tecu"
# EKF_TEC_COLUMN = "ekf_prior_tec_mean_per_ray_tecu"

# required_tec_columns = {
#     "lat",
#     "lon",
#     "alt_km",
#     "ray_count",
#     KF_TEC_COLUMN,
#     EKF_TEC_COLUMN,
# }

# missing_tec_columns = required_tec_columns.difference(
#     tec_voxels.columns
# )

# if missing_tec_columns:
#     raise KeyError(
#         "TEC CSV is missing required columns: "
#         + ", ".join(sorted(missing_tec_columns))
#     )


# # Only use voxels crossed by at least one ray
# tec_valid = tec_voxels[
#     (tec_voxels["ray_count"] > 0)
#     & np.isfinite(tec_voxels[KF_TEC_COLUMN])
#     & np.isfinite(tec_voxels[EKF_TEC_COLUMN])
# ].copy()

# tec_valid["kf_ekf_tec_difference_tecu"] = (
#     tec_valid[KF_TEC_COLUMN]
#     - tec_valid[EKF_TEC_COLUMN]
# )


# # One TEC RMSE value per horizontal column
# figure8_rows = []

# for (lat, lon), group in tec_valid.groupby(
#     ["lat", "lon"],
#     sort=False,
# ):

#     difference = group[
#         "kf_ekf_tec_difference_tecu"
#     ].to_numpy(dtype=float)

#     finite = np.isfinite(difference)

#     if finite.sum() == 0:
#         tec_rmse = np.nan
#     else:
#         tec_rmse = float(
#             np.sqrt(
#                 np.mean(difference[finite] ** 2)
#             )
#         )

#     figure8_rows.append({
#         "lat": float(lat),
#         "lon": float(lon),
#         "kf_ekf_prior_tec_rmse_tecu": tec_rmse,
#         "n_altitude_voxels": int(finite.sum()),
#         "total_ray_voxel_intersections": int(
#             np.nansum(group["ray_count"])
#         ),
#     })


# figure8_df = pd.DataFrame(figure8_rows)

# figure8_df = figure8_df[
#     np.isfinite(
#         figure8_df["kf_ekf_prior_tec_rmse_tecu"]
#     )
#     & (
#         figure8_df["kf_ekf_prior_tec_rmse_tecu"] > 0
#     )
# ].copy()

# figure8_df = figure8_df.sort_values(
#     "kf_ekf_prior_tec_rmse_tecu",
#     ascending=False,
# ).reset_index(drop=True)

# figure8_csv = OUTPUT_DIR / (
#     "Figure8_KF_EKF_prior_TEC_RMSE_by_column.csv"
# )

# figure8_df.to_csv(
#     figure8_csv,
#     index=False,
# )

# print(f"Saved: {figure8_csv}")


# if figure8_df.empty:

#     print("Figure 8 skipped: no valid TEC RMSE values.")

# else:

#     tec_rmse_values = figure8_df[
#         "kf_ekf_prior_tec_rmse_tecu"
#     ].to_numpy(dtype=float)

#     figure8_norm = LogNorm(
#         vmin=float(np.nanmin(tec_rmse_values)),
#         vmax=float(np.nanmax(tec_rmse_values)),
#     )

#     largest_tec_rmse = figure8_df.iloc[0]

#     fig = plt.figure(figsize=(13, 11))

#     ax = fig.add_subplot(
#         1,
#         1,
#         1,
#         projection=POLAR_PROJECTION,
#     )

#     configure_polar_axis(ax)

#     scatter = ax.scatter(
#         figure8_df["lon"],
#         figure8_df["lat"],
#         c=figure8_df[
#             "kf_ekf_prior_tec_rmse_tecu"
#         ],
#         s=70,
#         cmap="jet",
#         norm=figure8_norm,
#         edgecolors="black",
#         linewidths=0.35,
#         transform=ccrs.PlateCarree(),
#         zorder=8,
#     )

#     # Highlight the largest-RMSE column
#     ax.scatter(
#         largest_tec_rmse["lon"],
#         largest_tec_rmse["lat"],
#         marker="X",
#         s=300,
#         color="white",
#         edgecolors="black",
#         linewidths=1.8,
#         transform=ccrs.PlateCarree(),
#         zorder=25,
#         label=(
#             "Largest TEC RMSE\n"
#             f"{largest_tec_rmse['kf_ekf_prior_tec_rmse_tecu']:.3e} TECU"
#         ),
#     )

#     cbar = fig.colorbar(
#         scatter,
#         ax=ax,
#         orientation="vertical",
#         shrink=0.72,
#         pad=0.04,
#     )

#     cbar.set_label(
#         "KF–EKF prior TEC-contribution RMSE (TECU)",
#         fontsize=label_fontsize,
#     )

#     cbar.ax.tick_params(
#         labelsize=label_fontsize - 2,
#     )

#     ax.set_title(
#         "Figure 8: Geographic Distribution of KF–EKF "
#         "Prior TEC Differences\n"
#         "RMSE Across Observed Altitude Voxels",
#         fontsize=label_fontsize + 4,
#     )

#     ax.legend(
#         loc="lower left",
#         fontsize=label_fontsize - 3,
#     )

#     fig.tight_layout()

#     figure8_path = OUTPUT_DIR / (
#         "Figure8_KF_EKF_prior_TEC_RMSE_polar_map.png"
#     )

#     fig.savefig(
#         figure8_path,
#         dpi=200,
#         bbox_inches="tight",
#     )

#     plt.close(fig)

#     print(f"Saved: {figure8_path}")

#     print(
#         "Largest TEC RMSE column: "
#         f"lat={largest_tec_rmse['lat']:.2f}, "
#         f"lon={largest_tec_rmse['lon']:.2f}, "
#         f"RMSE="
#         f"{largest_tec_rmse['kf_ekf_prior_tec_rmse_tecu']:.3e} TECU"
#     )


# ============================================================
# Figure 9:
# Observation count in each voxel, grouped every 100 km
# ============================================================

required_obs_columns = {
    "lat",
    "lon",
    "alt_km",
    "observation_count",
}

missing_obs_columns = required_obs_columns.difference(
    obs_count.columns
)

if missing_obs_columns:
    raise KeyError(
        "Observation-count CSV is missing required columns: "
        + ", ".join(sorted(missing_obs_columns))
    )


figure9_band_data = []

for altitude_min, altitude_max in ALTITUDE_BANDS:

    band_df = obs_count[
        (obs_count["alt_km"] >= altitude_min)
        & (obs_count["alt_km"] < altitude_max)
    ].copy()

    if band_df.empty:
        continue

    # Accumulate all altitude voxels within this 100-km range
    band_sum = (
        band_df.groupby(
            ["lat", "lon"],
            as_index=False,
        )["observation_count"]
        .sum(min_count=1)
    )

    band_sum["altitude_min_km"] = altitude_min
    band_sum["altitude_max_km"] = altitude_max

    figure9_band_data.append(band_sum)


if not figure9_band_data:

    print("Figure 9 skipped: no observation-count data.")

else:

    figure9_df = pd.concat(
        figure9_band_data,
        ignore_index=True,
    )

    figure9_csv = OUTPUT_DIR / (
        "Figure9_observation_counts_by_100km_band.csv"
    )

    figure9_df.to_csv(
        figure9_csv,
        index=False,
    )

    count_values = figure9_df[
        "observation_count"
    ].to_numpy(dtype=float)

    count_values = count_values[
        np.isfinite(count_values)
        & (count_values >= 0)
    ]

    figure9_vmax = float(
        np.nanmax(count_values)
    )

    if figure9_vmax <= 0:
        figure9_vmax = 1.0

    figure9_norm = Normalize(
        vmin=0,
        vmax=figure9_vmax*0.5,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(23, 13),
        subplot_kw={
            "projection": POLAR_PROJECTION,
        },
    )

    axes = axes.ravel()
    last_scatter = None

    for ax, (altitude_min, altitude_max) in zip(
        axes,
        ALTITUDE_BANDS,
    ):

        configure_polar_axis(ax)

        panel_df = figure9_df[
            np.isclose(
                figure9_df["altitude_min_km"],
                altitude_min,
            )
            & np.isclose(
                figure9_df["altitude_max_km"],
                altitude_max,
            )
        ]

        if not panel_df.empty:

            last_scatter = ax.scatter(
                panel_df["lon"],
                panel_df["lat"],
                c=panel_df["observation_count"],
                s=42,
                cmap="jet",
                norm=figure9_norm,
                edgecolors="black",
                linewidths=0.20,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )

        display_max = (
            800
            if altitude_max > 800
            else int(altitude_max)
        )

        ax.set_title(
            f"{int(altitude_min)}–{display_max} km",
            fontsize=label_fontsize,
        )

    if last_scatter is not None:

        cbar = fig.colorbar(
            last_scatter,
            ax=axes.tolist(),
            orientation="vertical",
            shrink=0.77,
            pad=0.025,
        )

        cbar.set_label(
            "Accumulated number of observation-ray intersections",
            fontsize=label_fontsize,
        )

        cbar.ax.tick_params(
            labelsize=label_fontsize - 2,
        )

    fig.suptitle(
        "Figure 9: Number of Observations in KF Voxels\n"
        "Accumulated Within Each 100-km Altitude Band",
        fontsize=label_fontsize + 5,
        y=0.98,
    )

    figure9_path = OUTPUT_DIR / (
        "Figure9_voxel_observation_counts_100km_bands.png"
    )

    fig.savefig(
        figure9_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {figure9_path}")
    print(f"Saved: {figure9_csv}")


# # ============================================================
# # Figure 10:
# # Three altitude profiles where KF–EKF TEC RMSE is largest
# # ============================================================

# N_TEC_EXAMPLES = 3

# if figure8_df.empty:

#     print("Figure 10 skipped: Figure 8 produced no valid columns.")

# else:

#     worst_tec_columns = figure8_df.head(
#         N_TEC_EXAMPLES
#     ).copy()

#     fig, axes = plt.subplots(
#         1,
#         N_TEC_EXAMPLES,
#         figsize=(6 * N_TEC_EXAMPLES, 9),
#         sharey=True,
#     )

#     if N_TEC_EXAMPLES == 1:
#         axes = [axes]

#     for ax, (_, row) in zip(
#         axes,
#         worst_tec_columns.iterrows(),
#     ):

#         latitude = float(row["lat"])
#         longitude = float(row["lon"])

#         profile = tec_valid[
#             np.isclose(
#                 tec_valid["lat"],
#                 latitude,
#             )
#             & np.isclose(
#                 tec_valid["lon"],
#                 longitude,
#             )
#         ].sort_values("alt_km")

#         ax.plot(
#             profile[KF_TEC_COLUMN],
#             profile["alt_km"],
#             linewidth=2.2,
#             label="KF prior TEC contribution",
#         )

#         ax.plot(
#             profile[EKF_TEC_COLUMN],
#             profile["alt_km"],
#             linewidth=2.2,
#             linestyle="--",
#             label="EKF prior TEC contribution",
#         )

#         ax.set_xlabel(
#             "Mean voxel TEC contribution\n"
#             "per intersecting ray (TECU)",
#             fontsize=label_fontsize,
#         )

#         ax.tick_params(
#             axis="both",
#             labelsize=label_fontsize - 2,
#         )

#         ax.set_xlim(left=0)

#         ax.grid(
#             True,
#             alpha=0.3,
#         )

#         ax.set_title(
#             f"lat={latitude:.1f}°, lon={longitude:.1f}°\n"
#             f"TEC RMSE="
#             f"{row['kf_ekf_prior_tec_rmse_tecu']:.3e} TECU",
#             fontsize=label_fontsize + 1,
#         )

#     axes[0].set_ylabel(
#         "Altitude (km)",
#         fontsize=label_fontsize,
#     )

#     axes[0].legend(
#         fontsize=label_fontsize - 4,
#     )

#     fig.suptitle(
#         "Figure 10: KF and EKF Prior TEC Contributions\n"
#         "Three Horizontal Columns With Largest TEC RMSE",
#         fontsize=label_fontsize + 5,
#     )

#     fig.tight_layout()

#     figure10_path = OUTPUT_DIR / (
#         "Figure10_largest_KF_EKF_TEC_RMSE_altitude_profiles.png"
#     )

#     fig.savefig(
#         figure10_path,
#         dpi=200,
#         bbox_inches="tight",
#     )

#     plt.close(fig)

#     print(f"Saved: {figure10_path}")

#     print("\nLargest three TEC RMSE columns:")

#     print(
#         worst_tec_columns[
#             [
#                 "lat",
#                 "lon",
#                 "kf_ekf_prior_tec_rmse_tecu",
#                 "n_altitude_voxels",
#             ]
#         ].to_string(index=False)
#     )

# ============================================================
# Figure 10:
# Geographic distribution of the 8 EKF PRIOR parameters
# One polar-map subplot for each parameter
# ============================================================

from matplotlib.colors import Normalize


EKF_PRIOR_PARAMETER_NAMES = [
    "log10(NmF2)",
    "hmF2",
    "H0",
    "gamma",
    "B0",
    "B1",
    "log10(NmE)",
    "hmE",
]

EKF_PARAMETER_LABELS = {
    "log10(NmF2)": r"log$_{10}$(NmF2)",
    "hmF2": "hmF2 (km)",
    "H0": "H0 (km)",
    "gamma": r"$\gamma$",
    "B0": "B0",
    "B1": "B1",
    "log10(NmE)": r"log$_{10}$(NmE)",
    "hmE": "hmE (km)",
}


# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

required_parameter_columns = {
    "lat",
    "lon",
}

required_parameter_columns.update(
    f"prior_{name}"
    for name in EKF_PRIOR_PARAMETER_NAMES
)

missing_parameter_columns = (
    required_parameter_columns.difference(
        ekf_columns.columns
    )
)

if missing_parameter_columns:
    raise KeyError(
        "EKF-columns CSV is missing required columns: "
        + ", ".join(
            sorted(missing_parameter_columns)
        )
    )


# ------------------------------------------------------------
# 2 x 4 maps
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    4,
    figsize=(24, 13),
    subplot_kw={
        "projection": POLAR_PROJECTION,
    },
)

axes = axes.ravel()


for ax, parameter_name in zip(
    axes,
    EKF_PRIOR_PARAMETER_NAMES,
):

    configure_polar_axis(ax)

    column_name = f"prior_{parameter_name}"

    parameter_values = ekf_columns[
        column_name
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(ekf_columns["lat"])
        & np.isfinite(ekf_columns["lon"])
        & np.isfinite(parameter_values)
    )

    plot_df = ekf_columns.loc[
        valid
    ].copy()

    values = plot_df[
        column_name
    ].to_numpy(dtype=float)

    if values.size == 0:
        ax.set_title(
            EKF_PARAMETER_LABELS[parameter_name],
            fontsize=label_fontsize,
        )
        continue

    parameter_min = float(
        np.nanmin(values)
    )

    parameter_max = float(
        np.nanmax(values)
    )

    if parameter_max <= parameter_min:
        parameter_max = parameter_min + 1e-12

    parameter_norm = Normalize(
        vmin=parameter_min,
        vmax=parameter_max,
    )

    scatter = ax.scatter(
        plot_df["lon"],
        plot_df["lat"],
        c=values,
        s=55,
        cmap="jet",
        norm=parameter_norm,
        edgecolors="black",
        linewidths=0.25,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )

    # Each parameter has different units/range,
    # so give every subplot its own colorbar.
    cbar = fig.colorbar(
        scatter,
        ax=ax,
        orientation="vertical",
        shrink=0.72,
        pad=0.025,
    )

    cbar.set_label(
        EKF_PARAMETER_LABELS[
            parameter_name
        ],
        fontsize=label_fontsize,
    )

    cbar.ax.tick_params(
        labelsize=label_fontsize - 3,
    )

    ax.set_title(
        EKF_PARAMETER_LABELS[
            parameter_name
        ],
        fontsize=label_fontsize,
    )


fig.suptitle(
    "EKF Prior 8-Parameter Values @ Each Voxel",
    fontsize=label_fontsize + 5,
    y=0.98,
)

figure10_path = OUTPUT_DIR / (
    "Figure10_EKF_prior_8_parameters_maps.png"
)

fig.savefig(
    figure10_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {figure10_path}")


# ============================================================
# Figures 11a and 11b:
# Geographic KF-EKF PRIOR Ne percentage difference
#
# For each horizontal voxel and each 50-km layer:
#
#   Difference (%) =
#
#       mean( |Ne_EKF - Ne_KF| )
#       ------------------------- x 100
#             mean( Ne_KF )
#
# KF is therefore the reference/background Ne.
#
# 50-100, 100-150, ..., 750-800 km
# ============================================================


# ------------------------------------------------------------
# 15 altitude bands
# ------------------------------------------------------------

NE_DIFFERENCE_ALTITUDE_BANDS = [
    (altitude_min, altitude_min + 50.0)
    for altitude_min in np.arange(
        50.0,
        1000.0,
        50.0,
    )
]

print(
    "Number of 50-km Ne-difference bands:",
    len(NE_DIFFERENCE_ALTITUDE_BANDS),
)


# ------------------------------------------------------------
# Use the new direct KF/EKF prior-Ne comparison CSV
#
# If this code is in the existing script, make sure the file
# was loaded earlier as, for example:
#
# prior_ne_compare = pd.read_csv(PRIOR_NE_COMPARE_FILE)
# ------------------------------------------------------------


required_ne_columns = {
    "lat",
    "lon",
    "alt_km",
    "kf_prior_ne_m3",
    "ekf_prior_ne_m3",
}

missing_ne_columns = (
    required_ne_columns.difference(
        prior_ne_compare.columns
    )
)

if missing_ne_columns:
    raise KeyError(
        "KF/EKF prior-Ne comparison CSV is "
        "missing required columns: "
        + ", ".join(
            sorted(missing_ne_columns)
        )
    )


# ------------------------------------------------------------
# Calculate one percentage difference for every
# (lat, lon, 50-km altitude band)
# ------------------------------------------------------------

ne_difference_band_data = []


for altitude_min, altitude_max in (
    NE_DIFFERENCE_ALTITUDE_BANDS
):

    band_df = prior_ne_compare[
        (prior_ne_compare["alt_km"] >= altitude_min)
        & (prior_ne_compare["alt_km"] < altitude_max)
    ].copy()

    if band_df.empty:
        continue

    band_rows = []

    for (lat, lon), group in band_df.groupby(
        ["lat", "lon"],
        sort=False,
    ):

        kf_ne = group[
            "kf_prior_ne_m3"
        ].to_numpy(dtype=float)

        ekf_ne = group[
            "ekf_prior_ne_m3"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(kf_ne)
            & np.isfinite(ekf_ne)
            & (kf_ne > 0)
        )

        if valid.sum() == 0:
            percent_difference = np.nan
            mean_kf_ne = np.nan
            mean_abs_difference = np.nan

        else:

            kf_valid = kf_ne[valid]
            ekf_valid = ekf_ne[valid]

            # Numerator:
            # vertically averaged absolute EKF-KF difference
            mean_abs_difference = float(
                np.mean(
                    np.abs(
                        ekf_valid
                        - kf_valid
                    )
                )
            )

            # Denominator:
            # mean KF background Ne in this 50-km layer
            mean_kf_ne = float(
                np.mean(kf_valid)
            )

            if mean_kf_ne > 0:

                percent_difference = (
                    100.0
                    * mean_abs_difference
                    / mean_kf_ne
                )

            else:
                percent_difference = np.nan

        band_rows.append({
            "lat": float(lat),
            "lon": float(lon),
            "altitude_min_km": float(
                altitude_min
            ),
            "altitude_max_km": float(
                altitude_max
            ),
            "mean_kf_prior_ne_m3":
                mean_kf_ne,
            "mean_abs_ekf_minus_kf_ne_m3":
                mean_abs_difference,
            "mean_percent_difference":
                percent_difference,
            "n_altitude_levels":
                int(valid.sum()),
        })

    if band_rows:

        ne_difference_band_data.append(
            pd.DataFrame(
                band_rows
            )
        )


if not ne_difference_band_data:

    print(
        "Figures 11a/11b skipped: "
        "no valid KF/EKF Ne differences."
    )

else:

    ne_difference_50km_df = pd.concat(
        ne_difference_band_data,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Save numerical values
    # --------------------------------------------------------

    figure11_csv = OUTPUT_DIR / (
        "Figure11_KF_EKF_prior_Ne_"
        "percentage_difference_50km_bands.csv"
    )

    ne_difference_50km_df.to_csv(
        figure11_csv,
        index=False,
    )

    print(f"Saved: {figure11_csv}")


    # --------------------------------------------------------
    # Shared color scale across ALL 15 panels
    #
    # This is important: otherwise 50-100 km and
    # 700-750 km could visually look equally different
    # while representing completely different percentages.
    # --------------------------------------------------------

    percent_values = ne_difference_50km_df[
        "mean_percent_difference"
    ].to_numpy(dtype=float)

    percent_values = percent_values[
        np.isfinite(percent_values)
        & (percent_values >= 0)
    ]

    if percent_values.size == 0:
        raise ValueError(
            "No finite Ne percentage differences "
            "available for Figures 11a/11b."
        )


    difference_vmin = 0.0

    difference_vmax = float(
        np.nanmax(
            percent_values
        )
    )

    if difference_vmax <= 0:
        difference_vmax = 1.0


    difference_norm = Normalize(
        vmin=difference_vmin,
        vmax=100,
    )


    # ========================================================
    # Helper: make one 2 x 4 figure
    # ========================================================

    def plot_ne_difference_band_figure(
        altitude_bands,
        output_filename,
        figure_title,
    ):

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(25, 15),
            subplot_kw={
                "projection": POLAR_PROJECTION,
            },
        )

        axes = axes.ravel()

        last_scatter = None


        for panel_index, ax in enumerate(
            axes
        ):

            configure_polar_axis(ax)

            # The second figure only has seven bands,
            # so hide the unused eighth subplot.
            if panel_index >= len(
                altitude_bands
            ):

                ax.set_visible(False)
                continue


            altitude_min, altitude_max = (
                altitude_bands[
                    panel_index
                ]
            )

            panel_df = ne_difference_50km_df[
                np.isclose(
                    ne_difference_50km_df[
                        "altitude_min_km"
                    ],
                    altitude_min,
                )
                & np.isclose(
                    ne_difference_50km_df[
                        "altitude_max_km"
                    ],
                    altitude_max,
                )
            ].copy()


            panel_df = panel_df[
                np.isfinite(
                    panel_df[
                        "mean_percent_difference"
                    ]
                )
            ]


            if not panel_df.empty:

                last_scatter = ax.scatter(
                    panel_df["lon"],
                    panel_df["lat"],
                    c=panel_df[
                        "mean_percent_difference"
                    ],
                    s=48,
                    cmap="jet",
                    norm=difference_norm,
                    edgecolors="black",
                    linewidths=0.20,
                    transform=ccrs.PlateCarree(),
                    zorder=8,
                )


            ax.set_title(
                f"{int(altitude_min)}–"
                f"{int(altitude_max)} km",
                fontsize=label_fontsize,
            )


        # ----------------------------------------------------
        # One common colorbar for all eight panels
        # ----------------------------------------------------

        if last_scatter is not None:

            visible_axes = [
                ax
                for ax in axes
                if ax.get_visible()
            ]

            cbar = fig.colorbar(
                last_scatter,
                ax=visible_axes,
                orientation="vertical",
                shrink=0.77,
                pad=0.025,
            )

            cbar.set_label(
                "Averaged absolute KF–EKF prior Ne "
                "difference relative to KF (%)",
                fontsize=label_fontsize,
            )

            cbar.ax.tick_params(
                labelsize=label_fontsize - 2,
            )


        fig.suptitle(
            figure_title,
            fontsize=label_fontsize + 5,
            y=0.98,
        )


        output_path = (
            OUTPUT_DIR
            / output_filename
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")


    # ========================================================
    # Figure 11a:
    # 50–450 km = first 8 bands
    # ========================================================

    plot_ne_difference_band_figure(
        altitude_bands=(
            NE_DIFFERENCE_ALTITUDE_BANDS[
                :8
            ]
        ),
        output_filename=(
            "Figure11a_KF_EKF_prior_Ne_"
            "percent_difference_50_450km.png"
        ),
        figure_title=(
            "KF–EKF Prior Ne "
            "Percentage Difference\n"
            "50-km Altitude Bands: 50–450 km"
        ),
    )


    # ========================================================
    # Figure 11b:
    # 450–800 km = remaining 7 bands
    # ========================================================

    plot_ne_difference_band_figure(
        altitude_bands=(
            NE_DIFFERENCE_ALTITUDE_BANDS[
                8:
            ]
        ),
        output_filename=(
            "Figure11b_KF_EKF_prior_Ne_"
            "percent_difference_450_850km.png"
        ),
        figure_title=(
            "KF–EKF Prior Ne "
            "Percentage Difference\n"
            "50-km Altitude Bands: 450–850 km"
        ),
    )

    plot_ne_difference_band_figure(
        altitude_bands=(
            NE_DIFFERENCE_ALTITUDE_BANDS[
                8:
            ]
        ),
        output_filename=(
            "Figure12b_KF_EKF_prior_Ne_"
            "percent_difference_850_1000km.png"
        ),
        figure_title=(
            "KF–EKF Prior Ne "
            "Percentage Difference\n"
            "50-km Altitude Bands: 850–1000 km"
        ),
    )

for IRI_VOXEL_INDEX in range(1, 56):

    profile_fit = iri_8param[
        iri_8param["voxel_idx"] == IRI_VOXEL_INDEX
    ].copy()

    if profile_fit.empty:
        print(
            f"Skip voxel {IRI_VOXEL_INDEX}: "
            "no IRI/8-param profile found"
        )
        continue

    profile_fit = profile_fit.sort_values("altitude_km")

    # Location
    fit_lat = float(profile_fit["lat_deg"].iloc[0])
    fit_lon = float(profile_fit["lon_deg"].iloc[0])

    # Extract profile
    alt_fit = profile_fit["altitude_km"].to_numpy(dtype=float)
    iri_ne  = profile_fit["iri_ne_m3"].to_numpy(dtype=float)
    fit_ne  = profile_fit["fit8_ne_m3"].to_numpy(dtype=float)

    valid = (
        np.isfinite(alt_fit)
        & np.isfinite(iri_ne)
        & np.isfinite(fit_ne)
        & (iri_ne > 0)
        & (fit_ne > 0)
    )

    if valid.sum() < 2:
        print(
            f"Skip voxel {IRI_VOXEL_INDEX}: "
            "not enough valid Ne points"
        )
        continue

    # log10 RMSE
    log_rmse = np.sqrt(
        np.mean(
            (
                np.log10(fit_ne[valid])
                - np.log10(iri_ne[valid])
            )**2
        )
    )

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 9))

    ax.plot(
        iri_ne,
        alt_fit,
        linewidth=2.5,
        label="IRI Ne",
    )

    ax.plot(
        fit_ne,
        alt_fit,
        linewidth=2.5,
        linestyle="--",
        label="8-parameter fit",
    )

    ax.set_xscale("log")
    ax.set_xlim(1e7, 1e13)

    ax.set_xlabel(
        r"Ne (m$^{-3}$)",
        fontsize=label_fontsize,
    )

    ax.set_ylabel(
        "Altitude (km)",
        fontsize=label_fontsize,
    )

    ax.tick_params(
        axis="both",
        labelsize=label_fontsize,
    )

    ax.set_title(
        f"IRI vs. 8-Parameter Ne Profile\n"
        f"Voxel {IRI_VOXEL_INDEX}: "
        f"lat={fit_lat:.2f}°, lon={fit_lon:.2f}°\n"
        f"log$_{{10}}$ RMSE = {log_rmse:.4f}",
        fontsize=label_fontsize + 3,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(
        fontsize=label_fontsize - 2,
    )

    fig.tight_layout()

    plot_path = OUTPUT_DIR / (
        f"voxel_{IRI_VOXEL_INDEX:03d}_"
        "IRI_vs_8param_Ne.png"
    )

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved voxel {IRI_VOXEL_INDEX:02d}: "
        f"{plot_path}"
    )


# ============================================================
# Final plot:
# EKF posterior Ne profile for EVERY horizontal voxel
# One individual figure per voxel
# + posterior 8-parameter state shown on figure
# ============================================================

POSTERIOR_OUTPUT_DIR = (
    OUTPUT_DIR / "EKF_posterior_Ne_all_voxels"
)

POSTERIOR_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Unique horizontal voxel locations
voxel_locations = (
    ekf_voxels[["lat", "lon"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    f"\nPlotting posterior EKF Ne for "
    f"{len(voxel_locations)} horizontal voxels ..."
)

for voxel_idx, row in voxel_locations.iterrows():

    lat = float(row["lat"])
    lon = float(row["lon"])

    # ========================================================
    # 1. Get posterior Ne profile for this voxel
    # ========================================================

    profile = ekf_voxels[
        np.isclose(ekf_voxels["lat"], lat)
        & np.isclose(ekf_voxels["lon"], lon)
    ].copy()

    profile = profile.sort_values("alt_km")

    alt = profile["alt_km"].to_numpy(dtype=float)
    ne_post = profile["post_ne_m3"].to_numpy(dtype=float)

    valid = (
        np.isfinite(alt)
        & np.isfinite(ne_post)
        & (ne_post > 0.0)
    )

    if valid.sum() < 2:
        print(
            f"  Skipping voxel {voxel_idx}: "
            "not enough valid posterior Ne points"
        )
        continue

    alt = alt[valid]
    ne_post = ne_post[valid]

    # ========================================================
    # 2. Find corresponding posterior 8-parameter state
    # ========================================================

    param_match = ekf_columns[
        np.isclose(ekf_columns["lat"], lat)
        & np.isclose(ekf_columns["lon"], lon)
    ]

    if param_match.empty:
        print(
            f"  Warning: no EKF parameter row for "
            f"voxel {voxel_idx}"
        )

        parameter_text = "Posterior parameters unavailable"

    else:

        p = param_match.iloc[0]

        logNmF2 = float(p["post_log10(NmF2)"])
        hmF2    = float(p["post_hmF2"])
        H0      = float(p["post_H0"])
        gamma   = float(p["post_gamma"])
        B0      = float(p["post_B0"])
        B1      = float(p["post_B1"])
        logNmE  = float(p["post_log10(NmE)"])
        hmE     = float(p["post_hmE"])

        # Optional physical density values
        NmF2 = 10.0 ** logNmF2
        NmE  = 10.0 ** logNmE

        parameter_text = (
            "Posterior 8-parameter state\n"
            f"log10(NmF2) = {logNmF2:.3f}\n"
            f"hmF2 = {hmF2:.1f} km\n"
            f"H0 = {H0:.1f} km\n"
            f"gamma = {gamma:.3f}\n"
            f"B0 = {B0:.1f} km\n"
            f"B1 = {B1:.3f}\n"
            f"log10(NmE) = {logNmE:.3f}\n"
            f"hmE = {hmE:.1f} km"
        )

    # ========================================================
    # 3. Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.5, 9)
    )

    ax.plot(
        ne_post,
        alt,
        linewidth=2.5,
        label="EKF posterior Ne",
    )

    ax.set_xscale("log")

    ax.set_xlim(
        1e7,
        1e13,
    )

    ax.set_xlabel(
        r"Ne (m$^{-3}$)",
        fontsize=label_fontsize,
    )

    ax.set_ylabel(
        "Altitude (km)",
        fontsize=label_fontsize,
    )

    ax.tick_params(
        axis="both",
        labelsize=label_fontsize,
    )

    ax.set_title(
        "EKF Posterior Ne\n"
        f"Voxel {voxel_idx}: "
        f"lat={lat:.2f}°, lon={lon:.2f}°",
        fontsize=label_fontsize + 3,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    # ========================================================
    # 4. Posterior-parameter box
    # ========================================================

    ax.text(
        0.04,
        0.96,
        parameter_text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
        ),
    )

    ax.legend(
        loc="upper right",
        fontsize=14,
    )

    fig.tight_layout()

    # ========================================================
    # 5. Save
    # ========================================================

    out_path = (
        POSTERIOR_OUTPUT_DIR
        / f"voxel_{voxel_idx:03d}_"
          f"lat_{lat:+07.2f}_"
          f"lon_{lon:+08.2f}_"
          f"posterior_Ne.png"
    )

    fig.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

print(
    f"Saved posterior voxel profiles → "
    f"{POSTERIOR_OUTPUT_DIR}"
)