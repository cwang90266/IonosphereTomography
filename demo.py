#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11

"""
demo.py — End-to-end demonstration of the IonosphereTomography codebase.

Region of interest: 60–65 °N, 150–140 °W (high-latitude Alaska)

Setup (one time):
    pipenv install        # creates venv and installs from Pipfile
    pipenv run python demo.py

Or, with any venv that has the packages installed:
    python demo.py

Sections
--------
 §1  Single IRI altitude profile at the region center (default solar indices)
 §2  Three solar-activity scenarios compared on one altitude profile
 §3  Sensitivity to ionospheric shape parameters (foF2, hmF2, B0, B1)
 §4  24-hour time profile at a fixed location
 §5  Latitude sweep 60→65 °N across the Alaska region
 §6  Rectangular triangular mesh: generate and plot
 §7  Polar-cap triangular mesh (60°N → North Pole): generate and plot
 §8  EDPSamples: populate a rectangular Alaska mesh with IRI across 3 solar scenarios
 §9  Bundled solar-index data: read and visualise apf107 / ig_rz
§10  IRI_Sample_Inputs: download live data and build a quantile-sample DataFrame
     (requires internet access; skipped gracefully if unavailable)

"""

import sys
import importlib.resources as impr
from pathlib import Path

ROOT = Path(__file__).parent
# Make EDPSamples and IRI_Sample_Inputs importable (they are not installed packages)
sys.path.insert(0, str(ROOT))
# locate_in_mesh.py lives inside a directory with a space in its name
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src" ))

print(ROOT)
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.parser import parse

# ── iri2020 package (editable install from iri2020_new/src) ─────────────────
from iri2020 import IRI, timeprofile, geoprofile
import iri2020.plots as iri_plots
from iri2020.get_iri_inputs import read_apf107, read_ig_rz

# ── EDPSamples utilities ─────────────────────────────────────────────────────
from EDPSamples.edp_samples import (
    EDPSamples,
    plot_tri_mesh,
    plot_polar_mesh,
    interp_heights,
)
from EDPSamples.generate_rect_tri_mesh import generate_rect_tri_mesh
from EDPSamples.generate_polar_mesh import generate_ploar_mesh   # note: typo in source

from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import Ionosphere_Tomography_Inverter
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter_RelTEC import Ionosphere_Tomography_Inverter_RelTEC

# Standalone spherical point-in-triangle algorithm (no WGS84 altitude bug)
from locate_in_mesh import find_containing_triangles as find_triangle_sphere

# ── IRI_Sample_Inputs ────────────────────────────────────────────────────────
from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs

from IRI_ARR_Samples.iri_arr_samples import calculate_iri_electron_density
from datetime import datetime, timedelta
# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

TIME     = "2025-06-15 19:00"      # simulation UTC time
LAT_C    = 40                    # center latitude  (°N)
LON_C    = -105.0                  # center longitude (°E, = 145 °W)
ALT_KM   = [60, 1000, 1]         # altitude grid: [start, stop, step] km

# Alaska region bounds (used for sweeps and mesh generation)
LAT_MIN, LAT_MAX, DLAT = 60.0, 65.0, 1.5
LON_MIN, LON_MAX, DLON = -150.0, -140.0, 1.5

# Three solar-activity scenarios spanning the realistic range for Alaska
SCENARIOS = [
    {"label": "Low activity",  "f107D":  70, "ap":  3, "IG12":  20, "Rz12":  15},
    {"label": "Nominal",       "f107D": 120, "ap": 10, "IG12":  60, "Rz12":  50},
    {"label": "High activity", "f107D": 200, "ap": 30, "IG12": 100, "Rz12":  90},
]
COLORS = ["royalblue", "forestgreen", "firebrick"]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# =============================================================================
# def _iri_time_sweep(times, altkmrange, glat, glon, **solar_kw) -> xr.Dataset:
#     """
#     Run IRI for each datetime in `times` and concatenate along the time
#     dimension. Implements the core of vprofile.timeprofile without the
#     iri.f107 attribute bug present in that function.
#     """
#     iono = xr.Dataset()
#     for t in times:
#         iri = IRI(t, altkmrange, glat, glon, **solar_kw)
#         iono = iri if not iono else xr.concat([iono, iri], dim="time")
#     return iono
# 
# 
# def _iri_lat_sweep(lats, altkm, glon, time, **solar_kw) -> xr.Dataset:
#     """
#     Run IRI at a single altitude for each latitude and concatenate along glat.
#     Implements the core of vprofile.geoprofile without the iri.f107 bug.
#     """
#     iono = xr.Dataset()
#     for lat in lats:
#         iri = IRI(time, [altkm] * 3, lat, glon, **solar_kw)
#         iono = iri if not iono else xr.concat([iono, iri], dim="glat")
#     return iono
# 
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  §1  Single IRI altitude profile at the region center
# ─────────────────────────────────────────────────────────────────────────────

# Assuming IRI, calculate_iri_electron_density, iri_plots, etc. are imported

def section1() -> xr.Dataset:
    # _banner("§1  Single IRI altitude profile (default solar indices)") # Un-comment if _banner is defined
    START_TIME = datetime(2025,6,15,0)
    for hour_i in [5]:#range(0,23,1):
        time_in = START_TIME + timedelta(hours=hour_i)
        # 1. Run the official IRI executable via your base.py wrapper
        iono = IRI(time_in, ALT_KM, LAT_C, LON_C)
    
        # 2. Print the standard output parameters
        peak_idx = int(iono["ne"].argmax())
        print(f"  ne peak  : {float(iono['ne'][peak_idx]):.3e} m⁻³"
              f"  at {float(iono.alt_km[peak_idx]):.0f} km")
        print(f"  NmF2     : {float(iono['NmF2']):.3e} m⁻³")
        print(f"  hmF2     : {float(iono['hmF2']):.1f} km")
        print(f"  foF2     : {float(iono['foF2']):.2f} MHz")
        print(f"  TEC      : {float(iono['TEC']):.3e} m⁻²")
        print(f"  B0 / B1  : {float(iono['B0']):.1f} km / {float(iono['B1']):.3f}")
        print(f"  f107D    : {iono.attrs['f107D']:.1f}   ap : {iono.attrs['ap']:.1f}")
        print(f"  IG12     : {iono.attrs['IG12']:.1f}   Rz12: {iono.attrs['Rz12']:.1f}")
    
        # 3. Construct the parameter dictionary for the piecewise Python function
        def get_param(name):
            """Helper to extract float values and treat missing layers (-1 or 0) as None"""
            val = float(iono[name].values[0])
            return val if val > 0 else None
    
        from scipy.optimize import curve_fit
        
        def fit_topside_profile(altitudes, ne_official, hmF2, NmF2):
            """
            Simultaneously solves for the optimal H0 and gamma by fitting the 
            analytical Epstein layer directly to the IRI OUTF topside array.
            """
            # 1. Isolate only the topside data points from the arrays
            mask = altitudes >= hmF2
            h_top = altitudes[mask]
            ne_top = ne_official[mask]
            
            if len(h_top) < 2:
                return 50.0, 0.15 # Fallback if array is too short
                
            # 2. Define the Epstein function exactly as curve_fit expects it: f(x, *params)
            def epstein_model(h, H0, gamma):
                r = 100.0 # NeQuick saturation factor
                dh = h - hmF2
                
                # Safe calculation for the restricted scale height
                # (Added 1e-9 to denominator to prevent division by zero during optimizer guesses)
                H_top = H0 * (1.0 + (r * gamma * dh) / (r * H0 + gamma * dh + 1e-9))
                
                z = dh / H_top
                # Clip z to prevent exponential overflow warnings during wild optimizer guesses
                z = np.clip(z, -100, 100) 
                
                return 4.0 * NmF2 * np.exp(z) / ((1.0 + np.exp(z))**2)
    
            # 3. Run the curve fit
            # We provide a reasonable starting guess (p0) and physical bounds
            try:
                popt, pcov = curve_fit(
                    epstein_model, 
                    h_top, 
                    ne_top, 
                    p0=[70.0, 0.15], 
                    bounds=([20.0, 0.001], [300.0, 1.5]) # [Lower bounds], [Upper bounds]
                )
                optimized_H0 = popt[0]
                optimized_gamma = popt[1]
                return optimized_H0, optimized_gamma
                
            except RuntimeError:
                print("Curve fit failed to converge. Defaulting.")
                return 70.0, 0.15
    
        # --- How to call it in your script ---
        
        # Extract peak parameters
        NmF2 = float(iono['NmF2'].values[0])
        hmF2 = float(iono['hmF2'].values[0])
        
        # Run the curve fitter using the actual xarray data
        alt_array = iono.alt_km.values
        ne_array = iono["ne"].values.squeeze()
        
        H0, optimized_gamma = fit_topside_profile(alt_array, ne_array, hmF2, NmF2)
        
        print(f"  Curve-Fit H0       : {H0:.1f} km")
        print(f"  Curve-Fit Gamma    : {optimized_gamma:.3f}")
        
    
        iri_params = {
            'NMF2': get_param('NmF2'),
            'HMF2': get_param('hmF2'),
            'NMF1': get_param('NmF1'),
            'HMF1': get_param('hmF1'),
            'NME':  get_param('NmE'),
            'HME':  get_param('hmE'),
            'NMD':  get_param('NmD'),
            'HMD':  get_param('hmD'),
            'B0':   get_param('B0'),
            'B1':   get_param('B1'),
            'VNER': get_param('VNER'),
            'HEF':  get_param('HEF'),
            'C1': get_param('C1')
        }
        # Add these variables to your parameter dictionary for the piecewise function
        iri_params['H0'] = H0
        iri_params['gamma'] = optimized_gamma
        # 4. Run your vectorized piecewise function
        altitudes = iono.alt_km.values
        calculated_edp = calculate_iri_electron_density(altitudes, iri_params)
    
        
        import matplotlib.pyplot as plt
        
        def plot_edp_comparison(iono):
            """
            Plots the official IRI electron density profile against the custom piecewise calculation,
            including a side-by-side subplot for relative error.
            """
            # Extract data, ensuring they are 1D arrays
            altitudes = iono.alt_km.values
            ne_official = iono["ne"].values.squeeze()
            ne_calculated = iono["ne_calculated"].values.squeeze()
        
            # Calculate Relative Error (%)
            # np.errstate prevents warnings if the official array contains zeros at very low altitudes
            with np.errstate(divide='ignore', invalid='ignore'):
                error_pct = np.where(ne_official > 0, 
                                     ((ne_calculated - ne_official) / ne_official) * 100.0, 
                                     0.0)
        
            # Create a figure with 1 row and 2 columns, sharing the y-axis (altitude)
            # The width_ratios gives the main plot more horizontal space than the error plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), sharey=True, 
                                           gridspec_kw={'width_ratios': [3, 1]})
            
            # ==========================================
            # Subplot 1: The Profiles
            # ==========================================
            ax1.plot(ne_official, altitudes, label='Official IRI (OUTF)', color='#1f77b4', linewidth=2.5)
            ax1.plot(ne_calculated, altitudes, label='Piecewise Reconstruction', color='#d62728', linestyle='--', linewidth=2.5)
            
            # Standard Ionospheric Plot Formatting
            ax1.set_xscale('log')
            ax1.set_xlim(left=1e8) 
            
            ax1.set_xlabel('Electron Density (m⁻³)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Altitude (km)', fontsize=12, fontweight='bold')
            ax1.set_title('IRI Electron Density Profile Comparison', fontsize=14, fontweight='bold')
            
            ax1.grid(True, which="major", linestyle="-", alpha=0.6)
            ax1.grid(True, which="minor", linestyle=":", alpha=0.4)
            
            # Highlight the anchor heights if they exist
            anchor_heights = {
                'hmF2': float(iono['hmF2'].values[0]) if float(iono['hmF2'].values[0]) > 0 else None,
                'hmF1': float(iono['hmF1'].values[0]) if float(iono['hmF1'].values[0]) > 0 else None,
                'hEF':  float(iono['HEF'].values[0]) if float(iono['HEF'].values[0]) > 0 else None,
                'hmE':  float(iono['hmE'].values[0])  if float(iono['hmE'].values[0]) > 0 else None,
                'hmD':  float(iono['hmD'].values[0])  if float(iono['hmD'].values[0]) > 0 else None
            }
            
            for label, height in anchor_heights.items():
                if height:
                    ax1.axhline(height, color='gray', linestyle=':', alpha=0.8)
                    ax1.text(1.5e8, height + 3, label, color='gray', fontsize=10, fontweight='bold')
                    # Extend the anchor lines across to the error plot as well
                    ax2.axhline(height, color='gray', linestyle=':', alpha=0.8)

            # --- COLOR CODING REGIONS ---
            # Get current y-limits to bound the top and bottom regions seamlessly
            y_min, y_max = ax1.get_ylim()

            # Define hex colors matched to the plot
            c_topside = '#9F9FE5'     # 1: Topside (Purple)
            c_f2      = '#75CFEE'     # 2: F2 (Light Blue)
            c_f1      = '#8DD8A4'     # 3: F1 (Light Green)
            c_inter   = '#F8F28A'     # 4: Intermediate (Yellow)
            c_evalley = '#F9C59A'     # 5: E-Valley (Peach)
            c_ed      = '#EE7E8B'     # 6: E/D (Red/Pink)

            # 1. Topside (> hmF2)
            if anchor_heights['hmF2']:
                ax1.axhspan(anchor_heights['hmF2'], y_max, color=c_topside, alpha=0.5, zorder=0)

            # 2. F2 (hmF1 to hmF2)
            if anchor_heights['hmF2'] and anchor_heights['hmF1']:
                ax1.axhspan(anchor_heights['hmF1'], anchor_heights['hmF2'], color=c_f2, alpha=0.5, zorder=0)
            elif anchor_heights['hmF2'] and anchor_heights['hEF']:
                ax1.axhspan(anchor_heights['hEF'], anchor_heights['hmF2'], color=c_f2, alpha=0.5, zorder=0)

            # 3. F1 (hEF to hmF1)
            if anchor_heights['hmF1'] and anchor_heights['hEF']:
                ax1.axhspan(anchor_heights['hEF'], anchor_heights['hmF1'], color=c_f1, alpha=0.5, zorder=0)

            # 4 & 5. Intermediate and E-Valley (hmE to hEF)
            if anchor_heights['hEF'] and anchor_heights['hmE']:
                # Calculate a midpoint to separate Region 4 and 5 since h_VT is not in the dataset
                h_mid = (anchor_heights['hmE'] + anchor_heights['hEF']) / 2
                ax1.axhspan(h_mid, anchor_heights['hEF'], color=c_inter, alpha=0.5, zorder=0)   # 4. Intermediate
                ax1.axhspan(anchor_heights['hmE'], h_mid, color=c_evalley, alpha=0.5, zorder=0) # 5. E-Valley

            # 6. E/D (< hmE)
            if anchor_heights['hmE']:
                # Colors from the very bottom of the plot up to hmE
                ax1.axhspan(y_min, anchor_heights['hmE'], color=c_ed, alpha=0.5, zorder=0)
                
            # Optional: Re-apply the y-limits so the colored spans don't accidentally expand the plot axes
            ax1.set_ylim(y_min, y_max)
        
            ax1.legend(fontsize=12, loc='upper left')
        
            # ==========================================
            # Subplot 2: The Relative Error
            # ==========================================
            ax2.plot(error_pct, altitudes, color='purple', linewidth=2)
            
            # Draw a solid reference line at 0% error
            ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
            
            ax2.set_xlabel('Error (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Relative Error', fontsize=14, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            # Clean up the layout so the plots don't overlap
            plt.tight_layout()
            plt.show()
        # 5. Append the calculated profile back to the xarray dataset for easy comparison
        iono["ne_calculated"] = (("alt_km"), calculated_edp)
    
        # Run the comparison plot
        plot_edp_comparison(iono)
        
        plt.show() # Ensure the plot renders
        
    return iono
# ─────────────────────────────────────────────────────────────────────────────
#  §2  Three solar-activity scenarios compared on one altitude profile
# ─────────────────────────────────────────────────────────────────────────────

def section2() -> list[xr.Dataset]:
    _banner("§2  Three solar-activity scenarios (3 IRI calls)")

    results = []
    for sc in SCENARIOS:
        print(f"  {sc['label']:16s}  f107D={sc['f107D']:3d}  ap={sc['ap']:2d}"
              f"  IG12={sc['IG12']:3d}  Rz12={sc['Rz12']:2d}", end="  →  ")
        iono = IRI(TIME, ALT_KM, LAT_C, LON_C,
                   f107D=sc["f107D"], ap=sc["ap"],
                   IG12=sc["IG12"], Rz12=sc["Rz12"])
        print(f"NmF2={float(iono['NmF2']):.2e} m⁻³  "
              f"hmF2={float(iono['hmF2']):.0f} km  "
              f"TEC={float(iono['TEC']):.2e} m⁻²")
        results.append(iono)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"§2  Solar scenario comparison\n"
        f"lat {LAT_C} °N   lon {LON_C} °E   {TIME}"
    )

    for iono, sc, color in zip(results, SCENARIOS, COLORS):
        alt = iono.alt_km.values
        axes[0].plot(iono["ne"].values,  alt, color=color, label=sc["label"])
        axes[1].plot(iono["Ti"].values,  alt, color=color, label=sc["label"])
        axes[1].plot(iono["Te"].values,  alt, color=color, ls="--", lw=1)
        axes[2].plot(iono["nO+"].values, alt, color=color, label=sc["label"])

    axes[0].set_xlabel("Electron density ne (m⁻³)")
    axes[1].set_xlabel("Ion temp Ti  /  Electron temp Te (K)  [dashed]")
    axes[2].set_xlabel("O⁺ density (m⁻³)")
    for ax in axes:
        ax.set_ylabel("Altitude (km)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    axes[1].set_xscale("linear")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  §3  Sensitivity to ionospheric shape parameters (foF2, hmF2, B0, B1)
# ─────────────────────────────────────────────────────────────────────────────

def section3() -> None:
    _banner("§3  Sensitivity to shape parameters (foF2, hmF2, B0, B1)")
    print("  1 nominal + 4 × 10 perturbed IRI calls …")

    iono_nom = IRI(TIME, ALT_KM, LAT_C, LON_C)
    nom = {
        "foF2": float(iono_nom["foF2"]),
        "hmF2": float(iono_nom["hmF2"]),
        "B0":   float(iono_nom["B0"]),
        "B1":   float(iono_nom["B1"]),
    }
    print(f"  Nominal: foF2={nom['foF2']:.2f} MHz  hmF2={nom['hmF2']:.0f} km"
          f"  B0={nom['B0']:.1f} km  B1={nom['B1']:.3f}")

    factors = np.linspace(0.5, 2.0, 10)

    def sweep(vary_key: str) -> list[xr.Dataset]:
        out = []
        for f in factors:
            kw = dict(nom)
            kw[vary_key] = kw[vary_key] * f
            out.append(IRI(TIME, ALT_KM, LAT_C, LON_C, **kw))
        return out

    results_foF2 = sweep("foF2")
    results_hmF2 = sweep("hmF2")
    results_B0   = sweep("B0")
    results_B1   = sweep("B1")

    # iri_plots.altprofile_sensitivity expects iParam_Switch=1 for shape params
    iri_plots.altprofile_sensitivity(
        1, iono_nom,
        results_foF2, results_hmF2,
        results_B0, results_B1,
    )
    plt.gcf().suptitle(
        f"§3  Shape-parameter sensitivity\n"
        f"lat {LAT_C} °N   lon {LON_C} °E   {TIME}",
        fontsize=9,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  §4  24-hour time profile at a fixed location
# ─────────────────────────────────────────────────────────────────────────────

def section4() -> xr.Dataset:
    _banner("§4  24-hour time profile  (lat 62.5 °N  lon -145 °E)")
    print("  Running 12 IRI calls (one per 2 h) …")

    iono = timeprofile(
        ("2022-06-15 00:00", "2022-06-16 00:00"),
        timedelta(hours=0.5),
        ALT_KM, LAT_C, LON_C,)
    print(f"  Dataset dims: {dict(iono.sizes)}")

    iri_plots.timeprofile(iono)
    # timeprofile creates two figures; label the most recently opened pair
    open_figs = plt.get_fignums()
    for fn in open_figs[-2:]:
        plt.figure(fn).suptitle(
            f"§4  Time profile   lat {LAT_C} °N   lon {LON_C} °E",
            fontsize=9,
        )
    return iono


# ─────────────────────────────────────────────────────────────────────────────
#  §5  Latitude sweep 60→65 °N across the Alaska region
# ─────────────────────────────────────────────────────────────────────────────

def section5() -> xr.Dataset:
    _banner("§5  Latitude sweep  60→65 °N  at 300 km,  lon -145 °E")
    print("  Running IRI calls at lats = [60.0, 62.5, 65.0] …")

    iono = geoprofile(
        latrange=(LAT_MIN, LAT_MAX + 0.1, DLAT),
        glon=LON_C,
        altkm=300.0,
        time=TIME,)

    iri_plots.latprofile(iono)
    plt.gcf().suptitle(
        f"§5  Latitude profile   lon {LON_C} °E   {TIME}",
        fontsize=9,
    )
    return iono


# ─────────────────────────────────────────────────────────────────────────────
#  §6  Rectangular triangular mesh: generate and plot
# ─────────────────────────────────────────────────────────────────────────────

def section6() -> tuple[np.ndarray, np.ndarray]:
    _banner("§6  Rectangular triangular mesh  (Alaska region)")
    from EDPSamples.plot_mesh_globe import plot_globe_occultation_mesh
    # generate_rect_tri_mesh(minLat, maxLat, dLat, minLon, maxLon, dLon)
    # Returns vertices as (N, 2) columns = (longitude, latitude)
    vertices, triangles = generate_rect_tri_mesh(
        LAT_MIN, LAT_MAX, DLAT,
        LON_MIN, LON_MAX, DLON,
    )
    print(f"  Vertices : {vertices.shape[0]}")
    print(f"  Triangles: {triangles.shape[0]}")
    print(f"  First 3 vertices (lon, lat):\n{vertices[:3]}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tri_mesh(vertices, triangles, ax=ax)
    ax.set_title(
        f"§6  Rectangular mesh   lat {LAT_MIN}–{LAT_MAX} °N   "
        f"lon {LON_MIN}–{LON_MAX} °E\n"
        f"dLat={DLAT}°   dLon={DLON}°   →   "
        f"{vertices.shape[0]} vertices,  {triangles.shape[0]} triangles"
    )
    save_path = "./Figures/Examples/Alaska_region.png"
    
    print("Running plotting code...")
    plot_globe_occultation_mesh(vertices, triangles, LAT_C, LON_C, save_path)
    print("Complete\n")
    return vertices, triangles


# ─────────────────────────────────────────────────────────────────────────────
#  §7  Polar-cap triangular mesh: generate and plot
# ─────────────────────────────────────────────────────────────────────────────

def section7() -> tuple[np.ndarray, np.ndarray]:
    _banner("§7  Polar-cap mesh   60 °N → North Pole")

    # generate_ploar_mesh is the name in the source (typo: "ploar")
    vertices, triangles = generate_ploar_mesh("north", minLat=60.0, dLat=1.0)
    vertices  = np.array(vertices)
    triangles = np.array(triangles)
    print(f"  Vertices : {vertices.shape[0]}")
    print(f"  Triangles: {len(triangles)}")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    plot_polar_mesh(vertices, triangles, pole="north", ax=ax)
    ax.set_title("§7  Polar-cap mesh   60 °N → North Pole", va="bottom")
    return vertices, triangles


# ─────────────────────────────────────────────────────────────────────────────
#  §8  EDPSamples: fill the Alaska rectangle with IRI for 3 solar scenarios
# ─────────────────────────────────────────────────────────────────────────────

def section8(rect_vertices: np.ndarray, rect_triangles: np.ndarray) -> EDPSamples:
    _banner("§8  EDPSamples  —  Alaska rect mesh × 3 solar scenarios")

    # Altitude grid (1-D numpy array)
    alt_grid = np.arange(ALT_KM[0], ALT_KM[1] + 1, ALT_KM[2], dtype=float)
    n_height = len(alt_grid)

    # sampling_parameters DataFrame: one row per solar scenario
    # Columns must be: hour, f107, ap, ig12, rz12
    sampling_df = pd.DataFrame([
        {
            "hour": 12.0,
            "f107": float(sc["f107D"]),
            "ap":   float(sc["ap"]),
            "ig12": float(sc["IG12"]),
            "rz12": float(sc["Rz12"]),
        }
        for sc in SCENARIOS
    ])
    n_sample = len(sampling_df)
    n_geo    = rect_vertices.shape[0]

    print(f"  Altitude levels : {n_height}  ({alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km)")
    print(f"  Mesh vertices   : {n_geo}")
    print(f"  Solar scenarios : {n_sample}")
    print(f"  Total IRI calls : {n_geo * n_sample}")

    # ── Run IRI for every vertex × scenario, fill edps array ─────────────────
    # edps shape: (height, geo_vertex, sample)
    edps = np.full((n_height, n_geo, n_sample), np.nan)

    for i_s, sc in enumerate(SCENARIOS):
        for i_g in range(n_geo):
            lon_v, lat_v = rect_vertices[i_g]      # vertices are (lon, lat)
            # print(f"  scenario {i_s+1}/{n_sample}  "
            #       f"vertex {i_g+1:2d}/{n_geo}  "
            #       f"({lat_v:.1f} °N, {lon_v:.1f} °E)",
            #       end="\r", flush=True)
            iono = IRI(
                TIME, ALT_KM, lat_v, lon_v,
                f107D=sc["f107D"], ap=sc["ap"],
                IG12=sc["IG12"], Rz12=sc["Rz12"],
            )
            edps[:, i_g, i_s] = iono["ne"].values
    print()  # clear the \r progress line

    # ── Construct EDPSamples ─────────────────────────────────────────────────
    # Passing a pre-computed edps array bypasses the evaluate_iri / hardcoded-
    # path code path in __init__.  The DIM_VERTEX patch (applied at import
    # time above) prevents the AttributeError from the missing class attribute.
    eds = EDPSamples(
        DateTime=TIME,
        geo_type="Rectangle",
        altitude=alt_grid,
        sampling_parameters=sampling_df,
        edps=edps,
        minLon=LON_MIN, maxLon=LON_MAX, dLon=DLON,
        minLat=LAT_MIN, maxLat=LAT_MAX, dLat=DLAT,
    )

    print(f"  EDPSamples dims  : {dict(eds.sizes)}")
    print(f"  EDPs shape       : {eds.edps.shape}   (height, geo_vertex, sample)")
    print(f"  Sampling params  :\n{sampling_df.to_string(index=False)}")

    # ── Plot: ne altitude profiles by solar scenario ──────────────────────────
    fig, axes = plt.subplots(1, n_sample, figsize=(5 * n_sample, 7), sharey=True)
    fig.suptitle(
        f"§8  EDPSamples — ne profiles per solar scenario\n"
        f"Alaska rect mesh  ({n_geo} vertices)   {TIME}"
    )

    # Vertex closest to the center of the region
    dists    = np.hypot(rect_vertices[:, 0] - LON_C, rect_vertices[:, 1] - LAT_C)
    i_center = int(np.argmin(dists))

    for i_s, (sc, ax, color) in enumerate(zip(SCENARIOS, axes, COLORS)):
        for i_g in range(n_geo):
            ax.plot(edps[:, i_g, i_s], alt_grid, color="lightgray", lw=0.7)
        ax.plot(
            edps[:, i_center, i_s], alt_grid,
            color=color, lw=2.5,
            label=(f"center  ({rect_vertices[i_center,1]:.1f} °N, "
                   f"{rect_vertices[i_center,0]:.1f} °E)"),
        )
        ax.set_xscale("log")
        ax.set_xlabel("ne (m⁻³)")
        ax.set_title(sc["label"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("Altitude (km)")

    # ── Save / load roundtrip ─────────────────────────────────────────────────
    ncpath = ROOT / "alaska_edp_demo.nc"
    try:
        eds.saveNetCDF(ncpath)
        print(f"\n  Saved  → {ncpath.name}  ({ncpath.stat().st_size // 1024} KB)")
        eds_loaded = EDPSamples.fromNetCDF(ncpath)
        print(f"  Loaded   EDPSamples dims: {dict(eds_loaded.sizes)}")
    except Exception as exc:
        print(f"\n  NetCDF roundtrip skipped: {exc}")

    # ── Demonstrate interp_heights ────────────────────────────────────────────
    query_alts = np.array([120.0, 250.0, 350.0, 450.0])
    idx, w = interp_heights(alt_grid, query_alts)
    print(f"\n  interp_heights — query altitudes: {query_alts}")
    print(f"    bracket indices : {idx}")
    print(f"    lower weights   : {w[:, 0].round(3)}")
    print(f"    upper weights   : {w[:, 1].round(3)}")

    # ── Demonstrate find_containing_triangles (spherical version) ─────────────
    # geolocation passed as (lat, lon) — the convention of locate_in_mesh.py
    geoloc_latlon = rect_vertices[:, [1, 0]]   # swap (lon,lat) → (lat,lon)
    query_pts = np.array([
        [62.5, -145.0],   # region center
        [61.0, -148.0],   # SW
        [64.0, -142.0],   # NE
    ])
    tri_idx, bary = find_triangle_sphere(
        query_pts, geoloc_latlon, rect_triangles, return_bary=True
    )
    print(f"\n  find_containing_triangles ({len(query_pts)} queries):")
    for q, ti, b in zip(query_pts, tri_idx, bary):
        print(f"    ({q[0]:.1f} °N, {q[1]:.1f} °E)  →  triangle {ti}"
              f"   bary = {b.round(3)}")

    return eds


# ─────────────────────────────────────────────────────────────────────────────
#  §9  Bundled solar-index data files: read_apf107, read_ig_rz, show_iri_inputs
# ─────────────────────────────────────────────────────────────────────────────

def section9() -> None:
    _banner("§9  Bundled solar-index data  (read_apf107, read_ig_rz)")

    with impr.as_file(impr.files("iri2020").joinpath("data")) as data_path:
        apf107 = read_apf107(str(data_path))
        ig_rz  = read_ig_rz(str(data_path))

    n_days = len(apf107["yr"])
    print(f"  apf107 : {n_days} daily records  "
          f"({apf107['yr'][0]}–{apf107['yr'][-1]})")
    print(f"  F10.7  range : {min(apf107['f107']):.1f} – {max(apf107['f107']):.1f} sfu")
    print(f"  Ap     range : {min(apf107['iapda'])} – {max(apf107['iapda'])}")
    print(f"  ig_rz  : {len(ig_rz['ig'])} monthly IG records  /  "
          f"{len(ig_rz['rz'])} Rz records")
    print(f"  IG12   range : {min(ig_rz['ig']):.1f} – {max(ig_rz['ig']):.1f}")
    print(f"  Rz12   range : {min(ig_rz['rz']):.1f} – {max(ig_rz['rz']):.1f}")

    from iri2020.get_iri_inputs import show_iri_inputs
    show_iri_inputs(apf107, ig_rz)
    plt.gcf().suptitle("§9  Bundled IRI solar-index time histories", fontsize=16)


# ─────────────────────────────────────────────────────────────────────────────
# §10  IRI_Sample_Inputs: download live data + quantile-sample DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def section10() -> None:
    _banner("§10  IRI_Sample_Inputs  (downloads live apf107 + ig_rz)")
    print("  Connecting to chain-new.chain-project.net …")
    try:
        inp = IRI_Sample_Inputs(2022, 6, 15, 12, 0, 0)
        print(f"  apf107  : {len(inp.apf107['yr'])} records  "
              f"({inp.apf107['yr'][0]}–{inp.apf107['yr'][-1]})")
        print(f"  ig_rz   : {len(inp.ig_rz['ig'])} IG values")

        # Build a sample plan: ±81 days of f107, ±27 days of ap
        samples = inp.quantileSamples(
            f107_sample_range=81,
            ap_sample_range=27,
        )
        print(f"  Sample plan: {len(samples)} rows × {len(samples.columns)} params")
        print(f"  Columns: {list(samples.columns)}")
        print(samples.describe().round(2).to_string())

        # Show the live time-history plot
        from IRI_Sample_Inputs.IRI_Sample_inputs import show_iri_inputs
        show_iri_inputs(inp.apf107, inp.ig_rz)
        plt.gcf().suptitle("§10  Live IRI input parameter time histories", fontsize=16)

    except Exception as exc:
        print(f"  Skipped — {type(exc).__name__}: {exc}")
        print("  (No network, server unreachable, or date not in downloaded data)")


# ─────────────────────────────────────────────────────────────────────────────
#  §11  Point-mode EDPSamples: quantile ensemble + NetCDF roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def section11() -> None:
    _banner("§11  Point-mode EDPSamples: quantile ensemble + NetCDF roundtrip")

    # 2003-11-21 is four days after the Halloween geomagnetic storm peak
    # (solar cycle 23 maximum), which makes the wide ig/rz sample ranges
    # physically meaningful.
    POINT_TIME = "2003-11-21T12"

    print("  Fetching live apf107 / ig_rz for 2003-11-21 …")
    try:
        inp = IRI_Sample_Inputs(2003, 11, 21, 12, 0, 0)
    except Exception as exc:
        print(f"  Skipped — {type(exc).__name__}: {exc}")
        print("  (No network or server unreachable)")
        return

    # Build quantile sample plan across all five axes.
    # NOTE: quantileSamples() has a known bug — it uses f107_range instead of
    # ig12_range in the IG12 loop, so the ig12 column values may be incorrect
    # until that bug is fixed in IRI_Sample_inputs.py.
    samples = inp.quantileSamples(
        hour_sample_range=6,
        ap_sample_range=31,
        f107_sample_range=31,
        ig_sample_range=48,
        rz_sample_range=48,
    )
    print(f"  Sample plan: {len(samples)} rows × {len(samples.columns)} columns")
    print(f"  Columns: {list(samples.columns)}")
    print(f"  → {len(samples)} IRI calls will be made internally")

    alt_grid = np.arange(80, 501, 10, dtype=float)   # 80–500 km, 43 levels
    print(f"  Altitude grid: {alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km "
          f"({len(alt_grid)} levels)")

    # evaluate_iri=1 tells EDPSamples to call IRI internally for each sample
    # row. This code path uses internal path construction that may fail outside
    # the original developer's environment; fall back to manual filling if so.
    eds = None
    try:
        eds = EDPSamples(
            DateTime=POINT_TIME,
            geo_type="Point",
            altitude=alt_grid,
            sampling_parameters=samples,
            evaluate_iri=0,
            Lon=LON_C,
            Lat=LAT_C,
        )
        print(f"  EDPSamples (internal IRI) dims: {dict(eds.sizes)}")
    except Exception as exc:
        print(f"  evaluate_iri=1 path failed ({type(exc).__name__}: {exc})")
        print("  Falling back to manual IRI loop …")

    if eds is None:
        # Manual fallback: loop over sample rows and call IRI() directly,
        # mirroring the approach used in §8.
        edps = np.full((len(alt_grid), 1, len(samples)), np.nan)
        for i_s, row in samples.iterrows():
            print(f"  sample {i_s+1}/{len(samples)}", end="\r", flush=True)
            iono = IRI(
                POINT_TIME,
                [float(alt_grid[0]), float(alt_grid[-1]), float(alt_grid[1] - alt_grid[0])],
                LAT_C, LON_C,
                f107D=float(row["f107"]),
                ap=float(row["ap"]),
                IG12=float(row["ig12"]),
                Rz12=float(row["rz12"]),
            )
            edps[:, 0, i_s] = iono["ne"].values
        print()
        eds = EDPSamples(
            DateTime=POINT_TIME,
            geo_type="Point",
            altitude=alt_grid,
            sampling_parameters=samples,
            edps=edps,
            Lon=LON_C,
            Lat=LAT_C,
        )
        print(f"  EDPSamples (manual fallback) dims: {dict(eds.sizes)}")

    # ── NetCDF roundtrip ──────────────────────────────────────────────────────
    ncpath = ROOT / "point_ensemble_demo.nc"
    eds.saveNetCDF(ncpath)
    print(f"\n  Saved  → {ncpath.name}  ({ncpath.stat().st_size // 1024} KB)")

    eds_loaded = EDPSamples.fromNetCDF(ncpath)
    print(f"  Loaded   EDPSamples dims: {dict(eds_loaded.sizes)}")

    match = np.allclose(
        eds.edps.values,
        eds_loaded.edps.values,
        equal_nan=True,
    )
    print(f"  Roundtrip data integrity : {'PASS ✓' if match else 'FAIL ✗'}")

    # ── Plot: ensemble spread of ne profiles at the single point ──────────────
    edps_vals = eds.edps.values[:, 0, :]   # shape (height, sample)

    fig, ax = plt.subplots(figsize=(7, 8))
    for i_s in range(edps_vals.shape[1]):
        ax.plot(edps_vals[:, i_s], alt_grid, color="lightsteelblue", lw=0.5, alpha=0.6)

    ne_median = np.nanmedian(edps_vals, axis=1)
    ne_p10    = np.nanpercentile(edps_vals, 10, axis=1)
    ne_p90    = np.nanpercentile(edps_vals, 90, axis=1)

    ax.fill_betweenx(alt_grid, ne_p10, ne_p90, alpha=0.25,
                     color="steelblue", label="10th–90th percentile")
    ax.plot(ne_median, alt_grid, color="steelblue", lw=2.5, label="Median")

    ax.set_xscale("log")
    ax.set_xlabel("Electron density ne (m⁻³)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        f"§11  ne ensemble spread at ({LAT_C} °N, {LON_C} °E)\n"
        f"{POINT_TIME}   {edps_vals.shape[1]} quantile samples"
    )
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    
# ─────────────────────────────────────────────────────────────────────────────
# §12  Occultation Defined Grid Points
# ─────────────────────────────────────────────────────────────────────────────
def section12(podTc2_file: str) -> tuple[np.ndarray, np.ndarray, tuple, tuple, tuple]:
    from EDPSamples.generate_occultation_tri_mesh import generate_occultation_mesh
    from EDPSamples.plot_mesh_globe import plot_globe_occultation_mesh
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file
    
    _banner("§12 Occultation Defined Grid Points")
    
    # Extract the filename from the path to use in the save string
    podTc2_string = podTc2_file.split('/')[-1]

    save_path = f"./Figures/Examples/{podTc2_string}_mesh_geometry.png"
    
    podTc_data = parse_podTc2_nc_file(podTc2_file)
    
    print("Generating Occultation Mesh...")
    vertices_podTc, triangles_podTc, pt1, pt2, pt3 = generate_occultation_mesh(filename=podTc2_file, dLat=5, dLon=5)
    print("Complete\n")
    
    print("Running plotting code...")
    plot_globe_occultation_mesh(vertices_podTc, triangles_podTc, podTc_data['lat_tecmax_tangent'], podTc_data['lon_tecmax_tangent'], save_path)
    print("Complete\n")
    
    return vertices_podTc, triangles_podTc, pt1, pt2, pt3   

def section13(rect_vertices: np.ndarray, rect_triangles: np.ndarray, pt1: tuple, pt2: tuple, pt3: tuple) -> EDPSamples:
    from EDPSamples.EDPS_plotting import plot_edp_statistics
    _banner("§13  EDPSamples  —  Occultation Mesh × 3 solar scenarios")

    # Altitude grid (1-D numpy array)
    alt_grid = np.arange(ALT_KM[0], ALT_KM[1] + 1, ALT_KM[2], dtype=float)
    n_height = len(alt_grid)

    # sampling_parameters DataFrame: one row per solar scenario
    # Columns must be: hour, f107, ap, ig12, rz12
    sampling_df = pd.DataFrame([
        {
            "hour": 12.0,
            "f107": float(sc["f107D"]),
            "ap":   float(sc["ap"]),
            "ig12": float(sc["IG12"]),
            "rz12": float(sc["Rz12"]),
        }
        for sc in SCENARIOS
    ])
    n_sample = len(sampling_df)
    n_geo    = rect_vertices.shape[0]

    print(f"  Altitude levels : {n_height}  ({alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km)")
    print(f"  Mesh vertices   : {n_geo}")
    print(f"  Solar scenarios : {n_sample}")
    print(f"  Total IRI calls : {n_geo * n_sample}")

    # ── Run IRI for every vertex × scenario, fill edps array ─────────────────
    # edps shape: (height, geo_vertex, sample)
    edps = np.full((n_height, n_geo, n_sample), np.nan)

    for i_s, sc in enumerate(SCENARIOS):
        for i_g in range(n_geo):
            lon_v, lat_v = rect_vertices[i_g]      # vertices are (lon, lat)
            # print(f"  scenario {i_s+1}/{n_sample}  "
            #       f"vertex {i_g+1:2d}/{n_geo}  "
            #       f"({lat_v:.1f} °N, {lon_v:.1f} °E)",
            #       end="\r", flush=True)
            iono = IRI(
                TIME, ALT_KM, lat_v, lon_v,
                f107D=sc["f107D"], ap=sc["ap"],
                IG12=sc["IG12"], Rz12=sc["Rz12"],
            )
            edps[:, i_g, i_s] = iono["ne"].values
    print()  # clear the \r progress line

    # ── Construct EDPSamples ─────────────────────────────────────────────────
    # Passing a pre-computed edps array bypasses the evaluate_iri / hardcoded-
    # path code path in __init__.  The DIM_VERTEX patch (applied at import
    # time above) prevents the AttributeError from the missing class attribute.
    eds = EDPSamples(
        DateTime=TIME,
        geo_type="Occultation",
        altitude=alt_grid,
        sampling_parameters=sampling_df,
        edps=edps,
        pt1 = pt1,
        pt2 = pt2,
        pt3 = pt3
    )

    print(f"  EDPSamples dims  : {dict(eds.sizes)}")
    print(f"  EDPs shape       : {eds.edps.shape}   (height, geo_vertex, sample)")
    print(f"  Sampling params  :\n{sampling_df.to_string(index=False)}")

    # ── Plot: ne altitude profiles by solar scenario ──────────────────────────
    fig, axes = plt.subplots(1, n_sample, figsize=(5 * n_sample, 7), sharey=True)
    fig.suptitle(
        f"§8  EDPSamples — ne profiles per solar scenario\n"
        f"Alaska rect mesh  ({n_geo} vertices)   {TIME}"
    )

    # Vertex closest to the center of the region
    dists    = np.hypot(rect_vertices[:, 0] - LON_C, rect_vertices[:, 1] - LAT_C)
    i_center = int(np.argmin(dists))

    for i_s, (sc, ax, color) in enumerate(zip(SCENARIOS, axes, COLORS)):
        for i_g in range(n_geo):
            ax.plot(edps[:, i_g, i_s], alt_grid, color="lightgray", lw=0.7)
        ax.plot(
            edps[:, i_center, i_s], alt_grid,
            color=color, lw=2.5,
            label=(f"center  ({rect_vertices[i_center,1]:.1f} °N, "
                   f"{rect_vertices[i_center,0]:.1f} °E)"),
        )
        ax.set_xscale("log")
        ax.set_xlabel("ne (m⁻³)")
        ax.set_title(sc["label"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("Altitude (km)")

    # ── Save / load roundtrip ─────────────────────────────────────────────────
    ncpath = ROOT / "alaska_edp_demo.nc"
    try:
        eds.saveNetCDF(ncpath)
        print(f"\n  Saved  → {ncpath.name}  ({ncpath.stat().st_size // 1024} KB)")
        eds_loaded = EDPSamples.fromNetCDF(ncpath)
        print(f"  Loaded   EDPSamples dims: {dict(eds_loaded.sizes)}")
    except Exception as exc:
        print(f"\n  NetCDF roundtrip skipped: {exc}")

    # ── Demonstrate interp_heights ────────────────────────────────────────────
    query_alts = np.array([120.0, 250.0, 350.0, 450.0])
    idx, w = interp_heights(alt_grid, query_alts)
    print(f"\n  interp_heights — query altitudes: {query_alts}")
    print(f"    bracket indices : {idx}")
    print(f"    lower weights   : {w[:, 0].round(3)}")
    print(f"    upper weights   : {w[:, 1].round(3)}")

    # ── Demonstrate find_containing_triangles (spherical version) ─────────────
    # geolocation passed as (lat, lon) — the convention of locate_in_mesh.py
    geoloc_latlon = rect_vertices[:, [1, 0]]   # swap (lon,lat) → (lat,lon)
    query_pts = np.array([
        [62.5, -145.0],   # region center
        [61.0, -148.0],   # SW
        [64.0, -142.0],   # NE
    ])
    tri_idx, bary = find_triangle_sphere(
        query_pts, geoloc_latlon, rect_triangles, return_bary=True
    )
    print(f"\n  find_containing_triangles ({len(query_pts)} queries):")
    for q, ti, b in zip(query_pts, tri_idx, bary):
        print(f"    ({q[0]:.1f} °N, {q[1]:.1f} °E)  →  triangle {ti}"
              f"   bary = {b.round(3)}")
        
    plot_edp_statistics(eds)
    return eds


def section14(eds, podTc_filename: str):
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file
    from TEC_model.podTc_file_processing import forward_model_mesh_tec
    from TEC_model.podTc_file_processing import rayTangent
    """
    Tests the modeled TEC against measured TEC from a podTc2 file.
    
    Parameters:
    -----------
    eds : EDPSamples
        The initialized and populated EDPSamples dataset (from section 13).
    podTc_filename : str
        Path to the podTc2 NetCDF file containing the measured RO pass.
    """
    try:
        _banner("§8  Section 14  —  Modeled vs Measured TEC Comparison")
    except NameError:
        print("\n=== Section 14 — Modeled vs Measured TEC Comparison ===")

    # 1. Parse the observed data
    print(f"  Loading measured data from : {podTc_filename}")
    data = parse_podTc2_nc_file(podTc_filename)
    
    LEO = data['LEO']
    GNSS = data['GNSS']
    
    # Safely extract TEC (keys can vary based on your parser implementation)
    if 'TEC' in data:
        measured_tec = data['TEC']
    elif 'TEC_podTc2' in data:
        measured_tec = data['TEC_podTc2']
    else:
        print("  [!] Warning: Could not find 'TEC' or 'absolute_TEC' key in podTc2 data.")
        measured_tec = np.zeros(LEO.shape[1])

    n_rays = LEO.shape[1]
    n_samples = eds.sizes.get('sample', 1) # Support 1 or multiple solar scenarios
    print(f"  Total Occultation Rays   : {n_rays}")
    print(f"  Evaluating Scenarios     : {n_samples}")

    # 2. Calculate Tangent Altitudes for the plot's Y-axis
    print("  Calculating tangent altitudes for profile geometry...")
    _, _, tangent_alt = rayTangent(LEO, GNSS, units='km')

    # 3. Compute Forward Modeled TEC for each scenario
    modeled_tecs = []
    for i_s in range(n_samples):
        print(f"  Running forward model for scenario {i_s + 1}/{n_samples}...")
        # Make sure forward_model_mesh_tec is imported/available in this scope
        tec = forward_model_mesh_tec(eds, data, sample_idx=i_s, num_segments=1000)
        modeled_tecs.append(tec)

    # 4. Generate the Comparison Plot
    print("  Generating comparison plot...")
    fig, ax = plt.subplots(figsize=(7, 9))
    fig.suptitle(
        f"§8  Section 14 — Measured vs. Modeled TEC\n"
        f"Occultation Pass: {podTc_filename.split('/')[-1]}",
        fontsize=12
    )

    # Plot Measured TEC
    print(f"{measured_tec.shape},  {tangent_alt.shape}")
    print(f"Measured TEC - Min: {np.nanmin(measured_tec)}, Max: {np.nanmax(measured_tec)}, NaNs: {np.isnan(measured_tec).sum()}")
    print(f"Modeled TEC 1 - Min: {np.nanmin(modeled_tecs[0])}, Max: {np.nanmax(modeled_tecs[0])}, NaNs: {np.isnan(modeled_tecs[0]).sum()}")
    ax.plot(measured_tec, tangent_alt*1e-3, color='black', lw=2.5, label="Measured TEC (podTc2)")

    # Plot Modeled TEC Scenarios
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    
    # Try to grab scenario labels if they exist in your sampling parameters
    try:
        sample_params = eds.data_vars['sampling_parameters'].values
    except KeyError:
        sample_params = None

    for i_s in range(n_samples):
        c = colors[i_s % len(colors)]
        label = f"Modeled TEC (Scenario {i_s + 1})"
        
        ax.plot(modeled_tecs[i_s], tangent_alt*1e-3, color=c, lw=1.5, linestyle='--', label=label)

    # Formatting the plot to match typical RO profiles
    ax.set_ylabel("Tangent Altitude (km)")
    ax.set_xlabel("Total Electron Content (TECU)")
    
    # Restrict Y-axis to the valid altitude envelope
    valid_alts = tangent_alt[tangent_alt >= 0]
    if len(valid_alts) > 0:
        ax.set_ylim(0, min(np.max(valid_alts) + 50, eds.coords['altitude'].values[-1]))
    else:
        ax.set_ylim(0, 600)
        
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    # plt.show()

    return modeled_tecs, tangent_alt



def section15(lon_point: float = -145.0, lat_point: float = 62.5, n_mc_samples: int = 100):
    """
    Evaluates a single geographical point across a large Monte Carlo ensemble 
    of IRI input parameters to test statistical EDPSamples generation.
    """
    from EDPSamples.EDPS_plotting import plot_edp_statistics
    
    try:
        _banner("§15  EDPSamples  —  Single Point × Monte Carlo Ensemble")
    except NameError:
        print("\n=== Section 15 — Single Point × Monte Carlo Ensemble ===")

    # 1. Altitude grid (1-D numpy array)
    alt_grid = np.arange(ALT_KM[0], ALT_KM[1] + 1, ALT_KM[2], dtype=float)
    n_height = len(alt_grid)
    n_geo = 1  # Single point

    # 2. Generate a large range of varying input parameters (Monte Carlo)
    print(f"  Generating {n_mc_samples} synthetic solar scenarios...")
    np.random.seed(42) # Set seed for reproducible testing
    
    # Create normal distributions for typical solar/geomagnetic parameters, 
    # clipped to realistic boundaries to prevent IRI model crashes.
    sampling_df = pd.DataFrame({
        "hour": np.full(n_mc_samples, 12.0),
        "f107": np.random.normal(loc=130, scale=30, size=n_mc_samples).clip(70, 250),
        "ap":   np.random.normal(loc=15, scale=12, size=n_mc_samples).clip(0, 400),
        "ig12": np.random.normal(loc=100, scale=25, size=n_mc_samples).clip(50, 200),
        "rz12": np.random.normal(loc=100, scale=25, size=n_mc_samples).clip(50, 200),
    })

    print(f"  Altitude levels : {n_height}  ({alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km)")
    print(f"  Mesh vertices   : {n_geo}  (Point: {lat_point}°N, {lon_point}°E)")
    print(f"  Solar scenarios : {n_mc_samples} (Monte Carlo Distribution)")
    print(f"  Total IRI calls : {n_geo * n_mc_samples}")

    # 3. Run IRI for the single vertex across ALL scenarios
    # edps shape: (height, geo_vertex, sample)
    edps = np.full((n_height, n_geo, n_mc_samples), np.nan)

    for i_s in range(n_mc_samples):
        sc = sampling_df.iloc[i_s]
        
        # Display a simple progress tracker
        if (i_s + 1) % 10 == 0 or i_s == 0:
            print(f"    Running sample {i_s + 1}/{n_mc_samples}...", end="\r", flush=True)
            
        iono = IRI(
            TIME, ALT_KM, lat_point, lon_point,
            f107D=sc["f107"], ap=sc["ap"],
            IG12=sc["ig12"], Rz12=sc["rz12"],
        )
        edps[:, 0, i_s] = iono["ne"].values
        
    print(f"    Running sample {n_mc_samples}/{n_mc_samples}... Done!  ")

    # 4. Construct the EDPSamples Object
    # Note: geo_type is explicitly "Point", and we pass Lon/Lat instead of pt1/pt2/pt3
    eds = EDPSamples(
        DateTime=TIME,
        geo_type="Point",
        altitude=alt_grid,
        sampling_parameters=sampling_df,
        edps=edps,
        Lon=lon_point,
        Lat=lat_point
    )

    print(f"\n  EDPSamples dims  : {dict(eds.sizes)}")
    print(f"  EDPs shape       : {eds.edps.shape}   (height, geo_vertex, sample)")

    # 5. Plot the raw "Spaghetti" Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.suptitle(f"§15 EDPSamples — Raw Monte Carlo Ensemble\nPoint: {lat_point}°N, {lon_point}°E at {TIME}")
    
    # Plot all samples lightly in the background
    for i_s in range(n_mc_samples):
        ax.plot(edps[:, 0, i_s], alt_grid, color="tab:blue", alpha=0.1, lw=1.0)
        
    # Plot the mean line over the top
    mean_profile = np.nanmean(edps[:, 0, :], axis=1)
    ax.plot(mean_profile, alt_grid, color="black", lw=2.5, label="Ensemble Mean")
    
    ax.set_xscale("log")
    ax.set_xlabel("ne (m⁻³)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.legend()
    plt.tight_layout()
    # plt.show()

    # 6. Call the Statistical Plotting Code
    print("\n  Generating Statistical Distribution Plots...")
    plot_edp_statistics(eds)

    return eds


def section16(podTc2_file: str, alt_grid: np.ndarray, sampling_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    §16 Comparative Mesh Analysis
    Generates 4 distinct EDPSamples meshes (Point, Occultation, Rectangle, Polar)
    across an ensemble of states (±3 hours), evaluates the nominal TEC, and plots 
    the spatial and state covariances.
    """
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
    from EDPSamples.edp_samples import EDPSamples
    
    try:
        _banner("§16  Comparing 4 Mesh Types Across State Range")
    except NameError:
        print("\n=== §16  Comparing 4 Mesh Types Across State Range ===")
        
    filename = podTc2_file.split('/')[-1]
    print(f"Processing File: {filename}")

    # 1. Parse podTc2 data to anchor the geometry
    podTc_data = parse_podTc2_nc_file(podTc2_file)
    if podTc_data is None:
        print(" [!] Invalid or skipped podTc data. Aborting comparison.")
        return {}, {}

    lat_c = podTc_data['lat_tecmax_tangent']
    lon_c = podTc_data['lon_tecmax_tangent']
    dlat_step = 5
    dlon_step = 5

    # ---------------------------------------------------------
    # 2. Generate State Ensemble (±3 Hours + Solar Variance)
    # ---------------------------------------------------------
    print("\n  -> Generating State Ensemble (±3 hours, varying solar parameters)")
    time_str = "2015-06-01 12:00:00"  # Base nominal time
    
    # Extract the base scenario
    base_sc = sampling_df.iloc[0]
    n_mc = 50  # Number of states to simulate for the covariance matrix
    np.random.seed(42)

    # Vary hour by ±3 hours, wrapping around midnight
    mc_hours = (base_sc['hour'] + np.random.uniform(-3, 3, size=n_mc)) % 24

    mc_df = pd.DataFrame({
        "hour": mc_hours,
        "f107": np.random.normal(loc=base_sc.get('f107', 130), scale=10, size=n_mc).clip(70, 250),
        "ap":   np.random.normal(loc=base_sc.get('ap', 15), scale=5, size=n_mc).clip(0, 400),
        "ig12": np.random.normal(loc=base_sc.get('ig12', 100), scale=10, size=n_mc).clip(50, 200),
        "rz12": np.random.normal(loc=base_sc.get('rz12', 100), scale=10, size=n_mc).clip(50, 200),
    })

    # CRITICAL: Force the 0th sample to be the EXACT nominal base state 
    # so the forward TEC model calculates exactly as it did before.
    mc_df.iloc[0] = base_sc

    eds_dict = {}

    # ---------------------------------------------------------
    # 3. Generate EDPSamples for each Geo Type
    # ---------------------------------------------------------
    
    # A. POINT (State Distribution anchor)
    print("  -> Generating [Point] EDPSamples")
    eds_dict['Point'] = EDPSamples(
        DateTime=time_str, geo_type="Point", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1,
        Lon=lon_c, Lat=lat_c
    )

    # B. OCCULTATION
    print("  -> Generating [Occultation] EDPSamples")
    eds_dict['Occultation'] = EDPSamples(
        DateTime=time_str, geo_type="Occultation", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1,
        filename=podTc2_file, dLat=dlat_step, dLon=dlon_step
    )
    
    lat_lons = np.array([
        eds_dict['Occultation'].attrs['pt1'], 
        eds_dict['Occultation'].attrs['pt2'], 
        eds_dict['Occultation'].attrs['pt3']
    ])

    # C. RECTANGLE
    print("  -> Generating [Rectangle] EDPSamples")
    eds_dict['Rectangle'] = EDPSamples(
        DateTime=time_str, geo_type="Rectangle", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1,
        minLon=np.min(lat_lons[:,1]), maxLon=np.max(lat_lons[:,1]), dLon=dlon_step,
        minLat=np.min(lat_lons[:,0]), maxLat=np.max(lat_lons[:,0]), dLat=dlat_step
    )

    # D. POLAR
    print("  -> Generating [Polar] EDPSamples")
    lats = lat_lons[:, 0]
    if np.mean(lats) > 0: 
        min_lat_polar = np.max([np.min(lats[lats > 0]), 1.0]) if np.any(lats > 0) else 1.0
    else:
        min_lat_polar = np.min([np.max(lats[lats <= 0]), -1.0]) if np.any(lats <= 0) else -1.0
        
    print(f"     Polar Plot edge: {min_lat_polar:.2f}°")
    eds_dict['Polar'] = EDPSamples(
        DateTime=time_str, geo_type="Polar", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1,
        minLat=min_lat_polar, dLat=dlat_step
    )
    
    # ---------------------------------------------------------
    # 4. Analysis: Plots, Covariance, and TEC Models
    # ---------------------------------------------------------
    print("\n  -> Running Analysis and Forward Models...")
    results_tec = {}
    
    for name, eds in eds_dict.items():
        print(f"\n     --- Evaluating {name} mesh ---")
        
        try:
            if name == 'Point':
                print("      -> Plotting State Distribution (Temporal variance at Center Point)")
                eds.plot_edp_statistics(f"{name} Geometry ({n_mc} states)")
            else:
                print(f"      -> Plotting {name} Geometric Distribution (Spatial variance at base time)")
                eds_spatial = EDPSamples.from_xarray(eds.isel(sample=[0]))
                eds_spatial.plot_edp_statistics(f"{name} Geometry ({n_mc} states)")
                
                tecmax_lat = podTc_data['lat_tecmax_tangent']
                tecmax_lon = podTc_data['lon_tecmax_tangent']
                
                eds_spatial.plot_mesh_globe(
                    tecmax_lat, tecmax_lon,
                    save_path=f"./Figures/{name}_mesh_globe.png",
                    podTc_data=podTc_data
                )
                target_alts = [200.0, 300.0, 400.0]  # <--- Change these altitudes to whatever you want to investigate
                
                for alt in target_alts:
                    print(f"      -> Plotting {name} mesh data at {alt} km")
                    
                    # 1. Plot Electron Density (Ne) for the Nominal State (sample=0)
                    eds.plot_mesh_globe(
                        tecmax_lat, tecmax_lon,
                        save_path=f"./Figures/{name}_mesh_globe_Ne_{int(alt)}km.png",
                        podTc_data=podTc_data,
                        mesh_scalars="Ne",
                        target_alt=alt,
                        target_sample=0,
                        scalar_cmap='plasma'
                    )
                    
                    # 2. Plot Standard Deviation across the full MC ensemble
                    eds.plot_mesh_globe(
                        tecmax_lat, tecmax_lon,
                        save_path=f"./Figures/{name}_mesh_globe_StdDev_{int(alt)}km.png",
                        podTc_data=podTc_data,
                        mesh_scalars="StdDev",
                        target_alt=alt,
                        scalar_cmap='magma'
                    )
                print(f"      -> Plotting {name} State Covariance Matrix")
                eds.plot_edp_covariance()
                
        except Exception as e:
            print(f"      [!] Statistical plotting failed: {e}")
            
        print("      -> Running Forward TEC Integration")
        
        # ---------------------------------------------------------
        # NEW: Find the 5% and 95% sample indices based on total density
        # ---------------------------------------------------------
        # Sum the electron density across height and geo dimensions for each sample
        edp_totals = np.nansum(eds.edps, axis=(0, 1)) 
        sorted_indices = np.argsort(edp_totals)
        n_samples = len(sorted_indices)
        
        idx_05 = sorted_indices[int(0.05 * n_samples)]
        idx_95 = sorted_indices[int(0.95 * n_samples)]
        
        print(f"         [Ensemble Selection] Nominal: 0 | 5%: {idx_05} | 95%: {idx_95}")
        
        # Run forward models for nominal and the two boundary indices
        tec_nom = eds.forward_model_mesh_tec(podTc_data, sample_idx=0, num_segments=1000)
        tec_05  = eds.forward_model_mesh_tec(podTc_data, sample_idx=idx_05, num_segments=1000)
        tec_95  = eds.forward_model_mesh_tec(podTc_data, sample_idx=idx_95, num_segments=1000)
        
        # Store as a dictionary to easily unpack in the plotting loop
        results_tec[name] = {
            'nominal': tec_nom,
            'lower': tec_05,
            'upper': tec_95
        }

    # ---------------------------------------------------------
    # 5. Generate the Comparison Plot
    # ---------------------------------------------------------
    print("\n  -> Generating TEC Comparison Plot...")
    
    _, _, tangent_alt_raw = rayTangent(podTc_data['LEO'], podTc_data['GNSS'], units='km')
    tangent_alt_km = tangent_alt_raw * 1e-3

    measured_tec = podTc_data.get('TEC_podTc2', podTc_data.get('TEC', np.zeros_like(tangent_alt_km)))

    fig, ax = plt.subplots(figsize=(8, 10))
    fig.suptitle(f"§16 Mesh Geometry Comparison\nOccultation: {filename}", fontsize=12)

    ax.plot(measured_tec, tangent_alt_km, color='black', lw=3, label="Measured TEC")

    styles = {
        'Point':       {'color': 'tab:red',    'ls': ':'},
        'Occultation': {'color': 'tab:blue',   'ls': '--'},
        'Rectangle':   {'color': 'tab:green',  'ls': '-.'},
        'Polar':       {'color': 'tab:purple', 'ls': '-'}
    }

    for name in eds_dict.keys():
        nom_tec = results_tec[name]['nominal']
        low_tec = results_tec[name]['lower']
        upp_tec = results_tec[name]['upper']
        
        # Plot nominal line
        ax.plot(
            nom_tec, 
            tangent_alt_km, 
            color=styles[name]['color'], 
            ls=styles[name]['ls'], 
            lw=2, 
            label=f"Modeled ({name})"
        )
        
        # NEW: Plot the 5% to 95% bounds as a shaded background region
        ax.fill_betweenx(
            tangent_alt_km,
            low_tec,
            upp_tec,
            color=styles[name]['color'],
            alpha=0.1,            # Light transparency so it doesn't block other lines
            edgecolor='none'
        )
        ax.plot(
            low_tec, 
            tangent_alt_km, 
            color=styles[name]['color'], 
            ls=styles[name]['ls'],  # Match the line style of the nominal line
            lw=1.0,                 # Thinner line width to distinguish from nominal
            alpha=0.6,              # Slight transparency
            label='_nolegend_'      # Keeps it from cluttering the legend
        )
        
        # Upper bound
        ax.plot(
            upp_tec, 
            tangent_alt_km, 
            color=styles[name]['color'], 
            ls=styles[name]['ls'], 
            lw=1.0, 
            alpha=0.6,
            label='_nolegend_'
        )
        
    ax.set_ylabel("Tangent Altitude (km)")
    ax.set_xlabel("Total Electron Content (TECU)")
    
    valid_alts = tangent_alt_km[tangent_alt_km >= 0]
    if len(valid_alts) > 0:
        ax.set_ylim(0, min(np.max(valid_alts) + 50, alt_grid[-1]))
    else:
        ax.set_ylim(0, 600)
        
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    save_dir = "./Figures/Section16_Comparisons/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}_mesh_comparison.png")
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved figure to {save_path}\n")

    return eds_dict, results_tec

def section17(podTc2_file: str, alt_grid: np.ndarray, sampling_df: pd.DataFrame):
    """
    §17 Tomography Data Assimilation Test
    Initializes the Ionosphere_Tomography_Inverter, assimilates measured TEC data,
    and plots the Prior vs. Posterior states to evaluate filter performance.
    """
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
    from EDPSamples.edp_samples import EDPSamples
    
    # Make sure this matches where you saved the inverter class!
    # from EDPSamples.tomography_inverter import Ionosphere_Tomography_Inverter 
    
    try:
        _banner("§17  Tomography Data Assimilation (Kalman Filter)")
    except NameError:
        print("\n=== §17  Tomography Data Assimilation (Kalman Filter) ===")
        
    filename = podTc2_file.split('/')[-1]
    print(f"Processing File: {filename}")

    # 1. Parse Data and Clean NaNs
    podTc_data = parse_podTc2_nc_file(podTc2_file)
    if podTc_data is None:
        print(" [!] Invalid or skipped podTc data. Aborting.")
        return

    _, _, tangent_alt_raw = rayTangent(podTc_data['LEO'], podTc_data['GNSS'], units='km')
    tangent_alt_km = tangent_alt_raw * 1e-3
    measured_tec = podTc_data.get('TEC_podTc2', podTc_data.get('TEC', np.zeros_like(tangent_alt_km)))
    
    # CRITICAL: Kalman Filters cannot process NaNs. We must filter out invalid rays.
    valid_mask = ~np.isnan(measured_tec) & (measured_tec > 0)
    measured_tec_clean = measured_tec[valid_mask]
    tangent_alt_clean = tangent_alt_km[valid_mask]
    
    # Create a subset of the geometry dictionary for the filter
    podTc_clean = {
        'LEO': podTc_data['LEO'][:, valid_mask],
        'GNSS': podTc_data['GNSS'][:, valid_mask]
    }
    
    print(f"  -> Filtered {np.sum(~valid_mask)} invalid/NaN rays. {np.sum(valid_mask)} rays remaining.")

    # 2. Generate Prior Ensemble (Required for P and Q matrices)
    print("  -> Generating Prior State Ensemble...")
    base_sc = sampling_df.iloc[0]
    n_mc = 50  
    np.random.seed(42)

    mc_hours = (base_sc['hour'] + np.random.uniform(-3, 3, size=n_mc)) % 24
    mc_df = pd.DataFrame({
        "hour": mc_hours,
        "f107": np.random.normal(loc=base_sc.get('f107', 130), scale=10, size=n_mc).clip(70, 250),
        "ap":   np.random.normal(loc=base_sc.get('ap', 15), scale=5, size=n_mc).clip(0, 400),
        "ig12": np.random.normal(loc=base_sc.get('ig12', 100), scale=10, size=n_mc).clip(50, 200),
        "rz12": np.random.normal(loc=base_sc.get('rz12', 100), scale=10, size=n_mc).clip(50, 200),
    })
    mc_df.iloc[0] = base_sc  # Ensure 0th index is the baseline

    eds_occ = EDPSamples(
        DateTime=podTc_data['date'], geo_type="Occultation", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1,
        filename=podTc2_file, dLat=5, dLon=5
    )

    # 3. Initialize the Tomography Inverter
    print("  -> Initializing Kalman Filter...")
    # Initialize using meanscale=1 to use fractional perturbations
    inverter = Ionosphere_Tomography_Inverter(EDPSam=eds_occ, meanscale=1)
    
    # Calculate the unscaled H matrix explicitly so we can use it to calculate absolute TEC
    print("  -> Building Observation Operator (H)...")
    H_unscaled = inverter.get_observation_operator(podTc_clean)
    # ---------------------------------------------------------
    # Plot the H Matrix Structure
    # ---------------------------------------------------------
    print("  -> Plotting H Matrix Structure...")
    from matplotlib.colors import LogNorm
    
    fig_H, ax_H = plt.subplots(figsize=(10, 8))
    fig_H.suptitle(f"Observation Operator (H Matrix)\n{filename}", fontsize=14)
    
    # Mask exactly zero values so they appear blank (white) instead of skewing the colormap
    H_masked = np.ma.masked_where(H_unscaled == 0, H_unscaled)
    
    # Plot using a heatmap. aspect='auto' prevents it from squishing to a tiny square
    im_H = ax_H.imshow(H_masked, aspect='auto', cmap='viridis', interpolation='none', norm=LogNorm())
    
    ax_H.set_title(f"Matrix Shape: {H_unscaled.shape[0]} Rays × {H_unscaled.shape[1]} State Variables", fontsize=11)
    ax_H.set_xlabel("State Vector Index (Flattened Altitude + Geo)")
    ax_H.set_ylabel("Observation Index (RO Ray Number)")
    
    # Add a colorbar
    cbar_H = fig_H.colorbar(im_H, ax=ax_H)
    cbar_H.set_label('Ray Path Length within Grid Cell (Scaled)')
    
    plt.tight_layout()
    plt.show()
    # ---------------------------------------------------------
    # 4. Run Assimilation
    print("\n  -> Running Data Assimilation...")
    posterior_state_flat = inverter.assimilate(
        obs=measured_tec_clean, 
        podTc2_data=None, # Passed None because we pre-calculated H below
        obs_operator=H_unscaled, 
        relaxation=0.95, 
        measurement_err=10.0 # Assumes ~1 TECU variance in measurement noise
    )

    # 5. Evaluate Results (Prior vs Posterior)
    print("  -> Calculating TEC and Reshaping States...")
    # Get Prior Mean State and Posterior State
    prior_state_flat = inverter.attrs['initial_edps_mean']
    
    # Strip masked array metadata to prevent NumPy mask broadcasting bugs
    H_clean = np.asarray(H_unscaled)
    prior_state_clean = np.asarray(prior_state_flat)
    posterior_state_clean = np.asarray(posterior_state_flat)

    # Calculate TEC: Z = H * x
    # NOTE: .flatten() is added here to convert (N, 1) column vectors to (N,) 1D arrays for matplotlib
    prior_tec = (H_clean @ prior_state_clean).flatten()
    posterior_tec = (H_clean @ posterior_state_clean).flatten()
    
    print("\n  --- DEBUG: TEC Arrays ---")
    print(f"  Y-Axis (tangent_alt_clean) shape: {tangent_alt_clean.shape}")
    print(f"  prior_tec shape: {prior_tec.shape} | NaNs: {np.isnan(prior_tec).sum()} | Min: {np.nanmin(prior_tec):.2e} | Max: {np.nanmax(prior_tec):.2e}")
    print(f"  posterior_tec shape: {posterior_tec.shape} | NaNs: {np.isnan(posterior_tec).sum()} | Min: {np.nanmin(posterior_tec):.2e} | Max: {np.nanmax(posterior_tec):.2e}")
    
    # --- NEW: Calculate Forward Modeled TEC directly via the integration method ---
    print("  -> Running standard Forward TEC Integration for verification...")
    # Compute forward model TEC for the base state (sample 0) using the full uncleaned data
    forward_tec_full = eds_occ.forward_model_mesh_tec(podTc_data, sample_idx=0, num_segments=1000)
    
    # Filter it down to the same valid rays used in the Kalman filter for a direct 1:1 plot match
    forward_tec_clean = forward_tec_full[valid_mask]
    
    # Reshape the 1D state vectors back into (n_height, n_geo) for plotting the vertical profiles
    n_height = len(alt_grid)
    n_geo = eds_occ.geolocation.shape[0]
    
    prior_edp_3d = prior_state_flat.reshape(n_height, n_geo)
    posterior_edp_3d = posterior_state_flat.reshape(n_height, n_geo)
    
    # Extract the center vertex to plot a representative vertical profile
    center_idx = n_geo // 2 
    prior_profile = prior_edp_3d[:, center_idx]
    posterior_profile = posterior_edp_3d[:, center_idx]
    
    print("\n  --- DEBUG: EDP Profiles ---")
    print(f"  Y-Axis (alt_grid) shape: {alt_grid.shape}")
    print(f"  prior_profile shape: {prior_profile.shape} | NaNs: {np.isnan(prior_profile).sum()} | Min: {np.nanmin(prior_profile):.2e} | Max: {np.nanmax(prior_profile):.2e}")
    print(f"  posterior_profile shape: {posterior_profile.shape} | NaNs: {np.isnan(posterior_profile).sum()} | Min: {np.nanmin(posterior_profile):.2e} | Max: {np.nanmax(posterior_profile):.2e}")
    
    # Check for Log-Scale Violations
    if np.nanmin(posterior_profile) <= 0:
        print("  [!] WARNING: Posterior profile contains negative or zero values. Matplotlib's log scale will hide these points!")

    # ---------------------------------------------------------
    # 6. Generate the Assessment Plot
    # ---------------------------------------------------------
    print("  -> Plotting Assimilation Results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle(f"§17 Tomography Assimilation Results\n{filename}", fontsize=14)

    # --- Panel 1: Observation Space (TEC) ---
    ax1 = axes[0]
    ax1.plot(measured_tec_clean, tangent_alt_clean, color='black', lw=3, label="Measured TEC")
    
    # Plot both the Matrix Multiplied Prior and the Forward Integrated Prior
    ax1.plot(prior_tec, tangent_alt_clean, color='tab:red', lw=3, ls='--', label="Prior (H Matrix)")
    ax1.plot(forward_tec_clean, tangent_alt_clean, color='tab:green', lw=2, ls=':', label="Prior (Forward Model)")
    
    ax1.plot(posterior_tec, tangent_alt_clean, color='tab:blue', lw=2, label="Posterior (Assimilated)")
    
    ax1.set_ylabel("Tangent Altitude (km)")
    ax1.set_xlabel("Total Electron Content (TECU)")
    ax1.set_title("Observation Space: TEC Adjustment")
    ax1.grid(True, alpha=0.4, linestyle=':')
    ax1.legend(loc='upper right')
    
    if len(tangent_alt_clean) > 0:
        ax1.set_ylim(0, min(np.max(tangent_alt_clean) + 50, alt_grid[-1]))

    # --- Panel 2: State Space (Electron Density) ---
    ax2 = axes[1]
    ax2.plot(prior_profile, alt_grid, color='tab:red', lw=2, ls='--', label="Prior Mean Density")
    ax2.plot(posterior_profile, alt_grid, color='tab:blue', lw=2, label="Posterior Density")
    
    # Highlight the difference
    ax2.fill_betweenx(alt_grid, prior_profile, posterior_profile, color='tab:blue', alpha=0.15, label="Assimilation Delta")

    ax2.set_xlabel("Electron Density (m⁻³)")
    ax2.set_title("State Space: 3D Mesh Adjustment (Center Vertex)")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.4, linestyle=':')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    
    # Save output
    save_dir = "./Figures/Section17_Assimilation/"
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}_assimilation.png")
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved figure to {save_path}\n")

    return prior_state_flat, posterior_state_flat
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def section18(podTc2_file: str, alt_grid: np.ndarray, sampling_df: pd.DataFrame):
    """
    §18 Tomography Data Assimilation Visual Assessment
    Assimilates measured TEC data and produces three sets of plots:
    1. TEC Fit & Linear Center EDP
    2. Vertical Profile Spread (All mesh points)
    3. 3x3 Globe Grid (Altitudes vs. Prior/Posterior/Delta)
    """
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
    from EDPSamples.edp_samples import EDPSamples
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    from matplotlib.ticker import ScalarFormatter
    
    try:
        _banner("§18  Tomography Data Assimilation Visual Assessment")
    except NameError:
        print("\n=== §18  Tomography Data Assimilation Visual Assessment ===")
        
    filename = podTc2_file.split('/')[-1]
    print(f"Processing File: {filename}")

    # 1. Parse Data and Clean NaNs
    podTc_data = parse_podTc2_nc_file(podTc2_file)
    if podTc_data is None:
        print(" [!] Invalid or skipped podTc data. Aborting.")
        return

    _, _, tangent_alt_raw = rayTangent(podTc_data['LEO'], podTc_data['GNSS'], units='km')
    tangent_alt_km = tangent_alt_raw * 1e-3
    measured_tec = podTc_data.get('TEC_podTc2', podTc_data.get('TEC', np.zeros_like(tangent_alt_km)))
    
    valid_mask = ~np.isnan(measured_tec) & (measured_tec > 0)
    measured_tec_clean = measured_tec[valid_mask]
    tangent_alt_clean = tangent_alt_km[valid_mask]
    
    podTc_clean = {
        'LEO': podTc_data['LEO'][:, valid_mask],
        'GNSS': podTc_data['GNSS'][:, valid_mask]
    }
    print(f"  -> Filtered {np.sum(~valid_mask)} invalid/NaN rays. {np.sum(valid_mask)} rays remaining.")

    # 2. Generate Prior Ensemble
    print("  -> Generating Prior State Ensemble...")
    base_sc = sampling_df.iloc[0]
    base_sc['hour'] = podTc_data['hour']
    n_mc = 50  
    np.random.seed(42)

    mc_hours = (base_sc['hour'] + np.random.uniform(-3, 3, size=n_mc)) % 24
    mc_df = pd.DataFrame({
        "hour": mc_hours,
        "f107": np.random.normal(loc=base_sc.get('f107', 130), scale=10, size=n_mc).clip(70, 250),
        "ap":   np.random.normal(loc=base_sc.get('ap', 15), scale=5, size=n_mc).clip(0, 400),
        "ig12": np.random.normal(loc=base_sc.get('ig12', 100), scale=10, size=n_mc).clip(50, 200),
        "rz12": np.random.normal(loc=base_sc.get('rz12', 100), scale=10, size=n_mc).clip(50, 200),
    })
    mc_df.iloc[0] = base_sc

    eds_occ = EDPSamples(
        DateTime="2015-06-01 12:00:00", geo_type="Occultation", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1,
        filename=podTc2_file, dLat=5, dLon=5
    )

    # 3. Initialize Inverter
    print("  -> Initializing Kalman Filter...")
    # Make sure Ionosphere_Tomography_Inverter is imported locally or globally
    # from EDPSamples.tomography_inverter import Ionosphere_Tomography_Inverter 
    inverter = Ionosphere_Tomography_Inverter(EDPSam=eds_occ, meanscale=1)
    
    print("  -> Building Observation Operator (H)...")
    H_unscaled = inverter.get_observation_operator(podTc_clean)

    # 4. Run Assimilation
    print("\n  -> Running Data Assimilation...")
    posterior_state_flat = inverter.assimilate(
        obs=measured_tec_clean,
        podTc2_data=None,
        obs_operator=H_unscaled,
        relaxation=0.95,
        measurement_err=1.0
    )

    # 4b. RelTEC Assimilation (no topside approximation)
    print("  -> Running RelTEC Assimilation...")
    rel_inverter = Ionosphere_Tomography_Inverter_RelTEC(EDPSam=eds_occ, meanscale=1)
    H_rel = rel_inverter.get_observation_operator(podTc_clean)
    reltec_state_flat = rel_inverter.assimilate(
        obs=measured_tec_clean,
        obs_operator=H_rel,
        tangent_alt_km=tangent_alt_clean,
        relaxation=0.95,
        measurement_err=1.0,
    )

    # 5. Evaluate Results & Reshape
    print("  -> Calculating TEC and Reshaping States...")
    prior_state_flat = inverter.attrs['initial_edps_mean']
    
    H_clean = np.asarray(H_unscaled)
    prior_state_clean = np.asarray(prior_state_flat)
    posterior_state_clean = np.asarray(posterior_state_flat)

    prior_tec = (H_clean @ prior_state_clean).flatten()
    posterior_tec = (H_clean @ posterior_state_clean).flatten()
    reltec_tec = (np.asarray(H_rel) @ np.asarray(reltec_state_flat)).flatten()

    n_height = len(alt_grid)
    n_geo = eds_occ.geolocation.shape[0]

    prior_edp_3d = prior_state_flat.reshape(n_height, n_geo)
    prior_edp_3d[prior_edp_3d == 0] = np.nan

    posterior_edp_3d = posterior_state_flat.reshape(n_height, n_geo)
    posterior_edp_3d[posterior_edp_3d == 0] = np.nan

    reltec_edp_3d = np.asarray(reltec_state_flat).reshape(n_height, n_geo)
    reltec_edp_3d[reltec_edp_3d == 0] = np.nan

    center_idx = n_geo // 2
    prior_profile = prior_edp_3d[:, center_idx]
    posterior_profile = posterior_edp_3d[:, center_idx]
    reltec_profile = reltec_edp_3d[:, center_idx]

    save_dir = "./Figures/Section18_Visual_Assessments2/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate Abel Inversion:
    from Abel_Inverter import run_abel_inversion
    abel = run_abel_inversion(podTc_data)
    plt.plot(abel["Ne"],abel["alt_km"])

    # =========================================================
    # PLOT 1: Observation Space (TEC) & State Space (Linear EDP)
    # =========================================================
    print("  -> Generating Plot 1: TEC Fit & Center Profile...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig1.suptitle(f"§18 Tomography Assimilation Results\n{filename}", fontsize=14)

    # Panel 1: TEC
    ax1_1 = axes1[0]
    ax1_1.plot(measured_tec_clean, tangent_alt_clean, color='black', lw=3, label="Measured TEC")
    ax1_1.plot(prior_tec, tangent_alt_clean, color='tab:red', lw=2, ls='--', label="Prior (H Matrix)")
    ax1_1.plot(posterior_tec, tangent_alt_clean, color='tab:blue', lw=2, label="Posterior (Assimilated)")
    ax1_1.plot(reltec_tec, tangent_alt_clean, color='tab:purple', lw=2, ls='-.', label="RelTEC Posterior")
    ax1_1.set_ylabel("Tangent Altitude (km)")
    ax1_1.set_xlabel("Total Electron Content (TECU)")
    ax1_1.set_title("Observation Space: TEC Adjustment")
    ax1_1.grid(True, alpha=0.4, linestyle=':')
    ax1_1.legend(loc='upper right')
    if len(tangent_alt_clean) > 0:
        ax1_1.set_ylim(0, min(np.max(tangent_alt_clean) + 50, alt_grid[-1]))

    # Panel 2: EDP (Linear Scale)
    ax1_2 = axes1[1]
    ax1_2.plot(prior_profile, alt_grid, color='tab:red', lw=2, ls='--', label="Prior Mean Density")
    ax1_2.plot(posterior_profile, alt_grid, color='tab:blue', lw=2, label="Posterior Density")
    ax1_2.fill_betweenx(alt_grid, prior_profile, posterior_profile, color='tab:blue', alpha=0.15, label="Assimilation Delta")
    ax1_2.plot(reltec_profile, alt_grid, color='tab:purple', lw=2, ls='-.', label="RelTEC Posterior")
    if abel is not None:
        ax1_2.plot(abel['Ne'], abel['alt_km'],
                   color='tab:green', lw=2, ls=':', label="Abel Inversion (Lei)")
        ax1_2.legend(loc='upper right')  # re-draw legend with new entries
        
    ax1_2.set_xlabel("Electron Density (m⁻³)")
    ax1_2.set_title("State Space: Center Vertex (Linear Scale)")
    
    # Format X axis for scientific notation so linear values don't overlap
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax1_2.xaxis.set_major_formatter(formatter)
    
    # Optional: adjust xlim to ensure non-NaN data is clearly visible
    max_edp = max(np.nanmax(prior_profile), np.nanmax(posterior_profile), np.nanmax(reltec_profile[~np.isnan(reltec_profile)]))
    ax1_2.set_xlim(left=0, right=max_edp * 1.1)
    
    ax1_2.grid(True, alpha=0.4, linestyle=':')
    ax1_2.legend(loc='upper right')
    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, f"{filename}_plot1_center_linear.png"), dpi=150)

    # =========================================================
    # PLOT 2: Spaghetti Plot of ALL Vertical Profiles
    # =========================================================
    print("  -> Generating Plot 2: All Vertical Profiles...")
    fig2, ax2 = plt.subplots(figsize=(8, 10))
    fig2.suptitle(f"Vertical Profile Dispersion across Mesh\n{filename}", fontsize=14)

    # Plot all prior profiles (red, transparent)
    ax2.plot(prior_edp_3d, alt_grid, color='tab:red', alpha=0.1, lw=1)
    # Plot all posterior profiles (blue, transparent)
    ax2.plot(posterior_edp_3d, alt_grid, color='tab:blue', alpha=0.1, lw=1)
    # Plot all RelTEC profiles (purple, transparent)
    ax2.plot(reltec_edp_3d, alt_grid, color='tab:purple', alpha=0.1, lw=1)

    # Add strong lines for the center profile to act as a reference
    ax2.plot(prior_profile, alt_grid, color='darkred', lw=2, ls='--', label="Prior (Center Vertex)")
    ax2.plot(posterior_profile, alt_grid, color='darkblue', lw=2, label="Posterior (Center Vertex)")
    ax2.plot(reltec_profile, alt_grid, color='purple', lw=2, ls='-.', label="RelTEC (Center Vertex)")

    if abel is not None:
        print("Plotting Abel Inversion")
        ax2.plot(abel['Ne'], abel['alt_km'],
                 color='tab:green', lw=2.5, ls=':', label="Abel Inversion (Lei)")
        
    ax2.set_ylabel("Altitude (km)")
    ax2.set_xlabel("Electron Density (m⁻³)")
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlim(left=0, right=max(np.nanmax(prior_edp_3d), np.nanmax(posterior_edp_3d), np.nanmax(reltec_edp_3d[~np.isnan(reltec_edp_3d)])) * 1.1)
    ax2.set_ylim(0, alt_grid[-1])
    ax2.grid(True, alpha=0.4, linestyle=':')
    
    # Custom legend to explain the transparent lines
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='tab:red', lw=2, alpha=0.5),
        Line2D([0], [0], color='tab:blue', lw=2, alpha=0.5),
        Line2D([0], [0], color='tab:purple', lw=2, alpha=0.5),
        Line2D([0], [0], color='darkred', lw=2, ls='--'),
        Line2D([0], [0], color='darkblue', lw=2),
        Line2D([0], [0], color='purple', lw=2, ls='-.'),
    ]
    ax2.legend(custom_lines, ['Prior (All Vertices)', 'Posterior (All Vertices)', 'RelTEC (All Vertices)',
                               'Prior (Center)', 'Posterior (Center)', 'RelTEC (Center)'], loc='upper right')
    
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"{filename}_plot2_all_profiles.png"), dpi=150)

    # =========================================================
    # PLOT 3: 3x3 Globe Grid (Altitudes vs States)
    # =========================================================
    print("  -> Generating Plot 3: Globe Map...")
    
    target_alts = [200.0, 300.0, 400.0, 500.0, 600.0]
    
    # Attempt to extract center lat/lon for the Orthographic projection
    try:
        tecmax_lat = podTc_data['lat_tecmax_tangent']
        tecmax_lon = podTc_data['lon_tecmax_tangent']
    except KeyError:
        # Fallback if specific tecmax keys aren't in this data structure
        tecmax_lon = np.nanmean(eds_occ.geolocation[:, 0])
        tecmax_lat = np.nanmean(eds_occ.geolocation[:, 1])

    proj = ccrs.Orthographic(central_longitude=tecmax_lon, central_latitude=tecmax_lat)
    
    # Added squeeze=False and layout='constrained'
    fig3, axes3 = plt.subplots(
        len(target_alts), 3, 
        figsize=(15, len(target_alts) * 5 + 1), 
        subplot_kw={'projection': proj},
        squeeze=False,
        layout='constrained'
    )
    
    # Dropped the y=0.95 since layout='constrained' handles title spacing perfectly
    fig3.suptitle(f"Spatial Mesh Assimilation Mapping\n{filename}", fontsize=16)
    
    verts = eds_occ.geolocation
    tris = eds_occ.mesh
    
    # Pre-filter triangles whose vertices carry NaN coordinates.
    # triplot routes coordinates through shapely for geodetic projection;
    # NaN coordinates produce a RuntimeWarning there.
    _nan_vert = np.any(np.isnan(verts), axis=1)
    clean_tris = tris[~np.any(_nan_vert[tris], axis=1)]

    # --- NEW: Calculate global color bounds across ALL target altitudes ---
    # This ensures the colormap scale is identical for every row.
    alt_indices = [int(np.argmin(np.abs(alt_grid - alt))) for alt in target_alts]
    
    vmin_edp = 0
    vmax_edp = max(np.nanmax(prior_edp_3d[alt_indices, :]), np.nanmax(posterior_edp_3d[alt_indices, :]))
    max_delta = np.nanmax(np.abs(posterior_edp_3d[alt_indices, :] - prior_edp_3d[alt_indices, :]))
    # ----------------------------------------------------------------------

    for i, target_alt in enumerate(target_alts):
        # Find closest altitude index
        alt_idx = int(np.argmin(np.abs(alt_grid - target_alt)))
        actual_alt = alt_grid[alt_idx]
        
        # Extract data for this altitude slice
        prior_slice = prior_edp_3d[alt_idx, :]
        post_slice = posterior_edp_3d[alt_idx, :]
        
        delta_slice = post_slice - prior_slice

        # Column 0: Prior
        ax_prior = axes3[i, 0]
        ax_prior.set_global()
        ax_prior.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')
        tc1 = ax_prior.tripcolor(verts[:, 0], verts[:, 1], tris, prior_slice,
                                 transform=ccrs.Geodetic(), cmap='plasma', shading='flat', 
                                 edgecolors='face', vmin=vmin_edp, vmax=vmax_edp)
        if i == 0: ax_prior.set_title("Prior EDP", fontsize=14)
        ax_prior.text(-0.1, 0.5, f"{actual_alt:.0f} km", va='center', ha='center', 
                      rotation='vertical', fontsize=14, transform=ax_prior.transAxes, fontweight='bold')
                      
        # Column 1: Posterior
        ax_post = axes3[i, 1]
        ax_post.set_global()
        ax_post.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')
        tc2 = ax_post.tripcolor(verts[:, 0], verts[:, 1], tris, post_slice,
                                transform=ccrs.Geodetic(), cmap='plasma', shading='flat', 
                                edgecolors='face', vmin=vmin_edp, vmax=vmax_edp)
        if i == 0: ax_post.set_title("Posterior EDP", fontsize=14)
        
        # Column 2: Delta
        ax_delta = axes3[i, 2]
        ax_delta.set_global()
        ax_delta.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')
        tc3 = ax_delta.tripcolor(verts[:, 0], verts[:, 1], tris, delta_slice,
                                 transform=ccrs.Geodetic(), cmap='coolwarm', shading='flat', 
                                 edgecolors='face', vmin=-max_delta, vmax=max_delta)
        if i == 0: ax_delta.set_title("Delta (Post - Prior)", fontsize=14)
        
        # Plot faint wireframe overlay on all
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
            for ax in [ax_prior, ax_post, ax_delta]:
                ax.triplot(verts[:, 0], verts[:, 1], clean_tris, transform=ccrs.Geodetic(), color='black', linewidth=0.2, alpha=0.4)

    # --- NEW: Add a single set of global colorbars OUTSIDE the loop ---
    # Global Colorbar for Prior/Posterior (spans the first two columns)
    cbar1 = fig3.colorbar(tc2, ax=axes3[:, 0:2].ravel().tolist(), orientation='vertical', shrink=0.7, pad=0.02)
    cbar1.set_label("Electron Density (m⁻³)", fontsize=14)
    cbar1.formatter.set_powerlimits((-2, 2))

    # Global Colorbar for Delta (spans the third column)
    cbar2 = fig3.colorbar(tc3, ax=axes3[:, 2].ravel().tolist(), orientation='vertical', shrink=0.7, pad=0.02)
    cbar2.set_label("Δ Density (m⁻³)", fontsize=14)
    cbar2.formatter.set_powerlimits((-2, 2))
    # ------------------------------------------------------------------

    # Save the plot
    fig3.savefig(os.path.join(save_dir, f"{filename}_plot3_globe_grid.png"), dpi=150, bbox_inches='tight')
    
    # =========================================================
    # PLOT 4: Vertical Slice of EDP (Prior, Posterior, Delta)
    # =========================================================
    print("  -> Generating Plot 4: Vertical EDP Slice along Lowest Ray...")
    from scipy.interpolate import LinearNDInterpolator
    import pyproj

    # 1. Identify the lowest ray to define our 2D cross-section
    idx_lowest = np.argmin(tangent_alt_clean)
    leo_pt = podTc_clean['LEO'][:, idx_lowest:idx_lowest+1]
    gnss_pt = podTc_clean['GNSS'][:, idx_lowest:idx_lowest+1]

    # Calculate tangent point to use as the distance origin (0 km)
    tangent_pt, _, _ = EDPSamples.rayTangent(leo_pt, gnss_pt, units='km')

    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
        pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
        always_xy=True
    )

    # 2. Sample 300 points along the straight ECEF path between GNSS and LEO
    t = np.linspace(0, 1, 300)
    ray_ecef = gnss_pt + (leo_pt - gnss_pt) * t  # Shape: (3, 300)

    # Calculate distance along the ground relative to the tangent point
    tangent_dist = np.linalg.norm(ray_ecef - tangent_pt, axis=0)
    
    # Determine which side of the tangent point each point is on (GNSS side = negative)
    leo_dist = np.linalg.norm(leo_pt - tangent_pt)
    dist_to_leo = np.linalg.norm(ray_ecef - leo_pt, axis=0)
    signs = np.where(dist_to_leo < leo_dist, 1.0, -1.0)
    distances_km = tangent_dist * signs

    # Convert ray points to Lat/Lon for mesh interpolation
    ray_lons, ray_lats, _ = transformer.transform(ray_ecef[0]*1e3, ray_ecef[1]*1e3, ray_ecef[2]*1e3)

    # Project mesh and ray to a local metric space (AEQD) to avoid spherical distortion bugs
    tan_lon, tan_lat, _ = transformer.transform(tangent_pt[0]*1e3, tangent_pt[1]*1e3, tangent_pt[2]*1e3)
    proj_aeqd = pyproj.Proj(proj='aeqd', lat_0=tan_lat[0], lon_0=tan_lon[0], ellps='WGS84')
    
    mesh_x, mesh_y = proj_aeqd(eds_occ.geolocation[:, 0], eds_occ.geolocation[:, 1])
    ray_x, ray_y = proj_aeqd(ray_lons, ray_lats)

    # 3. Interpolate the 3D grid along this 2D slice
    prior_slice  = np.full((n_height, len(t)), np.nan)
    post_slice   = np.full((n_height, len(t)), np.nan)
    reltec_slice = np.full((n_height, len(t)), np.nan)

    mesh_points = np.column_stack((mesh_x, mesh_y))
    ray_points = np.column_stack((ray_x, ray_y))

    # Fast 2D interpolation height-by-height
    # (Yields NaNs when the ray leaves the bounds of your occultation mesh)
    for h in range(n_height):
        interp_prior  = LinearNDInterpolator(mesh_points, prior_edp_3d[h, :])
        interp_post   = LinearNDInterpolator(mesh_points, posterior_edp_3d[h, :])
        interp_reltec = LinearNDInterpolator(mesh_points, reltec_edp_3d[h, :])

        prior_slice[h, :]  = interp_prior(ray_points)
        post_slice[h, :]   = interp_post(ray_points)
        reltec_slice[h, :] = interp_reltec(ray_points)

    delta_slice = post_slice - prior_slice

    # 4. Plotting the 4x1 Vertical Slices
    fig4, axes4 = plt.subplots(4, 1, figsize=(10, 16), sharex=True, sharey=True, layout='constrained')
    fig4.suptitle(f"Vertical EDP Slice Along Lowest Occultation Ray\n{filename}", fontsize=16)

    vmin_edp = 0
    vmax_edp = max(np.nanmax(prior_edp_3d), np.nanmax(posterior_edp_3d), np.nanmax(reltec_edp_3d[~np.isnan(reltec_edp_3d)]))
    max_delta = np.nanmax(np.abs(posterior_edp_3d - prior_edp_3d))
    # ---------------------------------------------------------------

    X, Y = np.meshgrid(distances_km, alt_grid)

    # --- Subplot 1: Prior ---
    ax4_1 = axes4[0]
    pcm1 = ax4_1.pcolormesh(X, Y, prior_slice, cmap='plasma', shading='auto', vmin=vmin_edp, vmax=vmax_edp)
    ax4_1.set_title("Prior EDP Cross-Section", fontsize=14)
    ax4_1.set_ylabel("Altitude (km)")
    
    # --- Subplot 2: Posterior ---
    ax4_2 = axes4[1]
    pcm2 = ax4_2.pcolormesh(X, Y, post_slice, cmap='plasma', shading='auto', vmin=vmin_edp, vmax=vmax_edp)
    ax4_2.set_title("Posterior EDP Cross-Section", fontsize=14)
    ax4_2.set_ylabel("Altitude (km)")

    # --- Subplot 3: RelTEC Posterior ---
    ax4_3 = axes4[2]
    pcm3 = ax4_3.pcolormesh(X, Y, reltec_slice, cmap='plasma', shading='auto', vmin=vmin_edp, vmax=vmax_edp)
    ax4_3.set_title("RelTEC Posterior Cross-Section", fontsize=14)
    ax4_3.set_ylabel("Altitude (km)")

    # Shared Colorbar for subplots 1, 2, and 3
    cbar1 = fig4.colorbar(pcm3, ax=axes4[:3], orientation='vertical', fraction=0.03, pad=0.03)
    cbar1.set_label("Electron Density (m⁻³)", fontsize=14)
    cbar1.formatter.set_powerlimits((-2, 2))

    # --- Subplot 4: Delta (Posterior - Prior) ---
    ax4_4 = axes4[3]
    pcm4 = ax4_4.pcolormesh(X, Y, delta_slice, cmap='coolwarm', shading='auto', vmin=-max_delta, vmax=max_delta)
    ax4_4.set_title("Delta (Post - Prior)", fontsize=14)
    ax4_4.set_xlabel("Ground Distance from Tangent Point (km) [ GNSS ←  0  → LEO ]")
    ax4_4.set_ylabel("Altitude (km)")

    # Mark the tangent point (distance = 0)
    for ax in axes4:
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Tangent Pt' if ax == axes4[0] else "")
        ax.grid(True, alpha=0.3, linestyle=':')
    axes4[0].legend(loc='upper right')

    # Colorbar for Delta
    cbar2 = fig4.colorbar(pcm4, ax=ax4_4, orientation='vertical', fraction=0.03, pad=0.03)
    cbar2.set_label("Δ Density (m⁻³)", fontsize=14)
    cbar2.formatter.set_powerlimits((-2, 2))

    # Zoom X-axis perfectly to the valid mesh boundaries
    valid_dist_mask = ~np.isnan(prior_slice).all(axis=0)
    if np.any(valid_dist_mask):
        min_dist = distances_km[valid_dist_mask].min()
        max_dist = distances_km[valid_dist_mask].max()
        pad = (max_dist - min_dist) * 0.05
        ax4_4.set_xlim(min_dist - pad, max_dist + pad)

    # Clip Y-axis to the actual alt_grid limits
    ax4_4.set_ylim(alt_grid[0], alt_grid[-1])


    fig4.savefig(os.path.join(save_dir, f"{filename}_plot4_vertical_slice.png"), dpi=150)
    print(f"  -> Saved all figures to {save_dir}\n")
    
    return prior_state_flat, posterior_state_flat

def section19(podTc2_file: str, alt_grid: np.ndarray, sampling_df: pd.DataFrame):
    

    """
    §18 Tomography Data Assimilation Visual Assessment
    Assimilates measured TEC data and produces three sets of plots:
    1. TEC Fit & Linear Center EDP
    2. Vertical Profile Spread (All mesh points)
    3. 3x3 Globe Grid (Altitudes vs. Prior/Posterior/Delta)
    """
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
    from EDPSamples.edp_samples import EDPSamples
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    from matplotlib.ticker import ScalarFormatter
    
    try:
        _banner("§18  Tomography Data Assimilation Visual Assessment")
    except NameError:
        print("\n=== §18  Tomography Data Assimilation Visual Assessment ===")
        
    filename = podTc2_file.split('/')[-1]
    print(f"Processing File: {filename}")

    # 1. Parse Data and Clean NaNs
    podTc_data = parse_podTc2_nc_file(podTc2_file)
    if podTc_data is None:
        print(" [!] Invalid or skipped podTc data. Aborting.")
        return
    
    from Abel_Inverter import run_abel_inversion
    abel = run_abel_inversion(podTc_data)
    
def extract_robust_f2_peak(profile: np.ndarray, alt_grid: np.ndarray, min_alt: float = 150.0, max_alt: float = 650.0):
    """
    Robustly finds NmF2 and hmF2 by restricting the search to physical 
    F-region altitudes and applying a 3-point parabolic fit for sub-grid resolution.
    """
    # 1. Restrict search to physical F-region bounds
    search_mask = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    
    if not np.any(search_mask):
        return np.nan, np.nan
        
    # Extract the restricted window
    search_alts = alt_grid[search_mask]
    search_prof = profile[search_mask]
    
    # 2. Find the discrete maximum within the F-region window
    local_max_idx = np.argmax(search_prof)
    discrete_hmF2 = search_alts[local_max_idx]
    discrete_NmF2 = search_prof[local_max_idx]
    
    # 3. Sub-grid refinement (3-point parabolic interpolation)
    # Only apply if the peak has neighboring points within our search window
    if 0 < local_max_idx < len(search_prof) - 1:
        h1, h2, h3 = search_alts[local_max_idx-1 : local_max_idx+2]
        n1, n2, n3 = search_prof[local_max_idx-1 : local_max_idx+2]
        
        # Fit a parabola: Ne(h) = a*h^2 + b*h + c
        try:
            coeffs = np.polyfit([h1, h2, h3], [n1, n2, n3], deg=2)
            a, b, c = coeffs
            
            # Ensure the parabola opens downwards (a < 0) indicating a true peak
            if a < 0:
                # The peak of a parabola is at h = -b / (2a)
                refined_hmF2 = -b / (2.0 * a)
                refined_NmF2 = (a * refined_hmF2**2) + (b * refined_hmF2) + c
                
                # Safety check: Ensure the interpolated peak didn't swing wildly out of bounds
                if abs(refined_hmF2 - discrete_hmF2) <= (h3 - h1):
                    return refined_NmF2, refined_hmF2
        except np.linalg.LinAlgError:
            pass # If the fit fails, quietly fall back to the discrete maximum
            
    return discrete_NmF2, discrete_hmF2
    
def section20(podTc2_file: str, alt_grid: np.ndarray, sampling_df: pd.DataFrame, 
              generate_plots: bool = True, save_dir: str = "./Figures/Section20_Batch/") -> dict:
    """
    §20 Tomography vs Abel Statistical Analysis (With Automated Plotting)
    Runs Data Assimilation and Abel inversions, calculates comparative statistics,
    and generates a 3-panel summary plot (TEC Fits, hmF2 Delta Map, EDP Dispersion).
    """
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
    from EDPSamples.edp_samples import EDPSamples
    from Abel_Inverter import run_abel_inversion
    import numpy as np
    import pandas as pd
    import gc
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    from scipy.interpolate import interp1d
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    filename = podTc2_file.split('/')[-1]
    print(f"--- Processing: {filename} ---")

    stats = {
        "File": filename,
        "Valid_Rays": 0,
        "Prior_NmF2": np.nan, "Prior_hmF2": np.nan,
        "Post_NmF2": np.nan,  "Post_hmF2": np.nan,
        "Abel_NmF2": np.nan,  "Abel_hmF2": np.nan,
        "Post_Abel_RMSE": np.nan,
        "Prior_TEC_RMSE": np.nan, "Prior_TEC_MAE": np.nan,
        "Post_TEC_RMSE": np.nan,  "Post_TEC_MAE": np.nan,
        "Status": "Failed"
    }

    try:
        # 1. Parse Data
        podTc_data = parse_podTc2_nc_file(podTc2_file)
        if podTc_data is None:
            return stats
        
        _, _, tangent_alt_raw = rayTangent(podTc_data['LEO'], podTc_data['GNSS'], units='km')
        tangent_alt_km = tangent_alt_raw * 1e-3
        measured_tec = podTc_data.get('TEC_podTc2', podTc_data.get('TEC', np.zeros_like(tangent_alt_km)))
        
        valid_mask = ~np.isnan(measured_tec) & (measured_tec > 0)
        measured_tec_clean = np.asarray(measured_tec[valid_mask], dtype=np.float64).flatten()
        tangent_alt_clean = tangent_alt_km[valid_mask].flatten()
        podTc_clean = {
            'LEO': podTc_data['LEO'][:, valid_mask],
            'GNSS': podTc_data['GNSS'][:, valid_mask]
        }
        stats["Valid_Rays"] = int(np.sum(valid_mask))
        
        if stats["Valid_Rays"] < 50:
            stats["Status"] = "Insufficient Rays"
            return stats

        # 2. Prior Generation
        base_sc = sampling_df.iloc[0]
        n_mc = 50  
        np.random.seed(42)
        mc_hours = (base_sc['hour'] + np.random.uniform(-3, 3, size=n_mc)) % 24
        mc_df = pd.DataFrame({
            "hour": mc_hours,
            "f107": np.random.normal(loc=base_sc.get('f107', 130), scale=10, size=n_mc).clip(70, 250),
            "ap":   np.random.normal(loc=base_sc.get('ap', 15), scale=5, size=n_mc).clip(0, 400),
            "ig12": np.random.normal(loc=base_sc.get('ig12', 100), scale=10, size=n_mc).clip(50, 200),
            "rz12": np.random.normal(loc=base_sc.get('rz12', 100), scale=10, size=n_mc).clip(50, 200),
        })
        mc_df.iloc[0] = base_sc

        eds_occ = EDPSamples(
            DateTime="2015-06-01 12:00:00", geo_type="Occultation", altitude=alt_grid,
            sampling_parameters=mc_df, evaluate_iri=1,
            filename=podTc2_file, dLat=5, dLon=5
        )

        # 3. Assimilation
        inverter = Ionosphere_Tomography_Inverter(EDPSam=eds_occ, meanscale=1)
        H_unscaled = inverter.get_observation_operator(podTc_clean)
        
        posterior_state_flat = inverter.assimilate(
            obs=measured_tec_clean, podTc2_data=None, obs_operator=H_unscaled, relaxation=0.95, measurement_err=1.0
        )
        
        prior_state_flat = inverter.attrs['initial_edps_mean']
        
        # Calculate TEC Residuals (Projects states back into TEC observation space)
        prior_tec = (np.asarray(H_unscaled) @ np.asarray(prior_state_flat)).flatten()
        post_tec = (np.asarray(H_unscaled) @ np.asarray(posterior_state_flat)).flatten()
        
        # Cast to pure float64 arrays to suppress NumPy 2.0 DeprecationWarnings
        prior_residuals = np.asarray(measured_tec_clean - prior_tec, dtype=np.float64)
        post_residuals = np.asarray(measured_tec_clean - post_tec, dtype=np.float64)
        
        stats["Prior_TEC_RMSE"] = np.sqrt(np.mean(prior_residuals**2))
        stats["Prior_TEC_MAE"] = np.mean(np.abs(prior_residuals))
        stats["Post_TEC_RMSE"] = np.sqrt(np.mean(post_residuals**2))
        stats["Post_TEC_MAE"] = np.mean(np.abs(post_residuals))

        # Reconstruct 3D Arrays for Mapping & Spaghetti Plots
        n_height = len(alt_grid)
        n_geo = eds_occ.geolocation.shape[0]
        center_idx = n_geo // 2 
        
        prior_edp_3d = prior_state_flat.reshape(n_height, n_geo)
        prior_edp_3d[prior_edp_3d == 0] = np.nan
        
        posterior_edp_3d = posterior_state_flat.reshape(n_height, n_geo)
        posterior_edp_3d[posterior_edp_3d == 0] = np.nan
        
        prior_profile = prior_edp_3d[:, center_idx]
        post_profile = posterior_edp_3d[:, center_idx]

        # 4. Extract Tomography F2 Peak Stats (Robust Method)
        prior_nm, prior_hm = extract_robust_f2_peak(prior_profile, alt_grid)
        stats["Prior_NmF2"] = prior_nm
        stats["Prior_hmF2"] = prior_hm
        
        post_nm, post_hm = extract_robust_f2_peak(post_profile, alt_grid)
        stats["Post_NmF2"] = post_nm
        stats["Post_hmF2"] = post_hm

        # 5. Abel Inversion & Comparison
        abel = run_abel_inversion(podTc_data)
        if abel is not None and len(abel['Ne']) > 0:
            abel_nm, abel_hm = extract_robust_f2_peak(abel['Ne'], abel['alt_km'])
            stats["Abel_NmF2"] = abel_nm
            stats["Abel_hmF2"] = abel_hm
            
            valid_abel = ~np.isnan(abel['Ne']) & ~np.isnan(abel['alt_km'])
            if np.sum(valid_abel) > 2:
                interp_func = interp1d(abel['alt_km'][valid_abel], abel['Ne'][valid_abel], 
                                       bounds_error=False, fill_value=np.nan)
                abel_on_grid = interp_func(alt_grid)
                
                valid_rmse = ~np.isnan(post_profile) & ~np.isnan(abel_on_grid)
                if np.any(valid_rmse):
                    mse = np.mean((post_profile[valid_rmse] - abel_on_grid[valid_rmse])**2)
                    stats["Post_Abel_RMSE"] = np.sqrt(mse)

        # =========================================================
        # 6. OPTIONAL BATCH PLOTTING (3 Subplots)
        # =========================================================
        if generate_plots:
            os.makedirs(save_dir, exist_ok=True)
            
            # Setup Figure and Projection
            fig = plt.figure(figsize=(18, 6))
            fig.suptitle(f"Tomography Batch Audit: {filename}\nTEC RMSE: Prior {stats['Prior_TEC_RMSE']:.2f} -> Post {stats['Post_TEC_RMSE']:.2f} TECU", fontsize=15)
            
            # ---------------------------------------------------------
            # Subplot 1: TEC Fits
            # ---------------------------------------------------------
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(measured_tec_clean, tangent_alt_clean, color='black', lw=3, label="Measured TEC")
            ax1.plot(prior_tec, tangent_alt_clean, color='tab:red', lw=2, ls='--', label="Prior TEC")
            ax1.plot(post_tec, tangent_alt_clean, color='tab:blue', lw=2, label="Posterior TEC")
            ax1.set_ylabel("Tangent Altitude (km)")
            ax1.set_xlabel("Total Electron Content (TECU)")
            ax1.set_title("Observation Space: TEC Adjustment")
            ax1.grid(True, alpha=0.4, linestyle=':')
            ax1.legend(loc='upper right')
            if len(tangent_alt_clean) > 0:
                ax1.set_ylim(0, min(np.max(tangent_alt_clean) + 50, alt_grid[-1]))

            # ---------------------------------------------------------
            # Subplot 2: Global Map of Deltas at Posterior hmF2
            # ---------------------------------------------------------
            # Calculate Map Center
            try:
                lon_center = podTc_data['lon_tecmax_tangent']
                lat_center = podTc_data['lat_tecmax_tangent']
            except KeyError:
                lon_center = np.nanmean(eds_occ.geolocation[:, 0])
                lat_center = np.nanmean(eds_occ.geolocation[:, 1])

            proj = ccrs.Orthographic(central_longitude=lon_center, central_latitude=lat_center)
            ax2 = fig.add_subplot(1, 3, 2, projection=proj)
            ax2.set_global()
            ax2.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')

            if not np.isnan(stats["Post_hmF2"]):
                # Find closest altitude slice to the Posterior hmF2
                alt_idx = int(np.argmin(np.abs(alt_grid - stats["Post_hmF2"])))
                actual_alt = alt_grid[alt_idx]
                delta_slice = posterior_edp_3d[alt_idx, :] - prior_edp_3d[alt_idx, :]
                
                verts = eds_occ.geolocation
                tris = eds_occ.mesh
                
                # Plot the Delta Map
                max_delta = np.nanmax(np.abs(delta_slice))
                tc = ax2.tripcolor(verts[:, 0], verts[:, 1], tris, delta_slice,
                                   transform=ccrs.Geodetic(), cmap='coolwarm', shading='flat',
                                   edgecolors='face', vmin=-max_delta, vmax=max_delta)
                
                cbar = fig.colorbar(tc, ax=ax2, orientation='horizontal', shrink=0.8, pad=0.05)
                cbar.set_label("Δ Density (m⁻³)")
                cbar.formatter.set_powerlimits((-2, 2))
                ax2.set_title(f"Δ Assimilation Map at hmF2 (~{actual_alt:.0f} km)")
            else:
                ax2.set_title("hmF2 Assimilation Map Unavailable")

            # ---------------------------------------------------------
            # Subplot 3: EDP Comparisons (Spaghetti Plot + F2 Peaks)
            # ---------------------------------------------------------
            ax3 = fig.add_subplot(1, 3, 3, sharey=ax1)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            
            # Spaghetti Lines
            ax3.plot(prior_edp_3d, alt_grid, color='tab:red', alpha=0.1, lw=1)
            ax3.plot(posterior_edp_3d, alt_grid, color='tab:blue', alpha=0.1, lw=1)
            
            # Center Profiles
            ax3.plot(prior_profile, alt_grid, color='darkred', lw=2, ls='--', label="Prior (Center)")
            ax3.plot(post_profile, alt_grid, color='darkblue', lw=2, label="Posterior (Center)")

            # --- NEW: Plot the extracted F2 Peaks ---
            if not np.isnan(stats["Prior_NmF2"]):
                ax3.plot(stats["Prior_NmF2"], stats["Prior_hmF2"], marker='o', markersize=8, 
                         color='darkred', markeredgecolor='black', zorder=5)
                         
            if not np.isnan(stats["Post_NmF2"]):
                ax3.plot(stats["Post_NmF2"], stats["Post_hmF2"], marker='o', markersize=8, 
                         color='darkblue', markeredgecolor='black', zorder=5)
            # ----------------------------------------

            # Legend Setup
            custom_lines = [
                Line2D([0], [0], color='tab:red', lw=2, alpha=0.5),
                Line2D([0], [0], color='tab:blue', lw=2, alpha=0.5),
                Line2D([0], [0], color='darkred', lw=2, ls='--'),
                Line2D([0], [0], color='darkblue', lw=2),
                # Add a legend entry for the peak markers
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8)
            ]
            legend_labels = ['Prior (All Vertices)', 'Posterior (All Vertices)', 'Prior (Center)', 'Posterior (Center)', 'F2 Peak']

            # Plot Abel Inversion if available
            if abel is not None and len(abel['Ne']) > 0:
                ax3.plot(abel['Ne'], abel['alt_km'], color='tab:green', lw=2.5, ls=':', label="Abel Inversion (Lei)")
                
                # Plot Abel Peak
                if not np.isnan(stats["Abel_NmF2"]):
                    ax3.plot(stats["Abel_NmF2"], stats["Abel_hmF2"], marker='o', markersize=8, 
                             color='tab:green', markeredgecolor='black', zorder=5)
                             
                custom_lines.insert(4, Line2D([0], [0], color='tab:green', lw=2.5, ls=':'))
                legend_labels.insert(4, 'Abel Inversion (Lei)')
                
            ax3.set_xlabel("Electron Density (m⁻³)")
            ax3.set_title("State Space: Vertical Profile Dispersion")
            ax3.xaxis.set_major_formatter(formatter)
            
            # Set dynamic limits
            max_edp = max(np.nanmax(prior_edp_3d), np.nanmax(posterior_edp_3d))
            if abel is not None and len(abel['Ne']) > 0:
                max_edp = max(max_edp, np.nanmax(abel['Ne']))
                
            ax3.set_xlim(left=0, right=max_edp * 1.1)
            ax3.grid(True, alpha=0.4, linestyle=':')
            ax3.legend(custom_lines, legend_labels, loc='upper right')
            
            # Final Layout & Save
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{filename}_summary.png"), dpi=100)
            plt.close(fig)

        stats["Status"] = "Success"

    except Exception as e:
        print(f" [!] Error processing {filename}: {e}")
        stats["Status"] = f"Error: {str(e)}"
        
    finally:
        plt.close('all')
        
        local_vars = ['podTc_data', 'eds_occ', 'inverter', 'H_unscaled', 
                      'prior_state_flat', 'posterior_state_flat', 'prior_tec', 'post_tec']
        for var in local_vars:
            if var in locals():
                del locals()[var]
        gc.collect()

    return stats

def _process_hourly_mesh(args: tuple) -> tuple:
    """
    Parallel worker function to process a single hour's Global EDP generation.
    Must remain at the top level of the module for multiprocessing to pickle it.
    """
    import os
    import gc
    import warnings
    import numpy as np
    import pandas as pd
    from EDPSamples.edp_samples import EDPSamples
    
    hr, alt_grid, base_sc, n_mc, dLat, dLon, data_dir = args
    nc_filename = os.path.join(data_dir, f"Global_EDPS_{dLat}_Hour_{hr:02d}.nc")
    n_height = len(alt_grid)
    
    # --- ROBUST Checkpoint Loading ---
    eds_global = None
    if os.path.exists(nc_filename):
        try:
            eds_global = EDPSamples.fromNetCDF(nc_filename)
        except Exception:
            eds_global = None # Force regeneration if corrupted
            
    if eds_global is None:
        np.random.seed(42 + hr) 
        mc_df = pd.DataFrame({
            "hour": np.full(n_mc, hr),
            "f107": np.random.normal(loc=base_sc.get('f107', 130), scale=10, size=n_mc).clip(70, 250),
            "ap":   np.random.normal(loc=base_sc.get('ap', 15), scale=5, size=n_mc).clip(0, 400),
            "ig12": np.random.normal(loc=base_sc.get('ig12', 100), scale=10, size=n_mc).clip(50, 200),
            "rz12": np.random.normal(loc=base_sc.get('rz12', 100), scale=10, size=n_mc).clip(50, 200),
        })
        
        time_str = f"2015-06-01 {hr:02d}:00:00"
        eds_global = EDPSamples(
            DateTime=time_str, 
            geo_type="Global", 
            altitude=alt_grid,
            sampling_parameters=mc_df, 
            evaluate_iri=1,
            equal_spaced=False, 
            dLat=dLat, 
            dLon=dLon
        )
        eds_global.saveNetCDF(nc_filename)

    # ==========================================
    # Calculate Covariance for this hour (Masked Array Approach)
    # ==========================================
    edps_3d = eds_global.edps
    
    # 1. Safely calculate mean, ignoring warnings if an altitude is 100% NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_edps = np.nanmean(edps_3d, axis=2, keepdims=True)
    
    # 2. Subtract to get anomalies
    anomalies = edps_3d - mean_edps 
    anomalies_flat = anomalies.reshape(n_height, -1) 
    
    # 3. CRITICAL: Use Masked Arrays to hide NaNs element-by-element
    masked_anomalies = np.ma.masked_invalid(anomalies_flat)
    
    # 4. Calculate covariance (np.ma.cov handles the mask automatically!)
    cov_matrix_ma = np.ma.cov(masked_anomalies)
    
    # 5. Convert back to a standard numpy array
    cov_matrix = cov_matrix_ma.filled(np.nan)
        
    # --- Extracted Variance at 300 km ---
    idx_300 = int(np.argmin(np.abs(alt_grid - 300.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        var_300km = np.nanvar(edps_3d[idx_300, :, :], axis=1) 
    # ==========================================
    
    # Aggressive cleanup
    del eds_global, edps_3d, mean_edps, anomalies, anomalies_flat
    gc.collect()
    
    return hr, nc_filename, cov_matrix, var_300km

def section21(alt_grid: np.ndarray, sampling_df: pd.DataFrame, 
              dLat: float = 10.0, dLon: float = 15.0, num_workers: int = 6) -> dict:
    """
    §21 Universal Global Covariance Generator (Parallelized)
    Generates a global mesh for all 24 hours using parallel processing. 
    Saves each ensemble to NetCDF, calculates vertical covariances, 
    and outputs a Universal Covariance Matrix and 300km Variance map.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    import os
    import warnings
    import multiprocessing as mp
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    
    try:
        _banner("§21  Universal Global Covariance Generator")
    except NameError:
        print("\n=== §21  Universal Global Covariance Generator ===")
        
    data_dir = "./Data/Section21_Global_EDPS/"
    save_dir = "./Figures/Section21_Universal_Covariance/"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    n_mc = 50 
    base_sc = sampling_df.iloc[0]
    
    hourly_nc_paths = {}
    hourly_covariances = {}
    hourly_var_300 = {}
    
    print(f" -> Generating Global Meshes ({dLat}° x {dLon}°) for 24 hours...")
    print(f"    Ensemble size: {n_mc} | CPU Workers: {num_workers}")

    tasks = [(hr, alt_grid, base_sc, n_mc, dLat, dLon, data_dir) for hr in range(24)]
    
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_hourly_mesh, tasks)):
            hr, nc_filename, cov_matrix, var_300 = result
            hourly_nc_paths[hr] = nc_filename
            hourly_covariances[hr] = cov_matrix
            hourly_var_300[hr] = var_300
            print(f"    [+] Finished Hour {hr:02d}:00  ({i+1}/24 completed)")

    print("\n -> All 24 hours processed successfully. Consolidating matrix...")

    sorted_covs = [hourly_covariances[h] for h in sorted(hourly_covariances.keys())]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        universal_cov = np.nanmean(sorted_covs, axis=0)
        
    np.save(os.path.join(save_dir, "Universal_Vertical_Covariance.npy"), universal_cov)
    
    sorted_vars = [hourly_var_300[h] for h in sorted(hourly_var_300.keys())]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        universal_var_300 = np.nanmean(sorted_vars, axis=0)

    # =========================================================
    # Plot 1: 24-Hour Covariance Grid
    # =========================================================
    print(" -> Generating 24-Panel Hourly Covariance Visualization...")
    fig1, axes1 = plt.subplots(4, 6, figsize=(20, 14), sharex=True, sharey=True)
    fig1.suptitle("Hourly Vertical EDP Covariance Matrices (Global Average)", fontsize=18)
    
    all_covs = np.array(sorted_covs)
    
    # Bulletproof vmax calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if np.all(np.isnan(all_covs)):
            vmax = 1e20
        else:
            vmax = np.nanmax(np.abs(all_covs))
            
    if np.isnan(vmax) or vmax == 0:
        vmax = 1e20
        
    linthresh = max(vmax * 1e-5, 1e-10) 
    alt_extent = [alt_grid[0], alt_grid[-1], alt_grid[0], alt_grid[-1]]
    
    for hr in range(24):
        ax = axes1[hr // 6, hr % 6]
        cov_hr = hourly_covariances[hr]
        
        pcm = ax.imshow(cov_hr, 
                        norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax), 
                        cmap='coolwarm', 
                        extent=alt_extent, 
                        origin='lower', 
                        aspect='auto')
                        
        ax.set_title(f"{hr:02d}:00 UT")
        if hr % 6 == 0:
            ax.set_ylabel("Altitude (km)")
        if hr >= 18:
            ax.set_xlabel("Altitude (km)")
            
    cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7])
    fig1.colorbar(pcm, cax=cbar_ax, label="Covariance (m⁻⁶)")
    
    fig1.savefig(os.path.join(save_dir, "Hourly_Covariances_24Panel.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # =========================================================
    # Plot 2: The Universal Covariance & Correlation Matrices
    # =========================================================
    print(" -> Generating Universal Matrix Summary Plot...")
    
    std_devs = np.sqrt(np.diag(universal_cov))
    outer_std = np.outer(std_devs, std_devs)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        universal_corr = universal_cov / np.where(outer_std == 0, 1e-10, outer_std)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle("Universal Global Vertical Error Statistics", fontsize=16)
    
    pcm1 = axes2[0].imshow(universal_cov, 
                           norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax), 
                           cmap='coolwarm', 
                           extent=alt_extent, 
                           origin='lower', 
                           aspect='auto')
                           
    axes2[0].set_title("Universal Covariance Matrix ($P$)")
    axes2[0].set_xlabel("Altitude (km)")
    axes2[0].set_ylabel("Altitude (km)")
    fig2.colorbar(pcm1, ax=axes2[0], label="Covariance (m⁻⁶)")
    
    pcm2 = axes2[1].imshow(universal_corr, 
                           cmap='coolwarm', 
                           vmin=-1, vmax=1, 
                           extent=alt_extent, 
                           origin='lower', 
                           aspect='auto')
                           
    axes2[1].set_title("Universal Correlation Matrix")
    axes2[1].set_xlabel("Altitude (km)")
    fig2.colorbar(pcm2, ax=axes2[1], label="Correlation Coefficient")
    
    fig2.savefig(os.path.join(save_dir, "Universal_Covariance_Summary.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # =========================================================
    # Plot 3: Global Variance Map at 300 km
    # =========================================================
    print(" -> Generating Global Variance Map at 300 km...")
    
    # Fast bypass read using xarray
    with xr.open_dataset(hourly_nc_paths[0]) as ds:
        verts = ds['geolocation'].values
        tris = ds['Mesh'].values
    
    fig3 = plt.figure(figsize=(12, 6))
    ax3 = fig3.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax3.set_global()
    ax3.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')

    tc = ax3.tripcolor(verts[:, 0], verts[:, 1], tris, universal_var_300,
                       transform=ccrs.Geodetic(), cmap='magma', shading='flat')

    cbar = fig3.colorbar(tc, ax=ax3, orientation='horizontal', shrink=0.7, pad=0.05)
    cbar.set_label("Daily Average Density Variance (m⁻⁶) at 300 km")
    cbar.formatter.set_powerlimits((-2, 2))
    ax3.set_title("Universal EDP Variance at 300 km (Monte Carlo Ensemble)")

    fig3.savefig(os.path.join(save_dir, "Universal_Variance_Map_300km.png"), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    import gc
    del verts, tris, universal_var_300
    gc.collect()

    return hourly_nc_paths

# ────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    import os
    import pandas as pd
    
    print("=" * 60)
    print("IonosphereTomography demo")
    print("Alaska region:  60–65 °N,  150–140 °W")
    print("=" * 60)

    # section1()
    # section2()
    # section3()
    # section4()
    # section5()
    # v_rect, t_rect = section6()
    # section7()
    # section8(v_rect, t_rect)
    # section9()
    # section10()
    # section11()

    # --- Define the file here so both section12 and section14 can use it ---
    # podTc2_string = "podTc2_GN05.2025.152.06.09.0026.C21.01_0000.0001_nc" #North polar
    # podTc2_string = "podTc2_GN05.2025.152.06.07.0026.C33.00_0000.0001_nc" #West coast pacific
    # podTc2_string = "podTc2_GN05.2025.152.06.07.0024.E08.01_0000.0001_nc" #Wide occultation polar
    # podTc2_string = "podTc2_GN05.2025.152.03.55.0027.E06.01_0000.0001_nc" #South America vertical occultation
    # podTc2_string = "podTc2_GN05.2025.152.03.53.0031.C39.01_0000.0001_nc" #South America wider occultation
    # podTc2_string = "podTc2_GN05.2025.152.03.52.0027.G24.01_0000.0001_nc" #Easter coast of South America
    # podTc2_string = 'podTc2_GN05.2025.152.02.51.0025.G10.01_0000.0001_nc' #North America vertical occultation
    # podTc2_file = f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.152/{podTc2_string}"

    # # Pass the file to section12
    # v_occ, t_occ, pt1, pt2, pt3 = section12(podTc2_file)
    
    # # Capture the generated EDPSamples dataset from section13
    # eds = section13(v_occ, t_occ, pt1, pt2, pt3)
    
    # # Call section14 using the dataset and the original file
    # tec = section14(eds, podTc2_file)
    
    # section15()
    

    # alt_grid = np.arange(60.0, 1000.0, 10.0, dtype=float)
    # sampling_df = pd.DataFrame([{
    #     "hour": 12.0, "f107": 150.0, "ap": 15.0, "ig12": 100.0, "rz12": 100.0
    # }])

    # podTc2_files = [
    #     "podTc2_GN05.2025.152.06.09.0026.C21.01_0000.0001_nc", # North polar
    #     # "podTc2_GN05.2025.152.06.07.0026.C33.00_0000.0001_nc", # West coast pacific
    #     # "podTc2_GN05.2025.152.06.07.0024.E08.01_0000.0001_nc", # Wide occultation polar
    #     # "podTc2_GN05.2025.152.03.55.0027.E06.01_0000.0001_nc", # South America vertical occultation
    #     # "podTc2_GN05.2025.152.03.53.0031.C39.01_0000.0001_nc", # South America wider occultation
    #     # "podTc2_GN05.2025.152.03.52.0027.G24.01_0000.0001_nc", # Eastern coast of South America
    #     # "podTc2_GN05.2025.152.02.51.0025.G10.01_0000.0001_nc"  # North America vertical occultation
    #     # "podTc2_GN04.2025.152.06.23.0026.G31.01_0000.0001_nc" # Very wide occultation
    #     # "podTc2_GN05.2025.152.06.04.0032.C36.00_0000.0001_nc"
    #     # "podTc2_GN04.2025.152.06.27.0042.C40.01_0000.0001_nc"
    #     # "podTc2_GN05.2025.152.03.01.0027.E03.01_0000.0001_nc"
    # ]

    # base_path = "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.152/"

    # for f_string in podTc2_files:
    #         full_path = os.path.join(base_path, f_string)
            
    #         # Run the geometry comparison suite
    #         # eds_dict, tec_results = section16(full_path, alt_grid, sampling_df)
            
    #         # Run the assimilation test
    #         prior_x, post_x = section18(full_path, alt_grid, sampling_df)
    #         # section19(full_path, alt_grid, sampling_df)

    # print("\n" + "=" * 60)
    # print("All sections complete — displaying figures.")
    # print("=" * 60)
    # # plt.tight_layout()
    # plt.show()
    

    
    # print("=" * 60)
    # print("IonosphereTomography Batch Processing")
    # print("=" * 60)

    alt_grid = np.arange(60.0, 1000.0, 10.0, dtype=float)
    sampling_df = pd.DataFrame([{
        "hour": 12.0, "f107": 150.0, "ap": 15.0, "ig12": 100.0, "rz12": 100.0
    }])

    # # Updated path to 2025.151
    # base_path = "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.151/"
    
    # # Dynamically find all .0001_nc files in the target directory
    # if not os.path.exists(base_path):
    #     print(f"Directory not found: {base_path}")
    #     return

    # podTc2_files = [f for f in os.listdir(base_path) if f.endswith(".0001_nc")]
    # podTc2_files.sort() # Ensure consistent ordering
    
    # print(f"Found {len(podTc2_files)} files to process in {base_path}\n")

    # # List to hold the statistical dictionaries
    # batch_statistics = []
    
    # # Set to None to run all files, or an integer (e.g., 30) to limit the run
    # MAX_FILES = 5 
    
    # files_to_process = podTc2_files[5:10] if MAX_FILES else podTc2_files
    # print(f"Processing {len(files_to_process)} out of {len(podTc2_files)} available files...\n")

    # for idx, f_string in enumerate(files_to_process):
    #     print(f"\n[{idx+1}/{len(podTc2_files)}] ", end="")
    #     full_path = os.path.join(base_path, f_string)
        
    #     # Run Section 20 and collect stats
    #     file_stats = section20(full_path, alt_grid, sampling_df)
    #     batch_statistics.append(file_stats)
        
    #     # Optional: Save a rolling backup in case it crashes midway
    #     if (idx + 1) % 10 == 0:
    #         temp_df = pd.DataFrame(batch_statistics)
    #         temp_df.to_csv("rolling_stats_backup.csv", index=False)

    # print("\n" + "=" * 60)
    # print("Batch processing complete. Compiling results...")
    # print("=" * 60)
    
    # # Convert list of dictionaries to a pandas DataFrame
    # stats_df = pd.DataFrame(batch_statistics)
    
    # # Save final results
    # stats_df.to_csv("Tomography_vs_Abel_Stats_2025_151.csv", index=False)
    
    # # Print a quick summary of the successful runs
    # success_df = stats_df[stats_df["Status"] == "Success"]
    # print(f"\nSuccessfully processed {len(success_df)} out of {len(stats_df)} files.")
    # if len(success_df) > 0:
    #     print("\n--- Summary Statistics ---")
    #     print("Mean hmF2 Comparison:")
    #     print(f"  Prior:     {success_df['Prior_hmF2'].mean():.1f} km")
    #     print(f"  Posterior: {success_df['Post_hmF2'].mean():.1f} km")
    #     print(f"  Abel:      {success_df['Abel_hmF2'].mean():.1f} km")
        
    #     print("\nMean TEC Residuals (RMSE):")
    #     print(f"  Prior:     {success_df['Prior_TEC_RMSE'].mean():.3f} TECU")
    #     print(f"  Posterior: {success_df['Post_TEC_RMSE'].mean():.3f} TECU")
        
    #     # Calculate the average percentage improvement
    #     improvement = ((success_df['Prior_TEC_RMSE'].mean() - success_df['Post_TEC_RMSE'].mean()) 
    #                    / success_df['Prior_TEC_RMSE'].mean()) * 100
    #     print(f"  Avg Filter Assimilation Improvement: {improvement:.1f}%")
        
        
    # Ensure alt_grid and sampling_df have been generated before this point
    # Run the universal covariance generator (using 6 workers to protect RAM)
    hourly_edp_arrays = section21(
        alt_grid=alt_grid, 
        sampling_df=sampling_df, 
        dLat=10.0, 
        dLon=5.0, 
        num_workers=12
    )

if __name__ == "__main__":
    main()
