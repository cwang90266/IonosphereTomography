#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11

# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:24:12 2026

@author: austinhunter

Ionosphere Tomography: Geometry Comparison Runner
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Local Imports
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from EDPSamples.edp_samples import EDPSamples
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import Ionosphere_Tomography_Inverter
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter_RelTEC import Ionosphere_Tomography_Inverter_RelTEC
from Abel_Inverter import run_abel_inversion
from iri2020.get_iri_inputs import read_apf107, read_ig_rz

def get_center_profile(edp_3d: np.ndarray, geolocation: np.ndarray, tgt_lat: float, tgt_lon: float) -> np.ndarray:
    """Helper to extract the vertical profile closest to the target coordinate."""
    if geolocation.shape[0] == 1:
        return edp_3d[:, 0]

    # Calculate squared Euclidean distance to find the closest vertex
    dists = (geolocation[:, 0] - tgt_lon)**2 + (geolocation[:, 1] - tgt_lat)**2
    idx = np.argmin(dists)
    return edp_3d[:, idx]


def _lookup_solar_indices(apf107: dict, ig_rz: dict, dt) -> dict:
    """Return solar/geomagnetic indices for the given datetime, with safe defaults."""
    result = {"f107": 130.0, "f107_81": 130.0, "ap": 15.0, "ig12": 80.0, "rz12": 60.0}
    try:
        yr, mn = dt.year, dt.month
        yr_arr = np.array(apf107['yr'])
        mn_arr = np.array(apf107['mn'])
        dy_arr = np.array(apf107['dy'])
        match = np.where((yr_arr == yr) & (mn_arr == mn) & (dy_arr == dt.day))[0]
        if match.size > 0:
            idx = match[0]
            result['f107']    = float(apf107['f107'][idx])
            result['f107_81'] = float(apf107['f107_81'][idx])
            result['ap']      = float(apf107['iapda'][idx])
    except Exception:
        pass
    try:
        se = ig_rz['Start_end_month']
        start_mn, start_yr = se[0], se[1]
        offset = (yr - start_yr) * 12 + (mn - start_mn)
        if 0 <= offset < len(ig_rz['ig']):
            result['ig12'] = float(ig_rz['ig'][offset])
            result['rz12'] = float(ig_rz['rz'][offset])
    except Exception:
        pass
    return result


def run_geometry_comparison(podTc2_file: str, alt_grid: np.ndarray, apf107: dict | None = None, ig_rz: dict | None = None, save_dir: str = "./Figures/Geometry_Comparison/"):
    """
    Assimilates measured TEC data using three distinct spatial geometries:
    1. Point (0D - strictly vertical)
    2. Regional (Rectangle or Polar, depending on latitude)
    3. Occultation (Optimized track-aligned mesh)

    Uses actual solar/geomagnetic indices from apf107 and ig_rz if provided;
    falls back to climatological defaults otherwise.

    Produces comparative plots to evaluate geometry influence on state estimation.
    """
    print(f"\n" + "=" * 60)
    print(f"Tomography Geometry Comparison")
    print("=" * 60)
    
    filename = os.path.basename(podTc2_file)
    os.makedirs(save_dir, exist_ok=True)
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
    print(f"  -> Filtered {np.sum(~valid_mask)} invalid rays. {np.sum(valid_mask)} rays remaining.")

    # 2. Extract Geometric Bounds dynamically
    print("  -> Calculating spatial boundaries...")
    pt1, pt2, pt3 = EDPSamples.get_occultation_extrema(podTc_clean['LEO'], podTc_clean['GNSS'], alt_limit=700.0)
    
    lats = [pt1[0], pt2[0], pt3[0]]
    lons = [pt1[1], pt2[1], pt3[1]]
    
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    min_lat_b, max_lat_b = min(lats) - 5, max(lats) + 5
    min_lon_b, max_lon_b = min(lons) - 5, max(lons) + 5

    # 3. Generate Prior Ensemble (Monte Carlo)
    print("  -> Generating Background Ensemble...")
    file_dt = podTc_data['date']
    datetime_str = file_dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"     DateTime: {datetime_str}")

    if apf107 is not None and ig_rz is not None:
        sol = _lookup_solar_indices(apf107, ig_rz, file_dt)
    else:
        sol = {"f107": 130.0, "f107_81": 130.0, "ap": 15.0, "ig12": 80.0, "rz12": 60.0}

    n_mc = 50
    np.random.seed(42)
    mc_df = pd.DataFrame({
        "hour": (podTc_data['hour'] + np.random.uniform(-3, 3, size=n_mc)) % 24,
        "f107": np.random.normal(loc=sol['f107'],    scale=10, size=n_mc).clip(70, 250),
        "ap":   np.random.normal(loc=sol['ap'],      scale=5,  size=n_mc).clip(0, 400),
        "ig12": np.random.normal(loc=sol['ig12'],    scale=10, size=n_mc).clip(50, 200),
        "rz12": np.random.normal(loc=sol['rz12'],    scale=10, size=n_mc).clip(50, 200),
    })
    mc_df.iloc[0] = {
        "hour": podTc_data['hour'], "f107": sol['f107'], "ap": sol['ap'],
        "ig12": sol['ig12'], "rz12": sol['rz12'],
    }

    # --- GEOMETRY 1: POINT ---
    print("\n  [Geometry 1] Processing Point...")
    eds_point = EDPSamples(
        DateTime=datetime_str, geo_type="Point", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1, Lon=center_lon, Lat=center_lat
    )
    inv_point = Ionosphere_Tomography_Inverter(EDPSam=eds_point, meanscale=1)
    H_point = inv_point.get_observation_operator(podTc_clean)
    post_state_point = inv_point.assimilate(obs=measured_tec_clean, podTc2_data=podTc_clean, obs_operator=H_point, relaxation=0.95, measurement_err=1.0)
    
    tec_prior = (H_point @ inv_point.attrs['initial_edps_mean']).flatten()
    tec_post_point = (H_point @ post_state_point).flatten()
    prof_prior = inv_point.attrs['initial_edps_mean'].reshape(len(alt_grid), -1)[:, 0]
    prof_post_point = post_state_point.reshape(len(alt_grid), -1)[:, 0]

    # --- GEOMETRY 2: REGIONAL (Rectangle or Polar) ---
    if abs(center_lat) > 65:
        print("\n  [Geometry 2] Processing Polar Mesh...")
        geo_type_reg = "Polar"
        bound_lat = 60.0 if center_lat > 0 else -60.0
        eds_reg = EDPSamples(
            DateTime=datetime_str, geo_type="Polar", altitude=alt_grid,
            sampling_parameters=mc_df, evaluate_iri=1, minLat=bound_lat, dLat=5.0
        )
    else:
        print("\n  [Geometry 2] Processing Rectangle Mesh...")
        geo_type_reg = "Rectangle"
        eds_reg = EDPSamples(
            DateTime=datetime_str, geo_type="Rectangle", altitude=alt_grid,
            sampling_parameters=mc_df, evaluate_iri=1, 
            minLon=min_lon_b, maxLon=max_lon_b, dLon=5.0, 
            minLat=min_lat_b, maxLat=max_lat_b, dLat=5.0
        )
        
    inv_reg = Ionosphere_Tomography_Inverter(EDPSam=eds_reg, meanscale=1)
    H_reg = inv_reg.get_observation_operator(podTc_clean)
    post_state_reg = inv_reg.assimilate(obs=measured_tec_clean, podTc2_data=podTc_clean, obs_operator=H_reg, relaxation=0.95, measurement_err=1.0)
    
    tec_post_reg = (H_reg @ post_state_reg).flatten()
    prof_post_reg = get_center_profile(post_state_reg.reshape(len(alt_grid), -1), eds_reg.geolocation, center_lat, center_lon)

    # --- GEOMETRY 3: OCCULTATION ---
    print("\n  [Geometry 3] Processing Occultation Mesh...")
    eds_occ = EDPSamples(
        DateTime=datetime_str, geo_type="Occultation", altitude=alt_grid,
        sampling_parameters=mc_df, evaluate_iri=1, pt1=pt1, pt2=pt2, pt3=pt3, dLat=5, dLon=5
    )
    inv_occ = Ionosphere_Tomography_Inverter(EDPSam=eds_occ, meanscale=1)
    H_occ = inv_occ.get_observation_operator(podTc_clean)
    post_state_occ = inv_occ.assimilate(obs=measured_tec_clean, podTc2_data=podTc_clean, obs_operator=H_occ, relaxation=0.95, measurement_err=1.0)

    tec_post_occ = (H_occ @ post_state_occ).flatten()
    prof_post_occ = get_center_profile(post_state_occ.reshape(len(alt_grid), -1), eds_occ.geolocation, center_lat, center_lon)

    # --- ABEL INVERSION (for comparison in fig1 only) ---
    print("\n  [Abel] Running Abel inversion...")
    try:
        abel = run_abel_inversion(podTc_data)
        if abel is None or len(abel.get('Ne', [])) == 0:
            abel = None
            print("     [!] Abel inversion returned empty result.")
        else:
            print(f"     Abel profile: {len(abel['Ne'])} points, max alt {abel['alt_km'].max():.0f} km")
    except Exception as e:
        abel = None
        print(f"     [!] Abel inversion failed: {e}")

    # =========================================================
    # PLOT 1: Observation Space and 1D Profile Comparison
    # =========================================================
    print("\n  -> Generating Line Plots...")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))

    fig1, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig1.suptitle(f"Tomography Geometry Comparison\n{filename}", fontsize=15, fontweight='bold')

    ax0 = axes[0]
    ax0.plot(measured_tec_clean, tangent_alt_clean, color='black', lw=3, label="Measured TEC")
    ax0.plot(tec_prior, tangent_alt_clean, color='gray', lw=2, ls='--', label="Prior TEC")
    ax0.plot(tec_post_point, tangent_alt_clean, color='tab:red', lw=2, label="Post (Point)")
    ax0.plot(tec_post_reg, tangent_alt_clean, color='tab:green', lw=2, label=f"Post ({geo_type_reg})")
    ax0.plot(tec_post_occ, tangent_alt_clean, color='tab:blue', lw=2, label="Post (Occultation)")
    if abel is not None:
        ax0.plot(abel['TEC_cal'],     abel['alt_km'], color='tab:orange', lw=2, ls=':', label="Abel TEC (Cal.)")
        ax0.plot(abel['TEC_forward'], abel['alt_km'], color='tab:orange', lw=2, ls='--', label="Abel TEC (Fwd.)")
    ax0.set_ylabel("Tangent Altitude (km)")
    ax0.set_xlabel("Total Electron Content (TECU)")
    ax0.set_title("Observation Space Fit")
    ax0.grid(True, alpha=0.4, linestyle=':')
    ax0.legend(loc='upper right')

    ax1 = axes[1]
    ax1.plot(prof_prior, alt_grid, color='gray', lw=2, ls='--', label="Prior Density")
    ax1.plot(prof_post_point, alt_grid, color='tab:red', lw=2, label="Post (Point)")
    ax1.plot(prof_post_reg, alt_grid, color='tab:green', lw=2, label=f"Post ({geo_type_reg})")
    ax1.plot(prof_post_occ, alt_grid, color='tab:blue', lw=2, label="Post (Occultation)")
    if abel is not None:
        ax1.plot(abel['Ne'], abel['alt_km'], color='tab:orange', lw=2, label="Abel Inversion")
    ax1.set_xlabel("Electron Density (m⁻³)")
    ax1.set_title(f"State Space: Profile at Occultation Center\nLat: {center_lat:.1f}°, Lon: {center_lon:.1f}°")
    ax1.xaxis.set_major_formatter(formatter)

    max_edp_vals = [np.nanmax(prof_prior), np.nanmax(prof_post_point), np.nanmax(prof_post_reg), np.nanmax(prof_post_occ)]
    if abel is not None:
        max_edp_vals.append(np.nanmax(abel['Ne']))
    max_edp = max(max_edp_vals)
    ax1.set_xlim(left=0, right=max_edp * 1.1)
    ax1.grid(True, alpha=0.4, linestyle=':')
    ax1.legend(loc='upper right')

    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, f"{filename}_plot1_geometry_lines.png"), dpi=150)
    
    # =========================================================
    # PLOT 2 & 3: Globe Grid Comparison (Absolute & Delta)
    # =========================================================
    print("  -> Generating Globe Grid Plots (Absolute & Delta)...")
    
    # Reverse order: 400km is plotted in row 0 (top), 100km in row 3 (bottom)
    target_alts = [400.0, 300.0, 200.0, 100.0] 
    
    # Reshape posterior states to 2D (alt, geo)
    edp_point = post_state_point.reshape(len(alt_grid), -1)
    edp_reg   = post_state_reg.reshape(len(alt_grid), -1)
    edp_occ   = post_state_occ.reshape(len(alt_grid), -1)

    # Calculate Delta (Posterior - Prior Climatology)
    delta_point = edp_point - inv_point.attrs['initial_edps_mean'].reshape(len(alt_grid), -1)
    delta_reg   = edp_reg - inv_reg.attrs['initial_edps_mean'].reshape(len(alt_grid), -1)
    delta_occ   = edp_occ - inv_occ.attrs['initial_edps_mean'].reshape(len(alt_grid), -1)

    alt_indices = [int(np.argmin(np.abs(alt_grid - alt))) for alt in target_alts]
    
    # Calculate global color bounds for Absolute EDP
    vmin_edp = 0
    vmax_edp = max(
        np.nanmax(edp_point[alt_indices, :]),
        np.nanmax(edp_reg[alt_indices, :]),
        np.nanmax(edp_occ[alt_indices, :])
    )

    # Calculate global symmetric color bounds for Delta EDP
    max_delta = max(
        np.nanmax(np.abs(delta_point[alt_indices, :])),
        np.nanmax(np.abs(delta_reg[alt_indices, :])),
        np.nanmax(np.abs(delta_occ[alt_indices, :]))
    )

    # Pre-filter triangles whose vertices carry NaN coordinates
    def get_clean_tris(verts, tris):
        if tris is None: return None
        _nan_vert = np.any(np.isnan(verts), axis=1)
        return tris[~np.any(_nan_vert[tris], axis=1)]

    clean_tris_reg = get_clean_tris(eds_reg.geolocation, eds_reg.mesh)
    clean_tris_occ = get_clean_tris(eds_occ.geolocation, eds_occ.mesh)
    
    # Helper to safely color the cartopy spine/outline across versions
    def outline_ax(ax, color):
        try:
            ax.spines['geo'].set_edgecolor(color)
            ax.spines['geo'].set_linewidth(3.5)
        except KeyError:
            ax.outline_patch.set_edgecolor(color)
            ax.outline_patch.set_linewidth(3.5)

    proj = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    
    # Setup both figures
    fig3, axes3 = plt.subplots(len(target_alts), 3, figsize=(15, len(target_alts) * 5 + 1), subplot_kw={'projection': proj}, squeeze=False, layout='constrained')
    fig3.suptitle(f"Spatial Mesh Geometry Comparison (Absolute Density)\n{filename}", fontsize=16)

    fig4, axes4 = plt.subplots(len(target_alts), 3, figsize=(15, len(target_alts) * 5 + 1), subplot_kw={'projection': proj}, squeeze=False, layout='constrained')
    fig4.suptitle(f"Spatial Mesh Geometry Comparison (Change vs Climatology)\n{filename}", fontsize=16)

    import warnings
    for i, target_alt in enumerate(target_alts):
        alt_idx = alt_indices[i]
        actual_alt = alt_grid[alt_idx]

        # Extract Absolute Slices
        slice_point = edp_point[alt_idx, :]
        slice_reg   = edp_reg[alt_idx, :]
        slice_occ   = edp_occ[alt_idx, :]

        # Extract Delta Slices
        d_slice_point = delta_point[alt_idx, :]
        d_slice_reg   = delta_reg[alt_idx, :]
        d_slice_occ   = delta_occ[alt_idx, :]

        # ==========================================
        # Column 0: Point Geometry (Both Figures)
        # ==========================================
        for ax, slc, cmap, vmin, vmax in zip([axes3[i, 0], axes4[i, 0]], 
                                             [slice_point, d_slice_point], 
                                             ['plasma', 'coolwarm'], 
                                             [vmin_edp, -max_delta], 
                                             [vmax_edp, max_delta]):
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')
            ax.scatter(eds_point.geolocation[:, 0], eds_point.geolocation[:, 1], c=slc, 
                       transform=ccrs.Geodetic(), cmap=cmap, s=250, vmin=vmin, vmax=vmax, edgecolors='black', zorder=5)
            
            outline_ax(ax, 'tab:red')
            if i == 0: 
                ax.set_title("Point Geometry", fontsize=15, color='tab:red', fontweight='bold')
                
            ax.text(-0.1, 0.5, f"{actual_alt:.0f} km", va='center', ha='center', 
                       rotation='vertical', fontsize=14, transform=ax.transAxes, fontweight='bold')

        # ==========================================
        # Column 1: Regional Geometry (Both Figures)
        # ==========================================
        for ax, slc, cmap, vmin, vmax in zip([axes3[i, 1], axes4[i, 1]], 
                                             [slice_reg, d_slice_reg], 
                                             ['plasma', 'coolwarm'], 
                                             [vmin_edp, -max_delta], 
                                             [vmax_edp, max_delta]):
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')
            tc_reg = ax.tripcolor(eds_reg.geolocation[:, 0], eds_reg.geolocation[:, 1], clean_tris_reg, slc,
                                  transform=ccrs.Geodetic(), cmap=cmap, shading='flat', edgecolors='face', vmin=vmin, vmax=vmax)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
                ax.triplot(eds_reg.geolocation[:, 0], eds_reg.geolocation[:, 1], clean_tris_reg, 
                           transform=ccrs.Geodetic(), color='black', linewidth=0.2, alpha=0.4)
                
            outline_ax(ax, 'tab:green')
            if i == 0: 
                ax.set_title(f"{geo_type_reg} Geometry", fontsize=15, color='tab:green', fontweight='bold')
                
            # Save handle for colorbars
            if cmap == 'plasma': tc_abs = tc_reg
            else: tc_del = tc_reg

        # ==========================================
        # Column 2: Occultation Geometry (Both Figures)
        # ==========================================
        for ax, slc, cmap, vmin, vmax in zip([axes3[i, 2], axes4[i, 2]], 
                                             [slice_occ, d_slice_occ], 
                                             ['plasma', 'coolwarm'], 
                                             [vmin_edp, -max_delta], 
                                             [vmax_edp, max_delta]):
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='gray')
            ax.tripcolor(eds_occ.geolocation[:, 0], eds_occ.geolocation[:, 1], clean_tris_occ, slc,
                         transform=ccrs.Geodetic(), cmap=cmap, shading='flat', edgecolors='face', vmin=vmin, vmax=vmax)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
                ax.triplot(eds_occ.geolocation[:, 0], eds_occ.geolocation[:, 1], clean_tris_occ, 
                           transform=ccrs.Geodetic(), color='black', linewidth=0.2, alpha=0.4)
                
            outline_ax(ax, 'tab:blue')
            if i == 0: 
                ax.set_title("Occultation Geometry", fontsize=15, color='tab:blue', fontweight='bold')

    # Colorbar for Figure 3 (Absolute)
    cbar3 = fig3.colorbar(tc_abs, ax=axes3.ravel().tolist(), orientation='vertical', shrink=0.7, pad=0.03)
    cbar3.set_label("Electron Density (m⁻³)", fontsize=14)
    cbar3.formatter.set_powerlimits((-2, 2))

    # Colorbar for Figure 4 (Delta)
    cbar4 = fig4.colorbar(tc_del, ax=axes4.ravel().tolist(), orientation='vertical', shrink=0.7, pad=0.03)
    cbar4.set_label("Δ Electron Density (m⁻³)", fontsize=14)
    cbar4.formatter.set_powerlimits((-2, 2))

    fig3.savefig(os.path.join(save_dir, f"{filename}_plot2_globe_absolute.png"), dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(save_dir, f"{filename}_plot3_globe_delta.png"), dpi=150, bbox_inches='tight')
    
    print(f"  -> Successfully generated geometry comparison in {save_dir}")
    return post_state_point, post_state_reg, post_state_occ

def main() -> None:
    print("=" * 60)
    print("Ionosphere Tomography Geometry Comparison")
    print("=" * 60)

    # 1. Define standard testing grids
    alt_grid = np.arange(60.0, 1000.0, 10.0, dtype=float)

    # Load solar index files once (apf107.dat and ig_rz.dat expected in script directory)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        apf107 = read_apf107(data_dir)
        ig_rz  = read_ig_rz(data_dir)
        print(f"  Solar indices loaded from {data_dir}")
    except FileNotFoundError as e:
        print(f"  [!] Solar index files not found ({e}); using climatological defaults.")
        apf107, ig_rz = None, None

    # 2. Files to process
    podTc2_files = [
        "podTc2_GN05.2025.152.06.09.0026.C21.01_0000.0001_nc", # North polar
        "podTc2_GN05.2025.152.06.07.0026.C33.00_0000.0001_nc", # West coast pacific
        "podTc2_GN05.2025.152.06.07.0024.E08.01_0000.0001_nc", # Wide occultation polar
        "podTc2_GN05.2025.152.03.55.0027.E06.01_0000.0001_nc", # South America vertical occultation
        "podTc2_GN05.2025.152.03.53.0031.C39.01_0000.0001_nc", # South America wider occultation
        "podTc2_GN05.2025.152.03.52.0027.G24.01_0000.0001_nc", # Eastern coast of South America
        "podTc2_GN05.2025.152.02.51.0025.G10.01_0000.0001_nc"  # North America vertical occultation
    ]

    base_path = "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.152/"

    # 3. Execution Loop
    for f_string in podTc2_files:
        full_path = os.path.join(base_path, f_string)
        
        # Check if file exists to prevent hard crashing during the loop
        if not os.path.exists(full_path):
            print(f"\n[!] File not found: {full_path}")
            continue
            
        # Run the geometry comparison suite
        post_x_point, post_x_reg, post_x_occ = run_geometry_comparison(full_path, alt_grid, apf107=apf107, ig_rz=ig_rz)

    print("\n" + "=" * 60)
    print("All comparison processing complete — displaying figures.")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()