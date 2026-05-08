import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4
import pyproj
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -------------------------------------------------------------------------
# Assumed External Imports (Adjust these to match your actual module names)
# -------------------------------------------------------------------------
# from your_utils import parse_podTc2_nc_file, check_colocation, ECEFtolla, rayTangent

# -------------------------------------------------------------------------
# Utility & Math Functions
# -------------------------------------------------------------------------
def interpolate_ray_path(leo_pos: np.ndarray, gnss_pos: np.ndarray, num_points: int = 300) -> tuple:
    """Interpolates linear points between LEO and GNSS satellite coordinates."""
    x_leo, y_leo, z_leo = leo_pos
    x_gnss, y_gnss, z_gnss = gnss_pos

    x_vals = np.linspace(x_leo, x_gnss, num_points)
    y_vals = np.linspace(y_leo, y_gnss, num_points)
    z_vals = np.linspace(z_leo, z_gnss, num_points)

    return x_vals, y_vals, z_vals

def ECEFtolla(ECEF):
    # ECEF can be shape (3,) for single point or (3, n) for multiple points
    if ECEF.ndim == 1:
        x, y, z = ECEF[0], ECEF[1], ECEF[2]
    else:
        x, y, z = ECEF[0, :], ECEF[1, :], ECEF[2, :]
    
    # Rest stays the same - pyproj handles arrays automatically
    transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True
        )
    lon_deg, lat_deg, alt = transformer.transform(1e3*x, 1e3*y, 1e3*z)
    
    return lat_deg, lon_deg, alt

def rayTangent(LEO, GNSS, units = 'km'):
    v = LEO - GNSS
    t_s = np.clip(-np.sum(v * GNSS, axis=0) / np.sum(v * v, axis=0), 0.0, 1.0)
    tangent_point = GNSS + v * t_s[np.newaxis, :]
    p = np.linalg.norm(tangent_point, axis=0)
    alt = p - 6371.0

   
     # Replace deprecated pyproj.transform with pyproj.Transformer
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
        pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
        always_xy=True
    )
    if units == 'km':
        _, _, alt = transformer.transform(
            1e3 * tangent_point[0, :],
            1e3 * tangent_point[1, :],
            1e3 * tangent_point[2, :]
        )
    elif units == 'm':
        _, _, alt = transformer.transform(
            tangent_point[0, :],
            tangent_point[1, :],
            tangent_point[2, :]
        )

    return tangent_point, p, alt

def parse_podTc2_nc_file(file_path):
    """Parses the NetCDF file, extracts variables, and performs optional QC cleanup."""
    
    clean_podTc2 = True  # Changed from 1 to True for Pythonic standard
    
    with netCDF4.Dataset(file_path, 'r') as nc:
        podTc2_data = {}

        # Variables
        podTc2_data['time'] = nc.variables['time'][:]
        podTc2_data['time'] = podTc2_data['time'] - podTc2_data['time'][0]
        podTc2_data['TEC_podTc2'] = nc.variables['TEC'][:]
        podTc2_data['elevation'] = nc.variables['elevation'][:]
        podTc2_data['mp_cal'] = nc.variables['mp_cal'][:]
        podTc2_data['S4_L1'] = nc.variables['S4_L1'][:]
        podTc2_data['S4_L2'] = nc.variables['S4_L2'][:]
        podTc2_data['caL1_SNR'] = nc.variables['caL1_SNR'][:]
        podTc2_data['pL2_SNR'] = nc.variables['pL2_SNR'][:]
        
        podTc2_data['x_LEO'] = nc.variables['x_LEO'][:]
        podTc2_data['x_GNSS'] = nc.variables['x_GPS'][:]
        podTc2_data['y_LEO'] = nc.variables['y_LEO'][:]
        podTc2_data['y_GNSS'] = nc.variables['y_GPS'][:]
        podTc2_data['z_LEO'] = nc.variables['z_LEO'][:]
        podTc2_data['z_GNSS'] = nc.variables['z_GPS'][:]
        
        # NOTE: If these are in meters, remember to divide by 1000.0 here 
        # depending on which version of rayTangent you are using!
        podTc2_data['LEO'] = np.array([podTc2_data['x_LEO'], podTc2_data['y_LEO'], podTc2_data['z_LEO']])
        podTc2_data['GNSS'] = np.array([podTc2_data['x_GNSS'], podTc2_data['y_GNSS'], podTc2_data['z_GNSS']])

        # Attributes
        attr_list = [
            'start_time', 'conid', 'prn_id', 'leo_id', 'obs1', 'obs2',
            'lat_tecmax_tangent', 'lon_tecmax_tangent', 'slta_tecmax_tangent',
            'leveling_err', 'leodcb', 'gpsdcb', 'year', 'month', 'day',
            'hour', 'minute', 'second'
        ]
        for attr in attr_list:
            podTc2_data[attr] = nc.getncattr(attr)

        # Date/time processing
        date_str = f"{podTc2_data['year']}-{podTc2_data['month']:02d}-{podTc2_data['day']:02d} {podTc2_data['hour']:02d}:{podTc2_data['minute']:02d}:{podTc2_data['second']:02d}"
        podTc2_data['date'] = pd.to_datetime(date_str)
        local_time_offset = podTc2_data['lon_tecmax_tangent'] / 15.0  # 15 deg per hour
        local_time = podTc2_data['date'] + pd.to_timedelta(local_time_offset, unit='h')
        podTc2_data['local_time_hms'] = local_time.strftime('%H:%M:%S')
        podTc2_data['DOY'] = podTc2_data['date'].dayofyear
        
        # -----------------------------------------
        # QC and Cleanup Block
        # -----------------------------------------
        if clean_podTc2:
            # 1. Break for invalid latitudes
            if np.abs(podTc2_data['lat_tecmax_tangent']) > 90:
                print(f"Invalid latitude ({podTc2_data['lat_tecmax_tangent']}) for TEC max tangent point. Skipping file.")
                return None

            # 2. Calculate tangent info
            tangent_point, p1, tangent_alt_raw = rayTangent(podTc2_data['LEO'], podTc2_data['GNSS'])
            tangent_alt_km = tangent_alt_raw * 1e-3  

            # [NEW] Calculate distance from the tangent point to the LEO satellite
            # This identifies when the ray path physically detaches from the LEO and starts setting.
            dist_to_leo = np.linalg.norm(tangent_point - podTc2_data['LEO'], axis=0)

            # 3. Basic quality check for altitude bounds
            if np.max(tangent_alt_km) < 400:
                print("Data not high enough. Skipping.")
                return None

            # 4. Mask arrays strictly to valid altitudes AND ensure the ray has detached from LEO
            # We use > 5.0 (km) as a tiny buffer to avoid floating point fuzziness when they are clamped.
            valid_alt_mask = (
                (tangent_alt_km > 0) & (dist_to_leo > 5.0) 
            )
            
            # Guard clause: Ensure we actually have data left after masking
            if not np.any(valid_alt_mask):
                print(f"No valid descending tangent points found. Skipping.")
                return None
            
            tec_masked = podTc2_data['TEC_podTc2'][valid_alt_mask]
            leo_masked = podTc2_data['LEO'][:, valid_alt_mask]
            gnss_masked = podTc2_data['GNSS'][:, valid_alt_mask]
            time_masked = podTc2_data['time'][valid_alt_mask]
            tangent_alt_masked = tangent_alt_km[valid_alt_mask]

            # 5. Determine occultation type (Rising vs Setting) and conditionally flip
            # We determine this using the unmasked array to see the true direction of the pass
            is_setting = tangent_alt_km[0] > tangent_alt_km[-1]
            
            if is_setting:
                # Reassign back into dictionary, flipped
                podTc2_data['TEC_podTc2'] = np.flip(tec_masked)
                podTc2_data['LEO'] = np.flip(leo_masked, axis=1) # Axis 1 for 2D arrays
                podTc2_data['GNSS'] = np.flip(gnss_masked, axis=1)
                podTc2_data['time'] = np.flip(time_masked)
                podTc2_data['tangent_alt_km'] = np.flip(tangent_alt_masked)
                podTc2_data['occ_type'] = 'setting'
                print(f"Data Flipped: {podTc2_data['conid']}{podTc2_data['prn_id']} is setting")
            else:
                # Reassign back into dictionary, straight pass
                podTc2_data['TEC_podTc2'] = tec_masked
                podTc2_data['LEO'] = leo_masked
                podTc2_data['GNSS'] = gnss_masked
                podTc2_data['time'] = time_masked
                podTc2_data['tangent_alt_km'] = tangent_alt_masked
                podTc2_data['occ_type'] = 'rising'
                print(f"{podTc2_data['conid']}{podTc2_data['prn_id']} is rising")

    return podTc2_data
#
#
#


def get_last_processed(progress_file='last_processed.txt'):
    """Get the most recently processed file"""
    if not os.path.exists(progress_file):
        return None
    
    try:
        with open(progress_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                return lines[1].strip()
    except Exception as e:
        print(f"Warning: Could not read progress file: {e}")
    
    return None

def get_processed_files_from_csv(csv_file='JUNE_TEC_altitude_data.csv'):
    """
    Read the CSV file and return a set of already-processed filenames.
    
    Args:
        csv_file: Path to the CSV file containing processed filenames
        
    Returns:
        set: A set of filenames that have already been processed
    """
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found. Starting fresh.")
        return set()
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get the first column (filenames)
        if len(df.columns) == 0:
            print(f"CSV file {csv_file} is empty.")
            return set()
        
        # Extract filenames from first column
        first_column = df.iloc[:, 0]
        
        # Convert to set for fast lookup, removing any NaN values
        processed_files = set(first_column.dropna().astype(str))
        
        print(f"Found {len(processed_files)} already-processed files in {csv_file}")
        return processed_files
        
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return set()

# -------------------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------------------
def plot_tec_profile(tec_data: np.ndarray, tangent_alt: np.ndarray, 
                     conid: str, prn_id: str, occ_type: str, save_path: str):
    """Generates and saves a TEC vs. Tangent Altitude profile plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tec_data, tangent_alt, 'b.-')
    ax.set_xlabel('TEC (TECU)')
    ax.set_ylabel('Tangent Altitude (km)')
    ax.set_title(f'TEC Profile - {conid}{prn_id} ({occ_type})')
    ax.grid()
    
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_ray_paths(ax, LEO, GNSS, tangent_LLA, ALTITUDE_LIMIT):
        # Select 20 time points for plotting
        num_points = 20
        for i in range(0, LEO.shape[1], max(1, LEO.shape[1] // num_points)):
            leo = LEO[:, i]
            gnss = GNSS[:, i]
            X, Y, Z = interpolate_ray_path(leo, gnss)
            # Convert ECEF coordinates to latitude, longitude, altitude
            lats, lons, alts = [], [], []
            for x, y, z in zip(X, Y, Z):
                lat, lon, alt = ECEFtolla(np.array([x, y, z]))
                if np.isnan(lat) or np.isnan(lon) or np.isnan(alt):
                    print("NANS")
                lats.append(lat)
                lons.append(lon)
                alts.append(alt)
            lats = np.array(lats)
            lons = np.array(lons)
            alts = np.array(alts)
            mask = (alts <= ALTITUDE_LIMIT*1e3)
            m = np.argmin(alts)
            if np.any(mask):
                ax.plot((lons[mask]), (lats[mask]), transform=ccrs.Geodetic(), linewidth=1, color=plt.cm.viridis(i / LEO.shape[1]))
            if np.min(alts) > ALTITUDE_LIMIT*1e3:
                return
        return    

def plot_globe_ray_paths(leo_data, gnss_data, tangent_lla, tecmax_lon: float, 
                         tecmax_lat: float, alt_limit: float, save_path: str):
    """Plots the Radio Occultation ray paths over an orthographic globe projection."""
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=tecmax_lon, 
                                               central_latitude=tecmax_lat))
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    

    plot_ray_paths(ax, leo_data, gnss_data, tangent_lla, alt_limit)

    # Convert LEO and GNSS ECEF coordinates to lat/lon for plotting
    lat_leo, lon_leo, _ = ECEFtolla(leo_data[:, 0])
    lat_gnss, lon_gnss, _ = ECEFtolla(gnss_data[:, 0])
    
    # Plot LEO and GNSS start points
    ax.plot(lon_leo, lat_leo, transform=ccrs.Geodetic(), color='g', marker='o', markersize=4)
    ax.plot(lon_gnss, lat_gnss, transform=ccrs.Geodetic(), color='r', marker='o', markersize=4)
    
    # Plot tangent points below the altitude limit
    mask = tangent_lla[2, :] < alt_limit * 1e3
    ax.plot(tangent_lla[1, mask], tangent_lla[0, mask], transform=ccrs.Geodetic(), color='m', linewidth=2)
    
    plt.title(f"Ray Paths Below {alt_limit} km Altitude")
    # Save the figure (change DPI and bbox_inches as needed)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -------------------------------------------------------------------------
# Main Processing Pipeline
# -------------------------------------------------------------------------
def process_podtc_retrieval(filepath_podTc: str, file: str, subfolder: str, generate_figs: bool = True):
    """
    Parses podTc2 netCDF files, calculates tangent point altitudes, filters 
    for rising/setting occultations, and optionally generates plots.
    """
    doy_dir = filepath_podTc.removesuffix(file)
    podTc2_data = parse_podTc2_nc_file(filepath_podTc)


    # Break for invalid latitudes
    if np.abs(podTc2_data['lat_tecmax_tangent']) > 90:
        print("Invalid latitude for TEC max tangent point. Skipping file.")
        return None

    # Calculate tangent info
    tangent_point, p1, tangent_alt_raw = rayTangent(podTc2_data['LEO'], podTc2_data['GNSS'])
    tangent_alt_km = tangent_alt_raw * 1e-3  

    # Basic quality check
    if np.max(tangent_alt_km) < 400:
        print("Data not high enough. Skipping.")
        return None

    # Determine occultation type (Rising vs Setting)
    is_setting = tangent_alt_km[0] > tangent_alt_km[-1]
    valid_alt_mask = tangent_alt_km > 0
    
    # Extract and conditionally flip arrays
    tec_input = podTc2_data['TEC_podTc2'][valid_alt_mask]
    leo_input = podTc2_data['LEO'][:, valid_alt_mask]
    gnss_input = podTc2_data['GNSS'][:, valid_alt_mask]
    time_input = podTc2_data['time'][valid_alt_mask]
    tangent_alt = tangent_alt_km[valid_alt_mask]
    
    if is_setting:
        tec_input = np.flip(tec_input)
        leo_input = np.flip(leo_input, axis=1)
        gnss_input = np.flip(gnss_input, axis=1)
        time_input = np.flip(time_input)
        tangent_alt = np.flip(tangent_alt)
        occ_type = 'setting'
        print(f"Data Flipped: {podTc2_data['conid']}{podTc2_data['prn_id']} is setting")
    else:
        occ_type = 'rising'
        print(f"{podTc2_data['conid']}{podTc2_data['prn_id']} is rising")

    # Final Mask to limit to < 600km altitude
    mask_600 = tangent_alt < 600
    leo_input = leo_input[:, mask_600]
    gnss_input = gnss_input[:, mask_600]
    tangent_alt = tangent_alt[mask_600]
    tec_input = tec_input[mask_600]

    # Generate visual outputs if requested
    if generate_figs:
        # Plot 1: TEC Profile
        prof_save_path = os.path.join(doy_dir, f'{file}_TEC_profile.png')
        plot_tec_profile(tec_input, tangent_alt, podTc2_data["conid"], podTc2_data["prn_id"], occ_type, prof_save_path)

        # Plot 2: Geographical Ray Paths
        alt_limit = 500.0  
        tangent_point_filtered, _, _ = rayTangent(leo_input, gnss_input)
        lat_all, lon_all, alt_all = ECEFtolla(tangent_point_filtered)
        tangent_lla = np.array([lat_all, lon_all, alt_all])

        globe_save_path = os.path.join(doy_dir, f'{file}_ray_paths.png')
        plot_globe_ray_paths(leo_input, gnss_input, tangent_lla, 
                             podTc2_data['lon_tecmax_tangent'], 
                             podTc2_data['lat_tecmax_tangent'], 
                             alt_limit, globe_save_path)

    # Return the processed dictionary so the user can actually use the data programmatically
    return {
        "tec": tec_input,
        "tangent_alt": tangent_alt,
        "occ_type": occ_type
    }

def find_midpoints(arr):
    if len(arr) < 2:
        return []

    midpoints = np.zeros(len(arr)-1)
    for i in range(len(arr) - 1):
        midpoints[i] = (arr[i] + arr[i+1]) / 2
        # midpoints.append(midpoint)
    return midpoints

def distance(X,Y,Z):
    dist = np.zeros(len(X)-1)
    for i in range(len(X)-1):
        dist[i] = np.sqrt(np.abs(X[i] - X[i+1])**2 + np.abs(Y[i] - Y[i+1])**2 + np.abs(Z[i] - Z[i+1])**2)
    return dist


def process_geometry(i,LEO,GNSS,grid_lats,grid_lons,grid_alt):
    from EDPSamples.edp_samples import interp_heights
    from EDPSamples.edp_samples import _ecef_to_geodetic
    leo = LEO[:, i]
    gnss = GNSS[:, i]
    X, Y, Z = interpolate_ray_path(leo, gnss)

    result = {}

    dist = distance(X, Y, Z)
    raylength = dist * 1e3

    # Instead of the lats/lons/alts loop:
    xyz_m = np.stack([X, Y, Z], axis=-1) * 1e3   # km → metres; shape (N, 3)
    lats, lons, alts_m = _ecef_to_geodetic(xyz_m)
    alts = alts_m * 1e-3                          # metres → km

    lats = np.round(find_midpoints(np.array(lats)) * 2) / 2
    lons = np.round(find_midpoints(np.array(lons)) * 2) / 2
    alts = np.round(find_midpoints(np.array(alts)))
    
    lat_idx = np.searchsorted(grid_lats, lats).clip(0, len(grid_lats)-1)
    lon_idx = np.searchsorted(grid_lons, lons).clip(0, len(grid_lons)-1)
    
    alt_idx, alt_weights = interp_heights(grid_alt, alts)

    result['i'] = i
    result['raylength'] = raylength
    result['lat_idx'] = lat_idx
    result['lon_idx'] = lon_idx
    result['alt_idx'] = alt_idx

    return result

def forward_model_func(Ne_grid,grid_alt,LEO,GNSS,grid_lats,grid_lons,tangent_radius, file):

    INDEX_store = np.zeros((LEO.shape[1],3,999))
    raylengths = np.zeros((LEO.shape[1],999))
    
    filename = f"Geometry_for_{file}.npz"

    leo = LEO[:, 0]
    gnss = GNSS[:, 0]
    X, Y, Z = interpolate_ray_path(leo, gnss)
    radius = np.linalg.norm([X, Y, Z], axis=0)
    min_r = np.min(radius)
    idx = np.argmin(radius)
    theta = np.arccos(min_r / radius)
    # Multiply by -1 if the idices are less than the index of the minimum radius
    for j in range(len(theta)):
        if j < idx:
            theta[j] = -theta[j]
        # Check if file exists
    if os.path.exists(filename):
        print(f"\nLoading IRI profiles from {filename}")
        data = np.load(filename)
        INDEX_store = data['INDEX_store']
        raylengths = data['raylengths']
        grid_lats = data['grid_lats']
        grid_lons = data['grid_lons']
        print("Done")
    else:
        tasks = [ (i, LEO,GNSS,grid_lats,grid_lons,grid_alt) for i in range(LEO.shape[1])] 
        # Parallel execution
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_geometry, *task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)): 
                res = future.result()
                i = res['i']
                raylengths[i, :] = res['raylength']
                INDEX_store[i, 0, :] = res['lat_idx']
                INDEX_store[i, 1, :] = res['lon_idx']
                INDEX_store[i, 2, :] = res['alt_idx']
                # print(f'\rGeometry Definition: {i+1}/{LEO.shape[1]} ({(i+1)/LEO.shape[1]*100:.2f}%)', end='', flush=True)

        # Save to file
        np.savez_compressed(filename, INDEX_store=INDEX_store, raylengths=raylengths, grid_lats=grid_lats, grid_lons=grid_lons)
        print(f"Geometry saved to: {filename}")

    # Final computation
    Ne_tangent = Ne_grid[int(INDEX_store[0, 0, idx]), int(INDEX_store[0, 1, idx]), :]

                  
    TEC = np.zeros(LEO.shape[1])

    for i in range(LEO.shape[1]):
        print(f'\rTEC Calc: {i+1}/{LEO.shape[1]} ({(i+1)/LEO.shape[1]*100:.2f}%)', end='', flush=True)
        if i == 0:
            Ne_x = Ne_grid[INDEX_store[i,0,:].astype(int),INDEX_store[i,1,:].astype(int),:]
            x = (6371+300) * theta[0:999]
        for j in range(999):
            k      = INDEX_store[i, 2, j]      # lower bracket index
            w0, w1 = alt_weights[j]            # stored alongside alt_idx
            ne_interp = (w0 * Ne_grid[lat_i, lon_i, k] + w1 * Ne_grid[lat_i, lon_i, k+1])
            TEC[i] += ne_interp * raylengths[i, j]    # Convert TEC to TECU
    TEC = TEC / 1e16  # Convert to TECU (1 TECU = 1e16 electrons/m^2)
    print('\n')
    return TEC, Ne_x, x, Ne_tangent

import numpy as np
from scipy.interpolate import LinearNDInterpolator

def forward_model_mesh_tec(edp_dataset, podTc2_data, sample_idx=0, num_segments=1000):
    """
    Calculates TEC for radio occultation passes using an unstructured EDPSamples dataset.
    
    Parameters:
    -----------
    edp_dataset : xarray.Dataset
        The generated EDPSamples dataset containing 'altitude', 'geolocation', and 'EDPs'.
    podTc2_data : dict
        Dictionary containing 'LEO' and 'GNSS' position arrays (in km). Shape: (3, N_rays).
    sample_idx : int
        Index of the sampling parameter to evaluate (default is 0).
    num_segments : int
        Number of segments to divide the raypath into for integration.
    """
    
    # 1. Extract Mesh and EDP Data
    altitude = edp_dataset.coords['altitude'].values  # 1D array of heights (km)
    geolocation = edp_dataset.data_vars['geolocation'].values  # (n_geo, 2) array [lat, lon]
    
    # Extract EDPs for the specific parameter sample. 
    # Original shape: (n_height, n_geo, n_sample)
    # We slice out the sample, then transpose to (n_geo, n_height) for the interpolator
    edps_2d = edp_dataset.data_vars['EDPs'].values[:, :, sample_idx].T
    
    # Create an unstructured 2D interpolator. For any given (lat, lon), 
    # this will return the full 1D vertical altitude profile.
    print("Building spatial interpolator for the occultation mesh...")
    spatial_interp = LinearNDInterpolator(geolocation, edps_2d, fill_value=0.0)

    # 2. Extract Satellite Geometry
    LEO = podTc2_data['LEO']
    GNSS = podTc2_data['GNSS']
    n_rays = LEO.shape[1]
    
    TEC = np.zeros(n_rays)
    
    # Coordinate transformer: Cartesian ECEF to Lat/Lon/Alt
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
        pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
        always_xy=True
    )

    # 3. Ray Tracing & Integration
    for i in tqdm(range(n_rays), desc="Processing Occultation Rays"):
        
        # Parameterized steps along the ray (from 0 to 1)
        t = np.linspace(0, 1, num_segments)
        
        # Ray positions in ECEF (km) - Shape: (3, num_segments)
        ray_points = GNSS[:, i:i+1] + (LEO[:, i:i+1] - GNSS[:, i:i+1]) * t
        
        # Calculate integration step lengths (dl) in meters
        diffs = np.diff(ray_points, axis=1) # km
        dl_km = np.linalg.norm(diffs, axis=0) # Shape: (num_segments - 1)
        dl_m = dl_km * 1000.0 
        
        # Get midpoints for the integration segments
        midpoints = (ray_points[:, :-1] + ray_points[:, 1:]) / 2.0
        
        # Convert midpoints to Lat, Lon, Alt
        # (Assuming midpoints are in km, multiply by 1e3 for pyproj)
        lons, lats, alts_m = transformer.transform(
            midpoints[0, :] * 1e3, 
            midpoints[1, :] * 1e3, 
            midpoints[2, :] * 1e3
        )
        alts_km = alts_m / 1000.0 # Convert back to match EDP altitude limits
        
        # 4. Interpolate Electron Density
        # Feed the lats/lons into the interpolator.
        # Returns shape: (num_segments - 1, n_height)
        ray_profiles = spatial_interp(lats, lons)
        
        Ne_along_ray = np.zeros(num_segments - 1)
        
        for j in range(num_segments - 1):
            # If the ray steps outside the generated Lat/Lon triangular mesh, skip it
            if np.isnan(ray_profiles[j, 0]):
                continue
                
            # Perform a fast 1D interpolation along the altitude axis for this specific point
            Ne_along_ray[j] = np.interp(
                alts_km[j], 
                altitude, 
                ray_profiles[j, :], 
                left=0.0, 
                right=0.0
            )
            
        # 5. Integrate to calculate TEC (Ne * dl)
        # Sums electrons/m^3 * meters
        tec_ray = np.sum(Ne_along_ray * dl_m)
        
        # Convert to TECU (1 TECU = 1e16 electrons/m^2)
        TEC[i] = tec_ray / 1e16 

    return TEC