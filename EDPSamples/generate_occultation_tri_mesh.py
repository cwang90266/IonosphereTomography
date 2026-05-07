import numpy as np
import netCDF4
import pandas as pd
import pyproj
import warnings
from TEC_model.podTc_file_processing import parse_podTc2_nc_file

def latlon_to_cartesian(lat, lon):
    """Converts Lat/Lon (degrees) to a 3D Cartesian unit vector."""
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    return np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

def cartesian_to_latlon(vec):
    """Converts a 3D Cartesian unit vector back to Lat/Lon (degrees)."""
    # Normalize to snap the point back onto the surface of the sphere
    vec = vec / np.linalg.norm(vec)  
    lat = np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(vec[1], vec[0]))
    return lat, lon

def rayTangent(LEO, GNSS, units='km'):
    """Calculates the tangent points and altitude of the raypath."""
    v = LEO - GNSS
    t_s = np.clip(-np.sum(v * GNSS, axis=0) / np.sum(v * v, axis=0), 0.0, 1.0)
    tangent_point = GNSS + v * t_s[np.newaxis, :]
    p = np.linalg.norm(tangent_point, axis=0)
    
    # Convert to ellipsoidal altitude
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
    else:
        raise ValueError(f"units must be 'km' or 'm', got '{units}'")

    return tangent_point, p, alt

def get_occultation_extrema(LEO, GNSS, alt_limit=600.0, r_earth_km=6371.0):
    """
    Analytically computes 3 bounding points (lat, lon) by finding the tangent 
    point of the highest valid ray, and the entry/exit points of the deepest ray.
    """
    # 1. Calculate tangent points and filter by valid altitude limits [0, alt_limit]
    tangent_point, _, tangent_alt = rayTangent(LEO, GNSS, units='km')
    
    valid_mask = (tangent_alt >= 0) & (tangent_alt <= (alt_limit*1e3))
    if not np.any(valid_mask):
        raise ValueError(f"No tangent points found within bounds [0, {alt_limit}] km.")
        
    # Extract only the data within the valid altitude envelope
    LEO_v = LEO[:, valid_mask]
    GNSS_v = GNSS[:, valid_mask]
    Tan_v = tangent_point[:, valid_mask]
    alts_v = tangent_alt[valid_mask]
    
    # 2. Identify the indices for the highest (shallowest) and lowest (deepest) tangent points
    idx_high = np.argmax(alts_v)
    idx_low = np.argmin(alts_v)
    
    # 3. p1: The tangent point of the highest altitude ray
    ecef_p1 = Tan_v[:, idx_high]
    
    # 4. p2 & p3: The endpoints (entry/exit) of the lowest ray at the alt_limit boundary
    leo_low = LEO_v[:, idx_low]
    gnss_low = GNSS_v[:, idx_low]
    v_low = leo_low - gnss_low
    
    # Solve quadratic equation for the single lowest ray intersecting the alt_limit sphere
    R_target = r_earth_km + alt_limit
    
    a = np.dot(v_low, v_low)
    b = 2.0 * np.dot(gnss_low, v_low)
    c = np.dot(gnss_low, gnss_low) - (R_target ** 2)
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        # Fallback if precision issues cause the deepest ray to theoretically miss the shell
        raise ValueError("The deepest ray does not intersect the altitude limit shell.")
        
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    
    ecef_p2 = gnss_low + v_low * t1  # Entry point
    ecef_p3 = gnss_low + v_low * t2  # Exit point
    
    # 5. Convert all 3 ECEF points back to Lat/Lon
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
        pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
        always_xy=True
    )
    
    points_ecef = np.column_stack((ecef_p1, ecef_p2, ecef_p3))
    
    lons, lats, _ = transformer.transform(
        1e3 * points_ecef[0, :],
        1e3 * points_ecef[1, :],
        1e3 * points_ecef[2, :]
    )
    
    # Return as tuples: (lat, lon)
    p1 = (lats[0], lons[0])
    p2 = (lats[1], lons[1])
    p3 = (lats[2], lons[2])
    
    return p1, p2, p3

def generate_occultation_mesh(pt1=None, pt2=None, pt3=None, filename=None, dLat=0.5, dLon=0.5, alt_limit=600.0):
    """
    Generates a rectangular grid mesh of vertices (lat, lon) aligned with an 
    occultation geometry.
    
    Provide EITHER:
    - pt1, pt2, pt3: Tuple coordinates (lat, lon) defining the occultation plane.
    - filename: String path to the podTc2 netCDF file to analytically determine bounds.
    """
    
    if filename is not None:
        # Load data and dynamically calculate the bounds based on the raypath extrema
        data = parse_podTc2_nc_file(filename)
        # Function now outputs exactly what the rest of the script needs
        pt1, pt2, pt3 = get_occultation_extrema(
            data['LEO'], data['GNSS'], alt_limit=alt_limit
        )
    
    if pt1 is not None and pt2 is not None and pt3 is not None:
        
        # --- Mesh Generation Core (3D Spherical Barycentric Subdivision) ---
        target_res = (dLat + dLon) / 2.0
        
        # 1. Convert boundary points to 3D Cartesian unit vectors
        v1 = latlon_to_cartesian(pt1[0], pt1[1])
        v2 = latlon_to_cartesian(pt2[0], pt2[1])
        v3 = latlon_to_cartesian(pt3[0], pt3[1])
        
        # 2. Calculate Great Circle distances (in degrees) for the edges
        # This prevents the distance calculation from failing at the poles
        e1 = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        e2 = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))
        e3 = np.degrees(np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0)))
        max_edge = max(e1, e2, e3)
        
        N = max(1, int(np.ceil(max_edge / target_res)))
        
        vertices_list = []
        indices = {}
        idx_counter = 0
        
        for i in range(N + 1):
            for j in range(N + 1 - i):
                k = N - i - j
                w1, w2, w3 = i / N, j / N, k / N
                
                # 3. Interpolate strictly in 3D space
                interp_v = w1 * v1 + w2 * v2 + w3 * v3
                
                # 4. Convert back to Lat/Lon
                lat, lon = cartesian_to_latlon(interp_v)
                
                vertices_list.append([lon, lat])
                indices[(i, j)] = idx_counter
                idx_counter += 1
                
        vertices = np.array(vertices_list)
        
        # Generate Triangles linking the vertices (Unchanged)
        triangles = []
        for i in range(N):
            for j in range(N - i):
                v0 = indices[(i, j)]
                v1 = indices[(i + 1, j)]
                v2 = indices[(i, j + 1)]
                triangles.append([v0, v1, v2])
                
                if j < N - i - 1:
                    v0 = indices[(i + 1, j)]
                    v1 = indices[(i + 1, j + 1)]
                    v2 = indices[(i, j + 1)]
                    triangles.append([v0, v1, v2])
                    
        triangles = np.array(triangles, dtype=int)
    else:
        raise ValueError("You must provide either (pt1, pt2, pt3) OR a valid (filename).")
        
    return vertices, triangles, pt1, pt2, pt3