#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class of Tomography Data Assimilation Filter

Created on Mon May 11 13:47:34 2026

@author: cwang
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class of Tomography Data Assimilation Filter
"""

import numpy as np
import pyproj
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

# Ensure you import these from your edp_samples module
from EDPSamples.edp_samples import EDPSamples, interp_heights, find_containing_triangles

class Ionosphere_Tomography_Inverter(KalmanFilter):
    """
    The class Ionosphere_Tomography_Inverter is a subclass of the KalmanFilter 
    which is dedicated for ionosphere tomographic inversion from the total 
    electron content (TEC) measurements to electron density profiles. The key 
    difference between the standard Kalman filter object is that the dimension
    of the observation and the observation operator change in each iteration. 
    The state transition matrix is the identity matrix multiplied by a scalar
    coefficient exp(-delta). This is the Gauss Markov model for a stationary 
    process. 
    """
    
    def __init__(self, EDPSam: EDPSamples, meanscale: int = 0):
        self.EDPSam = EDPSam
        edps = EDPSam.edps  # Original shape: (n_height, n_geo, n_sample)
        n_height, n_geo, n_sample = edps.shape
        n_state_vars = n_height * n_geo
        
        # 1. Flatten spatial and altitude dimensions into a state vector
        edps_flat = edps.reshape(n_state_vars, n_sample)
        
        # CRITICAL FIX: Sanitize the IRI background state. 
        # IRI returns NaNs for regions outside its defined altitude bounds. 
        # We must replace these with 0.0 or the Kalman matrix math will explode.
        edps_flat = np.nan_to_num(edps_flat, nan=0.0)
        
        # Extract the base state (0th index) instead of the ensemble mean
        edps_base = edps_flat[:, 0:1]
        
        # 2. Scale to fractional perturbations if requested
        if meanscale == 1:
            # Avoid division by zero
            safe_base = np.where(edps_base == 0, 1e-10, edps_base)
            edps_flat = edps_flat / safe_base

        # 3. Initialize the FilterPy parent class
        super().__init__(dim_x=n_state_vars, dim_z=1)

        # 4. Set Initial State and Covariance
        self.x = np.zeros((n_state_vars, 1))
        self.P = np.cov(edps_flat)

        # 5. Store metadata (Update the key name here too if you like)
        self.attrs = {
            "meanscale": meanscale,
            "initial_edps": edps_flat,
            "initial_edps_mean": edps_base  # Keeping the old dict key so you don't have to change the assimilate() code
        }

    def get_observation_operator(self, podTc2_data: dict, num_segments: int = 1000, topside_scale_height_m: float = 500000.0) -> np.ndarray:
        """
        Generates the Observation Operator (H matrix) for the Kalman Filter.
        Maps the integration path of RO rays to the flattened state vector.
        """
        altitude = self.EDPSam.altitude
        geolocation = self.EDPSam.geolocation
        n_height = len(altitude)
        n_geo = geolocation.shape[0]
        n_state_vars = n_height * n_geo
        
        LEO = podTc2_data['LEO']
        GNSS = podTc2_data['GNSS']
        n_rays = LEO.shape[1]
        
        H = np.zeros((n_rays, n_state_vars), dtype=np.float32)
        
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True
        )

        if n_geo > 1:
            tree = cKDTree(geolocation)

        for i in tqdm(range(n_rays), desc="Building H Matrix"):
            t = np.linspace(0, 1, num_segments)
            ray_points = GNSS[:, i:i+1] + (LEO[:, i:i+1] - GNSS[:, i:i+1]) * t
            
            diffs = np.diff(ray_points, axis=1)
            dl_m = np.linalg.norm(diffs, axis=0) * 1000.0  # Lengths in meters
            
            midpoints = (ray_points[:, :-1] + ray_points[:, 1:]) / 2.0
            
            lons, lats, alts_m = transformer.transform(
                midpoints[0, :] * 1e3, midpoints[1, :] * 1e3, midpoints[2, :] * 1e3
            )
            alts_km = alts_m / 1000.0
            # -----------------------------------------------------------------
            # 1. PROCESS 3D MESH
            # -----------------------------------------------------------------
            valid_mask = (alts_km >= altitude[0]) & (alts_km <= altitude[-1])
            if not np.any(valid_mask):
                continue
                
            dl_m_v = dl_m[valid_mask]
            lats_v = lats[valid_mask]
            lons_v = lons[valid_mask]
            alts_km_v = alts_km[valid_mask]
            
            idx_alt, w_alt = interp_heights(altitude, alts_km_v)
            a_idx0, a_idx1 = idx_alt, idx_alt + 1
            aw0, aw1 = w_alt[:, 0], w_alt[:, 1]
            
            if n_geo == 1:
                # Spherical Symmetry
                np.add.at(H[i], a_idx0, dl_m_v * aw0)
                np.add.at(H[i], a_idx1, dl_m_v * aw1)
            else:
                # Mesh Geometry
                tri_idx, bary = find_containing_triangles(
                    np.column_stack([lats_v, lons_v]), geolocation, self.EDPSam.mesh, return_bary=True
                )
                
                inside = tri_idx != -1
                outside = tri_idx == -1
                
                if np.any(inside):
                    t_idx = tri_idx[inside]
                    v0, v1, v2 = self.EDPSam.mesh[t_idx, 0], self.EDPSam.mesh[t_idx, 1], self.EDPSam.mesh[t_idx, 2]
                    bw0, bw1, bw2 = bary[inside, 0], bary[inside, 1], bary[inside, 2]
                    a0_in, a1_in = a_idx0[inside], a_idx1[inside]
                    aw0_in, aw1_in = aw0[inside], aw1[inside]
                    dl_in = dl_m_v[inside]
                    
                    np.add.at(H[i], a0_in * n_geo + v0, dl_in * bw0 * aw0_in)
                    np.add.at(H[i], a0_in * n_geo + v1, dl_in * bw1 * aw0_in)
                    np.add.at(H[i], a0_in * n_geo + v2, dl_in * bw2 * aw0_in)
                    
                    np.add.at(H[i], a1_in * n_geo + v0, dl_in * bw0 * aw1_in)
                    np.add.at(H[i], a1_in * n_geo + v1, dl_in * bw1 * aw1_in)
                    np.add.at(H[i], a1_in * n_geo + v2, dl_in * bw2 * aw1_in)

                if np.any(outside):
                    _, near_v = tree.query(np.column_stack([lats_v[outside], lons_v[outside]]))
                    a0_out, a1_out = a_idx0[outside], a_idx1[outside]
                    aw0_out, aw1_out = aw0[outside], aw1[outside]
                    dl_out = dl_m_v[outside]
                    
                    np.add.at(H[i], a0_out * n_geo + near_v, dl_out * aw0_out)
                    np.add.at(H[i], a1_out * n_geo + near_v, dl_out * aw1_out)
            # -----------------------------------------------------------------
            # 2. PROCESS THE TOPSIDE APPROXIMATION
            # -----------------------------------------------------------------        
            topside_mask = (alts_km > altitude[-1])
            
            if np.any(topside_mask):
                # Find the segment closest to the mesh boundary (the pierce point)
                top_indices = np.where(topside_mask)[0]
                boundary_idx = top_indices[np.argmin(np.abs(alts_km[top_indices] - altitude[-1]))]
                
                pierce_pt_m = midpoints[:, boundary_idx] * 1000.0  # Convert km to meters
                lat_p = lats[boundary_idx]
                lon_p = lons[boundary_idx]
                
                # Calculate Ray Direction
                ray_vec = LEO[:, i] - GNSS[:, i]
                ray_dir = ray_vec / np.linalg.norm(ray_vec)
                
                # Calculate Earth Normal at Pierce Point
                normal_vec = pierce_pt_m / np.linalg.norm(pierce_pt_m)
                
                # Calculate Obliquity Factor: sec(chi) = 1 / |cos(chi)|
                # We use absolute value because the ray direction could be up or down relative to the normal
                cos_zenith = np.abs(np.dot(ray_dir, normal_vec))
                cos_zenith = max(cos_zenith, 0.087) # Cap zenith at ~85 degrees to prevent infinite path lengths
                obliquity = 1.0 / cos_zenith
                
                # Calculate Effective Topside Path Length
                topside_dl_m = topside_scale_height_m * obliquity
                
                # Target the highest altitude index in your state vector
                top_a_idx = n_height - 1
                
                if n_geo == 1:
                    np.add.at(H[i], top_a_idx, topside_dl_m)
                else:
                    # Find horizontal nodes for the pierce point
                    tri_idx, bary = find_containing_triangles(
                        np.array([[lat_p, lon_p]]), geolocation, self.EDPSam.mesh, return_bary=True
                    )
                    
                    if tri_idx[0] != -1:
                        # Inside horizontal mesh
                        v0, v1, v2 = self.EDPSam.mesh[tri_idx[0]]
                        bw0, bw1, bw2 = bary[0]
                        
                        np.add.at(H[i], top_a_idx * n_geo + v0, topside_dl_m * bw0)
                        np.add.at(H[i], top_a_idx * n_geo + v1, topside_dl_m * bw1)
                        np.add.at(H[i], top_a_idx * n_geo + v2, topside_dl_m * bw2)
                    else:
                        # Outside horizontal mesh - snap to nearest
                        _, near_v = tree.query(np.array([[lat_p, lon_p]]))
                        np.add.at(H[i], top_a_idx * n_geo + near_v[0], topside_dl_m)

        H /= 1e16 # Convert to TECU
        return H
        H /= 1e16 # Convert to TECU
        return H

    def assimilate(self, obs: np.ndarray, podTc2_data: dict = None, obs_operator: np.ndarray = None,
                   relaxation: float = 1.0, measurement_err: float = 0.0) -> np.ndarray:
        """
        Runs a single Kalman Filter assimilation step.
        """
        # Ensure obs is a column vector
        obs = obs.reshape(-1, 1)

        # 1. Generate or validate the H matrix
        if obs_operator is None:
            assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
            print("Dynamically calculating observation operator (H)...")
            obs_operator = self.get_observation_operator(podTc2_data)

        assert obs.shape[0] == obs_operator.shape[0], "Observation length must match H matrix rows"
        assert obs_operator.shape[1] == self.dim_x, "H matrix columns must match State vector length"
        
        # 2. Dynamically resize the filter's observation dimension
        self.dim_z = obs.shape[0]

        # 3. Apply Mean Scaling to the H matrix if needed
        H_scaled = np.copy(obs_operator)
        if self.attrs["meanscale"] == 1:
            # Broadcast multiply across columns to map fractional state to absolute TEC
            H_scaled = H_scaled * self.attrs["initial_edps_mean"].T

        self.H = H_scaled
        
        # 4. Gauss-Markov State Transition & Process Noise
        self.F = relaxation * np.eye(self.dim_x)
        self.Q = (1.0 - relaxation) * np.cov(self.attrs["initial_edps"])
        
        # 5. Dynamic Observation Covariance (R)
        # Map the initial ensemble to observation space: Z = H * Ensemble
        # initial_predict_obs = self.H @ self.attrs["initial_edps"]
        # self.R = np.cov(initial_predict_obs) + measurement_err * np.eye(self.dim_z)
        self.R = measurement_err * np.eye(self.dim_z)
        
        # 6. Execute Kalman Filter Steps
        # predict() advances self.x and self.P based on self.F and self.Q
        self.predict()
        
        # --- CRITICAL FIX ---
        # Calculate the Background TEC using the unscaled operator and base state
        background_tec = obs_operator @ self.attrs["initial_edps_mean"]
        
        # Because self.x tracks the *perturbation* (initialized to 0), 
        # the filter must assimilate the difference between the measurement and the background.
        obs_anomaly = obs - background_tec
        
        # update() calculates innovation (obs_anomaly - H*x) and updates self.x and self.P
        self.update(obs_anomaly)
        
        # 7. Reconstruct Absolute Electron Density for output
        # self.x currently represents the anomaly/perturbation from the background mean
        if self.attrs["meanscale"] == 1:
            # Total State = Mean * (1 + Fractional Anomaly)
            analysis_x = self.attrs["initial_edps_mean"] * (1.0 + self.x)
        else:
            # Total State = Mean + Absolute Anomaly
            analysis_x = self.attrs["initial_edps_mean"] + self.x

        return analysis_x