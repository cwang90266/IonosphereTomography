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

def _process_single_ray(
    gnss_i: np.ndarray,
    leo_i: np.ndarray,
    t: np.ndarray,
    altitude: np.ndarray,
    n_height: int,
    n_geo: int,
    n_state_vars: int,
    geolocation: np.ndarray,
    mesh: np.ndarray,
    tree,
    transformer,
    topside_scale_height_m: float,
    topside_n_steps: int = 10,
) -> np.ndarray:
    H_row = np.zeros(n_state_vars, dtype=np.float32)

    ray_points = gnss_i[:, np.newaxis] + (leo_i[:, np.newaxis] - gnss_i[:, np.newaxis]) * t
    dl_m = np.linalg.norm(np.diff(ray_points, axis=1), axis=0) * 1000.0
    midpoints = (ray_points[:, :-1] + ray_points[:, 1:]) / 2.0

    lons, lats, alts_m = transformer.transform(
        midpoints[0, :] * 1e3, midpoints[1, :] * 1e3, midpoints[2, :] * 1e3
    )
    alts_km = alts_m / 1000.0

    # --- 1. PROCESS 3D MESH ---
    valid_mask = (alts_km >= altitude[0]) & (alts_km <= altitude[-1])
    if np.any(valid_mask):
        dl_m_v   = dl_m[valid_mask]
        lats_v   = lats[valid_mask]
        lons_v   = lons[valid_mask]
        alts_km_v = alts_km[valid_mask]

        idx_alt, w_alt = interp_heights(altitude, alts_km_v)
        a_idx0, a_idx1 = idx_alt, idx_alt + 1
        aw0, aw1 = w_alt[:, 0], w_alt[:, 1]

        if n_geo == 1:
            np.add.at(H_row, a_idx0, dl_m_v * aw0)
            np.add.at(H_row, a_idx1, dl_m_v * aw1)
        else:
            tri_idx, bary = find_containing_triangles(
                np.column_stack([lats_v, lons_v]), geolocation, mesh, return_bary=True
            )
            inside  = tri_idx != -1
            outside = tri_idx == -1

            if np.any(inside):
                t_idx = tri_idx[inside]
                v0, v1, v2   = mesh[t_idx, 0], mesh[t_idx, 1], mesh[t_idx, 2]
                bw0, bw1, bw2 = bary[inside, 0], bary[inside, 1], bary[inside, 2]
                a0_in, a1_in  = a_idx0[inside], a_idx1[inside]
                aw0_in, aw1_in = aw0[inside], aw1[inside]
                dl_in = dl_m_v[inside]

                flat_idx = np.concatenate([
                    a0_in * n_geo + v0, a0_in * n_geo + v1, a0_in * n_geo + v2,
                    a1_in * n_geo + v0, a1_in * n_geo + v1, a1_in * n_geo + v2,
                ])
                flat_val = np.concatenate([
                    dl_in * bw0 * aw0_in, dl_in * bw1 * aw0_in, dl_in * bw2 * aw0_in,
                    dl_in * bw0 * aw1_in, dl_in * bw1 * aw1_in, dl_in * bw2 * aw1_in,
                ])
                H_row += np.bincount(flat_idx, weights=flat_val, minlength=n_state_vars)

            if np.any(outside):
                _, near_v = tree.query(np.column_stack([lats_v[outside], lons_v[outside]]))
                a0_out, a1_out = a_idx0[outside], a_idx1[outside]
                aw0_out, aw1_out = aw0[outside], aw1[outside]
                dl_out = dl_m_v[outside]

                flat_idx_out = np.concatenate([a0_out * n_geo + near_v, a1_out * n_geo + near_v])
                flat_val_out = np.concatenate([dl_out * aw0_out, dl_out * aw1_out])
                H_row += np.bincount(flat_idx_out, weights=flat_val_out, minlength=n_state_vars)

    # --- 2. PROCESS THE TOPSIDE: stepped exponential integration ---
    # Integrates ne(h) = ne(h_top) * exp(-(h - h_top) / H_s) above the grid.
    # Uses n_steps half-scale-height slabs; recomputes local cos_zenith each step.
    topside_mask = alts_km > altitude[-1]
    if np.any(topside_mask):
        top_indices  = np.where(topside_mask)[0]
        boundary_idx = top_indices[0]   # first midpoint above the grid top

        pierce_pt_m  = midpoints[:, boundary_idx] * 1000.0   # ECEF metres
        lat_p, lon_p = lats[boundary_idx], lons[boundary_idx]

        ray_dir       = (leo_i - gnss_i) / np.linalg.norm(leo_i - gnss_i)
        normal_0      = pierce_pt_m / np.linalg.norm(pierce_pt_m)
        cos_zenith_0  = max(np.abs(np.dot(ray_dir, normal_0)), 0.15)

        H_s_m      = topside_scale_height_m          # e-folding scale height [m]
        dh_m       = H_s_m / 2.0                     # vertical slab thickness [m]
        top_a_idx  = n_height - 1
        topside_contribution = 0.0

        for k in range(topside_n_steps):
            # Slant distance to the midpoint of slab k from the pierce point.
            # Uses the initial cos_zenith to convert vertical distance to path length.
            slant_m  = (k + 0.5) * dh_m / cos_zenith_0
            pos_k_m  = pierce_pt_m + slant_m * ray_dir

            # Recompute local vertical at this position (Earth's curvature rotates it).
            normal_k    = pos_k_m / np.linalg.norm(pos_k_m)
            cos_zen_k   = max(np.abs(np.dot(ray_dir, normal_k)), 0.15)

            # Path length through the vertical slab dh_m at this obliquity.
            dl_k_m = dh_m / cos_zen_k

            # Midpoint-rule exponential weight (more accurate than left-edge rule).
            weight_k = np.exp(-(k + 0.5) * dh_m / H_s_m)

            topside_contribution += dl_k_m * weight_k

        if n_geo == 1:
            H_row[top_a_idx] += topside_contribution
        else:
            tri_idx_top, bary_top = find_containing_triangles(
                np.array([[lat_p, lon_p]]), geolocation, mesh, return_bary=True
            )
            if tri_idx_top[0] != -1:
                v0, v1, v2    = mesh[tri_idx_top[0]]
                bw0, bw1, bw2 = bary_top[0]
                H_row[top_a_idx * n_geo + v0] += topside_contribution * bw0
                H_row[top_a_idx * n_geo + v1] += topside_contribution * bw1
                H_row[top_a_idx * n_geo + v2] += topside_contribution * bw2
            else:
                _, near_v = tree.query(np.array([[lat_p, lon_p]]))
                H_row[top_a_idx * n_geo + near_v[0]] += topside_contribution

    return H_row

    return H_row

class Ionosphere_Tomography_Inverter(KalmanFilter):
    """
    Extended Kalman Filter (EKF) for ionospheric tomographic inversion.

    The state is the log-ratio anomaly  x = log(ne / ne_bg), so the
    reconstructed electron density  ne = ne_bg * exp(x)  is guaranteed
    positive for any finite x — no separate non-negativity constraint needed.

    The observation model  TEC = H @ ne  is nonlinear in x; the EKF
    linearises it at the current predicted state each step:

        H_eff = H * ne_hat.T      (each column j scaled by ne_hat[j])
        y     = TEC_obs - H @ ne_hat          (nonlinear innovation)

    Prior covariance is computed in log-ratio space from the EDP ensemble.
    The Gauss-Markov model restores variance toward the prior each step.
    """

    def __init__(self, EDPSam: EDPSamples, ne_floor: float = 1.0, meanscale: int = 0):
        self.EDPSam = EDPSam
        edps = EDPSam.edps  # shape: (n_height, n_geo, n_sample)
        n_height, n_geo, n_sample = edps.shape
        n_state_vars = n_height * n_geo

        edps_flat = edps.reshape(n_state_vars, n_sample)
        edps_flat = np.nan_to_num(edps_flat, nan=0.0)

        # Background: first ensemble member.  Floor prevents log(0) at low
        # altitudes where IRI returns near-zero values; 1 e/m^3 is negligible
        # physically but keeps the log well-defined everywhere.
        ne_bg = np.maximum(edps_flat[:, 0:1], ne_floor)

        # Log-ratio ensemble: x_i = log(ne_i / ne_bg).
        # Prior mean x = 0 corresponds to ne = ne_bg.
        log_ensemble = np.log(np.maximum(edps_flat, ne_floor)) - np.log(ne_bg)

        super().__init__(dim_x=n_state_vars, dim_z=1)

        self.x = np.zeros((n_state_vars, 1))
        self.P = np.cov(log_ensemble)

        self.attrs = {
            "ne_bg":             ne_bg,          # background EDP (n, 1), always > 0
            "initial_edps_mean": ne_bg,          # alias kept for backward compatibility
            "initial_edps_cov":  self.P.copy(),  # log-ratio prior covariance
        }
    def get_observation_operator(self, podTc2_data: dict, num_segments: int = 1000,
                                 topside_scale_height_m: float = 150000.0,
                                 topside_n_steps: int = 10) -> np.ndarray:
        from joblib import Parallel, delayed
    
        altitude    = self.EDPSam.altitude
        geolocation = self.EDPSam.geolocation
        n_height    = len(altitude)
        n_geo       = geolocation.shape[0]
        n_state_vars = n_height * n_geo
    
        LEO    = podTc2_data['LEO']
        GNSS   = podTc2_data['GNSS']
        n_rays = LEO.shape[1]
    
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True
        )
        tree = cKDTree(geolocation) if n_geo > 1 else None
        t    = np.linspace(0, 1, num_segments)  # computed once, shared across all rays
    
        print(f"  -> Building H Matrix ({n_rays} rays, parallel)...")
        rows = Parallel(n_jobs=-1)(
            delayed(_process_single_ray)(
                GNSS[:, i], LEO[:, i], t,
                altitude, n_height, n_geo, n_state_vars,
                geolocation, self.EDPSam.mesh, tree,
                transformer, topside_scale_height_m,
                topside_n_steps,
            )
            for i in range(n_rays)
        )
    
        H = np.array(rows, dtype=np.float32)
        H /= 1e16
        return H
    
    def assimilate(self, obs: np.ndarray, podTc2_data: dict = None, obs_operator: np.ndarray = None,
                   relaxation: float = 1.0, measurement_err: float = 0.0) -> np.ndarray:
        """
        Single EKF assimilation step in log-electron-density space.

        State x = log(ne / ne_bg), reconstructed as ne = ne_bg * exp(x) > 0.
        The EKF Jacobian H_eff = H * ne_hat.T is re-evaluated at the predicted
        state each call, so the linearisation tracks the evolving solution.
        """
        obs = np.asarray(obs).reshape(-1, 1)
        n_obs = obs.shape[0]

        if obs_operator is None:
            assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
            print("Dynamically calculating observation operator (H)...")
            obs_operator = self.get_observation_operator(podTc2_data)

        assert obs.shape[0] == obs_operator.shape[0], "Observation length must match H matrix rows"
        assert obs_operator.shape[1] == self.dim_x,   "H matrix columns must match state vector length"

        # Gauss-Markov predict
        self.x = relaxation * self.x
        self.P = (relaxation ** 2) * self.P + (1.0 - relaxation) * self.attrs["initial_edps_cov"]

        # EDP at predicted state — always positive by construction
        ne_hat = self.attrs["ne_bg"] * np.exp(self.x)          # (n, 1)

        # EKF Jacobian: d(H @ ne) / dx = H * ne_hat.T
        H_eff = obs_operator * ne_hat.T                         # (n_obs, n)

        # Innovation: observed TEC minus nonlinear forward model
        y = obs - obs_operator @ ne_hat                         # (n_obs, 1)

        # Kalman update  (O(n² × n_obs), same as original)
        PHT = self.P @ H_eff.T                                  # (n, n_obs)
        S   = H_eff @ PHT + max(measurement_err, 1e-6) * np.eye(n_obs)
        K   = np.linalg.solve(S, PHT.T).T
        self.x = self.x + K @ y
        self.P = self.P - K @ PHT.T

        # Reconstruct absolute EDP — guaranteed positive for any finite x
        return self.attrs["ne_bg"] * np.exp(self.x)