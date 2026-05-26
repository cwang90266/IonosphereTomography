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
    topside_H_H_m: float = 1000000.0,
    topside_alpha: float = 0.05,
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

    # --- 2. PROCESS THE TOPSIDE: Two-Ion Plasmasphere Extension ---
    topside_mask = alts_km > altitude[-1]
    if np.any(topside_mask):
        top_indices  = np.where(topside_mask)[0]
        boundary_idx = top_indices[-1]  # last topside midpoint = ray entry into topside from below

        pierce_pt_m = midpoints[:, boundary_idx] * 1000.0  # ECEF metres

        # Integration direction: from pierce point (grid top) upward toward GNSS
        ray_dir    = (gnss_i - leo_i) / np.linalg.norm(gnss_i - leo_i)
        normal_0   = pierce_pt_m / np.linalg.norm(pierce_pt_m)
        cos_zenith_0 = max(np.abs(np.dot(ray_dir, normal_0)), 0.05)

        H_O_m = topside_scale_height_m
        H_H_m = topside_H_H_m
        alpha  = topside_alpha

        dh_m       = H_O_m / max(topside_n_steps, 1)
        max_dist_m = np.linalg.norm(gnss_i * 1000.0 - pierce_pt_m)
        r_pierce   = np.linalg.norm(pierce_pt_m)
        top_a_idx  = n_height - 1

        # --- Vectorised topside integration (replaces per-step Python loop) ---
        # Upper bound on steps: H+ term < threshold when h > -H_H * ln(threshold / alpha)
        h_stop_m = -H_H_m * np.log(max(1e-8 / alpha, 1e-30))
        k_max    = int(h_stop_m / dh_m) + 2

        k_arr     = np.arange(k_max, dtype=np.float64) + 0.5
        slant_arr = k_arr * dh_m / cos_zenith_0          # (k_max,) slant distances
        slant_arr = slant_arr[slant_arr < max_dist_m]    # truncate at GNSS satellite

        if slant_arr.size > 0:
            # All positions along the topside path in one shot: shape (3, n_steps)
            positions = pierce_pt_m[:, None] + slant_arr[None, :] * ray_dir[:, None]

            r_arr    = np.linalg.norm(positions, axis=0)              # (n_steps,)
            normals  = positions / r_arr[None, :]                     # (3, n_steps)
            cos_zens = np.maximum(np.abs(ray_dir @ normals), 0.05)   # (n_steps,)

            h_diff   = r_arr - r_pierce                               # (n_steps,)
            weights  = ((1.0 - alpha) * np.exp(-h_diff / H_O_m)
                        + alpha       * np.exp(-h_diff / H_H_m))

            valid        = weights >= 1e-8
            dl_arr       = dh_m / cos_zens                            # (n_steps,)
            contributions = (dl_arr * weights * valid).astype(np.float32)

            if np.any(valid):
                if n_geo == 1:
                    H_row[top_a_idx] += contributions.sum()
                else:
                    # Single bulk coordinate transform for all valid steps
                    pos_v = positions[:, valid]
                    lons_t, lats_t, _ = transformer.transform(
                        pos_v[0], pos_v[1], pos_v[2]
                    )
                    contribs_v = contributions[valid]

                    tri_arr, bary_arr = find_containing_triangles(
                        np.column_stack([lats_t, lons_t]),
                        geolocation, mesh, return_bary=True,
                    )

                    inside  = tri_arr != -1
                    outside = tri_arr == -1

                    if np.any(inside):
                        t_idx        = tri_arr[inside]
                        v0, v1, v2   = mesh[t_idx, 0], mesh[t_idx, 1], mesh[t_idx, 2]
                        bw0, bw1, bw2 = bary_arr[inside, 0], bary_arr[inside, 1], bary_arr[inside, 2]
                        c_in         = contribs_v[inside]
                        flat_idx = np.concatenate([
                            top_a_idx * n_geo + v0,
                            top_a_idx * n_geo + v1,
                            top_a_idx * n_geo + v2,
                        ])
                        flat_val = np.concatenate([c_in * bw0, c_in * bw1, c_in * bw2])
                        H_row += np.bincount(flat_idx, weights=flat_val,
                                             minlength=n_state_vars)

                    if np.any(outside):
                        _, near_vs = tree.query(
                            np.column_stack([lats_t[outside], lons_t[outside]])
                        )
                        c_out        = contribs_v[outside]
                        flat_idx_out = top_a_idx * n_geo + near_vs
                        H_row += np.bincount(flat_idx_out, weights=c_out,
                                             minlength=n_state_vars)

    return H_row

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

        # 2. CRITICAL FIX for Log-State: Sanitize the IRI background state.
        # We use a realistic ionospheric floor (1e8 m^-3) to prevent the ensemble
        # variance from exploding when comparing normal values to near-zero values.
        physical_floor = 1e8  
        edps_flat = np.nan_to_num(edps_flat, nan=physical_floor)
        edps_flat = np.clip(edps_flat, physical_floor, None)

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
            "initial_edps_mean": edps_base,  # Keeping the old dict key so you don't have to change the assimilate() code
            "initial_edps_cov": self.P.copy()
        }
    def get_observation_operator(self, podTc2_data: dict, num_segments: int = 1000,
                                 topside_scale_height_m: float = 150000.0,
                                 topside_n_steps: int = 10,
                                 topside_H_H_m: float = 1000000.0,
                                 topside_alpha: float = 0.05) -> np.ndarray:
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
                topside_n_steps, topside_H_H_m, topside_alpha,
            )
            for i in range(n_rays)
        )

        H = np.array(rows, dtype=np.float32)
        H /= 1e16
        return H

    def assimilate(self, obs: np.ndarray, podTc2_data: dict = None, obs_operator: np.ndarray = None,
                   relaxation: float = 1.0, measurement_err: float = 0.0) -> np.ndarray:
        """
        Runs a single Kalman Filter assimilation step.
        """
        # Strip masked-array wrapper (netCDF4 returns np.ma.MaskedArray by default).
        # np.asarray extracts the underlying data; any fill values become ordinary floats.
        obs = np.asarray(obs).reshape(-1, 1)

        # 1. Generate or validate the H matrix
        if obs_operator is None:
            assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
            print("Dynamically calculating observation operator (H)...")
            obs_operator = self.get_observation_operator(podTc2_data)

        assert obs.shape[0] == obs_operator.shape[0], "Observation length must match H matrix rows"
        assert obs_operator.shape[1] == self.dim_x, "H matrix columns must match State vector length"

        n_obs = obs.shape[0]

        # 2. Apply Mean Scaling to the H matrix if needed (no copy when unscaled)
        if self.attrs["meanscale"] == 1:
            H = obs_operator * self.attrs["initial_edps_mean"].T
        else:
            H = obs_operator

        # 3. Gauss-Markov Predict: F = relaxation * I — avoids two O(d³) matrix multiplies
        self.x = relaxation * self.x
        self.P = (relaxation ** 2) * self.P + (1.0 - relaxation) * self.attrs["initial_edps_cov"]

        # 4. Innovation
        background_tec = obs_operator @ self.attrs["initial_edps_mean"]
        y = (obs - background_tec) - H @ self.x

        # 5. Efficient Update: O(d² × n_obs) vs filterpy's Joseph form O(d³)
        #    PHT = P @ Hᵀ         (d × n_obs)
        #    S   = H @ PHT + R    (n_obs × n_obs) — small, cheap to factor
        #    K   = PHT @ S⁻¹      solved without explicit inverse
        #    P   = P - K @ PHᵀ   (d² × n_obs flops, not d³)
        PHT      = self.P @ H.T
        S        = H @ PHT
        S       += max(measurement_err, 1e-6) * np.eye(n_obs)
        K        = np.linalg.solve(S, PHT.T).T
        self.x   = self.x + K @ y
        self.P   = self.P - K @ PHT.T

        # 7. Reconstruct Absolute Electron Density for output
        # self.x currently represents the anomaly/perturbation from the background mean
        if self.attrs["meanscale"] == 1:
            # Total State = Mean * (1 + Fractional Anomaly)
            analysis_x = self.attrs["initial_edps_mean"] * (1.0 + self.x)
        else:
            # Total State = Mean + Absolute Anomaly
            analysis_x = self.attrs["initial_edps_mean"] + self.x

        return analysis_x

    def plot_covariance_correlation(self, title: str = None, P: np.ndarray = None) -> np.ndarray:
        """
        Plot the altitude-altitude correlation matrix derived from self.P.

        Averages P over geo-point pairs to produce an (n_height, n_height)
        covariance, then normalises to a Pearson correlation matrix and plots
        it with altitude axes, matching the style of the universal covariance
        plots in demo.py.

        Parameters
        ----------
        title : str, optional
            Figure title.  Defaults to "Altitude-Altitude Correlation (P)".

        Returns
        -------
        corr_alt : np.ndarray, shape (n_height, n_height)
            Altitude-altitude correlation matrix.
        """
        import matplotlib.pyplot as plt
        import warnings

        altitude = self.EDPSam.altitude                  # (n_height,)
        n_height = len(altitude)
        n_geo    = self.EDPSam.geolocation.shape[0]

        # Reshape (n_h*n_g, n_h*n_g) → (n_h, n_g, n_h, n_g), average over geo
        _P      = P if P is not None else self.P
        P_4d    = _P.reshape(n_height, n_geo, n_height, n_geo)
        cov_alt = P_4d.mean(axis=(1, 3))                 # (n_height, n_height)

        std_devs  = np.sqrt(np.diag(cov_alt))
        outer_std = np.outer(std_devs, std_devs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr_alt = cov_alt / np.where(outer_std == 0, 1e-10, outer_std)

        alt_extent = [float(altitude[0]), float(altitude[-1]),
                      float(altitude[0]), float(altitude[-1])]

        fig, ax = plt.subplots(figsize=(7, 6))
        pcm = ax.imshow(corr_alt, cmap='coolwarm', vmin=-1, vmax=1,
                        extent=alt_extent, origin='lower', aspect='auto')
        ax.set_title(title or "Altitude-Altitude Correlation (P)")
        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel("Altitude (km)")
        fig.colorbar(pcm, ax=ax, label="Correlation Coefficient")
        plt.tight_layout()

        return corr_alt
