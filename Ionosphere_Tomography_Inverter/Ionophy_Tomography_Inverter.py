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
    n_state_vars_aug: int,
    geolocation: np.ndarray,
    mesh: np.ndarray,
    tree,
    transformer,
    H_eff_m: float,
) -> np.ndarray:
    H_row = np.zeros(n_state_vars_aug, dtype=np.float32)

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
                H_row += np.bincount(flat_idx, weights=flat_val, minlength=n_state_vars_aug)

            if np.any(outside):
                _, near_v = tree.query(np.column_stack([lats_v[outside], lons_v[outside]]))
                a0_out, a1_out = a_idx0[outside], a_idx1[outside]
                aw0_out, aw1_out = aw0[outside], aw1[outside]
                dl_out = dl_m_v[outside]

                flat_idx_out = np.concatenate([a0_out * n_geo + near_v, a1_out * n_geo + near_v])
                flat_val_out = np.concatenate([dl_out * aw0_out, dl_out * aw1_out])
                H_row += np.bincount(flat_idx_out, weights=flat_val_out, minlength=n_state_vars_aug)

    # --- 2. TOPSIDE COLUMNS — Numerical Integration ---
    # N_e(h) = (VTEC_top / H_eff) * exp(-(h - h_top) / H_eff)
    # The integration weight for a segment dl is: (dl / H_eff) * exp(...)
    # Multiplied by x_top[j] (VTEC in TECU), this gives slant TECU directly.
    topside_mask = alts_km > altitude[-1]
    if np.any(topside_mask):
        dl_m_t    = dl_m[topside_mask]
        lats_t    = lats[topside_mask]
        lons_t    = lons[topside_mask]
        alts_km_t = alts_km[topside_mask]

        H_eff_km = H_eff_m / 1000.0
        h_top_km = altitude[-1]

        # Calculate dimensionless integration weights for all topside segments
        decay_factors = np.exp(-(alts_km_t - h_top_km) / H_eff_km)
        topside_weights = (dl_m_t / H_eff_m) * decay_factors

        if n_geo == 1:
            H_row[n_state_vars] += np.sum(topside_weights)
        else:
            # Map topside segments to spatial geo-nodes using barycentric coordinates
            tri_t, bary_t = find_containing_triangles(
                np.column_stack([lats_t, lons_t]), geolocation, mesh, return_bary=True
            )
            inside_t = tri_t != -1
            outside_t = tri_t == -1

            if np.any(inside_t):
                t_idx = tri_t[inside_t]
                v0, v1, v2 = mesh[t_idx, 0], mesh[t_idx, 1], mesh[t_idx, 2]
                bw0, bw1, bw2 = bary_t[inside_t, 0], bary_t[inside_t, 1], bary_t[inside_t, 2]
                w_in = topside_weights[inside_t]
                
                flat_idx_t = np.concatenate([
                    n_state_vars + v0, n_state_vars + v1, n_state_vars + v2
                ])
                flat_val_t = np.concatenate([
                    w_in * bw0, w_in * bw1, w_in * bw2
                ])
                H_row += np.bincount(flat_idx_t, weights=flat_val_t, minlength=n_state_vars_aug)

            if np.any(outside_t):
                _, near_v = tree.query(np.column_stack([lats_t[outside_t], lons_t[outside_t]]))
                w_out = topside_weights[outside_t]
                flat_idx_out = n_state_vars + near_v
                H_row += np.bincount(flat_idx_out, weights=w_out, minlength=n_state_vars_aug)

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

    def __init__(self, EDPSam: EDPSamples, meanscale: int = 0,
                 topside_scale_height_m: float = 150000.0,
                 topside_H_H_m: float = 1000000.0,
                 topside_alpha: float = 0.05,
                 topside_prior_sigma: float = 5.0):
        self.EDPSam = EDPSam
        edps = EDPSam.edps  # Original shape: (n_height, n_geo, n_sample)
        n_height, n_geo, n_sample = edps.shape
        n_state_vars     = n_height * n_geo
        n_state_vars_aug = n_state_vars + n_geo   # grid + one topside TECU per geo node

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

        # Scale to fractional perturbations if requested
        if meanscale == 1:
            safe_base = np.where(edps_base == 0, 1e-10, edps_base)
            edps_flat = edps_flat / safe_base

        # Topside prior: vertical TECU above the grid top at each geo node.
        # ne_top (m^-3) × analytic two-ion vertical integral (m) / 1e16 → TECU.
        ne_top      = edps_base.reshape(n_height, n_geo)[-1, :]   # (n_geo,) m^-3
        x_top_prior = (ne_top * ((1.0 - topside_alpha) * topside_scale_height_m
                                  + topside_alpha * topside_H_H_m) / 1e16)  # (n_geo,) TECU

        # 3. Initialize the FilterPy parent class with the augmented state size
        super().__init__(dim_x=n_state_vars_aug, dim_z=1)

        # 4. Set Initial State: anomaly from background; topside anomaly starts at zero
        self.x = np.zeros((n_state_vars_aug, 1))

        # 5. Augmented prior covariance: block-diagonal (grid block | topside block)
        P_grid = np.cov(edps_flat)                                     # (n_state_vars, n_state_vars)
        P_top  = (topside_prior_sigma ** 2) * np.eye(n_geo)           # (n_geo, n_geo)
        self.P = np.block([[P_grid,                          np.zeros((n_state_vars, n_geo))],
                            [np.zeros((n_geo, n_state_vars)), P_top]])

        # Calculate effective topside scale height
        H_eff_m = (1.0 - topside_alpha) * topside_scale_height_m + topside_alpha * topside_H_H_m

        # 6. Store metadata
        self.attrs = {
            "topside_H_eff_m":   H_eff_m,  # Added for numerical topside integration
            "meanscale":         meanscale,
            "initial_edps":      edps_flat,
            "initial_edps_mean": edps_base,        # (n_state_vars, 1) m^-3 (or fractional)
            "initial_edps_cov":  self.P.copy(),    # augmented prior covariance
            "x_top_prior":       x_top_prior,      # (n_geo,) vertical TECU background
            "n_state_vars":      n_state_vars,
            "n_geo":             n_geo,
        }
    def get_observation_operator(self, podTc2_data: dict,
                                 num_segments: int = 1000) -> np.ndarray:
        from joblib import Parallel, delayed

        altitude     = self.EDPSam.altitude
        geolocation  = self.EDPSam.geolocation
        n_height     = len(altitude)
        n_geo        = geolocation.shape[0]
        n_state_vars     = n_height * n_geo
        n_state_vars_aug = n_state_vars + n_geo

        LEO    = podTc2_data['LEO']
        GNSS   = podTc2_data['GNSS']
        n_rays = LEO.shape[1]

        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True
        )
        tree = cKDTree(geolocation) if n_geo > 1 else None
        t    = np.linspace(0, 1, num_segments)

        H_eff_m = self.attrs.get("topside_H_eff_m", 150000.0)

        print(f"  -> Building H Matrix ({n_rays} rays, parallel)...")
        rows = Parallel(n_jobs=-1)(
            delayed(_process_single_ray)(
                GNSS[:, i], LEO[:, i], t,
                altitude, n_height, n_geo, n_state_vars, n_state_vars_aug,
                geolocation, self.EDPSam.mesh, tree, transformer, H_eff_m,
            )
            for i in range(n_rays)
        )

        H = np.array(rows, dtype=np.float32)
        # Grid columns are path length (m); divide by 1e16 to express as m / TECU.
        # Topside columns are dimensionless (sec θ × bary weight) and must not be scaled.
        H[:, :n_state_vars] /= 1e16

# =============================================================================
#         # --- Visualise H ---------------------------------------------------
#         import matplotlib.pyplot as plt
#         import matplotlib.colors as mcolors
# 
#         H_grid = H[:, :n_state_vars]
#         H_top  = H[:, n_state_vars:]
#         H_2d   = H_grid.reshape(n_rays, n_height, n_geo).mean(axis=2)
# 
#         fig, axes = plt.subplots(1, 4, figsize=(22, 5))
#         fig.suptitle(
#             f"Observation Operator H  —  {n_rays} rays × {n_state_vars_aug} state vars  |  "
#             f"grid {altitude[0]:.0f}–{altitude[-1]:.0f} km + {n_geo} topside TECU nodes",
#             fontsize=10,
#         )
# 
#         # Panel 1: H_grid heatmap (ray × altitude), log scale
#         ax0 = axes[0]
#         pos_vals = H_2d[H_2d > 0]
#         if pos_vals.size:
#             im0 = ax0.imshow(
#                 H_2d, aspect='auto', origin='upper',
#                 extent=[altitude[0], altitude[-1], n_rays - 0.5, -0.5],
#                 norm=mcolors.LogNorm(
#                     vmin=float(np.percentile(pos_vals, 5)),
#                     vmax=float(pos_vals.max()),
#                 ),
#                 cmap='plasma',
#             )
#             fig.colorbar(im0, ax=ax0, label='H  (m / TECU normalisation)')
#         ax0.set_xlabel('Altitude (km)')
#         ax0.set_ylabel('Ray index')
#         ax0.set_title('H_grid — geo-averaged (log scale)')
# 
#         # Panel 2: per-ray in-grid row norm
#         ax1 = axes[1]
#         row_norms = np.linalg.norm(H_grid, axis=1)
#         ax1.plot(row_norms, np.arange(n_rays), 'o-', ms=2, lw=0.8, color='steelblue')
#         ax1.axvline(0, color='red', ls='--', lw=1.0, alpha=0.7, label='zero')
#         ax1.set_xlabel('Row ‖H_grid‖₂')
#         ax1.set_ylabel('Ray index')
#         ax1.set_title('Per-ray in-grid sensitivity\n(zero = unobservable ray)')
#         ax1.invert_yaxis()
#         ax1.legend(fontsize=8)
#         ax1.grid(True, alpha=0.3)
# 
#         # Panel 3: altitude sensitivity profile (in-grid only)
#         ax2 = axes[2]
#         col_sum = np.abs(H_grid).sum(axis=0).reshape(n_height, n_geo).mean(axis=1)
#         ax2.plot(col_sum, altitude, lw=1.5, color='darkorange')
#         ax2.set_xlabel('Σ|H_grid| across rays (geo-averaged)')
#         ax2.set_ylabel('Altitude (km)')
#         ax2.set_title('Altitude sensitivity profile\n(in-grid only)')
#         ax2.set_ylim(altitude[0], altitude[-1])
#         ax2.grid(True, alpha=0.3)
# 
#         # Panel 4: topside TECU state sensitivity per ray (≈ sec θ)
#         ax3 = axes[3]
#         top_per_ray = H_top.sum(axis=1)   # (n_rays,) effective sec(θ)
#         ax3.plot(top_per_ray, np.arange(n_rays), 'o-', ms=2, lw=0.8, color='mediumseagreen')
#         ax3.set_xlabel('Σ H_top  (sec θ × bary weights)')
#         ax3.set_ylabel('Ray index')
#         ax3.set_title('Topside TECU state sensitivity\nper ray  (≈ 1/cos θ)')
#         ax3.invert_yaxis()
#         ax3.grid(True, alpha=0.3)
# 
#         plt.tight_layout()
#         plt.show()
#         # -------------------------------------------------------------------
# =============================================================================

        return H

    def assimilate(self, obs: np.ndarray, podTc2_data: dict = None, obs_operator: np.ndarray = None,
                   relaxation: float = 1.0, relaxation_top: float = 0.99,
                   measurement_err: float = 0.0) -> np.ndarray:
        """
        Runs a single Kalman Filter assimilation step.

        relaxation     : Gauss-Markov decay for the in-grid electron density state.
        relaxation_top : Gauss-Markov decay for the topside TECU state (should be
                         close to 1.0 — plasmasphere varies slowly).
        """
        obs = np.asarray(obs).reshape(-1, 1)

        # 1. Generate or validate the H matrix
        if obs_operator is None:
            assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
            print("Dynamically calculating observation operator (H)...")
            obs_operator = self.get_observation_operator(podTc2_data)

        assert obs.shape[0] == obs_operator.shape[0], "Observation length must match H matrix rows"
        assert obs_operator.shape[1] == self.dim_x, "H matrix columns must match State vector length"

        n_obs        = obs.shape[0]
        n_state_vars = self.attrs["n_state_vars"]

        # 2. Apply Mean Scaling to the in-grid columns only.
        # The topside columns are already in TECU units and need no scaling.
        if self.attrs["meanscale"] == 1:
            H = obs_operator.copy()
            H[:, :n_state_vars] *= self.attrs["initial_edps_mean"].T
        else:
            H = obs_operator

        # 3. Gauss-Markov Predict with separate relaxation for grid and topside.
        # Build a per-state relaxation vector; outer product gives F⊗F elementwise on P.
        r_vec                = np.full(self.dim_x, relaxation)
        r_vec[n_state_vars:] = relaxation_top
        self.x = r_vec[:, None] * self.x
        r_outer = np.outer(r_vec, r_vec)
        self.P  = r_outer * self.P + (1.0 - r_outer) * self.attrs["initial_edps_cov"]

        # 4. Innovation — background includes grid mean and topside TECU prior
        x_prior_aug    = np.vstack([self.attrs["initial_edps_mean"],
                                    self.attrs["x_top_prior"][:, None]])
        background_tec = obs_operator @ x_prior_aug
        y = (obs - background_tec) - H @ self.x

        # 5. Efficient Update: O(d² × n_obs)
        PHT    = self.P @ H.T
        S      = H @ PHT
        S     += max(measurement_err, 1e-6) * np.eye(n_obs)
        K      = np.linalg.solve(S, PHT.T).T
        self.x = self.x + K @ y
        self.P = self.P - K @ PHT.T

        # 6. Split state: reconstruct grid EDP and expose topside TECU estimate
        x_grid = self.x[:n_state_vars]
        x_top  = self.x[n_state_vars:]
        # Absolute topside TECU per geo node (prior + estimated anomaly)
        self.x_top_tecu = self.attrs["x_top_prior"][:, None] + x_top  # (n_geo, 1)

        if self.attrs["meanscale"] == 1:
            analysis_x = self.attrs["initial_edps_mean"] * (1.0 + x_grid)
        else:
            analysis_x = self.attrs["initial_edps_mean"] + x_grid

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

        # Slice the grid block from the augmented P, then reshape and average over geo
        _P      = P if P is not None else self.P
        n_sv    = n_height * n_geo
        P_grid  = _P[:n_sv, :n_sv]
        P_4d    = P_grid.reshape(n_height, n_geo, n_height, n_geo)
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
