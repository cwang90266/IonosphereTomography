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


def _gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """
    Gaspari-Cohn (1999) compact-support localization function.

    Smoothly decreases from 1 at r=0 to 0 at r=2; identically zero beyond r=2.
    Provides 5th-order piecewise polynomial smoothness with compact support,
    making it the standard choice for covariance localization in data assimilation.

    Parameters
    ----------
    r : ndarray
        Normalized distance  d / localization_radius.  The function is identically
        zero for r >= 2, so localization_radius acts as the half-support radius.

    Returns
    -------
    ndarray, same shape as r, values in [0, 1].
    """
    out = np.zeros_like(r, dtype=float)
    m1  = r <= 1.0
    m2  = (r > 1.0) & (r <= 2.0)
    r1, r2 = r[m1], r[m2]
    out[m1] = (
        1.0 - (5.0 / 3.0) * r1**2 + (5.0 / 8.0) * r1**3
        + (1.0 / 2.0) * r1**4 - (1.0 / 4.0) * r1**5
    )
    out[m2] = (
        4.0 - 5.0 * r2 + (5.0 / 3.0) * r2**2 + (5.0 / 8.0) * r2**3
        - (1.0 / 2.0) * r2**4 + (1.0 / 12.0) * r2**5 - 2.0 / (3.0 * r2)
    )
    return out


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
            # geolocation is stored as [lon, lat] (col 0 = lon, col 1 = lat) by all
            # EDPSamples constructors, but find_containing_triangles expects [lat, lon].
            # Swap columns here; the mesh vertex indices are unaffected.
            # The cKDTree was built on the original [lon, lat] ordering, so the
            # nearest-neighbour fallback queries must also use [lon, lat].
            geo_latlon = geolocation[:, [1, 0]]
            tri_idx, bary = find_containing_triangles(
                np.column_stack([lats_v, lons_v]), geo_latlon, mesh, return_bary=True
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
                # tree built on geolocation [lon, lat] — query must match that order
                _, near_v = tree.query(np.column_stack([lons_v[outside], lats_v[outside]]))
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
            # Map topside segments to spatial geo-nodes using barycentric coordinates.
            # Same [lon, lat] → [lat, lon] swap needed for find_containing_triangles.
            geo_latlon = geolocation[:, [1, 0]]
            tri_t, bary_t = find_containing_triangles(
                np.column_stack([lats_t, lons_t]), geo_latlon, mesh, return_bary=True
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
                # tree built on [lon, lat] — query must match
                _, near_v = tree.query(np.column_stack([lons_t[outside_t], lats_t[outside_t]]))
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
                 topside_prior_sigma: float = 5.0,
                 topside_prior_floor_tecu: float = 1.0):
        """
        Parameters
        ----------
        topside_prior_floor_tecu : float
            Minimum vertical TECU for the topside prior at every geo node.
            IRI-2020 has no plasmasphere model and clips electron density to a
            physical floor (1e8 m^-3) above ~450 km in nighttime/low-latitude
            conditions.  When this happens, the naive x_top_prior collapses to
            ~0.002 TECU, making the forward-modeled TEC essentially zero for
            high-tangent-altitude rays and preventing the Kalman filter from
            assimilating those observations.  A floor of 1.0 TECU (default)
            represents a conservative estimate of the plasmaspheric vertical TEC
            content above the grid top (~800 km) under quiet conditions.
            Set to 0.0 to disable the floor (original behavior).
        """
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

        # Apply a minimum floor to guard against IRI's lack of a plasmasphere model.
        # Without this, any geo node where IRI clips ne to the physical_floor gives
        # x_top_prior ≈ 0.002 TECU, collapsing the forward-modeled TEC to zero for
        # high-tangent-altitude rays and making those observations unassimilatable.
        if topside_prior_floor_tecu > 0.0:
            x_top_prior = np.maximum(x_top_prior, topside_prior_floor_tecu)

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
            "initial_edps_cov":        self.P.copy(),    # augmented prior covariance
            "x_top_prior":             x_top_prior,      # (n_geo,) vertical TECU background
            "topside_prior_floor_tecu": topside_prior_floor_tecu,  # floor for re-anchoring
            "n_state_vars":            n_state_vars,
            "n_geo":                   n_geo,
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

    def _build_localization_matrix(
        self,
        podTc2_data: dict,
        localization_radius_km: float,
        localization_mode: str,
    ) -> np.ndarray:
        """
        Build the (n_state_vars_aug, n_obs) distance-based localization matrix L.

        L[j, i] ∈ [0, 1] scales the influence of observation i on state variable j.
        The weight decreases monotonically with the 3-D Euclidean distance between
        the centre of voxel j and the midpoint of the GNSS–LEO ray for observation i.

        The ray midpoint is used as the representative observation location (it
        approximates the Ionospheric Pierce Point and keeps cost O(n_sv × n_obs)).
        Topside TECU nodes are placed at altitude[-1] + 0.5 * H_eff_km for the
        purpose of distance computation.

        Parameters
        ----------
        podTc2_data : dict
            Must contain 'GNSS' (3, n_obs) and 'LEO' (3, n_obs), ECEF positions in km.
        localization_radius_km : float
            Characteristic length scale (km).  Interpretation depends on mode:
            - 'gaussian':          1-σ e-folding radius
            - 'inverse_distance':  half-weight distance  (L = 1 / (1 + d/r₀))
            - 'gaspari_cohn':      half-support radius; L = 0 for d > 2 * r₀
        localization_mode : str
            One of 'gaussian', 'inverse_distance', 'gaspari_cohn'.

        Returns
        -------
        L : ndarray, float32, shape (n_state_vars_aug, n_obs)
        """
        GNSS  = podTc2_data['GNSS']   # (3, n_obs) ECEF km
        LEO   = podTc2_data['LEO']    # (3, n_obs) ECEF km
        n_obs = GNSS.shape[1]

        # Ray midpoints in ECEF km — representative location for each observation
        mid_ecef = (GNSS + LEO) / 2.0   # (3, n_obs)

        # Convert ECEF km → geodetic (lon °, lat °, alt m)
        xfm = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True,
        )
        lons_o, lats_o, alts_o_m = xfm.transform(
            mid_ecef[0, :] * 1e3, mid_ecef[1, :] * 1e3, mid_ecef[2, :] * 1e3
        )
        alts_o = alts_o_m / 1000.0   # km

        # Build voxel centre coordinates
        altitude = self.EDPSam.altitude       # (n_height,) km
        geo      = self.EDPSam.geolocation    # (n_geo, 2): col 0 = lon, col 1 = lat
        n_height = len(altitude)
        n_geo    = geo.shape[0]
        H_eff_km = self.attrs["topside_H_eff_m"] / 1000.0

        # In-grid voxels — state-vector ordering is (alt_index * n_geo + geo_index).
        # geolocation col 0 = lon, col 1 = lat (all EDPSamples constructors store [lon, lat]).
        v_lon = np.tile(geo[:, 0], n_height)    # (n_sv,)
        v_lat = np.tile(geo[:, 1], n_height)
        v_alt = np.repeat(altitude, n_geo)      # (n_sv,) km

        # Topside nodes: representative altitude = top of grid + half scale height
        t_alt = np.full(n_geo, altitude[-1] + 0.5 * H_eff_km)

        all_lat = np.concatenate([v_lat, geo[:, 1]])   # (n_sv_aug,)
        all_lon = np.concatenate([v_lon, geo[:, 0]])
        all_alt = np.concatenate([v_alt, t_alt])

        # Spherical-Earth Cartesian conversion for consistent 3-D distance (km)
        R_e = 6371.0
        def _to_xyz(lat_d, lon_d, alt_km):
            lr = np.deg2rad(lat_d)
            lo = np.deg2rad(lon_d)
            r  = R_e + alt_km
            return np.stack(
                [r * np.cos(lr) * np.cos(lo),
                 r * np.cos(lr) * np.sin(lo),
                 r * np.sin(lr)], axis=-1
            )

        v_xyz = _to_xyz(all_lat, all_lon, all_alt)   # (n_sv_aug, 3) km
        o_xyz = _to_xyz(lats_o,  lons_o,  alts_o)   # (n_obs, 3) km

        # Pairwise 3-D Euclidean distances D[j, i] in km
        diff = v_xyz[:, np.newaxis, :] - o_xyz[np.newaxis, :, :]   # (n_sv_aug, n_obs, 3)
        D    = np.sqrt((diff ** 2).sum(axis=2))                      # (n_sv_aug, n_obs)

        # Apply the chosen inverse-distance localization kernel
        r_norm = D / localization_radius_km
        if localization_mode == 'gaussian':
            # L = exp(-d² / 2r₀²)  — smooth, unbounded support
            L = np.exp(-0.5 * r_norm ** 2)
        elif localization_mode == 'inverse_distance':
            # L = 1 / (1 + d/r₀)  — heavy-tailed, never exactly zero
            L = 1.0 / (1.0 + r_norm)
        elif localization_mode == 'gaspari_cohn':
            # Compact-support polynomial; zero for d > 2*r₀ (see _gaspari_cohn)
            L = _gaspari_cohn(r_norm)
        else:
            raise ValueError(
                f"Unknown localization_mode '{localization_mode}'. "
                "Choose from: 'gaussian', 'inverse_distance', 'gaspari_cohn'."
            )

        return L.astype(np.float32)

    def assimilate(self, obs: np.ndarray, podTc2_data: dict = None, obs_operator: np.ndarray = None,
                   relaxation: float = 1.0, relaxation_top: float = 0.99,
                   measurement_err: float = 0.0,
                   # -----------------------------------------------------------------------
                   # Distance-based localization (Schur-product covariance tapering)
                   # -----------------------------------------------------------------------
                   # Each element of P @ H.T is multiplied by a weight L[j, i] ∈ [0, 1]
                   # that decreases with the distance between voxel j and the ray midpoint
                   # of observation i.  This suppresses spurious long-range Kalman updates
                   # that arise from background-covariance structure unrelated to the true
                   # spatial correlation of the ionosphere.
                   #
                   # TOGGLE:  set distance_localization=True  to enable (default: False).
                   #          Requires podTc2_data to be supplied.
                   #
                   # localization_radius_km — characteristic length scale (km):
                   #   'gaussian'        : 1-σ e-folding radius
                   #   'inverse_distance': half-weight distance  (L = 1/(1+d/r₀))
                   #   'gaspari_cohn'    : half-support radius; exactly zero for d > 2·r₀
                   # -----------------------------------------------------------------------
                   distance_localization:  bool  = False,
                   localization_radius_km: float = 1000.0,
                   localization_mode:      str   = 'gaussian') -> np.ndarray:
        """
        Runs a single Kalman Filter assimilation step.

        Parameters
        ----------
        relaxation : float
            Gauss-Markov decay for the in-grid electron density state.
        relaxation_top : float
            Gauss-Markov decay for the topside TECU state (should be close to 1.0 —
            plasmasphere varies slowly).
        distance_localization : bool
            Enable Schur-product distance-based covariance localization.  Each entry
            of P @ H.T is multiplied by L[j, i] ∈ [0, 1] where L decreases with the
            3-D Euclidean distance from voxel j to the ray midpoint of observation i.
            Reduces spurious long-range updates. Requires podTc2_data.  Default: False.
        localization_radius_km : float
            Characteristic length scale (km) for the localization kernel.  Default: 1000.
        localization_mode : str
            Localization kernel: 'gaussian', 'inverse_distance', or 'gaspari_cohn'.
            Default: 'gaussian'.
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

        # 5. Compute cross-covariance P @ H.T and optionally apply localization.
        #    When distance_localization=True, each element PHT[j, i] is multiplied
        #    by L[j, i] ∈ [0, 1] — a weight inversely related to the distance from
        #    voxel j to the ray midpoint of observation i.  Both the Kalman gain K
        #    and the covariance update use the same localized PHT for consistency.
        #    Toggle: pass distance_localization=True to assimilate() to enable.
        PHT = self.P @ H.T   # (n_state_vars_aug, n_obs)

        if distance_localization:
            if podTc2_data is None:
                raise ValueError(
                    "podTc2_data must be provided when distance_localization=True "
                    "(needed to compute ray midpoints for the localization matrix)."
                )
            L   = self._build_localization_matrix(
                podTc2_data, localization_radius_km, localization_mode
            )
            PHT = PHT * L   # Schur product: taper cross-covariances by distance

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
