#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ionosphere Tomography Inverter — Relative TEC variant.

Assimilates differential (relative) TEC instead of absolute TEC, eliminating
the requirement to approximate the topside ionosphere.

Physics
-------
For a sequence of N occultation rays each with absolute TEC measurement:
    TEC_i = H_i @ x_true  +  topside_i  +  bias      (i = 0 … N-1)

Taking the difference against a reference epoch r:
    ΔTEC_{i,r} = (H_i − H_r) @ x_true  +  (topside_i − topside_r)

Two cases that make the topside term negligible:
  1. Reference epoch above the grid (tangent alt > max(altitude_grid)):
       H_r ≈ 0  →  ΔH_i = H_i  and  ΔTEC_i ≈ H_i @ x_true (within-grid only)
  2. Consecutive or near-consecutive rays have nearly identical topside geometry:
       topside_i − topside_r ≈ 0

In either case no scale-height approximation for the topside is required.

Created: 2026
"""

import numpy as np
import pyproj
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

from EDPSamples.edp_samples import EDPSamples, interp_heights, find_containing_triangles


# ---------------------------------------------------------------------------
# Ray-path helper (no topside section)
# ---------------------------------------------------------------------------

def _process_single_ray_no_topside(
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
) -> np.ndarray:
    """
    Compute a single row of the observation operator H for one GNSS-LEO ray,
    integrating path lengths only within the modelled altitude grid.

    No topside approximation is applied; contributions above altitude[-1] are
    simply ignored because the differential TEC formulation eliminates them.
    """
    H_row = np.zeros(n_state_vars, dtype=np.float32)

    ray_points = gnss_i[:, np.newaxis] + (leo_i[:, np.newaxis] - gnss_i[:, np.newaxis]) * t
    dl_m = np.linalg.norm(np.diff(ray_points, axis=1), axis=0) * 1000.0
    midpoints = (ray_points[:, :-1] + ray_points[:, 1:]) / 2.0

    lons, lats, alts_m = transformer.transform(
        midpoints[0, :] * 1e3, midpoints[1, :] * 1e3, midpoints[2, :] * 1e3
    )
    alts_km = alts_m / 1000.0

    # Only integrate within the modelled altitude range — topside is skipped entirely.
    valid_mask = (alts_km >= altitude[0]) & (alts_km <= altitude[-1])
    if not np.any(valid_mask):
        return H_row

    dl_m_v    = dl_m[valid_mask]
    lats_v    = lats[valid_mask]
    lons_v    = lons[valid_mask]
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
            v0, v1, v2    = mesh[t_idx, 0], mesh[t_idx, 1], mesh[t_idx, 2]
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

    return H_row


# ---------------------------------------------------------------------------
# Inverter class
# ---------------------------------------------------------------------------

class Ionosphere_Tomography_Inverter_RelTEC(KalmanFilter):
    """
    EKF-based ionospheric tomography inverter using **relative (differential) TEC**.

    Assimilates  ΔTEC_i = TEC_i − TEC_ref  with differential operator
    ΔH_i = H_i − H_ref, eliminating the need to model the topside ionosphere
    (see ``compute_relative_tec`` for reference-epoch selection).

    The state is the log-ratio anomaly  x = log(ne / ne_bg), so the
    reconstructed EDP  ne = ne_bg * exp(x)  is guaranteed positive for any
    finite x.  The EKF Jacobian  H_eff = ΔH * ne_hat.T  is re-evaluated at
    the predicted state each assimilation call.
    """

    def __init__(self, EDPSam: EDPSamples, ne_floor: float = 1.0, meanscale: int = 0):
        self.EDPSam = EDPSam
        edps = EDPSam.edps  # shape: (n_height, n_geo, n_sample)
        n_height, n_geo, n_sample = edps.shape
        n_state_vars = n_height * n_geo

        edps_flat = edps.reshape(n_state_vars, n_sample)
        edps_flat = np.nan_to_num(edps_flat, nan=0.0)

        # Background: first ensemble member, floored so log is defined everywhere.
        ne_bg = np.maximum(edps_flat[:, 0:1], ne_floor)

        # Log-ratio ensemble: x_i = log(ne_i / ne_bg).
        log_ensemble = np.log(np.maximum(edps_flat, ne_floor)) - np.log(ne_bg)

        super().__init__(dim_x=n_state_vars, dim_z=1)

        self.x = np.zeros((n_state_vars, 1))
        self.P = np.cov(log_ensemble)

        self.attrs = {
            "ne_bg":             ne_bg,          # background EDP (n, 1), always > 0
            "initial_edps_mean": ne_bg,          # alias kept for backward compatibility
            "initial_edps_cov":  self.P.copy(),  # log-ratio prior covariance
        }

    # ------------------------------------------------------------------
    # Observation operator (within-grid only, no topside)
    # ------------------------------------------------------------------

    def get_observation_operator(
        self,
        podTc2_data: dict,
        num_segments: int = 1000,
    ) -> np.ndarray:
        """
        Build the H matrix by integrating path lengths within the altitude grid.

        No topside approximation is applied.  The caller is responsible for
        differencing H rows via ``compute_relative_tec`` before assimilation.

        Parameters
        ----------
        podTc2_data : dict
            Must contain keys ``'LEO'`` and ``'GNSS'`` with shapes (3, n_rays)
            in km (geocentric Cartesian).
        num_segments : int
            Number of linear segments per ray for numerical integration.

        Returns
        -------
        H : ndarray, shape (n_rays, n_state_vars), float32
            Unscaled path-length matrix.  Divide by 1e16 before use with TEC
            in TECU.
        """
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
            always_xy=True,
        )
        tree = cKDTree(geolocation) if n_geo > 1 else None
        t    = np.linspace(0, 1, num_segments)

        print(f"  -> Building H Matrix ({n_rays} rays, no topside, parallel)...")
        rows = Parallel(n_jobs=-1)(
            delayed(_process_single_ray_no_topside)(
                GNSS[:, i], LEO[:, i], t,
                altitude, n_height, n_geo, n_state_vars,
                geolocation, self.EDPSam.mesh, tree,
                transformer,
            )
            for i in range(n_rays)
        )

        H = np.array(rows, dtype=np.float32)
        H /= 1e16
        return H

    # ------------------------------------------------------------------
    # Differential TEC computation
    # ------------------------------------------------------------------

    def compute_relative_tec(
        self,
        obs: np.ndarray,
        H: np.ndarray,
        tangent_alt_km: np.ndarray | None = None,
        ref_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute differential observations and differential H matrix.

        Reference-epoch selection (in priority order):

        1. If ``ref_idx`` is provided explicitly, use it.
        2. If ``tangent_alt_km`` is supplied, select the last epoch whose
           tangent-point altitude exceeds the top of the altitude grid.
           At that epoch H_ref ≈ 0, so no within-grid path is subtracted.
        3. Fall back to epoch 0.

        Parameters
        ----------
        obs : ndarray, shape (n_rays,)
            Absolute (or phase-bias-affected) TEC in TECU.
        H : ndarray, shape (n_rays, n_state_vars)
            Observation operator from ``get_observation_operator``.
        tangent_alt_km : ndarray, shape (n_rays,), optional
            Tangent-point altitudes in km.  Used for automatic reference
            selection.
        ref_idx : int, optional
            Force a specific reference epoch index.

        Returns
        -------
        delta_obs : ndarray, shape (n_used,)
            Differential TEC values (ΔTEC = TEC_i − TEC_ref) for all epochs
            i ≠ ref_idx.
        delta_H : ndarray, shape (n_used, n_state_vars)
            Differential observation operator (ΔH = H_i − H_ref).
        ref_idx : int
            The reference epoch index that was chosen.
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        H   = np.asarray(H,   dtype=np.float64)
        n_rays = obs.shape[0]

        if ref_idx is None:
            if tangent_alt_km is not None:
                alt_top = self.EDPSam.altitude[-1]
                above   = np.where(tangent_alt_km > alt_top)[0]
                ref_idx = int(above[np.argmax(tangent_alt_km[above])]) if above.size > 0 else 0
            else:
                ref_idx = 0

        keep = np.ones(n_rays, dtype=bool)
        keep[ref_idx] = False

        delta_obs = obs[keep] - obs[ref_idx]
        delta_H   = H[keep]   - H[ref_idx]

        if tangent_alt_km is not None:
            print(f"  -> RelTEC: reference epoch {ref_idx} "
                  f"(tangent alt ≈ {tangent_alt_km[ref_idx]:.1f} km), "
                  f"{np.sum(keep)} differential observations.")
        else:
            print(f"  -> RelTEC: reference epoch {ref_idx}, "
                  f"{np.sum(keep)} differential observations.")

        return delta_obs, delta_H, ref_idx

    # ------------------------------------------------------------------
    # Assimilation
    # ------------------------------------------------------------------

    def assimilate(
        self,
        obs: np.ndarray,
        podTc2_data: dict | None = None,
        obs_operator: np.ndarray | None = None,
        tangent_alt_km: np.ndarray | None = None,
        ref_idx: int | None = None,
        relaxation: float = 1.0,
        measurement_err: float = 1.0,
        delta_obs: np.ndarray | None = None,
        delta_H: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Single EKF assimilation step using relative (differential) TEC.

        State x = log(ne / ne_bg); reconstructed as ne = ne_bg * exp(x) > 0.
        The EKF Jacobian  H_eff = ΔH * ne_hat.T  is evaluated at the current
        predicted state, so the linearisation tracks the evolving solution.

        Option A — pre-computed differential data (fastest):

            posterior = inverter.assimilate(
                obs=None, obs_operator=H,
                delta_obs=delta_tec, delta_H=delta_H_matrix,
            )

        Option B — raw absolute TEC + geometry dict (differenced internally):

            posterior = inverter.assimilate(
                obs=measured_tec, podTc2_data=podTc_clean,
                tangent_alt_km=tangent_alt_km,
            )
        """
        # ---- Option A: caller already differenced ----
        if delta_obs is not None and delta_H is not None:
            delta_obs = np.asarray(delta_obs, dtype=np.float64).reshape(-1, 1)
            delta_H   = np.asarray(delta_H,   dtype=np.float64)
        else:
            # ---- Option B: difference internally ----
            assert obs is not None, "Must supply obs when delta_obs/delta_H are not provided."
            obs = np.asarray(obs, dtype=np.float64).ravel()

            if obs_operator is None:
                assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
                print("  -> Dynamically calculating H matrix...")
                obs_operator = self.get_observation_operator(podTc2_data)

            delta_obs_1d, delta_H, _ = self.compute_relative_tec(
                obs, obs_operator, tangent_alt_km=tangent_alt_km, ref_idx=ref_idx
            )
            delta_obs = delta_obs_1d.reshape(-1, 1)

        n_diff = delta_obs.shape[0]
        assert delta_H.shape[0] == n_diff,    "delta_H row count must match delta_obs length."
        assert delta_H.shape[1] == self.dim_x, "delta_H column count must match state dimension."

        # ---- Gauss-Markov predict ----
        self.x = relaxation * self.x
        self.P = (relaxation ** 2) * self.P + (1.0 - relaxation) * self.attrs["initial_edps_cov"]

        # ---- EDP at predicted state — always positive by construction ----
        ne_hat = self.attrs["ne_bg"] * np.exp(self.x)          # (n, 1)

        # ---- EKF Jacobian: d(ΔH @ ne) / dx = ΔH * ne_hat.T ----
        H_eff = delta_H * ne_hat.T                              # (n_diff, n)

        # ---- Innovation: observed ΔTEC minus nonlinear forward model ----
        y = delta_obs - delta_H @ ne_hat                        # (n_diff, 1)

        # ---- Kalman update ----
        PHT = self.P @ H_eff.T                                  # (n, n_diff)
        S   = H_eff @ PHT + max(measurement_err, 1e-6) * np.eye(n_diff)
        K   = np.linalg.solve(S, PHT.T).T
        self.x = self.x + K @ y
        self.P = self.P - K @ PHT.T

        # ---- Reconstruct absolute EDP — guaranteed positive for any finite x ----
        return self.attrs["ne_bg"] * np.exp(self.x)
