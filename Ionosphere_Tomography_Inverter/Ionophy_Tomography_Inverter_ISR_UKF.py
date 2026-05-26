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
import scipy.linalg
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    The class Ionosphere_Tomography_Inverter is a subclass of the KalmanFilter
    dedicated for ionosphere tomographic inversion from TEC measurements.
    
    UPDATED: Now implements an Iterated Square-Root Unscented Kalman Filter (ISR-UKF).
    The state vector self.x now tracks the natural logarithm of the electron density
    (ln(Ne)) to mathematically guarantee strictly positive physical electron densities.
    """

    def __init__(self, EDPSam):
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

        # 3. Transform the entire ensemble into Logarithmic Space
        edps_log = np.log(edps_flat)

        # Extract the base state (0th index) in log space
        edps_base_log = edps_log[:, 0:1]

        # 4. Initialize the FilterPy parent class
        super().__init__(dim_x=n_state_vars, dim_z=1)

        # 5. Set Initial State and Covariance in Log Space
        # Note: self.x is now the FULL log-state, not a perturbation.
        self.x = edps_base_log.copy().flatten()
        self.P = np.cov(edps_log)
        
        # Keep original linear base for reference if needed, but it's unused in the UKF math
        self.attrs = {
            "initial_edps_mean": np.exp(edps_base_log), 
            "initial_edps_log_mean": edps_base_log.flatten(),
            "initial_edps_log_cov": self.P.copy()
        }

    def get_observation_operator(self, podTc2_data: dict, num_segments: int = 1000,
                                 topside_scale_height_m: float = 150000.0,
                                 topside_n_steps: int = 10) -> np.ndarray:
        # (This function remains EXACTLY the same as your original code)
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
        t    = np.linspace(0, 1, num_segments)

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
        Diagnostic ISR-UKF assimilation step with safety checks and plotting.
        """
        obs = np.asarray(obs).reshape(-1)

        if obs_operator is None:
            assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
            print("\n[DIAGNOSTIC] Dynamically calculating observation operator (H)...")
            obs_operator = self.get_observation_operator(podTc2_data)

        assert obs.shape[0] == obs_operator.shape[0], "Observation length must match H matrix rows"
        assert obs_operator.shape[1] == self.dim_x, "H matrix columns must match State vector length"

        # Gauss-Markov Predict
        mean_log = self.attrs["initial_edps_log_mean"]
        self.x = mean_log + relaxation * (self.x - mean_log)
        self.P = (relaxation ** 2) * self.P + (1.0 - relaxation) * self.attrs["initial_edps_log_cov"]

        n_dim = len(self.x)
        n_obs = len(obs)
        max_iter = 5       
        tol = 1e-3         

        # Unscented Transform Parameters
        # For L=7000+, alpha=1.0 prevents the central weight from becoming wildly negative
        alpha = 1.0       
        beta = 0.0         
        kappa = 0.0

        lam = (alpha ** 2) * (n_dim + kappa) - n_dim
        # This is the wrong gamma, but it might be necessary to change it
        gamma = 10#np.sqrt(n_dim + lam)

        W_m = np.full(2 * n_dim + 1, 1.0 / (2.0 * (n_dim + lam)))
        W_c = np.full(2 * n_dim + 1, 1.0 / (2.0 * (n_dim + lam)))
        W_m[0] = lam / (n_dim + lam)
        W_c[0] = W_m[0] + (1.0 - alpha**2 + beta)

        x_prior = self.x.copy()
        x_iter = self.x.copy()
        R = max(measurement_err, 1e-6) * np.eye(n_obs)

        # --- DIAGNOSTICS INIT ---
        print(f"\n{'='*40}")
        print(f"ASSIMILATION DIAGNOSTICS: START")
        print(f"{'='*40}")
        print(f"State Dimension (L): {n_dim}")
        print(f"UT Gamma Factor: {gamma:.2f}")
        print(f"Initial Log-State Min/Max: {self.x.min():.2f} / {self.x.max():.2f}")
        print(f"Max Covariance (P) Variance: {np.diag(self.P).max():.4f}")
        
        # Trackers for plotting
        res_history = []
        state_diff_history = []

        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            try:
                S_x = scipy.linalg.cholesky(self.P, lower=True)
            except scipy.linalg.LinAlgError:
                print("[WARNING] P matrix lost positive-definiteness. Adding jitter.")
                self.P += 1e-6 * np.eye(n_dim)
                S_x = scipy.linalg.cholesky(self.P, lower=True)
                
            sigmas_x = np.zeros((2 * n_dim + 1, n_dim))
            sigmas_x[0] = x_iter
            for i in range(n_dim):
                sigmas_x[i + 1]         = x_iter + gamma * S_x[:, i]
                sigmas_x[n_dim + i + 1] = x_iter - gamma * S_x[:, i]

            # --- DIAGNOSTIC: Check Sigma Points ---
            max_sig = sigmas_x.max()
            min_sig = sigmas_x.min()
            print(f"Sigma Points Log-Space Spread: {min_sig:.2f} to {max_sig:.2f}")
            if max_sig > 50:
                print(f"[CRITICAL] Sigma points exceed log=50. np.exp() will likely overflow!")

            sigmas_y = np.zeros((2 * n_dim + 1, n_obs))
            for i in range(2 * n_dim + 1):
                # We use np.clip to strictly prevent np.exp from overflowing during tests
                # 80 is chosen because np.exp(80) is ~5.5e34, keeping it safely inside float64 limits
                safe_sigmas = np.clip(sigmas_x[i], -80, 80)
                physical_Ne = np.exp(safe_sigmas) 
                sigmas_y[i] = obs_operator @ physical_Ne
                
            y_hat = np.sum(W_m[:, None] * sigmas_y, axis=0)
            
            # --- DIAGNOSTIC: Check Predictions ---
            print(f"Observed TEC Min/Max: {obs.min():.2e} / {obs.max():.2e}")
            print(f"Predicted TEC Min/Max: {y_hat.min():.2e} / {y_hat.max():.2e}")

            X_diff = sigmas_x - x_prior 
            Y_diff = sigmas_y - y_hat
            
            P_xy = np.zeros((n_dim, n_obs))
            P_yy = np.zeros((n_obs, n_obs))
            
            for i in range(2 * n_dim + 1):
                P_xy += W_c[i] * np.outer(X_diff[i], Y_diff[i])
                P_yy += W_c[i] * np.outer(Y_diff[i], Y_diff[i])
                
            P_yy += R  
            
            # --- DIAGNOSTIC: Check Matrix Conditioning ---
            cond_Pyy = np.linalg.cond(P_yy)
            print(f"Condition Number of P_yy: {cond_Pyy:.2e}")
            if cond_Pyy > 1e12:
                print("[WARNING] P_yy is ill-conditioned! Kalman Gain calculation may be unstable.")

            # E. Compute Kalman Gain and Update
            K = np.linalg.solve(P_yy, P_xy.T).T
            innovation = obs - y_hat
            
            # --- NON-LINEAR TRUST REGION (Step-Size Limiter) ---
            # Standard filters take the full step K @ innovation, which will 
            # overshoot and diverge severely in exponential space.
            delta_x = K @ innovation
            
            # Cap the maximum step a single parameter can take per iteration.
            # A max_step of 0.5 restricts max physical density changes to ~64% per iteration
            max_step = 0.5
            step_magnitudes = np.abs(delta_x)
            excessive_steps = step_magnitudes > max_step
            
            if np.any(excessive_steps):
                num_capped = np.sum(excessive_steps)
                print(f"      [TRUST REGION] Capping {num_capped} aggressive parameter updates.")
                delta_x = np.clip(delta_x, -max_step, max_step)
            
            x_new = x_prior + delta_x
            
            state_diff = np.linalg.norm(x_new - x_iter)
            res_norm = np.linalg.norm(innovation)
            
            print(f"Innovation Norm: {res_norm:.4e}")
            print(f"State Update Norm: {state_diff:.4e}")
            
            res_history.append(res_norm)
            state_diff_history.append(state_diff)
            
            x_iter = x_new
            
            if state_diff < tol:
                print(f"[SUCCESS] Iteration converged at step {iteration + 1}.")
                break
        else:
            print(f"[WARNING] IUKF reached max iterations ({max_iter}) without converging.")

        # --- Plotting the Convergence ---
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(res_history) + 1), res_history, marker='o', color='red')
        plt.title('Innovation Norm (Residuals)')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(state_diff_history) + 1), state_diff_history, marker='o', color='blue')
        plt.title('State Update Norm (Convergence)')
        plt.xlabel('Iteration')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Final Update
        self.x = x_iter
        self.P = self.P - K @ P_yy @ K.T
        self.P = 0.5 * (self.P + self.P.T) 

        analysis_x = np.exp(np.clip(self.x, -80, 80))
        return analysis_x.reshape(-1, 1)
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
