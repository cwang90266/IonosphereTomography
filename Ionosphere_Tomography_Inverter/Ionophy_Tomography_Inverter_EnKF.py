import numpy as np
import pyproj
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

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

    # --- 2. PROCESS THE TOPSIDE: Two-Ion Plasmasphere Extension ---
    topside_mask = alts_km > altitude[-1]
    if np.any(topside_mask):
        top_indices  = np.where(topside_mask)[0]
        boundary_idx = top_indices[0]   # first midpoint above the grid top

        pierce_pt_m  = midpoints[:, boundary_idx] * 1000.0   # ECEF metres
        lat_p, lon_p = lats[boundary_idx], lons[boundary_idx]

        ray_dir      = (leo_i - gnss_i) / np.linalg.norm(leo_i - gnss_i)
        normal_0     = pierce_pt_m / np.linalg.norm(pierce_pt_m)
        
        # Prevent division by zero for perfectly horizontal rays
        cos_zenith_0 = max(np.abs(np.dot(ray_dir, normal_0)), 0.05) 

        # --- Two-Ion Model Parameters ---
        # Note: You can promote these to function arguments for tuning
        H_O_m = 150000.0       # O+ scale height (150 km)
        H_H_m = 1000000.0      # H+ scale height (1000 km, for plasmasphere)
        alpha = 0.05           # Fraction of H+ at the grid boundary (5%)
        
        dh_m = H_O_m / 2.0     # Base vertical slab thickness off the smaller scale height
        max_dist_m = np.linalg.norm(gnss_i * 1000.0 - pierce_pt_m) # Dist to GNSS

        top_a_idx = n_height - 1
        topside_contribution = 0.0
        
        k = 0
        slant_m = 0.0
        
        # Integrate until we hit the GNSS sat OR density becomes negligible
        while slant_m < max_dist_m:
            slant_m = (k + 0.5) * dh_m / cos_zenith_0
            
            if slant_m > max_dist_m:
                break # Reached the transmitter

            pos_k_m = pierce_pt_m + slant_m * ray_dir

            # Recompute local vertical
            normal_k  = pos_k_m / np.linalg.norm(pos_k_m)
            cos_zen_k = max(np.abs(np.dot(ray_dir, normal_k)), 0.05)

            dl_k_m = dh_m / cos_zen_k
            h_diff_m = (k + 0.5) * dh_m  # Vertical distance above boundary
            
            # Two-Ion Diffusive Equilibrium Weight
            weight_k = (1.0 - alpha) * np.exp(-h_diff_m / H_O_m) + \
                       (alpha) * np.exp(-h_diff_m / H_H_m)

            # Early stopping: if contribution is extremely small, truncate to save compute
            if weight_k < 1e-5:
                break

            topside_contribution += dl_k_m * weight_k
            k += 1

        # Apply the contribution to the observation operator row
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


class Ionosphere_Tomography_Inverter:
    """
    Ionosphere Tomography Inverter utilizing a Non-Linear Ensemble Kalman Filter (EnKF).
    Operates in Log-Space to guarantee strictly positive physical electron densities.
    """

    def __init__(self, EDPSam):
        self.EDPSam = EDPSam
        edps = EDPSam.edps  # Original shape: (n_height, n_geo, n_sample)
        n_height, n_geo, n_sample = edps.shape
        self.n_state_vars = n_height * n_geo
        self.n_sample = n_sample

        # 1. Flatten spatial and altitude dimensions
        edps_flat = edps.reshape(self.n_state_vars, n_sample)

        # 2. Enforce physical floor to prevent log(0)
        physical_floor = 1e7  
        edps_flat = np.nan_to_num(edps_flat, nan=physical_floor)
        edps_flat = np.clip(edps_flat, physical_floor, None)

        # 3. Transform the entire ensemble into Logarithmic Space
        # self.E is our Ensemble Matrix (State Dimension x Number of Samples)
        self.E = np.log(edps_flat)

        # Save base state for relaxation/predict steps and prior correlation plots
        self.E_prior = self.E.copy()

        # Prior physical-space mean state (mirrors attrs["initial_edps_mean"] on other filters)
        self.attrs = {
            "initial_edps_mean": np.exp(np.mean(self.E_prior, axis=1, keepdims=True)),
        }

    def get_observation_operator(self, podTc2_data: dict, num_segments: int = 1000,
                                 topside_scale_height_m: float = 150000.0,
                                 topside_n_steps: int = 10) -> np.ndarray:
        # (This function remains EXACTLY the same as your original code)
        altitude    = self.EDPSam.altitude
        geolocation = self.EDPSam.geolocation
        n_height    = len(altitude)
        n_geo       = geolocation.shape[0]

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
                altitude, n_height, n_geo, self.n_state_vars,
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
                   relaxation: float = 1.0, measurement_err: float = 2.0) -> np.ndarray:
        """
        Runs a single Non-Linear EnKF assimilation step.
        """
        obs = np.asarray(obs).reshape(-1)

        if obs_operator is None:
            assert podTc2_data is not None, "Must provide podTc2_data if obs_operator is None."
            obs_operator = self.get_observation_operator(podTc2_data)

        n_obs = len(obs)

        # 1. Gauss-Markov Predict (Relaxation back to prior)
        if relaxation < 1.0:
            self.E = self.E_prior + relaxation * (self.E - self.E_prior)

        # 2. Perturb Observations (Stochastic EnKF requirement)
        # We add random measurement noise to the observations for each ensemble member
        R_matrix = max(measurement_err, 1e-6) * np.eye(n_obs)
        pert_noise = np.random.randn(n_obs, self.n_sample) * np.sqrt(measurement_err)
        obs_ens = obs[:, None] + pert_noise

        # 3. Non-Linear Measurement Mapping
        # Convert log ensemble back to physical density to model TEC
        physical_E = np.exp(np.clip(self.E, -80, 80)) 
        Y_ens = obs_operator @ physical_E  # Shape: (n_obs, n_sample)

        # 4. Compute Ensemble Anomalies
        x_mean = np.mean(self.E, axis=1, keepdims=True)
        y_mean = np.mean(Y_ens, axis=1, keepdims=True)
        
        dX = self.E - x_mean
        dY = Y_ens - y_mean

        # 5. Compute Covariances (Efficiently, without forming 7238x7238 P)
        factor = 1.0 / (self.n_sample - 1)
        P_yy = factor * (dY @ dY.T) + R_matrix
        P_xy = factor * (dX @ dY.T)

        # 6. Compute Kalman Gain
        K = np.linalg.solve(P_yy, P_xy.T).T

        # 7. Compute Updates and Apply Trust Region
        innovations = obs_ens - Y_ens
        delta_E = K @ innovations
        
        # Prevent aggressive log-steps on any single iteration
        max_step = 1.0 
        delta_E = np.clip(delta_E, -max_step, max_step)

        # 8. Update Ensemble
        self.E = self.E + delta_E

        # 9. Return the Mean Analysis in Physical Space
        analysis_mean_log = np.mean(self.E, axis=1)
        analysis_x_physical = np.exp(np.clip(analysis_mean_log, -80, 80))

        return analysis_x_physical.reshape(-1, 1)

    def plot_covariance_correlation(self, title: str = None, P: np.ndarray = None,
                                    use_prior: bool = False) -> np.ndarray:
        """
        Plot the altitude-altitude correlation matrix.

        Computes the altitude-altitude covariance directly from the ensemble
        without forming the full (n_state_vars × n_state_vars) matrix, then
        normalises to a Pearson correlation and plots it with altitude axes.

        Parameters
        ----------
        title : str, optional
            Figure title.
        P : np.ndarray, optional
            Pre-computed (n_state_vars, n_state_vars) covariance matrix.  When
            supplied, bypasses the ensemble computation (kept for API parity with
            the other filter classes).
        use_prior : bool, optional
            When True and P is None, uses self.E_prior (before assimilation)
            instead of the current ensemble self.E.

        Returns
        -------
        corr_alt : np.ndarray, shape (n_height, n_height)
            Altitude-altitude correlation matrix.
        """
        import matplotlib.pyplot as plt
        import warnings

        altitude = self.EDPSam.altitude          # (n_height,)
        n_height = len(altitude)
        n_geo    = self.EDPSam.geolocation.shape[0]

        if P is not None:
            # Full covariance matrix passed directly — reshape and average over geo
            P_4d    = P.reshape(n_height, n_geo, n_height, n_geo)
            cov_alt = P_4d.mean(axis=(1, 3))
        else:
            # Memory-efficient: compute (n_height, n_height) cov from ensemble
            # without ever materialising the (n_state_vars, n_state_vars) matrix
            E        = self.E_prior if use_prior else self.E
            n_sample = E.shape[1]
            E_3d     = E.reshape(n_height, n_geo, n_sample)
            E_c      = E_3d - E_3d.mean(axis=2, keepdims=True)
            # cov_alt[h1,h2] = mean over geo of cov(E[h1,g,:], E[h2,g,:])
            cov_alt  = np.einsum('hgs,kgs->hk', E_c, E_c) / (n_geo * max(n_sample - 1, 1))

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
        ax.set_title(title or "Altitude-Altitude Correlation (EnKF)")
        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel("Altitude (km)")
        fig.colorbar(pcm, ax=ax, label="Correlation Coefficient")
        plt.tight_layout()

        return corr_alt