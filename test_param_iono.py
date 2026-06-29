#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
test_param_iono.py — Validation and smoothness tests for the parameterized
ionosphere forward model in enkf_update.py / observation_operator.py.

Workflow
--------
1. Scan a day directory for podTc2 occultation files; select up to N_OCC_MAX
   arcs distributed evenly across GNSS constellations (round-robin).
2. Build two Fibonacci sphere grids:
     • 1-deg spacing  — fine grid for the "truth" ensemble (9 deterministic
                        members, one per parameter perturbation).
     • 5-deg spacing  — coarser grid for the model ensemble (N_MEMBERS stochastic
                        members drawn from the IRI background covariance).
   Both grids are anchored on the full arc geometry (all decimated tangent-point
   locations) connected by MST + SLERP great-circle waypoints to guarantee
   continuous spatial coverage even when arcs are geographically separated.
3. Run the parametric forward model (ObservationOperator) for every arc on both
   grids using IDW interpolation (12 nearest neighbours).
4. Produce diagnostic figures:
     • param_iono_test_{YYYY}_{DOY}.png        — 2×2 TEC panels + globe map
     • ensemble_histograms_{YYYY}_{DOY}.png    — 8×1 per-parameter histograms
     • param_sensitivity_{YYYY}_{DOY}.png      — 8×2 signed sensitivity sweep
     • param_sensitivity_abs_{YYYY}_{DOY}.png  — 8×2 absolute sensitivity sweep
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import warnings
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import netCDF4
from scipy.spatial import cKDTree

from TEC_model.podTc_file_processing import parse_podTc2_nc_file
from Ionosphere_Tomography_Inverter.ionospheric_state import (
    IonosphericState, N_STATE, PARAM_NAMES, LOG_INDICES,
)
from Ionosphere_Tomography_Inverter.observation_operator import ObservationOperator
from Ionosphere_Tomography_Inverter.enkf_update import (
    _haversine_km, ParametricEnKF, build_ray_localisation_matrix,
)
from demo_compare_kf_enkf import (
    _build_gnss_to_leo_ray, _tangent_latlon_single, _solar_sampling_df,
    _get_iri_edp_and_features_batch, _state_from_iri_direct,
    _default_background_covariance, _TRANSFORMER,
    _parametric_to_edp, _parametric_to_edp_ensemble,
    _precompute_ray_phi, _plot_arc_innovation_diagnostic,
    _idw_weights as _idw_weights_enkf,
)
from demo_group import CONSTELLATION_CONFIG, _CONST_FALLBACK_CMAP


# ─────────────────────────────────────────────────────────────────────────────
# §0  Configuration
# ─────────────────────────────────────────────────────────────────────────────

DOY           = 154
YYYY          = 2025
BASE_PATH     = (f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
                 f"{YYYY}.{DOY}/")
SAVE_DIR      = "./Figures/test_param_iono/"
IRI_CACHE_DIR = "./Data/IRI_param_cache/"

ALT_MIN_TEC   = 100.0   # km — integration band for measured TEC (lower bound)
ALT_MAX_TEC   = 400.0   # km — integration band for measured TEC (upper bound)

N_OCC_MAX     = 8      # maximum number of occultations to process

# Altitude grid for Ne integration (log-spaced 60–800 km)
ALT_GRID = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)

GRID_MARGIN_DEG = 15.0   # degrees of margin added around arc tangent tracks
CORR_LENGTH_KM  = 500.0  # exponential spatial correlation length (km)

N_MEMBERS  = 200   # stochastic model ensemble size
MAX_EPOCHS = 200   # maximum ray epochs per arc (decimated if more)

# Per-parameter 1-σ perturbation for the truth ensemble members
# log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE
_TRUTH_SIGMA = np.array([0.5, 38.0, 33.0, 0.2, 23.0, 0.30, 0.4, 8.0])


# IDW interpolation settings (applied identically to both 1-deg and 5-deg grids)
_IDW_POWER   = 2.0
_IDW_NEAREST = 12

# Parameter sensitivity sweep
_N_SWEEP = 61      # number of sweep points (odd → symmetric about baseline)
_N_SIGMA = 5.0     # sweep range in units of _TRUTH_SIGMA

# Constellation panel layout (matches demo_group.CONSTELLATION_CONFIG)
_CONST_POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}

# ── §7  EnKF retrieval experiment ─────────────────────────────────────────────
# Truth ionosphere is IRI evaluated +TRUTH_HOUR_OFFSET hours after the arc
# representative time, with F10.7 increased by TRUTH_F107_DELTA solar-flux units.
TRUTH_HOUR_OFFSET    = 1       # hours added to mean time for truth ionosphere
# A +10 F10.7 increment is a realistic active-to-moderate solar-activity step
# that gives ~15–30 TECU innovations — large enough to test the filter but
# small enough to stay within the stochastic EnKF's linear regime.
TRUTH_F107_DELTA     = 10.0    # solar flux unit increment for truth conditions

ENKF_N_MEMBERS       = 500     # EnKF ensemble size (model prior)
ENKF_LOC_RADIUS_KM   = 200.0   # Gaspari-Cohn half-support radius (km)
# sigma_obs must be large enough that R_mda = n_mda * sigma^2 keeps the
# Kalman gain moderate (members don't saturate physical bounds after step 1).
ENKF_SIGMA_OBS       = 1.0     # observation noise std-dev (TECU)
ENKF_INFLATION       = 1.0     # multiplicative prior-ensemble inflation
ENKF_N_MDA           = 1       # ES-MDA iterations (1 = standard single-step EnKF)
# Keep update rays << n_members to avoid rank-deficiency in D = P_yy + R.
# ~10 levels per arc gives ~80 total for 8 arcs, well within the 200-member rank.
ENKF_MAX_UPDATE_RAYS = 500      # maximum rays per arc used in the EnKF update
ENKF_MAX_UPDATE_STEP = 1.0     # per-element log-space update clip

# Altitudes (km) for the 5×2 EDP spatial-error orthographic plots
ERROR_ALTITUDES      = [100, 200, 300, 400, 500]


# ─────────────────────────────────────────────────────────────────────────────
# §1  File scanning and selection
# ─────────────────────────────────────────────────────────────────────────────

def scan_and_select_files(
    base_path: str,
    alt_min: float        = ALT_MIN_TEC,
    alt_max: float        = ALT_MAX_TEC,
    alt_min_tangent: float = 90.0,
    max_files: int        = N_OCC_MAX,
    file_suffix: str      = ".0001_nc",
    time_window_min: int  = 30,
) -> list[dict]:
    """
    Scan *base_path* for podTc2 netCDF files, filter to those whose tangent
    altitude range overlaps [alt_min, alt_max] km AND whose full arc probes
    below *alt_min_tangent* km (default 90 km), group by 30-minute time
    window, pick the window with the most occultations, then select up to
    *max_files* records distributed evenly across GNSS constellations
    (round-robin by constellation letter G/R/E/C).

    Parameters
    ----------
    alt_min_tangent : float
        Require the arc's minimum tangent altitude to be below this value (km).
        Ensures the ray path probes deep enough into the lower ionosphere /
        upper mesosphere.  Default 90 km.

    Returns
    -------
    selected : list of dicts with keys
        path, leo_id, prn_id, conid, lat, lon, slta, min_tang_km, date
    """
    records = []
    for fname in sorted(os.listdir(base_path)):
        if not fname.endswith(file_suffix):
            continue
        fpath = os.path.join(base_path, fname)
        try:
            ds = netCDF4.Dataset(fpath, "r")
            # Global attributes
            slta_tec = float(ds.getncattr("slta_tecmax_tangent"))
            lat_tec  = float(ds.getncattr("lat_tecmax_tangent"))
            lon_tec  = float(ds.getncattr("lon_tecmax_tangent"))
            yr       = int(ds.getncattr("year"))
            mo       = int(ds.getncattr("month"))
            dy       = int(ds.getncattr("day"))
            hh       = int(ds.getncattr("hour"))
            mm       = int(ds.getncattr("minute"))
            ss       = int(ds.getncattr("second"))
            prn_id   = str(ds.getncattr("prn_id"))
            leo_id   = str(ds.getncattr("leo_id"))
            conid    = str(ds.getncattr("conid"))

            # Time and TEC variables — used to find the TEC-max epoch datetime
            time_arr = np.asarray(ds.variables["time"][:],  dtype=float)
            tec_arr  = np.asarray(ds.variables["TEC"][:],   dtype=float)

            # Position arrays (km) — used to compute minimum tangent altitude
            x_leo = np.asarray(ds.variables["x_LEO"][:], dtype=float)
            y_leo = np.asarray(ds.variables["y_LEO"][:], dtype=float)
            z_leo = np.asarray(ds.variables["z_LEO"][:], dtype=float)
            x_gps = np.asarray(ds.variables["x_GPS"][:], dtype=float)
            y_gps = np.asarray(ds.variables["y_GPS"][:], dtype=float)
            z_gps = np.asarray(ds.variables["z_GPS"][:], dtype=float)
            ds.close()
        except Exception as exc:
            warnings.warn(f"Could not read {fpath}: {exc}")
            continue

        # slta_tecmax_tangent may be in metres — convert if needed
        slta_km = slta_tec / 1000.0 if slta_tec > 1000.0 else slta_tec

        # Constellation letter
        conid = conid.upper()
        if conid not in "GREC":
            conid = prn_id[0].upper() if prn_id[0].upper() in "GREC" else "G"

        # Filter 1: TEC-max tangent altitude within the integration band
        if not (alt_min <= slta_km <= alt_max):
            continue

        # Filter 2: arc must probe below alt_min_tangent km.
        # Compute per-epoch tangent altitudes with a spherical approximation
        # (R_E = 6371 km) — accurate to <1 km for this screening purpose.
        leo  = np.array([x_leo, y_leo, z_leo])   # (3, n)
        gnss = np.array([x_gps, y_gps, z_gps])   # (3, n)
        v    = leo - gnss
        denom = np.einsum("ij,ij->j", v, v)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        t_s   = np.clip(-np.einsum("ij,ij->j", gnss, v) / denom, 0.0, 1.0)
        tang_pt     = gnss + v * t_s[np.newaxis, :]
        tang_alt_km = np.linalg.norm(tang_pt, axis=0) - 6371.0
        min_tang_km = float(np.min(tang_alt_km))

        if min_tang_km > alt_min_tangent:
            continue

        # File start datetime (from year/month/day/hour/minute/second attributes)
        start_dt = pd.Timestamp(yr, mo, dy, hh, mm, ss)

        # TEC-max epoch datetime.
        # time_arr uses GPS seconds (epoch 1980-01-06), not Unix time, so we
        # derive the TEC-max time as an elapsed offset from the file start
        # rather than converting the absolute GPS timestamp.
        tec_max_idx    = int(np.argmax(tec_arr))
        tec_max_offset = float(time_arr[tec_max_idx] - time_arr[0])
        tec_max_dt     = start_dt + pd.Timedelta(seconds=tec_max_offset)

        records.append(dict(
            path        = fpath,
            leo_id      = leo_id,
            prn_id      = prn_id,
            conid       = conid,
            lat         = lat_tec,
            lon         = lon_tec,
            slta        = slta_km,
            min_tang_km = min_tang_km,
            date        = start_dt,      # file start time
            tec_max_dt  = tec_max_dt,    # datetime of TEC maximum within arc
        ))

    if not records:
        raise RuntimeError(f"No qualifying podTc2 files found in {base_path}")

    # Group by 30-minute window keyed on the TEC-max epoch datetime, so that
    # only arcs whose ionospheric peak occurs within the same half-hour window
    # are grouped together — much tighter colocation than using the file start.
    def _window_key(r):
        t = r["tec_max_dt"]
        slot = (t.hour * 60 + t.minute) // time_window_min
        return f"{t.date()}_{slot:04d}"

    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[_window_key(r)].append(r)

    best_key = max(groups, key=lambda k: len(groups[k]))
    tec_times = sorted(r["tec_max_dt"] for r in groups[best_key])
    span_min  = (tec_times[-1] - tec_times[0]).total_seconds() / 60.0
    print(f"  Best window: {best_key}  "
          f"({len(groups[best_key])} occultations, "
          f"TEC-max span {span_min:.1f} min)")

    # Round-robin constellation selection
    const_pool: dict[str, list] = defaultdict(list)
    for r in groups[best_key]:
        const_pool[r["conid"]].append(r)

    queues = [list(const_pool[c])
              for c in sorted(const_pool, key=lambda c: -len(const_pool[c]))]
    selected: list = []
    while len(selected) < max_files and any(queues):
        for q in queues:
            if q and len(selected) < max_files:
                selected.append(q.pop(0))
        queues = [q for q in queues if q]

    # Print constellation breakdown
    breakdown: dict[str, int] = defaultdict(int)
    for r in selected:
        breakdown[r["conid"]] += 1
    print(f"  Selected {len(selected)} occultations: "
          + "  ".join(f"{c}:{n}" for c, n in sorted(breakdown.items())))

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# §2  Fibonacci sphere grid construction
# ─────────────────────────────────────────────────────────────────────────────

def _fibonacci_sphere_latlons(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate *n_points* approximately uniformly distributed lat/lon positions
    using the golden-ratio Fibonacci sphere method.

    Avoids polar over-sampling and longitude wrap-around artefacts that arise
    from rectangular lat/lon grids.
    """
    phi  = (1.0 + np.sqrt(5.0)) / 2.0
    i    = np.arange(n_points, dtype=float)
    lats = np.degrees(np.arcsin(1.0 - 2.0 * (i + 0.5) / n_points))
    lons = (np.degrees(2.0 * np.pi * i / phi) + 180.0) % 360.0 - 180.0
    return lats, lons


def _mst_edges(lats: np.ndarray, lons: np.ndarray) -> list[tuple[int, int]]:
    """
    Compute a Minimum Spanning Tree over a set of geographic points using
    Prim's algorithm with haversine distances.

    Returns
    -------
    edges : list of (i, j) index pairs forming the MST.
    """
    n = len(lats)
    if n <= 1:
        return []

    # Full pairwise haversine distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        dist[i] = _haversine_km(lats[i], lons[i], lats, lons)

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    edges: list[tuple[int, int]] = []

    for _ in range(n - 1):
        in_idx  = np.where(in_tree)[0]
        out_idx = np.where(~in_tree)[0]
        d_sub   = dist[np.ix_(in_idx, out_idx)]
        flat    = int(np.argmin(d_sub))
        r_best, c_best = np.unravel_index(flat, d_sub.shape)
        i = int(in_idx[r_best])
        j = int(out_idx[c_best])
        in_tree[j] = True
        edges.append((i, j))

    return edges


def _great_circle_waypoints(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    step_deg: float,
) -> list[tuple[float, float]]:
    """
    Return intermediate (lat, lon) waypoints along the great circle from
    (lat1, lon1) to (lat2, lon2) at approximately *step_deg* intervals.

    Endpoints are excluded.  Uses SLERP interpolation in 3-D Cartesian space.
    """
    def _v(lat_d: float, lon_d: float) -> np.ndarray:
        lr, lo = np.radians(lat_d), np.radians(lon_d)
        return np.array([np.cos(lr) * np.cos(lo),
                         np.cos(lr) * np.sin(lo),
                         np.sin(lr)])

    v1, v2   = _v(lat1, lon1), _v(lat2, lon2)
    cos_th   = float(np.clip(v1 @ v2, -1.0, 1.0))
    theta_deg = np.degrees(np.arccos(cos_th))

    if theta_deg < step_deg:
        return []

    n_seg  = int(np.ceil(theta_deg / step_deg))
    sin_th = np.sin(np.radians(theta_deg))
    waypoints = []
    for k in range(1, n_seg):
        t = k / n_seg
        p = (np.sin((1 - t) * np.radians(theta_deg)) * v1
             + np.sin(t * np.radians(theta_deg)) * v2) / sin_th
        waypoints.append((
            float(np.degrees(np.arcsin(np.clip(p[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(p[1], p[0]))),
        ))
    return waypoints


def _make_fibonacci_grid(
    tp_lats: np.ndarray,
    tp_lons: np.ndarray,
    spacing_deg: float,
    margin_deg: float  = GRID_MARGIN_DEG,
    min_points: int    = _IDW_NEAREST + 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a Fibonacci sphere sub-grid that covers the geometry of all
    occultation arc tangent points.

    Anchor set = tangent points  +  MST great-circle waypoints (at
    spacing_deg/2 intervals) to guarantee connectivity between arcs that
    may be geographically separated.

    Parameters
    ----------
    tp_lats, tp_lons : anchor positions (all decimated tangent-point locations
                       across all arcs and epochs).
    spacing_deg      : target inter-node spacing (degrees).
    margin_deg       : angular margin around each anchor.
    min_points       : minimum number of nodes to keep (expanded if needed).

    Returns
    -------
    lats, lons : (n_grid,) selected Fibonacci node positions.
    """
    tp_lats = np.asarray(tp_lats)
    tp_lons = np.asarray(tp_lons)

    # Total nodes on the full sphere for this spacing
    n_total = max(int(np.round(4.0 * np.pi / np.radians(spacing_deg) ** 2)), 12)
    fib_lats, fib_lons = _fibonacci_sphere_latlons(n_total)

    # Build anchor set: tangent points + MST SLERP waypoints
    anchor_lats = tp_lats.tolist()
    anchor_lons = tp_lons.tolist()
    step = spacing_deg / 2.0
    for i, j in _mst_edges(tp_lats, tp_lons):
        for wp_lat, wp_lon in _great_circle_waypoints(
                tp_lats[i], tp_lons[i], tp_lats[j], tp_lons[j], step):
            anchor_lats.append(wp_lat)
            anchor_lons.append(wp_lon)
    anchor_lats = np.array(anchor_lats)
    anchor_lons = np.array(anchor_lons)

    # Convert to unit 3-D Cartesian for chord-distance search
    def _xyz(lat_d: np.ndarray, lon_d: np.ndarray) -> np.ndarray:
        lr = np.radians(lat_d)
        ln = np.radians(lon_d)
        return np.column_stack([np.cos(lr) * np.cos(ln),
                                 np.cos(lr) * np.sin(ln),
                                 np.sin(lr)])

    fib_xyz    = _xyz(fib_lats, fib_lons)
    anchor_xyz = _xyz(anchor_lats, anchor_lons)
    tree       = cKDTree(fib_xyz)

    r: float          = float(margin_deg)
    indices: set[int] = set()
    while True:
        chord = 2.0 * np.sin(np.radians(r) / 2.0)
        for xyz in anchor_xyz:
            indices.update(tree.query_ball_point(xyz, r=chord))
        if len(indices) >= min_points or r >= 180.0:
            break
        r *= 1.5

    if r > margin_deg:
        print(f"    [info] Fibonacci grid margin expanded "
              f"{margin_deg:.1f}° → {r:.1f}° ({len(indices)} points)")

    idx_arr = np.array(sorted(indices))
    return fib_lats[idx_arr], fib_lons[idx_arr]


# ─────────────────────────────────────────────────────────────────────────────
# §3  IRI state grid construction
# ─────────────────────────────────────────────────────────────────────────────

def _exp_spatial_corr(
    lats: np.ndarray,
    lons: np.ndarray,
    corr_length_km: float = CORR_LENGTH_KM,
) -> np.ndarray:
    """
    Build an exponential spatial correlation matrix for *n* grid points.

    C[i,j] = exp(-d_km[i,j] / corr_length_km)

    Normalised to unit diagonal (correlation matrix, not covariance).
    A small nugget (1e-6 × I) is added for numerical conditioning.
    """
    n = len(lats)
    C = np.empty((n, n))
    for i in range(n):
        d    = _haversine_km(lats[i], lons[i], lats, lons)
        C[i] = np.exp(-d / corr_length_km)
    C = 0.5 * (C + C.T)
    C += 1e-6 * np.eye(n)
    std = np.sqrt(np.diag(C))
    C /= np.outer(std, std)
    return C


def build_iri_state_grid(
    time_dt: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    alt_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Call IRI2020 for every grid point and assemble the 8-parameter state matrix.

    Parameters
    ----------
    time_dt  : observation time.
    lats, lons : (n_grid,) grid-point coordinates.
    alt_grid : (n_alt,) altitude levels (km).

    Returns
    -------
    mean_state  : (N_STATE, n_grid) IRI background mean in log/km/dim-less units.
    ne_profiles : (n_alt, n_grid)   IRI electron density profiles (m⁻³).
    """
    n_grid     = len(lats)
    sampling_df = _solar_sampling_df(time_dt)
    ne_profiles, feature_vecs = _get_iri_edp_and_features_batch(
        time_dt, lats, lons, alt_grid, sampling_df=sampling_df,
    )
    mean_state = np.empty((N_STATE, n_grid))
    for g in range(n_grid):
        mean_state[:, g] = _state_from_iri_direct(
            ne_profiles[:, g], feature_vecs[:, g], alt_grid,
        )
    return mean_state, ne_profiles


def _iri_cache_path(
    time_dt: pd.Timestamp,
    spacing_deg: float,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str = IRI_CACHE_DIR,
) -> str:
    """Return the .npz cache file path for a given IRI grid run."""
    fname = (
        f"iri_{time_dt.year}_{time_dt.dayofyear:03d}_"
        f"{time_dt.hour:02d}{time_dt.minute:02d}_"
        f"{spacing_deg:.1f}deg_"
        f"lat{lat_min:.1f}_{lat_max:.1f}_"
        f"lon{lon_min:.1f}_{lon_max:.1f}.npz"
    )
    return os.path.join(cache_dir, fname)


def build_iri_state_grid_cached(
    time_dt: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    alt_grid: np.ndarray,
    spacing_deg: float,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str = IRI_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper around build_iri_state_grid with .npz caching.

    Cache key includes time, spacing, and actual bounding box of the Fibonacci
    grid so a changed grid layout invalidates the cache automatically.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _iri_cache_path(
        time_dt, spacing_deg, lat_min, lat_max, lon_min, lon_max, cache_dir,
    )

    if os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            # Validate alt_grid match
            if (data["alt_grid"].shape == alt_grid.shape
                    and np.allclose(data["alt_grid"], alt_grid, rtol=1e-5)):
                print(f"  [cache hit]  {os.path.basename(cache_path)}")
                return data["mean_state"], data["ne_profiles"]
            else:
                print(f"  [cache miss] alt_grid mismatch — rebuilding.")
        except Exception as exc:
            print(f"  [cache miss] Could not load {cache_path}: {exc}")

    print(f"  Building IRI grid  ({len(lats)} pts, {spacing_deg:.1f}°)…")
    mean_state, ne_profiles = build_iri_state_grid(time_dt, lats, lons, alt_grid)
    np.savez(
        cache_path,
        mean_state  = mean_state,
        ne_profiles = ne_profiles,
        lats        = lats,
        lons        = lons,
        alt_grid    = alt_grid,
    )
    print(f"  [cache write] {os.path.basename(cache_path)}")
    return mean_state, ne_profiles


def build_truth_state(
    mean_state: np.ndarray,
    truth_sigma: np.ndarray = _TRUTH_SIGMA,
) -> IonosphericState:
    """
    Build 9 deterministic ensemble members representing the "truth":

      member 0        : IRI baseline (no perturbation)
      member k+1      : baseline with parameter k shifted by +truth_sigma[k]

    Parameters
    ----------
    mean_state   : (N_STATE, n_grid)  IRI background mean.
    truth_sigma  : (N_STATE,)         per-parameter 1-σ offsets.

    Returns
    -------
    IonosphericState with ensemble clamped to physical bounds.
    """
    n_grid    = mean_state.shape[1]
    n_members = N_STATE + 1   # member 0 + one per parameter

    ensemble = np.broadcast_to(
        mean_state[:, :, np.newaxis],
        (N_STATE, n_grid, n_members),
    ).copy()

    for k in range(N_STATE):
        ensemble[k, :, k + 1] += truth_sigma[k]

    state = IonosphericState(n_grid_points=n_grid, n_members=n_members)
    state.ensemble = ensemble
    state.clamp_to_physical_bounds()
    return state


def build_model_ensemble(
    mean_state: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    n_members: int  = N_MEMBERS,
    corr_length_km: float = CORR_LENGTH_KM,
) -> IonosphericState:
    """
    Draw a stochastic ensemble from the IRI background distribution using the
    Kronecker-structured spatial covariance (param_cov ⊗ spatial_corr).

    Parameters
    ----------
    mean_state     : (N_STATE, n_grid)  IRI background mean.
    lats, lons     : (n_grid,) grid coordinates for the spatial correlation.
    n_members      : ensemble size.
    corr_length_km : exponential spatial correlation length.

    Returns
    -------
    IonosphericState with ensemble clamped to physical bounds.
    """
    n_grid      = mean_state.shape[1]
    param_cov   = _default_background_covariance()
    spatial_corr = _exp_spatial_corr(lats, lons, corr_length_km)

    state = IonosphericState(n_grid_points=n_grid, n_members=n_members)
    try:
        state.generate_ensemble_spatial(mean_state, param_cov, spatial_corr,
                                         n_members=n_members)
    except np.linalg.LinAlgError:
        warnings.warn("generate_ensemble_spatial failed; falling back to "
                      "independent (non-spatial) sampling.")
        state.generate_ensemble(mean_state, param_cov, n_members=n_members)

    # Jensen bias correction for log-space parameters.
    #
    # Drawing ε ~ N(0, σ²) and storing θ = μ + ε means:
    #   E[10^θ] = 10^μ · exp(½ σ² ln²10)  >  10^μ
    # The ensemble-mean Ne is systematically above the IRI baseline Ne, which
    # biases innovations negative and drives the filter in the wrong direction.
    # Subtracting σ²·ln(10)/2 from every log-space member shifts the ensemble
    # mean in log-space so that E[10^θ] = 10^μ exactly.
    log_var = np.diag(param_cov)[LOG_INDICES]          # variance per log param
    bias    = log_var * np.log(10) / 2                 # log₁₀ units
    state.ensemble[LOG_INDICES] -= bias[:, np.newaxis, np.newaxis]

    # Pin member 0 to the unperturbed IRI baseline so callers can use
    # ensemble[:, :, 0] as a deterministic prior mean.
    state.ensemble[:, :, 0] = mean_state
    state.clamp_to_physical_bounds()
    return state


# ─────────────────────────────────────────────────────────────────────────────
# §4  Forward model
# ─────────────────────────────────────────────────────────────────────────────

def _arc_tangent_tracks(
    parsed_list: list[dict],
    max_epochs: int = MAX_EPOCHS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the full set of decimated tangent-point (lat, lon) locations across
    all arcs and epochs.

    Used to anchor the Fibonacci grids so that both ends of each arc (including
    high-altitude tangent points) are included in the coverage region.

    This function is lightweight — it calls _tangent_latlon_single per epoch
    but does NOT build ray objects.

    Parameters
    ----------
    parsed_list : list of arc dicts from parse_podTc2_nc_file.
    max_epochs  : maximum epochs to retain per arc (same decimation as
                  _build_arc_rays to keep the grid consistent).

    Returns
    -------
    all_lats, all_lons : flat arrays of tangent-point coordinates.
    """
    all_lats: list[float] = []
    all_lons: list[float] = []

    for arc in parsed_list:
        leo  = arc["LEO"]    # (3, n_epochs) ECEF km
        gnss = arc["GNSS"]   # (3, n_epochs) ECEF km
        tang_alts = np.asarray(arc["tangent_alt_km"])
        n_eps = leo.shape[1]

        # Decimate in the same way as _build_arc_rays
        if n_eps > max_epochs:
            stride  = int(np.ceil(n_eps / max_epochs))
            idx     = np.arange(0, n_eps, stride)
            deepest = int(np.argmin(tang_alts))
            if deepest not in idx:
                idx = np.sort(np.append(idx, deepest))
        else:
            idx = np.arange(n_eps)

        for i in idx:
            lat, lon = _tangent_latlon_single(gnss[:, i], leo[:, i])
            all_lats.append(float(lat))
            all_lons.append(float(lon))

    return np.array(all_lats), np.array(all_lons)


def _idw_weights(
    tp_lat: float,
    tp_lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    power: float = _IDW_POWER,
    n_nearest: int = _IDW_NEAREST,
) -> np.ndarray:
    """
    Compute IDW weights from tangent point (tp_lat, tp_lon) to the *n_nearest*
    grid nodes.  Returns a full-length weight vector (zeros for non-neighbours).
    """
    d_km = _haversine_km(tp_lat, tp_lon, grid_lats, grid_lons)
    d_km = np.maximum(d_km, 0.01)
    idx  = np.argsort(d_km)[:n_nearest]
    w    = np.zeros(len(grid_lats))
    w[idx] = 1.0 / (d_km[idx] ** power)
    w /= w.sum()
    return w


def _build_arc_rays(
    arc: dict,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the ray trajectory objects and per-epoch metadata for one arc.

    Parameters
    ----------
    arc        : parsed arc dict from parse_podTc2_nc_file.
    max_epochs : maximum epochs to retain (decimated if more).

    Returns
    -------
    rays      : list of (n_iono, 3) ray arrays [lat, lon, alt_km].
    tp_lats   : (n_ep,) tangent-point latitudes.
    tp_lons   : (n_ep,) tangent-point longitudes.
    tang_alts : (n_ep,) tangent-point altitudes (km).
    tec_meas  : (n_ep,) measured podTc2 sTEC (TECU) — raw, not band-limited.
    """
    leo       = arc["LEO"]    # (3, n_epochs) ECEF km
    gnss      = arc["GNSS"]   # (3, n_epochs) ECEF km
    tang_alts_full = np.asarray(arc["tangent_alt_km"])
    tec_full  = np.asarray(arc.get("TEC_podTc2", np.full(leo.shape[1], np.nan)))
    n_eps     = leo.shape[1]

    # Decimation
    if n_eps > max_epochs:
        stride  = int(np.ceil(n_eps / max_epochs))
        idx     = np.arange(0, n_eps, stride)
        deepest = int(np.argmin(tang_alts_full))
        if deepest not in idx:
            idx = np.sort(np.append(idx, deepest))
    else:
        idx = np.arange(n_eps)

    rays: list     = []
    tp_lats: list  = []
    tp_lons: list  = []
    tang_list: list = []
    tec_list: list  = []

    for i in idx:
        ray = _build_gnss_to_leo_ray(gnss[:, i], leo[:, i])
        rays.append(ray)
        lat, lon = _tangent_latlon_single(gnss[:, i], leo[:, i])
        tp_lats.append(float(lat))
        tp_lons.append(float(lon))
        tang_list.append(float(tang_alts_full[i]))
        tec_list.append(float(tec_full[i]))

    return (rays,
            np.array(tp_lats),
            np.array(tp_lons),
            np.array(tang_list),
            np.array(tec_list))


def forward_model_arc(
    rays: list,
    tp_lats: np.ndarray,
    tp_lons: np.ndarray,
    obs_op: ObservationOperator,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    use_idw: bool = True,
) -> np.ndarray:
    """
    Compute the forward-modelled sTEC ensemble for every epoch of one arc.

    Parameters
    ----------
    rays         : list of ray arrays (length n_ep).
    tp_lats/lons : tangent-point coordinates (n_ep,).
    obs_op       : ObservationOperator bound to the appropriate state.
    grid_lats/lons : (n_grid,) grid coordinates.
    use_idw      : if True use IDW weights; if False use nearest-neighbour.

    Returns
    -------
    Y_f : (n_ep, n_members) simulated sTEC ensemble (TECU).
    """
    n_ep = len(rays)

    if use_idw:
        # Build weight matrix (n_ep, n_grid) — one IDW row per tangent point
        W = np.stack([
            _idw_weights(tp_lats[k], tp_lons[k], grid_lats, grid_lons)
            for k in range(n_ep)
        ])  # (n_ep, n_grid)
        Y_f = obs_op.compute_stec_ensemble(rays, grid_point_weights=W)
    else:
        tree = cKDTree(np.column_stack([grid_lats, grid_lons]))
        _, gpi = tree.query(np.column_stack([tp_lats, tp_lons]))
        gpi = gpi.ravel().astype(int)
        Y_f = obs_op.compute_stec_ensemble(rays, grid_point_indices=gpi)

    return Y_f


def run_forward_models(
    parsed_list: list[dict],
    truth_state: IonosphericState,
    model_state: IonosphericState,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """
    Run the parametric forward model for every arc on both the truth (1-deg)
    and model (5-deg) grids.

    Returns
    -------
    truth_arcs : list of dicts per arc with keys:
        tangent_km, tec_all (n_ep, 9), tec_measured (n_ep,),
        tp_lats, tp_lons, conid, prn_id, leo_id
    model_arcs : list of dicts per arc with keys:
        tangent_km, tec_mean (n_ep,), tec_std (n_ep,),
        tp_lats, tp_lons, conid, prn_id, leo_id
    """
    obs_truth = ObservationOperator(truth_state, alt_grid)
    obs_model = ObservationOperator(model_state, alt_grid)

    truth_arcs: list[dict] = []
    model_arcs: list[dict] = []

    for arc_idx, arc in enumerate(parsed_list):
        prn_id = str(arc.get("prn_id", arc.get("prn", "?")))
        leo_id = str(arc.get("leo_id", "?"))
        conid  = str(arc.get("conid", prn_id[0].upper()
                              if prn_id[0].upper() in "GREC" else "?")).upper()

        print(f"  Arc {arc_idx+1}/{len(parsed_list)}: "
              f"{leo_id} {conid}{prn_id} …", end=" ", flush=True)

        rays, tp_lats, tp_lons, tang_km, tec_meas = _build_arc_rays(arc)

        # Truth forward model on 1-deg grid (IDW)
        Y_truth = forward_model_arc(
            rays, tp_lats, tp_lons, obs_truth,
            grid_lats_1deg, grid_lons_1deg, use_idw=True,
        )  # (n_ep, 9)

        # Model forward model on 5-deg grid (IDW)
        Y_model = forward_model_arc(
            rays, tp_lats, tp_lons, obs_model,
            grid_lats_5deg, grid_lons_5deg, use_idw=True,
        )  # (n_ep, N_MEMBERS)

        print(f"done  ({len(rays)} epochs)")

        truth_arcs.append(dict(
            tangent_km   = tang_km,
            tec_all      = Y_truth,           # (n_ep, 9)
            tec_measured = tec_meas,          # (n_ep,)  measured podTc2
            tp_lats      = tp_lats,
            tp_lons      = tp_lons,
            conid        = conid,
            prn_id       = prn_id,
            leo_id       = leo_id,
        ))
        model_arcs.append(dict(
            tangent_km = tang_km,
            tec_mean   = Y_model.mean(axis=1),
            tec_std    = Y_model.std(axis=1),
            tp_lats    = tp_lats,
            tp_lons    = tp_lons,
            conid      = conid,
            prn_id     = prn_id,
            leo_id     = leo_id,
        ))

    return truth_arcs, model_arcs


# ─────────────────────────────────────────────────────────────────────────────
# §5  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _occ_colors(parsed_list: list[dict]) -> list:
    """
    Assign a unique colour to each occultation, deepening shade within each
    GNSS constellation using that constellation's colourmap.
    """
    # Count per constellation
    const_counts: dict[str, int] = defaultdict(int)
    for arc in parsed_list:
        prn_id = str(arc.get("prn_id", arc.get("prn", "?")))
        conid  = str(arc.get("conid", prn_id[0].upper()
                              if prn_id[0].upper() in "GREC" else "?")).upper()
        const_counts[conid] += 1

    const_idx: dict[str, int] = defaultdict(int)
    colors = []
    for arc in parsed_list:
        prn_id = str(arc.get("prn_id", arc.get("prn", "?")))
        conid  = str(arc.get("conid", prn_id[0].upper()
                              if prn_id[0].upper() in "GREC" else "?")).upper()
        cfg    = CONSTELLATION_CONFIG.get(conid, {})
        cmap   = mpl.colormaps.get_cmap(cfg.get("cmap", _CONST_FALLBACK_CMAP))
        n      = max(const_counts[conid], 1)
        # Shade range 0.4–0.85 to avoid very light / very dark extremes
        shade  = 0.4 + 0.45 * const_idx[conid] / n
        colors.append(cmap(shade))
        const_idx[conid] += 1

    return colors


def plot_results(
    truth_arcs: list[dict],
    model_arcs: list[dict],
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    time_dt: pd.Timestamp,
    save_path: str,
) -> None:
    """
    Produce the main diagnostic figure:

    Layout — GridSpec(2, 3, width_ratios=[1, 1, 1.6])
    ──────────────────────────────────────────────────
    [0,0] GPS TEC     [0,1] Galileo TEC   [0:2, 2] Globe map
    [1,0] GLONASS TEC [1,1] BeiDou TEC

    TEC panels
    ----------
    • Black solid       : measured podTc2 sTEC
    • Thick coloured solid : 1-deg IRI baseline (truth member 0)
    • 8 thin tab10 lines   : truth members 1–8 (one per parameter perturbation)
    • Coloured dashed      : 5-deg model ensemble mean
    • Coloured shading     : 5-deg model ±1σ

    Shared parameter legend below the 4 TEC panels (tab10 palette, PARAM_NAMES).

    Globe panel
    -----------
    • Small grey dots    : 1-deg Fibonacci grid nodes
    • Steelblue squares  : 5-deg Fibonacci grid nodes
    • Coloured dots      : arc track tangent points (colour = constellation)
    • Large circle       : TEC-max tangent point per arc
    """
    occ_colors = _occ_colors(
        [{"conid": tr["conid"], "prn_id": tr["prn_id"]} for tr in truth_arcs]
    )
    tab10 = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(18, 9), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.6],
                   hspace=0.35, wspace=0.28,
                   left=0.06, right=0.97, top=0.92, bottom=0.14)

    # Panel axes for 4 constellations
    tec_axes: dict[str, plt.Axes] = {}
    for const, (row, col) in _CONST_POS.items():
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#2b2b2b")
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const,
                                                "title_color": "white"})
        ax.set_title(cfg["name"], color=cfg.get("title_color", "white"),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("sTEC (TECU)",      color="lightgray", fontsize=7)
        ax.set_ylabel("Tang. alt. (km)",  color="lightgray", fontsize=7)
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        tec_axes[const] = ax

    # Orthographic globe centerd on the mean tangent-point location of all arcs
    all_tp_lats = np.concatenate([tr["tp_lats"] for tr in truth_arcs])
    all_tp_lons = np.concatenate([tr["tp_lons"] for tr in truth_arcs])
    cen_lat = float(np.mean(all_tp_lats))
    cen_lon = float(np.mean(all_tp_lons))

    ax_globe = fig.add_subplot(gs[:, 2],
                                projection=ccrs.Orthographic(
                                    central_longitude=cen_lon,
                                    central_latitude=cen_lat,
                                ))
    ax_globe.set_facecolor("#2b2b2b")
    ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.4,
                          edgecolor="#aaaaaa")
    ax_globe.add_feature(cfeature.BORDERS,   linewidth=0.2,
                          edgecolor="#888888")
    ax_globe.gridlines(draw_labels=False, linewidth=0.3,
                       color="gray", alpha=0.5)

    # Plot 1-deg grid nodes
    ax_globe.scatter(
        grid_lons_1deg, grid_lats_1deg,
        s=1, color="lightgray", alpha=0.3, transform=ccrs.PlateCarree(),
        zorder=2, label="1° grid",
    )
    # Plot 5-deg grid nodes
    ax_globe.scatter(
        grid_lons_5deg, grid_lats_5deg,
        s=8, color="steelblue", alpha=0.6, marker="s",
        transform=ccrs.PlateCarree(), zorder=3, label="5° grid",
    )

    globe_legend_handles: list = [
        Line2D([0], [0], color="lightgray", marker="o", ms=3,
               linestyle="none", label="1° grid"),
        Line2D([0], [0], color="steelblue", marker="s", ms=4,
               linestyle="none", label="5° grid"),
    ]

    for arc_i, (tr, mo, col) in enumerate(
            zip(truth_arcs, model_arcs, occ_colors)):
        const  = str(tr["conid"]).upper()
        prn_id = str(tr["prn_id"])
        leo_id = str(tr["leo_id"])
        tang   = tr["tangent_km"]
        label  = f"{leo_id} {const}{prn_id}"

        ax = tec_axes.get(const, tec_axes["G"])

        # Measured TEC (black solid)  — sTEC on x, altitude on y
        tec_meas = tr["tec_measured"]
        if np.any(np.isfinite(tec_meas)):
            ax.plot(tec_meas, tang, color="black", linewidth=1.4,
                    zorder=6, label="Measured" if arc_i == 0 else None)

        # Truth member 0: IRI baseline (thick coloured solid)
        ax.plot(tr["tec_all"][:, 0], tang, color=col,
                linewidth=2.0, zorder=5, label=label)

        # Truth members 1–8: parameter perturbations (thin tab10 lines)
        for k in range(1, N_STATE + 1):
            ax.plot(tr["tec_all"][:, k], tang,
                    color=tab10(k - 1), linewidth=0.7,
                    alpha=0.8, zorder=4)

        # Model ensemble mean ± 1σ (coloured dashed + shading)
        ax.plot(mo["tec_mean"], tang, color=col, linewidth=1.2,
                linestyle="--", zorder=5)
        ax.fill_betweenx(
            tang,
            mo["tec_mean"] - mo["tec_std"],
            mo["tec_mean"] + mo["tec_std"],
            color=col, alpha=0.18, zorder=3,
        )

        # Globe: arc track
        ax_globe.scatter(
            tr["tp_lons"], tr["tp_lats"],
            s=4, color=col, alpha=0.6, transform=ccrs.PlateCarree(), zorder=4,
        )
        # Globe: TEC-max marker
        tec_max_idx = int(np.argmin(tang))
        ax_globe.scatter(
            tr["tp_lons"][tec_max_idx], tr["tp_lats"][tec_max_idx],
            s=60, color=col, edgecolors="white", linewidth=0.6,
            marker="o", transform=ccrs.PlateCarree(), zorder=5,
        )
        globe_legend_handles.append(
            Line2D([0], [0], color=col, marker="o", ms=5,
                   linestyle="none", label=label)
        )

    # Add legend on TEC axes
    for const, ax in tec_axes.items():
        if ax.lines:
            ax.legend(fontsize=5, facecolor="#2b2b2b",
                      labelcolor="lightgray", loc="best",
                      framealpha=0.7)

    # Shared parameter legend below TEC panels
    param_handles = [
        Line2D([0], [0], color=tab10(k), linewidth=1.5,
               label=PARAM_NAMES[k])
        for k in range(N_STATE)
    ]
    param_handles.insert(0, Line2D([0], [0], color="black", linewidth=1.5,
                                    label="Measured"))
    fig.legend(
        handles=param_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.38, 0.01),
        fontsize=7,
        facecolor="#2b2b2b",
        labelcolor="lightgray",
        framealpha=0.85,
        title="Truth perturbation",
        title_fontsize=7,
    )

    # Globe legend
    ax_globe.legend(
        handles=globe_legend_handles,
        fontsize=6, facecolor="#2b2b2b",
        labelcolor="lightgray", loc="lower left",
        framealpha=0.7, markerscale=1.3,
    )

    title_str = (f"Parametric IonosphericState forward model — "
                 f"{time_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    fig.suptitle(title_str, color="white", fontsize=11, y=0.97)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5.1  Ensemble parameter histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_ensemble_histograms(
    model_state: IonosphericState,
    mean_state: np.ndarray,
    time_dt: pd.Timestamp,
    save_path: str,
    n_bins: int = 40,
) -> None:
    """
    8×1 figure showing the ensemble distribution of each IRI parameter across
    all grid points and ensemble members.

    For each parameter:
      • Histogram of ensemble[k].ravel()  (n_grid × n_members samples)
      • Red dashed : IRI mean (mean_state[k].mean())
      • Gray dotted : ±1σ sample std lines

    Parameters
    ----------
    model_state : IonosphericState containing the drawn ensemble.
    mean_state  : (N_STATE, n_grid) IRI background mean.
    time_dt     : observation time (used in title).
    save_path   : output file path.
    n_bins      : histogram bin count.
    """
    ens = model_state.ensemble   # (N_STATE, n_grid, n_members)

    fig, axes = plt.subplots(N_STATE, 1, figsize=(7, 14),
                              facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle(
        f"Ensemble parameter distributions — "
        f"{time_dt.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"({model_state.n_members} members × "
        f"{model_state.n_grid_points} grid points)",
        color="white", fontsize=10, y=0.99,
    )

    for k, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        samples = ens[k].ravel()
        mu      = float(mean_state[k].mean())
        sigma   = float(samples.std())

        ax.set_facecolor("#2b2b2b")
        ax.hist(samples, bins=n_bins, color="steelblue",
                alpha=0.75, density=True)
        ax.axvline(mu, color="red",  linestyle="--", linewidth=1.4,
                   label=f"IRI mean = {mu:.3g}")
        ax.axvline(mu - sigma, color="lightgray", linestyle=":",
                   linewidth=1.0)
        ax.axvline(mu + sigma, color="lightgray", linestyle=":",
                   linewidth=1.0, label=f"σ = {sigma:.3g}")
        ax.set_ylabel(name, color="lightgray", fontsize=7, rotation=0,
                      labelpad=70, va="center")
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.legend(fontsize=6, facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="upper right", framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5.5  Parameter sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────

def _sweep_one_param(
    arc_rays: list,
    mean_state_1gp: np.ndarray,
    param_idx: int,
    alt_grid: np.ndarray,
    n_sweep: int   = _N_SWEEP,
    n_sigma: float = _N_SIGMA,
) -> dict:
    """
    Sweep parameter *param_idx* over ±n_sigma·σ around its IRI baseline value
    and compute the forward-modelled sTEC for each sweep member.

    The state has a single grid point so no IDW interpolation is needed.

    Parameters
    ----------
    arc_rays       : list of ray arrays for one arc.
    mean_state_1gp : (N_STATE,)  IRI baseline at the representative grid point.
    param_idx      : index into PARAM_NAMES / state vector to vary.
    alt_grid       : altitude integration grid.
    n_sweep        : number of sweep values (odd → symmetric about baseline).
    n_sigma        : sweep half-range in units of _TRUTH_SIGMA[param_idx].

    Returns
    -------
    dict with keys:
        param_values   : (n_sweep,) actual swept values after clamping.
        tec_matrix     : (n_ep, n_sweep) forward-modelled sTEC.
        baseline_idx   : index of the nearest sweep member to the IRI baseline.
        baseline_val   : IRI baseline value for this parameter.
        truth_idx      : index of member at baseline + _TRUTH_SIGMA[param_idx].
        truth_val      : baseline + _TRUTH_SIGMA[param_idx].
    """
    baseline_val = float(mean_state_1gp[param_idx])
    sigma        = float(_TRUTH_SIGMA[param_idx])
    truth_val    = baseline_val + sigma

    param_values = np.linspace(
        baseline_val - n_sigma * sigma,
        baseline_val + n_sigma * sigma,
        n_sweep,
    )

    # Build single-gp ensemble: shape (N_STATE, 1, n_sweep)
    ens = np.tile(mean_state_1gp[:, np.newaxis, np.newaxis],
                  (1, 1, n_sweep)).copy()
    ens[param_idx, 0, :] = param_values

    state = IonosphericState(n_grid_points=1, n_members=n_sweep)
    state.ensemble = ens
    state.clamp_to_physical_bounds()
    param_values = state.ensemble[param_idx, 0, :].copy()

    obs_op    = ObservationOperator(state, alt_grid)
    gp_indices = np.zeros(len(arc_rays), dtype=int)
    Y_f = obs_op.compute_stec_ensemble(
        arc_rays, grid_point_indices=gp_indices,
    )  # (n_ep, n_sweep)

    baseline_idx = int(np.argmin(np.abs(param_values - baseline_val)))
    truth_idx    = int(np.argmin(np.abs(param_values - truth_val)))

    return dict(
        param_values  = param_values,
        tec_matrix    = Y_f,
        baseline_idx  = baseline_idx,
        baseline_val  = baseline_val,
        truth_idx     = truth_idx,
        truth_val     = truth_val,
    )


def plot_parameter_influence(
    arc_rays: list,
    arc_tang_km: np.ndarray,
    mean_state_1gp: np.ndarray,
    alt_grid: np.ndarray,
    time_dt: pd.Timestamp,
    arc_label: str,
    save_path: str,
    n_sweep: int    = _N_SWEEP,
    n_sigma: float  = _N_SIGMA,
    abs_error: bool = False,
    log_err_scale: bool = False,
    tec_truth_perturbed: np.ndarray | None = None,
) -> None:
    """
    8×2 parameter sensitivity figure.

    For each of the 8 IRI parameters:
      Left  : pcolormesh(param_values, tang_km, sTEC) — colourmap "cividis"
               white dashed line = IRI baseline
               cyan solid line   = +1σ perturbation marker
      Right : pcolormesh(param_values, tang_km, sTEC − truth_col)
               signed → "coolwarm"  (abs_error=False)
               absolute → "magma"   (abs_error=True)

    Parameters
    ----------
    arc_rays       : ray list from _build_arc_rays for one arc.
    arc_tang_km    : (n_ep,) tangent altitudes (km).
    mean_state_1gp : (N_STATE,)  IRI state at the representative 1-deg grid point.
    alt_grid       : altitude integration grid.
    time_dt        : observation time (for plot title).
    arc_label      : string label, e.g. "GN05 G15".
    save_path      : output file path.
    n_sweep, n_sigma, abs_error : sweep settings.
    log_err_scale  : if True, use logarithmic colour normalization on the error
                     panels.  For abs_error=True uses LogNorm; for abs_error=False
                     uses SymLogNorm (handles both signs).
    tec_truth_perturbed : (n_ep, N_STATE) TEC from the 1×1-deg IDW forward model
                     with each parameter individually perturbed by +1σ.
                     Column k is the truth reference for parameter panel k —
                     i.e. truth_arcs[arc_idx]["tec_all"][:, 1:].  Must be in the
                     same epoch order as arc_tang_km.
    """
    if log_err_scale:
        save_path += "_log"
    # Sort epochs by ascending tangent altitude for pcolormesh
    sort_idx  = np.argsort(arc_tang_km)
    tang_plot = arc_tang_km[sort_idx]
    rays_sort = [arc_rays[i] for i in sort_idx]

    # Sort the perturbed truth array into ascending-altitude order.
    # Result shape: (n_ep, N_STATE) — column k is the truth for parameter panel k.
    tec_truth_sorted = (
        tec_truth_perturbed[sort_idx, :]
        if tec_truth_perturbed is not None else None
    )

    fig, axes = plt.subplots(
        N_STATE, 2, figsize=(12, 22),
        facecolor="#1e1e1e",
    )
    fig.patch.set_facecolor("#1e1e1e")

    truth_src  = "1°-grid perturbed TEC" if tec_truth_perturbed is not None else "+1σ perturbation"
    title_type = "Absolute error" if abs_error else "Signed error"
    scale_tag  = "  [log scale]" if log_err_scale else ""
    fig.suptitle(
        f"Parameter sensitivity — {arc_label} — "
        f"{time_dt.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"(left: sTEC TECU;  right: {title_type} vs {truth_src}{scale_tag})",
        color="white", fontsize=10, y=0.995,
    )

    # ── Pre-pass: compute all sweeps and errors so we can set a shared scale ──
    sweep_results = []
    for k in range(N_STATE):
        result = _sweep_one_param(
            rays_sort, mean_state_1gp, k, alt_grid, n_sweep, n_sigma,
        )
        tec   = result["tec_matrix"]
        t_idx = result["truth_idx"]
        if tec_truth_sorted is not None:
            truth_col = tec_truth_sorted[:, k:k+1]
        else:
            truth_col = tec[:, t_idx:t_idx+1]
        err = np.abs(tec - truth_col) if abs_error else (tec - truth_col)
        sweep_results.append((result, err))

    # Global colour limits across all 8 error panels
    all_errors   = np.concatenate([e.ravel() for _, e in sweep_results])
    global_vmax  = float(np.nanpercentile(np.abs(all_errors), 98)) or 1.0

    if abs_error:
        cmap_r = "magma"
        if log_err_scale:
            pos_vals        = all_errors[all_errors > 0]
            global_vmin_log = float(np.nanpercentile(pos_vals, 2)) if len(pos_vals) else 1e-4
            global_vmin_log = max(global_vmin_log, 1e-4)
            global_norm_r   = LogNorm(vmin=global_vmin_log, vmax=global_vmax)
        else:
            global_norm_r = mpl.colors.Normalize(vmin=0, vmax=global_vmax)
    else:
        cmap_r = "coolwarm"
        if log_err_scale:
            linthresh     = max(global_vmax * 0.01, 1e-4)
            global_norm_r = SymLogNorm(linthresh=linthresh,
                                        vmin=-global_vmax, vmax=global_vmax)
        else:
            global_norm_r = mpl.colors.Normalize(vmin=-global_vmax, vmax=global_vmax)

    # ── Plotting loop ─────────────────────────────────────────────────────────
    for k, (result, err) in enumerate(sweep_results):
        pv    = result["param_values"]
        tec   = result["tec_matrix"]
        b_val = result["baseline_val"]
        t_val = result["truth_val"]

        ax_left  = axes[k, 0]
        ax_right = axes[k, 1]

        for ax in (ax_left, ax_right):
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="lightgray", labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")

        # Bin edges for pcolormesh
        pv_edges   = np.concatenate([[2*pv[0]-pv[1]],
                                      0.5*(pv[:-1]+pv[1:]),
                                      [2*pv[-1]-pv[-2]]])
        tang_edges = np.concatenate([[2*tang_plot[0]-tang_plot[1]],
                                      0.5*(tang_plot[:-1]+tang_plot[1:]),
                                      [2*tang_plot[-1]-tang_plot[-2]]])

        # ── Left panel: raw sTEC ─────────────────────────────────────────────
        pcm_l  = ax_left.pcolormesh(pv_edges, tang_edges, tec,
                                     cmap="cividis", shading="flat")
        cbar_l = fig.colorbar(pcm_l, ax=ax_left, fraction=0.04, pad=0.02)
        cbar_l.set_label("sTEC (TECU)", color="lightgray")
        cbar_l.ax.yaxis.label.set_color("lightgray")
        cbar_l.ax.tick_params(colors="lightgray")

        ax_left.axvline(b_val, color="white", linestyle="--", linewidth=1.0)
        ax_left.axvline(t_val, color="cyan",  linestyle="-",  linewidth=1.0)
        ax_left.set_ylabel("Tang. alt. (km)", color="lightgray", fontsize=7)
        ax_left.set_title(PARAM_NAMES[k], color="lightgray", fontsize=8)

        # ── Right panel: error (shared norm, individual colorbar) ────────────
        pcm_r  = ax_right.pcolormesh(pv_edges, tang_edges, err,
                                      cmap=cmap_r, norm=global_norm_r, shading="flat")
        cbar_r = fig.colorbar(pcm_r, ax=ax_right, fraction=0.04, pad=0.02)
        cbar_r.set_label(title_type + " (TECU)", color="lightgray")
        cbar_r.ax.yaxis.label.set_color("lightgray")
        cbar_r.ax.tick_params(colors="lightgray")

        ax_right.axvline(b_val, color="white", linestyle="--", linewidth=1.0)
        ax_right.axvline(t_val, color="cyan",  linestyle="-",  linewidth=1.0)
        ax_right.set_title(PARAM_NAMES[k], color="lightgray", fontsize=8)

        if k == N_STATE - 1:
            ax_left.set_xlabel("Parameter value", color="lightgray", fontsize=7)
            ax_right.set_xlabel("Parameter value", color="lightgray", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §6  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(SAVE_DIR,      exist_ok=True)
    os.makedirs(IRI_CACHE_DIR, exist_ok=True)

    # ── Step 1: scan files ───────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Scanning podTc2 files …")
    meta_list = scan_and_select_files(BASE_PATH)
    if not meta_list:
        print("No files selected.  Exiting.")
        return

    print(f"  Parsing {len(meta_list)} files …")
    parsed_list: list[dict] = []
    for rec in meta_list:
        try:
            data = parse_podTc2_nc_file(rec["path"])
            # Carry metadata into parsed dict
            data["conid"]  = rec["conid"]
            data["prn_id"] = rec["prn_id"]
            data["leo_id"] = rec["leo_id"]
            parsed_list.append(data)
        except Exception as exc:
            warnings.warn(f"Could not parse {rec['path']}: {exc}")

    if not parsed_list:
        print("All files failed to parse.  Exiting.")
        return

    # Representative time: median of file times
    times = [rec["date"] for rec in meta_list]
    mid   = len(times) // 2
    time_dt: pd.Timestamp = sorted(times)[mid]
    print(f"  Representative time: {time_dt}")

    # ── Step 2: build Fibonacci grids ────────────────────────────────────────
    print("\nStep 2: Building Fibonacci grids …")
    tp_lats_all, tp_lons_all = _arc_tangent_tracks(parsed_list)
    print(f"  Arc tangent-point anchors: {len(tp_lats_all)}")

    grid_lats_1deg, grid_lons_1deg = _make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=1.0,
    )
    print(f"  1-deg grid: {len(grid_lats_1deg)} nodes")

    grid_lats_5deg, grid_lons_5deg = _make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=5.0,
    )
    print(f"  5-deg grid: {len(grid_lats_5deg)} nodes")

    # ── Step 2a: 1-deg IRI grid → truth state ────────────────────────────────
    print("\nStep 2a: Building 1-deg IRI state (truth) …")
    lat_min_1 = float(grid_lats_1deg.min())
    lat_max_1 = float(grid_lats_1deg.max())
    lon_min_1 = float(grid_lons_1deg.min())
    lon_max_1 = float(grid_lons_1deg.max())

    mean_1deg, _ = build_iri_state_grid_cached(
        time_dt, grid_lats_1deg, grid_lons_1deg, ALT_GRID,
        spacing_deg=1.0,
        lat_min=lat_min_1, lat_max=lat_max_1,
        lon_min=lon_min_1, lon_max=lon_max_1,
    )
    truth_state = build_truth_state(mean_1deg)
    print(f"  Truth ensemble: {truth_state.n_members} members "
          f"× {truth_state.n_grid_points} grid points")

    # ── Step 2b: 5-deg IRI grid → model ensemble ─────────────────────────────
    print("\nStep 2b: Building 5-deg IRI state (model ensemble) …")
    lat_min_5 = float(grid_lats_5deg.min())
    lat_max_5 = float(grid_lats_5deg.max())
    lon_min_5 = float(grid_lons_5deg.min())
    lon_max_5 = float(grid_lons_5deg.max())

    mean_5deg, _ = build_iri_state_grid_cached(
        time_dt, grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        spacing_deg=5.0,
        lat_min=lat_min_5, lat_max=lat_max_5,
        lon_min=lon_min_5, lon_max=lon_max_5,
    )
    model_state = build_model_ensemble(
        mean_5deg, grid_lats_5deg, grid_lons_5deg,
        n_members=N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
    )
    print(f"  Model ensemble: {model_state.n_members} members "
          f"× {model_state.n_grid_points} grid points")

    # Ensemble variance summary
    ens_var = model_state.ensemble.var(axis=2)  # (N_STATE, n_grid)
    print("  Per-parameter ensemble std (mean over grid):")
    for k, name in enumerate(PARAM_NAMES):
        print(f"    {name:20s}  σ = {float(np.sqrt(ens_var[k].mean())):.4g}")

    # Ensemble histograms
    plot_ensemble_histograms(
        model_state, mean_5deg, time_dt,
        save_path=os.path.join(
            SAVE_DIR, f"ensemble_histograms_{YYYY}_{DOY:03d}.png"
        ),
    )

    # ── Steps 3–4: forward models ────────────────────────────────────────────
    print("\nStep 3: Running forward models …")
    truth_arcs, model_arcs = run_forward_models(
        parsed_list,
        truth_state, model_state,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        ALT_GRID,
    )

    # Summary statistics
    print("\nArc summary (IRI baseline vs 5-deg model mean):")
    for tr, mo in zip(truth_arcs, model_arcs):
        diff = tr["tec_all"][:, 0] - mo["tec_mean"]
        print(f"  {tr['leo_id']} {tr['conid']}{tr['prn_id']:>3s}  "
              f"mean_diff={diff.mean():+.3f} TECU  "
              f"rmse={float(np.sqrt((diff**2).mean())):.3f} TECU")

    # ── Step 5: main results plot ─────────────────────────────────────────────
    print("\nStep 5: Plotting results …")
    save_results = os.path.join(
        SAVE_DIR, f"param_iono_test_{YYYY}_{DOY:03d}.png"
    )
    plot_results(
        truth_arcs, model_arcs,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        time_dt, save_results,
    )

    # ── Step 6: parameter sensitivity sweep ──────────────────────────────────
    print("\nStep 6: Parameter sensitivity sweep …")

    # Representative 1-deg grid point: nearest node to the TEC-max of arc 0
    tree_1deg = cKDTree(np.column_stack([grid_lats_1deg, grid_lons_1deg]))
    ref_lat = meta_list[0]["lat"]
    ref_lon = meta_list[0]["lon"]
    _, _gp_idx = tree_1deg.query([[ref_lat, ref_lon]])
    gp_idx = int(_gp_idx[0])
    mean_1gp = mean_1deg[:, gp_idx]

    # Build rays for arc 0 — epoch order matches truth_arcs[0]
    rays_0, tp_lats_0, tp_lons_0, tang_km_0, _ = _build_arc_rays(parsed_list[0])

    # 1-deg grid perturbed TECs for arc 0: columns 1–8 correspond to each
    # parameter individually shifted by +1σ — column k is the truth for panel k.
    tec_truth_arc0 = truth_arcs[0]["tec_all"][:, 1:]   # (n_ep, N_STATE)

    arc0 = parsed_list[0]
    arc_label_0 = (f"{str(arc0.get('leo_id', '?'))} "
                   f"{str(arc0.get('conid', '?'))}"
                   f"{str(arc0.get('prn_id', '?'))}")

    save_sens = os.path.join(
        SAVE_DIR, f"param_sensitivity_{YYYY}_{DOY:03d}"
    )
    save_abs = os.path.join(
        SAVE_DIR, f"param_sensitivity_abs_{YYYY}_{DOY:03d}"
    )

    plot_parameter_influence(
        rays_0, tang_km_0, mean_1gp, ALT_GRID,
        time_dt, arc_label_0, save_sens,
        n_sweep=_N_SWEEP, n_sigma=_N_SIGMA, abs_error=False, log_err_scale=True,
        tec_truth_perturbed=tec_truth_arc0,
    )
    plot_parameter_influence(
        rays_0, tang_km_0, mean_1gp, ALT_GRID,
        time_dt, arc_label_0, save_abs,
        n_sweep=_N_SWEEP, n_sigma=_N_SIGMA, abs_error=True, log_err_scale=True,
        tec_truth_perturbed=tec_truth_arc0,
    )

    print("\n✓ test_param_iono.py completed successfully.")
    print(f"  Figures in: {SAVE_DIR}")

    # ── Step 7–13: EnKF retrieval experiment ──────────────────────────────────
    _run_enkf_retrieval_experiment(
        parsed_list      = parsed_list,
        meta_list        = meta_list,
        mean_5deg        = mean_5deg,
        grid_lats_1deg   = grid_lats_1deg,
        grid_lons_1deg   = grid_lons_1deg,
        grid_lats_5deg   = grid_lats_5deg,
        grid_lons_5deg   = grid_lons_5deg,
        time_dt          = time_dt,
        save_dir         = SAVE_DIR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §7  Truth ionosphere: 1×1 grid, +1 h, adjusted F10.7
# ─────────────────────────────────────────────────────────────────────────────

def _truth_solar_conditions(
    base_time: pd.Timestamp,
    hour_offset: float = TRUTH_HOUR_OFFSET,
    f107_delta: float  = TRUTH_F107_DELTA,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    Return (truth_time, sampling_df) for the truth ionosphere.

    truth_time    = base_time + hour_offset hours
    sampling_df   = solar-index DataFrame with F10.7 incremented by f107_delta
    """
    truth_time   = base_time + pd.Timedelta(hours=hour_offset)
    df           = _solar_sampling_df(truth_time).copy()
    df["f107"]   = df["f107"] + f107_delta
    return truth_time, df


def build_truth_iri_grid(
    truth_time: pd.Timestamp,
    truth_sampling_df: pd.DataFrame,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    alt_grid: np.ndarray,
    label: str = "",
) -> tuple[IonosphericState, np.ndarray, np.ndarray]:
    """
    Evaluate IRI at every grid point using the modified solar conditions and
    assemble the 8-parameter state.

    Returns
    -------
    truth_state  : single-member IonosphericState (the deterministic truth).
    ne_profiles  : (n_alt, n_grid) electron density profiles (m⁻³).
    mean_state   : (N_STATE, n_grid) parameter state in log/km/dim-less units.
    """
    n_grid = len(grid_lats)
    tag    = f" [{label}]" if label else ""
    print(f"  Building IRI truth grid{tag}  ({n_grid} pts, "
          f"{truth_time.strftime('%H:%M')} UTC, "
          f"F10.7+{TRUTH_F107_DELTA:.0f}) …")

    ne_profiles, feature_vecs = _get_iri_edp_and_features_batch(
        truth_time, grid_lats, grid_lons, alt_grid, truth_sampling_df,
    )
    mean_state = np.empty((N_STATE, n_grid))
    for g in range(n_grid):
        mean_state[:, g] = _state_from_iri_direct(
            ne_profiles[:, g], feature_vecs[:, g], alt_grid,
        )

    truth_state          = IonosphericState(n_grid_points=n_grid, n_members=1)
    truth_state.ensemble = mean_state[:, :, np.newaxis].copy()
    truth_state.clamp_to_physical_bounds()
    return truth_state, ne_profiles, mean_state


# ─────────────────────────────────────────────────────────────────────────────
# §8  Generate truth TEC measurements
# ─────────────────────────────────────────────────────────────────────────────

def generate_truth_tec(
    parsed_list: list[dict],
    truth_state: IonosphericState,
    grid_lats_truth: np.ndarray,
    grid_lons_truth: np.ndarray,
    alt_grid: np.ndarray,
) -> list[dict]:
    """
    Evaluate the truth IonosphericState along every arc ray to produce
    synthetic sTEC observations.

    Parameters
    ----------
    parsed_list      : list of parsed podTc2 arc dicts.
    truth_state      : single-member IonosphericState on the 1-deg truth grid.
    grid_lats/lons_truth : (n_truth,) 1-deg Fibonacci grid coordinates.
    alt_grid         : altitude integration grid (km).

    Returns
    -------
    List of dicts — one per arc — with keys:
        rays, tec_truth, tp_lats, tp_lons, tang_km, conid, prn_id, leo_id
    """
    obs_truth = ObservationOperator(truth_state, alt_grid)
    arc_list: list[dict] = []

    for arc_idx, arc in enumerate(parsed_list):
        prn_id = str(arc.get("prn_id", arc.get("prn", "?")))
        leo_id = str(arc.get("leo_id", "?"))
        conid  = str(arc.get("conid",
                    prn_id[0].upper() if prn_id[0].upper() in "GREC" else "?")).upper()

        rays, tp_lats, tp_lons, tang_km, _ = _build_arc_rays(arc)
        n_ep = len(rays)

        # IDW weights from each epoch's tangent point to the 1-deg truth grid
        W = np.stack([
            _idw_weights(tp_lats[k], tp_lons[k],
                         grid_lats_truth, grid_lons_truth)
            for k in range(n_ep)
        ])  # (n_ep, n_truth)

        Y_truth = obs_truth.compute_stec_ensemble(
            rays, grid_point_weights=W,
        )  # (n_ep, 1)
        tec_truth = Y_truth[:, 0]   # (n_ep,)

        print(f"  Arc {arc_idx+1}/{len(parsed_list)}: "
              f"{leo_id} {conid}{prn_id}  "
              f"{n_ep} epochs  "
              f"TEC {tec_truth.min():.1f}–{tec_truth.max():.1f} TECU")

        arc_list.append(dict(
            rays      = rays,
            tec_truth = tec_truth,
            tp_lats   = tp_lats,
            tp_lons   = tp_lons,
            tang_km   = tang_km,
            conid     = conid,
            prn_id    = prn_id,
            leo_id    = leo_id,
        ))

    return arc_list


# ─────────────────────────────────────────────────────────────────────────────
# §9  ParametricEnKF assimilation
# ─────────────────────────────────────────────────────────────────────────────

def run_parametric_enkf(
    arc_truth_list: list[dict],
    model_state: IonosphericState,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    loc_radius_km: float  = ENKF_LOC_RADIUS_KM,
    sigma_obs: float      = ENKF_SIGMA_OBS,
    inflation: float      = ENKF_INFLATION,
    n_mda: int            = ENKF_N_MDA,
    max_update_step: float = ENKF_MAX_UPDATE_STEP,
    max_update_rays: int  = ENKF_MAX_UPDATE_RAYS,
) -> dict:
    """
    Assimilate truth-generated sTEC observations into the 5-deg model ensemble.

    Uses ES-MDA (Emerick & Reynolds 2013) with Gaspari-Cohn ray-path
    localisation.  Mirrors the logic of _run_parametric_enkf in
    demo_compare_kf_enkf.py but operates directly on arc_truth_list.

    Returns
    -------
    dict with keys:
        model_state, prior_ensemble, prior_edp, posterior_edp,
        posterior_ne_5deg (same as posterior_edp), posterior_mean_5deg,
        tec_slices, prior_rmse, post_rmse,
        all_prior_resid, all_post_resid,
        arc_prior_mean, arc_post_mean, arc_prior_rmse, arc_post_rmse,
        arc_lats, arc_lons, arc_labels,
        mda_arc_means_list, mda_flat_list, grid_lats, grid_lons
    """
    n_geo     = model_state.n_grid_points
    n_members = model_state.n_members
    n_occ     = len(arc_truth_list)

    _idw_k = min(4, n_geo)

    # ── Build per-arc ray sets ────────────────────────────────────────────────
    # (A) All per-epoch rays  → TEC profile evaluation and RMSE diagnostics
    # (B) Decimated subset   → EnKF update (compact observation vector)
    per_arc_sample_rays: list[list] = []
    per_arc_tp_lats:     list       = []
    per_arc_tp_lons:     list       = []
    ray_counts:          list[int]  = []
    arc_all_tec:         list       = []

    rep_rays:          list      = []
    rep_tp_lats_list:  list      = []
    rep_tp_lons_list:  list      = []
    rep_tec_obs_list:  list      = []
    arc_update_counts: list[int] = []

    for arc in arc_truth_list:
        rays    = arc["rays"]
        tec     = arc["tec_truth"]
        tp_lats = arc["tp_lats"]
        tp_lons = arc["tp_lons"]
        n_s     = len(rays)

        per_arc_sample_rays.append(rays)
        per_arc_tp_lats.append(tp_lats)
        per_arc_tp_lons.append(tp_lons)
        ray_counts.append(n_s)
        arc_all_tec.append(tec)

        # Uniform-stride sub-decimation for the update rays
        if n_s > max_update_rays:
            stride = int(np.ceil(n_s / max_update_rays))
            chosen = list(range(0, n_s, stride))
            if n_s - 1 not in chosen:
                chosen.append(n_s - 1)
        else:
            chosen = list(range(n_s))

        for idx in chosen:
            rep_rays.append(rays[idx])
            rep_tp_lats_list.append(float(tp_lats[idx]))
            rep_tp_lons_list.append(float(tp_lons[idx]))
            rep_tec_obs_list.append(float(tec[idx]))
        arc_update_counts.append(len(chosen))

    rep_tp_lats = np.array(rep_tp_lats_list)
    rep_tp_lons = np.array(rep_tp_lons_list)
    y_obs_arc   = np.array(rep_tec_obs_list)
    y_obs_all   = np.concatenate(arc_all_tec)

    # Flat per-epoch tangent points (for IDW weight computation)
    all_tp_lats_flat = np.concatenate([a.tolist() for a in per_arc_tp_lats])
    all_tp_lons_flat = np.concatenate([a.tolist() for a in per_arc_tp_lons])
    all_sample_rays  = [r for arc_r in per_arc_sample_rays for r in arc_r]

    all_sample_W = _idw_weights_enkf(
        all_tp_lats_flat, all_tp_lons_flat,
        grid_lats_5deg, grid_lons_5deg, k=_idw_k,
    )
    rep_W = _idw_weights_enkf(
        rep_tp_lats, rep_tp_lons,
        grid_lats_5deg, grid_lons_5deg, k=_idw_k,
    )

    n_update_obs = len(rep_rays)
    print(f"  [EnKF] {n_occ} arcs  |  "
          f"{len(all_sample_rays)} profile rays  |  "
          f"{n_update_obs} update rays  |  "
          f"{n_members} members")
    if n_update_obs >= n_members:
        print(f"  [EnKF] WARNING: n_update_obs ({n_update_obs}) >= n_members ({n_members}). "
              f"D = P_yy + R will be rank-deficient — reduce ENKF_MAX_UPDATE_RAYS.")

    # ── Prior forward model ───────────────────────────────────────────────────
    op                    = ObservationOperator(model_state, alt_grid)
    prior_ensemble_snap   = model_state.ensemble.copy()

    Y_all_prior_ens  = op.compute_stec_ensemble(
        all_sample_rays, grid_point_weights=all_sample_W,
    )
    Y_all_prior_mean = Y_all_prior_ens.mean(axis=1)

    Y_rep_prior_ens  = op.compute_stec_ensemble(rep_rays, grid_point_weights=rep_W)
    Y_rep_prior_mean = Y_rep_prior_ens.mean(axis=1)

    prior_inno = y_obs_arc - Y_rep_prior_mean
    print(f"  [EnKF] Prior innovations  "
          f"mean={prior_inno.mean():.2f}  std={prior_inno.std():.2f}  "
          f"max_abs={np.abs(prior_inno).max():.2f} TECU")

    # ── Localisation matrix ───────────────────────────────────────────────────
    if n_geo > 1 and np.isfinite(loc_radius_km) and loc_radius_km > 0:
        L_ray = build_ray_localisation_matrix(
            grid_lats_5deg, grid_lons_5deg, rep_rays, loc_radius_km,
        )
        print(f"  [EnKF] GC ray-path localisation  loc_radius={loc_radius_km:.0f} km")
    else:
        L_ray = None

    # ── Prior inflation ───────────────────────────────────────────────────────
    if inflation > 1.0:
        X_mu              = model_state.ensemble.mean(axis=2, keepdims=True)
        model_state.ensemble = X_mu + (model_state.ensemble - X_mu) * inflation

    # ── EnKF / ES-MDA update ─────────────────────────────────────────────────
    R     = (sigma_obs ** 2) * np.eye(len(rep_rays))
    R_mda = n_mda * R

    enkf = ParametricEnKF(
        state         = model_state,
        grid_lats     = grid_lats_5deg,
        grid_lons     = grid_lons_5deg,
        loc_radius_km = loc_radius_km,
        inflation     = 1.0,   # inflation already applied above
    )

    mda_inno_list: list[np.ndarray] = []
    for mda_i in range(n_mda):
        Y_mda_ens = op.compute_stec_ensemble(rep_rays, grid_point_weights=rep_W)
        inno_mda  = y_obs_arc - Y_mda_ens.mean(axis=1)
        mda_inno_list.append(inno_mda.copy())
        label_i   = (f"ES-MDA {mda_i+1}/{n_mda}"
                     if n_mda > 1 else "EnKF update")
        print(f"  [{label_i}]  "
              f"mean={inno_mda.mean():.2f}  std={inno_mda.std():.2f}  "
              f"max_abs={np.abs(inno_mda).max():.2f} TECU")
        # Apply bounds every step: prevents overflow when extreme ensemble members
        # produce NaN/Inf in the forward model on the next iteration.
        # The positive-feedback loop (clamping → larger innovations → more clamping)
        # is broken by keeping ENKF_SIGMA_OBS large enough that the Kalman gain is
        # modest and few members saturate bounds after each step.
        enkf.assimilate(
            Y_f                 = Y_mda_ens,
            y_obs               = y_obs_arc,
            R                   = R_mda,
            localisation_matrix = L_ray,
            max_update_step     = max_update_step,
            deterministic       = False,
            apply_bounds        = True,
        )

    # ── Posterior forward model ───────────────────────────────────────────────
    Y_all_post_ens  = op.compute_stec_ensemble(
        all_sample_rays, grid_point_weights=all_sample_W,
    )
    Y_all_post_mean = Y_all_post_ens.mean(axis=1)
    Y_rep_post_ens  = op.compute_stec_ensemble(rep_rays, grid_point_weights=rep_W)
    Y_rep_post_mean = Y_rep_post_ens.mean(axis=1)
    post_inno       = y_obs_arc - Y_rep_post_mean

    prior_rmse = float(np.sqrt(np.nanmean((y_obs_all - Y_all_prior_mean) ** 2)))
    post_rmse  = float(np.sqrt(np.nanmean((y_obs_all - Y_all_post_mean) ** 2)))
    print(f"  [EnKF] Prior RMSE {prior_rmse:.3f} TECU  →  "
          f"Post RMSE {post_rmse:.3f} TECU")

    # ── Convert parametric state → Ne profile grids ───────────────────────────
    prior_state_enc          = IonosphericState(n_geo, n_members)
    prior_state_enc.ensemble = prior_ensemble_snap
    prior_edp   = _parametric_to_edp(prior_state_enc, prior_ensemble_snap, alt_grid)
    post_edp    = _parametric_to_edp(model_state,     model_state.ensemble, alt_grid)

    posterior_mean_5deg = model_state.ensemble.mean(axis=2)   # (N_STATE, n_geo)

    # ── TEC slices (one per arc) ──────────────────────────────────────────────
    tec_slices: list[dict] = []
    soff = 0
    for i, arc in enumerate(arc_truth_list):
        n_s = ray_counts[i]
        sl  = slice(soff, soff + n_s)
        tec_slices.append(dict(
            tec_truth = arc["tec_truth"],
            prior_tec = Y_all_prior_mean[sl].copy(),
            post_tec  = Y_all_post_mean[sl].copy(),
            tang_km   = arc["tang_km"],
        ))
        soff += n_s

    # ── Per-arc innovation statistics ─────────────────────────────────────────
    all_prior_resid = y_obs_all - Y_all_prior_mean
    all_post_resid  = y_obs_all - Y_all_post_mean

    arc_prior_mean_l, arc_post_mean_l  = [], []
    arc_prior_rmse_l, arc_post_rmse_l  = [], []
    arc_lats_l, arc_lons_l, arc_lbl_l  = [], [], []

    soff = 0
    for i, arc in enumerate(arc_truth_list):
        n_s = ray_counts[i]
        sl  = slice(soff, soff + n_s)
        rp  = all_prior_resid[sl]
        ra  = all_post_resid[sl]
        arc_prior_mean_l.append(float(np.nanmean(rp)))
        arc_post_mean_l.append(float(np.nanmean(ra)))
        arc_prior_rmse_l.append(float(np.sqrt(np.nanmean(rp ** 2))))
        arc_post_rmse_l.append(float(np.sqrt(np.nanmean(ra ** 2))))
        arc_lats_l.append(float(arc["tp_lats"].mean()))
        arc_lons_l.append(float(arc["tp_lons"].mean()))
        arc_lbl_l.append(f"{arc['conid']}{arc['prn_id']}")
        soff += n_s

    # Per-MDA-step mean residuals per arc (for the innovation bar chart)
    arc_upd_offs = np.concatenate([[0], np.cumsum(arc_update_counts)])
    mda_arc_means_list: list[np.ndarray] = []
    for inno_step in mda_inno_list:
        step_means = []
        for ai in range(n_occ):
            sl_u = slice(int(arc_upd_offs[ai]), int(arc_upd_offs[ai + 1]))
            step_means.append(float(np.nanmean(inno_step[sl_u])))
        mda_arc_means_list.append(np.array(step_means))

    return dict(
        model_state         = model_state,
        prior_ensemble      = prior_ensemble_snap,
        posterior_ensemble  = model_state.ensemble.copy(),  # (N_STATE, n_geo, n_members)
        prior_edp           = prior_edp,          # (n_alt, n_geo)
        posterior_edp       = post_edp,           # (n_alt, n_geo)  alias for clarity
        posterior_ne_5deg   = post_edp,           # (n_alt, n_geo)
        posterior_mean_5deg = posterior_mean_5deg, # (N_STATE, n_geo)
        tec_slices          = tec_slices,
        prior_rmse          = prior_rmse,
        post_rmse           = post_rmse,
        all_prior_resid     = all_prior_resid,
        all_post_resid      = all_post_resid,
        arc_prior_mean      = np.array(arc_prior_mean_l),
        arc_post_mean       = np.array(arc_post_mean_l),
        arc_prior_rmse      = np.array(arc_prior_rmse_l),
        arc_post_rmse       = np.array(arc_post_rmse_l),
        arc_lats            = np.array(arc_lats_l),
        arc_lons            = np.array(arc_lons_l),
        arc_labels          = arc_lbl_l,
        mda_arc_means_list  = mda_arc_means_list,
        mda_flat_list       = mda_inno_list,
        grid_lats           = grid_lats_5deg,
        grid_lons           = grid_lons_5deg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §10  EnKF TEC + globe + EDP spaghetti plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_tec_edp_figure(
    arc_truth_list: list[dict],
    tec_slices: list[dict],
    prior_edp: np.ndarray,
    post_edp: np.ndarray,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    suptitle: str,
    save_path: str,
) -> None:
    """
    Shared TEC + globe + EDP spaghetti figure used by both EnKF and KF wrappers.

    Layout — GridSpec(2, 4, width_ratios=[1,1,1.5,1.2])
    ──────────────────────────────────────────────────────
    [0,0] GPS        [0,1] Galileo     [0:2,2] Globe   [0:2,3] EDP spaghetti
    [1,0] GLONASS    [1,1] BeiDou                       (full-height column)
    """
    occ_colors = _occ_colors(
        [{"conid": a["conid"], "prn_id": a["prn_id"]} for a in arc_truth_list]
    )

    fig = plt.figure(figsize=(26, 9), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    gs  = GridSpec(2, 4, figure=fig,
                   width_ratios=[1, 1, 1.5, 1.2],
                   wspace=0.40, hspace=0.45,
                   left=0.05, right=0.97, top=0.92, bottom=0.10)

    # ── TEC panels ───────────────────────────────────────────────────────────
    tec_axes: dict[str, plt.Axes] = {}
    first_ax = None
    for const, (row, col) in _CONST_POS.items():
        ax = fig.add_subplot(gs[row, col],
                             sharey=first_ax if first_ax is not None else None)
        ax.set_facecolor("#2b2b2b")
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "white"})
        ax.set_title(cfg["name"], color=cfg.get("title_color", "white"),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("sTEC (TECU)", color="lightgray", fontsize=7)
        ax.set_ylabel("Tang. alt. (km)", color="lightgray", fontsize=7)
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        tec_axes[const] = ax
        if first_ax is None:
            first_ax = ax

    # ── Globe ─────────────────────────────────────────────────────────────────
    all_tp_lats = np.concatenate([a["tp_lats"] for a in arc_truth_list])
    all_tp_lons = np.concatenate([a["tp_lons"] for a in arc_truth_list])
    cen_lat = float(np.mean(all_tp_lats))
    cen_lon = float(np.mean(all_tp_lons))
    ax_globe = fig.add_subplot(
        gs[0:2, 2],
        projection=ccrs.Orthographic(
            central_longitude=cen_lon, central_latitude=cen_lat,
        ),
    )
    ax_globe.set_facecolor("#2b2b2b")
    ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#aaaaaa")
    ax_globe.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor="#888888")
    ax_globe.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
    ax_globe.scatter(grid_lons_1deg, grid_lats_1deg,
                     s=1, color="lightgray", alpha=0.3,
                     transform=ccrs.PlateCarree(), zorder=2)
    ax_globe.scatter(grid_lons_5deg, grid_lats_5deg,
                     s=10, color="steelblue", alpha=0.7, marker="s",
                     transform=ccrs.PlateCarree(), zorder=3)

    # ── EDP spaghetti — full right column ────────────────────────────────────
    ax_edp = fig.add_subplot(gs[0:2, 3])
    ax_edp.set_facecolor("#2b2b2b")
    ax_edp.set_title("EDP — prior vs posterior", color="white", fontsize=8)
    ax_edp.set_xlabel("Ne (m⁻³)", color="lightgray", fontsize=7)
    ax_edp.set_ylabel("Altitude (km)", color="lightgray", fontsize=7)
    ax_edp.tick_params(colors="lightgray", labelsize=6)
    for sp in ax_edp.spines.values():
        sp.set_edgecolor("#555")

    # center of the 5-deg model grid (nearest node to the mean lat/lon)
    cen_5deg_dist = _haversine_km(
        float(np.mean(grid_lats_5deg)), float(np.mean(grid_lons_5deg)),
        grid_lats_5deg, grid_lons_5deg,
    )
    cen_5deg_idx = int(np.argmin(cen_5deg_dist))

    n_geo = prior_edp.shape[1]
    for g in range(n_geo):
        if g == cen_5deg_idx:
            continue  # drawn last so it sits on top
        ax_edp.plot(prior_edp[:, g], alt_grid, color="steelblue",
                    linewidth=0.7, alpha=0.4, zorder=3)
        ax_edp.plot(post_edp[:, g],  alt_grid, color="tomato",
                    linewidth=0.7, alpha=0.6, zorder=3)

    # Bold center profiles drawn last (highest zorder)
    ax_edp.plot(prior_edp[:, cen_5deg_idx], alt_grid, color="steelblue",
                linewidth=2.5, alpha=1.0, zorder=6)
    ax_edp.plot(post_edp[:, cen_5deg_idx],  alt_grid, color="tomato",
                linewidth=2.5, alpha=1.0, zorder=6)

    cen_1deg_dist = _haversine_km(
        float(np.mean(grid_lats_5deg)), float(np.mean(grid_lons_5deg)),
        grid_lats_1deg, grid_lons_1deg,
    )
    cen_1deg_idx = int(np.argmin(cen_1deg_dist))
    ax_edp.plot(truth_ne_1deg[:, cen_1deg_idx], alt_grid,
                color="yellow", linewidth=1.5, linestyle="--", zorder=7)

    edp_handles = [
        Line2D([0], [0], color="steelblue", lw=0.7, alpha=0.4, label="Prior Ne"),
        Line2D([0], [0], color="tomato",    lw=0.7, alpha=0.6, label="Posterior Ne"),
        Line2D([0], [0], color="steelblue", lw=2.5, alpha=1.0, label="Prior Ne (center)"),
        Line2D([0], [0], color="tomato",    lw=2.5, alpha=1.0, label="Posterior Ne (center)"),
        Line2D([0], [0], color="yellow",    lw=1.5, linestyle="--",
               label="Truth Ne (center)"),
    ]
    ax_edp.legend(handles=edp_handles, fontsize=7,
                  facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="lower right", framealpha=0.8)

    # ── Populate TEC and globe panels ─────────────────────────────────────────
    globe_handles: list = [
        Line2D([0], [0], color="lightgray", marker="o", ms=3,
               linestyle="none", label="1° truth grid"),
        Line2D([0], [0], color="steelblue", marker="s", ms=4,
               linestyle="none", label="5° model grid"),
    ]
    style_placed = False

    for arc, sl, col in zip(arc_truth_list, tec_slices, occ_colors):
        const = arc["conid"]
        label = f"{arc['leo_id']} {const}{arc['prn_id']}"
        tang  = sl["tang_km"]
        ax    = tec_axes.get(const, tec_axes.get("G"))

        ax.plot(sl["tec_truth"], tang, color=col, linewidth=2.2,
                zorder=6, label=label)
        ax.plot(sl["prior_tec"], tang, color=col, linewidth=1.1,
                linestyle="--", alpha=0.6, zorder=4,
                label="Prior" if not style_placed else None)
        ax.plot(sl["post_tec"], tang, color=col, linewidth=1.4,
                linestyle=":", alpha=0.9, zorder=5,
                label="Posterior" if not style_placed else None)
        style_placed = True

        ax_globe.scatter(arc["tp_lons"], arc["tp_lats"],
                         s=4, color=col, alpha=0.6,
                         transform=ccrs.PlateCarree(), zorder=4)
        peak_idx = int(np.argmin(tang))
        ax_globe.scatter(arc["tp_lons"][peak_idx], arc["tp_lats"][peak_idx],
                         s=60, color=col, edgecolors="white", linewidth=0.5,
                         marker="o", transform=ccrs.PlateCarree(), zorder=5)
        globe_handles.append(
            Line2D([0], [0], color=col, marker="o", ms=5,
                   linestyle="none", label=label)
        )

    for ax in tec_axes.values():
        if ax.lines:
            ax.legend(fontsize=5, facecolor="#2b2b2b",
                      labelcolor="lightgray", loc="best", framealpha=0.7)

    ax_globe.legend(handles=globe_handles, fontsize=6,
                    facecolor="#2b2b2b", labelcolor="lightgray",
                    loc="lower left", framealpha=0.7, markerscale=1.2)

    fig.suptitle(suptitle, color="white", fontsize=10, y=0.98)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_enkf_tec_edp(
    arc_truth_list: list[dict],
    enkf_result: dict,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    truth_time: pd.Timestamp,
    save_path: str,
) -> None:
    """TEC + globe + EDP summary figure for the ParametricEnKF."""
    _plot_tec_edp_figure(
        arc_truth_list,
        enkf_result["tec_slices"],
        enkf_result["prior_edp"],
        enkf_result["posterior_edp"],
        truth_ne_1deg,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        alt_grid,
        suptitle=(
            f"ParametricEnKF retrieval — truth ionosphere "
            f"{truth_time.strftime('%Y-%m-%d %H:%M')} UTC  "
            f"(+{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f})\n"
            f"Prior RMSE {enkf_result['prior_rmse']:.2f} TECU  →  "
            f"Posterior RMSE {enkf_result['post_rmse']:.2f} TECU"
        ),
        save_path=save_path,
    )


def plot_kf_tec_edp(
    arc_truth_list: list[dict],
    kf_result: dict,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    truth_time: pd.Timestamp,
    save_path: str,
) -> None:
    """TEC + globe + EDP summary figure for the gridded Ne KF."""
    _plot_tec_edp_figure(
        arc_truth_list,
        kf_result["tec_slices"],
        kf_result["prior_edp"],
        kf_result["posterior_edp"],
        truth_ne_1deg,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        alt_grid,
        suptitle=(
            f"Gridded Ne KF retrieval — truth ionosphere "
            f"{truth_time.strftime('%Y-%m-%d %H:%M')} UTC  "
            f"(+{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f})\n"
            f"Prior RMSE {kf_result['prior_rmse']:.2f} TECU  →  "
            f"Posterior RMSE {kf_result['post_rmse']:.2f} TECU"
        ),
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §11  EDP spatial error — 5×2 orthographic
# ─────────────────────────────────────────────────────────────────────────────

def _meshgrid_interp(
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    n_pts: int      = 120,
    margin_deg: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearly interpolate scattered lat/lon values onto a regular meshgrid.

    Returns
    -------
    LO, LA    : (n_pts, n_pts) regular lon/lat coordinate grids (degrees).
    vals_grid : (n_pts, n_pts) interpolated values; NaN outside convex hull.
    """
    from scipy.interpolate import griddata as _gd
    lat_min = lats.min() - margin_deg;  lat_max = lats.max() + margin_deg
    lon_min = lons.min() - margin_deg;  lon_max = lons.max() + margin_deg
    lats_r  = np.linspace(lat_min, lat_max, n_pts)
    lons_r  = np.linspace(lon_min, lon_max, n_pts)
    LO, LA  = np.meshgrid(lons_r, lats_r)
    fin     = np.isfinite(vals)
    if fin.sum() < 3:
        return LO, LA, np.full_like(LO, np.nan)
    vg = _gd(
        np.column_stack([lons[fin], lats[fin]]),
        vals[fin],
        (LO, LA),
        method="linear",
    )
    return LO, LA, vg


def _ax_pcolormesh(
    ax, LO, LA, vg,
    cmap: str, vmin: float, vmax: float,
    fig,
    label: str,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
) -> None:
    """Shared helper: pcolormesh + grid-point dots + colorbar on a cartopy ax."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    pm = ax.pcolormesh(
        LO, LA, vg,
        cmap=cmap_obj, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), zorder=3, shading="auto",
    )
    # Overlay original grid points as small markers for reference
    ax.scatter(
        grid_lons, grid_lats, c="k", s=5, alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(label, fontsize=8)


def plot_edp_spatial_error(
    truth_ne_5deg: np.ndarray,
    posterior_ne_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    error_altitudes: list = None,
    truth_time: pd.Timestamp = None,
    show_truth_col: bool = False,
    save_path: str = "./Figures/test_param_iono/edp_spatial_error.png",
) -> None:
    """
    5×N orthographic meshgrid plots of EDP retrieval error at five altitudes.

    Rows are ordered highest-to-lowest altitude (top → bottom).
    All panels in the same column share the same colorbar limits, computed
    from the 98th percentile across all displayed altitudes.

    Col 0  : [if show_truth_col] truth Ne  (m⁻³, viridis)
    Col +0 : absolute Ne error  |Ne_post − Ne_truth|  (m⁻³, magma)
    Col +1 : signed percent error  100·(Ne_post − Ne_truth)/Ne_truth  (%, RdBu_r)
    """
    if error_altitudes is None:
        error_altitudes = ERROR_ALTITUDES

    # Highest altitude at the top of the figure
    error_altitudes_sorted = sorted(error_altitudes, reverse=True)

    n_alt_plot = len(error_altitudes_sorted)
    n_cols     = 3 if show_truth_col else 2
    cen_lat    = float(np.mean(grid_lats_5deg))
    cen_lon    = float(np.mean(grid_lons_5deg))
    proj       = ccrs.Orthographic(central_longitude=cen_lon,
                                   central_latitude=cen_lat)

    fig_w  = 6 * n_cols
    fig, axes = plt.subplots(
        n_alt_plot, n_cols,
        figsize=(fig_w, 4 * n_alt_plot),
        subplot_kw={"projection": proj},
    )
    if n_alt_plot == 1:
        axes = axes[np.newaxis, :]

    title_str = "EDP spatial error — posterior vs truth"
    if show_truth_col:
        title_str = "EDP — truth / absolute error / percent error"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(title_str, fontsize=11, y=1.01)

    ext_lat_min = grid_lats_5deg.min() - 3.0
    ext_lat_max = grid_lats_5deg.max() + 3.0
    ext_lon_min = grid_lons_5deg.min() - 3.0
    ext_lon_max = grid_lons_5deg.max() + 3.0

    # Column metadata (order matches show_truth_col branches below)
    if show_truth_col:
        col_cmaps    = ["viridis", "magma",  "RdBu_r"]
        col_symm     = [False,     False,     True   ]
        col_cb_labels= ["Truth Ne  (m⁻³)", "|ΔNe|  (m⁻³)", "ΔNe  (%)"]
    else:
        col_cmaps    = ["magma",  "RdBu_r"]
        col_symm     = [False,     True   ]
        col_cb_labels= ["|ΔNe|  (m⁻³)", "ΔNe  (%)"]

    # First pass — collect all finite values per column to find shared limits
    col_vals_global = [[] for _ in range(n_cols)]
    for alt_km in error_altitudes_sorted:
        alt_idx = int(np.argmin(np.abs(alt_grid - alt_km)))
        ne_tru  = truth_ne_5deg[alt_idx, :]
        ne_pst  = posterior_ne_5deg[alt_idx, :]
        abs_err = np.abs(ne_pst - ne_tru)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_err = 100.0 * (ne_pst - ne_tru) / np.where(ne_tru > 0, ne_tru, np.nan)
        row_vals = ([ne_tru] if show_truth_col else []) + [abs_err, pct_err]
        for c, v in enumerate(row_vals):
            fin = np.isfinite(v)
            if fin.any():
                col_vals_global[c].append(v[fin])

    col_limits = []
    for c in range(n_cols):
        if col_vals_global[c]:
            all_v = np.concatenate(col_vals_global[c])
            v_abs = float(np.nanpercentile(np.abs(all_v), 98)) or 1.0
            col_limits.append((-v_abs, v_abs) if col_symm[c] else (0.0, v_abs))
        else:
            col_limits.append((0.0, 1.0))

    # Second pass — plot with shared limits
    for row_i, alt_km in enumerate(error_altitudes_sorted):
        alt_idx = int(np.argmin(np.abs(alt_grid - alt_km)))
        ne_tru  = truth_ne_5deg[alt_idx, :]
        ne_pst  = posterior_ne_5deg[alt_idx, :]
        abs_err = np.abs(ne_pst - ne_tru)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_err = 100.0 * (ne_pst - ne_tru) / np.where(ne_tru > 0, ne_tru, np.nan)
        row_vals = ([ne_tru] if show_truth_col else []) + [abs_err, pct_err]

        for col_i, vals in enumerate(row_vals):
            ax      = axes[row_i, col_i]
            cmap    = col_cmaps[col_i]
            vmin, vmax = col_limits[col_i]
            cb_label = col_cb_labels[col_i]
            ax_title = f"~{alt_grid[alt_idx]:.0f} km — {cb_label}"

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#444")
            ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
            ax.set_facecolor("#d8d8d8")
            ax.set_extent(
                [ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max],
                crs=ccrs.PlateCarree(),
            )

            if not np.isfinite(vals).any():
                ax.set_title(f"{ax_title} — no data", fontsize=8)
                continue

            LO, LA, vg = _meshgrid_interp(grid_lats_5deg, grid_lons_5deg, vals)
            _ax_pcolormesh(ax, LO, LA, vg, cmap, vmin, vmax,
                           fig, cb_label, grid_lons_5deg, grid_lats_5deg)
            ax.set_title(ax_title, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §12  Parameter spatial error — 8×2 orthographic
# ─────────────────────────────────────────────────────────────────────────────

def plot_parameter_spatial_error(
    truth_mean_5deg: np.ndarray,
    posterior_mean_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: pd.Timestamp = None,
    save_path: str = "./Figures/test_param_iono/param_spatial_error.png",
) -> None:
    """
    8×2 orthographic meshgrid plots of IRI parameter retrieval.

    Row k  : PARAM_NAMES[k]
    Col 0  : posterior parameter value (pcolormesh, viridis)
    Col 1  : signed error  posterior_param − truth_param  (pcolormesh, RdBu_r)

    Values are linearly interpolated onto a regular lat/lon meshgrid
    (scipy.interpolate.griddata) and displayed with pcolormesh.  The original
    5-deg Fibonacci grid points are overlaid as small black dots for reference.
    """
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    proj    = ccrs.Orthographic(central_longitude=cen_lon, central_latitude=cen_lat)

    fig, axes = plt.subplots(
        N_STATE, 2,
        figsize=(12, 4 * N_STATE),
        subplot_kw={"projection": proj},
    )

    title_str = "Parameter retrieval — posterior value and error vs truth"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(title_str, fontsize=11, y=1.005)

    ext_lat_min = grid_lats_5deg.min() - 3.0
    ext_lat_max = grid_lats_5deg.max() + 3.0
    ext_lon_min = grid_lons_5deg.min() - 3.0
    ext_lon_max = grid_lons_5deg.max() + 3.0

    for k, pname in enumerate(PARAM_NAMES):
        pst_vals = posterior_mean_5deg[k, :]    # (n_geo,)
        tru_vals = truth_mean_5deg[k, :]        # (n_geo,)
        err_vals = pst_vals - tru_vals

        for col_i, (vals, label, cmap, symm) in enumerate([
            (pst_vals, f"{pname}  posterior",          "viridis", False),
            (err_vals, f"{pname}  error (post−truth)", "RdBu_r",  True),
        ]):
            ax = axes[k, col_i]
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#444")
            ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
            ax.set_facecolor("#d8d8d8")
            ax.set_extent(
                [ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max],
                crs=ccrs.PlateCarree(),
            )

            finite_mask = np.isfinite(vals)
            if not finite_mask.any():
                ax.set_title(f"{label} — no data", fontsize=8)
                continue

            v_abs = float(np.nanpercentile(np.abs(vals[finite_mask]), 98)) or 1e-6
            if symm:
                vmin, vmax = -v_abs, v_abs
            else:
                vmin = float(np.nanmin(vals[finite_mask]))
                vmax = float(np.nanmax(vals[finite_mask]))
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-6

            LO, LA, vg = _meshgrid_interp(grid_lats_5deg, grid_lons_5deg, vals)
            _ax_pcolormesh(ax, LO, LA, vg, cmap, vmin, vmax,
                           fig, label, grid_lons_5deg, grid_lats_5deg)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §13b  EnKF Δ log10(NmF2) — filter correction vs. required correction
# ─────────────────────────────────────────────────────────────────────────────

def plot_enkf_delta_nmf2(
    prior_edp: np.ndarray,
    posterior_mean_5deg: np.ndarray,
    truth_mean_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: pd.Timestamp = None,
    save_path: str = "./Figures/test_param_iono/enkf_delta_nmf2.png",
) -> None:
    """
    Compare the filter correction to the required correction in log10(NmF2).

    prior_edp : (n_alt, n_geo) — ensemble-mean Ne prior, same as used in the
        EDP spaghetti plot.  NmF2 is derived as peak Ne along the altitude axis
        so both plots reference the same prior.

    Col 0  : Actual  Δ = truth − prior            (correction needed)
    Col 1  : Filter  Δ = posterior − prior         (EnKF update)
    Col 2  : Residual  = posterior − truth         (remaining error)

    Columns 0 and 1 share a symmetric colorbar so the magnitudes are directly
    comparable.  Column 2 has its own symmetric scale.
    """
    nmf2_prior = np.log10(np.maximum(prior_edp.max(axis=0), 1.0))  # (n_geo,)
    nmf2_post  = posterior_mean_5deg[0, :]
    nmf2_truth = truth_mean_5deg[0, :]

    actual_delta  = nmf2_truth - nmf2_prior   # what should have changed
    filter_delta  = nmf2_post  - nmf2_prior   # what the filter did
    residual      = nmf2_post  - nmf2_truth   # remaining error after update

    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    proj    = ccrs.Orthographic(central_longitude=cen_lon, central_latitude=cen_lat)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5),
        subplot_kw={"projection": proj},
    )

    title_str = "EnKF  Δ log₁₀(NmF₂) — filter correction vs. required correction"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(title_str, fontsize=11, y=1.02)

    ext_lat_min = grid_lats_5deg.min() - 3.0
    ext_lat_max = grid_lats_5deg.max() + 3.0
    ext_lon_min = grid_lons_5deg.min() - 3.0
    ext_lon_max = grid_lons_5deg.max() + 3.0

    # Shared symmetric scale for the two Δ panels
    shared_vals = np.concatenate([
        actual_delta[np.isfinite(actual_delta)],
        filter_delta[np.isfinite(filter_delta)],
    ])
    v_shared = float(np.nanpercentile(np.abs(shared_vals), 98)) or 1e-4

    # Residual scale
    res_fin  = residual[np.isfinite(residual)]
    v_resid  = float(np.nanpercentile(np.abs(res_fin), 98)) if res_fin.size else 1e-4
    v_resid  = v_resid or 1e-4

    col_specs = [
        (actual_delta, "Actual Δ = truth − prior",     "RdBu_r", v_shared),
        (filter_delta, "Filter Δ = posterior − prior", "RdBu_r", v_shared),
        (residual,     "Residual = posterior − truth", "RdBu_r", v_resid),
    ]

    for col_i, (vals, label, cmap, v_abs) in enumerate(col_specs):
        ax = axes[col_i]
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#444")
        ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
        ax.set_facecolor("#d8d8d8")
        ax.set_extent(
            [ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max],
            crs=ccrs.PlateCarree(),
        )

        if not np.isfinite(vals).any():
            ax.set_title(f"{label} — no data", fontsize=9)
            continue

        LO, LA, vg = _meshgrid_interp(grid_lats_5deg, grid_lons_5deg, vals)
        _ax_pcolormesh(ax, LO, LA, vg, cmap, -v_abs, v_abs,
                       fig, "Δ log₁₀(NmF₂)", grid_lons_5deg, grid_lats_5deg)
        ax.set_title(label, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_nmf2_by_index(
    prior_mean_5deg: np.ndarray,
    kf_result: dict,
    enkf_result: dict,
    truth_mean_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: pd.Timestamp = None,
    save_path: str = "./Figures/test_param_iono/nmf2_by_index.png",
) -> None:
    """
    Line plot of log10(NmF2) vs. 5-deg grid-point index.

    Prior, EnKF posterior, and truth use the log10(NmF2) Chapman parameter
    (index 0) directly.  The KF posterior is in Ne-space, so NmF2 is derived
    as log10(max-altitude Ne) — the F2 peak of the posterior Ne profile.

    A secondary x-axis shows the lat/lon of each grid node for spatial context.
    """
    n_geo = len(grid_lats_5deg)
    idx   = np.arange(n_geo)

    prior_nmf2 = prior_mean_5deg[0, :]
    enkf_nmf2  = enkf_result["posterior_mean_5deg"][0, :]
    truth_nmf2 = truth_mean_5deg[0, :]
    kf_nmf2    = np.log10(
        np.maximum(kf_result["posterior_ne_5deg"].max(axis=0), 1.0)
    )

    fig, ax = plt.subplots(figsize=(max(10, n_geo // 2 + 4), 5))

    ax.plot(idx, prior_nmf2, "o--", color="#888888", lw=1.5, ms=5,
            label="Prior (IRI baseline)")
    ax.plot(idx, truth_nmf2, "s-",  color="#e05050", lw=2.0, ms=6,
            label="Truth")
    ax.plot(idx, enkf_nmf2,  "^-",  color="#4da6ff", lw=1.8, ms=5,
            label="EnKF posterior")
    ax.plot(idx, kf_nmf2,    "D-",  color="#50c870", lw=1.8, ms=5,
            label="KF posterior (peak Ne)")

    ax.set_xlabel("Grid point index", fontsize=10)
    ax.set_ylabel("log₁₀(NmF₂)  [log₁₀ m⁻³]", fontsize=10)
    ax.set_xticks(idx)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9, loc="best")

    title_str = "log₁₀(NmF₂) by grid-point index — prior / KF / EnKF / truth"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    ax.set_title(title_str, fontsize=10)

    # Secondary x-axis: lat / lon labels for each grid node
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(idx)
    ax2.set_xticklabels(
        [f"{lat:.1f}°,{lon:.1f}°"
         for lat, lon in zip(grid_lats_5deg, grid_lons_5deg)],
        fontsize=5, rotation=60, ha="left",
    )
    ax2.set_xlabel("Grid (lat, lon)", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §13d  KF / EnKF EDP profile comparison (prior | posterior | ΔNe)
# ─────────────────────────────────────────────────────────────────────────────

def plot_edp_profiles_kf_enkf(
    kf_result: dict,
    enkf_result: dict,
    truth_ne_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    truth_time: pd.Timestamp = None,
    save_path: str = "./Figures/test_param_iono/edp_profiles_kf_enkf.png",
) -> None:
    """
    Two-row × four-column EDP profile figure, mirroring the structure of
    ``plot_edp_profiles`` in demo_ground_station_kf.py.

    Row 0 (top)    : Gridded Ne KF
    Row 1 (bottom) : Parametric EnKF

    Col 0  : Prior Ne(h)
    Col 1  : Posterior Ne(h)
    Col 2  : ΔNe(h) = posterior − prior      (filter update)
    Col 3  : ΔNe(h) = posterior − truth       (residual error vs truth)

    Each panel overlays all grid-point profiles (faint) with the centre grid
    point bolded and the spatial mean as a dashed black line.  Truth Ne is
    plotted in green on the Prior and Posterior panels.

    Prior/Posterior x-limits are shared across both rows for direct comparison.
    Both ΔNe columns share a common symmetric x-limit across all four panels
    so filter update and residual magnitudes are directly comparable.
    """
    _COL_PRI_FAINT  = "#7EB6D9"   # light steel-blue — non-centre prior
    _COL_PRI_BOLD   = "#1A5276"   # dark blue — centre prior
    _COL_POST_FAINT = "#F1948A"   # light tomato — non-centre posterior
    _COL_POST_BOLD  = "#922B21"   # dark red — centre posterior
    _COL_DELTA      = "#A0522D"   # sienna — ΔNe(post−prior) centre
    _COL_RESID      = "#6C3483"   # purple — ΔNe(post−truth) centre
    _COL_TRUTH      = "#27AE60"   # green — truth
    _COL_MEAN       = "black"

    n_geo = len(grid_lats_5deg)

    # Centre grid point: nearest node to spatial centroid
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    cen_idx = int(np.argmin(_haversine_km(cen_lat, cen_lon,
                                          grid_lats_5deg, grid_lons_5deg)))

    row_labels = ["Gridded Ne KF", "Parametric EnKF"]
    row_prior  = [kf_result["prior_edp"],     enkf_result["prior_edp"]]
    row_post   = [kf_result["posterior_edp"], enkf_result["posterior_edp"]]

    # ── Global x-limits ───────────────────────────────────────────────────────
    all_ne = np.concatenate([
        row_prior[0].ravel(),  row_prior[1].ravel(),
        row_post[0].ravel(),   row_post[1].ravel(),
        truth_ne_5deg.ravel(),
    ])
    all_ne_pos = all_ne[all_ne > 0]
    ne_xmax = float(np.nanpercentile(all_ne_pos, 99)) if all_ne_pos.size else 1e12
    ne_xmin = 0.0

    # Both Δ columns (update + residual) share the same symmetric limit so
    # magnitudes are directly comparable across the two rightmost columns.
    all_delta = np.concatenate([
        (row_post[0] - row_prior[0]).ravel(),
        (row_post[1] - row_prior[1]).ravel(),
        (row_post[0] - truth_ne_5deg).ravel(),
        (row_post[1] - truth_ne_5deg).ravel(),
    ])
    delta_abs = float(np.nanpercentile(np.abs(all_delta), 99)) if all_delta.size else 1.0
    delta_abs = max(delta_abs, 1.0)

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 4,
        figsize=(19, 9),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.35},
    )

    title_str = "EDP Profiles — Prior / Posterior / ΔNe(post−prior) / ΔNe(post−truth)  ·  KF vs EnKF"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(title_str, fontsize=11, y=1.01)

    for row in range(2):
        ax_pri  = axes[row, 0]
        ax_post = axes[row, 1]
        ax_del  = axes[row, 2]
        ax_res  = axes[row, 3]

        prior_edp  = row_prior[row]            # (n_alt, n_geo)
        post_edp   = row_post[row]
        delta_edp  = post_edp - prior_edp      # filter update
        resid_edp  = post_edp - truth_ne_5deg  # residual vs truth

        pri_mean   = prior_edp.mean(axis=1)
        post_mean  = post_edp.mean(axis=1)
        delta_mean = delta_edp.mean(axis=1)
        resid_mean = resid_edp.mean(axis=1)

        # ── Prior ──────────────────────────────────────────────────────────────
        for g in range(n_geo):
            if g == cen_idx:
                continue
            ax_pri.plot(np.maximum(prior_edp[:, g], 1.0), alt_grid,
                        color=_COL_PRI_FAINT, lw=0.6, alpha=0.35, zorder=2)
        ax_pri.plot(np.maximum(prior_edp[:, cen_idx], 1.0), alt_grid,
                    color=_COL_PRI_BOLD, lw=2.2, alpha=1.0, zorder=4,
                    label=f"Centre (#{cen_idx})")
        ax_pri.plot(np.maximum(pri_mean, 1.0), alt_grid,
                    color=_COL_MEAN, lw=1.4, ls="--", zorder=5, label="Mean")
        ax_pri.plot(np.maximum(truth_ne_5deg[:, cen_idx], 1.0), alt_grid,
                    color=_COL_TRUTH, lw=2.0, zorder=6, label="Truth")
        ax_pri.set_xlim(ne_xmin, ne_xmax)
        ax_pri.set_title(f"{row_labels[row]}\nPrior Ne(h)", fontsize=8)
        ax_pri.set_xlabel("Ne  (m⁻³)", fontsize=7)
        ax_pri.tick_params(labelsize=6)
        ax_pri.grid(True, alpha=0.3, ls=":")
        ax_pri.legend(fontsize=6, loc="upper right")

        # ── Posterior ──────────────────────────────────────────────────────────
        for g in range(n_geo):
            if g == cen_idx:
                continue
            ax_post.plot(np.maximum(post_edp[:, g], 1.0), alt_grid,
                         color=_COL_POST_FAINT, lw=0.6, alpha=0.35, zorder=2)
        ax_post.plot(np.maximum(post_edp[:, cen_idx], 1.0), alt_grid,
                     color=_COL_POST_BOLD, lw=2.2, alpha=1.0, zorder=4,
                     label=f"Centre (#{cen_idx})")
        ax_post.plot(np.maximum(post_mean, 1.0), alt_grid,
                     color=_COL_MEAN, lw=1.4, ls="--", zorder=5, label="Mean")
        ax_post.plot(np.maximum(truth_ne_5deg[:, cen_idx], 1.0), alt_grid,
                     color=_COL_TRUTH, lw=2.0, zorder=6, label="Truth")
        ax_post.set_xlim(ne_xmin, ne_xmax)
        ax_post.set_title(f"{row_labels[row]}\nPosterior Ne(h)", fontsize=8)
        ax_post.set_xlabel("Ne  (m⁻³)", fontsize=7)
        ax_post.tick_params(labelsize=6)
        ax_post.grid(True, alpha=0.3, ls=":")
        ax_post.legend(fontsize=6, loc="upper right")

        # ── ΔNe: filter update (post − prior) ──────────────────────────────────
        for g in range(n_geo):
            if g == cen_idx:
                continue
            ax_del.plot(delta_edp[:, g], alt_grid,
                        color=_COL_PRI_FAINT, lw=0.6, alpha=0.35, zorder=2)
        ax_del.plot(delta_edp[:, cen_idx], alt_grid,
                    color=_COL_DELTA, lw=2.2, alpha=1.0, zorder=4,
                    label=f"Centre (#{cen_idx})")
        ax_del.plot(delta_mean, alt_grid,
                    color=_COL_MEAN, lw=1.4, ls="--", zorder=5, label="Mean Δ")
        ax_del.axvline(0, color="gray", lw=0.8, ls=":", zorder=1)
        ax_del.set_xlim(-delta_abs, delta_abs)
        ax_del.set_title(f"{row_labels[row]}\nΔNe  (post − prior)", fontsize=8)
        ax_del.set_xlabel("ΔNe  (m⁻³)", fontsize=7)
        ax_del.tick_params(labelsize=6)
        ax_del.grid(True, alpha=0.3, ls=":")
        ax_del.legend(fontsize=6, loc="upper right")

        # ── ΔNe: residual vs truth (post − truth) ──────────────────────────────
        for g in range(n_geo):
            if g == cen_idx:
                continue
            ax_res.plot(resid_edp[:, g], alt_grid,
                        color=_COL_POST_FAINT, lw=0.6, alpha=0.35, zorder=2)
        ax_res.plot(resid_edp[:, cen_idx], alt_grid,
                    color=_COL_RESID, lw=2.2, alpha=1.0, zorder=4,
                    label=f"Centre (#{cen_idx})")
        ax_res.plot(resid_mean, alt_grid,
                    color=_COL_MEAN, lw=1.4, ls="--", zorder=5, label="Mean Δ")
        ax_res.axvline(0, color="gray", lw=0.8, ls=":", zorder=1)
        ax_res.set_xlim(-delta_abs, delta_abs)
        ax_res.set_title(f"{row_labels[row]}\nΔNe  (post − truth)", fontsize=8)
        ax_res.set_xlabel("ΔNe  (m⁻³)", fontsize=7)
        ax_res.tick_params(labelsize=6)
        ax_res.grid(True, alpha=0.3, ls=":")
        ax_res.legend(fontsize=6, loc="upper right")

    axes[0, 0].set_ylabel("Altitude  (km)", fontsize=8)
    axes[1, 0].set_ylabel("Altitude  (km)", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §14  Gridded Ne Kalman Filter — linear KF in direct Ne-space
# ─────────────────────────────────────────────────────────────────────────────

def run_gridded_ne_kf(
    arc_truth_list: list[dict],
    model_state_prior: "IonosphericState",
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    sigma_obs: float     = ENKF_SIGMA_OBS,
    max_update_rays: int = ENKF_MAX_UPDATE_RAYS,
) -> dict:
    """
    Standard linear Kalman Filter assimilation directly in Ne-space.

    State vector  : x  (n_alt * n_geo,)   — Ne at every altitude × grid point.
                    Layout: x[k * n_geo + g] = Ne(alt_k, gp_g)  [m⁻³]
    Forward model : H  (n_obs, n_alt * n_geo) — trapezoidal arc integration
                    H[i, k*n_geo+g] = phi_k(ray_i) × w_ig  [TECU / m⁻³]
    Update        : P_xy  S⁻¹  (y − H x̄)   — single-step linear update.

    Expects ``model_state_prior`` built with ``build_truth_state(mean_5deg)``:
      member 0 = IRI baseline at current conditions (no perturbation)
      members 1…N_STATE = single-parameter ±σ shifts

    Member 0's Ne profile is used as the prior mean x̄.  The full ensemble
    spread provides the background-error covariance.

    Returns
    -------
    dict with the same keys as ``run_parametric_enkf`` for direct comparison.
    """
    n_occ    = len(arc_truth_list)
    n_geo    = len(grid_lats_5deg)
    n_alt    = len(alt_grid)
    n_state  = n_alt * n_geo
    n_members = model_state_prior.n_members

    print(f"  [KF] State: {n_alt} alt × {n_geo} geo = {n_state} elements | "
          f"{n_members}-member prior ensemble (member 0 = IRI baseline)")

    # ── 1. Convert parametric ensemble → Ne-space ─────────────────────────────
    ne_ens = _parametric_to_edp_ensemble(
        model_state_prior, model_state_prior.ensemble, alt_grid,
    )  # (n_alt, n_geo, n_members)
    # Flatten: X[k * n_geo + g, m] = ne_ens[k, g, m]   (C-order)
    X = ne_ens.reshape(n_state, n_members)

    # Prior mean = member 0, the unperturbed IRI baseline at current conditions
    x_bar = X[:, 0].copy()                            # (n_state,)
    X_c   = X - x_bar[:, np.newaxis]                  # centered   (n_state, n_members)

    # ── 2. Decimate update rays (same stride as EnKF) ─────────────────────────
    rep_rays:          list  = []
    rep_tp_lats_list:  list  = []
    rep_tp_lons_list:  list  = []
    rep_tec_obs_list:  list  = []
    arc_update_counts: list  = []
    ray_counts:        list  = []
    arc_all_tec:       list  = []
    per_arc_tp_lats:   list  = []
    per_arc_tp_lons:   list  = []

    for arc in arc_truth_list:
        rays    = arc["rays"]
        tec     = arc["tec_truth"]
        tp_lats = arc["tp_lats"]
        tp_lons = arc["tp_lons"]
        n_s     = len(rays)

        per_arc_tp_lats.append(tp_lats)
        per_arc_tp_lons.append(tp_lons)
        ray_counts.append(n_s)
        arc_all_tec.append(tec)

        if n_s > max_update_rays:
            stride = int(np.ceil(n_s / max_update_rays))
            chosen = list(range(0, n_s, stride))
            if n_s - 1 not in chosen:
                chosen.append(n_s - 1)
        else:
            chosen = list(range(n_s))

        for idx in chosen:
            rep_rays.append(rays[idx])
            rep_tp_lats_list.append(float(tp_lats[idx]))
            rep_tp_lons_list.append(float(tp_lons[idx]))
            rep_tec_obs_list.append(float(tec[idx]))
        arc_update_counts.append(len(chosen))

    y_obs_arc = np.array(rep_tec_obs_list)
    y_obs_all = np.concatenate(arc_all_tec)
    n_obs     = len(rep_rays)
    print(f"  [KF] {n_obs} update rays across {n_occ} arcs")

    if n_obs >= n_members:
        print(f"  [KF] WARNING: n_obs ({n_obs}) >= n_members ({n_members}); "
              f"S = Y_c Y_c^T + R may be rank-deficient in ensemble block.")

    # ── 3. Build H matrix (n_obs, n_state) ───────────────────────────────────
    # H[i, k*n_geo+g] = phi_k(ray_i) × w_ig
    # Each row = np.outer(phi_i, w_i).ravel()  (shape: n_alt × n_geo → n_state)
    print(f"  [KF] Building H ({n_obs} × {n_state}) …")
    H = np.zeros((n_obs, n_state), dtype=float)
    for i in range(n_obs):
        phi_i = _precompute_ray_phi(rep_rays[i], alt_grid)      # (n_alt,)
        w_i   = _idw_weights(rep_tp_lats_list[i], rep_tp_lons_list[i],
                             grid_lats_5deg, grid_lons_5deg)     # (n_geo,)
        H[i]  = np.outer(phi_i, w_i).ravel()                    # (n_state,)

    # ── 4. Prior TEC and innovations ──────────────────────────────────────────
    y_prior_arc = H @ x_bar
    prior_inno  = y_obs_arc - y_prior_arc
    print(f"  [KF] Prior innovations  "
          f"mean={prior_inno.mean():.2f}  std={prior_inno.std():.2f}  "
          f"max_abs={np.abs(prior_inno).max():.2f} TECU")

    # ── 5. Ensemble-based KF update ───────────────────────────────────────────
    # Avoids forming P (n_state × n_state) explicitly.
    # P_yy = Y_c Y_c^T / (N−1),   P_xy = X_c Y_c^T / (N−1)
    # S    = P_yy + R
    # K    = P_xy S⁻¹
    # x_post = x_bar + K (y − H x_bar)
    Y_c   = H @ X_c                                   # (n_obs, n_members)
    nm1   = max(n_members - 1, 1)
    P_yy  = Y_c @ Y_c.T / nm1                         # (n_obs, n_obs)
    P_xy  = X_c @ Y_c.T / nm1                         # (n_state, n_obs)
    R_mat = (sigma_obs ** 2) * np.eye(n_obs)
    S     = P_yy + R_mat                              # (n_obs, n_obs)
    # Solve S x = P_xy^T, then K = result^T
    K     = np.linalg.solve(S.T, P_xy.T).T            # (n_state, n_obs)

    x_post  = x_bar + K @ prior_inno                  # (n_state,)
    ne_post = np.maximum(x_post.reshape(n_alt, n_geo), 0.0)

    # Linearized posterior ensemble anomaly for covariance visualization:
    # X_post_c ≈ (I − K H) X_c  (ignores obs-noise contribution K R K^T)
    X_post_c = X_c - K @ (H @ X_c)                  # (n_state, n_members)

    post_inno_arc = y_obs_arc - H @ x_post
    print(f"  [KF] Posterior innovations  "
          f"mean={post_inno_arc.mean():.2f}  std={post_inno_arc.std():.2f}  "
          f"max_abs={np.abs(post_inno_arc).max():.2f} TECU")

    # ── 6. Build H_all for all sample rays ───────────────────────────────────
    all_tp_lats_flat = np.concatenate([a.tolist() for a in per_arc_tp_lats])
    all_tp_lons_flat = np.concatenate([a.tolist() for a in per_arc_tp_lons])
    all_sample_rays  = [r for arc in arc_truth_list for r in arc["rays"]]
    n_all = len(all_sample_rays)

    print(f"  [KF] Building H_all ({n_all} × {n_state}) for RMSE …")
    H_all = np.zeros((n_all, n_state), dtype=float)
    for i in range(n_all):
        phi_i    = _precompute_ray_phi(all_sample_rays[i], alt_grid)
        w_i      = _idw_weights(float(all_tp_lats_flat[i]),
                               float(all_tp_lons_flat[i]),
                               grid_lats_5deg, grid_lons_5deg)
        H_all[i] = np.outer(phi_i, w_i).ravel()

    y_prior_all = H_all @ x_bar
    y_post_all  = H_all @ x_post

    prior_rmse = float(np.sqrt(np.nanmean((y_obs_all - y_prior_all) ** 2)))
    post_rmse  = float(np.sqrt(np.nanmean((y_obs_all - y_post_all) ** 2)))
    print(f"  [KF] Prior RMSE {prior_rmse:.3f} TECU  →  "
          f"Post RMSE {post_rmse:.3f} TECU")

    # ── 7. Per-arc stats and tec_slices ──────────────────────────────────────
    tec_slices:     list[dict] = []
    all_prior_resid = y_obs_all - y_prior_all
    all_post_resid  = y_obs_all - y_post_all

    arc_prior_mean_l, arc_post_mean_l = [], []
    arc_prior_rmse_l, arc_post_rmse_l = [], []
    arc_lats_l, arc_lons_l, arc_lbl_l = [], [], []

    soff = 0
    for i, arc in enumerate(arc_truth_list):
        n_s = ray_counts[i]
        sl  = slice(soff, soff + n_s)
        tec_slices.append(dict(
            tec_truth = arc["tec_truth"],
            prior_tec = y_prior_all[sl].copy(),
            post_tec  = y_post_all[sl].copy(),
            tang_km   = arc["tang_km"],
        ))
        rp = all_prior_resid[sl]
        ra = all_post_resid[sl]
        arc_prior_mean_l.append(float(np.nanmean(rp)))
        arc_post_mean_l.append(float(np.nanmean(ra)))
        arc_prior_rmse_l.append(float(np.sqrt(np.nanmean(rp ** 2))))
        arc_post_rmse_l.append(float(np.sqrt(np.nanmean(ra ** 2))))
        arc_lats_l.append(float(arc["tp_lats"].mean()))
        arc_lons_l.append(float(arc["tp_lons"].mean()))
        arc_lbl_l.append(f"{arc['conid']}{arc['prn_id']}")
        soff += n_s

    return dict(
        prior_ne_5deg       = x_bar.reshape(n_alt, n_geo),
        posterior_ne_5deg   = ne_post,
        prior_edp           = x_bar.reshape(n_alt, n_geo),
        posterior_edp       = ne_post,
        tec_slices          = tec_slices,
        prior_rmse          = prior_rmse,
        post_rmse           = post_rmse,
        all_prior_resid     = all_prior_resid,
        all_post_resid      = all_post_resid,
        arc_prior_mean      = np.array(arc_prior_mean_l),
        arc_post_mean       = np.array(arc_post_mean_l),
        arc_prior_rmse      = np.array(arc_prior_rmse_l),
        arc_post_rmse       = np.array(arc_post_rmse_l),
        arc_lats            = np.array(arc_lats_l),
        arc_lons            = np.array(arc_lons_l),
        arc_labels          = arc_lbl_l,
        grid_lats           = grid_lats_5deg,
        grid_lons           = grid_lons_5deg,
        mda_arc_means_list  = None,
        mda_flat_list       = None,
        prior_Xc            = X_c,       # (n_state, n_members) centered prior ensemble
        post_Xc             = X_post_c,  # (n_state, n_members) linearized posterior ensemble
    )


# ─────────────────────────────────────────────────────────────────────────────
# §15b  Covariance structure panels — KF and EnKF
# ─────────────────────────────────────────────────────────────────────────────

def plot_kf_covariance_panels(
    prior_Xc: np.ndarray,
    post_Xc: np.ndarray,
    prior_edp: np.ndarray,
    alt_grid: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: "pd.Timestamp | None" = None,
    save_path: str = "./Figures/test_param_iono/kf_covariance.png",
) -> None:
    """
    Four-panel covariance structure figure for the gridded Ne KF.

    Layout (2 rows × 2 cols):
      Row 0 — Prior:     Alt-Alt Ne correlation  |  Horizontal Ne correlation at hmF2
      Row 1 — Posterior: Alt-Alt Ne correlation  |  Horizontal Ne correlation at hmF2

    The alt-alt panels average the grid-point pair covariance over all geo
    vertices and normalise to a Pearson correlation matrix.

    The horizontal panels fix the centre grid point at the prior F2-peak
    altitude and plot the Pearson correlation of that state element with every
    other grid vertex at the same altitude, projected on an orthographic globe.

    Parameters
    ----------
    prior_Xc : (n_state, n_members) centered prior ensemble anomaly.
               Layout: x[k * n_geo + g] = Ne(alt_k, gp_g).
    post_Xc  : (n_state, n_members) linearized posterior ensemble anomaly,
               computed as (I − K H) X_c (ignores K R K^T obs-noise term).
    prior_edp : (n_alt, n_geo) prior-mean Ne profiles — used to locate hmF2.
    """
    import warnings

    n_state, n_members = prior_Xc.shape
    n_alt = len(alt_grid)
    n_geo = len(grid_lats_5deg)
    nm1   = max(n_members - 1, 1)

    # Centre grid point (nearest to spatial centroid)
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    cen_idx = int(np.argmin(_haversine_km(cen_lat, cen_lon,
                                          grid_lats_5deg, grid_lons_5deg)))

    # F2-peak reference altitude from the prior-mean centre profile
    alt_ref_idx  = int(np.argmax(prior_edp[:, cen_idx]))
    true_alt_ref = float(alt_grid[alt_ref_idx])

    def _alt_corr(Xc):
        # Average the alt-alt covariance over all geo points.
        # Xc shape: (n_alt * n_geo, n_members) → reshape to (n_alt, n_geo*n_members)
        # so that a single matrix product gives the geo-averaged alt-alt cov.
        Xm  = Xc.reshape(n_alt, n_geo * n_members)
        cov = (Xm @ Xm.T) / (n_geo * nm1)          # (n_alt, n_alt)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer = np.outer(std, std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov / np.where(outer == 0, 1e-10, outer)

    def _horiz_corr(Xc):
        Xc4 = Xc.reshape(n_alt, n_geo, n_members)
        ref  = Xc4[alt_ref_idx, cen_idx, :]         # (n_members,)
        std_ref = float(np.sqrt(max(ref @ ref / nm1, 0.0)))
        horiz = np.empty(n_geo)
        for g in range(n_geo):
            gp      = Xc4[alt_ref_idx, g, :]
            cov_g   = float(ref @ gp / nm1)
            std_g   = float(np.sqrt(max(gp @ gp / nm1, 0.0)))
            denom   = std_ref * std_g
            horiz[g] = cov_g / denom if denom > 1e-30 else 0.0
        return horiz

    prior_alt_corr = _alt_corr(prior_Xc)
    post_alt_corr  = _alt_corr(post_Xc)
    prior_horiz    = _horiz_corr(prior_Xc)
    post_horiz     = _horiz_corr(post_Xc)

    clon = float(np.nanmean(grid_lons_5deg))
    clat = float(np.nanmean(grid_lats_5deg))
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    alt_extent = [float(alt_grid[0]), float(alt_grid[-1]),
                  float(alt_grid[0]), float(alt_grid[-1])]

    title_str = "Gridded Ne KF — Prior and Posterior Covariance Structure"
    if truth_time is not None:
        title_str += f"\n{truth_time.strftime('%Y-%m-%d %H:%M')} UTC truth"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        title_str + f"\nHorizontal slice at {true_alt_ref:.0f} km  ·  ★ = centre vertex",
        fontsize=12,
    )
    gs = GridSpec(2, 2, figure=fig,
                  left=0.06, right=0.97, top=0.88, bottom=0.07,
                  wspace=0.30, hspace=0.35)

    for row, (row_lbl, alt_corr, horiz_corr) in enumerate([
        ("Prior",     prior_alt_corr, prior_horiz),
        ("Posterior", post_alt_corr,  post_horiz),
    ]):
        # ── Alt-Alt correlation ───────────────────────────────────────────────
        ax_aa = fig.add_subplot(gs[row, 0])
        pcm = ax_aa.imshow(
            alt_corr, cmap="coolwarm", vmin=-1, vmax=1,
            extent=alt_extent, origin="lower", aspect="auto",
        )
        ax_aa.axhline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.axvline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.set_xlabel("Altitude (km)", fontsize=9)
        ax_aa.set_ylabel("Altitude (km)", fontsize=9)
        ax_aa.set_title(f"{row_lbl} — Alt-Alt Ne Correlation", fontsize=10)
        fig.colorbar(pcm, ax=ax_aa, label="Pearson r", fraction=0.046, pad=0.04)

        # ── Horizontal correlation globe ──────────────────────────────────────
        ax_gl = fig.add_subplot(gs[row, 1], projection=proj)
        ax_gl.set_global()
        ax_gl.add_feature(cfeature.LAND,  facecolor="whitesmoke", zorder=0)
        ax_gl.add_feature(cfeature.OCEAN, facecolor="aliceblue",  zorder=0)
        ax_gl.add_feature(cfeature.COASTLINE.with_scale("110m"),
                          lw=0.4, edgecolor="gray")
        ax_gl.gridlines(lw=0.2, alpha=0.3)

        sc = ax_gl.scatter(
            grid_lons_5deg, grid_lats_5deg,
            c=horiz_corr, cmap="coolwarm", vmin=-1, vmax=1,
            s=80, transform=ccrs.Geodetic(), zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax_gl, orientation="horizontal",
                          shrink=0.75, pad=0.04, fraction=0.04)
        cb.set_label("Pearson r", fontsize=8)

        ax_gl.plot(
            float(grid_lons_5deg[cen_idx]), float(grid_lats_5deg[cen_idx]),
            transform=ccrs.Geodetic(),
            marker="*", color="gold", ms=14, mec="black", mew=0.8, zorder=8,
        )
        ax_gl.set_title(
            f"{row_lbl} — Horizontal Ne Correlation at {true_alt_ref:.0f} km",
            fontsize=10,
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_enkf_covariance_panels(
    prior_ensemble: np.ndarray,
    post_ensemble: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: "pd.Timestamp | None" = None,
    save_path: str = "./Figures/test_param_iono/enkf_covariance.png",
) -> None:
    """
    Four-panel covariance structure figure for the ParametricEnKF.

    Layout (2 rows × 2 cols):
      Row 0 — Prior:     8×8 parameter correlation  |  NmF2 spatial correlation globe
      Row 1 — Posterior: 8×8 parameter correlation  |  NmF2 spatial correlation globe

    The 8×8 panels show the cross-parameter Pearson correlation matrix averaged
    over all grid points (i.e. the mean covariance among the 8 Chapman/profile
    parameters at the same location).

    The globe panels show the Pearson correlation of each grid point's
    log₁₀(NmF₂) to the central grid point's log₁₀(NmF₂), mapped on an
    orthographic projection centred on the grid region.

    Parameters
    ----------
    prior_ensemble : (N_STATE, n_geo, n_members) ensemble BEFORE the EnKF update.
    post_ensemble  : (N_STATE, n_geo, n_members) ensemble AFTER the EnKF update.
    """
    import warnings

    n_params, n_geo, n_members = prior_ensemble.shape
    nm1 = max(n_members - 1, 1)

    # Centre grid point (nearest to spatial centroid)
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    cen_idx = int(np.argmin(_haversine_km(cen_lat, cen_lon,
                                          grid_lats_5deg, grid_lons_5deg)))

    def _param_corr(ens):
        """8×8 parameter correlation matrix averaged over all grid points."""
        ens_c = ens - ens.mean(axis=2, keepdims=True)   # centre over members
        cov = np.zeros((n_params, n_params))
        for g in range(n_geo):
            Xg   = ens_c[:, g, :]          # (N_STATE, n_members)
            cov += Xg @ Xg.T / nm1
        cov /= n_geo
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer = np.outer(std, std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov / np.where(outer == 0, 1e-10, outer)

    def _nmf2_spatial_corr(ens):
        """Pearson correlation of each grid point's log₁₀(NmF₂) to the centre."""
        ens_c   = ens - ens.mean(axis=2, keepdims=True)
        ref     = ens_c[0, cen_idx, :]             # (n_members,) — NmF₂ at centre
        std_ref = float(np.sqrt(max(ref @ ref / nm1, 0.0)))
        corr    = np.empty(n_geo)
        for g in range(n_geo):
            gp      = ens_c[0, g, :]
            cov_g   = float(ref @ gp / nm1)
            std_g   = float(np.sqrt(max(gp @ gp / nm1, 0.0)))
            denom   = std_ref * std_g
            corr[g] = cov_g / denom if denom > 1e-30 else 0.0
        return corr

    prior_param_corr = _param_corr(prior_ensemble)
    post_param_corr  = _param_corr(post_ensemble)
    prior_nmf2_corr  = _nmf2_spatial_corr(prior_ensemble)
    post_nmf2_corr   = _nmf2_spatial_corr(post_ensemble)

    clon = float(np.nanmean(grid_lons_5deg))
    clat = float(np.nanmean(grid_lats_5deg))
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    title_str = "ParametricEnKF — Prior and Posterior Covariance Structure"
    if truth_time is not None:
        title_str += f"\n{truth_time.strftime('%Y-%m-%d %H:%M')} UTC truth"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title_str + "  ·  ★ = centre vertex", fontsize=12)
    gs = GridSpec(2, 2, figure=fig,
                  left=0.06, right=0.97, top=0.90, bottom=0.07,
                  wspace=0.30, hspace=0.35)

    ticks = list(range(n_params))

    for row, (row_lbl, param_corr, nmf2_corr) in enumerate([
        ("Prior",     prior_param_corr, prior_nmf2_corr),
        ("Posterior", post_param_corr,  post_nmf2_corr),
    ]):
        # ── 8×8 parameter correlation ─────────────────────────────────────────
        ax_pp = fig.add_subplot(gs[row, 0])
        pcm = ax_pp.imshow(param_corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax_pp.set_xticks(ticks)
        ax_pp.set_yticks(ticks)
        ax_pp.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=8)
        ax_pp.set_yticklabels(PARAM_NAMES, fontsize=8)
        ax_pp.set_title(f"{row_lbl} — 8×8 Parameter Correlation (avg over grid)", fontsize=10)
        fig.colorbar(pcm, ax=ax_pp, label="Pearson r", fraction=0.046, pad=0.04)
        for i in range(n_params):
            for j in range(n_params):
                val = param_corr[i, j]
                ax_pp.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=6,
                           color="white" if abs(val) > 0.65 else "black")

        # ── NmF2 spatial correlation globe ────────────────────────────────────
        ax_gl = fig.add_subplot(gs[row, 1], projection=proj)
        ax_gl.set_global()
        ax_gl.add_feature(cfeature.LAND,  facecolor="whitesmoke", zorder=0)
        ax_gl.add_feature(cfeature.OCEAN, facecolor="aliceblue",  zorder=0)
        ax_gl.add_feature(cfeature.COASTLINE.with_scale("110m"),
                          lw=0.4, edgecolor="gray")
        ax_gl.gridlines(lw=0.2, alpha=0.3)

        sc = ax_gl.scatter(
            grid_lons_5deg, grid_lats_5deg,
            c=nmf2_corr, cmap="coolwarm", vmin=-1, vmax=1,
            s=80, transform=ccrs.Geodetic(), zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax_gl, orientation="horizontal",
                          shrink=0.75, pad=0.04, fraction=0.04)
        cb.set_label("Pearson r  (log₁₀(NmF₂) vs centre)", fontsize=8)

        ax_gl.plot(
            float(grid_lons_5deg[cen_idx]), float(grid_lats_5deg[cen_idx]),
            transform=ccrs.Geodetic(),
            marker="*", color="gold", ms=14, mec="black", mew=0.8, zorder=8,
        )
        ax_gl.set_title(
            f"{row_lbl} — NmF₂ Spatial Correlation to Centre",
            fontsize=10,
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §15  KF vs EnKF comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_kf_enkf_comparison(
    arc_truth_list: list[dict],
    kf_result: dict,
    enkf_result: dict,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    truth_time: "pd.Timestamp",
    save_path: str,
) -> None:
    """
    Side-by-side KF vs EnKF comparison figure.

    Layout — GridSpec(2, 3)
    ────────────────────────────────────────────────────────
    [0,0] GPS TEC      [0,1] GLONASS TEC    [0,2] EDP spaghetti
    [1,0] Galileo TEC  [1,1] BeiDou TEC     [1,2] RMSE comparison

    TEC panels
    ----------
    • Thick solid  : truth TEC (from 1-deg truth ionosphere)
    • Dashed       : prior model ensemble mean
    • Dash-dot blue: gridded Ne KF posterior
    • Dotted orange: ParametricEnKF posterior

    EDP panel
    ---------
    • Grey dashed  : prior Ne profiles (5-deg grid)
    • Blue solid   : KF posterior Ne profiles
    • Orange solid : EnKF posterior Ne profiles
    • Red dashed   : truth Ne at 5-deg grid centroid (1-deg grid nearest)

    RMSE panel
    ----------
    Grouped bar chart per arc plus global RMSE (prior / KF / EnKF).
    """
    from matplotlib.lines import Line2D

    occ_colors = _occ_colors(
        [{"conid": a["conid"], "prn_id": a["prn_id"]} for a in arc_truth_list]
    )

    fig = plt.figure(figsize=(22, 9), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    gs  = plt.GridSpec(2, 3, figure=fig,
                       width_ratios=[1, 1, 1.3],
                       wspace=0.38, hspace=0.45,
                       left=0.06, right=0.97, top=0.91, bottom=0.10)

    # ── TEC panels ───────────────────────────────────────────────────────────
    tec_axes: dict[str, plt.Axes] = {}
    first_ax = None
    for const, (row, col) in _CONST_POS.items():
        ax = fig.add_subplot(gs[row, col],
                             sharey=first_ax if first_ax is not None else None)
        ax.set_facecolor("#2b2b2b")
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "white"})
        ax.set_title(cfg["name"], color=cfg.get("title_color", "white"),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("sTEC (TECU)", color="lightgray", fontsize=7)
        ax.set_ylabel("Tang. alt. (km)", color="lightgray", fontsize=7)
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        tec_axes[const] = ax
        if first_ax is None:
            first_ax = ax

    # ── EDP spaghetti ─────────────────────────────────────────────────────────
    ax_edp = fig.add_subplot(gs[0, 2])
    ax_edp.set_facecolor("#2b2b2b")
    ax_edp.set_title("EDP — prior / KF / EnKF / truth", color="white", fontsize=8)
    ax_edp.set_xlabel("Ne (m⁻³)", color="lightgray", fontsize=7)
    ax_edp.set_ylabel("Altitude (km)", color="lightgray", fontsize=7)
    ax_edp.tick_params(colors="lightgray", labelsize=6)
    for sp in ax_edp.spines.values():
        sp.set_edgecolor("#555")

    kf_prior_edp   = kf_result["prior_edp"]      # (n_alt, n_geo)
    kf_post_edp    = kf_result["posterior_edp"]   # (n_alt, n_geo)
    enkf_post_edp  = enkf_result["posterior_edp"] # (n_alt, n_geo)
    n_geo          = kf_prior_edp.shape[1]

    for g in range(n_geo):
        ax_edp.plot(kf_prior_edp[:, g],   alt_grid, color="gray",
                    linewidth=0.6, alpha=0.4, linestyle="--")
        ax_edp.plot(kf_post_edp[:, g],    alt_grid, color="steelblue",
                    linewidth=0.8, alpha=0.7)
        ax_edp.plot(enkf_post_edp[:, g],  alt_grid, color="darkorange",
                    linewidth=0.8, alpha=0.7)

    # Truth at centroid of 5-deg grid
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    dist_1deg = _haversine_km(cen_lat, cen_lon, grid_lats_1deg, grid_lons_1deg)
    cen_idx   = int(np.argmin(dist_1deg))
    ax_edp.plot(truth_ne_1deg[:, cen_idx], alt_grid,
                color="red", linewidth=1.8, linestyle="--",
                label="Truth (centroid)")

    edp_handles = [
        Line2D([0], [0], color="gray",       lw=1.2, linestyle="--",
               label=f"Prior Ne"),
        Line2D([0], [0], color="steelblue",  lw=1.4, alpha=0.8,
               label=f"KF posterior"),
        Line2D([0], [0], color="darkorange", lw=1.4, alpha=0.8,
               label=f"EnKF posterior"),
        Line2D([0], [0], color="red",        lw=1.8, linestyle="--",
               label="Truth Ne (centroid)"),
    ]
    ax_edp.legend(handles=edp_handles, fontsize=6,
                  facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="upper right", framealpha=0.8)

    # ── RMSE comparison bar chart ─────────────────────────────────────────────
    ax_rmse = fig.add_subplot(gs[1, 2])
    ax_rmse.set_facecolor("#2b2b2b")
    ax_rmse.set_title("Per-arc RMSE: prior / KF / EnKF", color="white", fontsize=8)
    ax_rmse.set_xlabel("Arc", color="lightgray", fontsize=7)
    ax_rmse.set_ylabel("RMSE (TECU)", color="lightgray", fontsize=7)
    ax_rmse.tick_params(colors="lightgray", labelsize=6)
    for sp in ax_rmse.spines.values():
        sp.set_edgecolor("#555")

    arc_labels_kf   = kf_result["arc_labels"]
    kf_prior_rmse   = kf_result["arc_prior_rmse"]
    kf_post_rmse    = kf_result["arc_post_rmse"]
    enkf_post_rmse  = enkf_result["arc_post_rmse"]
    n_arcs          = len(arc_labels_kf)

    x_pos = np.arange(n_arcs, dtype=float)
    bw    = 0.24
    ax_rmse.bar(x_pos - bw,   kf_prior_rmse,  width=bw, color="#4c72b0", alpha=0.85,
                label=f"Prior (KF)")
    ax_rmse.bar(x_pos,        kf_post_rmse,   width=bw, color="#55a868", alpha=0.85,
                label=f"KF post")
    ax_rmse.bar(x_pos + bw,   enkf_post_rmse, width=bw, color="#c44e52", alpha=0.85,
                label=f"EnKF post")
    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels(arc_labels_kf, rotation=45, ha="right",
                             fontsize=6, color="lightgray")
    ax_rmse.legend(fontsize=6, facecolor="#2b2b2b", labelcolor="lightgray",
                   loc="upper right", framealpha=0.8)

    # Global RMSE as text
    gkf_rmse   = kf_result["post_rmse"]
    genkf_rmse = enkf_result["post_rmse"]
    gpr_rmse   = kf_result["prior_rmse"]
    ax_rmse.text(0.02, 0.97,
                 f"Global:  Prior {gpr_rmse:.2f}  KF {gkf_rmse:.2f}  EnKF {genkf_rmse:.2f} TECU",
                 transform=ax_rmse.transAxes, fontsize=6.5, color="lightgray",
                 va="top", ha="left")
    ax_rmse.grid(axis="y", lw=0.3, alpha=0.4)

    # ── TEC panels: truth / prior / KF / EnKF ────────────────────────────────
    style_placed = False
    kf_slices   = kf_result["tec_slices"]
    enkf_slices = enkf_result["tec_slices"]

    for i, (arc, col) in enumerate(zip(arc_truth_list, occ_colors)):
        const  = arc["conid"]
        label  = f"{arc['leo_id']} {const}{arc['prn_id']}"
        tang   = arc["tang_km"]
        ax     = tec_axes.get(const, tec_axes.get("G"))
        ksl    = kf_slices[i]
        esl    = enkf_slices[i]

        ax.plot(ksl["tec_truth"], tang, color=col,
                linewidth=2.2, zorder=6, label=label)
        ax.plot(ksl["prior_tec"], tang, color=col,
                linewidth=1.0, linestyle="--", alpha=0.55, zorder=3,
                label="Prior" if not style_placed else None)
        ax.plot(ksl["post_tec"], tang, color="steelblue",
                linewidth=1.4, linestyle="-.", alpha=0.9, zorder=4,
                label="KF post" if not style_placed else None)
        ax.plot(esl["post_tec"], tang, color="darkorange",
                linewidth=1.4, linestyle=":", alpha=0.9, zorder=5,
                label="EnKF post" if not style_placed else None)
        style_placed = True

    for ax in tec_axes.values():
        if ax.lines:
            ax.legend(fontsize=5, facecolor="#2b2b2b",
                      labelcolor="lightgray", loc="best", framealpha=0.7)

    fig.suptitle(
        f"KF vs EnKF comparison — truth {truth_time.strftime('%Y-%m-%d %H:%M')} UTC  "
        f"(+{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f})\n"
        f"Prior {gpr_rmse:.2f} TECU  →  KF {gkf_rmse:.2f} TECU  |  "
        f"EnKF {genkf_rmse:.2f} TECU",
        color="white", fontsize=10, y=0.98,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §6 (continued)  main() — EnKF retrieval experiment steps
# ─────────────────────────────────────────────────────────────────────────────

def _run_enkf_retrieval_experiment(
    parsed_list: list[dict],
    meta_list: list[dict],
    mean_5deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    time_dt: pd.Timestamp,
    save_dir: str = SAVE_DIR,
) -> None:
    """
    Execute the full filter retrieval experiment (EnKF + gridded Ne KF)
    and produce all diagnostic plots.

    Steps
    -----
    7  . Build truth ionosphere on the 1×1 Fibonacci grid (+1 h, F10.7+Δ).
         Also evaluate the truth at the 5×5 model grid for error comparison.
    8  . Generate synthetic sTEC measurements from the truth state.
    9  . Build shared prior ensemble (n=ENKF_N_MEMBERS).
    9a . Run ParametricEnKF (ES-MDA) — assimilate into Chapman-param state.
    9b . Run Gridded Ne KF (linear, single-step) — assimilate into Ne-space state.
    10 . Plot: EnKF TEC profiles + globe + EDP spaghetti.
    11 . Plot: EnKF per-arc innovation diagnostic.
    11b. Plot: KF per-arc innovation diagnostic.
    12 . Plot: EnKF 5×2 EDP spatial error.
    12b. Plot: KF 5×2 EDP spatial error.
    13 . Plot: 8×2 parameter spatial error (EnKF only — KF has no param state).
    14 . Plot: KF vs EnKF comparison (TEC, EDP, RMSE bar chart).
    """
    # ── Step 7: truth ionosphere ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 7: Building truth ionosphere "
          f"(1×1 grid, +{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f}) …")

    truth_time, truth_sdf = _truth_solar_conditions(time_dt)
    print(f"  Base time:  {time_dt}")
    print(f"  Truth time: {truth_time}  (F10.7 → {truth_sdf['f107'].iloc[0]:.1f})")

    truth_state_1deg, truth_ne_1deg, truth_mean_1deg = build_truth_iri_grid(
        truth_time, truth_sdf,
        grid_lats_1deg, grid_lons_1deg, ALT_GRID,
        label="1-deg truth",
    )

    # Truth at the 5-deg model grid for error comparison
    truth_state_5deg, truth_ne_5deg, truth_mean_5deg = build_truth_iri_grid(
        truth_time, truth_sdf,
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        label="5-deg truth (model grid)",
    )
    print(f"  Truth 1-deg: {truth_state_1deg.n_grid_points} grid points")
    print(f"  Truth 5-deg: {truth_state_5deg.n_grid_points} grid points")

    # ── Step 8: truth TEC measurements ───────────────────────────────────────
    print(f"\nStep 8: Generating truth sTEC from {len(parsed_list)} arcs …")
    arc_truth_list = generate_truth_tec(
        parsed_list, truth_state_1deg,
        grid_lats_1deg, grid_lons_1deg, ALT_GRID,
    )

    # ── Step 9a: EnKF assimilation ────────────────────────────────────────────
    print(f"\nStep 9a: Running ParametricEnKF "
          f"({ENKF_N_MEMBERS} members, ES-MDA {ENKF_N_MDA} iter) …")
    model_state_enkf = build_model_ensemble(
        mean_5deg, grid_lats_5deg, grid_lons_5deg,
        n_members=ENKF_N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
    )
    enkf_result = run_parametric_enkf(
        arc_truth_list, model_state_enkf,
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
    )

    # ── Step 9b: Gridded Ne KF assimilation ──────────────────────────────────
    print(f"\nStep 9b: Running gridded Ne KF (linear update, IRI baseline prior) …")
    # build_model_ensemble pins member 0 to the unperturbed IRI baseline;
    # the remaining members are drawn stochastically for the covariance.
    model_state_kf = build_model_ensemble(
        mean_5deg, grid_lats_5deg, grid_lons_5deg,
        n_members=ENKF_N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
    )
    kf_result = run_gridded_ne_kf(
        arc_truth_list, model_state_kf,
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
    )

    # ── Step 9c: KF covariance panels ────────────────────────────────────────
    print("\nStep 9c: Plotting KF covariance panels (alt-alt + horizontal) …")
    plot_kf_covariance_panels(
        prior_Xc       = kf_result["prior_Xc"],
        post_Xc        = kf_result["post_Xc"],
        prior_edp      = kf_result["prior_edp"],
        alt_grid       = ALT_GRID,
        grid_lats_5deg = grid_lats_5deg,
        grid_lons_5deg = grid_lons_5deg,
        truth_time     = truth_time,
        save_path      = os.path.join(save_dir, f"kf_covariance_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 9d: EnKF covariance panels ──────────────────────────────────────
    print("\nStep 9d: Plotting EnKF covariance panels (8×8 param + NmF2 spatial) …")
    plot_enkf_covariance_panels(
        prior_ensemble = enkf_result["prior_ensemble"],
        post_ensemble  = enkf_result["posterior_ensemble"],
        grid_lats_5deg = grid_lats_5deg,
        grid_lons_5deg = grid_lons_5deg,
        truth_time     = truth_time,
        save_path      = os.path.join(save_dir, f"enkf_covariance_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 10: TEC + globe + EDP plot (EnKF) ───────────────────────────────
    print("\nStep 10: Plotting EnKF TEC profiles, globe, and EDP spaghetti …")
    plot_enkf_tec_edp(
        arc_truth_list, enkf_result, truth_ne_1deg,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        ALT_GRID, truth_time,
        save_path=os.path.join(save_dir, f"enkf_tec_edp_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 10b: TEC + globe + EDP plot (KF) ────────────────────────────────
    print("\nStep 10b: Plotting KF TEC profiles, globe, and EDP spaghetti …")
    plot_kf_tec_edp(
        arc_truth_list, kf_result, truth_ne_1deg,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        ALT_GRID, truth_time,
        save_path=os.path.join(save_dir, f"kf_tec_edp_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 11: arc innovation diagnostic (EnKF) ────────────────────────────
    print("\nStep 11: Plotting EnKF per-arc innovation diagnostic …")
    _plot_arc_innovation_diagnostic(
        arc_labels          = enkf_result["arc_labels"],
        arc_prior_mean      = enkf_result["arc_prior_mean"],
        arc_post_mean       = enkf_result["arc_post_mean"],
        arc_prior_rmse      = enkf_result["arc_prior_rmse"],
        arc_post_rmse       = enkf_result["arc_post_rmse"],
        arc_lats            = enkf_result["arc_lats"],
        arc_lons            = enkf_result["arc_lons"],
        all_prior           = enkf_result["all_prior_resid"],
        all_post_main       = enkf_result["all_post_resid"],
        group_key           = f"{YYYY}_{DOY:03d}_enkf",
        save_dir            = save_dir,
        filter_name         = "ParametricEnKF",
        prior_rmse          = enkf_result["prior_rmse"],
        post_rmse           = enkf_result["post_rmse"],
        mda_arc_means_list  = enkf_result["mda_arc_means_list"],
        mda_flat_list       = enkf_result["mda_flat_list"],
    )

    # ── Step 11b: arc innovation diagnostic (KF) ──────────────────────────────
    print("\nStep 11b: Plotting KF per-arc innovation diagnostic …")
    _plot_arc_innovation_diagnostic(
        arc_labels          = kf_result["arc_labels"],
        arc_prior_mean      = kf_result["arc_prior_mean"],
        arc_post_mean       = kf_result["arc_post_mean"],
        arc_prior_rmse      = kf_result["arc_prior_rmse"],
        arc_post_rmse       = kf_result["arc_post_rmse"],
        arc_lats            = kf_result["arc_lats"],
        arc_lons            = kf_result["arc_lons"],
        all_prior           = kf_result["all_prior_resid"],
        all_post_main       = kf_result["all_post_resid"],
        group_key           = f"{YYYY}_{DOY:03d}_kf",
        save_dir            = save_dir,
        filter_name         = "GriddedNeKF",
        prior_rmse          = kf_result["prior_rmse"],
        post_rmse           = kf_result["post_rmse"],
        mda_arc_means_list  = None,
        mda_flat_list       = None,
    )

    # ── Step 12: EDP spatial error (5×2) — EnKF ──────────────────────────────
    print("\nStep 12: Plotting EnKF EDP spatial error (5×2 orthographic) …")
    plot_edp_spatial_error(
        truth_ne_5deg, enkf_result["posterior_ne_5deg"],
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        truth_time=truth_time,
        save_path=os.path.join(save_dir, f"edp_spatial_error_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 12b: EDP spatial error (5×3) — KF with truth column ─────────────
    print("\nStep 12b: Plotting KF EDP spatial error (5×3 orthographic) …")
    plot_edp_spatial_error(
        truth_ne_5deg, kf_result["posterior_ne_5deg"],
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        truth_time=truth_time,
        show_truth_col=True,
        save_path=os.path.join(save_dir, f"edp_spatial_error_kf_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 13: parameter spatial error (8×2) — EnKF only ───────────────────
    print("\nStep 13: Plotting parameter spatial error (8×2 orthographic) …")
    plot_parameter_spatial_error(
        truth_mean_5deg, enkf_result["posterior_mean_5deg"],
        grid_lats_5deg, grid_lons_5deg,
        truth_time=truth_time,
        save_path=os.path.join(save_dir, f"param_spatial_error_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 13b: EnKF Δ log10(NmF2) — filter vs. required correction ────────
    print("\nStep 13b: Plotting EnKF Δ log10(NmF2) correction comparison …")
    plot_enkf_delta_nmf2(
        enkf_result["prior_edp"], enkf_result["posterior_mean_5deg"], truth_mean_5deg,
        grid_lats_5deg, grid_lons_5deg,
        truth_time=truth_time,
        save_path=os.path.join(save_dir, f"enkf_delta_nmf2_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 13c: log10(NmF2) vs. grid-point index ───────────────────────────
    print("\nStep 13c: Plotting log10(NmF2) by grid-point index …")
    plot_nmf2_by_index(
        mean_5deg, kf_result, enkf_result, truth_mean_5deg,
        grid_lats_5deg, grid_lons_5deg,
        truth_time=truth_time,
        save_path=os.path.join(save_dir, f"nmf2_by_index_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 13d: EDP profiles — prior / posterior / ΔNe (KF top, EnKF bottom) ─
    print("\nStep 13d: Plotting EDP profile comparison (KF vs EnKF) …")
    plot_edp_profiles_kf_enkf(
        kf_result, enkf_result, truth_ne_5deg,
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        truth_time=truth_time,
        save_path=os.path.join(save_dir, f"edp_profiles_kf_enkf_{YYYY}_{DOY:03d}.png"),
    )

    # ── Step 14: KF vs EnKF comparison plot ──────────────────────────────────
    print("\nStep 14: Plotting KF vs EnKF comparison …")
    plot_kf_enkf_comparison(
        arc_truth_list    = arc_truth_list,
        kf_result         = kf_result,
        enkf_result       = enkf_result,
        truth_ne_1deg     = truth_ne_1deg,
        grid_lats_1deg    = grid_lats_1deg,
        grid_lons_1deg    = grid_lons_1deg,
        grid_lats_5deg    = grid_lats_5deg,
        grid_lons_5deg    = grid_lons_5deg,
        alt_grid          = ALT_GRID,
        truth_time        = truth_time,
        save_path         = os.path.join(
            save_dir, f"kf_vs_enkf_comparison_{YYYY}_{DOY:03d}.png"
        ),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Filter comparison summary")
    print("=" * 60)
    print(f"  Prior  RMSE:  {kf_result['prior_rmse']:.3f} TECU")
    print(f"  KF     RMSE:  {kf_result['post_rmse']:.3f} TECU"
          f"  (Δ = {kf_result['post_rmse'] - kf_result['prior_rmse']:+.3f})")
    print(f"  EnKF   RMSE:  {enkf_result['post_rmse']:.3f} TECU"
          f"  (Δ = {enkf_result['post_rmse'] - enkf_result['prior_rmse']:+.3f})")
    print(f"\n✓ EnKF + KF retrieval experiment complete.  Figures in: {save_dir}")


if __name__ == "__main__":
    main()
