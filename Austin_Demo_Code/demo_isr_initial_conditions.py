#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_isr_initial_conditions.py

For every day in the ISR EDP cache this script:

  1. Identifies the instrument (ESR at 78°N or JRO at -12°N) and records
     the solar conditions (F10.7, Ap, ig12, rz12) at that day's noon UTC.

  2. Builds two IRI-based background grids centred on the instrument:

       Voxel grid   (2°  spacing) — (n_alt, n_lat, n_lon) Ne array [m⁻³]
       Parametric   (2°  spacing) — (N_STATE, n_pts) Chapman 8-param state

     Both grids are saved as .npz in  Data/ISR_IC/  and are cached, so
     re-running only processes new days.

  3. Saves a JSON catalog  Data/ISR_IC/catalog.json  containing the solar
     conditions, grid filenames, and ISR coverage summary for every day.

  4. Produces one diagnostic PNG per day  (Figures/ISR_IRI_IC/)  showing
     all ISR Ne profiles for that day (colour-coded by hour) alongside the
     co-located IRI profile at noon.

Usage
-----
    python3.11 demo_isr_initial_conditions.py           # normal run (cached)
    python3.11 demo_isr_initial_conditions.py --force   # rebuild every day
    python3.11 demo_isr_initial_conditions.py --no-plot # skip figures
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

from scipy.optimize import minimize

from demo_esr_isr import load_edps, isr_days
from demo_compare_kf_enkf import (
    _solar_sampling_df,
    _get_iri_edp_and_features_batch,
    _state_from_iri_direct,
    _fit_log_rmse,
    _OPT_BOUNDS_PER_PARAM,
)
from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs
from Ionosphere_Tomography_Inverter.ionospheric_state import N_STATE, PARAM_NAMES
from Ionosphere_Tomography_Inverter.observation_operator import _ne_profile_ensemble

# ── Output directories ────────────────────────────────────────────────────────
IC_DIR   = ROOT / "Data"   / "ISR_IC"
FIG_DIR  = ROOT / "Figures" / "ISR_IRI_IC"
IC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = IC_DIR / "catalog.json"

# ── Altitude grid (matches test_param_iono.py) ───────────────────────────────
ALT_GRID = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)

# ── Grid spacings ─────────────────────────────────────────────────────────────
VOXEL_DEG = 2.0   # fine Ne grid
PARAM_DEG  = 2.0  # coarse Chapman-parameter grid

# ── Instrument definitions ────────────────────────────────────────────────────
INSTRUMENTS = {
    "ESR": {
        "lat": 78.09, "lon": 16.02,
        "lat_bounds": (60.0, 88.0),
        "lon_bounds": (-20.0, 60.0),
        "label": "EISCAT Svalbard Radar",
    },
    "TRO": {
        "lat": 69.583, "lon": 19.21,
        "lat_bounds": (55.0, 80.0),
        "lon_bounds": (-5.0,  45.0),
        "label": "EISCAT Tromsø UHF Radar",
    },
    "JRO": {
        "lat": -11.95, "lon": -76.87,   # 283.13°E → -76.87°E
        "lat_bounds": (-25.0, 15.0),
        "lon_bounds": (-105.0, -45.0),  # 255–315°E → -105 to -45°E
        "label": "Jicamarca IS Radar",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Instrument identification
# ─────────────────────────────────────────────────────────────────────────────

def _identify_instrument(lat: float) -> str:
    """Map instrument site latitude to instrument key."""
    if lat > 75.0:
        return "ESR"   # EISCAT Svalbard (78.09°N)
    elif lat > 30.0:
        return "TRO"   # EISCAT Tromsø   (69.58°N)
    else:
        return "JRO"   # Jicamarca       (-11.95°N)


# ─────────────────────────────────────────────────────────────────────────────
# Regular lat/lon grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid(lat_bounds: tuple, lon_bounds: tuple, spacing: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats_1d, lons_1d) covering the bounding box at *spacing* degrees."""
    lats = np.arange(lat_bounds[0], lat_bounds[1] + spacing * 0.5, spacing)
    lons = np.arange(lon_bounds[0], lon_bounds[1] + spacing * 0.5, spacing)
    return lats, lons


def _flatten_grid(lats_1d: np.ndarray, lons_1d: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_flat, lon_flat) as all (lat, lon) pairs on the grid."""
    lat_g, lon_g = np.meshgrid(lats_1d, lons_1d, indexing="ij")
    return lat_g.ravel(), lon_g.ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Solar condition extractor
# ─────────────────────────────────────────────────────────────────────────────

def get_solar_conditions(time_dt: pd.Timestamp) -> dict:
    """Return solar-index scalars for *time_dt* from IRI_Sample_Inputs."""
    inp = IRI_Sample_Inputs(time_dt.strftime("%Y-%m-%d %H:%M:%S"))
    return {
        "f107":     float(inp.apf107["f107"][inp.current_idx_f107]),
        "f107_81":  float(inp.apf107["f107_81"][inp.current_idx_f107]),
        "ap":       int(inp.apf107["iapda"][inp.current_idx_f107]),
        "ig12":     float(inp.ig_rz["ig"][inp.current_idx_igrz]),
        "rz12":     float(inp.ig_rz["rz"][inp.current_idx_igrz]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IRI grid builders
# ─────────────────────────────────────────────────────────────────────────────

def build_voxel_grid(
    time_dt: pd.Timestamp,
    inst_name: str,
    spacing: float = VOXEL_DEG,
    cache_dir: Path = IC_DIR,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a (n_alt, n_lat, n_lon) IRI Ne voxel grid.

    Returns
    -------
    lats_1d, lons_1d, alt_km, ne_m3
    """
    inst  = INSTRUMENTS[inst_name]
    lats_1d, lons_1d = _make_grid(inst["lat_bounds"], inst["lon_bounds"], spacing)
    tag  = (f"{inst_name}_{time_dt.year}_{time_dt.dayofyear:03d}_"
            f"voxel_{spacing:.0f}deg")
    path = cache_dir / f"{tag}.npz"

    if path.exists() and not force:
        d = np.load(path)
        return d["lats"], d["lons"], d["alt_km"], d["ne_m3"]

    lat_f, lon_f = _flatten_grid(lats_1d, lons_1d)
    print(f"  [voxel] IRI @ {len(lat_f)} pts …", flush=True)
    sampling_df = _solar_sampling_df(time_dt)
    ne_flat, _ = _get_iri_edp_and_features_batch(
        time_dt, lat_f, lon_f, ALT_GRID, sampling_df=sampling_df
    )
    ne_m3 = ne_flat.reshape(len(ALT_GRID), len(lats_1d), len(lons_1d))

    np.savez_compressed(path,
        lats=lats_1d, lons=lons_1d, alt_km=ALT_GRID, ne_m3=ne_m3)
    return lats_1d, lons_1d, ALT_GRID, ne_m3


def build_parametric_grid(
    time_dt: pd.Timestamp,
    inst_name: str,
    spacing: float = PARAM_DEG,
    cache_dir: Path = IC_DIR,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a (N_STATE, n_pts) Chapman 8-parameter state grid.

    Returns
    -------
    lats_1d, lons_1d, mean_state (N_STATE, n_pts), ne_m3 (n_alt, n_pts)
    """
    inst  = INSTRUMENTS[inst_name]
    lats_1d, lons_1d = _make_grid(inst["lat_bounds"], inst["lon_bounds"], spacing)
    tag  = (f"{inst_name}_{time_dt.year}_{time_dt.dayofyear:03d}_"
            f"param_{spacing:.0f}deg")
    path = cache_dir / f"{tag}.npz"

    if path.exists() and not force:
        d = np.load(path)
        return d["lats"], d["lons"], d["mean_state"], d["ne_m3"]

    lat_f, lon_f = _flatten_grid(lats_1d, lons_1d)
    n_pts = len(lat_f)
    print(f"  [param] IRI @ {n_pts} pts …", flush=True)
    sampling_df = _solar_sampling_df(time_dt)
    ne_flat, feat_flat = _get_iri_edp_and_features_batch(
        time_dt, lat_f, lon_f, ALT_GRID, sampling_df=sampling_df
    )
    mean_state = np.empty((N_STATE, n_pts))
    for g in range(n_pts):
        mean_state[:, g] = _state_from_iri_direct(
            ne_flat[:, g], feat_flat[:, g], ALT_GRID
        )

    np.savez_compressed(path,
        lats=lats_1d, lons=lons_1d,
        mean_state=mean_state, ne_m3=ne_flat,
        alt_km=ALT_GRID, param_names=np.array(PARAM_NAMES))
    return lats_1d, lons_1d, mean_state, ne_flat


# ─────────────────────────────────────────────────────────────────────────────
# IRI helper: profile at the instrument site for an arbitrary time
# ─────────────────────────────────────────────────────────────────────────────

def _iri_at_instrument(
    time_dt: pd.Timestamp,
    inst_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (iri_ne, feat) at the instrument location at *time_dt*.
    iri_ne : (n_alt,) m⁻³  on ALT_GRID
    feat   : (13,)  IRI feature vector
    """
    inst = INSTRUMENTS[inst_name]
    sampling_df = _solar_sampling_df(time_dt)
    ne, feat = _get_iri_edp_and_features_batch(
        time_dt,
        np.array([inst["lat"]]),
        np.array([inst["lon"]]),
        ALT_GRID,
        sampling_df=sampling_df,
    )
    # IRI2020 can return NaN/Inf for isolated (time, location) edge cases;
    # floor to 1 m⁻³ (same convention as run_kf_window's P_f regularisation)
    # so a bad Fortran call can't propagate NaN into the KF state/covariance.
    ne_col = np.nan_to_num(ne[:, 0], nan=1.0, posinf=1.0, neginf=1.0)
    ne_col = np.maximum(ne_col, 1.0)
    return ne_col, feat[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Chapman 8-parameter fit to an ISR profile
# ─────────────────────────────────────────────────────────────────────────────

def _fit_chapman_to_profile(
    ne_obs: np.ndarray,
    alt_obs: np.ndarray,
    iri_ne: np.ndarray,
    iri_feat: np.ndarray,
    alt_min_km: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit the 8-parameter Chapman/IRI model to a single ISR Ne profile.

    Uses the IRI profile at the same location/time as the initial guess and
    optimises with L-BFGS-B in log₁₀-RMSE on the common altitude range.

    Parameters
    ----------
    ne_obs     : (n_obs,) measured Ne [m⁻³] on alt_obs.
    alt_obs    : (n_obs,) altitude [km] of observations.
    iri_ne     : (n_alt,) IRI Ne on ALT_GRID (initial-guess forward model).
    iri_feat   : (13,)    IRI feature vector for the same location/time.
    alt_min_km : ignore ISR gates below this altitude (noisy E-region floor).

    Returns
    -------
    params_fit : (N_STATE,)  fitted parameters in log/linear mixed convention.
    ne_fit     : (n_alt,)    fitted Ne profile on ALT_GRID [m⁻³].
    rmse       : log₁₀-RMSE of the fit evaluated on the common altitude range.
    """
    # ── Initial guess from IRI at same location/time ──────────────────────────
    x0 = _state_from_iri_direct(iri_ne, iri_feat, ALT_GRID)

    # ── Interpolate ISR obs onto ALT_GRID, masking bad gates ─────────────────
    valid = np.isfinite(ne_obs) & (ne_obs > 1e6) & (alt_obs >= alt_min_km)
    if valid.sum() < 5:
        ne_fit = _eval_chapman(x0)
        return x0, ne_fit, np.nan

    ne_interp = np.interp(ALT_GRID, alt_obs[valid], ne_obs[valid],
                          left=np.nan, right=np.nan)
    mask = np.isfinite(ne_interp) & (ne_interp > 1e6)
    if mask.sum() < 5:
        ne_fit = _eval_chapman(x0)
        return x0, ne_fit, np.nan

    ne_target = ne_interp.copy()
    ne_target[~mask] = np.nan

    # ── Objective: log₁₀-RMSE on valid gates only ────────────────────────────
    def cost(p):
        return _fit_log_rmse(p, ne_target, ALT_GRID)

    bounds = _OPT_BOUNDS_PER_PARAM
    result = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 300, "ftol": 1e-9})

    params_fit = result.x
    ne_fit     = _eval_chapman(params_fit)
    rmse       = float(result.fun)
    return params_fit, ne_fit, rmse


def _eval_chapman(params_log: np.ndarray) -> np.ndarray:
    """Evaluate Chapman Ne profile on ALT_GRID from log/linear params."""
    lin = params_log.copy()
    lin[0] = 10.0 ** params_log[0]   # NmF2
    lin[6] = 10.0 ** params_log[6]   # NmE
    ne = _ne_profile_ensemble(ALT_GRID, lin[:, np.newaxis])[:, 0]
    return np.maximum(ne, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4-panel comparison plot  (one panel per 6-hour UTC window)
# ─────────────────────────────────────────────────────────────────────────────

# Instrument display colours
_INST_COLOR = {"ESR": "crimson", "TRO": "steelblue", "JRO": "forestgreen"}
_INST_SHORT = {"ESR": "ESR (Svalbard)", "TRO": "TRO (Tromsø)", "JRO": "JRO (Peru)"}

# 6-hour windows: (start_h, end_h, central_h, panel_title)
_WINDOWS = [
    (0,  6,  3,  "00:00 – 06:00 UTC"),
    (6,  12, 9,  "06:00 – 12:00 UTC"),
    (12, 18, 15, "12:00 – 18:00 UTC"),
    (18, 24, 21, "18:00 – 24:00 UTC"),
]


def plot_isr_vs_iri(
    day_edps: list[dict],
    inst_name: str,
    day: object,
    solar: dict,
    save_dir: Path = FIG_DIR,
) -> None:
    """
    Four-panel plot — one panel per 6-hour UTC window.

    Each panel shows:
      • All ISR Ne profiles in that window (thin, coloured by instrument).
      • Hourly-median ISR profiles with time labels in the legend.
      • 8-parameter Chapman fit to the window's aggregate median (bold dashed).
      • IRI profile at the window's central time (orange solid).
    """
    inst      = INSTRUMENTS[inst_name]
    isr_color = _INST_COLOR[inst_name]
    isr_short = _INST_SHORT[inst_name]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, sharex=True)
    axes_flat = axes.ravel()

    fig.suptitle(
        f"{inst['label']}  ·  {day}   "
        f"F10.7 = {solar['f107']:.0f}    Ap = {solar['ap']}    "
        f"ig12 = {solar['ig12']:.0f}    rz12 = {solar['rz12']:.0f}",
        fontsize=12, fontweight="bold",
    )

    for panel_idx, (h0, h1, h_ctr, win_title) in enumerate(_WINDOWS):
        ax = axes_flat[panel_idx]

        # ── Collect profiles in this window ───────────────────────────────────
        win_edps = [e for e in day_edps if h0 <= e["time"].hour < h1]

        if not win_edps:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=11)
            ax.set_title(win_title, fontsize=10)
            continue

        # ── Raw traces (thin background) ──────────────────────────────────────
        for edp in win_edps:
            ax.plot(edp["ne_m3"], edp["alt_km"],
                    color=isr_color, alpha=0.08, lw=0.6)

        # ── Hourly medians with legend labels ─────────────────────────────────
        alt_ref   = win_edps[0]["alt_km"]
        hourly_ne = {}
        for h in range(h0, h1):
            bucket = [e for e in win_edps if e["time"].hour == h]
            if not bucket:
                continue
            ne_mat = np.vstack([np.interp(alt_ref, e["alt_km"], e["ne_m3"])
                                 for e in bucket])
            hourly_ne[h] = np.nanmedian(ne_mat, axis=0)
            label = f"{isr_short}  {h:02d}:30 UTC  (n={len(bucket)})"
            ax.plot(hourly_ne[h], alt_ref,
                    color=isr_color, lw=1.6, alpha=0.85, label=label)

        # ── 6-hour aggregate median for Chapman fit ───────────────────────────
        ne_all = np.vstack([np.interp(alt_ref, e["alt_km"], e["ne_m3"])
                             for e in win_edps])
        win_median_ne  = np.nanmedian(ne_all, axis=0)
        win_median_alt = alt_ref

        # IRI prior at central time for fit initialisation
        ctr_dt = pd.Timestamp(
            year=day.year, month=day.month, day=day.day,
            hour=h_ctr, tz="UTC"
        )
        iri_ne_ctr, iri_feat_ctr = _iri_at_instrument(ctr_dt, inst_name)

        params_fit, ne_fit, rmse = _fit_chapman_to_profile(
            win_median_ne, win_median_alt, iri_ne_ctr, iri_feat_ctr
        )
        rmse_str = f"{rmse:.3f}" if np.isfinite(rmse) else "n/a"
        fit_label = f"{isr_short}  Chapman fit  (log₁₀-RMSE = {rmse_str})"
        ax.plot(ne_fit, ALT_GRID,
                color=isr_color, lw=2.2, ls="--", label=fit_label)

        # ── IRI at central time (orange) ──────────────────────────────────────
        iri_label = f"IRI  {h_ctr:02d}:00 UTC"
        ax.plot(iri_ne_ctr, ALT_GRID,
                color="darkorange", lw=2.0, ls="-", label=iri_label)

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_xscale("log")
        ax.set_xlim(1e9, 1e12)
        ax.set_ylim(50, 900)
        ax.set_title(win_title, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(labelsize=8)

        legend = ax.legend(fontsize=7.5, loc="upper left",
                           framealpha=0.85, handlelength=2.0)

        if panel_idx in (0, 2):
            ax.set_ylabel("Altitude  (km)", fontsize=10)
        if panel_idx in (2, 3):
            ax.set_xlabel("Ne  (m⁻³)", fontsize=10)

    fname = save_dir / f"{inst_name}_{day}_edp_comparison.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ISR initial-condition grids and comparison plots.")
    parser.add_argument("--force",   action="store_true",
                        help="Rebuild IRI grids even if cached.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip diagnostic plots.")
    args = parser.parse_args()

    # ── Load ISR EDP cache ────────────────────────────────────────────────────
    print("[ISR] Loading EDP cache …")
    edps = load_edps()
    days = isr_days(edps)
    print(f"[ISR] {len(edps)} profiles across {len(days)} days\n")

    # Load existing catalog (if any)
    catalog: list[dict] = []
    done_keys: set[str] = set()
    if CATALOG_PATH.exists() and not args.force:
        with open(CATALOG_PATH) as fh:
            catalog = json.load(fh)
        done_keys = {f"{e['instrument']}_{e['date']}" for e in catalog}

    # ── Process each day ──────────────────────────────────────────────────────
    for day in days:
        day_edps = [e for e in edps if e["time"].date() == day]
        inst_name = _identify_instrument(day_edps[0]["lat"])
        key = f"{inst_name}_{day}"

        if key in done_keys and not args.force:
            print(f"[skip] {key} already in catalog")
            continue

        print(f"\n── {day}  ({inst_name}, {len(day_edps)} profiles) ──")

        # ── 1. Solar conditions at noon UTC ───────────────────────────────────
        noon_dt = pd.Timestamp(
            year=day.year, month=day.month, day=day.day,
            hour=12, tz="UTC"
        )
        solar = get_solar_conditions(noon_dt)
        print(f"  Solar: F10.7={solar['f107']:.0f}  Ap={solar['ap']}"
              f"  ig12={solar['ig12']:.0f}  rz12={solar['rz12']:.0f}")

        # ── 2. Voxel grid ─────────────────────────────────────────────────────
        lats_v, lons_v, alt_km, ne_vox = build_voxel_grid(
            noon_dt, inst_name, VOXEL_DEG, force=args.force
        )
        voxel_file = (f"{inst_name}_{day.year}_{day.timetuple().tm_yday:03d}"
                      f"_voxel_{VOXEL_DEG:.0f}deg.npz")
        print(f"  Voxel  grid: {len(lats_v)}×{len(lons_v)} lat/lon, "
              f"{len(alt_km)} alt levels")

        # ── 3. Parametric grid ────────────────────────────────────────────────
        lats_p, lons_p, mean_state, ne_param = build_parametric_grid(
            noon_dt, inst_name, PARAM_DEG, force=args.force
        )
        param_file = (f"{inst_name}_{day.year}_{day.timetuple().tm_yday:03d}"
                      f"_param_{PARAM_DEG:.0f}deg.npz")
        n_pts = mean_state.shape[1]
        print(f"  Param  grid: {len(lats_p)}×{len(lons_p)} lat/lon → {n_pts} pts")
        print(f"  State ranges: "
              + "  ".join(f"{PARAM_NAMES[i]}=[{mean_state[i].min():.2g},"
                          f"{mean_state[i].max():.2g}]"
                          for i in range(N_STATE)))

        # ── 4. Catalog entry ──────────────────────────────────────────────────
        alt_all = np.concatenate([e["alt_km"] for e in day_edps])
        ne_all  = np.concatenate([e["ne_m3"]  for e in day_edps])
        entry = {
            "date":               str(day),
            "instrument":         inst_name,
            "instrument_label":   INSTRUMENTS[inst_name]["label"],
            "lat":                INSTRUMENTS[inst_name]["lat"],
            "lon":                INSTRUMENTS[inst_name]["lon"],
            "doy":                int(day.timetuple().tm_yday),
            "n_profiles":         len(day_edps),
            "n_edp_points":       int((~np.isnan(ne_all)).sum()),
            "alt_km_min":         float(np.nanmin(alt_all)),
            "alt_km_max":         float(np.nanmax(alt_all)),
            "ne_m3_median":       float(np.nanmedian(ne_all)),
            "ne_m3_p10":          float(np.nanpercentile(ne_all, 10)),
            "ne_m3_p90":          float(np.nanpercentile(ne_all, 90)),
            "representative_time_utc": str(noon_dt),
            "solar":              solar,
            "voxel_grid_file":    voxel_file,
            "param_grid_file":    param_file,
            "voxel_spacing_deg":  VOXEL_DEG,
            "param_spacing_deg":  PARAM_DEG,
            "param_names":        PARAM_NAMES,
            "param_mean_min":     [float(mean_state[i].min()) for i in range(N_STATE)],
            "param_mean_max":     [float(mean_state[i].max()) for i in range(N_STATE)],
        }
        catalog = [e for e in catalog
                   if not (e["instrument"] == inst_name and e["date"] == str(day))]
        catalog.append(entry)
        catalog.sort(key=lambda e: (e["date"], e["instrument"]))

        with open(CATALOG_PATH, "w") as fh:
            json.dump(catalog, fh, indent=2)

        # ── 5. Comparison plot ────────────────────────────────────────────────
        if not args.no_plot:
            plot_isr_vs_iri(day_edps, inst_name, day, solar)

    print(f"\n[done] Catalog: {CATALOG_PATH}  ({len(catalog)} entries)")
    print(f"       Grids:   {IC_DIR}/")
    print(f"       Figures: {FIG_DIR}/")


if __name__ == "__main__":
    main()
