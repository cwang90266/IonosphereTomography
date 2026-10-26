#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_occultation_animation.py

Animate a single GNSS radio-occultation (RO) pass straight from a real podTc2
NetCDF file, so the link between the *geometry* of the pass and the *measured
TEC* is visible frame by frame.

Left panel  — the occultation-plane geometry (Earth disk, the 60-700 km
              ionosphere shell shaded, the LEO orbit arc, and the LEO->GNSS ray
              sweeping down through the atmosphere). The stretch of ray that lies
              inside the 60-700 km ionosphere is drawn thick/highlighted and the
              tangent point (ray's closest approach to Earth) is marked, colour-
              coded by whether it is currently inside that band.
Right-top   — the measured slant TEC versus time, drawn up to the current frame.
Right-bot   — the classic RO profile: TEC versus tangent altitude, built up
              sample by sample, with the 60-700 km ionosphere band shaded.

Geometry note
-------------
The three points {Earth-centre, LEO_i, GNSS_i} plus the tangent point are exactly
coplanar for every sample (the tangent point lies on the LEO->GNSS chord). Over
the ~6-minute occultation sweep the LEO moves only ~20 deg of its orbit, so a
single plane taken at the middle of the sweep reproduces the tangent-point radius
to <3 km. We therefore project every sample onto that one fixed plane, which lets
the Earth stay still while the LEO visibly travels its arc and the ray sweeps the
limb — an honest 2-D rendering of an almost-planar 3-D geometry.

The podTc2 x/y/z_LEO and x/y/z_GPS variables are ECEF kilometres; TEC is in TECU.

Optional `--voxels` overlay
----------------------------
Bins the ray's 60-700 km in-band segment into a lat/lon/alt grid at the
project's EnKF state-grid resolution (see `build_voxel_grid`) and highlights
whichever cells the ray has swept through, coloured viridis and faded by how
long ago (bright/opaque = just crossed, dark/transparent = about to drop out).
Works in both `--mode 2d` and `--mode 3d`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import netCDF4
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Annulus, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

sys.path.insert(0, str(Path(__file__).parent))
from TEC_model.podTc_file_processing import rayTangent  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
PODTC_BASE = Path("/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2")
PODTC_SUFFIX = ".0001_nc"
EARTH_R = 6371.0                 # km, mean Earth radius used by rayTangent
IONO_MIN_KM, IONO_MAX_KM = 60.0, 700.0
FIGURES_DIR = Path(__file__).parent / "Figures" / "Occultation_Animation"

_BAND_COLOR = "#ffd24d"          # 60-700 km ionosphere highlight
_RAY_COLOR = "#3a86ff"
_LEO_COLOR = "#e63946"
_EARTH_COLOR = "#c9d6df"


# ── data loading ─────────────────────────────────────────────────────────────
def load_occultation(fpath: Path) -> dict:
    """
    Read one podTc2 file and return only the occultation *sweep* — the contiguous
    stretch where the ray's tangent point has detached from the LEO and descends
    through the atmosphere. Positions are ECEF km, TEC in TECU.
    """
    with netCDF4.Dataset(str(fpath), "r") as nc:
        LEO = np.array([nc.variables["x_LEO"][:], nc.variables["y_LEO"][:],
                        nc.variables["z_LEO"][:]], dtype=float)
        GNSS = np.array([nc.variables["x_GPS"][:], nc.variables["y_GPS"][:],
                         nc.variables["z_GPS"][:]], dtype=float)
        tec = np.asarray(nc.variables["TEC"][:], dtype=float)
        tsec = np.asarray(nc.variables["time"][:], dtype=float)
        attrs = {a: nc.getncattr(a) for a in (
            "prn_id", "leo_id", "conid", "year", "month", "day",
            "hour", "minute", "second", "lat_tecmax_tangent",
            "lon_tecmax_tangent") if a in nc.ncattrs()}

    tp, _p, alt_m = rayTangent(LEO, GNSS)
    alt_km = alt_m * 1e-3
    dist_leo = np.linalg.norm(tp - LEO, axis=0)

    # The occultation is the part where the tangent has detached from the LEO
    # (>5 km) and sits below the ~720 km top of interest, with finite TEC.
    detached = (dist_leo > 5.0) & (alt_km < 720.0) & np.isfinite(tec)
    idx = np.where(detached)[0]
    if idx.size < 5:
        raise ValueError(f"{fpath.name}: no usable occultation sweep found.")
    sl = slice(int(idx.min()), int(idx.max()) + 1)

    LEO, GNSS, tp = LEO[:, sl], GNSS[:, sl], tp[:, sl]
    alt_km, tec = alt_km[sl], tec[sl]
    tsec = tsec[sl] - tsec[sl][0]
    occ_type = "setting" if alt_km[0] > alt_km[-1] else "rising"

    return dict(LEO=LEO, GNSS=GNSS, tp=tp, alt=alt_km, tec=tec, tsec=tsec,
                occ_type=occ_type, attrs=attrs, name=fpath.name, n=LEO.shape[1])


def auto_pick_file(day_dir: Path, scan_max: int = 150) -> Path:
    """
    Scan up to `scan_max` files in a day directory and return the one whose
    occultation sweep spans the widest tangent-altitude range while reaching low
    (a deep, clean pass makes the most legible animation).
    """
    files = sorted(day_dir.glob(f"*{PODTC_SUFFIX}"))[:scan_max]
    best, best_score = None, -np.inf
    for f in files:
        try:
            occ = load_occultation(f)
        except Exception:
            continue
        span = float(np.nanmax(occ["alt"]) - np.nanmin(occ["alt"]))
        low = float(np.nanmin(occ["alt"]))
        # Favour a wide span that dips near/below the surface, penalise gaps.
        score = span - max(0.0, low)
        if score > best_score:
            best, best_score = f, score
    if best is None:
        raise FileNotFoundError(f"No readable podTc2 files under {day_dir}")
    return best


# ── occultation-plane projection ─────────────────────────────────────────────
def plane_basis(occ: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the fixed 2-D basis (e_h, e_r) of the occultation plane from the sample
    at the middle of the sweep. e_r points radially "up" through the tangent
    point; e_h is the in-plane horizontal, oriented so the GNSS lies at +u.
    """
    L, G, T = occ["LEO"], occ["GNSS"], occ["tp"]
    mid = L.shape[1] // 2
    
    # 1. Anchor the radial vector strictly to the true tangent point
    #    to prevent warping caused by Earth's oblateness.
    e_r = T[:, mid].copy()
    e_r /= np.linalg.norm(e_r)
    
    # 2. Derive the horizontal vector from the ray itself (LEO -> GNSS).
    #    Projecting this ray onto the horizontal plane ensures e_h 
    #    perfectly aligns with the occultation geometry.
    ray = G[:, mid] - L[:, mid]
    e_h = ray - e_r * (e_r @ ray)
    e_h /= np.linalg.norm(e_h)
    
    # No need for an orientation check! Because 'ray' points from LEO 
    # to GNSS, the resulting e_h inherently points towards the GNSS, 
    # guaranteeing the GNSS lies at +u.
    
    return e_h, e_r


def project(P: np.ndarray, e_h: np.ndarray, e_r: np.ndarray) -> tuple:
    """Project ECEF points into 2D while preserving true geometric altitude."""
    u = e_h @ P
    
    # Get the true 3D distance from the center of the Earth
    r_true = np.linalg.norm(P, axis=0)
    
    # Calculate 'v' so that u^2 + v^2 exactly equals r_true^2.
    # This "rotates" out-of-plane motion into the 2D plane instead of squashing it,
    # ensuring the LEO arc stays at its true altitude.
    v = np.sqrt(np.maximum(r_true**2 - u**2, 0.0))
    v = np.copysign(v, e_r @ P)
    
    return u, v


def ray_band_segment(leo: np.ndarray, gnss: np.ndarray, e_h, e_r,
                     n_samp: int = 4000) -> tuple:
    """
    Densely sample the LEO->GNSS chord and return the projected (u, v) of the
    portion whose geocentric altitude is within the 60-700 km ionosphere band.
    """
    t = np.linspace(0.0, 1.0, n_samp)
    pts = leo[:, None] + (gnss - leo)[:, None] * t[None, :]
    alt = np.linalg.norm(pts, axis=0) - EARTH_R
    m = (alt >= IONO_MIN_KM) & (alt <= IONO_MAX_KM)
    if not np.any(m):
        return np.array([]), np.array([])
    return project(pts[:, m], e_h, e_r)


# ── Kalman-filter voxel grid overlay ────────────────────────────────────────
# The ParametricEnKF state (Ionosphere_Tomography_Inverter/ionospheric_state.py)
# is a horizontal (lat, lon) grid of analytic profile columns, not a discretised
# 3-D voxel grid — see demo_ground_station_kf.py's GRID_DLAT/GRID_DLON (2 deg)
# and ALT_GRID_KM. For visualisation we bin that same lat/lon resolution against
# altitude to form genuine 3-D cells, then highlight whichever cells the ray
# actually passes through inside the 60-700 km band.
VOXEL_DLAT_DEG = 2.0              # matches GRID_DLAT in demo_ground_station_kf.py
VOXEL_DLON_DEG = 2.0              # matches GRID_DLON in demo_ground_station_kf.py
VOXEL_DALT_KM = 40.0
VOXEL_DECAY_SEC = 45.0            # wall-clock seconds a crossed voxel stays lit


def _ecef_to_geodetic(P: np.ndarray) -> tuple:
    """ECEF km (3, N) -> spherical geodetic lat/lon (deg) and altitude (km),
    consistent with the spherical EARTH_R already used throughout this file."""
    x, y, z = P[0], P[1], P[2]
    r = np.linalg.norm(P, axis=0)
    lat = np.degrees(np.arcsin(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, r - EARTH_R


def _geodetic_to_ecef(lat_deg, lon_deg, alt_km) -> np.ndarray:
    """Inverse of `_ecef_to_geodetic`; returns a (3, N) (or (3,)) ECEF km array."""
    latr, lonr = np.radians(lat_deg), np.radians(lon_deg)
    r = EARTH_R + alt_km
    return np.array([r * np.cos(latr) * np.cos(lonr),
                     r * np.cos(latr) * np.sin(lonr),
                     r * np.sin(latr)])


def build_voxel_grid(occ: dict, frames: np.ndarray, dlat: float = VOXEL_DLAT_DEG,
                     dlon: float = VOXEL_DLON_DEG, dalt: float = VOXEL_DALT_KM,
                     pad_deg: float = 1.0, n_samp: int = 400) -> dict:
    """
    Build a regular lat/lon/alt voxel grid, at the EnKF state grid's horizontal
    resolution, sized just to cover the ground footprint this pass's ray sweeps
    through the 60-700 km ionosphere band (so the overlay stays sparse and
    legible instead of tiling the whole globe).
    """
    L, G = occ["LEO"], occ["GNSS"]
    lat_min, lat_max = np.inf, -np.inf
    lon_min, lon_max = np.inf, -np.inf
    t = np.linspace(0.0, 1.0, n_samp)
    for i in frames:
        pts = L[:, i, None] + (G[:, i] - L[:, i])[:, None] * t[None, :]
        ralt = np.linalg.norm(pts, axis=0) - EARTH_R
        m = (ralt >= IONO_MIN_KM) & (ralt <= IONO_MAX_KM)
        if not np.any(m):
            continue
        lat, lon, _ = _ecef_to_geodetic(pts[:, m])
        lat_min, lat_max = min(lat_min, lat.min()), max(lat_max, lat.max())
        lon_min, lon_max = min(lon_min, lon.min()), max(lon_max, lon.max())

    if not np.isfinite(lat_min):          # sweep never entered the band
        lat_min, lat_max, lon_min, lon_max = -pad_deg, pad_deg, -pad_deg, pad_deg

    lat_edges = np.arange(np.floor(lat_min - pad_deg), np.ceil(lat_max + pad_deg) + dlat, dlat)
    lon_edges = np.arange(np.floor(lon_min - pad_deg), np.ceil(lon_max + pad_deg) + dlon, dlon)
    alt_edges = np.arange(IONO_MIN_KM, IONO_MAX_KM + dalt, dalt)
    nlat, nlon, nalt = lat_edges.size - 1, lon_edges.size - 1, alt_edges.size - 1

    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    alt_c = 0.5 * (alt_edges[:-1] + alt_edges[1:])
    LAc, LOc, ALc = np.meshgrid(lat_c, lon_c, alt_c, indexing="ij")
    centers_ecef = _geodetic_to_ecef(LAc.ravel(), LOc.ravel(), ALc.ravel())

    return dict(lat_edges=lat_edges, lon_edges=lon_edges, alt_edges=alt_edges,
                nlat=nlat, nlon=nlon, nalt=nalt, centers_ecef=centers_ecef,
                last_hit=np.full(nlat * nlon * nalt, -1e9, dtype=float))


def voxel_hits(pts_ecef: np.ndarray, grid: dict) -> np.ndarray:
    """Return the unique flat voxel indices that `pts_ecef` (3, K) fall inside."""
    if pts_ecef.size == 0:
        return np.array([], dtype=int)
    lat, lon, alt = _ecef_to_geodetic(pts_ecef)
    ilat = np.digitize(lat, grid["lat_edges"]) - 1
    ilon = np.digitize(lon, grid["lon_edges"]) - 1
    ialt = np.digitize(alt, grid["alt_edges"]) - 1
    valid = ((ilat >= 0) & (ilat < grid["nlat"]) &
             (ilon >= 0) & (ilon < grid["nlon"]) &
             (ialt >= 0) & (ialt < grid["nalt"]))
    if not np.any(valid):
        return np.array([], dtype=int)
    flat = (ilat[valid] * grid["nlon"] + ilon[valid]) * grid["nalt"] + ialt[valid]
    return np.unique(flat)


def _voxel_colors(grid: dict, t_now: float, decay_sec: float) -> tuple:
    """
    Age every voxel against the current wall-clock time (seconds since sweep
    start, i.e. `occ["tsec"]`) and return (active_mask, RGBA) for whichever
    ones were crossed within the last `decay_sec` seconds. Colour is mapped
    directly from that elapsed *time* (viridis_r, vmin=0/vmax=decay_sec) —
    not a normalized 0-1 recency fraction — so the colour scale reads in
    actual seconds-since-crossed; alpha is faded on top of that so cells
    fully disappear once older than `decay_sec`.
    """
    age = t_now - grid["last_hit"]
    active = (age >= 0) & (age <= decay_sec)
    if not np.any(active):
        return active, None
    frac = np.clip(age[active] / decay_sec, 0.0, 1.0)   # 0 = just crossed
    colors = plt.cm.viridis_r(frac)
    colors[:, 3] = 0.15 + 0.75 * (1.0 - frac)
    return active, colors


# ── animation ────────────────────────────────────────────────────────────────
def animate_occultation(occ: dict, max_frames: int = 180, fps: int = 15,
                        save_path: Path | None = None, show_voxels: bool = False,
                        voxel_decay_sec: float = VOXEL_DECAY_SEC,
                        voxel_dlat: float = VOXEL_DLAT_DEG,
                        voxel_dlon: float = VOXEL_DLON_DEG,
                        voxel_dalt: float = VOXEL_DALT_KM) -> Path:
    e_h, e_r = plane_basis(occ)
    L, G, T = occ["LEO"], occ["GNSS"], occ["tp"]
    alt, tec, tsec = occ["alt"], occ["tec"], occ["tsec"]
    n = occ["n"]

    # Pre-project full sweep tracks (static context).
    leo_u, leo_v = project(L, e_h, e_r)
    tan_u, tan_v = project(T, e_h, e_r)

    frames = np.unique(np.linspace(0, n - 1, min(max_frames, n)).astype(int))

    a = occ["attrs"]
    sv = f"{a.get('conid', '?')}{int(a.get('prn_id', 0)):02d}"
    leo_id = f"GN{int(a.get('leo_id', 0)):02d}"
    date = (f"{int(a.get('year', 0))}-{int(a.get('month', 0)):02d}-"
            f"{int(a.get('day', 0)):02d}")

    # ── figure scaffold ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 7.6))
    gs = GridSpec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1],
                  wspace=0.22, hspace=0.28,
                  left=0.04, right=0.975, top=0.9, bottom=0.09)
    ax_geo = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # ---- geometry panel (static elements) ----
    ax_geo.set_aspect("equal")
    lim = 7700.0
    ax_geo.set_xlim(-3400, 3400)
    ax_geo.set_ylim(EARTH_R - 3000, lim)
    ax_geo.axis("off")

    ax_geo.add_patch(Circle((0, 0), EARTH_R, facecolor=_EARTH_COLOR,
                            edgecolor="0.4", lw=0.8, zorder=1))
    # 60-700 km ionosphere shell.
    ax_geo.add_patch(Annulus((0, 0), EARTH_R + IONO_MAX_KM,
                             IONO_MAX_KM - IONO_MIN_KM, facecolor=_BAND_COLOR,
                             alpha=0.35, edgecolor="none", zorder=2))
    for r in (EARTH_R + IONO_MIN_KM, EARTH_R + IONO_MAX_KM):
        ax_geo.add_patch(Circle((0, 0), r, facecolor="none",
                                edgecolor="#b8860b", lw=0.8, ls="--", zorder=3))
    # Faint full LEO orbit arc over the sweep.
    ax_geo.plot(leo_u, leo_v, color=_LEO_COLOR, lw=1.0, alpha=0.25, zorder=3)
    ax_geo.text(0, EARTH_R + 350, "ionosphere\n60–700 km", ha="center",
                va="center", fontsize=9, color="#7a5c00", zorder=4)
    # GNSS is ~29,000 km away (off-frame); label its direction.
    mid = len(frames) // 2
    gdir_u, gdir_v = project(G[:, mid], e_h, e_r)
    tmid_u, tmid_v = tan_u[mid], tan_v[mid]
    
    # 1. Calculate angle from the tangent point (not the Earth's surface!)
    ang = np.arctan2(gdir_v - tmid_v, gdir_u - tmid_u)
    
    # 2. Anchor the arrow exactly where the horizontal ray crosses x=3150
    y_cross = tmid_v + (3150 - tmid_u) * np.tan(ang)
    
    ax_geo.annotate(f"→ to GNSS {sv}\n(~29,000 km)",
                    xy=(3150, y_cross),
                    xytext=(1500, lim - 350), fontsize=8.5, color="0.3",
                    ha="center", va="top",
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2, alpha=0.8))

    # ---- TEC-vs-time panel (static) ----
    ax_t.set_xlim(tsec[0], tsec[-1])
    ax_t.set_ylim(np.nanmin(tec) * 0.95, np.nanmax(tec) * 1.05)
    ax_t.set_xlabel("time since sweep start (s)")
    ax_t.set_ylabel("slant TEC (TECU)")
    ax_t.set_title("Measured TEC vs time", fontsize=10)
    ax_t.grid(alpha=0.3)
    ax_t.plot(tsec, tec, color="0.8", lw=1.0, zorder=1)  # ghost of full curve

    # ---- TEC-vs-altitude profile panel (static) ----
    ax_p.set_xlim(np.nanmin(tec) * 0.95, np.nanmax(tec) * 1.05)
    ax_p.set_ylim(min(0, np.nanmin(alt)) - 20, max(IONO_MAX_KM + 40, np.nanmax(alt) + 40))
    ax_p.axhspan(IONO_MIN_KM, IONO_MAX_KM, color=_BAND_COLOR, alpha=0.30,
                 zorder=0, label="ionosphere 60–700 km")
    ax_p.set_xlabel("slant TEC (TECU)")
    ax_p.set_ylabel("tangent altitude (km)")
    ax_p.set_title("RO profile: TEC vs tangent altitude", fontsize=10)
    ax_p.grid(alpha=0.3)
    ax_p.plot(tec, alt, color="0.8", lw=1.0, zorder=1)  # ghost of full profile
    ax_p.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"GNSS radio-occultation pass — {leo_id} × {sv}   ({occ['occ_type']}, "
        f"{date})\nfile: {occ['name']}", fontsize=12)

    vox = None
    if show_voxels:
        vox = build_voxel_grid(occ, frames, dlat=voxel_dlat, dlon=voxel_dlon,
                               dalt=voxel_dalt)
        vox["u"], vox["v"] = project(vox["centers_ecef"], e_h, e_r)
        sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(0, voxel_decay_sec))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_geo, fraction=0.035, pad=0.02)
        cbar.set_label("time since ray crossed voxel (s)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

    dynamic: list = []

    def update(fi: int):
        for art in dynamic:
            art.remove()
        dynamic.clear()
        i = int(frames[fi])
        lu, lv = leo_u[i], leo_v[i]
        tu, tv = tan_u[i], tan_v[i]
        gu, gv = project(G[:, i], e_h, e_r)

        # 1. Generate dense points for the FULL ray (LEO to GNSS)
        t_ray = np.linspace(0.0, 1.0, 500)
        ray_3d = L[:, i, None] + (G[:, i] - L[:, i])[:, None] * t_ray[None, :]
        
        # Project all points through the non-linear projection
        ray_u, ray_v = project(ray_3d, e_h, e_r)
        
        (ray,) = ax_geo.plot(ray_u, ray_v, color=_RAY_COLOR, lw=1.5,
                             alpha=0.8, zorder=5)
        dynamic.append(ray)

        # 2. Generate dense points for the HIGHLIGHTED band
        t_band = np.linspace(0.0, 1.0, 4000)
        pts = L[:, i, None] + (G[:, i] - L[:, i])[:, None] * t_band[None, :]
        ralt = np.linalg.norm(pts, axis=0) - EARTH_R
        
        m = (ralt >= IONO_MIN_KM) & (ralt <= IONO_MAX_KM)
        if np.any(m):
            band_u, band_v = project(pts[:, m], e_h, e_r)
            (band,) = ax_geo.plot(band_u, band_v, color="#c1121f",
                                  lw=4.0, solid_capstyle="round", zorder=6)
            dynamic.append(band)

        # Voxel overlay: light up whichever EnKF-grid cells the ray just swept
        # through, coloured by actual elapsed time (viridis_r) since crossed.
        if vox is not None:
            if np.any(m):
                idx = voxel_hits(pts[:, m], vox)
                if idx.size:
                    vox["last_hit"][idx] = tsec[i]
            active, colors = _voxel_colors(vox, tsec[i], voxel_decay_sec)
            if colors is not None:
                vox_art = ax_geo.scatter(vox["u"][active], vox["v"][active],
                                         c=colors, s=55, marker="s",
                                         edgecolors="none", zorder=4)
                dynamic.append(vox_art)

        # LEO marker.
        (leo_m,) = ax_geo.plot([lu], [lv], marker="o", ms=9,
                               color=_LEO_COLOR, mec="k", mew=0.7, zorder=8)
        dynamic.append(leo_m)

        # Tangent point — gold when inside the ionosphere band, else grey.
        in_band = IONO_MIN_KM <= alt[i] <= IONO_MAX_KM
        (tan_m,) = ax_geo.plot([tu], [tv], marker="v", ms=11,
                               color=(_BAND_COLOR if in_band else "0.5"),
                               mec="k", mew=0.8, zorder=9)
        dynamic.append(tan_m)

        readout = ax_geo.text(
            0.02, 0.02,
            f"tangent altitude: {alt[i]:6.1f} km\nslant TEC: {tec[i]:7.1f} TECU\n"
            f"t = {tsec[i]:5.1f} s",
            transform=ax_geo.transAxes, fontsize=10, va="bottom", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
            zorder=10)
        dynamic.append(readout)

        # TEC-vs-time trace + marker.
        (t_line,) = ax_t.plot(tsec[:i + 1], tec[:i + 1], color=_RAY_COLOR,
                              lw=1.6, zorder=3)
        (t_dot,) = ax_t.plot([tsec[i]], [tec[i]], "o", color=_LEO_COLOR,
                             ms=7, zorder=4)
        dynamic.extend([t_line, t_dot])

        # TEC-vs-altitude profile build-up + marker.
        (p_line,) = ax_p.plot(tec[:i + 1], alt[:i + 1], color=_RAY_COLOR,
                              lw=1.6, zorder=3)
        (p_dot,) = ax_p.plot([tec[i]], [alt[i]], "o",
                             color=(_BAND_COLOR if in_band else "0.5"),
                             mec="k", mew=0.6, ms=8, zorder=4)
        dynamic.extend([p_line, p_dot])
        return dynamic

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / max(fps, 1), blit=False)

    if save_path is None:
        save_path = _default_save_path(occ, suffix="_voxels" if show_voxels else "")
    return _write_animation(anim, fig, len(frames), fps, save_path)


# ── 3D geometry animation ────────────────────────────────────────────────────
def _sphere(radius: float, n: int = 40) -> tuple:
    """Return x, y, z meshgrid arrays for a sphere of the given radius."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _coastline_ecef(radius: float, scale: str = "110m") -> tuple:
    """
    Natural Earth coastlines (via cartopy, already a project dependency) draped
    onto a sphere of the given radius, as one NaN-separated (x, y, z) polyline so
    the 3-D globe reads as an actual map rather than a bare sphere.
    """
    feature = cfeature.NaturalEarthFeature("physical", "coastline", scale)
    xs, ys, zs = [], [], []
    for geom in feature.geometries():
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for part in parts:
            lon, lat = np.asarray(part.coords).T
            latr, lonr = np.radians(lat), np.radians(lon)
            xs.append(radius * np.cos(latr) * np.cos(lonr))
            ys.append(radius * np.cos(latr) * np.sin(lonr))
            zs.append(radius * np.sin(latr))
            xs.append([np.nan]); ys.append([np.nan]); zs.append([np.nan])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def _clip_ray_to_box(leo: np.ndarray, gnss: np.ndarray, lim: float) -> np.ndarray:
    """
    Truncate the LEO->GNSS ray at the cube [-lim, lim]^3 boundary (GNSS sits
    ~29,000 km out, far outside the view box) so the drawn segment stays in frame.
    """
    d = gnss - leo
    length = np.linalg.norm(d)
    d = d / length
    tmax = length
    for k in range(3):
        if d[k] > 1e-9:
            tmax = min(tmax, (lim - leo[k]) / d[k])
        elif d[k] < -1e-9:
            tmax = min(tmax, (-lim - leo[k]) / d[k])
    return leo + d * max(0.0, tmax)


def animate_occultation_3d(occ: dict, max_frames: int = 180, fps: int = 15,
                           save_path: Path | None = None,
                           coastline_scale: str = "110m",
                           show_voxels: bool = False,
                           voxel_decay_sec: float = VOXEL_DECAY_SEC,
                           voxel_dlat: float = VOXEL_DLAT_DEG,
                           voxel_dlon: float = VOXEL_DLON_DEG,
                           voxel_dalt: float = VOXEL_DALT_KM) -> Path:
    """
    Render the same occultation as a true 3-D ECEF scene: a translucent Earth
    textured with Natural Earth coastlines, a 700 km ionosphere shell, the LEO
    orbit arc, and the LEO->GNSS ray sweeping down through the limb with its
    60-700 km segment highlighted and the tangent point tracing its descent.

    The camera is held static, positioned directly above the file's own TEC-max
    tangent point (`lat_tecmax_tangent`/`lon_tecmax_tangent` attrs) looking
    straight down at it, so the whole sweep is viewed from one fixed, physically
    meaningful vantage rather than an arbitrary orbiting shot.
    """
    
    L, G, T = occ["LEO"], occ["GNSS"], occ["tp"]
    alt, tec, tsec = occ["alt"], occ["tec"], occ["tsec"]
    n = occ["n"]
    frames = np.unique(np.linspace(0, n - 1, min(max_frames, n)).astype(int))

    a = occ["attrs"]
    sv = f"{a.get('conid', '?')}{int(a.get('prn_id', 0)):02d}"
    leo_id = f"GN{int(a.get('leo_id', 0)):02d}"
    date = (f"{int(a.get('year', 0))}-{int(a.get('month', 0)):02d}-"
            f"{int(a.get('day', 0)):02d}")

    # Static camera: elev/azim of a point on a sphere are exactly its geocentric
    # lat/lon, so this looks straight down along the local vertical at the tecmax
    # tangent point.
    cam_lat = float(a["lat_tecmax_tangent"]); cam_lon = float(a["lon_tecmax_tangent"])
    ns = "N" if cam_lat >= 0 else "S"
    ew = "E" if cam_lon >= 0 else "W"
    offset = 65 if cam_lat >=0 else -45

    lim = EARTH_R+IONO_MAX_KM+20
    fig = plt.figure(figsize=(9.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.set_box_aspect((1, 1, 1))
    for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setlim(-lim, lim)
    ax.set_axis_off()
    ax.view_init(elev=cam_lat-offset, azim=cam_lon-15)

    # Static scene: a translucent Earth (so the ray/tangent read through it),
    # Natural Earth coastlines draped just above the surface, a 700 km
    # ionosphere-top wireframe shell, and the full LEO orbit arc.
    ex, ey, ez = _sphere(EARTH_R, 40)
    
    # Earth Surface: Base solid object
    ax.plot_surface(ex, ey, ez, color="#aac4e0", alpha=0.5, linewidth=0,
                    zorder=1, shade=False)
                    
    # ---------------------------------------------------------
    # Coastlines: Draped onto the surface with back-face culling
    # ---------------------------------------------------------
    cx, cy, cz = _coastline_ecef(EARTH_R + 8.0, scale=coastline_scale)
    
    # 1. Calculate the normalized vector pointing from the origin to the camera.
    #    (Using the exact elev/azim we passed to ax.view_init)
    elev_rad = np.radians(cam_lat - offset)
    azim_rad = np.radians(cam_lon)
    cam_dx = np.cos(elev_rad) * np.cos(azim_rad)
    cam_dy = np.cos(elev_rad) * np.sin(azim_rad)
    cam_dz = np.sin(elev_rad)
    
    # 2. Take the dot product of every coastline point against the camera vector.
    #    Positive means it faces the camera, negative means it's on the back.
    dot_prod = (cx * cam_dx) + (cy * cam_dy) + (cz * cam_dz)
    
    # 3. Mask out the back side by setting those coordinates to NaN.
    back_mask = dot_prod < 0
    cx[back_mask] = np.nan
    cy[back_mask] = np.nan
    cz[back_mask] = np.nan
    
    ax.plot(cx, cy, cz, color="#3a3a3a", lw=0.5, alpha=0.65, zorder=2)
    
    # Ionosphere Shell: Floating above the Earth
    sx, sy, sz = _sphere(EARTH_R + IONO_MAX_KM, 30)
    ax.plot_wireframe(sx, sy, sz, color="#b8860b", alpha=0.16, linewidth=0.5, zorder=4)
    
    # LEO Orbit Arc: Background track
    ax.plot(L[0], L[1], L[2], color=_LEO_COLOR, lw=1.4, alpha=0.5, zorder=0)
    
    # TEC-Max Marker: Static crosshair on the surface
    tm_latr, tm_lonr = np.radians(cam_lat), np.radians(cam_lon)
    ax.scatter([EARTH_R * np.cos(tm_latr) * np.cos(tm_lonr)],
               [EARTH_R * np.cos(tm_latr) * np.sin(tm_lonr)],
               [EARTH_R * np.sin(tm_latr)], color="k", marker="+", s=120, zorder=5)

    fig.suptitle(
        f"GNSS radio-occultation geometry (3-D ECEF) — {leo_id} × {sv}\n"
        f"{occ['occ_type']}, {date}   ·   camera fixed above TEC-max tangent "
        f"({abs(cam_lat):.1f}°{ns}, {abs(cam_lon):.1f}°{ew})", fontsize=11)

    vox = None
    if show_voxels:
        vox = build_voxel_grid(occ, frames, dlat=voxel_dlat, dlon=voxel_dlon,
                               dalt=voxel_dalt)
        sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(0, voxel_decay_sec))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.7)
        cbar.set_label("time since ray crossed voxel (s)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    dynamic: list = []

    def update(fi: int):
        for art in dynamic:
            art.remove()
        dynamic.clear()
        i = int(frames[fi])

        # Ray Line
        end = _clip_ray_to_box(L[:, i], G[:, i], lim)
        (ray,) = ax.plot([L[0, i], end[0]], [L[1, i], end[1]],
                         [L[2, i], end[2]], color=_RAY_COLOR, lw=2.0,
                         alpha=0.95, zorder=6)
        dynamic.append(ray)

        # 60-700 km highlighted ray segment
        t = np.linspace(0.0, 1.0, 4000)
        pts = L[:, i, None] + (G[:, i] - L[:, i])[:, None] * t[None, :]
        ralt = np.linalg.norm(pts, axis=0) - EARTH_R
        m = (ralt >= IONO_MIN_KM) & (ralt <= IONO_MAX_KM)
        if np.any(m):
            (band,) = ax.plot(pts[0, m], pts[1, m], pts[2, m], color="#c1121f",
                              lw=5.0, solid_capstyle="round", zorder=7)
            dynamic.append(band)

        # Voxel overlay: light up whichever EnKF-grid cells the ray just swept
        # through, coloured by actual elapsed time (viridis_r) since crossed.
        if vox is not None:
            if np.any(m):
                idx = voxel_hits(pts[:, m], vox)
                if idx.size:
                    vox["last_hit"][idx] = tsec[i]
            active, colors = _voxel_colors(vox, tsec[i], voxel_decay_sec)
            if colors is not None:
                cxa, cya, cza = vox["centers_ecef"][:, active]
                vox_art = ax.scatter(cxa, cya, cza, c=colors, s=40, marker="s",
                                     depthshade=False, zorder=5)
                dynamic.append(vox_art)

        # LEO Marker
        leo_m = ax.scatter([L[0, i]], [L[1, i]], [L[2, i]], color=_LEO_COLOR,
                           edgecolor="k", s=80, zorder=8)
        dynamic.append(leo_m)

        in_band = IONO_MIN_KM <= alt[i] <= IONO_MAX_KM
        
        # Tangent Marker
        tan_m = ax.scatter([T[0, i]], [T[1, i]], [T[2, i]],
                           color=(_BAND_COLOR if in_band else "0.5"),
                           edgecolor="k", s=90, marker="v", zorder=9)
                           
        # Tangent Trace
        (tan_tr,) = ax.plot(T[0, :i + 1], T[1, :i + 1], T[2, :i + 1],
                            color="#c1121f", lw=2.0, alpha=0.85, zorder=3)
        dynamic.extend([tan_m, tan_tr])

        txt = ax.text2D(
            0.02, 0.96,
            f"tangent altitude: {alt[i]:6.1f} km\nslant TEC: {tec[i]:7.1f} TECU\n"
            f"t = {tsec[i]:5.1f} s",
            transform=ax.transAxes, fontsize=10, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
        dynamic.append(txt)
        return dynamic

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / max(fps, 1), blit=False)
    if save_path is None:
        save_path = _default_save_path(occ, suffix="_3d_voxels" if show_voxels else "_3d")
    return _write_animation(anim, fig, len(frames), fps, save_path)

# ── shared save helper ───────────────────────────────────────────────────────
def _default_save_path(occ: dict, suffix: str) -> Path:
    stem = occ["name"].replace(PODTC_SUFFIX, "").replace(".", "_")
    return FIGURES_DIR / f"occultation_{stem}{suffix}.gif"


def _write_animation(anim, fig, nframes: int, fps: int, save_path: Path) -> Path:
    """Encode to MP4 if ffmpeg is available, else fall back to an animated GIF."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = None
    if FFMpegWriter.isAvailable():
        try:
            mp4 = save_path.with_suffix(".mp4")
            print(f"  Encoding MP4 → {mp4} ({nframes} frames @ {fps} fps) ...")
            anim.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=2600), dpi=110)
            wrote = mp4
        except Exception as exc:
            print(f"  FFMpeg failed ({exc}); falling back to GIF.")
    if wrote is None:
        gif = save_path.with_suffix(".gif")
        print(f"  Encoding GIF → {gif} ({nframes} frames @ {fps} fps) ...")
        anim.save(str(gif), writer=PillowWriter(fps=fps), dpi=95)
        wrote = gif
    plt.close(fig)
    print(f"  Saved → {wrote}")
    return wrote


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, default=None,
                   help="Path to a podTc2 NetCDF file. If omitted, auto-pick a "
                        "deep pass from --day.")
    p.add_argument("--day", default="2025.239",
                   help="Day directory under PODTC_BASE for auto-pick "
                        "(default: 2025.239).")
    p.add_argument("--mode", choices=("2d", "3d", "both"), default="2d",
                   help="2d = geometry+TEC panels (default); 3d = true 3-D ECEF "
                        "scene; both = render each.")
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--voxels", action="store_true",
                   help="Overlay the EnKF-grid voxels the ray sweeps through "
                        "inside the 60-700 km band, coloured by wall-clock "
                        "time since crossed (viridis_r, fades out after "
                        "--voxel-decay-sec seconds).")
    p.add_argument("--voxel-decay-sec", type=float, default=VOXEL_DECAY_SEC,
                   help=f"Seconds a crossed voxel stays lit before fading out "
                        f"(default: {VOXEL_DECAY_SEC}).")
    p.add_argument("--voxel-dlat", type=float, default=VOXEL_DLAT_DEG,
                   help=f"Voxel grid latitude resolution, deg (default: "
                        f"{VOXEL_DLAT_DEG}, matches the EnKF state grid).")
    p.add_argument("--voxel-dlon", type=float, default=VOXEL_DLON_DEG,
                   help=f"Voxel grid longitude resolution, deg (default: "
                        f"{VOXEL_DLON_DEG}, matches the EnKF state grid).")
    p.add_argument("--voxel-dalt", type=float, default=VOXEL_DALT_KM,
                   help=f"Voxel grid altitude resolution, km (default: "
                        f"{VOXEL_DALT_KM}).")
    args = p.parse_args()

    if args.file is not None:
        fpath = args.file
    else:
        day_dir = PODTC_BASE / args.day
        if not day_dir.is_dir():
            sys.exit(f"Day directory not found: {day_dir}")
        print(f"Auto-picking deepest occultation under {day_dir} ...")
        fpath = auto_pick_file(day_dir)
        print(f"  Selected: {fpath.name}")

    occ = load_occultation(fpath)
    print(f"Loaded {occ['name']}: {occ['n']} sweep samples, "
          f"{occ['occ_type']}, tangent alt "
          f"{occ['alt'].min():.1f}→{occ['alt'].max():.1f} km, "
          f"TEC {occ['tec'].min():.1f}–{occ['tec'].max():.1f} TECU.")
    voxel_kw = dict(show_voxels=args.voxels, voxel_decay_sec=args.voxel_decay_sec,
                    voxel_dlat=args.voxel_dlat, voxel_dlon=args.voxel_dlon,
                    voxel_dalt=args.voxel_dalt)
    if args.mode in ("2d", "both"):
        animate_occultation(occ, max_frames=args.max_frames, fps=args.fps,
                            save_path=args.out if args.mode == "2d" else None,
                            **voxel_kw)
    if args.mode in ("3d", "both"):
        animate_occultation_3d(occ, max_frames=args.max_frames, fps=args.fps,
                               save_path=args.out if args.mode == "3d" else None,
                               **voxel_kw)


if __name__ == "__main__":
    main()
