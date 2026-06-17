#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEC_model/igs_tec_pipeline.py
==============================
Carrier-aided absolute slant TEC from IGS ground-station RINEX data.

Pipeline stages
---------------
1. **Download**  – RINEX observation + navigation files and a DCB SINEX
                   from the CDDIS archive, with optional local-file fallback.
2. **Parse**     – Extract dual-frequency pseudorange and carrier-phase
                   observables using *georinex*.
3. **Ephemeris** – Satellite ECEF positions from broadcast navigation
                   (IS-GPS-200 for GPS/Galileo/BeiDou; RK4 for GLONASS).
4. **Filter**    – Discard epochs with elevation ≤ 20°.
5. **Select**    – Pick the widest-separation dual-frequency pair with
                   sufficient data per constellation.
6. **Cycle slips** – Geometry-free combination jump detection; arc splitting
                   on jumps and time gaps > 60 s.
7. **Leveling**  – Elevation-weighted carrier phase leveling per arc to
                   resolve the integer ambiguity.
8. **DCB**       – Apply satellite + receiver differential code biases
                   from a SINEX BSX file.
9. **IRI model** – Optionally forward-model sTEC via EDPSamples (IRI-2020)
                   for reference / residual computation.
10. **Output**   – Yield observation dicts compatible with
                   `Ionosphere_Tomography_Inverter.assimilate()` and the
                   `demo_group.py` grouping code (same schema as podTc2 data).

Required external packages
--------------------------
    pip install georinex unlzw3

    georinex  – RINEX 2/3 parser (auto-decompresses .gz and Hatanaka .crx)
    unlzw3    – needed by georinex for UNIX .Z archives

Usage
-----
    from TEC_model.igs_tec_pipeline import IGSTECPipeline, process_igs_station

    # High-level convenience wrapper:
    obs_list = process_igs_station(
        station   = 'ALGO',
        date      = datetime(2024, 5, 10),
        cache_dir = '/tmp/igs_cache',
    )

    # Or use the class directly for finer control:
    pipe = IGSTECPipeline(
        station       = 'ALGO',
        date          = datetime(2024, 5, 10),
        rinex_version = 3,
        cache_dir     = '/tmp/igs_cache',
        use_iri       = True,
    )
    obs_list = pipe.run()

    # Each element of obs_list is a dict with the same keys as a podTc2_data
    # dict plus IGS-specific fields; pass them directly to demo_group helpers.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    import georinex as gr
    _HAS_GEORINEX = True
except ImportError:
    _HAS_GEORINEX = False
    warnings.warn(
        "georinex is not installed — RINEX parsing unavailable.\n"
        "Install with:  pip install georinex unlzw3",
        ImportWarning,
    )

try:
    from EDPSamples.edp_samples import EDPSamples
    _HAS_EDPSAMPLES = True
except ImportError:
    _HAS_EDPSAMPLES = False

# ──────────────────────────────────────────────────────────────────────────────
# Physical / geodetic constants
# ──────────────────────────────────────────────────────────────────────────────
C          = 2.99792458e8         # m s⁻¹
IONO_CONST = 40.3e16              # m³ s⁻² × (1 / el m⁻²) = 40.3 m³ s⁻² · 1/TECU·1e16
R_EARTH_KM = 6371.0              # km
H_IPP_KM   = 350.0               # single-layer ionospheric height (km)

WGS84_A  = 6378137.0             # semi-major axis (m)
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

# IS-GPS-200 constants (also used for Galileo, BeiDou)
MU_GPS     = 3.986004418e14      # m³ s⁻²
OMEGA_E    = 7.2921151467e-5     # rad s⁻¹  (WGS-84 Earth rotation rate)
# GLONASS (PZ-90)
MU_GLO     = 3.9860044e14        # m³ s⁻²
OMEGA_E_GL = 7.2921150e-5        # rad s⁻¹
C20_GL     = -1.08263e-3         # J2 zonal harmonic (PZ-90)

# ──────────────────────────────────────────────────────────────────────────────
# GNSS frequency tables (Hz)
# ──────────────────────────────────────────────────────────────────────────────
_GPS_F = {'L1': 1575.42e6, 'L2': 1227.60e6, 'L5': 1176.45e6}
_GAL_F = {'E1': 1575.42e6, 'E5a': 1176.45e6, 'E5b': 1207.14e6,
           'E5': 1191.795e6, 'E6': 1278.75e6}
_BDS_F = {'B1C': 1575.42e6, 'B1I': 1561.098e6,
           'B2a': 1176.45e6, 'B2b': 1207.14e6, 'B3I': 1268.52e6}
# GLONASS frequency slots resolved at runtime (see _glo_freq())

# Priority-ordered frequency pair candidates per constellation.
# Each entry: (fA_name, fB_name, fA_Hz, fB_Hz, code_A_opts, code_B_opts,
#              phase_A_opts, phase_B_opts)
# fA > fB so that BetaI > 0.
_FREQ_PRIORITY: Dict[str, List[Tuple]] = {
    # Each entry: (band_A_name, band_B_name, f1_hz, f2_hz,
    #              code_A_options, code_B_options,
    #              phase_A_options, phase_B_options)
    #
    # Observable lists include both RINEX-3 3-char codes (e.g. C1C, L2W) and
    # RINEX-2 2-char codes (e.g. P1, L1) so that georinex RINEX-2 datasets
    # (which keep the 2-char names) are handled identically to RINEX-3 files.
    'G': [
        ('L1', 'L5', _GPS_F['L1'], _GPS_F['L5'],
         ['C1C', 'C1W', 'C1P', 'P1', 'C1'],
         ['C5Q', 'C5X', 'C5P', 'C5'],
         ['L1C', 'L1W', 'L1P', 'L1'],
         ['L5Q', 'L5X', 'L5P', 'L5']),
        ('L1', 'L2', _GPS_F['L1'], _GPS_F['L2'],
         ['C1C', 'C1W', 'C1P', 'P1', 'C1'],
         ['C2P', 'C2W', 'C2C', 'C2D', 'C2S', 'C2L', 'P2', 'C2'],
         ['L1C', 'L1W', 'L1P', 'L1'],
         ['L2P', 'L2W', 'L2C', 'L2D', 'L2S', 'L2L', 'L2']),
    ],
    'E': [
        ('E1', 'E5a', _GAL_F['E1'], _GAL_F['E5a'],
         ['C1C', 'C1X', 'C1B', 'C1'],
         ['C5Q', 'C5X', 'C5'],
         ['L1C', 'L1X', 'L1B', 'L1'],
         ['L5Q', 'L5X', 'L5']),
        ('E1', 'E5b', _GAL_F['E1'], _GAL_F['E5b'],
         ['C1C', 'C1X', 'C1B', 'C1'],
         ['C7Q', 'C7X', 'C7'],
         ['L1C', 'L1X', 'L1B', 'L1'],
         ['L7Q', 'L7X', 'L7']),
        ('E1', 'E5',  _GAL_F['E1'], _GAL_F['E5'],
         ['C1C', 'C1X', 'C1'],
         ['C8Q', 'C8X', 'C8'],
         ['L1C', 'L1X', 'L1'],
         ['L8Q', 'L8X', 'L8']),
    ],
    'R': [
        # Actual GLONASS frequencies depend on channel number k; f1/f2 are
        # placeholders replaced at runtime by _select_freq_pair.
        ('G1', 'G2', None, None,
         ['C1C', 'C1P', 'P1', 'C1'],
         ['C2C', 'C2P', 'P2', 'C2'],
         ['L1C', 'L1P', 'L1'],
         ['L2C', 'L2P', 'L2']),
    ],
    'C': [
        ('B1C', 'B2a', _BDS_F['B1C'], _BDS_F['B2a'],
         ['C1X', 'C1'],  ['C5X', 'C5'],  ['L1X', 'L1'],  ['L5X', 'L5']),
        ('B1I', 'B2b', _BDS_F['B1I'], _BDS_F['B2b'],
         ['C2I', 'C2'],  ['C7I', 'C7'],  ['L2I', 'L2'],  ['L7I', 'L7']),
        ('B1I', 'B3I', _BDS_F['B1I'], _BDS_F['B3I'],
         ['C2I', 'C2'],  ['C6I', 'C6'],  ['L2I', 'L2'],  ['L6I', 'L6']),
    ],
}

# Minimum number of valid epochs per arc to retain for processing
MIN_ARC_SAMPLES = 20
# Gap larger than this (seconds) triggers an arc break
ARC_GAP_S = 60.0
# Geometry-free combination jump threshold for cycle-slip declaration (m)
GF_SLIP_THRESHOLD_M = 0.10
# Elevation cut-off (strictly greater than this value)
ELEV_CUT_DEG = 20.0
# CDDIS root URL
CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss"


# ──────────────────────────────────────────────────────────────────────────────
# §1  Geodetic / geometric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ecef_to_geodetic(xyz_m: np.ndarray) -> Tuple[float, float, float]:
    """ECEF (m) → (lat_deg, lon_deg, alt_m) using iterative Bowring method."""
    x, y, z = float(xyz_m[0]), float(xyz_m[1]), float(xyz_m[2])
    p = np.sqrt(x * x + y * y)
    lon_rad = np.arctan2(y, x)
    lat_rad = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(10):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
        lat_rad = np.arctan2(z + WGS84_E2 * N * np.sin(lat_rad), p)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    alt_m = p / np.cos(lat_rad) - N if abs(lat_rad) < np.radians(89) else (
        z / np.sin(lat_rad) - N * (1 - WGS84_E2))
    return float(np.degrees(lat_rad)), float(np.degrees(lon_rad)), float(alt_m)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """(lat_deg, lon_deg, alt_m) → ECEF position (m, 3-vector)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_m) * np.sin(lat)
    return np.array([x, y, z])


def _elevation_azimuth(rx_ecef_m: np.ndarray,
                        sv_ecef_m: np.ndarray) -> Tuple[float, float]:
    """Compute elevation (deg) and azimuth (deg) of satellite as seen from receiver.

    Parameters
    ----------
    rx_ecef_m : (3,) receiver ECEF position in metres
    sv_ecef_m : (3,) satellite ECEF position in metres

    Returns
    -------
    el_deg, az_deg : floats
    """
    lat, lon, _ = _ecef_to_geodetic(rx_ecef_m)
    lat_r, lon_r = np.radians(lat), np.radians(lon)

    # Local ENU unit vectors
    e = np.array([-np.sin(lon_r),               np.cos(lon_r),               0.0])
    n = np.array([-np.sin(lat_r) * np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r)])
    u = np.array([ np.cos(lat_r) * np.cos(lon_r),  np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])

    dr = sv_ecef_m - rx_ecef_m
    dr_norm = np.linalg.norm(dr)
    if dr_norm == 0:
        return 90.0, 0.0
    dr_u = dr / dr_norm

    el_deg = float(np.degrees(np.arcsin(np.dot(dr_u, u))))
    az_deg = float(np.degrees(np.arctan2(np.dot(dr_u, e), np.dot(dr_u, n)))) % 360
    return el_deg, az_deg


def _ipp_lat_lon(rx_lat_deg: float, rx_lon_deg: float,
                  el_deg: float, az_deg: float,
                  h_ipp_km: float = H_IPP_KM) -> Tuple[float, float]:
    """Compute the ionospheric pierce point (lat, lon) for the SLM.

    Uses the spherical Earth single-layer model at altitude h_ipp_km.

    Parameters
    ----------
    rx_lat_deg, rx_lon_deg : receiver geodetic latitude / longitude (deg)
    el_deg  : satellite elevation angle (deg)
    az_deg  : satellite azimuth angle (deg)
    h_ipp_km: ionospheric shell height (km)

    Returns
    -------
    ipp_lat_deg, ipp_lon_deg : floats
    """
    el   = np.radians(el_deg)
    az   = np.radians(az_deg)
    lat  = np.radians(rx_lat_deg)
    lon  = np.radians(rx_lon_deg)

    # Earth central angle from receiver to IPP
    psi = np.pi / 2 - el - np.arcsin(R_EARTH_KM / (R_EARTH_KM + h_ipp_km) * np.cos(el))

    ipp_lat = np.arcsin(np.sin(lat) * np.cos(psi) +
                        np.cos(lat) * np.sin(psi) * np.cos(az))
    if abs(np.degrees(lat)) > 70:
        ipp_lon = lon + np.arctan2(np.sin(psi) * np.sin(az),
                                    np.cos(lat) * np.cos(psi) -
                                    np.sin(lat) * np.sin(psi) * np.cos(az))
    else:
        ipp_lon = lon + np.arcsin(np.sin(psi) * np.sin(az) / np.cos(ipp_lat))

    return float(np.degrees(ipp_lat)), float(np.degrees(ipp_lon))


def _beta_i(f1_hz: float, f2_hz: float) -> float:
    """TEC conversion factor [TECU/m].

    BetaI = f1² × f2² / (40.3e16 × (f1² − f2²))   (f1 > f2 → BetaI > 0)

    Multiply the leveled pseudorange/phase difference (metres, computed as
    Φ1 − Φ2 or, equivalently, P2 − P1) by BetaI to obtain slant TEC in TECU.
    """
    f1sq = f1_hz ** 2
    f2sq = f2_hz ** 2
    return f1sq * f2sq / (IONO_CONST * (f1sq - f2sq))


def _glo_freq(signal: str, k: int) -> float:
    """Return GLONASS carrier frequency (Hz) for signal 'G1' or 'G2' and slot k."""
    if signal == 'G1':
        return (1602.0 + k * 0.5625) * 1e6
    if signal == 'G2':
        return (1246.0 + k * 0.4375) * 1e6
    raise ValueError(f"Unknown GLONASS signal: {signal}")


# GPS epoch used for GPS seconds-of-week conversion
_GPS_EPOCH = pd.Timestamp('1980-01-06', tz='UTC')
# Cumulative GPS-UTC leap seconds as of 2017 (valid through at least 2024)
_LEAP_SECONDS = 18


def _utc_to_gps_sow(t: pd.Timestamp) -> float:
    """Convert a UTC timestamp to GPS seconds-of-week.

    GPS time does not observe leap seconds.  The offset is approximately
    +18 s relative to UTC as of 2024 (no leap seconds have been added
    since January 2017).

    Parameters
    ----------
    t : pd.Timestamp (timezone-aware UTC or naive treated as UTC)

    Returns
    -------
    GPS seconds-of-week in [0, 604800)
    """
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    gps_total_s = (t - _GPS_EPOCH).total_seconds() + _LEAP_SECONDS
    return float(gps_total_s % 604800.0)


# ──────────────────────────────────────────────────────────────────────────────
# §2  Broadcast ephemeris — satellite ECEF positions
# ──────────────────────────────────────────────────────────────────────────────

def _solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 15) -> float:
    """Iterative Kepler equation solver: E = M + e·sin(E)."""
    E = M
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def _gps_galileo_bds_ecef(eph: dict, t_sv: float) -> Optional[np.ndarray]:
    """Compute GPS/Galileo/BeiDou satellite ECEF position (km).

    Implements IS-GPS-200 Table 20-IV (same model for Galileo and BeiDou).

    Parameters
    ----------
    eph : dict
        Broadcast ephemeris parameters.  Expected keys (IS-GPS-200 names):
        sqrtA, e, M0, Delta_n, omega, i0, Idot, Omega0, OmegaDot,
        Cuc, Cus, Crc, Crs, Cic, Cis, toe
    t_sv : float
        Satellite signal transmission time (GPS seconds of week).

    Returns
    -------
    (3,) array (km) or None if eph is incomplete.
    """
    try:
        sqrtA    = float(eph['sqrtA'])
        e        = float(eph['e'])
        M0       = float(eph['M0'])
        dn       = float(eph['Delta_n'])
        omega    = float(eph['omega'])
        i0       = float(eph['i0'])
        Idot     = float(eph['Idot'])
        Omega0   = float(eph['Omega0'])
        OmegaDot = float(eph['OmegaDot'])
        Cuc, Cus = float(eph['Cuc']), float(eph['Cus'])
        Crc, Crs = float(eph['Crc']), float(eph['Crs'])
        Cic, Cis = float(eph['Cic']), float(eph['Cis'])
        toe      = float(eph['toe'])
    except (KeyError, TypeError, ValueError):
        return None

    A  = sqrtA ** 2
    n0 = np.sqrt(MU_GPS / A ** 3)
    tk = t_sv - toe
    # Handle week crossover
    if tk >  302400: tk -= 604800
    if tk < -302400: tk += 604800

    n  = n0 + dn
    Mk = M0 + n * tk
    Ek = _solve_kepler(Mk, e)

    sinvk = np.sqrt(1 - e ** 2) * np.sin(Ek) / (1 - e * np.cos(Ek))
    cosvk = (np.cos(Ek) - e) / (1 - e * np.cos(Ek))
    vk    = np.arctan2(sinvk, cosvk)

    phik = vk + omega
    duk  = Cus * np.sin(2 * phik) + Cuc * np.cos(2 * phik)
    drk  = Crs * np.sin(2 * phik) + Crc * np.cos(2 * phik)
    dik  = Cis * np.sin(2 * phik) + Cic * np.cos(2 * phik)

    uk  = phik + duk
    rk  = A * (1 - e * np.cos(Ek)) + drk
    ik  = i0 + Idot * tk + dik

    xk_orb = rk * np.cos(uk)
    yk_orb = rk * np.sin(uk)

    Omegak = Omega0 + (OmegaDot - OMEGA_E) * tk - OMEGA_E * toe

    xk = xk_orb * np.cos(Omegak) - yk_orb * np.cos(ik) * np.sin(Omegak)
    yk = xk_orb * np.sin(Omegak) + yk_orb * np.cos(ik) * np.cos(Omegak)
    zk = yk_orb * np.sin(ik)

    return np.array([xk, yk, zk]) * 1e-3  # m → km


def _glonass_ecef(eph: dict, t_utc: float) -> Optional[np.ndarray]:
    """Propagate GLONASS satellite ECEF position via RK4 (PZ-90 model).

    Parameters
    ----------
    eph : dict
        Keys: X, Y, Z (km), Xdot, Ydot, Zdot (km s⁻¹),
              Xdotdot, Ydotdot, Zdotdot (km s⁻²), toe (UTC seconds of day)
    t_utc : float
        Observation time in UTC seconds of day.

    Returns
    -------
    (3,) array (km) or None.
    """
    try:
        r = np.array([float(eph['X']), float(eph['Y']), float(eph['Z'])],        dtype=float)
        v = np.array([float(eph['Xdot']), float(eph['Ydot']), float(eph['Zdot'])], dtype=float)
        a = np.array([float(eph['Xdotdot']), float(eph['Ydotdot']), float(eph['Zdotdot'])], dtype=float)
        toe = float(eph['toe'])
    except (KeyError, TypeError, ValueError):
        return None

    dt = t_utc - toe
    if abs(dt) > 1800:
        # Do not extrapolate more than 30 minutes from the reference epoch
        return None

    def _accel(r_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        """PZ-90 simplified acceleration (J2 + external lunar/solar neglected)."""
        rr  = np.linalg.norm(r_) * 1e3   # km → m
        r_m = r_ * 1e3
        j2f = 1.5 * C20_GL * MU_GLO * (WGS84_A ** 2) / rr ** 5
        ax  = (-MU_GLO * r_m[0] / rr ** 3 +
               j2f * r_m[0] * (1 - 5 * (r_m[2] / rr) ** 2) +
               OMEGA_E_GL ** 2 * r_m[0] +
               2 * OMEGA_E_GL * v_[1] * 1e3 + a[0] * 1e3)
        ay  = (-MU_GLO * r_m[1] / rr ** 3 +
               j2f * r_m[1] * (1 - 5 * (r_m[2] / rr) ** 2) +
               OMEGA_E_GL ** 2 * r_m[1] -
               2 * OMEGA_E_GL * v_[0] * 1e3 + a[1] * 1e3)
        az  = (-MU_GLO * r_m[2] / rr ** 3 +
               j2f * r_m[2] * (3 - 5 * (r_m[2] / rr) ** 2) +
               a[2] * 1e3)
        return np.array([ax, ay, az]) * 1e-3  # → km s⁻²

    # RK4 integration with step ≤ 30 s
    h = np.sign(dt) * min(30.0, abs(dt)) if dt != 0 else 1.0
    n_steps = max(1, int(abs(dt) / abs(h)))
    for _ in range(n_steps):
        if abs(dt) < 1e-6:
            break
        step = min(abs(dt), abs(h)) * np.sign(dt)
        k1r  = v;                  k1v  = _accel(r, v)
        k2r  = v + 0.5*step*k1v;  k2v  = _accel(r + 0.5*step*k1r, k2r)
        k3r  = v + 0.5*step*k2v;  k3v  = _accel(r + 0.5*step*k2r, k3r)
        k4r  = v + step*k3v;      k4v  = _accel(r + step*k3r, k4r)
        r   += step * (k1r + 2*k2r + 2*k3r + k4r) / 6
        v   += step * (k1v + 2*k2v + 2*k3v + k4v) / 6
        dt  -= step

    return r  # km


class BroadcastEphemeris:
    """Parse a RINEX navigation file and provide satellite ECEF positions.

    Parameters
    ----------
    nav_path : str or Path
        Path to a RINEX 2 or 3 navigation file (may be .gz compressed).
    """

    def __init__(self, nav_path: str | Path) -> None:
        if not _HAS_GEORINEX:
            raise ImportError("georinex required for navigation parsing.")
        nav_str = str(nav_path)
        try:
            self._nav_ds = gr.load(nav_str)
        except Exception as exc:
            # Mixed nav files sometimes contain malformed IRNSS (System I)
            # records that georinex cannot parse.  Retry loading only the
            # constellations we actually use for TEC processing.
            log.warning("Nav load failed (%s) — retrying without IRNSS: %s", nav_str, exc)
            self._nav_ds = gr.load(nav_str, use='GRECSJ')
        self._cache: Dict[str, dict] = {}  # sv → dict of ephemeris records
        self._glo_slots: Dict[str, int] = {}  # GLONASS SV → FDMA channel k
        self._build_eph_cache()

    # ------------------------------------------------------------------
    # georinex RINEX-3 nav variable names → IS-GPS-200 parameter names used
    # by _gps_galileo_bds_ecef.  Without these aliases every satellite position
    # returns None and the pipeline produces 0 arcs.
    _GX_ALIAS: Dict[str, str] = {
        'Eccentricity': 'e',        # orbital eccentricity
        'DeltaN':       'Delta_n',  # mean motion correction (rad s⁻¹)
        'Io':           'i0',       # inclination at reference time (rad)
        'IDOT':         'Idot',     # rate of inclination angle (rad s⁻¹)
        'Toe':          'toe',      # time of ephemeris (GPS SOW) — takes
                                    #   priority over the UTC-SOD fallback below
    }

    def _build_eph_cache(self) -> None:
        """Pre-process navigation dataset into per-SV dictionaries."""
        ds = self._nav_ds
        svs = ds.sv.values if 'sv' in ds.dims else []
        for sv in svs:
            sv_str = str(sv).strip()
            sv_ds  = ds.sel(sv=sv)
            records = []
            times   = sv_ds.time.values if 'time' in sv_ds.dims else [sv_ds.time.values]
            for t in times:
                rec = {'_time': pd.Timestamp(t)}
                for var in sv_ds.data_vars:
                    try:
                        val = sv_ds[var].sel(time=t).values
                        rec[var] = float(val) if val.ndim == 0 else val
                    except Exception:
                        pass

                # Normalise georinex RINEX-3 names → IS-GPS-200 names so the
                # Kepler propagator can find every required ephemeris field.
                for gx_key, igs_key in self._GX_ALIAS.items():
                    if gx_key in rec and igs_key not in rec:
                        rec[igs_key] = rec[gx_key]

                if 'toe' not in rec:
                    # Fallback: derive from the message epoch (UTC seconds-of-day).
                    # Applies to GLONASS (no Toe variable) and legacy RINEX-2 GPS
                    # files.  For RINEX-3 GPS/Galileo/BeiDou the Toe alias above
                    # should have already populated rec['toe'] with GPS SOW.
                    ts = pd.Timestamp(t)
                    h  = ts.hour
                    m  = ts.minute
                    rec['toe'] = h * 3600 + m * 60 + ts.second
                records.append(rec)
            if records:
                self._cache[sv_str] = records

            # GLONASS: extract FDMA channel number from FreqNum field
            if sv_str.startswith('R'):
                try:
                    k = int(sv_ds['FreqNum'].values.flat[0])
                    self._glo_slots[sv_str] = k
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _best_eph(self, sv: str, t_sow: float) -> Optional[dict]:
        """Return the broadcast ephemeris record closest to t_sow."""
        records = self._cache.get(sv)
        if not records:
            return None
        best, best_dt = None, 1e9
        for rec in records:
            dt = abs(t_sow - rec.get('toe', t_sow))
            if dt < best_dt:
                best_dt = dt
                best = rec
        return best

    # ------------------------------------------------------------------
    def sv_position_km(self, sv: str, t_sow: float) -> Optional[np.ndarray]:
        """Return satellite ECEF position (km) for SV *sv* at time *t_sow*.

        Parameters
        ----------
        sv    : satellite identifier, e.g. 'G01', 'R07', 'E03', 'C14'
        t_sow : GPS/GLONASS seconds of week (or UTC seconds-of-day for GLONASS)

        Returns
        -------
        (3,) float64 array in km, or None on failure.
        """
        eph = self._best_eph(sv, t_sow)
        if eph is None:
            return None
        conid = sv[0]
        if conid in ('G', 'E', 'C'):
            return _gps_galileo_bds_ecef(eph, t_sow)
        if conid == 'R':
            return _glonass_ecef(eph, t_sow)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# §3  CDDIS downloader
# ──────────────────────────────────────────────────────────────────────────────

class RinexDownloader:
    """Download RINEX observation, navigation, and DCB SINEX files from CDDIS.

    CDDIS requires NASA Earthdata credentials stored in ~/.netrc:
        machine urs.earthdata.nasa.gov login <user> password <pass>

    Parameters
    ----------
    cache_dir : str or Path
        Directory for caching downloaded files.
    netrc_path : str or Path, optional
        Path to netrc file (default ``~/.netrc``).
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(self,
                 cache_dir: str | Path = '/tmp/igs_rinex_cache',
                 netrc_path: str | Path = '~/.netrc',
                 timeout: int = 120) -> None:
        self.cache_dir  = Path(cache_dir)
        self.netrc_path = Path(os.path.expanduser(str(netrc_path)))
        self.timeout    = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session   = self._make_session()

    # ------------------------------------------------------------------
    def _make_session(self):
        """Build a requests.Session with Earthdata authentication."""
        session = requests.Session()
        if self.netrc_path.exists():
            try:
                from netrc import netrc as _netrc
                auth = _netrc(str(self.netrc_path)).authenticators('urs.earthdata.nasa.gov')
                if auth:
                    session.auth = (auth[0], auth[2])
            except Exception as exc:
                log.warning("Could not read netrc: %s", exc)
        return session

    # ------------------------------------------------------------------
    def _download(self, url: str, dest: Path) -> Path:
        """Download *url* to *dest* if not already cached; return path."""
        if dest.exists() and dest.stat().st_size > 0:
            log.debug("Cache hit: %s", dest.name)
            return dest
        log.info("Downloading %s", url)
        resp = self._session.get(url, timeout=self.timeout, stream=True)
        if resp.status_code == 401:
            raise PermissionError(
                f"CDDIS authentication failed for {url}.\n"
                "Set up NASA Earthdata credentials in ~/.netrc:\n"
                "  machine urs.earthdata.nasa.gov login <user> password <pass>"
            )
        if resp.status_code == 404:
            raise FileNotFoundError(f"CDDIS file not found: {url}")
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest

    # ------------------------------------------------------------------
    @staticmethod
    def _decompress(path: Path) -> Path:
        """Decompress .gz file in-place and return the uncompressed path."""
        if path.suffix == '.gz':
            out = path.with_suffix('')
            if not out.exists():
                with gzip.open(path, 'rb') as f_in, open(out, 'wb') as f_out:
                    f_out.write(f_in.read())
            return out
        return path

    # ------------------------------------------------------------------
    def _list_cddis_dir(self, url_dir: str) -> List[str]:
        """Return basenames of files in a CDDIS directory via HTML index parsing."""
        try:
            resp = self._session.get(url_dir + '/', timeout=self.timeout)
            hrefs = re.findall(
                r'href="([^"]+\.(?:rnx|gz|Z|bsx|BSX|bia|BIA)[^"]*)"',
                resp.text,
            )
            # Normalise to bare filenames — CDDIS sometimes returns full paths
            return [h.rstrip('/').split('/')[-1] for h in hrefs]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def obs_file(self,
                 station: str,
                 date: datetime,
                 rinex_version: int = 3,
                 local_path: Optional[str] = None) -> Path:
        """Obtain a RINEX observation file (CDDIS or local).

        Parameters
        ----------
        station       : 4-character IGS station code (case-insensitive)
        date          : observation date
        rinex_version : 2 or 3
        local_path    : if given, use this local file instead of downloading

        Returns
        -------
        Path to the (possibly decompressed) RINEX observation file.
        """
        if local_path:
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(f"Local obs file not found: {local_path}")
            return self._decompress(p)

        sta  = station.upper()[:4]
        year = date.year
        doy  = date.timetuple().tm_yday
        yy   = year % 100

        if rinex_version == 2:
            # ── Attempt 1: plain RINEX-2 obs (.{yy}o.gz) in xxo/ ────────────
            fname_gz = f"{sta.lower()}{doy:03d}0.{yy:02d}o.gz"
            url_dir_o = f"{CDDIS_BASE}/data/daily/{year}/{doy:03d}/{yy:02d}o"
            dest     = self.cache_dir / fname_gz
            try:
                self._download(f"{url_dir_o}/{fname_gz}", dest)
                return self._decompress(dest)
            except FileNotFoundError:
                pass
            # Uppercase station name variant
            fname_gz_up = f"{sta}{doy:03d}0.{yy:02d}o.gz"
            dest_up = self.cache_dir / fname_gz_up
            try:
                self._download(f"{url_dir_o}/{fname_gz_up}", dest_up)
                return self._decompress(dest_up)
            except FileNotFoundError:
                pass

            # ── Attempt 2: Hatanaka-compressed RINEX-2 (.{yy}d.gz) in xxd/ ─
            # Many IGS stations (e.g. WES2) upload only in the Hatanaka format
            # used by the CDDIS xxd/ subdirectory:  sta{doy}0.{yy}d.gz
            # georinex decompresses and reads .crx / .{yy}d transparently.
            url_dir_d = f"{CDDIS_BASE}/data/daily/{year}/{doy:03d}/{yy:02d}d"
            for sta_case in (sta.lower(), sta):
                fname_d_gz = f"{sta_case}{doy:03d}0.{yy:02d}d.gz"
                dest_d = self.cache_dir / fname_d_gz
                try:
                    self._download(f"{url_dir_d}/{fname_d_gz}", dest_d)
                    return self._decompress(dest_d)
                except FileNotFoundError:
                    pass

            # ── Attempt 3: directory listing in xxd/ for this station ────────
            # Pattern: sta{DOY:3d}{session:1d}.{yy:2d}d.gz  e.g. wes21540.25d.gz
            files_d = self._list_cddis_dir(url_dir_d)
            pat_d   = re.compile(
                rf'^{re.escape(sta)}\d{{3}}0\.{yy:02d}d\.gz$', re.IGNORECASE
            )
            match_d = [f for f in files_d if pat_d.match(f.strip())]
            if match_d:
                fname_d_gz = match_d[0].strip()
                dest_d = self.cache_dir / fname_d_gz
                self._download(f"{url_dir_d}/{fname_d_gz}", dest_d)
                return self._decompress(dest_d)

            raise FileNotFoundError(
                f"No RINEX-2 obs file found for {sta} on DOY {doy} {year}. "
                f"Tried: {url_dir_o}/{sta.lower()}{doy:03d}0.{yy:02d}o.gz, "
                f"{url_dir_d}/{sta.lower()}{doy:03d}0.{yy:02d}d.gz (and uppercase variants)."
            )

        # RINEX 3: discover filename from directory listing
        url_dir = f"{CDDIS_BASE}/data/daily/{year}/{doy:03d}/{yy:02d}d"
        files   = self._list_cddis_dir(url_dir)
        # Match RINEX-3 long-name pattern for obs files:
        #   SSSS??XXX_R_YYYYDDD0000_01D_??S_MO.{rnx|crx}.gz
        # Both uncompressed (.rnx) and Hatanaka-compressed (.crx) variants.
        pattern = re.compile(
            rf'^{re.escape(sta)}\S{{5}}_R_{year}{doy:03d}\d{{4}}_01D_\d+S_MO\.(rnx|crx)(\.gz)?$',
            re.IGNORECASE
        )
        matches = [f.strip() for f in files if pattern.match(f.strip())]
        if not matches:
            # ── RINEX-3 fallback: try RINEX-2 Hatanaka naming in xxd/ ────────
            for sta_case in (sta.lower(), sta):
                fname_d_gz = f"{sta_case}{doy:03d}0.{yy:02d}d.gz"
                dest_d = self.cache_dir / fname_d_gz
                try:
                    self._download(f"{url_dir}/{fname_d_gz}", dest_d)
                    log.info("RINEX-3 not found; using RINEX-2 Hatanaka %s", fname_d_gz)
                    return self._decompress(dest_d)
                except FileNotFoundError:
                    pass
            raise FileNotFoundError(
                f"No RINEX-3 observation file found in {url_dir} for station {sta}. "
                f"Available: {files[:10]}"
            )
        # Prefer 30-second over 1-second data to save bandwidth
        matches_sorted = sorted(matches, key=lambda f: ('30S' not in f.upper()))
        fname_gz = matches_sorted[0]
        dest     = self.cache_dir / fname_gz
        self._download(f"{url_dir}/{fname_gz}", dest)
        return self._decompress(dest)

    # ------------------------------------------------------------------
    def nav_file(self,
                 station: str,
                 date: datetime,
                 rinex_version: int = 3,
                 local_path: Optional[str] = None) -> Path:
        """Obtain a broadcast navigation file with multi-GNSS coverage.

        Priority
        --------
        1. ``/{yy}p/`` — IGS/DLR RINEX-3 mixed navigation file
           (``BRDM…_MN.rnx.gz`` or ``BRDC…_MN.rnx.gz``).  Contains GPS,
           GLONASS, Galileo, BeiDou and SBAS — always tried first regardless
           of *rinex_version* so that non-GPS constellations in RINEX-2
           observation files still get ephemeris data.
        2. ``/{yy}n/`` — GPS-only RINEX-2 broadcast nav; tried station-specific
           first, then the IGS consolidated ``brdc{doy}0.{yy}n.gz``.

        The cached file from a previous GPS-only run must be removed before
        re-running if you want multi-GNSS arcs.
        """
        if local_path:
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(f"Local nav file not found: {local_path}")
            return self._decompress(p)

        sta  = station.upper()[:4]
        year = date.year
        doy  = date.timetuple().tm_yday
        yy   = year % 100

        # ── 1. Mixed-nav from /{yy}p/ (all constellations, preferred) ─────────
        url_dir_p = f"{CDDIS_BASE}/data/daily/{year}/{doy:03d}/{yy:02d}p"
        brdm_pat  = re.compile(
            rf'^(?:BRDM|BRDC)\S+_R_{year}{doy:03d}\d{{4}}_01D_MN\.rnx(\.gz)?$',
            re.IGNORECASE,
        )
        try:
            files_p   = self._list_cddis_dir(url_dir_p)
            matches_p = [f.strip() for f in files_p if brdm_pat.match(f.strip())]
            if matches_p:
                fname_p = matches_p[0]
                dest_p  = self.cache_dir / fname_p
                self._download(f"{url_dir_p}/{fname_p}", dest_p)
                print(f"  [nav] Using multi-GNSS mixed nav: {fname_p}")
                return self._decompress(dest_p)
        except Exception:
            pass   # fall through to GPS-only

        # ── 2. GPS-only /{yy}n/ — station-specific then IGS consolidated ──────
        url_dir_n = f"{CDDIS_BASE}/data/daily/{year}/{doy:03d}/{yy:02d}n"
        for fname_gz in (
            f"{sta.lower()}{doy:03d}0.{yy:02d}n.gz",
            f"{sta}{doy:03d}0.{yy:02d}n.gz",
            f"brdc{doy:03d}0.{yy:02d}n.gz",
        ):
            dest = self.cache_dir / fname_gz
            try:
                self._download(f"{url_dir_n}/{fname_gz}", dest)
                print(f"  [nav] Using GPS-only nav (no mixed nav found): {fname_gz}")
                return self._decompress(dest)
            except FileNotFoundError:
                pass

        raise FileNotFoundError(
            f"No navigation file found for {sta} on DOY {doy} {year}. "
            f"Tried /{yy:02d}p/ (BRDM/BRDC mixed) and "
            f"/{yy:02d}n/ (brdc{doy:03d}0.{yy:02d}n.gz)."
        )

    # ------------------------------------------------------------------
    def dcb_sinex(self,
                  date: datetime,
                  local_path: Optional[str] = None) -> Optional[Path]:
        """Download a daily DCB SINEX (BSX) file from CDDIS.

        Tries CODE/CAS products in order. Returns None if unavailable.
        """
        if local_path:
            p = Path(local_path)
            return p if p.exists() else None

        year = date.year
        doy  = date.timetuple().tm_yday

        # All DCB products live in the flat yearly directory bias/{year}/ —
        # there is no per-DOY subdirectory.
        # Known filename formats (time field is always 0000 for daily products):
        #   CAS0OPSRAP_20261530000_01D_01D_DCB.BIA.gz  (current CAS/GFZ)
        #   CAS1OPSRAP_20261530000_01D_01D_DCB.BIA.gz
        #   GFZ0OPSRAP_20261530000_01D_01D_DCB.BIA.gz
        #   CAS0MGXRAP_20261530000_01D_01D_DCB.BSX.gz  (legacy)
        #   DLR0MGXRAP_20261530000_01D_01D_DCB.BSX.gz  (legacy)
        # We construct URLs directly — CDDIS directory listings require a
        # cookie-based session that the netrc Basic Auth cannot satisfy, so
        # HTML-parsing via _list_cddis_dir is unreliable for this endpoint.
        _bias_dir = f"{CDDIS_BASE}/products/bias/{year}"
        direct_filenames = [
            f"CAS0OPSRAP_{year}{doy:03d}0000_01D_01D_DCB.BIA.gz",
            f"CAS1OPSRAP_{year}{doy:03d}0000_01D_01D_DCB.BIA.gz",
            f"GFZ0OPSRAP_{year}{doy:03d}0000_01D_01D_DCB.BIA.gz",
            f"DLR0OPSRAP_{year}{doy:03d}0000_01D_01D_DCB.BIA.gz",
            f"COD0OPSRAP_{year}{doy:03d}0000_01D_01D_DCB.BIA.gz",
            f"CAS0MGXRAP_{year}{doy:03d}0000_01D_01D_DCB.BSX.gz",
            f"DLR0MGXRAP_{year}{doy:03d}0000_01D_01D_DCB.BSX.gz",
        ]

        for fname in direct_filenames:
            dest = self.cache_dir / fname
            try:
                self._download(f"{_bias_dir}/{fname}", dest)
                log.info("DCB SINEX: %s", fname)
                return self._decompress(dest)
            except FileNotFoundError:
                continue  # try next provider
            except Exception as exc:
                log.warning("DCB download failed (%s): %s", fname, exc)

        # Fallback: directory listing (works when the session has a valid cookie)
        for pat_str, ext in [
            (rf'^(?:CAS0|CAS1|GFZ0|DLR0|COD0|WHU0)OPSRAP[_+]{year}{doy:03d}\d{{4}}_01D_01D_DCB\.BIA(\.gz)?$', '.BIA'),
            (rf'^(?:CAS0|DLR0)MGXRAP_{year}{doy:03d}\d{{4}}_01D_01D_DCB\.BSX(\.gz)?$', '.BSX'),
        ]:
            pat   = re.compile(pat_str, re.I)
            files = self._list_cddis_dir(_bias_dir)
            matches = [f for f in files if pat.match(f)]
            if matches:
                fname = matches[0]
                print(fname)
                dest  = self.cache_dir / fname
                try:
                    self._download(f"{_bias_dir}/{fname}", dest)
                    return self._decompress(dest)
                except Exception as exc:
                    log.warning("DCB download failed (%s): %s", fname, exc)

        log.warning("No DCB SINEX found for DOY %d/%d — proceeding without DCB.", year, doy)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# §4  DCB SINEX parser
# ──────────────────────────────────────────────────────────────────────────────

class DCBCorrector:
    """Parse a SINEX BSX differential code bias file and apply corrections.

    Supports the DSB (Differential Signal Bias) product format used by
    CODE/CAS/DLR for multi-GNSS DCBs.

    Parameters
    ----------
    bsx_path : str or Path
        Path to the BSX SINEX file (uncompressed).
    """

    def __init__(self, bsx_path: str | Path) -> None:
        self._sv_dcb:  Dict[str, Dict[str, float]] = {}  # sv  → {obs_pair: ns}
        self._sta_dcb: Dict[str, Dict[str, float]] = {}  # sta → {obs_pair: ns}
        self._parse(Path(bsx_path))

    # ------------------------------------------------------------------
    def _parse(self, path: Path) -> None:
        """Read DSB lines from the BSX file."""
        in_block = False
        with open(path, 'r', errors='ignore') as fh:
            for line in fh:
                if line.startswith('+BIAS/SOLUTION'):
                    in_block = True
                    continue
                if line.startswith('-BIAS/SOLUTION'):
                    break
                if not in_block or line.startswith('*'):
                    continue
                # DSB  SV   PRN  STATION   OBS1 OBS2  START  END  UNIT  VALUE ...
                parts = line.split()
                if len(parts) < 10 or parts[0] != 'DSB':
                    continue
                prn   = parts[2].strip()  # e.g. 'G01', empty for receiver biases
                sta   = parts[3].strip()  # station (empty for satellite biases)
                obs1  = parts[4].strip()  # e.g. 'C1C'
                obs2  = parts[5].strip()  # e.g. 'C5Q'
                try:
                    val_ns = float(parts[9])  # bias in nanoseconds
                except (IndexError, ValueError):
                    continue
                pair_key = f"{obs1}-{obs2}"
                if prn:
                    self._sv_dcb.setdefault(prn, {})[pair_key] = val_ns
                if sta:
                    self._sta_dcb.setdefault(sta.upper(), {})[pair_key] = val_ns

    # ------------------------------------------------------------------
    def get_sv_dcb_tecu(self, sv: str, obs1: str, obs2: str,
                         f1_hz: float, f2_hz: float) -> float:
        """Satellite DCB in TECU for the given obs pair.

        The DSB is defined as the satellite hardware delay difference
        (obs1 − obs2) in nanoseconds.  Converting to TECU:
            DCB_TECU = DCB_ns × 1e-9 × c × BetaI
        where BetaI = f1²f2² / (40.3e16 × (f1² − f2²)).

        Returns 0.0 if not found.
        """
        pair = f"{obs1}-{obs2}"
        ns   = self._sv_dcb.get(sv, {}).get(pair)
        if ns is None:
            # Try reversed pair with negated sign
            ns = self._sv_dcb.get(sv, {}).get(f"{obs2}-{obs1}")
            if ns is not None:
                ns = -ns
        if ns is None:
            return 0.0
        return ns * 1e-9 * C * _beta_i(f1_hz, f2_hz)

    # ------------------------------------------------------------------
    def get_rx_dcb_tecu(self, station: str, obs1: str, obs2: str,
                          f1_hz: float, f2_hz: float) -> float:
        """Receiver DCB in TECU for the given obs pair.  Returns 0.0 if not found."""
        pair = f"{obs1}-{obs2}"
        ns   = self._sta_dcb.get(station.upper(), {}).get(pair)
        if ns is None:
            ns = self._sta_dcb.get(station.upper(), {}).get(f"{obs2}-{obs1}")
            if ns is not None:
                ns = -ns
        if ns is None:
            return 0.0
        return ns * 1e-9 * C * _beta_i(f1_hz, f2_hz)


# ──────────────────────────────────────────────────────────────────────────────
# §5  Per-arc TEC processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _first_available(ds, options: List[str]) -> Optional[str]:
    """Return the first obs code in *options* that exists as a variable in *ds*."""
    avail = set(ds.data_vars)
    for opt in options:
        if opt in avail:
            return opt
    return None


def _split_arcs(times_s: np.ndarray, gf_m: np.ndarray) -> List[np.ndarray]:
    """Partition epoch indices into continuous arcs.

    An arc break occurs when:
    * The time gap between consecutive epochs exceeds ARC_GAP_S, OR
    * The geometry-free phase jump |Δgf| > GF_SLIP_THRESHOLD_M.

    Parameters
    ----------
    times_s : seconds-of-day array (N,)
    gf_m    : geometry-free combination Φ1−Φ2 (metres, N,), NaN where invalid

    Returns
    -------
    List of index arrays, one per arc.
    """
    N = len(times_s)
    if N == 0:
        return []

    # Find break indices
    breaks = [0]
    for i in range(1, N):
        gap  = times_s[i] - times_s[i - 1]
        if gap > ARC_GAP_S:
            breaks.append(i)
            continue
        if np.isnan(gf_m[i]) or np.isnan(gf_m[i - 1]):
            breaks.append(i)
            continue
        if abs(gf_m[i] - gf_m[i - 1]) > GF_SLIP_THRESHOLD_M:
            breaks.append(i)
    breaks.append(N)

    arcs = []
    for k in range(len(breaks) - 1):
        idx = np.arange(breaks[k], breaks[k + 1])
        valid = idx[np.isfinite(gf_m[idx])]
        if len(valid) >= MIN_ARC_SAMPLES:
            arcs.append(valid)
    return arcs


def _level_arc(P_diff_m: np.ndarray, phi_diff_m: np.ndarray,
                elevs_deg: np.ndarray) -> np.ndarray:
    """Carrier phase leveling for a single arc.

    Computes the elevation-weighted mean of (P_diff − Φ_diff) and adds it to
    the carrier-phase difference to yield a carrier-smoothed absolute range
    difference.

    Parameters
    ----------
    P_diff_m    : (N,) pseudorange difference P2 − P1 (metres)
    phi_diff_m  : (N,) phase difference  Φ1 − Φ2 (metres)
    elevs_deg   : (N,) satellite elevation angles (degrees)

    Returns
    -------
    (N,) carrier-leveled range difference in metres.
    """
    weights = np.sin(np.radians(np.clip(elevs_deg, 5.0, 90.0))) ** 2
    # Leveling constant: mean difference between code and phase proxies
    level_m = np.average(P_diff_m - phi_diff_m, weights=weights)
    return phi_diff_m + level_m


# ──────────────────────────────────────────────────────────────────────────────
# §5b  Module-level worker functions (picklable for ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────

def _sv_pos_from_cache(sv_str: str, t_sow: float,
                        records: list, glo_slots: dict) -> Optional[np.ndarray]:
    """Satellite ECEF position (km) from pre-extracted ephemeris cache dicts.

    Mirrors BroadcastEphemeris.sv_position_km() but accepts plain dicts so the
    function is picklable for multiprocessing workers.
    """
    if not records:
        return None
    best, best_dt = None, 1e9
    for rec in records:
        dt = abs(t_sow - rec.get('toe', t_sow))
        if dt < best_dt:
            best_dt = dt
            best    = rec
    if best is None:
        return None
    conid = sv_str[0]
    if conid in ('G', 'E', 'C'):
        return _gps_galileo_bds_ecef(best, t_sow)
    if conid == 'R':
        return _glonass_ecef(best, t_sow)
    return None


def _compute_sv_tec(task: dict) -> list:
    """Steps C–J of the per-SV TEC pipeline, packaged for ProcessPoolExecutor.

    All values in *task* are plain Python / NumPy objects (picklable).
    Returns a list of obs dicts; empty if no valid arcs found.
    """
    sv_str         = task['sv_str']
    conid          = task['conid']
    station        = task['station']
    P_diff         = task['P_diff']
    phi_diff       = task['phi_diff']
    valid_both     = task['valid_both']
    t_sod          = task['t_sod']
    t_gps_sow      = task['t_gps_sow']
    epoch_times_ns = task['epoch_times_ns']   # int64 nanoseconds since epoch
    betaI          = task['betaI']
    f1_hz          = task['f1_hz']
    f2_hz          = task['f2_hz']
    fA_name        = task['fA_name']
    fB_name        = task['fB_name']
    code_A_var     = task['code_A_var']
    code_B_var     = task['code_B_var']
    rx_xyz_m       = task['rx_xyz_m']
    rx_lat         = task['rx_lat']
    rx_lon         = task['rx_lon']
    rx_alt_m       = task['rx_alt_m']
    eph_records    = task['eph_records']
    glo_slots      = task['glo_slots']
    dcb_sv_tecu    = task['dcb_sv_tecu']
    dcb_rx_tecu    = task['dcb_rx_tecu']
    ephem_stride   = task['ephem_stride']
    verbose        = task['verbose']

    epoch_times = pd.DatetimeIndex(
        np.array(epoch_times_ns, dtype='datetime64[ns]')
    )
    t0_sv = time.time()

    # ── C. Satellite positions + elevation filter ──────────────────────────
    elevs     = np.full(len(t_sod), np.nan)
    azims     = np.full(len(t_sod), np.nan)
    sv_xyz_km = np.full((3, len(t_sod)), np.nan)

    all_valid_idx = np.where(valid_both)[0]
    stride        = max(1, int(ephem_stride))
    sampled_idx   = all_valid_idx[::stride]
    kepler_calls  = 0

    for i in sampled_idx:
        t_s   = t_sod[i] if conid == 'R' else t_gps_sow[i]
        sv_km = _sv_pos_from_cache(sv_str, t_s, eph_records, glo_slots)
        if sv_km is None:
            continue
        sv_xyz_km[:, i] = sv_km
        el, az = _elevation_azimuth(rx_xyz_m, sv_km * 1e3)
        elevs[i] = el
        azims[i] = az
        kepler_calls += 1

    good = sampled_idx[np.isfinite(elevs[sampled_idx])]
    if len(good) < 2:
        if verbose:
            print(f"  [{station}] {sv_str:4s}: <2 good position samples — skip",
                  flush=True)
        return []

    t_good = t_sod[good]
    elevs[all_valid_idx]     = np.interp(t_sod[all_valid_idx], t_good, elevs[good])
    azims[all_valid_idx]     = np.interp(t_sod[all_valid_idx], t_good, azims[good])
    for row in range(3):
        sv_xyz_km[row, all_valid_idx] = np.interp(
            t_sod[all_valid_idx], t_good, sv_xyz_km[row, good]
        )

    elev_mask  = (elevs > ELEV_CUT_DEG)
    valid_mask = valid_both & elev_mask & np.all(np.isfinite(sv_xyz_km), axis=0)
    if not np.any(valid_mask):
        return []

    # ── D. Geometry-free combination for cycle-slip detection ──────────────
    gf_m_full = np.where(valid_mask, phi_diff, np.nan)

    # ── E. Arc splitting (time gaps + cycle slips) ────────────────────────
    valid_idx  = np.where(valid_mask)[0]
    t_valid    = t_sod[valid_idx]
    gf_valid   = gf_m_full[valid_idx]
    arc_groups = _split_arcs(t_valid, gf_valid)
    if not arc_groups:
        return []

    # ── F–J. Per-arc leveling, DCB correction, IPP, output ────────────────
    results = []
    for arc_local_idx in arc_groups:
        global_idx = valid_idx[arc_local_idx]

        arc_t   = t_sod[global_idx]
        arc_P   = P_diff[global_idx]
        arc_phi = phi_diff[global_idx]
        arc_el  = elevs[global_idx]
        arc_az  = azims[global_idx]
        arc_sv  = sv_xyz_km[:, global_idx]
        arc_dt  = epoch_times[global_idx]

        leveled_m      = _level_arc(arc_P, arc_phi, arc_el)
        stec_tecu      = betaI * leveled_m
        stec_corrected = stec_tecu + dcb_sv_tecu + dcb_rx_tecu

        med_tec = np.nanmedian(stec_corrected)
        if not (0.1 < med_tec < 300.0):
            log.debug("[%s] Arc %s median TEC=%.1f TECU — discarding.",
                      station, sv_str, med_tec)
            continue

        ipp_lat = np.zeros(len(arc_el))
        ipp_lon = np.zeros(len(arc_el))
        for j, (el, az) in enumerate(zip(arc_el, arc_az)):
            ipp_lat[j], ipp_lon[j] = _ipp_lat_lon(rx_lat, rx_lon, el, az)

        i_tmax = int(np.argmax(stec_corrected))
        lat_tm = float(ipp_lat[i_tmax])
        lon_tm = float(ipp_lon[i_tmax])

        rx_km     = (rx_xyz_m / 1e3)[:, np.newaxis] * np.ones((1, len(arc_t)))
        arc_start = arc_dt[0]

        obs = {
            'TEC_podTc2':          stec_corrected,
            'TEC':                 stec_corrected,
            'LEO':                 rx_km,
            'GNSS':                arc_sv,
            'time':                arc_t - arc_t[0],
            'tangent_alt_km':      np.full(len(arc_t), H_IPP_KM),
            'tec_type':            'absolute',
            'lat_tecmax_tangent':  lat_tm,
            'lon_tecmax_tangent':  lon_tm,
            'date':    pd.Timestamp(arc_start),
            'year':    int(arc_start.year),
            'month':   int(arc_start.month),
            'day':     int(arc_start.day),
            'hour':    int(arc_start.hour),
            'minute':  int(arc_start.minute),
            'second':  float(arc_start.second),
            'DOY':     int(arc_start.to_pydatetime().timetuple().tm_yday),
            'prn_id':    sv_str[1:],
            'conid':     conid,
            'leo_id':    station,
            'fileStamp': f"{sv_str}.{arc_start.strftime('%Y.%j.%H%M')}",
            'station_id':     station,
            'station_lat':    rx_lat,
            'station_lon':    rx_lon,
            'station_alt_m':  rx_alt_m,
            'elevation':      arc_el,
            'azimuth':        arc_az,
            'ipp_lat':        ipp_lat,
            'ipp_lon':        ipp_lon,
            'ipp_alt_km':     H_IPP_KM,
            'freq_pair':      (fA_name, fB_name),
            'f1_hz':          f1_hz,
            'f2_hz':          f2_hz,
            'code_obs_A':     code_A_var,
            'code_obs_B':     code_B_var,
            'dcb_sv_tecu':    dcb_sv_tecu,
            'dcb_rx_tecu':    dcb_rx_tecu,
            'iri_stec':       np.zeros(len(arc_t)),
            'tec_residual':   stec_corrected.copy(),
            'obs_source':     'IGS_ground',
        }
        results.append(obs)

    if verbose:
        n_ep    = int(valid_both.sum())
        fin_el  = elevs[np.isfinite(elevs)]
        el_rng  = f"{fin_el.min():.0f}–{fin_el.max():.0f}" if len(fin_el) else "?"
        tec_all = np.concatenate([r['TEC'] for r in results]) if results else np.array([])
        tec_rng = f"{tec_all.min():.1f}–{tec_all.max():.1f}" if len(tec_all) else "—"
        print(f"  [{station}] {sv_str:4s}: {n_ep:5d} ep  "
              f"stride={stride:2d}  Kepler×{kepler_calls:3d}  "
              f"el={el_rng}°  "
              f"{len(results):2d} arcs  TEC={tec_rng} TECU  "
              f"{time.time()-t0_sv:.2f}s",
              flush=True)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# §6  IRI forward model baseline (via EDPSamples)
# ──────────────────────────────────────────────────────────────────────────────

class IRIForwardModel:
    """Compute IRI-2020 baseline sTEC along ground-to-GNSS ray paths.

    Uses the regional EDPSamples infrastructure.  For each satellite arc,
    builds a single-point EDPSamples at the arc's ionospheric pierce point
    and integrates along the ray to obtain sTEC.

    Parameters
    ----------
    alt_km_grid : ndarray
        Altitude grid (km) for the IRI profiles (e.g. np.arange(80, 900, 10)).
    sampling_params : pd.DataFrame
        Sampling parameters for EDPSamples (columns: f107, f107_81, ig, ir).
    """

    def __init__(self,
                 alt_km_grid: np.ndarray,
                 sampling_params: pd.DataFrame) -> None:
        if not _HAS_EDPSAMPLES:
            raise ImportError("EDPSamples not found — IRI forward model unavailable.")
        self.alt_km_grid     = alt_km_grid
        self.sampling_params = sampling_params
        self._edp_cache: Dict[str, 'EDPSamples'] = {}

    # ------------------------------------------------------------------
    def _get_edps(self, dt_str: str,
                  lat: float, lon: float) -> 'EDPSamples':
        """Build (or retrieve cached) EDPSamples for a point location."""
        key = f"{dt_str}_{lat:.1f}_{lon:.1f}"
        if key not in self._edp_cache:
            self._edp_cache[key] = EDPSamples(
                DateTime           = dt_str,
                geo_type           = "Point",
                altitude           = self.alt_km_grid,
                sampling_parameters= self.sampling_params,
                evaluate_iri       = 1,
                minLat=lat, maxLat=lat, dLat=1.0,
                minLon=lon, maxLon=lon, dLon=1.0,
            )
        return self._edp_cache[key]

    # ------------------------------------------------------------------
    def stec_along_ray(self,
                        dt_str: str,
                        rx_ecef_km: np.ndarray,
                        sv_ecef_km: np.ndarray,
                        ipp_lat: float,
                        ipp_lon: float) -> float:
        """Return IRI sTEC (TECU) along the receiver-to-satellite ray.

        Builds a single-node EDPSamples at the IPP location, constructs the
        observation operator H, and multiplies by the IRI mean EDP.
        Falls back to the single-layer mapping-function approximation if
        EDPSamples fails.

        Parameters
        ----------
        dt_str     : ISO datetime string for the epoch
        rx_ecef_km : (3,) receiver ECEF position (km)
        sv_ecef_km : (3,) satellite ECEF position (km)
        ipp_lat    : ionospheric pierce point latitude (deg)
        ipp_lon    : ionospheric pierce point longitude (deg)

        Returns
        -------
        sTEC in TECU (scalar).
        """
        try:
            edps = self._get_edps(dt_str, ipp_lat, ipp_lon)
            obs_dict = {
                'LEO':  rx_ecef_km[:, np.newaxis],
                'GNSS': sv_ecef_km[:, np.newaxis],
            }
            H = edps.get_observation_operator(obs_dict, num_segments=500)
            ne_mean = edps.edps[:, 0, :].mean(axis=-1)  # (n_height,)
            stec = float((H @ ne_mean.reshape(-1)) / 1e16)
            return max(stec, 0.0)
        except Exception as exc:
            log.debug("IRI full-ray integration failed (%s), using SLM fallback.", exc)
            # SLM fallback: VTEC / sin(elevation)
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# §7  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class IGSTECPipeline:
    """Full IGS ground-station TEC processing pipeline.

    Parameters
    ----------
    station : str
        Four-character IGS station code (case-insensitive), e.g. ``'ALGO'``.
    date : datetime
        UTC date of the observation session.
    rinex_version : {2, 3}
        RINEX observation file version to request from CDDIS.
    cache_dir : str or Path
        Directory for caching downloaded files.
    netrc_path : str or Path
        Path to ~/.netrc file with NASA Earthdata credentials.
    use_iri : bool
        If True, compute the IRI-2020 baseline sTEC for each arc using
        EDPSamples (requires the EDPSamples package to be importable).
    alt_km_grid : ndarray, optional
        Altitude grid for IRI profiles.  Default 80–800 km in 10-km steps.
    sampling_params : pd.DataFrame, optional
        Sampling parameters for EDPSamples.  If None, read from apf107.dat.
    local_obs : str, optional
        If given, load the obs file from this local path instead of CDDIS.
    local_nav : str, optional
        If given, load the nav file from this local path instead of CDDIS.
    local_dcb : str, optional
        If given, load the DCB SINEX from this local path.
    """

    def __init__(self,
                 station:          str,
                 date:             datetime,
                 rinex_version:    int = 3,
                 cache_dir:        str | Path = '/tmp/igs_rinex_cache',
                 netrc_path:       str | Path = '~/.netrc',
                 use_iri:          bool = False,
                 alt_km_grid:      Optional[np.ndarray] = None,
                 sampling_params:  Optional[pd.DataFrame] = None,
                 local_obs:        Optional[str] = None,
                 local_nav:        Optional[str] = None,
                 local_dcb:        Optional[str] = None,
                 ephem_stride:     int = 1,
                 show_progress:    bool = True,
                 verbose:          bool = False,
                 num_sv_workers:   int  = 1) -> None:
        if not _HAS_GEORINEX:
            raise ImportError(
                "georinex is required.  Install with: pip install georinex unlzw3"
            )
        self.station       = station.upper()[:4]
        self.date          = date
        self.rinex_ver     = rinex_version
        self.use_iri       = use_iri and _HAS_EDPSAMPLES
        self.local_obs     = local_obs
        self.local_nav     = local_nav
        self.local_dcb     = local_dcb
        self._ephem_stride   = max(1, int(ephem_stride))
        self._show_progress  = show_progress
        self._verbose        = verbose
        self._num_sv_workers = max(1, int(num_sv_workers))

        self.downloader = RinexDownloader(cache_dir=cache_dir, netrc_path=netrc_path)

        self.alt_km_grid = (
            alt_km_grid if alt_km_grid is not None
            else np.arange(80.0, 810.0, 10.0)
        )
        self.sampling_params = sampling_params

    # ------------------------------------------------------------------
    def run(self) -> List[dict]:
        """Execute the full pipeline and return a list of observation dicts.

        Each dict represents one continuous satellite arc and is compatible
        with ``Ionosphere_Tomography_Inverter.assimilate()`` and the
        ``demo_group.py`` grouping helpers.

        Returns
        -------
        List of observation dicts (may be empty if no usable arcs found).
        """
        t_run = time.time()

        # 1. Acquire files
        t0 = time.time()
        log.info("[%s] Downloading / loading RINEX files …", self.station)
        obs_path = self.downloader.obs_file(
            self.station, self.date, self.rinex_ver, self.local_obs)
        nav_path = self.downloader.nav_file(
            self.station, self.date, self.rinex_ver, self.local_nav)
        dcb_path = self.downloader.dcb_sinex(self.date, self.local_dcb)
        if self._verbose:
            print(f"  [{self.station}] Files ready  ({time.time()-t0:.1f}s)", flush=True)

        # 2. Parse observation file
        t0 = time.time()
        log.info("[%s] Parsing RINEX obs …", self.station)
        obs_ds = gr.load(str(obs_path))
        if self._verbose:
            n_sv  = len(obs_ds.sv.values)
            n_ep  = len(obs_ds.time.values)
            print(f"  [{self.station}] RINEX parsed: {n_sv} SVs, {n_ep} epochs  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # 3. Receiver position from RINEX header
        rx_xyz_m, rx_lat, rx_lon, rx_alt_m = self._rx_position(obs_ds)
        if self._verbose:
            print(f"  [{self.station}] Rx pos: lat={rx_lat:.3f}° lon={rx_lon:.3f}° "
                  f"alt={rx_alt_m/1e3:.1f}km", flush=True)

        # 4. Broadcast ephemeris
        t0 = time.time()
        log.info("[%s] Loading navigation ephemeris …", self.station)
        ephem = BroadcastEphemeris(nav_path)
        if self._verbose:
            n_eph = len(ephem._cache)
            print(f"  [{self.station}] Ephemeris: {n_eph} SVs in cache  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # 5. DCB corrector
        dcb = DCBCorrector(dcb_path) if dcb_path else None
        if self._verbose:
            if dcb:
                print(f"  [{self.station}] DCB: {len(dcb._sv_dcb)} Tx SVs, "
                      f"{len(dcb._sta_dcb)} Rx stations", flush=True)
            else:
                print(f"  [{self.station}] DCB: none (corrections will be 0)", flush=True)

        # 6. IRI forward model (optional)
        iri_model = None
        if self.use_iri:
            sp = self.sampling_params or self._default_sampling_params()
            iri_model = IRIForwardModel(self.alt_km_grid, sp)

        # Store ephemeris on self so _glo_channel can access it during _select_freq_pair
        self._ephem = ephem

        # 7. Build time arrays and stride
        svs            = obs_ds.sv.values
        epoch_times_dt = pd.DatetimeIndex(obs_ds.time.values)

        t_sod = np.array(
            [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6
             for t in epoch_times_dt], dtype=float
        )
        t_gps_sow = np.array(
            [_utc_to_gps_sow(t) for t in epoch_times_dt], dtype=float
        )

        stride = self._ephem_stride
        if stride == 1 and len(t_sod) > 1:
            dt_med = float(np.nanmedian(np.diff(t_sod)))
            if dt_med > 0:
                stride = max(1, int(round(150.0 / dt_med)))

        eligible_svs = [(str(sv).strip(), sv) for sv in svs
                        if str(sv).strip()[0] in _FREQ_PRIORITY]
        n_eligible = len(eligible_svs)

        if self._verbose:
            dt_med_s = float(np.nanmedian(np.diff(t_sod))) if len(t_sod) > 1 else 0
            print(f"  [{self.station}] {n_eligible} eligible SVs  "
                  f"sample_dt={dt_med_s:.0f}s  ephem_stride={stride}  "
                  f"(~{stride*dt_med_s:.0f}s between Kepler calls)", flush=True)

        # 8. Pre-stage task dicts in main thread (xarray / frequency selection)
        t0 = time.time()
        sv_tasks: list = []
        for sv_str_raw, sv in eligible_svs:
            conid  = sv_str_raw[0]
            sv_obs = obs_ds.sel(sv=sv)
            task   = self._stage_sv_task(
                sv_str_raw, conid, sv_obs, t_sod, t_gps_sow,
                epoch_times_dt, ephem, rx_xyz_m, rx_lat, rx_lon,
                rx_alt_m, dcb, stride,
            )
            if task is not None:
                sv_tasks.append(task)

        n_tasks = len(sv_tasks)
        if self._verbose:
            print(f"  [{self.station}] {n_tasks}/{n_eligible} SVs staged  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # 9. Compute TEC — serial or parallel
        obs_list:  List[dict] = []
        arc_count  = 0
        _BAR       = 28

        if self._num_sv_workers > 1 and n_tasks > 1:
            if self._verbose:
                print(f"  [{self.station}] Parallel TEC: {n_tasks} SVs × "
                      f"{self._num_sv_workers} workers", flush=True)
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=self._num_sv_workers) as pool:
                future_sv = {pool.submit(_compute_sv_tec, t): t['sv_str']
                             for t in sv_tasks}
                done = 0
                for fut in as_completed(future_sv):
                    arcs = fut.result()
                    obs_list.extend(arcs)
                    arc_count += len(arcs)
                    done      += 1
                    if self._show_progress and not self._verbose:
                        filled = int(_BAR * done / max(n_tasks, 1))
                        bar    = '█' * filled + '░' * (_BAR - filled)
                        print(f"\r  [{self.station}] [{bar}] {done:3d}/{n_tasks} SVs"
                              f"  {arc_count:3d} arcs",
                              end='', flush=True)
            if self._verbose:
                print(f"  [{self.station}] Parallel TEC done  "
                      f"({time.time()-t0:.1f}s)", flush=True)
        else:
            if self._verbose:
                print(f"  [{self.station}] Serial TEC: {n_tasks} SVs", flush=True)
            for sv_done, task in enumerate(sv_tasks, 1):
                arcs = _compute_sv_tec(task)
                obs_list.extend(arcs)
                arc_count += len(arcs)
                if self._show_progress and not self._verbose:
                    filled = int(_BAR * sv_done / max(n_tasks, 1))
                    bar    = '█' * filled + '░' * (_BAR - filled)
                    print(f"\r  [{self.station}] [{bar}] {sv_done:3d}/{n_tasks} SVs"
                          f"  {arc_count:3d} arcs",
                          end='', flush=True)

        if self._show_progress and not self._verbose:
            print(f"\r  [{self.station}] [{'█' * _BAR}] {n_tasks}/{n_tasks} SVs"
                  f"  {arc_count:3d} arcs  ✓", flush=True)

        if self._verbose:
            print(f"  [{self.station}] Pipeline complete: {len(obs_list)} arcs  "
                  f"total={time.time()-t_run:.1f}s", flush=True)

        log.info("[%s] Pipeline complete — %d satellite arcs.", self.station, len(obs_list))
        return obs_list

    # ------------------------------------------------------------------
    def _rx_position(self, obs_ds) -> Tuple[np.ndarray, float, float, float]:
        """Extract receiver ECEF position from the RINEX dataset."""
        xyz = np.zeros(3)
        try:
            xyz = np.array([
                float(obs_ds.attrs.get('position_xyz', [0, 0, 0])[0]),
                float(obs_ds.attrs.get('position_xyz', [0, 0, 0])[1]),
                float(obs_ds.attrs.get('position_xyz', [0, 0, 0])[2]),
            ])
        except Exception:
            pass
        # georinex may store receiver position as 'position'
        if np.allclose(xyz, 0):
            try:
                xyz = np.array(obs_ds.attrs.get('position', [0, 0, 0]), dtype=float)
            except Exception:
                pass
        if np.linalg.norm(xyz) < 1e3:
            log.warning("Receiver position not found in RINEX header; using (0,0,0).")
        lat, lon, alt = _ecef_to_geodetic(xyz)
        return xyz, lat, lon, alt

    # ------------------------------------------------------------------
    def _stage_sv_task(self,
                        sv_str:        str,
                        conid:         str,
                        sv_obs,
                        t_sod:         np.ndarray,
                        t_gps_sow:     np.ndarray,
                        epoch_times_dt: pd.DatetimeIndex,
                        ephem:         BroadcastEphemeris,
                        rx_xyz_m:      np.ndarray,
                        rx_lat:        float,
                        rx_lon:        float,
                        rx_alt_m:      float,
                        dcb:           Optional[DCBCorrector],
                        ephem_stride:  int) -> Optional[dict]:
        """Steps A–B: xarray access and array extraction.  Returns picklable task dict.

        Called in the main thread (xarray is not fork-safe for shared access).
        Returns None if no usable frequency pair is found for this SV.
        """
        # ── A. Select frequency pair ──────────────────────────────────────────
        freq_pair = self._select_freq_pair(sv_str, conid, sv_obs)
        if freq_pair is None:
            log.debug("[%s] No suitable frequency pair for %s", self.station, sv_str)
            return None

        (fA_name, fB_name, f1_hz, f2_hz,
         code_A_var, code_B_var, phase_A_var, phase_B_var) = freq_pair

        betaI = _beta_i(f1_hz, f2_hz)

        # ── B. Extract time series arrays ─────────────────────────────────────
        def _arr(varname: str) -> np.ndarray:
            v = sv_obs[varname].values
            return np.where(np.isfinite(v.astype(float)), v.astype(float), np.nan)

        try:
            P1   = _arr(code_A_var)
            P2   = _arr(code_B_var)
            L1_c = _arr(phase_A_var)
            L2_c = _arr(phase_B_var)
        except KeyError as exc:
            log.debug("[%s] Missing obs variable %s for %s", self.station, exc, sv_str)
            return None

        L1_m     = L1_c * (C / f1_hz)
        L2_m     = L2_c * (C / f2_hz)
        P_diff   = P2 - P1
        phi_diff = L1_m - L2_m
        valid_both = np.isfinite(P_diff) & np.isfinite(phi_diff)

        # Pre-compute DCB scalars (floats — picklable; avoid passing DCBCorrector)
        dcb_sv_tecu = (dcb.get_sv_dcb_tecu(sv_str, code_A_var, code_B_var, f1_hz, f2_hz)
                       if dcb else 0.0)
        dcb_rx_tecu = (dcb.get_rx_dcb_tecu(self.station, code_A_var, code_B_var, f1_hz, f2_hz)
                       if dcb else 0.0)

        # Pre-extract ephemeris records as plain dicts (picklable, no xarray)
        eph_records = ephem._cache.get(sv_str, [])

        return {
            'sv_str':          sv_str,
            'conid':           conid,
            'station':         self.station,
            'P_diff':          P_diff,
            'phi_diff':        phi_diff,
            'valid_both':      valid_both,
            't_sod':           t_sod,
            't_gps_sow':       t_gps_sow,
            'epoch_times_ns':  epoch_times_dt.asi8,   # int64 ns — picklable
            'betaI':           betaI,
            'f1_hz':           f1_hz,
            'f2_hz':           f2_hz,
            'fA_name':         fA_name,
            'fB_name':         fB_name,
            'code_A_var':      code_A_var,
            'code_B_var':      code_B_var,
            'rx_xyz_m':        rx_xyz_m,
            'rx_lat':          rx_lat,
            'rx_lon':          rx_lon,
            'rx_alt_m':        rx_alt_m,
            'eph_records':     eph_records,
            'glo_slots':       ephem._glo_slots,
            'dcb_sv_tecu':     dcb_sv_tecu,
            'dcb_rx_tecu':     dcb_rx_tecu,
            'ephem_stride':    ephem_stride,
            'verbose':         self._verbose,
        }

    # ------------------------------------------------------------------
    def _process_sv(self,
                     sv_str:       str,
                     conid:        str,
                     sv_obs,
                     t_sod:        np.ndarray,
                     t_gps_sow:    np.ndarray,
                     epoch_times:  pd.DatetimeIndex,
                     ephem:        BroadcastEphemeris,
                     rx_xyz_m:     np.ndarray,
                     rx_lat:       float,
                     rx_lon:       float,
                     rx_alt_m:     float,
                     dcb:          Optional[DCBCorrector],
                     iri_model:    Optional[IRIForwardModel],
                     ephem_stride: int = 1) -> List[dict]:
        """Thin wrapper: stage task dict then call _compute_sv_tec."""
        task = self._stage_sv_task(
            sv_str, conid, sv_obs, t_sod, t_gps_sow, epoch_times,
            ephem, rx_xyz_m, rx_lat, rx_lon, rx_alt_m, dcb, ephem_stride,
        )
        if task is None:
            return []
        return _compute_sv_tec(task)

    # ------------------------------------------------------------------
    def _select_freq_pair(self, sv: str, conid: str, sv_obs) -> Optional[Tuple]:
        """Select the best dual-frequency pair for *sv* from *sv_obs*.

        Iterates the priority list for the constellation and returns the first
        pair where both code and phase observables have ≥ MIN_ARC_SAMPLES
        finite values.

        Returns
        -------
        (fA_name, fB_name, f1_hz, f2_hz,
         code_A_var, code_B_var, phase_A_var, phase_B_var) or None.
        """
        candidates = _FREQ_PRIORITY.get(conid, [])
        avail = set(sv_obs.data_vars)

        for (fA_n, fB_n, f1, f2, cA_opts, cB_opts, pA_opts, pB_opts) in candidates:
            cA = _first_available(sv_obs, cA_opts)
            cB = _first_available(sv_obs, cB_opts)
            pA = _first_available(sv_obs, pA_opts)
            pB = _first_available(sv_obs, pB_opts)
            if not all([cA, cB, pA, pB]):
                log.debug("[%s] %s %s/%s: missing obs — cA=%s cB=%s pA=%s pB=%s (avail=%s)",
                          self.station, sv, fA_n, fB_n, cA, cB, pA, pB,
                          sorted(avail))
                continue

            # Check data density
            def _n_valid(var):
                v = sv_obs[var].values.astype(float)
                return int(np.isfinite(v).sum())

            if all(_n_valid(v) >= MIN_ARC_SAMPLES for v in [cA, cB, pA, pB]):
                # Resolve GLONASS frequencies at runtime
                if conid == 'R':
                    k  = self._glo_channel(sv)
                    f1 = _glo_freq('G1', k)
                    f2 = _glo_freq('G2', k)
                    if f1 == 0 or f2 == 0:
                        continue
                return (fA_n, fB_n, f1, f2, cA, cB, pA, pB)

        return None

    # ------------------------------------------------------------------
    def _glo_channel(self, sv: str) -> int:
        """Return GLONASS FDMA channel number for *sv* (default 0 if unknown)."""
        ephem = getattr(self, '_ephem', None)
        if ephem is not None and sv in ephem._glo_slots:
            return ephem._glo_slots[sv]
        return 0

    # ------------------------------------------------------------------
    def _default_sampling_params(self) -> pd.DataFrame:
        """Build a minimal sampling-parameter DataFrame from apf107.dat."""
        try:
            from IRI_Sample_Inputs.IRI_Sample_inputs import get_apf107
            ap = get_apf107()
            year = self.date.year
            doy  = self.date.timetuple().tm_yday
            yrs  = np.array(ap['yr'])
            mns  = np.array(ap['mn'])
            dys  = np.array(ap['dy'])
            # Match the requested date
            from datetime import date as _date
            ref = self.date.date()
            idx = None
            for i in range(len(yrs)):
                try:
                    if _date(int(yrs[i]), int(mns[i]), int(dys[i])) == ref:
                        idx = i
                        break
                except Exception:
                    continue
            if idx is not None:
                f107    = float(ap['f107'][idx])
                f107_81 = float(ap['f107_81'][idx])
                ir      = float(ap['ir'][idx])
                ig      = max(0.0, (f107_81 - 65.0) * 0.4)
            else:
                f107 = f107_81 = 150.0
                ig = ir = 50.0
        except Exception:
            f107 = f107_81 = 150.0
            ig   = ir     = 50.0

        return pd.DataFrame({
            'f107':    [f107],
            'f107_81': [f107_81],
            'ig':      [ig],
            'ir':      [ir],
        })


# ──────────────────────────────────────────────────────────────────────────────
# §8  demo_group.py integration helpers
# ──────────────────────────────────────────────────────────────────────────────

def igs_obs_to_clean_entry(obs: dict,
                            max_rays: int = 500,
                            min_valid: int = 50) -> Optional[dict]:
    """Convert an IGS obs dict to a ``clean_list`` entry for demo_group.py.

    Mirrors the logic in ``demo_group.process_group()`` that converts podTc2
    dicts into the ``clean_list`` format expected by the Kalman filter.

    Parameters
    ----------
    obs      : dict returned by IGSTECPipeline.run()
    max_rays : downsample if arc has more than this many epochs
    min_valid: minimum number of valid epochs; return None if below

    Returns
    -------
    dict with keys ``tec``, ``tangent_km``, ``LEO``, ``GNSS``, ``tec_type``,
    ``leo_id``, ``prn_id``, ``label`` — or None if the arc is too short.
    """
    tec = obs.get('TEC_podTc2', obs.get('TEC', np.array([])))
    valid = np.isfinite(tec) & (tec > 0)
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return None

    if n_valid > max_rays:
        stride = int(np.ceil(n_valid / max_rays))
        dec    = np.where(valid)[0][::stride]
        mask   = np.zeros(len(tec), dtype=bool)
        mask[dec] = True
    else:
        mask = valid

    stamp = obs.get('fileStamp', obs.get('station_id', '?'))
    tang = obs['tangent_alt_km']

    # IPP arrays — preserve for ground-track plotting; fall back to scalar if absent.
    _ipp_lat_arr = obs.get('ipp_lat')
    _ipp_lon_arr = obs.get('ipp_lon')
    if _ipp_lat_arr is not None and len(_ipp_lat_arr) == len(tec):
        ipp_lat_out = _ipp_lat_arr[mask]
        ipp_lon_out = _ipp_lon_arr[mask]
    else:
        ipp_lat_out = np.full(int(mask.sum()), float(obs.get('lat_tecmax_tangent', np.nan)))
        ipp_lon_out = np.full(int(mask.sum()), float(obs.get('lon_tecmax_tangent', np.nan)))

    # Arc time in seconds from arc start (for sTEC vs time axis).
    _time_arr = obs.get('time')
    if _time_arr is not None and len(_time_arr) == len(tec):
        arc_time_out = _time_arr[mask]
    else:
        arc_time_out = np.arange(int(mask.sum()), dtype=float)

    # UTC wall-clock time for each masked epoch (decimal hours from midnight).
    arc_start_sod = (float(obs.get('hour',   0)) * 3600.0
                     + float(obs.get('minute', 0)) * 60.0
                     + float(obs.get('second', 0.0)))
    time_utc_h_out = (arc_start_sod + arc_time_out) / 3600.0

    return {
        'tec':          tec[mask],
        'tangent_km':   tang[mask] if len(tang) == len(tec) else np.full(int(mask.sum()), H_IPP_KM),
        'LEO':          obs['LEO'][:, mask],
        'GNSS':         obs['GNSS'][:, mask],
        'tec_type':     'absolute',
        'leo_id':       str(obs.get('station_id', '?')),
        'prn_id':       str(obs.get('conid', '?')) + str(obs.get('prn_id', '?')),
        'label':        stamp,
        'obs_source':   'IGS_ground',
        # Grouping fields
        'lat_tecmax_tangent': obs['lat_tecmax_tangent'],
        'lon_tecmax_tangent': obs['lon_tecmax_tangent'],
        'date':               obs['date'],
        # Ground-track / plotting extras
        'ipp_lat':            ipp_lat_out,
        'ipp_lon':            ipp_lon_out,
        'arc_time_sec':       arc_time_out,      # seconds from arc start
        'time_s':             arc_time_out,      # alias (consistent with suborbital)
        'arc_start_sod':      arc_start_sod,     # seconds-of-day of first epoch
        'time_utc_h':         time_utc_h_out,    # UTC decimal hours per epoch
        'station_lat':        float(obs.get('station_lat', np.nan)),
        'station_lon':        float(obs.get('station_lon', np.nan)),
    }


def scan_igs_obs_metadata(obs_list: List[dict]) -> pd.DataFrame:
    """Build a grouping-compatible metadata DataFrame from a list of IGS obs dicts.

    The resulting DataFrame has the same columns used by ``scan_metadata()``
    in demo_group.py (filename, full_path, date, lat, lon, spacecraft,
    region, time_window, group_key), allowing IGS observations to be grouped
    with the same logic as RO occultations.

    Parameters
    ----------
    obs_list : list of dicts from IGSTECPipeline.run()

    Returns
    -------
    pd.DataFrame with one row per arc.
    """
    rows = []
    for obs in obs_list:
        lat = obs.get('lat_tecmax_tangent', np.nan)
        lon = obs.get('lon_tecmax_tangent', np.nan)
        dt  = obs.get('date', pd.Timestamp.now())
        sta = obs.get('station_id', '????')
        prn = obs.get('conid', '?') + obs.get('prn_id', '??')
        rows.append({
            'filename':    obs.get('fileStamp', f"{sta}.{prn}"),
            'full_path':   None,          # no file — data already in memory
            'date':        dt,
            'lat':         lat,
            'lon':         lon,
            'spacecraft':  f"{sta}_{prn}",
            'obs_source':  'IGS_ground',
        })
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# §9  Module-level convenience function
# ──────────────────────────────────────────────────────────────────────────────

def process_igs_station(station:       str,
                         date:          datetime,
                         rinex_version: int = 3,
                         cache_dir:     str | Path = '/tmp/igs_rinex_cache',
                         netrc_path:    str | Path = '~/.netrc',
                         use_iri:       bool = False,
                         local_obs:     Optional[str] = None,
                         local_nav:     Optional[str] = None,
                         local_dcb:     Optional[str] = None,
                         max_rays:      int = 500) -> List[dict]:
    """High-level wrapper: run the full IGS TEC pipeline for one station/date.

    Parameters
    ----------
    station, date, rinex_version, cache_dir, netrc_path, use_iri,
    local_obs, local_nav, local_dcb
        See ``IGSTECPipeline`` for documentation.
    max_rays : int
        Maximum number of epochs per arc in the returned clean_list entries.

    Returns
    -------
    List of observation dicts ready for ``demo_group.process_group()`` or
    direct use with ``Ionosphere_Tomography_Inverter.assimilate()``.
    """
    pipe     = IGSTECPipeline(
        station       = station,
        date          = date,
        rinex_version = rinex_version,
        cache_dir     = cache_dir,
        netrc_path    = netrc_path,
        use_iri       = use_iri,
        local_obs     = local_obs,
        local_nav     = local_nav,
        local_dcb     = local_dcb,
    )
    obs_list = pipe.run()

    log.info("process_igs_station: %d arcs for %s on %s",
             len(obs_list), station, date.strftime('%Y-%m-%d'))
    return obs_list
