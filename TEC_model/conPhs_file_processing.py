# -*- coding: utf-8 -*-
"""
conPhs_file_processing.py

Utilities for finding and parsing conPhs (connected-phase) NetCDF files
and computing relative dual-frequency TEC for assimilation.

Public API
----------
load_conPhs(input_file_path, ...)
    Accept either a podTc2 or conPhs file path.  Locates the conPhs file,
    parses it, and returns a data dict analogous to parse_podTc2_nc_file().

find_conPhs_for_podTc(podTc_file_path, ...)
    Given a podTc2 file path, return the path of the best-matching conPhs
    file (matched on LEO ID, year, DOY, PRN, and approximate start time).

parse_conPhs_nc_file(file_path)
    Parse a single conPhs NetCDF file and return a structured data dict.

Naming conventions assumed
--------------------------
  podTc2 : podTc2_LEOX.YYYY.DDD.HH.MM.UUUU.PRN.TT_0000.0001_nc
  conPhs  : conPhs_LEOX.YYYY.DDD.HH.MM.PRN_0001.0001_nc

TEC formula
-----------
Relative TEC is derived from the dual-frequency excess-phase difference:

    rel_TEC [TECU] = (exL1 - exL2) * f1**2 * f2**2
                     / (40.3e16 * (f2**2 - f1**2))

exL1 / exL2 are in metres (positive group-delay convention used by CDAAC).
The result carries an arbitrary bias (unknown carrier-phase integer ambiguity);
absolute calibration or bias estimation is left to the calling code.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import netCDF4
from datetime import datetime, timedelta

# 40.3 m³ s⁻² scaled for TECU output (1 TECU = 1×10¹⁶ el m⁻²)
_IONO_CONST = 40.3e16


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _doy_to_date(year: int, doy: int) -> datetime:
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def _parse_podtc2_filename(filename: str) -> dict | None:
    """
    Extract identification fields from a podTc2 filename.
    Format: podTc2_LEOX.YYYY.DDD.HH.MM.UUUU.PRN.TT_…_nc
    """
    base = os.path.basename(filename)
    m = re.match(
        r'podTc2_([A-Z0-9]+)\.(\d{4})\.(\d{3})\.(\d{2})\.(\d{2})\.\d+\.([A-Z]\d+)\.\d+_',
        base
    )
    if not m:
        return None
    return {
        'leo_id': m.group(1),
        'year':   int(m.group(2)),
        'doy':    int(m.group(3)),
        'hour':   int(m.group(4)),
        'minute': int(m.group(5)),
        'prn':    m.group(6),
    }


def _parse_conphs_filename(filename: str) -> dict | None:
    """
    Extract identification fields from a conPhs filename.
    Format: conPhs_LEOX.YYYY.DDD.HH.MM.PRN_…_nc
    """
    base = os.path.basename(filename)
    m = re.match(
        r'conPhs_([A-Z0-9]+)\.(\d{4})\.(\d{3})\.(\d{2})\.(\d{2})\.([A-Z]\d+)_',
        base
    )
    if not m:
        return None
    return {
        'leo_id': m.group(1),
        'year':   int(m.group(2)),
        'doy':    int(m.group(3)),
        'hour':   int(m.group(4)),
        'minute': int(m.group(5)),
        'prn':    m.group(6),
    }


def _candidate_conphs_dirs(input_dir: str, year: int, doy: int) -> list:
    """
    Return a list of directories to search for conPhs files given the
    directory of the input podTc2 file.

    Tries, in order:
      1. Replace 'podTc2' with 'conPhs' in the full directory path
         (works for both AuroraData/podTc2/YYYY-MM-DD and
          level1b/podTc2/YYYY.DDD structures).
      2. Walk up to three parent levels looking for a 'conPhs' sibling,
         checking for matching YYYY-MM-DD / YYYY.DDD / YYYY.DDDD sub-dirs.
    """
    candidates = []

    # Strategy 1: simple string substitution
    direct = input_dir.replace('podTc2', 'conPhs')
    if os.path.isdir(direct):
        candidates.append(direct)

    # Strategy 2: look for a 'conPhs' sibling up the tree with matching sub-dir
    date = _doy_to_date(year, doy)
    date_fmt_variants = [
        date.strftime('%Y-%m-%d'),
        f'{year}.{doy:03d}',
        f'{year}.{doy:04d}',
    ]

    parent = input_dir
    for _ in range(4):
        parent = os.path.dirname(parent)
        sibling = os.path.join(parent, 'conPhs')
        if os.path.isdir(sibling):
            for fmt in date_fmt_variants:
                full = os.path.join(sibling, fmt)
                if os.path.isdir(full) and full not in candidates:
                    candidates.append(full)
            # Also add the sibling root itself in case no sub-dirs are used
            if sibling not in candidates:
                candidates.append(sibling)

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_conPhs_for_podTc(podTc_file_path: str,
                           conPhs_base_dir: str = None,
                           time_window_min: float = 15.0) -> str | None:
    """
    Find the conPhs file that corresponds to a given podTc2 file.

    Matching criteria (all must hold):
      - LEO satellite ID  (exact)
      - Year + DOY        (exact)
      - GNSS PRN          (exact)
      - Start time offset ≤ time_window_min minutes

    Parameters
    ----------
    podTc_file_path : str
        Absolute path to a podTc2 NetCDF file.
    conPhs_base_dir : str, optional
        Root directory to search for conPhs files.  When given, the function
        looks for YYYY-MM-DD / YYYY.DDD sub-directories inside it.  When
        omitted, the conPhs directory is inferred from the podTc2 path by
        replacing 'podTc2' with 'conPhs'.
    time_window_min : float
        Maximum allowed start-time difference in minutes.

    Returns
    -------
    str or None
        Path to the best-matching conPhs file, or None if not found.
    """
    fields = _parse_podtc2_filename(podTc_file_path)
    if fields is None:
        raise ValueError(
            f"Cannot parse podTc2 filename: {os.path.basename(podTc_file_path)}"
        )

    leo_id        = fields['leo_id']
    year          = fields['year']
    doy           = fields['doy']
    hour          = fields['hour']
    minute        = fields['minute']
    prn           = fields['prn']
    podtc_minutes = hour * 60 + minute

    # Build search directory list
    if conPhs_base_dir is not None:
        date = _doy_to_date(year, doy)
        search_dirs = []
        for fmt in [date.strftime('%Y-%m-%d'), f'{year}.{doy:03d}', f'{year}.{doy:04d}']:
            d = os.path.join(conPhs_base_dir, fmt)
            if os.path.isdir(d):
                search_dirs.append(d)
        if not search_dirs:
            search_dirs = [conPhs_base_dir]
    else:
        search_dirs = _candidate_conphs_dirs(
            os.path.dirname(podTc_file_path), year, doy
        )

    if not search_dirs:
        print(f"[conPhs] No candidate directories for {os.path.basename(podTc_file_path)}")
        return None

    best_path  = None
    best_delta = float('inf')
    pattern    = f'conPhs_{leo_id}.{year}.{doy:03d}.*.{prn}_*'

    for search_dir in search_dirs:
        for match in glob.glob(os.path.join(search_dir, pattern)):
            parsed = _parse_conphs_filename(match)
            if parsed is None:
                continue
            delta = abs((parsed['hour'] * 60 + parsed['minute']) - podtc_minutes)
            if delta < best_delta:
                best_delta = delta
                best_path  = match

    if best_path is None:
        print(f"[conPhs] No matching file for {os.path.basename(podTc_file_path)}")
        return None

    if best_delta > time_window_min:
        print(
            f"[conPhs] Closest match is {best_delta:.0f} min away "
            f"(threshold={time_window_min} min) — {os.path.basename(best_path)}"
        )
        return None

    return best_path


def parse_conPhs_nc_file(file_path: str) -> dict | None:
    """
    Parse a conPhs NetCDF file and return a structured data dictionary
    analogous to parse_podTc2_nc_file() from podTc_file_processing.py.

    Returned dictionary keys
    ------------------------
    time            : 1-D float array, seconds elapsed from occultation start
    rel_TEC         : 1-D float array, relative dual-frequency TEC (TECU)
    tangent_alt_km  : 1-D float array, tangent height filtered to h > 0 km
    LEO             : (3, N) float array, LEO ECEF positions interpolated to
                      the high-rate time grid (km)
    GNSS            : (3, N) float array, GNSS ECEF positions (km)
    occ_type        : 'setting' or 'rising' (after normalising to descending
                      heights, matching the convention of parse_podTc2_nc_file)
    f1, f2          : carrier frequencies (Hz)
    leo_id          : LEO satellite integer ID string
    prn_id          : GNSS PRN integer ID string
    conid           : GNSS constellation letter (G/R/E/C)
    fileStamp       : raw file-stamp attribute (e.g. 'GN04.2024.131.00.00.C43')
    DOY             : day-of-year (int)
    date            : pd.Timestamp of occultation start
    year/month/day/hour/minute/second : int/float scalars
    exL1, exL2      : unfiltered excess-phase arrays (m), full 100-Hz record
    caL1Snr, pL2Snr : SNR arrays (full record)

    Returns None if the file is missing, flagged bad, or contains no valid
    ionospheric data (positive tangent heights with finite excess phase).
    """
    if not os.path.isfile(file_path):
        print(f"[conPhs] File not found: {file_path}")
        return None

    with netCDF4.Dataset(file_path, 'r') as nc:
        data = {}

        # -- Attributes --------------------------------------------------
        data['year']      = int(nc.getncattr('year'))
        data['month']     = int(nc.getncattr('month'))
        data['day']       = int(nc.getncattr('day'))
        data['hour']      = int(nc.getncattr('hour'))
        data['minute']    = int(nc.getncattr('minute'))
        data['second']    = float(nc.getncattr('second'))
        data['DOY']       = int(nc.getncattr('dayOfYear'))
        data['conid']     = str(nc.getncattr('conId'))
        data['leo_id']    = str(int(nc.getncattr('leoId')))
        data['prn_id']    = str(int(nc.getncattr('occsatId')))
        data['f1']        = float(nc.getncattr('occfreq1'))
        data['f2']        = float(nc.getncattr('occfreq2'))
        data['fileStamp'] = str(nc.getncattr('fileStamp'))
        data['startTime'] = float(nc.getncattr('startTime'))  # GPS seconds
        data['stopTime']  = float(nc.getncattr('stopTime'))

        if str(nc.getncattr('bad')).strip().upper() not in ('0', 'OK', 'FALSE', ''):
            print(f"[conPhs] File flagged bad ({nc.getncattr('bad')}): "
                  f"{os.path.basename(file_path)}")
            return None

        data['date'] = pd.Timestamp(
            year=data['year'], month=data['month'], day=data['day'],
            hour=data['hour'], minute=data['minute'],
            second=int(data['second'])
        )

        # -- High-rate time and phase arrays -----------------------------
        time_hr     = np.array(nc.variables['time'][:], dtype=float)
        occheight   = np.ma.filled(nc.variables['occheight'][:], np.nan).astype(float)
        exL1        = np.ma.filled(nc.variables['exL1'][:],      np.nan).astype(float)
        exL2        = np.ma.filled(nc.variables['exL2'][:],      np.nan).astype(float)
        caL1Snr     = np.ma.filled(nc.variables['caL1Snr'][:],   np.nan).astype(float)
        pL2Snr      = np.ma.filled(nc.variables['pL2Snr'][:],    np.nan).astype(float)

        data['exL1']    = exL1
        data['exL2']    = exL2
        data['caL1Snr'] = caL1Snr
        data['pL2Snr']  = pL2Snr

        # -- Low-rate orbit positions → interpolate to high-rate grid ----
        # orbtime is in GPS absolute seconds; startTime is the same epoch.
        # orbtime - startTime therefore aligns with the 'time' variable.
        orbtime = np.array(nc.variables['orbtime'][:], dtype=float)
        orbit_t = orbtime - data['startTime']

        def _interp_orbit(var_name):
            raw = np.array(nc.variables[var_name][:], dtype=float)
            return np.interp(time_hr, orbit_t, raw)

        x_leo  = _interp_orbit('xLeoLR')
        y_leo  = _interp_orbit('yLeoLR')
        z_leo  = _interp_orbit('zLeoLR')
        x_gnss = _interp_orbit('xGnssLR')
        y_gnss = _interp_orbit('yGnssLR')
        z_gnss = _interp_orbit('zGnssLR')

    # -- Relative TEC ---------------------------------------------------
    # exL1, exL2 are excess phase in metres (positive = group-delay convention).
    # For f1 > f2 (e.g. L1 > L5), (f2² - f1²) < 0 and (exL1 - exL2) < 0,
    # so the product is positive → physically correct positive TEC.
    f1, f2 = data['f1'], data['f2']
    rel_TEC = -(exL1 - exL2) * (f1**2 * f2**2) / (_IONO_CONST * (f2**2 - f1**2))

    # -- QC: keep only samples with positive heights and finite phase ----
    valid = (
        np.isfinite(rel_TEC) &
        np.isfinite(occheight) &
        (occheight > 0.0)
    )
    if not np.any(valid):
        print(f"[conPhs] No valid ionospheric data: {os.path.basename(file_path)}")
        return None

    t_f   = time_hr[valid]
    tec_f = rel_TEC[valid]
    h_f   = occheight[valid]
    LEO   = np.array([x_leo[valid],  y_leo[valid],  z_leo[valid]])
    GNSS  = np.array([x_gnss[valid], y_gnss[valid], z_gnss[valid]])

    # -- Normalise to descending heights (same convention as ---------------
    # -- parse_podTc2_nc_file: heights go from high to low) ---------------
    if h_f[0] < h_f[-1]:
        # Data is ascending (rising geometry) → flip to descending
        t_f   = np.flip(t_f)
        tec_f = np.flip(tec_f)
        h_f   = np.flip(h_f)
        LEO   = np.flip(LEO,  axis=1)
        GNSS  = np.flip(GNSS, axis=1)
        occ_type = 'setting'
    else:
        occ_type = 'rising'

    data['time']           = t_f
    data['rel_TEC']        = tec_f
    data['tangent_alt_km'] = h_f
    data['LEO']            = LEO
    data['GNSS']           = GNSS
    data['occ_type']       = occ_type

    return data


def load_conPhs(input_file_path: str,
                conPhs_base_dir: str = None,
                time_window_min: float = 15.0) -> dict | None:
    """
    Accept either a podTc2 or conPhs file path, locate the corresponding
    conPhs file, parse it, and return the data dictionary.

    Parameters
    ----------
    input_file_path : str
        Absolute path to a podTc2 *or* conPhs NetCDF file.
    conPhs_base_dir : str, optional
        Explicit root directory to search when resolving a podTc2 → conPhs
        match.  Not used when input_file_path already points to a conPhs file.
    time_window_min : float
        Maximum allowed start-time offset (minutes) when matching
        podTc2 → conPhs.

    Returns
    -------
    dict or None
        Parsed conPhs data, or None if the file cannot be found or parsed.

    Raises
    ------
    ValueError
        If the input filename does not match either expected prefix.
    """
    base = os.path.basename(input_file_path)

    if base.startswith('conPhs_'):
        conphs_path = input_file_path
    elif base.startswith('podTc2_'):
        conphs_path = find_conPhs_for_podTc(
            input_file_path, conPhs_base_dir, time_window_min
        )
        if conphs_path is None:
            return None
    else:
        raise ValueError(
            f"Expected a podTc2 or conPhs filename, got: {base}"
        )

    return parse_conPhs_nc_file(conphs_path)
