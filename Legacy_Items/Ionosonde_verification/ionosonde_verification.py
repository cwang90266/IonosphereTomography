"""
Ionosonde Verification Module

Downloads GIRO/DIDBase digisonde soundings that are spatially and temporally
collocated with an occultation, then compares the true-height EDP against the
tomographic retrieval.

Public API
----------
run_ionosonde_verification(...)   main entry point, called from section20
STATIONS                          dict of {URSI_code: (lat_N, lon_E)}
"""

from __future__ import annotations

import os
import re
import time
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
#  GIRO Station Master List  {URSI_code: (lat_N, lon_E)}
# ─────────────────────────────────────────────────────────────────────────────
STATIONS: dict[str, tuple[float, float]] = {
    # ── North America ──────────────────────────────────────────────────────
    'BC840': ( 40.0, -105.3),   # Boulder, CO
    'EG931': ( 42.6,  -71.5),   # Millstone Hill, MA
    'WP937': ( 37.9,  -75.5),   # Wallops Island, VA
    'FF051': ( 50.2,  -66.7),   # Sept-Iles, QC
    'FP105': ( 45.4,  -75.9),   # Shirleys Bay, ON
    'PR118': ( 18.5,  -67.1),   # Ramey, Puerto Rico
    'JA836': ( 64.9, -147.8),   # Gakona, AK
    'MH453': ( 21.3, -158.0),   # Kauai, HI
    'VA112': ( 48.5, -123.4),   # Victoria, BC
    # ── Europe ─────────────────────────────────────────────────────────────
    'EG031': ( 37.1,   -6.7),   # El Arenosillo, Spain
    'RL052': ( 51.6,   -1.3),   # Chilton, UK
    'DB049': ( 50.1,    4.6),   # Dourbes, Belgium
    'JR055': ( 54.6,   13.4),   # Juliusruh, Germany
    'AT138': ( 38.0,   23.5),   # Athens, Greece
    'SO148': ( 43.5,   16.4),   # Split, Croatia
    'PQ052': ( 50.0,   14.6),   # Pruhonice, Czech Republic
    'RO041': ( 44.4,   26.0),   # Bucharest, Romania
    'MO155': ( 55.5,   37.3),   # Moscow, Russia
    'EB040': ( 47.7,    7.6),   # Ebenhausen, Germany
    # ── Scandinavia / High-Latitude Europe ─────────────────────────────────
    'TR164': ( 69.7,   19.0),   # Tromso, Norway
    'SO166': ( 67.4,   26.6),   # Sodankyla, Finland
    'HO156': ( 77.0,   15.5),   # Hornsund, Svalbard
    'MA046': ( 59.9,   30.7),   # St. Petersburg, Russia
    # ── Russia / Central Asia ──────────────────────────────────────────────
    'SA135': ( 66.5,   66.6),   # Salekhard, Russia
    'IC432': ( 52.9,  104.0),   # Irkutsk, Russia
    'NO135': ( 69.2,   88.1),   # Norilsk, Russia
    'KH466': ( 48.5,  135.1),   # Khabarovsk, Russia
    'YA135': ( 62.0,  129.6),   # Yakutsk, Russia
    # ── East Asia / Pacific ────────────────────────────────────────────────
    'BC418': ( 40.3,  116.2),   # Beijing, China
    'WU430': ( 30.5,  114.4),   # Wuhan, China
    'TK535': ( 45.2,  141.8),   # Wakkanai, Japan
    'TO536': ( 35.7,  139.5),   # Kokubunji, Japan
    'OK426': ( 26.7,  128.2),   # Okinawa, Japan
    'TW419': ( 25.0,  121.2),   # Zhongli (Taiwan)
    # ── South / Southeast Asia ─────────────────────────────────────────────
    'ID300': ( 22.0,   79.0),   # Hyderabad, India
    'TI305': (  8.9,   77.6),   # Trivandrum, India
    'KA315': ( 10.2,   77.5),   # Kodaikanal, India
    'BK328': (  1.3,  103.8),   # Singapore
    'MQ156': (  3.6,   98.4),   # Medan, Indonesia
    # ── Oceania / Australia ────────────────────────────────────────────────
    'LM854': (-23.5,  133.7),   # Alice Springs, Australia
    'CA852': (-12.4,  130.9),   # Darwin, Australia
    'CA836': (-22.3,  114.1),   # Learmonth, Australia
    'AS857': (-35.3,  149.0),   # Canberra, Australia
    'HS918': (-43.6,  172.6),   # Christchurch, NZ
    'AU931': (-38.0,  145.0),   # Melbourne, Australia
    # ── Africa / Middle East ───────────────────────────────────────────────
    'TA401': ( 35.7,   51.4),   # Tehran, Iran
    'AS00A': (-33.9,   18.5),   # Hermanus, South Africa
    'ES516': (-25.9,   28.2),   # Johannesburg, South Africa
    'MA280': (-22.4,   17.1),   # Tsumeb, Namibia
    'RO300': ( 12.4,   -1.5),   # Ouagadougou, Burkina Faso
    # ── South America ──────────────────────────────────────────────────────
    'SA418': (-31.7,  -64.5),   # Cordoba, Argentina
    'BR840': (-15.8,  -47.9),   # Brasilia, Brazil
    'TL083': ( -3.7,  -38.5),   # Fortaleza, Brazil
    'JI91J': (-11.9,  -76.9),   # Jicamarca, Peru
    'CC971': (-17.6, -149.6),   # Tahiti, French Polynesia
}

# DIDBase URL templates
_GIRO_FILE_URL  = "https://giro.uml.edu/didbase/v2/data/{station}/{year}/{month}/{day}/"
_LGDC_RAW_URL   = "https://lgdc.uml.edu/common/DIDBRawData"
_LGDC_SCALED    = "https://lgdc.uml.edu/common/DIDBGetValues"

# ─────────────────────────────────────────────────────────────────────────────
#  Day-Level Availability Cache
#  Populated by check_daily_station_availability(); consulted by
#  run_ionosonde_verification() so per-file calls never repeat day-level checks.
#  Key: "{URSI}_{YYYY-MM-DD}"  →  bool (True = station has data that day)
# ─────────────────────────────────────────────────────────────────────────────
_DAY_AVAILABILITY: dict[str, bool] = {}

# ─────────────────────────────────────────────────────────────────────────────
#  Day-Level Sounding Schedule Cache
#  Populated when availability is confirmed; used by fetch_sao_file so it
#  only tries timestamps that are known to carry soundings.
#  Key: "{URSI}_{YYYY-MM-DD}"  →  sorted list[datetime] of sounding times
#  Empty list means "active station but schedule not yet resolved"
# ─────────────────────────────────────────────────────────────────────────────
_DAY_SCHEDULE: dict[str, list[datetime]] = {}

# ─────────────────────────────────────────────────────────────────────────────
#  LGDC Service Health Flag
#  None  = not yet tested
#  True  = service is returning actual ionosonde data
#  False = service is down / returning no data for all queries
#
#  Checked once per batch run.  When False, all per-file HTTP requests to
#  LGDC are suppressed to avoid wasting time on thousands of guaranteed-fail
#  requests across the batch.
# ─────────────────────────────────────────────────────────────────────────────
_LGDC_HEALTHY: Optional[bool] = None

# Standard cadences in seconds (most common first)
_STANDARD_CADENCES_S = (300, 600, 900, 1200, 1800, 3600)


def _check_lgdc_health(batch_date: Optional[datetime] = None, timeout_s: int = 12) -> bool:
    """
    Probe LGDC to determine whether ionosonde data is available for the batch
    date (or a 30-day window around it).

    Uses BC840 (Boulder, CO) as a canary: it is one of the most continuously
    archived stations.  Queries a ±15-day window around ``batch_date`` so
    that a single-day gap on the batch date itself does not trigger a false
    negative.

    If ``batch_date`` is None, uses today minus 1 year as a conservative
    estimate of what an archive should contain.

    Sets and returns the module-level ``_LGDC_HEALTHY`` flag.

    False → all per-file LGDC requests are suppressed for the batch, saving
            ~15 guaranteed-fail HTTP calls per file processed.
    """
    global _LGDC_HEALTHY
    if _LGDC_HEALTHY is not None:
        return _LGDC_HEALTHY

    if batch_date is None:
        from datetime import date as _date_cls
        today = datetime.utcnow()
        batch_date = today - timedelta(days=365)

    window_start = batch_date - timedelta(days=15)
    window_end   = batch_date + timedelta(days=15)

    try:
        r = requests.get(
            _LGDC_SCALED,
            params={
                'ursiCode':   'BC840',
                'charName':   'foF2',
                'fromDate':   window_start.strftime('%Y-%m-%dT00:00:00'),
                'toDate':     window_end.strftime('%Y-%m-%dT23:59:59'),
                'fileformat': '2',
            },
            timeout=timeout_s,
        )
        if r.status_code == 200:
            data_rows = [
                ln for ln in r.text.splitlines()
                if ln.strip()
                and not ln.strip().startswith('#')
                and not ln.strip().upper().startswith('ERROR')
            ]
            _LGDC_HEALTHY = len(data_rows) > 0
        else:
            _LGDC_HEALTHY = False
    except requests.RequestException:
        _LGDC_HEALTHY = False

    if _LGDC_HEALTHY:
        print(f"  [Ionosonde] LGDC health check: service has data near {batch_date.strftime('%Y-%m-%d')}")
    else:
        print(
            f"  [Ionosonde] WARNING: LGDC has no data near {batch_date.strftime('%Y-%m-%d')} "
            f"(checked ±15 days on BC840/foF2).\n"
            "  [Ionosonde] All LGDC queries suppressed — ionosonde verification disabled.\n"
            "  [Ionosonde] Check https://lgdc.uml.edu/ for archive coverage."
        )
    return _LGDC_HEALTHY


def _extract_times_from_lgdc_rows(rows: list[str], date: datetime) -> list[datetime]:
    """
    Parse sounding timestamps from LGDC DIDBGetValues data rows.

    Each row looks like:  2024-10-10 12:00:00, BC840, foF2, 5.5, 44
    Returns a sorted, deduplicated list of datetimes on ``date``.
    """
    times: list[datetime] = []
    date_prefix = date.strftime('%Y-%m-%d')
    for ln in rows:
        ln = ln.strip()
        if not ln or ln.startswith('#') or ln.upper().startswith('ERROR'):
            continue
        # First token should be the date, second the time
        parts = ln.split(',')
        if not parts:
            continue
        ts_str = parts[0].strip()
        # Accept "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS"
        ts_str = ts_str.replace('T', ' ')
        if not ts_str.startswith(date_prefix):
            continue
        try:
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            times.append(dt)
        except ValueError:
            pass
    seen: set = set()
    unique: list[datetime] = []
    for t in times:
        key = t.replace(second=0)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    unique.sort()
    return unique


def _extract_times_from_sao_bytes(content: bytes, date: datetime) -> list[datetime]:
    """
    Scan raw SAO file bytes for "YYYY DOY HHMMSS" timestamp patterns.
    Returns sorted, deduplicated datetimes on ``date``.
    """
    text = content.decode('latin-1', errors='replace')
    year = date.year
    doy  = int(date.strftime('%j'))
    pat  = re.compile(r'\b(\d{4})\s+(\d{3})\s+(\d{6})\b')
    times: list[datetime] = []
    for m in pat.finditer(text):
        try:
            y, d, hms = int(m.group(1)), int(m.group(2)), m.group(3)
            if y != year or d != doy:
                continue
            hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
            times.append(datetime(y, date.month, date.day, hh, mm, ss))
        except (ValueError, OverflowError):
            pass
    seen: set = set()
    unique: list[datetime] = []
    for t in times:
        key = t.replace(second=0)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    unique.sort()
    return unique


def _extrapolate_schedule(known_times: list[datetime], date: datetime) -> list[datetime]:
    """
    Given a handful of known sounding times, infer the station cadence and
    extrapolate to cover the full UTC day (00:00 – 23:55).

    Algorithm:
    1. Compute pairwise gaps between consecutive times.
    2. Find the modal gap; snap it to the nearest standard cadence.
    3. Generate timestamps at that cadence from midnight to 23:59.
    4. Return only those within ±(cadence/2) of the grid.
    """
    if not known_times:
        return []
    if len(known_times) == 1:
        # Single point — can't infer cadence; just return what we have
        return list(known_times)

    gaps_s = [
        int((known_times[i + 1] - known_times[i]).total_seconds())
        for i in range(len(known_times) - 1)
        if (known_times[i + 1] - known_times[i]).total_seconds() > 0
    ]
    if not gaps_s:
        return list(known_times)

    # Modal gap → snap to standard cadence
    modal_gap = Counter(gaps_s).most_common(1)[0][0]
    cadence_s = min(_STANDARD_CADENCES_S, key=lambda c: abs(c - modal_gap))

    # Extrapolate grid: start at midnight, step by cadence
    midnight = datetime(date.year, date.month, date.day, 0, 0, 0)
    n_steps  = 86400 // cadence_s
    grid     = [midnight + timedelta(seconds=cadence_s * i) for i in range(n_steps)]
    return grid


def _fetch_day_schedule(
    code: str,
    date: datetime,
    timeout_s: int = 15,
) -> list[datetime]:
    """
    Determine the full list of sounding timestamps for ``code`` on ``date``.

    Strategy A: LGDC DIDBGetValues foF2 query for the full day — parse
                timestamps directly from the response rows.
    Strategy B: Four DIDBRawData probes (00h, 06h, 12h, 18h ±90 min) —
                extract timestamps from SAO headers, then extrapolate.

    Stores result in ``_DAY_SCHEDULE`` and returns it.
    Empty list returned (and cached) when the station is definitely silent.
    """
    sched_key = f"{code}_{date.strftime('%Y-%m-%d')}"
    if sched_key in _DAY_SCHEDULE:
        return _DAY_SCHEDULE[sched_key]

    times: list[datetime] = []

    # ── Strategy A: LGDC scaled params full-day query ─────────────────────
    # Use full ISO datetime so LGDC returns the complete day's rows.
    try:
        r = requests.get(
            _LGDC_SCALED,
            params={
                'ursiCode':   code,
                'charName':   'foF2',
                'fromDate':   date.strftime('%Y-%m-%dT00:00:00'),
                'toDate':     date.strftime('%Y-%m-%dT23:59:59'),
                'fileformat': '2',
            },
            timeout=timeout_s,
        )
        if r.status_code == 200:
            data_rows = [
                ln for ln in r.text.splitlines()
                if ln.strip()
                and not ln.strip().startswith('#')
                and not ln.strip().upper().startswith('ERROR')
            ]
            if data_rows:
                times = _extract_times_from_lgdc_rows(data_rows, date)
                if times:
                    # Strategy A covers the full day; extrapolate only when very
                    # sparse (< 4 rows), otherwise use the actual timestamps.
                    schedule = _extrapolate_schedule(times, date) if len(times) < 4 else times
                    _DAY_SCHEDULE[sched_key] = schedule
                    return schedule
    except requests.RequestException:
        pass

    # ── Strategy B: RAW data probes at 4 UTC anchor hours ─────────────────
    # Store only observed times (no extrapolation) so a daytime-only station
    # correctly returns an empty window for midnight files.
    anchor_hours = (0, 6, 12, 18)
    b_times: list[datetime] = []
    for i, anchor_h in enumerate(anchor_hours):
        if i > 0:
            time.sleep(0.20)   # stagger within-strategy probes
        anchor   = datetime(date.year, date.month, date.day, anchor_h, 0, 0)
        from_str = (anchor - timedelta(minutes=90)).strftime('%Y-%m-%dT%H:%M:%S')
        to_str   = (anchor + timedelta(minutes=90)).strftime('%Y-%m-%dT%H:%M:%S')
        try:
            r = requests.get(
                _LGDC_RAW_URL,
                params={'ursiCode': code, 'fromDate': from_str,
                        'toDate': to_str, 'fileformat': '1'},
                timeout=timeout_s,
            )
            if r.status_code == 429:
                break   # back off; keep what we have
            if r.status_code == 200 and len(r.content) > 500:
                found = _extract_times_from_sao_bytes(r.content, date)
                b_times.extend(found)
        except requests.RequestException:
            pass

    # Deduplicate and sort Strategy B observations
    seen: set = set()
    unique: list[datetime] = []
    for t in b_times:
        k = t.replace(second=0)
        if k not in seen:
            seen.add(k)
            unique.append(t)
    unique.sort()

    # Only extrapolate if we have a dense sample (≥ 12 observed times covering
    # at least two anchor windows); otherwise keep what we actually saw so that
    # time windows without real data correctly produce an empty hit list.
    if len(unique) >= 12:
        schedule = _extrapolate_schedule(unique, date)
    else:
        schedule = unique   # may be empty if station not reachable via LGDC RAW

    _DAY_SCHEDULE[sched_key] = schedule
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
#  Station Query
# ─────────────────────────────────────────────────────────────────────────────

def find_stations_in_region(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    center_lat: float | None = None, center_lon: float | None = None,
) -> list[tuple[str, float, float, float]]:
    """
    Return GIRO stations inside the bounding box, sorted by angular distance
    to the center point.

    Returns
    -------
    list of (ursi_code, lat, lon, dist_deg)
    """
    crosses_dateline = lon_min > lon_max
    c_lat = center_lat if center_lat is not None else 0.5 * (lat_min + lat_max)
    if crosses_dateline:
        raw_mid = (lon_min + lon_max + 360.0) / 2.0
        c_lon = (center_lon if center_lon is not None else (raw_mid % 360.0) - 180.0)
    else:
        c_lon = center_lon if center_lon is not None else 0.5 * (lon_min + lon_max)

    results: list[tuple[str, float, float, float]] = []
    for code, (lat, lon) in STATIONS.items():
        if not (lat_min <= lat <= lat_max):
            continue
        if crosses_dateline:
            if not ((lon >= lon_min) or (lon <= lon_max)):
                continue
        else:
            if not (lon_min <= lon <= lon_max):
                continue
        cos_lat = np.cos(np.radians(c_lat))
        d = np.sqrt((lat - c_lat) ** 2 + ((lon - c_lon) * cos_lat) ** 2)
        results.append((code, lat, lon, float(d)))

    results.sort(key=lambda x: x[3])
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Day-Level Availability Pre-Screen
# ─────────────────────────────────────────────────────────────────────────────

def _check_one_station(code: str, date: datetime, timeout_s: int) -> tuple[str, bool]:
    """
    Determine whether station ``code`` has ionosonde data on ``date``.

    Strategy 1 — stream-GET the DIDBase daily directory.
      HTTP 200 → confirmed data present.
      Any non-200 (including 404) → inconclusive; the /v2/ URL path is not
      guaranteed to be correct for every node, so we fall through rather than
      marking the station absent.

    Strategy 2 — LGDC DIDBGetValues foF2 query for the full day.
      Real data rows (not comment lines, not GIRO "ERROR:" messages) → True.
      Well-formed GIRO "no data" response (comment header + ERROR line) → False.
      HTTP 429 (rate-limited) or network error → conservative True.

    Strategy 3 — DIDBRawData probe (noon ±1 h window).
      HTTP 200 with meaningful content (>500 bytes) → True.
      Used when Strategy 2 is rate-limited or unavailable.

    Conservative default → True (never silently drop a station on ambiguity).
    """
    day_key = f"{code}_{date.strftime('%Y-%m-%d')}"
    if day_key in _DAY_AVAILABILITY:
        return code, _DAY_AVAILABILITY[day_key]

    year_s  = date.strftime('%Y')
    month_s = date.strftime('%m')
    day_s   = date.strftime('%d')

    # ── Strategy 1: DIDBase daily directory ──────────────────────────────
    # Only accept 200 as confirmation; treat 404 as inconclusive (URL may
    # not be the correct path for all DIDBase nodes).
    dir_url = _GIRO_FILE_URL.format(
        station=code, year=year_s, month=month_s, day=day_s
    )
    try:
        r = requests.get(dir_url, timeout=timeout_s, stream=True)
        r.close()
        if r.status_code == 200:
            _DAY_AVAILABILITY[day_key] = True
            return code, True
        # Non-200 (404, 403, 5xx …) → fall through, do NOT mark False here
    except requests.RequestException:
        pass

    # ── Strategy 2: LGDC DIDBGetValues foF2 query ─────────────────────────
    try:
        r = requests.get(
            _LGDC_SCALED,
            params={
                'ursiCode':   code,
                'charName':   'foF2',
                'fromDate':   date.strftime('%Y-%m-%dT00:00:00'),
                'toDate':     date.strftime('%Y-%m-%dT23:59:59'),
                'fileformat': '2',
            },
            timeout=timeout_s,
        )
        if r.status_code == 429:
            # Rate-limited — can't confirm either way; stay conservative
            _DAY_AVAILABILITY[day_key] = True
            return code, True

        if r.status_code == 200:
            lines = r.text.splitlines()
            # Real data rows: non-empty, not a comment (#), not a GIRO error message
            data_lines = [
                ln for ln in lines
                if ln.strip()
                and not ln.strip().startswith('#')
                and not ln.strip().upper().startswith('ERROR')
            ]
            comment_lines = [ln for ln in lines if ln.strip().startswith('#')]

            if data_lines:
                # Positive confirmation: actual foF2 values present
                _DAY_AVAILABILITY[day_key] = True
                # Free schedule extraction — we already have the full-day rows
                sched_key = day_key  # same key format: "{URSI}_{YYYY-MM-DD}"
                if sched_key not in _DAY_SCHEDULE:
                    times = _extract_times_from_lgdc_rows(data_lines, date)
                    if times:
                        sched = (_extrapolate_schedule(times, date)
                                 if len(times) < 4 else times)
                        _DAY_SCHEDULE[sched_key] = sched
                return code, True

            if comment_lines:
                # Well-formed GIRO response (has header) but zero data rows →
                # station has no data in LGDC for this day.  Note: LGDC coverage
                # is not universal; stations in other DIDBase nodes (e.g. EISCAT,
                # national networks) will appear here as "no data" even when active.
                # Strategy 3 gives them a second chance via the RAW file probe.
                pass  # fall through to Strategy 3 before concluding False

    except requests.RequestException:
        pass

    # ── Strategy 3: DIDBRawData noon probe ────────────────────────────────
    # Try to fetch a ~1-hour raw SAO window around local noon to catch stations
    # that are active but not indexed in LGDC's scaled-parameter database.
    try:
        r = requests.get(
            _LGDC_RAW_URL,
            params={
                'ursiCode':   code,
                'fromDate':   date.strftime('%Y-%m-%dT11:30:00'),
                'toDate':     date.strftime('%Y-%m-%dT12:30:00'),
                'fileformat': '1',
            },
            timeout=timeout_s,
        )
        if r.status_code == 429:
            _DAY_AVAILABILITY[day_key] = True
            return code, True
        if r.status_code == 200 and len(r.content) > 500:
            _DAY_AVAILABILITY[day_key] = True
            # Extract the sounding timestamps that fell in this noon window and
            # cache them so fetch_sao_file can skip files outside operating hours.
            sched_key = day_key
            if sched_key not in _DAY_SCHEDULE:
                noon_times = _extract_times_from_sao_bytes(r.content, date)
                if noon_times:
                    # Store only observed times (no extrapolation) — this gives
                    # accurate operating-hours info for the noon window.
                    _DAY_SCHEDULE[sched_key] = noon_times
            return code, True
        if r.status_code == 200:
            # 200 but tiny body → likely an empty/error response → no data
            _DAY_AVAILABILITY[day_key] = False
            return code, False
    except requests.RequestException:
        pass

    # Conservative default: neither confirmed present nor absent → assume present
    _DAY_AVAILABILITY[day_key] = True
    return code, True


def check_daily_station_availability(
    stations: list[str],
    date: datetime,
    timeout_s: int = 10,
    max_workers: int = 4,
) -> dict[str, bool]:
    """
    Pre-screen GIRO stations for ionosonde data on ``date``.
    Call this **once per batch day** before the per-file loop.

    Each station is checked with up to three strategies (see
    ``_check_one_station``).  Results are stored in ``_DAY_AVAILABILITY``
    so per-file calls to ``run_ionosonde_verification`` consult the cache
    instead of repeating network requests.

    ``max_workers`` is deliberately small (default 4) to avoid triggering
    LGDC's rate limiter; requests are also staggered by 150 ms between
    thread submissions.

    Parameters
    ----------
    stations    : URSI codes to check
    date        : day to screen (time component is ignored)
    timeout_s   : per-request timeout in seconds
    max_workers : thread-pool size (keep ≤ 5 to avoid 429s)

    Returns
    -------
    ``{ursi_code: has_data}`` for every code in ``stations``
    """
    # ── LGDC health gate ──────────────────────────────────────────────────
    # Run the canary query once per batch.  If the service has no data near
    # this date, mark every station False so no per-file requests are wasted.
    if not _check_lgdc_health(batch_date=date, timeout_s=timeout_s):
        day_str = date.strftime('%Y-%m-%d')
        dead: dict[str, bool] = {}
        for c in stations:
            key = f"{c}_{day_str}"
            _DAY_AVAILABILITY[key] = False
            dead[c] = False
        return dead

    day_str  = date.strftime('%Y-%m-%d')
    cached   = {c: _DAY_AVAILABILITY[f"{c}_{day_str}"]
                for c in stations if f"{c}_{day_str}" in _DAY_AVAILABILITY}
    to_check = [c for c in stations if c not in cached]

    results: dict[str, bool] = dict(cached)

    if to_check:
        n_workers = min(max_workers, len(to_check))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures: dict = {}
            for i, code in enumerate(to_check):
                if i > 0:
                    time.sleep(0.15)   # 150 ms stagger to avoid rate-limiting
                futures[pool.submit(_check_one_station, code, date, timeout_s)] = code
            for fut in as_completed(futures):
                code, has_data = fut.result()
                results[code] = has_data

    # Note: sounding schedules (_DAY_SCHEDULE) are NOT pre-fetched here for all
    # active stations; that would fire ~55 × 5 requests and trigger LGDC's rate
    # limiter.  Instead, _fetch_day_schedule is called lazily inside
    # fetch_sao_file the first time a station is needed for a specific file.
    # Strategy 2 above already populates _DAY_SCHEDULE for any stations that
    # are indexed in LGDC (free, from the same response).
    # Strategy 3 above populates _DAY_SCHEDULE with the noon-window observed
    # times for stations reachable via DIDBRawData.

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  File Download
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sao_file(
    station: str,
    target_dt: datetime,
    window_minutes: int = 30,
    download_dir: str = ".",
    timeout_s: int = 20,
) -> Optional[str]:
    """
    Search DIDBase for an .SAO file within ±window_minutes of target_dt.

    Tries the giro.uml.edu direct-file URL first; falls back to the
    lgdc.uml.edu raw-data endpoint.

    Returns
    -------
    Local file path if a file was downloaded, else None.
    """
    # Both LGDC raw and the DIDBase direct-file URL are currently unavailable
    # when the service is down; skip all HTTP work immediately.
    if _LGDC_HEALTHY is False:
        return None

    os.makedirs(download_dir, exist_ok=True)

    year_s  = target_dt.strftime('%Y')
    month_s = target_dt.strftime('%m')
    day_s   = target_dt.strftime('%d')
    doy_s   = target_dt.strftime('%j')

    # ── Use schedule cache when available ─────────────────────────────────
    sched_key = f"{station}_{target_dt.strftime('%Y-%m-%d')}"
    if sched_key not in _DAY_SCHEDULE:
        # Lazy population: fetch schedule on first per-file call
        _fetch_day_schedule(station, target_dt, timeout_s=timeout_s)

    schedule = _DAY_SCHEDULE.get(sched_key, [])

    window_s = window_minutes * 60
    if schedule:
        # Only try timestamps known to carry soundings
        times = [
            t for t in schedule
            if abs((t - target_dt).total_seconds()) <= window_s
        ]
        if not times:
            print(f"  [{station}] Schedule known — no soundings within "
                  f"{window_minutes} min of {target_dt.strftime('%H:%M')} UTC")
            return None
        times.sort(key=lambda x: abs((x - target_dt).total_seconds()))
    else:
        # No schedule info — fall back to blind 5-min cadence scan
        start = target_dt - timedelta(minutes=window_minutes)
        end   = target_dt + timedelta(minutes=window_minutes)
        t = start.replace(second=0, microsecond=0)
        t = t.replace(minute=(t.minute // 5) * 5)
        times = []
        while t <= end:
            times.append(t)
            t += timedelta(minutes=5)
        times.sort(key=lambda x: abs((x - target_dt).total_seconds()))

    # Strategy 1: direct file URL
    base_url = _GIRO_FILE_URL.format(
        station=station, year=year_s, month=month_s, day=day_s
    )
    for ct in times:
        hhmmss   = ct.strftime('%H%M%S')
        filename = f"{station}_{year_s}{doy_s}{hhmmss}.SAO"
        local    = os.path.join(download_dir, filename)
        if os.path.exists(local):
            print(f"  [{station}] Using cached: {filename}")
            return local
        try:
            r = requests.get(base_url + filename, timeout=timeout_s)
            if r.status_code == 200 and len(r.content) > 200:
                with open(local, 'wb') as fh:
                    fh.write(r.content)
                print(f"  [{station}] Downloaded {filename} ({len(r.content)//1024} KB)")
                return local
        except requests.RequestException:
            pass

    # Strategy 2: LGDC raw-data endpoint
    try:
        from_str = (target_dt - timedelta(minutes=window_minutes)).strftime('%Y-%m-%dT%H:%M:%S')
        to_str   = (target_dt + timedelta(minutes=window_minutes)).strftime('%Y-%m-%dT%H:%M:%S')
        r = requests.get(
            _LGDC_RAW_URL,
            params={'ursiCode': station, 'fromDate': from_str,
                    'toDate': to_str, 'fileformat': '1'},
            timeout=timeout_s * 2,
        )
        if r.status_code == 200 and len(r.content) > 500:
            filename = f"{station}_{year_s}{doy_s}_lgdc.SAO"
            local    = os.path.join(download_dir, filename)
            with open(local, 'wb') as fh:
                fh.write(r.content)
            print(f"  [{station}] Downloaded via LGDC ({len(r.content)//1024} KB)")
            # Cache the timestamps from this response so future files in the same
            # batch can skip this station if they're outside its operating hours.
            found_times = _extract_times_from_sao_bytes(r.content, target_dt)
            if found_times:
                existing = _DAY_SCHEDULE.get(sched_key, [])
                merged   = sorted(set(existing) | set(found_times))
                _DAY_SCHEDULE[sched_key] = merged
            return local
    except requests.RequestException:
        pass

    print(f"  [{station}] No SAO file found within {window_minutes} min of {target_dt}")
    return None


def fetch_scaled_params(
    station: str,
    target_dt: datetime,
    window_minutes: int = 30,
    timeout_s: int = 20,
) -> Optional[dict]:
    """
    Fetch autoscaled parameters (foF2, hmF2, …) from the LGDC REST API.
    Used as a last resort to construct a Chapman-profile proxy EDP.

    Returns a dict of floats keyed by parameter name, or None on failure.
    """
    if _LGDC_HEALTHY is False:
        return None
    from_str = (target_dt - timedelta(minutes=window_minutes)).strftime('%Y-%m-%dT%H:%M:%S')
    to_str   = (target_dt + timedelta(minutes=window_minutes)).strftime('%Y-%m-%dT%H:%M:%S')
    try:
        r = requests.get(
            _LGDC_SCALED,
            params={
                'ursiCode':   station,
                'charName':   'foF2,hmF2,foE,hmE,foF1,hmF1',
                'fromDate':   from_str,
                'toDate':     to_str,
                'fileformat': '2',
            },
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return None

        records: list[dict] = []
        headers: list[str] = []
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Some LGDC responses embed headers in comment lines
                candidate = re.sub(r'^#+\s*', '', line).split(',')
                if len(candidate) >= 3:
                    headers = [h.strip() for h in candidate]
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                if headers and len(headers) == len(parts):
                    row = {}
                    for h, v in zip(headers, parts):
                        try:
                            row[h] = float(v.strip())
                        except ValueError:
                            row[h] = v.strip()
                    records.append(row)
                else:
                    records.append({
                        'foF2': float(parts[1]) if len(parts) > 1 else np.nan,
                        'hmF2': float(parts[2]) if len(parts) > 2 else np.nan,
                    })
            except (ValueError, IndexError):
                continue

        if records:
            mid = len(records) // 2
            return records[mid]
    except requests.RequestException:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ne_from_fp(fp_mhz: np.ndarray) -> np.ndarray:
    """Plasma frequency (MHz) → electron density (m⁻³)."""
    return np.asarray(fp_mhz, dtype=float) ** 2 * 1.2401e10


def _fp_from_ne(ne_m3: np.ndarray) -> np.ndarray:
    """Electron density (m⁻³) → plasma frequency (MHz)."""
    return np.sqrt(np.maximum(np.asarray(ne_m3, dtype=float), 0.0) * 80.64) / 1e6


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return float(2.0 * R * np.arcsin(np.sqrt(a)))


def _extract_robust_f2_peak(
    ne: np.ndarray, alt: np.ndarray,
    min_alt: float = 150.0, max_alt: float = 650.0,
) -> tuple[float, float]:
    """Return (NmF2, hmF2) restricted to F-region altitudes."""
    mask = (alt >= min_alt) & (alt <= max_alt) & np.isfinite(ne)
    if not np.any(mask):
        return float(np.nan), float(np.nan)
    idx  = int(np.argmax(ne[mask]))
    alts = alt[mask]
    prof = ne[mask]
    return float(prof[idx]), float(alts[idx])


# ─────────────────────────────────────────────────────────────────────────────
#  SAO Parser
# ─────────────────────────────────────────────────────────────────────────────

def _find_section_start(lines: list[str], section_num: int) -> int:
    """
    Return the index of the line AFTER the SAO 2.0 section tag, or -1.

    A tag line is one where every whitespace-separated token equals section_num,
    e.g. "  63  63  63  63  63  63 ...".
    """
    target = str(section_num)
    for i, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) >= 5 and all(t == target for t in tokens):
            return i + 1
    return -1


def _read_section_values(lines: list[str], start: int) -> list[float]:
    """Read all floats from an SAO data section, stopping at the next tag line."""
    vals: list[float] = []
    for line in lines[start:]:
        tokens = line.strip().split()
        if not tokens:
            continue
        # Another section tag — stop
        if len(tokens) >= 5 and all(t == tokens[0] for t in tokens):
            try:
                int(tokens[0])
                break
            except ValueError:
                pass
        try:
            vals.extend(float(v) for v in tokens)
        except ValueError:
            pass
    return vals


def _parse_sao_simple(filename: str) -> Optional[pd.DataFrame]:
    """
    Fallback SAO 2.0 parser; tries three strategies in order:

    1. Section 63  — electron-density block (heights + Ne or heights + fp)
    2. Sections 55/56 or 3/4 — paired height + plasma-frequency blocks
    3. α-Chapman from foF2/hmF2 extracted anywhere in the header

    Returns a DataFrame with columns
        true_height (km), plasma_frequency (MHz), electron_density (m⁻³)
    or None if every strategy fails.
    """
    try:
        with open(filename, 'r', errors='replace') as fh:
            lines = fh.readlines()
    except OSError:
        return None

    # ── Strategy 1: Section 63 ────────────────────────────────────────────
    s63 = _find_section_start(lines, 63)
    if s63 >= 0:
        vals = _read_section_values(lines, s63)
        if len(vals) >= 6:
            # Try "N, [heights…], [Ne…]" interpretation first
            first = vals[0]
            n = int(first) if (first == int(first) and 1 < int(first) < 500) else None
            if n and len(vals) >= 1 + 2 * n:
                h_arr  = np.array(vals[1:n + 1])
                ne_raw = np.array(vals[n + 1: 2 * n + 1])
            else:
                mid    = len(vals) // 2
                h_arr  = np.array(vals[:mid])
                ne_raw = np.array(vals[mid:])

            # Auto-detect units: GIRO typically stores Ne in 10^10 m^-3
            ne_m3 = ne_raw * 1e10 if ne_raw.max() < 5000 else ne_raw
            fp    = _fp_from_ne(ne_m3)
            valid = (h_arr > 60) & (h_arr < 2000) & (ne_m3 > 0)
            if valid.sum() >= 4:
                return (pd.DataFrame({
                    'true_height':      h_arr[valid],
                    'plasma_frequency': fp[valid],
                    'electron_density': ne_m3[valid],
                }).sort_values('true_height').reset_index(drop=True))

    # ── Strategy 2: Paired height + plasma-frequency sections ─────────────
    for h_sec, fp_sec in [(55, 56), (3, 4)]:
        sh = _find_section_start(lines, h_sec)
        sf = _find_section_start(lines, fp_sec)
        if sh < 0 or sf < 0:
            continue
        h_vals  = _read_section_values(lines, sh)
        fp_vals = _read_section_values(lines, sf)
        if not h_vals or not fp_vals:
            continue
        n      = min(len(h_vals), len(fp_vals))
        h_arr  = np.array(h_vals[:n])
        fp_arr = np.array(fp_vals[:n])
        ne_m3  = _ne_from_fp(fp_arr)
        valid  = (h_arr > 60) & (h_arr < 2000) & (fp_arr > 0)
        if valid.sum() >= 4:
            return (pd.DataFrame({
                'true_height':      h_arr[valid],
                'plasma_frequency': fp_arr[valid],
                'electron_density': ne_m3[valid],
            }).sort_values('true_height').reset_index(drop=True))

    # ── Strategy 3: Chapman profile from header parameters ────────────────
    header_text = ''.join(lines[:50])
    fo_f2 = hm_f2 = None
    m = re.search(r'foF2[=:\s]+([0-9]+\.[0-9]+)', header_text, re.IGNORECASE)
    if m:
        fo_f2 = float(m.group(1))
    m = re.search(r'hmF2[=:\s]+([0-9]+\.?[0-9]*)', header_text, re.IGNORECASE)
    if m:
        hm_f2 = float(m.group(1))

    if fo_f2 and hm_f2 and fo_f2 > 0 and hm_f2 > 80:
        print(f"  SAO parser: Chapman fallback (foF2={fo_f2:.2f} MHz, hmF2={hm_f2:.1f} km)")
        return _chapman_profile(fo_f2, hm_f2)

    return None


def _chapman_profile(
    fo_f2: float, hm_f2: float,
    scale_km: float = 60.0, n_pts: int = 80,
) -> pd.DataFrame:
    """α-Chapman profile from foF2 (MHz) and hmF2 (km)."""
    nm_f2 = _ne_from_fp(np.array([fo_f2]))[0]
    h     = np.linspace(80.0, hm_f2 + 250.0, n_pts)
    z     = (h - hm_f2) / scale_km
    ne    = nm_f2 * np.exp(0.5 * (1.0 - z - np.exp(-z)))
    fp    = _fp_from_ne(ne)
    return pd.DataFrame({'true_height': h, 'plasma_frequency': fp,
                         'electron_density': ne})


def get_edp_from_sao(filename: str) -> Optional[pd.DataFrame]:
    """
    Extract the electron density profile from a downloaded .SAO file.

    Tries pynasonde first (if installed), then falls back to the built-in
    multi-strategy parser.

    Returns
    -------
    DataFrame with columns true_height (km), plasma_frequency (MHz),
    electron_density (m⁻³), or None on complete failure.
    """
    try:
        from pynasonde.digisonde import parse_sao as _pns_parse
        sao_obj = _pns_parse(filename)
        df = sao_obj.get_profile_dataframe()
        if df is not None and len(df) >= 4:
            return df[['true_height', 'plasma_frequency', 'electron_density']].copy()
    except Exception:
        pass

    return _parse_sao_simple(filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ionosonde_comparison(
    alt_grid: np.ndarray,
    closest_post: np.ndarray,          # (n_height,) – posterior at closest pt
    closest_prior: np.ndarray,         # (n_height,) – prior mean at closest pt
    all_post: np.ndarray,              # (n_height, n_geo) – full posterior
    prior_ensemble_at_pt: np.ndarray,  # (n_height, n_sample) – prior samples at closest pt
    sao_df: pd.DataFrame,
    station_code: str,
    station_lat: float,
    station_lon: float,
    dist_km: float,
    profile_dt: datetime,
    save_path: str,
    abel_ne: Optional[np.ndarray] = None,
    abel_alt_km: Optional[np.ndarray] = None,
    is_chapman_fallback: bool = False,
) -> None:
    """
    Two-panel figure:
      • Top  – 1-D comparison: ionosonde vs. tomographic profiles at the
               closest grid point (+ Abel if available).
      • Bottom – spaghetti: prior ensemble at the same point overlaid with
                 all posterior geo-points, highlighting the closest one.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))

    fig = plt.figure(figsize=(10, 12))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    sao_label = ("Ionosonde (α-Chapman approx.)" if is_chapman_fallback
                 else "Ionosonde (True-Height EDP)")

    # ── Interpolate SAO onto the alt_grid for RMSE ────────────────────────
    from scipy.interpolate import interp1d
    sao_h  = sao_df['true_height'].values
    sao_ne = sao_df['electron_density'].values
    valid  = np.isfinite(sao_h) & np.isfinite(sao_ne) & (sao_ne > 0)
    sao_on_grid: Optional[np.ndarray] = None
    if valid.sum() >= 3:
        interp = interp1d(sao_h[valid], sao_ne[valid],
                          bounds_error=False, fill_value=np.nan)
        sao_on_grid = interp(alt_grid)

    # ── Panel 1: 1-D comparison ───────────────────────────────────────────
    ax1.plot(closest_prior, alt_grid,
             color='tab:red',   lw=1.8, ls='--', alpha=0.85, label='Prior (closest pt)')
    ax1.plot(closest_post,  alt_grid,
             color='tab:blue',  lw=2.2,           label='Posterior KF (closest pt)')
    ax1.plot(sao_ne, sao_h,
             color='tab:green', lw=2.5, marker='o', markersize=3,
             markevery=5, label=sao_label, zorder=5)

    if abel_ne is not None and abel_alt_km is not None:
        ax1.plot(abel_ne, abel_alt_km,
                 color='dimgray', lw=1.8, ls=':', label='Abel inversion')

    # NmF2 / hmF2 markers
    for ne_arr, h_arr, col in [
        (closest_post, alt_grid, 'tab:blue'),
        (sao_ne,       sao_h,    'tab:green'),
    ]:
        nm, hm = _extract_robust_f2_peak(ne_arr, h_arr)
        if np.isfinite(nm):
            ax1.plot(nm, hm, marker='*', markersize=12,
                     color=col, markeredgecolor='black', zorder=6)

    ax1.set_xlabel("Electron Density (m⁻³)", fontsize=11)
    ax1.set_ylabel("Altitude (km)",           fontsize=11)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True, alpha=0.35, linestyle=':')
    ax1.legend(loc='upper right', fontsize=9)

    source_note = "(Chapman approx.)" if is_chapman_fallback else "(True-Height)"
    ax1.set_title(
        f"Ionosonde vs. Tomographic EDP  {source_note}\n"
        f"Station: {station_code}  ({station_lat:.1f}°N, {station_lon:.1f}°E)  "
        f"Δ {dist_km:.0f} km  |  {profile_dt.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=10,
    )

    # ── Panel 2: Spaghetti ────────────────────────────────────────────────
    # Prior ensemble at closest point (faint gray)
    ax2.plot(prior_ensemble_at_pt, alt_grid,
             color='lightgray', lw=0.6, alpha=0.5)

    # All posterior geo-points (faint blue)
    ax2.plot(all_post, alt_grid,
             color='tab:blue', lw=0.7, alpha=0.15)

    # Posterior at closest point highlighted
    ax2.plot(closest_post, alt_grid,
             color='darkblue', lw=2.2, label='Posterior (closest pt)')

    # Ionosonde overlaid
    ax2.plot(sao_ne, sao_h,
             color='tab:green', lw=2.5, marker='o', markersize=3,
             markevery=5, label=sao_label, zorder=5)

    ax2.set_xlabel("Electron Density (m⁻³)", fontsize=11)
    ax2.set_ylabel("Altitude (km)",           fontsize=11)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.grid(True, alpha=0.35, linestyle=':')
    ax2.set_title("Prior Ensemble & Full Posterior Coverage", fontsize=10)

    custom_lines = [
        Line2D([0], [0], color='lightgray', lw=1.5, alpha=0.7),
        Line2D([0], [0], color='tab:blue',  lw=1.0, alpha=0.4),
        Line2D([0], [0], color='darkblue',  lw=2.2),
        Line2D([0], [0], color='tab:green', lw=2.5),
    ]
    ax2.legend(custom_lines,
               ['Prior samples (closest pt)', 'Posterior (all pts)',
                'Posterior (closest pt)', sao_label],
               loc='upper right', fontsize=9)

    # Shared y-limits
    all_alts = np.concatenate([alt_grid, sao_h[valid]])
    ax1.set_ylim(max(0, all_alts.min() - 20), all_alts.max() + 30)

    # X-limit driven by data range
    all_ne = [v for v in [closest_post, closest_prior, sao_ne] if v is not None]
    if all_ne:
        finite_ne = np.concatenate([a[np.isfinite(a)] for a in all_ne])
        finite_ne = finite_ne[finite_ne > 0]
        if finite_ne.size:
            ax1.set_xlim(left=-0.05 * finite_ne.max(), right=1.15 * finite_ne.max())

    fig.suptitle(
        f"Ionosonde Collocation Verification  —  {profile_dt.strftime('%Y %b %d %H:%M')} UTC",
        fontsize=12, fontweight='bold', y=0.995,
    )

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Ionosonde] Saved comparison plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_ionosonde_verification(
    profile_dt: datetime,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    alt_grid: np.ndarray,
    posterior_edp: Optional[np.ndarray],   # (n_height, n_geo)
    prior_edp: Optional[np.ndarray],       # (n_height, n_geo)
    geolocation: np.ndarray,               # (n_geo, 2) = (lon, lat)
    prior_ensemble: Optional[np.ndarray] = None,  # (n_height, n_geo, n_sample)
    abel_ne: Optional[np.ndarray] = None,
    abel_alt_km: Optional[np.ndarray] = None,
    save_dir: str = "./Figures/Section20_Batch/Ionosonde/",
    filename_prefix: str = "ionosonde",
    window_minutes: int = 30,
    download_dir: str = "./Ionosonde_verification/downloads/",
    generate_plot: bool = True,
) -> dict:
    """
    Full ionosonde collocation pipeline:

    1. Find GIRO stations within the occultation bounding box.
    2. For each station (closest first), attempt to download an .SAO file
       within ±window_minutes.
    3. Parse the SAO file for the true-height EDP.  Falls back to the
       LGDC scaled-parameter API + Chapman profile if parsing fails.
    4. Locate the closest tomographic grid point to the ionosonde station.
    5. Compute RMSE between the ionosonde and posterior EDP on the shared
       altitude grid.
    6. Generate (and save) the two-panel comparison + spaghetti figure.

    Returns
    -------
    dict with keys prefixed ``Ionosonde_``:
        Ionosonde_Station, Ionosonde_Dist_km,
        Ionosonde_NmF2, Ionosonde_hmF2,
        Ionosonde_Post_RMSE, Ionosonde_Prior_RMSE,
        Ionosonde_Source   ("SAO" | "Chapman" | "None")
    """
    default = {
        'Ionosonde_Station':   'None',
        'Ionosonde_Dist_km':   np.nan,
        'Ionosonde_NmF2':      np.nan,
        'Ionosonde_hmF2':      np.nan,
        'Ionosonde_Post_RMSE': np.nan,
        'Ionosonde_Prior_RMSE': np.nan,
        'Ionosonde_Source':    'None',
    }

    try:
        # ── 0. Fast exit when the data service is known to be down ─────────
        if _LGDC_HEALTHY is False:
            return default

        # ── 1. Find candidate stations ─────────────────────────────────────
        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = (0.5 * (lon_min + lon_max) if lon_min <= lon_max
                      else ((lon_min + lon_max + 360.0) / 2.0 % 360.0) - 180.0)

        candidates = find_stations_in_region(
            lat_min, lat_max, lon_min, lon_max, center_lat, center_lon
        )
        print(f"  [Ionosonde] Stations in region: "
              f"{[c[0] for c in candidates] or 'None'}")

        if not candidates:
            return default

        # ── 2. Download .SAO file (try stations in order of proximity) ─────
        sao_file: Optional[str] = None
        chosen_code = chosen_lat = chosen_lon = None
        day_key_prefix = profile_dt.strftime('%Y-%m-%d')

        for code, slat, slon, _ in candidates:
            # Respect the day-level pre-screen: skip stations confirmed empty
            if _DAY_AVAILABILITY.get(f"{code}_{day_key_prefix}") is False:
                print(f"  [{code}] Skipped — pre-screen: no data on {day_key_prefix}")
                continue

            sao_file = fetch_sao_file(
                code, profile_dt,
                window_minutes=window_minutes,
                download_dir=download_dir,
            )
            if sao_file:
                chosen_code, chosen_lat, chosen_lon = code, slat, slon
                break

        # ── 3. Parse SAO  (or fall back to scaled params) ──────────────────
        sao_df: Optional[pd.DataFrame] = None
        is_chapman = False

        if sao_file:
            sao_df = get_edp_from_sao(sao_file)

        if sao_df is None and chosen_code is None:
            # Try the closest station anyway via scaled params
            code, slat, slon, _ = candidates[0]
            chosen_code, chosen_lat, chosen_lon = code, slat, slon

        if sao_df is None and chosen_code:
            print(f"  [Ionosonde] SAO parse failed; trying scaled params "
                  f"for {chosen_code}")
            scaled = fetch_scaled_params(
                chosen_code, profile_dt, window_minutes=window_minutes
            )
            fo_f2 = (scaled or {}).get('foF2', None)
            hm_f2 = (scaled or {}).get('hmF2', None)
            if fo_f2 and hm_f2 and float(fo_f2) > 0 and float(hm_f2) > 80:
                sao_df     = _chapman_profile(float(fo_f2), float(hm_f2))
                is_chapman = True

        if sao_df is None or len(sao_df) < 4:
            print("  [Ionosonde] No usable ionogram data found.")
            return default

        # ── 4. Closest tomographic grid point to the ionosonde station ─────
        # geolocation shape: (n_geo, 2) = (lon, lat)
        geo_lon = geolocation[:, 0]
        geo_lat = geolocation[:, 1]
        dists   = np.sqrt(
            (geo_lat - chosen_lat) ** 2
            + ((geo_lon - chosen_lon) * np.cos(np.radians(chosen_lat))) ** 2
        )
        closest_idx = int(np.argmin(dists))
        dist_km     = _haversine_km(chosen_lat, chosen_lon,
                                    float(geo_lat[closest_idx]),
                                    float(geo_lon[closest_idx]))

        # Select the best available EDP for the comparison
        best_post  = posterior_edp[:, closest_idx] if posterior_edp is not None else None
        best_prior = prior_edp[:,   closest_idx] if prior_edp    is not None else None

        if best_post is None and best_prior is None:
            print("  [Ionosonde] No tomographic EDP available for comparison.")
            return default

        tomo_profile = best_post if best_post is not None else best_prior

        # ── 5. Statistics ──────────────────────────────────────────────────
        from scipy.interpolate import interp1d
        sao_h  = sao_df['true_height'].values
        sao_ne = sao_df['electron_density'].values
        valid  = np.isfinite(sao_h) & np.isfinite(sao_ne) & (sao_ne > 0)
        sao_on_grid: Optional[np.ndarray] = None
        if valid.sum() >= 3:
            interp = interp1d(sao_h[valid], sao_ne[valid],
                              bounds_error=False, fill_value=np.nan)
            sao_on_grid = interp(alt_grid)

        result = dict(default)
        result['Ionosonde_Station'] = chosen_code
        result['Ionosonde_Dist_km'] = round(dist_km, 1)
        result['Ionosonde_Source']  = 'Chapman' if is_chapman else 'SAO'

        iono_nm, iono_hm = _extract_robust_f2_peak(sao_ne, sao_h)
        result['Ionosonde_NmF2'] = iono_nm
        result['Ionosonde_hmF2'] = iono_hm

        if sao_on_grid is not None:
            if best_post is not None:
                ok = np.isfinite(best_post) & np.isfinite(sao_on_grid)
                if ok.sum() >= 5:
                    result['Ionosonde_Post_RMSE'] = float(
                        np.sqrt(np.mean((best_post[ok] - sao_on_grid[ok]) ** 2))
                    )
            if best_prior is not None:
                ok = np.isfinite(best_prior) & np.isfinite(sao_on_grid)
                if ok.sum() >= 5:
                    result['Ionosonde_Prior_RMSE'] = float(
                        np.sqrt(np.mean((best_prior[ok] - sao_on_grid[ok]) ** 2))
                    )

        print(
            f"  [Ionosonde] Station={chosen_code}  Δ={dist_km:.0f} km  "
            f"NmF2={iono_nm:.2e}  hmF2={iono_hm:.1f} km  "
            f"Post RMSE={result['Ionosonde_Post_RMSE']:.2e}"
        )

        # ── 6. Plot ────────────────────────────────────────────────────────
        if generate_plot and best_prior is not None:
            # Prior ensemble at closest grid point
            if prior_ensemble is not None:
                ens_at_pt = prior_ensemble[:, closest_idx, :]   # (n_h, n_sample)
            else:
                ens_at_pt = best_prior[:, np.newaxis]

            # All posterior geo-points
            all_post = posterior_edp if posterior_edp is not None else prior_edp

            save_path = os.path.join(
                save_dir, f"{filename_prefix}_{chosen_code}_ionosonde.png"
            )
            _plot_ionosonde_comparison(
                alt_grid             = alt_grid,
                closest_post         = tomo_profile,
                closest_prior        = best_prior,
                all_post             = all_post,
                prior_ensemble_at_pt = ens_at_pt,
                sao_df               = sao_df,
                station_code         = chosen_code,
                station_lat          = chosen_lat,
                station_lon          = chosen_lon,
                dist_km              = dist_km,
                profile_dt           = profile_dt,
                save_path            = save_path,
                abel_ne              = abel_ne,
                abel_alt_km          = abel_alt_km,
                is_chapman_fallback  = is_chapman,
            )

        return result

    except Exception as exc:
        print(f"  [Ionosonde] Verification failed: {exc}")
        return default
