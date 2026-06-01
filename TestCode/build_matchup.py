"""
Build a matchup dataset linking COSMIC-2 atmPrf files with:
  1. Corresponding echPrf files  (direct name substitution: atmPrf_ -> echPrf_)
  2. EarthCare granule files whose start time is within TIME_WINDOW_MINUTES of the RO event

Directory structure:
  /Volumes/COSMIC_RO/atmPrf/{year}/{year.doy}/atmPrf_*
  /Volumes/COSMIC_RO/echPrf/{year}/{year.doy}/echPrf_*
  /Volumes/EarthCare/{year}/{year.doy}/ECA_*.h5

atmPrf filename format:  atmPrf_{instrument}.{year}.{doy}.{HH}.{MM}.{GNSS_SV}_{suffix}
EarthCare filename format: ECA_..._2B_{YYYYMMDDTHHMMSSz}_{proc_date}_{orbit_granule}.h5
  (the second timestamp is the processing date, not the granule end; it is ignored)

Output: dict keyed by atmPrf filename:
  {
      'atmPrf_C2E1.2024.207.10.08.G18_0001.0001_nc': {
          'atm_path':   Path,
          'ro_time':    datetime (UTC),
          'ech_path':   Path | None,   # None if echPrf file not found on disk
          'eca_files':  [Path, ...],   # sorted by start time, empty if no match
      },
      ...
  }
"""

import re
import pickle
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ATM_ROOT = Path("/Volumes/COSMIC_RO/atmPrf")
ECH_ROOT = Path("/Volumes/COSMIC_RO/echPrf")
ECA_ROOT = Path("/Volumes/EarthCare")

TIME_WINDOW_MINUTES = 30   # EarthCare files within ±N minutes of the RO event

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
# atmPrf_C2E1.2024.207.10.08.G18_0001.0001_nc
_ATM_RE = re.compile(
    r"atmPrf_[A-Za-z0-9]+\.(\d{4})\.(\d{3})\.(\d{2})\.(\d{2})\."
)

# First YYYYMMDDTHHmmSS timestamp in the EarthCare filename
_ECA_RE = re.compile(r"_(\d{8}T\d{6})Z_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ro_time(filename):
    """Return datetime of the RO event encoded in an atmPrf filename, or None."""
    m = _ATM_RE.search(filename)
    if not m:
        return None
    year, doy, hour, minute = (int(m.group(i)) for i in range(1, 5))
    return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute)


def parse_eca_start(filename):
    """Return the granule start datetime from an EarthCare filename, or None."""
    m = _ECA_RE.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def ech_path_for(atm_path):
    """Derive the expected echPrf Path from an atmPrf Path."""
    rel = atm_path.relative_to(ATM_ROOT)
    ech_name = atm_path.name.replace("atmPrf_", "echPrf_", 1)
    return ECH_ROOT / rel.parent / ech_name


def eca_dir_for(dt):
    """Return the EarthCare day-directory for a given datetime."""
    doy = dt.timetuple().tm_yday
    return ECA_ROOT / str(dt.year) / f"{dt.year}.{doy:03d}"


def find_eca_matches(ro_time, window_minutes=TIME_WINDOW_MINUTES):
    """
    Return sorted list of EarthCare Paths whose start time is within
    ±window_minutes of ro_time.  Checks the adjacent days as well so
    events near midnight are handled correctly.
    """
    window = timedelta(minutes=window_minutes)
    matched = []
    checked = set()

    for offset in (-1, 0, 1):
        day_dir = eca_dir_for(ro_time + timedelta(days=offset))
        if day_dir in checked or not day_dir.is_dir():
            continue
        checked.add(day_dir)
        for f in day_dir.iterdir():
            start = parse_eca_start(f.name)
            if start is not None and abs(start - ro_time) <= window:
                matched.append(f)

    return sorted(matched, key=lambda f: parse_eca_start(f.name))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_matchup(time_window_minutes=TIME_WINDOW_MINUTES, output_pickle=None):
    """
    Scan all atmPrf files under ATM_ROOT and return a matchup dict.

    Parameters
    ----------
    time_window_minutes : int
        Half-width of the time window (in minutes) for EarthCare matching.
    output_pickle : Path or str or None
        If given, the result dict is pickled to this path.

    Returns
    -------
    dict  (see module docstring for structure)
    """
    atm_files = sorted(ATM_ROOT.rglob("atmPrf_*"))
    print(f"Found {len(atm_files)} atmPrf files")

    matchup = {}
    for atm_path in atm_files:
        ro_time = parse_ro_time(atm_path.name)
        if ro_time is None:
            print(f"  [WARN] cannot parse time: {atm_path.name}")
            continue

        ech = ech_path_for(atm_path)
        eca = find_eca_matches(ro_time, window_minutes=time_window_minutes)

        matchup[atm_path.name] = {
            "atm_path":  atm_path,
            "ro_time":   ro_time,
            "ech_path":  ech if ech.exists() else None,
            "eca_files": eca,
        }

    n_ech = sum(1 for v in matchup.values() if v["ech_path"] is not None)
    n_eca = sum(1 for v in matchup.values() if v["eca_files"])
    print(f"echPrf matched : {n_ech} / {len(matchup)}")
    print(f"EarthCare match: {n_eca} / {len(matchup)} (±{time_window_minutes} min)")

    if output_pickle is not None:
        out = Path(output_pickle)
        with open(out, "wb") as fh:
            pickle.dump(matchup, fh)
        print(f"Saved to {out}")

    return matchup


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    matchup = build_matchup(
        time_window_minutes=TIME_WINDOW_MINUTES,
        output_pickle="ro_eca_matchup.pkl",
    )

    # Quick sanity check: print first 3 entries
    for key, val in list(matchup.items())[:3]:
        print(f"\n{key}")
        print(f"  ro_time  : {val['ro_time']}")
        print(f"  ech_path : {val['ech_path']}")
        print(f"  eca_files: {[f.name for f in val['eca_files']]}")
