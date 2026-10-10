"""
demo_esr_isr.py

Processes ISR data from three instruments from Data/ISR_Data/ and downloads
co-located GNSS RINEX data from nearby IGS stations.

Instruments
-----------
  ESR  — EISCAT Svalbard Radar (MAD6400_*.hdf5, kinst=95, 78.09°N 16.02°E)
  TRO  — EISCAT Tromsø UHF Radar (MAD6300_*.hdf5, kinst=72, 69.58°N 19.21°E)
  JRO  — Jicamarca Radio Observatory (jro*.hdf5, -11.95°N 283.13°E)

Steps
-----
1. Scan Data/ISR_Data/ for MAD6400, MAD6300, and jro HDF5 files.
2. Extract electron density profiles (EDPs) and cache to esr_edp_cache.pkl.
   Subsequent runs load from cache and only process new files.
3. Print a summary of days with ISR data.
4. Download RINEX obs + nav files from CDDIS for each day × station pair.

Usage
-----
    python demo_esr_isr.py                  # normal run (uses cache)
    python demo_esr_isr.py --force-reload   # re-process all ISR files
    python demo_esr_isr.py --no-rinex       # skip RINEX download
    python demo_esr_isr.py --list-days      # print days and exit

NASA Earthdata credentials in ~/.netrc are required for CDDIS downloads:
    machine urs.earthdata.nasa.gov login <user> password <pass>
"""
from __future__ import annotations

import argparse
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from TEC_model.igs_tec_pipeline import RinexDownloader

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
ISR_DIR    = ROOT / "Data" / "ISR_Data"
RINEX_CACHE = ROOT / "Data" / "RINEX_Cache"
CACHE_FILE = ISR_DIR / "esr_edp_cache.pkl"

# ── IGS stations near ESR (Longyearbyen, 78°N 16°E) ──────────────────────────
# 4-char codes used by RinexDownloader; full 9-char RINEX-3 name in comment.
IGS_STATIONS = ["TRO1", "WUTH", "NYA1"]   # TRO100NOR, WUTH00NOR, NYA100NOR
RINEX_VERSION = 3

# Minimum valid altitude gates per profile to keep it.
MIN_GATES = 5


# ─────────────────────────────────────────────────────────────────────────────
# ISR file loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_mad6400_file(fpath: Path) -> list[dict]:
    """
    Load a single MAD6400 HDF5 file and return a list of EDP dicts.

    Handles two on-disk layouts produced by the Madrigal download service:

    netCDF4 flat layout (--format=netCDF4 single-file downloads):
        Root attributes hold kindat_code, instrument_latitude, etc.
        Top-level datasets: timestamps (n_t,), gdalt/ne/… (n_t, n_gates).

    Classic Madrigal HDF5 layout (bulk globalDownload.py downloads):
        No root attributes; kindat/lat/lon in Metadata/Experiment Parameters.
        Data/Table Layout: structured record array, one row per (time, alt).

    Each returned dict has: time, lat, lon, alt_site_km, alt_km, ne_m3,
    dne_m3, ti_K, tr, source_file, kindat.

    kindat is always "6400" here ("GUISDAP params" -- full spectral fit:
    corrected ne/ti/tr, not the Te/Ti=1 power-profile approximation used by
    MAD6300/6301/6302). See _load_mad6300_file for the other kind.
    """
    # Madrigal fill value used in the classic layout (not NaN).
    MAD_FILL = 1.0e6

    edps: list[dict] = []
    with h5py.File(fpath, "r") as f:

        # ── Detect layout ─────────────────────────────────────────────────────
        is_classic = "Data" in f and "Table Layout" in f["Data"]

        if is_classic:
            exp = dict(zip(
                [r[0].decode() for r in f["Metadata/Experiment Parameters"][:]],
                [r[1].decode() for r in f["Metadata/Experiment Parameters"][:]],
            ))
            if exp.get("kindat code(s)", "").strip() != "6400":
                return []
            lat      = float(exp.get("instrument latitude",  "0"))
            lon      = float(exp.get("instrument longitude", "0"))
            alt_site = float(exp.get("instrument altitude",  "0"))

            tbl = f["Data/Table Layout"][:]
            for ts in np.unique(tbl["ut1_unix"]):
                rows  = tbl[tbl["ut1_unix"] == ts]
                alt_i = rows["gdalt"].astype(float)
                ne_i  = rows["ne"].astype(float)
                dne_i = rows["dne"].astype(float)
                ti_i  = rows["ti"].astype(float)
                tr_i  = rows["tr"].astype(float)

                valid = (np.isfinite(alt_i) & np.isfinite(ne_i)
                         & (ne_i > MAD_FILL) & (ne_i > 0))
                if valid.sum() < MIN_GATES:
                    continue

                idx = np.argsort(alt_i[valid])
                edps.append({
                    "time":        datetime.fromtimestamp(ts, tz=timezone.utc),
                    "lat":         lat,
                    "lon":         lon,
                    "alt_site_km": alt_site,
                    "alt_km":      alt_i[valid][idx].copy(),
                    "ne_m3":       ne_i[valid][idx].copy(),
                    "dne_m3":      dne_i[valid][idx].copy(),
                    "ti_K":        ti_i[valid][idx].copy(),
                    "tr":          tr_i[valid][idx].copy(),
                    "source_file": fpath.name,
                    "kindat":      "6400",
                })

        else:
            # netCDF4 flat layout
            kindat = f.attrs.get("kindat_code", b"").decode().strip()
            if kindat != "6400":
                return []

            lat      = float(f.attrs.get("instrument_latitude",  b"0").decode())
            lon      = float(f.attrs.get("instrument_longitude", b"0").decode())
            alt_site = float(f.attrs.get("instrument_altitude",  b"0").decode())

            timestamps = f["timestamps"][:]
            gdalt      = f["gdalt"][:]
            ne         = f["ne"][:]
            dne        = f["dne"][:]
            ti  = f["ti"][:] if "ti" in f else np.full_like(ne, np.nan)
            tr  = f["tr"][:] if "tr" in f else np.full_like(ne, np.nan)

            for i, ts in enumerate(timestamps):
                alt_i = gdalt[i]
                ne_i  = ne[i]
                dne_i = dne[i]
                ti_i  = ti[i]
                tr_i  = tr[i]

                valid = np.isfinite(alt_i) & np.isfinite(ne_i) & (ne_i > 0)
                if valid.sum() < MIN_GATES:
                    continue

                idx = np.argsort(alt_i[valid])
                edps.append({
                    "time":        datetime.fromtimestamp(ts, tz=timezone.utc),
                    "lat":         lat,
                    "lon":         lon,
                    "alt_site_km": alt_site,
                    "alt_km":      alt_i[valid][idx].copy(),
                    "ne_m3":       ne_i[valid][idx].copy(),
                    "dne_m3":      dne_i[valid][idx].copy(),
                    "ti_K":        ti_i[valid][idx].copy(),
                    "tr":          tr_i[valid][idx].copy(),
                    "source_file": fpath.name,
                    "kindat":      "6400",
                })

    return edps


def _load_mad6300_file(fpath: Path) -> list[dict]:
    """
    Load a single MAD6300 (EISCAT Tromsø UHF) HDF5 file.

    Same classic Madrigal HDF5 layout as MAD6400 but with different fields:
      - Electron density:  'pop'  (uncorrected, Te/Ti=1 assumption)  [m⁻³]
      - Density error:     'dpop'                                     [m⁻³]
      - No ti/tr fields   (set to NaN)
      - Altitude derived: alt_km = range_km * sin(elm_deg)
                          (accurate to <1 km for elm ≈ 78°, range < 800 km)

    Gates below 60 km are dropped (likely near-range clutter).

    kindat is always "6300" ("GUISDAP pp resolution 0" -- quick-look power
    profile, distinct from MAD6400's fitted params; see _load_mad6400_file).
    """
    MAD_FILL   = 1.0e6
    ALT_MIN_KM = 60.0

    edps: list[dict] = []
    with h5py.File(fpath, "r") as f:
        if "Data" not in f or "Table Layout" not in f["Data"]:
            return []

        exp = dict(zip(
            [r[0].decode() for r in f["Metadata/Experiment Parameters"][:]],
            [r[1].decode() for r in f["Metadata/Experiment Parameters"][:]],
        ))

        # Accept only Tromsø UHF (kinst 72)
        if exp.get("instrument code(s)", "").strip() != "72":
            return []

        lat      = float(exp.get("instrument latitude",  "0"))
        lon      = float(exp.get("instrument longitude", "0"))
        alt_site = float(exp.get("instrument altitude",  "0"))

        tbl = f["Data/Table Layout"][:]
        if "pop" not in tbl.dtype.names:
            return []

        for ts in np.unique(tbl["ut1_unix"]):
            rows = tbl[tbl["ut1_unix"] == ts]

            range_km = rows["range"].astype(float)
            elm_deg  = rows["elm"].astype(float)
            alt_km   = range_km * np.sin(np.deg2rad(elm_deg))

            ne_i  = rows["pop"].astype(float)
            dne_i = (rows["dpop"].astype(float)
                     if "dpop" in rows.dtype.names
                     else np.full(len(rows), np.nan))

            valid = (np.isfinite(alt_km) & np.isfinite(ne_i)
                     & (ne_i > MAD_FILL) & (alt_km >= ALT_MIN_KM))
            if valid.sum() < MIN_GATES:
                continue

            idx = np.argsort(alt_km[valid])
            n   = int(valid.sum())
            edps.append({
                "time":        datetime.fromtimestamp(ts, tz=timezone.utc),
                "lat":         lat,
                "lon":         lon,
                "alt_site_km": alt_site,
                "alt_km":      alt_km[valid][idx].copy(),
                "ne_m3":       ne_i[valid][idx].copy(),
                "dne_m3":      dne_i[valid][idx].copy(),
                "ti_K":        np.full(n, np.nan),
                "tr":          np.full(n, np.nan),
                "source_file": fpath.name,
                "kindat":      "6300",
            })

    return edps


def _load_jro_file(fpath: Path) -> list[dict]:
    """
    Load a Jicamarca IS Radar HDF5 file and return a list of EDP dicts.

    Layout differs from MAD6400:
        gdalt      – fixed 1-D altitude grid (n_alt,)  km
        ne         – (n_alt, n_times)  m⁻³   (transposed vs ESR)
        timestamps – (n_times,)  Unix seconds
        gdlatr/gdlonr – per-time lat/lon (constant for JRO)

    Drifts files (kindat 1910, no 'ne' dataset) are silently skipped.
    """
    edps: list[dict] = []
    with h5py.File(fpath, "r") as f:
        al = f["Data/Array Layout"]

        # Skip files that don't carry electron density (e.g. drifts)
        params_2d = al["2D Parameters"]
        if "ne" not in params_2d:
            return []

        exp = dict(zip(
            [r[0].decode() for r in f["Metadata/Experiment Parameters"][:]],
            [r[1].decode() for r in f["Metadata/Experiment Parameters"][:]],
        ))
        lat      = float(exp.get("instrument latitude", "0"))
        lon      = float(exp.get("instrument longitude", "0"))
        alt_site = float(exp.get("instrument altitude", "0"))

        gdalt      = al["gdalt"][:]                  # (n_alt,)  km
        timestamps = al["timestamps"][:]             # (n_times,)
        ne         = params_2d["ne"][:]              # (n_alt, n_times)
        dne        = params_2d["dne"][:]
        ti  = params_2d["ti"][:] if "ti" in params_2d else np.full_like(ne, np.nan)
        te  = params_2d["te"][:] if "te" in params_2d else np.full_like(ne, np.nan)
        tr  = np.where(np.isfinite(ti) & (ti > 0), te / ti, np.nan)

        for i, ts in enumerate(timestamps):
            ne_i  = ne[:, i]
            dne_i = dne[:, i]
            ti_i  = ti[:, i]
            tr_i  = tr[:, i]

            valid = np.isfinite(gdalt) & np.isfinite(ne_i) & (ne_i > 0)
            if valid.sum() < MIN_GATES:
                continue

            idx = np.argsort(gdalt[valid])
            edps.append({
                "time":        datetime.fromtimestamp(ts, tz=timezone.utc),
                "lat":         lat,
                "lon":         lon,
                "alt_site_km": alt_site,
                "alt_km":      gdalt[valid][idx].copy(),
                "ne_m3":       ne_i[valid][idx].copy(),
                "dne_m3":      dne_i[valid][idx].copy(),
                "ti_K":        ti_i[valid][idx].copy(),
                "tr":          tr_i[valid][idx].copy(),
                "source_file": fpath.name,
                "kindat":      "jro",
            })
    return edps


def _candidate_files(isr_dir: Path) -> list[Path]:
    """Return all EDP HDF5 files (MAD6400 ESR + MAD6300 Tromsø + JRO non-drifts)."""
    def _no_seq(files):
        """Drop sequentially-numbered duplicates (e.g. MAD6400_..._1.hdf5)."""
        return [f for f in files if not re.search(r"_\d+\.hdf5$", f.name)]

    mad6400 = _no_seq(sorted(isr_dir.glob("MAD6400_*.hdf5")))
    mad6300 = _no_seq(sorted(isr_dir.glob("MAD6300_*.hdf5")))
    jro_files = [f for f in sorted(isr_dir.glob("jro*.hdf5"))
                 if "drifts" not in f.name]
    return mad6400 + mad6300 + jro_files


# ─────────────────────────────────────────────────────────────────────────────
# Cache-aware EDP loader
# ─────────────────────────────────────────────────────────────────────────────

def load_edps(isr_dir: Path = ISR_DIR,
              cache_file: Path = CACHE_FILE,
              force: bool = False) -> list[dict]:
    """
    Load electron density profiles from all MAD6400 files, using a pickle
    cache so only new files are re-processed on subsequent runs.

    Parameters
    ----------
    force : if True, ignore cache and re-process every file.

    Returns
    -------
    List of EDP dicts sorted by time.
    """
    candidates = _candidate_files(isr_dir)
    if not candidates:
        print(f"[ISR] No ISR files found in {isr_dir}")
        return []

    # ── Load cache ────────────────────────────────────────────────────────────
    cached_edps: list[dict] = []
    cached_files: set[str] = set()

    if cache_file.exists() and not force:
        print(f"[ISR] Loading cache {cache_file.name} … ", end="", flush=True)
        with open(cache_file, "rb") as fh:
            store = pickle.load(fh)
        cached_edps  = store.get("edps", [])
        cached_files = {e["source_file"] for e in cached_edps}
        print(f"{len(cached_edps)} profiles from {len(cached_files)} files")

    # ── Process only new files ────────────────────────────────────────────────
    new_files = [f for f in candidates if f.name not in cached_files]
    if new_files:
        print(f"[ISR] Processing {len(new_files)} new file(s) …")
    else:
        print("[ISR] Cache is up to date.")

    new_edps: list[dict] = []
    for fpath in new_files:
        if fpath.name.startswith("jro"):
            loader = _load_jro_file
        elif fpath.name.startswith("MAD6300"):
            loader = _load_mad6300_file
        else:
            loader = _load_mad6400_file
        profiles = loader(fpath)
        new_edps.extend(profiles)
        print(f"  {fpath.name}: {len(profiles)} profiles")

    all_edps = cached_edps + new_edps
    all_edps.sort(key=lambda e: e["time"])

    # ── Update cache if anything changed ─────────────────────────────────────
    if new_edps or force:
        with open(cache_file, "wb") as fh:
            pickle.dump({"edps": all_edps}, fh, protocol=4)
        print(f"[ISR] Cache updated → {cache_file} ({len(all_edps)} total profiles)")

    return all_edps


# ─────────────────────────────────────────────────────────────────────────────
# Day listing
# ─────────────────────────────────────────────────────────────────────────────

def isr_days(edps: list[dict]) -> list:
    """Return sorted list of date objects that have at least one EDP."""
    return sorted({e["time"].date() for e in edps})


def print_day_summary(edps: list[dict]) -> None:
    days = isr_days(edps)
    print(f"\n[ISR] {len(days)} day(s) with ISR data:")
    for d in days:
        count = sum(1 for e in edps if e["time"].date() == d)
        alt_range = ""
        day_alts = np.concatenate([e["alt_km"] for e in edps if e["time"].date() == d])
        alt_range = f"  alt {day_alts.min():.0f}–{day_alts.max():.0f} km"
        print(f"  {d}   {count:4d} profiles{alt_range}")


# ─────────────────────────────────────────────────────────────────────────────
# RINEX download
# ─────────────────────────────────────────────────────────────────────────────

def download_rinex(days: list,
                   stations: list[str] = IGS_STATIONS,
                   rinex_cache: Path = RINEX_CACHE,
                   rinex_version: int = RINEX_VERSION) -> dict:
    """
    Download RINEX obs + nav files from CDDIS for each (station, day).

    Returns a dict keyed by (station_4char, date) → {'obs': Path, 'nav': Path}.
    Missing downloads are omitted from the result (a warning is printed).
    """
    dl = RinexDownloader(cache_dir=str(rinex_cache))
    results: dict = {}

    for day in days:
        dt = datetime(day.year, day.month, day.day)
        print(f"\n── {day} ──")

        # Shared BRDM mixed nav (one per day, same for every station)
        nav_shared = None
        try:
            nav_shared = dl.nav_file("BRDM", dt, rinex_version)
            print(f"  Nav (BRDM): {nav_shared.name}")
        except Exception as exc:
            print(f"  [warn] BRDM nav unavailable: {exc}")

        for sta in stations:
            try:
                obs_path = dl.obs_file(sta, dt, rinex_version)
                nav_path = nav_shared or dl.nav_file(sta, dt, rinex_version)
                results[(sta, day)] = {"obs": obs_path, "nav": nav_path}
                print(f"  {sta}: {obs_path.name}")
            except Exception as exc:
                print(f"  {sta}: SKIP — {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process ESR ISR EDPs and download co-located RINEX data.")
    parser.add_argument("--force-reload", action="store_true",
                        help="Re-process all HDF5 files, ignoring cache.")
    parser.add_argument("--no-rinex", action="store_true",
                        help="Skip RINEX download step.")
    parser.add_argument("--list-days", action="store_true",
                        help="Print day summary and exit without downloading RINEX.")
    args = parser.parse_args()

    # ── Step 1: load / cache EDPs ─────────────────────────────────────────────
    edps = load_edps(force=args.force_reload)
    if not edps:
        print("[ISR] No profiles found — check Data/ISR_Data/ for MAD6400_*.hdf5 files.")
        return

    # ── Step 2: day summary ───────────────────────────────────────────────────
    print_day_summary(edps)
    if args.list_days:
        return

    # ── Step 3: RINEX download ────────────────────────────────────────────────
    if not args.no_rinex:
        days = isr_days(edps)
        print(f"\n[RINEX] Downloading for {len(IGS_STATIONS)} stations × {len(days)} days …")
        results = download_rinex(days)
        n = len(results)
        total = len(days) * len(IGS_STATIONS)
        print(f"\n[RINEX] Done: {n}/{total} (station, day) pairs acquired.")
    else:
        print("\n[RINEX] Skipped (--no-rinex).")


if __name__ == "__main__":
    main()
