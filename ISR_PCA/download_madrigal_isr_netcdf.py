#!/usr/bin/env python3
"""
Robust Madrigal ISR downloader for:
  - Tromso ISR (TRO): instrument 72, kindat 6400
  - EISCAT Svalbard Radar (SVL): instrument 95, kindat 6400
  - Millstone Hill ISR (MIL): instrument 30, "Combined basic parameters file"

Downloads netCDF4 ONLY.

Behavior:
  * skips files already downloaded
  * retries temporary failures
  * if a file still fails after all retries, records it and continues
  * removes partial/empty files after failed downloads
  * writes failed_downloads.csv at the end

Requires:
    pip install madrigalweb
"""

from __future__ import annotations

import csv
import os
import socket
import time
import urllib.error
from pathlib import Path

import madrigalWeb.madrigalWeb


# ============================================================
# USER SETTINGS
# ============================================================

START = (2025, 1, 1, 0, 0, 0)
END   = (2025, 12, 31, 23, 59, 59)

USER_FULLNAME = "Pin-Hsuan Cheng"
USER_EMAIL = "pich8403@colorado.edu"
USER_AFFILIATION = "PlanetiQ"

BASE_DIR = Path("/home/pin/Desktop/tomography_project/Data/ISR_Data/")

# Number of attempts for each individual file.
MAX_ATTEMPTS = 3

# Wait between retry attempts, seconds.
RETRY_DELAYS = [10, 30, 60]

# Small pause after a successful download to avoid hammering the server.
SUCCESS_DELAY = 2


# ============================================================
# STATION CONFIGURATION
# ============================================================

STATIONS = {
    "TRO": {
        "name": "Tromso ISR",
        "urls": [
            "https://madrigal.eiscat.se/madrigal",
        ],
        "instrument": 72,
        "kindat": 6400,
        "selector": "kindat",
    },

    "SVL": {
        "name": "EISCAT Svalbard Radar",
        "urls": [
            "https://madrigal.eiscat.se/madrigal",
        ],
        "instrument": 95,
        "kindat": 6400,
        "selector": "kindat",
    },

    "MIL": {
        "name": "Millstone Hill ISR",
        # Madrigal documentation identifies Millstone Hill as instrument 30.
        # Try HTTPS first, then HTTP for compatibility with older deployments.
        "urls": [
            "https://madrigal.haystack.mit.edu/madrigal",
            "http://madrigal.haystack.mit.edu/madrigal",
        ],
        "instrument": 30,
        "selector": "millstone_combined",
    },
}


# ============================================================
# HELPERS
# ============================================================

def connect_madrigal(urls):
    """Connect to the first Madrigal URL that responds."""
    last_error = None

    for url in urls:
        print(f"  Trying Madrigal server: {url}")
        try:
            mad = madrigalWeb.madrigalWeb.MadrigalData(url)
            print(f"  Connected: {url}")
            return mad, url
        except Exception as exc:
            last_error = exc
            print(f"  Connection failed: {exc}")

    raise RuntimeError(
        f"Could not connect to any configured Madrigal server. "
        f"Last error: {last_error}"
    )


def wanted_file(station_key, config, file_obj):
    """Return (True/False, reason) for whether this file should be downloaded."""
    try:
        kindat = int(file_obj.kindat)
    except Exception:
        kindat = None

    filename = os.path.basename(str(file_obj.name))
    description = str(getattr(file_obj, "kindatdesc", "") or "")
    status = str(getattr(file_obj, "status", "") or "")

    text = f"{filename} {description} {status}".lower()

    if config["selector"] == "kindat":
        wanted_kindat = int(config["kindat"])

        if kindat == wanted_kindat:
            return True, f"kindat {wanted_kindat}"

        return False, f"kindat {kindat}"

    if config["selector"] == "millstone_combined":
        # Match the product selected manually in the Millstone web interface.
        # This deliberately does not assume one fixed Millstone kindat,
        # because Millstone kindat codes can vary between experiments/products.
        if "combined basic parameters file" not in text:
            return False, "not Combined basic parameters file"

        # Extra safety against products the user does not want.
        reject_terms = (
            "not for science",
            "not for scientific",
            "usrp",
            "plasma line",
        )
        if any(term in text for term in reject_terms):
            return False, "rejected non-science/plasma-line product"

        return True, "Combined basic parameters file"

    return False, "unknown selector"


def nc_output_path(output_dir: Path, remote_name: str) -> Path:
    """
    Madrigal source files are often .hdf5.
    We request netCDF4, so store locally with .nc.
    """
    base = os.path.basename(remote_name)

    # Strip the original extension.
    stem = os.path.splitext(base)[0]
    return output_dir / f"{stem}.nc"


def already_downloaded(path: Path) -> bool:
    """
    Treat a non-empty existing .nc as already downloaded.
    Empty files are removed and retried.
    """
    if not path.exists():
        return False

    if path.stat().st_size > 0:
        return True

    print(f"    Existing file is empty; removing: {path.name}")
    try:
        path.unlink()
    except OSError:
        pass

    return False


def remove_partial(path: Path):
    """Delete a partial local file left by a failed request."""
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        print(f"    Warning: could not remove partial file: {exc}")


def download_with_retries(
    mad,
    remote_file: str,
    local_file: Path,
):
    """
    Download one file as netCDF4.
    Returns (success: bool, error_message: str).
    """
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS}: netCDF4")
            mad.downloadFile(
                remote_file,
                str(local_file),
                USER_FULLNAME,
                USER_EMAIL,
                USER_AFFILIATION,
                format="netCDF4",
            )

            if not local_file.exists() or local_file.stat().st_size == 0:
                raise IOError("Download returned but local file is missing or empty")

            size_mb = local_file.stat().st_size / (1024 ** 2)
            print(f"    SUCCESS: {local_file.name} ({size_mb:.1f} MB)")
            return True, ""

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {last_error}")
            remove_partial(local_file)

        except Exception as exc:
            # Catch Madrigal/server-side errors that do not fall into
            # the standard network exception classes.
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {last_error}")
            remove_partial(local_file)

        if attempt < MAX_ATTEMPTS:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            print(f"    Waiting {delay} s before retry...")
            time.sleep(delay)

    print("    SKIPPING after final failed attempt.")
    return False, last_error


# ============================================================
# DOWNLOAD ONE STATION
# ============================================================

def download_station(station_key, config, failed_rows):
    output_dir = BASE_DIR / station_key
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"{station_key}: {config['name']}")
    print(f"Output: {output_dir}")
    print("=" * 90)

    try:
        mad, active_url = connect_madrigal(config["urls"])
    except Exception as exc:
        print(f"STATION FAILED: {exc}")
        failed_rows.append({
            "station": station_key,
            "experiment": "",
            "remote_file": "",
            "local_file": "",
            "error": f"Could not connect to station server: {exc}",
        })
        return

    sy, sm, sd, sh, smin, ss = START
    ey, em, ed, eh, emin, es = END

    try:
        experiments = mad.getExperiments(
            config["instrument"],
            sy, sm, sd, sh, smin, ss,
            ey, em, ed, eh, emin, es,
            local=1,
        )
    except Exception as exc:
        print(f"Could not list experiments: {exc}")
        failed_rows.append({
            "station": station_key,
            "experiment": "",
            "remote_file": "",
            "local_file": "",
            "error": f"Could not list experiments from {active_url}: {exc}",
        })
        return

    print(f"Found {len(experiments)} local experiments.")

    station_candidates = 0
    station_success = 0
    station_existing = 0
    station_failed = 0

    for exp_index, exp in enumerate(experiments, start=1):
        exp_name = str(getattr(exp, "name", ""))
        exp_id = getattr(exp, "id", None)

        print(
            f"\n[{exp_index}/{len(experiments)}] "
            f"{exp_name} (experiment id {exp_id})"
        )

        try:
            # False = default/realtime files, which Madrigal recommends
            # as the most reliable products.
            files = mad.getExperimentFiles(exp_id, getNonDefault=False)
        except Exception as exc:
            print(f"  Could not list files; skipping experiment: {exc}")
            failed_rows.append({
                "station": station_key,
                "experiment": exp_name,
                "remote_file": "",
                "local_file": "",
                "error": f"Could not list experiment files: {exc}",
            })
            continue

        selected = []
        for file_obj in files:
            use_it, reason = wanted_file(station_key, config, file_obj)
            if use_it:
                selected.append((file_obj, reason))

        if not selected:
            print("  No requested product in this experiment.")
            continue

        for file_obj, reason in selected:
            station_candidates += 1

            remote_file = str(file_obj.name)
            local_file = nc_output_path(output_dir, remote_file)

            print(f"  Selected: {os.path.basename(remote_file)}")
            print(f"    Reason: {reason}")
            print(f"    Output: {local_file}")

            if already_downloaded(local_file):
                size_mb = local_file.stat().st_size / (1024 ** 2)
                print(f"    ALREADY EXISTS -> SKIP ({size_mb:.1f} MB)")
                station_existing += 1
                continue

            ok, error = download_with_retries(
                mad,
                remote_file,
                local_file,
            )

            if ok:
                station_success += 1
                time.sleep(SUCCESS_DELAY)
            else:
                station_failed += 1
                failed_rows.append({
                    "station": station_key,
                    "experiment": exp_name,
                    "remote_file": remote_file,
                    "local_file": str(local_file),
                    "error": error,
                })

    print("\n" + "-" * 90)
    print(f"{station_key} summary")
    print(f"  Selected candidate files : {station_candidates}")
    print(f"  Downloaded this run      : {station_success}")
    print(f"  Already existed          : {station_existing}")
    print(f"  Failed / skipped         : {station_failed}")
    print("-" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    failed_rows = []
    stations_to_run = ["TRO", "SVL", "MIL"]

    for station_key in stations_to_run:
        try:
            download_station(
                station_key,
                STATIONS[station_key],
                failed_rows,
            )
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as exc:
            # Never let one station prevent the next station from running.
            print(f"\nUnexpected station-level error for {station_key}: {exc}")
            failed_rows.append({
                "station": station_key,
                "experiment": "",
                "remote_file": "",
                "local_file": "",
                "error": f"Unexpected station-level error: {exc}",
            })

    failed_csv = BASE_DIR / "failed_downloads.csv"

    with failed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station",
                "experiment",
                "remote_file",
                "local_file",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(failed_rows)

    print("\n" + "=" * 90)
    print("ALL REQUESTED STATIONS FINISHED")
    print("=" * 90)
    print(f"Failed-download log: {failed_csv}")

    if failed_rows:
        print(f"{len(failed_rows)} failure(s) were recorded.")
        print("Rerun this same script later: existing non-empty .nc files are skipped,")
        print("so previously failed files will be attempted again.")
    else:
        print("No failures recorded.")


if __name__ == "__main__":
    main()
