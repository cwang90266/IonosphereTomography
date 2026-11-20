#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download EISCAT Tromsø Dynasonde daily files from Madrigal (CEDAR) for
E-region (NmE/hmE) validation (E_Region_Retrieval_Plan.md, Section 5.7).

Product: instrument code 2950 ("EISCAT Tromsø Dynasonde"), kindat 4010
("DSND Realtime ionospheric data analysis"), one file per UTC day. It is
the same product as the existing
``Data/Tomography_data/Ionosonde_Data/tromso_dynasonde_2025-12-1*.nc``
files, which ``validation_preparation.ionosonde_parameter_validation``
already reads. Files are saved as netCDF4 with the same names
(``tromso_dynasonde_YYYY-MM-DD.nc``).

Madrigal records the user name, email and affiliation sent with each
download (its usage statistics), so they are required arguments here.

    python TwoStageTomography/fetch_tromso_dynasonde.py \
        --start 2025-10-14 --end 2025-10-17 \
        --out-dir /path/to/Ionosonde_Data \
        --user-name "First Last" --user-email you@example.com \
        --user-affiliation "PlanetIQ"
"""

from __future__ import annotations

import argparse
import datetime as dt
import urllib.parse
from pathlib import Path

import requests

MADRIGAL_URL = "https://cedar.openmadrigal.org/"
INSTRUMENT_CODE = 2950
KINDAT = 4010
NETCDF4_FILE_TYPE = -3
TIMEOUT_S = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="First UTC day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last UTC day, YYYY-MM-DD (inclusive)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--user-name", required=True)
    parser.add_argument("--user-email", required=True)
    parser.add_argument("--user-affiliation", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _get_lines(service: str, params: dict) -> list[str]:
    response = requests.get(MADRIGAL_URL + service, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    return [line for line in response.text.splitlines() if line.strip()]


def list_experiment_ids(start: dt.date, end: dt.date) -> list[int]:
    """Experiment ids of the Tromsø Dynasonde served by the CEDAR site itself."""
    params = {
        "code": INSTRUMENT_CODE,
        "startyear": start.year, "startmonth": start.month, "startday": start.day,
        "starthour": 0, "startmin": 0, "startsec": 0,
        "endyear": end.year, "endmonth": end.month, "endday": end.day,
        "endhour": 23, "endmin": 59, "endsec": 59,
        "local": 0,
    }
    ids = []
    for line in _get_lines("getExperimentsService.py", params):
        fields = line.split(",")
        # Each experiment is listed once per Madrigal site; only the CEDAR
        # copy can be downloaded from this server.
        if fields[1].startswith(MADRIGAL_URL):
            ids.append(int(fields[0]))
    return ids


def list_files(experiment_id: int) -> list[str]:
    """Server-side paths of the kindat-4010 files in one experiment."""
    paths = []
    for line in _get_lines("getExperimentFilesService.py", {"id": experiment_id}):
        fields = line.split(",")
        if int(fields[1]) == KINDAT:
            paths.append(fields[0])
    return paths


def download(server_path: str, out_path: Path, args: argparse.Namespace) -> None:
    query = urllib.parse.urlencode({
        "fileName": server_path,
        "fileType": NETCDF4_FILE_TYPE,
        "user_fullname": args.user_name,
        "user_email": args.user_email,
        "user_affiliation": args.user_affiliation,
    })
    response = requests.get(f"{MADRIGAL_URL}getMadfile.cgi?{query}", timeout=TIMEOUT_S)
    response.raise_for_status()
    # A netCDF4 file is HDF5 underneath; anything else is an error page.
    if not response.content.startswith(b"\x89HDF"):
        raise RuntimeError(f"{server_path}: server did not return a netCDF4 file")
    out_path.write_bytes(response.content)


def main() -> None:
    args = parse_args()
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for experiment_id in list_experiment_ids(start, end):
        for server_path in list_files(experiment_id):
            out_path = args.out_dir / (Path(server_path).stem + ".nc")
            if out_path.exists() and not args.overwrite:
                print(f"skip (exists): {out_path.name}")
                continue
            download(server_path, out_path, args)
            print(f"saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
