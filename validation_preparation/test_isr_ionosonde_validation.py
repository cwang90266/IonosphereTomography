"""Run on REAL files; only the input Ne/parameters are self-defined."""

import argparse
from pathlib import Path
import traceback

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from .isr_profile_validation import validate_isr_profile
from .isr_profile_validation import ISR_DATA_DIR
from .ionosonde_parameter_validation import (
    IONOSONDE_DATA_DIR, IONOSONDE_SITES, PARAMETERS,
    _read_product, validate_ionosonde_parameters,
)
from .raw_observation_io import observation_files, read_isr_file


def find_common_time(site, half_window_min, allow_mean_height_proxy):
    """Return earliest UTC centre with ISR + all four ionosonde values."""
    from demo_isr_initial_conditions import INSTRUMENTS
    isr_site = INSTRUMENTS[site]
    isr_times = []
    for path in observation_files(ISR_DATA_DIR):
        try:
            for record in read_isr_file(path):
                if (str(record.get("kindat")) == "6400"
                        and abs(record["lat"] - isr_site["lat"]) <= .5
                        and abs(record["lon"] - isr_site["lon"]) <= .5):
                    isr_times.append(pd.Timestamp(record["time"]))
        except Exception:
            continue
    iono_site = IONOSONDE_SITES[site if site in IONOSONDE_SITES else "TRO"]
    iono_records = []
    for path in observation_files(IONOSONDE_DATA_DIR):
        try:
            iono_records.extend(_read_product(path, iono_site, allow_mean_height_proxy))
        except Exception:
            continue
    complete = [r for r in iono_records if all(
        np.isfinite(r["parameters"].get(p, np.nan)) for p in PARAMETERS
    )]
    if not isr_times or not complete:
        return None
    half = pd.Timedelta(minutes=half_window_min)
    isr_times = sorted(set(isr_times))
    for center in isr_times:
        if any(abs(r["time"] - center) <= half for r in complete):
            return center
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # The supplied ISR NetCDF covers approximately 04:00--08:00 UTC.
    parser.add_argument("--centre", default="2025-12-15T06:00:00Z")
    parser.add_argument("--half-window-min", type=float, default=15.)
    parser.add_argument("--height-bin-km", type=float, default=10.)
    parser.add_argument("--site", choices=("TRO", "ESR"), default="TRO")
    parser.add_argument("--output-dir", default="Figures/validation_test")
    parser.add_argument("--allow-mean-height-proxy", action="store_true")
    parser.add_argument(
        "--find-common-time", action="store_true",
        help="Find the first window containing ISR Ne and all four ionosonde values.",
    )
    args = parser.parse_args()
    if args.half_window_min <= 0 or args.height_bin_km <= 0:
        parser.error("Time-window half-width and altitude bin width must be positive")

    # Reuse the project's model grid and station configuration.
    from demo_isr_initial_conditions import ALT_GRID, INSTRUMENTS

    centre = pd.Timestamp(args.centre)
    centre = centre.tz_localize("UTC") if centre.tzinfo is None else centre.tz_convert("UTC")
    delta = pd.Timedelta(minutes=args.half_window_min)
    if args.find_common_time:
        found = find_common_time(args.site, args.half_window_min,
                                 args.allow_mean_height_proxy)
        if found is None:
            raise SystemExit(
                "No common time found with ISR Ne and all four ionosonde "
                "parameters in the same window."
            )
        centre = found
        print(f"[Common time] Using {centre} UTC")
    window = dict(window_key=centre.strftime("%Y-%m-%d_%H%M"),
                  lo=centre-delta, hi=centre+delta, t_centre=centre)

    # SELF-DEFINED MODEL INPUT ONLY. No synthetic ISR/ionosonde truth or files.
    hmF2, NmF2, hmE, NmE = 280., 5e11, 110., 8e10
    altitude = np.asarray(ALT_GRID, dtype=float).ravel()
    def chapman(nm, hm, scale):
        z = (altitude-hm)/scale
        return nm*np.exp(.5*(1-z-np.exp(np.clip(-z, -700, 700))))
    input_ne = chapman(NmF2, hmF2, 50.) + chapman(NmE, hmE, 10.)
    lat, lon = (float(INSTRUMENTS[args.site][k]) for k in ("lat", "lon"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors = []

    try:
        result = validate_isr_profile(
            input_ne, altitude, lat, lon, time_window=window,
            height_bin_km=args.height_bin_km,
            output_path=output/"isr_validation.png",
        )
        if not np.isfinite(result["by_height"]["rmse_ne_m3"]).any():
            raise RuntimeError("No finite height RMSE")
        print("PASS: ISR reading/comparison/plot")
        print(result["by_time"].to_string(index=False))
    except Exception as error:
        traceback.print_exc()
        errors.append(f"ISR: {error}")

    try:
        result = validate_ionosonde_parameters(
            hmF2, NmF2, hmE, NmE, lat, lon, time_window=window,
            allow_mean_height_proxy=args.allow_mean_height_proxy,
            output_path=output/"ionosonde_validation.png",
        )
        print("PASS: ionosonde reading/available comparisons/plot")
        missing = [p for p,v in result["ionosonde"].items() if not np.isfinite(v)]
        if missing:
            print("NOT VALIDATED (missing at selected time):", ", ".join(missing))
        if result["height_proxies"]:
            print("MEAN-HEIGHT PROXIES, not measured peak heights:", result["height_proxies"])
    except Exception as error:
        traceback.print_exc()
        errors.append(f"Ionosonde: {error}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Figures and CSVs: {output.resolve()}")


if __name__ == "__main__":
    main()
