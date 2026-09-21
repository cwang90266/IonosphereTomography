"""Validate input F2/E parameters against real ionosonde product variables."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from .isr_profile_validation import (
    _utc, _resolve_window, _nearest_site, _availability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IONOSONDE_DATA_DIR = PROJECT_ROOT / "Data" / "Ionosonde_Data"

IONOSONDE_SITES = {
    "TRO": {"lat": 69.662, "lon": 18.940},
    "SVA": {"lat": 78.148, "lon": 16.043},
}

PARAMETERS = ("hmF2", "NmF2", "hmE", "NmE")


def _numeric(variable):
    return np.ma.asarray(variable[:], dtype=float).filled(np.nan).ravel()


def _time_values(ds):
    variable = ds.variables["timestamps"]
    values = _numeric(variable)
    units = str(getattr(variable, "units", "")).strip()
    if "unix" in units.lower():
        return [
            pd.to_datetime(v, unit="s", utc=True) if np.isfinite(v) else None
            for v in values
        ]
    if "since" in units.lower():
        calendar = str(getattr(variable, "calendar", "standard"))
        if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
            raise ValueError(f"Unsupported observation calendar: {calendar}")
        result = []
        for value in values:
            if not np.isfinite(value):
                result.append(None)
            else:
                date = netCDF4.num2date(value, units, calendar=calendar)
                result.append(_utc(date.isoformat()))
        return result
    raise ValueError(f"Unrecognized timestamps units: {units!r}")


def _unit_key(units):
    return (
        str(units).lower().replace(" ", "")
        .replace("**", "").replace("^", "")
        .replace("{", "").replace("}", "")
    )


def _series(ds, names, n, kind):
    """Read a scalar-per-time product variable with explicit unit conversion."""
    name = next((v for v in names if v in ds.variables), None)
    if name is None:
        return np.full(n, np.nan), None

    variable = ds.variables[name]
    values = _numeric(variable)
    if values.size != n:
        raise ValueError(f"{name}: expected {n} values, got {values.size}.")

    units = _unit_key(getattr(variable, "units", ""))
    scales = {
        "height": {"km": 1.0, "m": 1e-3},
        "density": {
            "m-3": 1.0, "1/m3": 1.0,
            "cm-3": 1e6, "1/cm3": 1e6,
        },
        "frequency": {"mhz": 1e6, "khz": 1e3, "hz": 1.0},
    }
    if units not in scales[kind]:
        raise ValueError(f"{name}: unsupported or missing units {units!r}.")

    values *= scales[kind][units]
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    return values, name


def _read_product(path, site, allow_mean_height_proxy):
    with netCDF4.Dataset(path) as ds:
        # Validate station identity from product metadata, not its filename.
        latitude = float(ds.getncattr("instrument_latitude"))
        longitude = float(ds.getncattr("instrument_longitude"))
        dlon = abs((longitude - site["lon"] + 180) % 360 - 180)
        if abs(latitude - site["lat"]) > 0.5 or dlon > 0.5:
            return []

        times = _time_values(ds)
        n = len(times)
        values, sources = {}, {}

        for parameter, aliases in (
            ("hmF2", ("hmF2", "hmf2")),
            ("hmE", ("hmE", "hme")),
            ("NmF2", ("NmF2", "nmf2")),
            ("NmE", ("NmE", "nme")),
        ):
            kind = "height" if parameter.startswith("hm") else "density"
            values[parameter], sources[parameter] = _series(
                ds, aliases, n, kind
            )

        # Density conversion from product critical frequency, not a Ne peak.
        for parameter, frequency in (("NmF2", "fof2"), ("NmE", "foe")):
            if sources[parameter] is None:
                hz, source = _series(ds, (frequency,), n, "frequency")
                values[parameter] = (hz / 8.98) ** 2
                sources[parameter] = (
                    f"{source}: Ne=(f_Hz/8.98)^2" if source else None
                )

        proxies = set()
        if allow_mean_height_proxy:
            for parameter, variable in (
                ("hmF2", "kmztrf"),
                ("hmE", "kmztre"),
            ):
                if sources[parameter] is None:
                    data, source = _series(ds, (variable,), n, "height")
                    values[parameter], sources[parameter] = data, source
                    if source:
                        proxies.add(parameter)

    return [
        {
            "time": timestamp,
            "parameters": {p: float(values[p][i]) for p in PARAMETERS},
            "sources": dict(sources),
            "height_proxies": set(proxies),
            "source_file": str(path),
        }
        for i, timestamp in enumerate(times) if timestamp is not None
    ]



def _window_tag(lo, hi):
    lo = _utc(lo)
    hi = _utc(hi)
    return f"{lo:%Y%m%d}_{lo:%H%M}{hi:%H%M}"


def _export_ionosonde_window_csv(records, site_name, site, lo, hi, output_dir):
    """Export every ionosonde parameter record in the requested time window."""
    rows = []
    for record in records:
        timestamp = _utc(record["time"])
        if not (lo <= timestamp <= hi):
            continue
        params = record["parameters"]
        rows.append({
            "ionosonde_id": site_name,
            "ionosonde_lat": float(site["lat"]),
            "ionosonde_lon": float(site["lon"]),
            "datetime_utc": timestamp,
            "hmF2_km": params.get("hmF2", np.nan),
            "NmF2_m3": params.get("NmF2", np.nan),
            "hmE_km": params.get("hmE", np.nan),
            "NmE_m3": params.get("NmE", np.nan),
            "hmF2_source": record.get("sources", {}).get("hmF2"),
            "NmF2_source": record.get("sources", {}).get("NmF2"),
            "hmE_source": record.get("sources", {}).get("hmE"),
            "NmE_source": record.get("sources", {}).get("NmE"),
            "source_file": record.get("source_file"),
        })
    path = Path(output_dir) / f"ionosonde_{site_name}_{_window_tag(lo, hi)}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def validate_ionosonde_parameters(
    hmF2,
    NmF2,
    hmE,
    NmE,
    lat,
    lon,
    time=None,
    doy=None,
    year=None,
    *,
    ionosonde_data_dir=None,
    time_window=None,
    time_window_minutes=15.0,
    output_path="ionosonde_validation.png",
    sites=None,
    max_site_distance_km=None,
    maximum_time_difference_minutes=None,
    allow_mean_height_proxy=False,
):
    """
    Read real product parameters, then plot ionosonde blue / input red.

    Missing parameters remain NaN. No observation profile peaks are computed.
    A single observation time is used for all four comparisons.
    """
    lo, hi, centre = _resolve_window(
        time_window, time, doy, year, time_window_minutes
    )
    name, site, distance = _nearest_site(
        lat, lon,
        IONOSONDE_SITES if sites is None else sites,
        max_site_distance_km,
    )

    root = (
        IONOSONDE_DATA_DIR
        if ionosonde_data_dir is None
        else Path(ionosonde_data_dir).expanduser()
    )
    print(f"[Ionosonde] Reading real observations from {root}")
    
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(set(root.rglob("*.nc")) | set(root.rglob("*.nc4")))
    else:
        raise FileNotFoundError(root)
    if not files:
        raise FileNotFoundError(f"No NetCDF ionosonde files under {root}.")

    records = []
    for path in files:
        try:
            records.extend(_read_product(path, site, allow_mean_height_proxy))
        except (OSError, KeyError, ValueError, AttributeError) as error:
            warnings.warn(f"Skipping {path}: {error}")

    records.sort(key=lambda r: r["time"])
    _availability(f"Ionosonde {name}", [r["time"] for r in records])
    for parameter in PARAMETERS:
        _availability(
            f"{name} {parameter}",
            [
                r["time"] for r in records
                if np.isfinite(r["parameters"][parameter])
            ],
        )

    candidates = [
        r for r in records
        if lo <= r["time"] <= hi
        and any(np.isfinite(v) for v in r["parameters"].values())
    ]
    if maximum_time_difference_minutes is not None:
        candidates = [
            r for r in candidates
            if abs((r["time"] - centre).total_seconds()) / 60
            <= maximum_time_difference_minutes
        ]
    if not candidates:
        raise LookupError("No usable ionosonde parameters in this window.")

    # Compute each parameter independently over the full validation window.
    # Different Dynasonde parameters can be missing at different timestamps, so
    # selecting only the nearest record can leave avoidable NaNs.
    observed = {}
    observed_counts = {}
    observed_sources = {}
    contributing_times = {}
    contributing_files = {}
    height_proxies = set()

    for parameter in PARAMETERS:
        values = []
        times = []
        files_used = []
        sources_used = []
        proxy_used = False

        for record in candidates:
            value = float(record["parameters"].get(parameter, np.nan))
            if not np.isfinite(value):
                continue

            values.append(value)
            times.append(record["time"])
            files_used.append(record["source_file"])

            source = record.get("sources", {}).get(parameter)
            if source:
                sources_used.append(source)

            if parameter in record.get("height_proxies", set()):
                proxy_used = True

        observed_counts[parameter] = len(values)
        observed[parameter] = (
            float(np.nanmedian(np.asarray(values, dtype=float)))
            if values else np.nan
        )
        contributing_times[parameter] = times
        contributing_files[parameter] = files_used

        unique_sources = list(dict.fromkeys(sources_used))
        observed_sources[parameter] = (
            unique_sources[0]
            if len(unique_sources) == 1
            else "; ".join(unique_sources)
            if unique_sources
            else None
        )

        if proxy_used:
            height_proxies.add(parameter)

    modeled = dict(zip(PARAMETERS, map(float, (hmF2, NmF2, hmE, NmE))))
    if not all(np.isfinite(v) and v > 0 for v in modeled.values()):
        raise ValueError("Input parameters must be finite and positive.")

    if height_proxies:
        warnings.warn(
            "Using mean real layer heights as explicit proxies, NOT measured "
            "peak heights: " + ", ".join(sorted(height_proxies))
        )

    table = pd.DataFrame([
        {
            "parameter": p,
            "input": modeled[p],
            "ionosonde": observed[p],
            "input_minus_ionosonde": modeled[p] - observed[p],
            "source_variable": observed_sources[p],
            "n_window_values": observed_counts[p],
            "height_is_proxy": p in height_proxies,
            "window_start_utc": lo,
            "window_end_utc": hi,
        }
        for p in PARAMETERS
    ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = _export_ionosonde_window_csv(
        records, name, site, lo, hi, output.parent
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 1))

    for ax, parameter in zip(axes, PARAMETERS):
        if np.isfinite(observed[parameter]):
            ax.bar(0, observed[parameter], color="tab:blue", label="Ionosonde")
        else:
            ax.text(
                0.03, 0.92, "Not available\nin product",
                transform=ax.transAxes, va="top", fontsize=9,
            )
        ax.bar(1, modeled[parameter], color="red", label="Input")
        ax.set_xticks([0, 1], ["Ionosonde", "Input"])
        ax.set_ylabel("km" if parameter.startswith("hm") else r"m$^{-3}$")
        title = parameter
        if parameter in height_proxies:
            title += "\n(mean-height proxy)"
        title += f"\nmedian n={observed_counts[parameter]}"
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if parameter.startswith("Nm"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle(
        f"{name}: input versus ionosonde window median\n"
        f"Window: {lo} to {hi}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(table[[
        "parameter", "input", "ionosonde",
        "n_window_values", "source_variable"
    ]])
    print(f"[Ionosonde] Saved {output}")
    print(f"[Ionosonde] Window CSV: {csv_path}")

    return {
        "site": name,
        "site_distance_km": distance,
        "target_time": centre,
        "observation_time": centre,
        "window_start": lo,
        "window_end": hi,
        "ionosonde": observed,
        "input": modeled,
        "comparison": table,
        "source_file": sorted({
            f for files in contributing_files.values() for f in files
        }),
        "height_proxies": height_proxies,
        "n_window_values": observed_counts,
        "contributing_times": contributing_times,
        "figure_path": str(output),
        "csv_path": str(csv_path),
    }