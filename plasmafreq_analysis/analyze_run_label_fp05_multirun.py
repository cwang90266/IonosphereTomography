#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run-level post-processing of tomography-vs-ISR plasma-frequency difference.
Run this only after a complete tomography run_label has finished.
Expected tomography output folder structure
-----------------
<TOMOGRAPHY_OUTPUT_ROOT>/<run_label>/
    2025-08-29_15001600/
        dual_isr_errors.csv
        no_boost/
            ..._isr_comparison_cross_style/
                pooled_rmse.csv
        boosted/
            ..._isr_comparison_cross_style/
                pooled_rmse.csv
    2025-08-29_16001700/
        ...


Plasma-frequency metric
-----------------------
The 0.5-MHz accepted region:

    |Delta fp| <= 0.5 MHz

Outputs
-------
<TOMOGRAPHY_OUTPUT_ROOT>/<run_label>/plasmafreq_analysis/

    fp_diff_percentiles_vs_altitude_all_results.csv

    no_boost/
        fp_diff_comparisons/
        fp_diff_percentiles_by_figure/

    boosted/
        fp_diff_comparisons/
        fp_diff_percentiles_by_figure/

Plot layout
-----------
Every plot has 2 + N panels, where N is the number of compared run labels:

    1. IRI ANCHOR prior
    2. IRI prior
    3+. one tomography posterior panel per run label

One run label -> 3 panels
Two run labels -> 4 panels
Three run labels -> 5 panels

Example bash command
-------
python -m plasmafreq_analysis.analyze_run_label_fp05_multirun \
    --run-label "label1" \
    (optional below, upto three)
    --run-label "label2" \
    --run-label "label3"
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


DEFAULT_TOMOGRAPHY_OUTPUT_ROOT = Path(
    "/home/pin/Desktop/tomography_output"
    #"/mnt/tmp-sbem-ionosphere-tomography/tomography_output"
)
DEFAULT_ISR_DIR = Path(
    "/home/pin/Desktop/ISR_PCA/PCA_data"
    #"/home/pyxis/Test/2026-10-13_ionosphere_tomography/ISR_data"
)
ISR_FILENAMES = {
    "MIL": "MILISR2025_nonan.nc",
    "TRO": "TROISR2025_nonan.nc",
    "SVL": "SVLISR2025_nonan.nc",
}

STATION_LABELS = {
    "MIL": "Millstone Hill ISR",
    "TRO": "Tromso ISR",
    "SVL": "Svalbard ISR",
}

CONDITION_ORDER = [
    "no_boost",
    "boosted",
]

CONDITION_LABELS = {
    "no_boost": "No boost",
    "boosted": "Boosted",
}

STYLE_ORDER_PREFERRED = [
    "ANCHOR",
    "PCA_3D",
    "PCA_3D_10ex",
    "PCA_1D",
    "PCA_1D_10ex",
    "PCA_1D_10ex_ISR",
]

IRI_ANCHOR_STYLE = "ANCHOR"
IRI_PCA1D_CANDIDATES = (
    "PCA_1D",
    "PCA_1D_10ex",
)

FP_COEFF_MHZ = 8.98e-6
FP_DIFF_LIMIT_MHZ = 0.5

ALT_MIN_KM = 80.0
ALT_MAX_KM = 700.0
ALT_TICK_KM = 50.0
FP_XMAX_MHZ = 2.5

DUAL_REQUIRED = [
    "style",
    "condition",
    "station",
    "window_start_utc",
    "window_end_utc",
    "n_ro_profiles",
    "n_tec_rays",
    "altitude_km",
    "analysis_error_m3",
]

POOLED_REQUIRED = [
    "style",
    "altitude_km",
    "rmse_forecast_m3",
]

HOUR_FOLDER_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start>\d{4})(?P<end>\d{4})$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--run-label",
        action="append",
        required=True,
        help=(
            "Completed tomography run-label folder name. "
            "Repeat --run-label up to three times to compare runs. "
            "Blank values or the literal 'None' are ignored. "
            "The first valid label owns the output directory and IRI prior panels."
        ),
    )

    p.add_argument(
        "--tomography-output-root",
        type=Path,
        default=DEFAULT_TOMOGRAPHY_OUTPUT_ROOT,
        help=(
            "Root containing run-label directories. "
            f"Default: {DEFAULT_TOMOGRAPHY_OUTPUT_ROOT}"
        ),
    )

    p.add_argument(
        "--isr-data",
        type=Path,
        default=DEFAULT_ISR_DIR,
        help=(
            "Directory containing MIL/TRO/SVL ISR NetCDF files. "
            f"Default: {DEFAULT_ISR_DIR}"
        ),
    )

    return p.parse_args()


def normalize_run_labels(values):
    """Return 1-3 usable run labels, preserving command-line order."""
    if isinstance(values, str) or values is None:
        values = [values]

    labels = []

    for value in values:
        if value is None:
            continue

        label = str(value).strip()

        if not label:
            continue

        if label.lower() in {"none", "null"}:
            continue

        labels.append(label)

    if not labels:
        raise ValueError(
            "At least one non-empty --run-label is required."
        )

    if len(labels) > 3:
        raise ValueError(
            "At most three non-empty --run-label values are supported."
        )

    return labels


def hour_info_from_name(name: str):
    m = HOUR_FOLDER_RE.match(str(name))

    if not m:
        return None

    day = datetime.strptime(
        m.group("date"),
        "%Y-%m-%d",
    )

    start_token = m.group("start")

    start = pd.Timestamp(
        day.replace(
            hour=int(start_token[:2]),
            minute=int(start_token[2:]),
        )
    )

    # Current tomography products are one-hour windows.
    end = start + pd.Timedelta(hours=1)

    return start, end


def hour_folder_from_path(
    path: Path,
    run_dir: Path,
):
    path = Path(path).resolve()
    run_dir = Path(run_dir).resolve()

    try:
        rel = path.relative_to(run_dir)
    except ValueError:
        return None

    for part in rel.parts:
        if HOUR_FOLDER_RE.match(part):
            info = hour_info_from_name(part)

            if info is not None:
                return (
                    part,
                    info[0],
                    info[1],
                )

    return None


def condition_from_path(path: Path):
    parts = {
        p.lower()
        for p in Path(path).parts
    }

    if "no_boost" in parts:
        return "no_boost"

    if "boosted" in parts:
        return "boosted"

    return None


def station_from_path(path: Path):
    """
    Infer which ISR station a pooled_rmse.csv belongs to from the
    *_isr_comparison_cross_style folder name.

    Important for runs containing more than one ISR:

        tromso_allstyles_..._isr_comparison_cross_style
            -> TRO

        tromso_allstyles_..._svl_isr_comparison_cross_style
            -> SVL

    The explicit station token immediately before
    "_isr_comparison_cross_style" overrides the run-center prefix.

    Single-ISR examples:
        mil_allstyles_..._isr_comparison_cross_style -> MIL
        tromso_allstyles_..._isr_comparison_cross_style -> TRO
        svalbard_allstyles_..._isr_comparison_cross_style -> SVL
    """
    path = Path(path)

    # Search folder names from nearest parent outward.
    for part in reversed(path.parts):
        name = part.lower()

        if "isr_comparison_cross_style" not in name:
            continue

        # Additional ISR suffix overrides the center-site prefix.
        suffix_map = [
            ("_svl_isr_comparison_cross_style", "SVL"),
            ("_tro_isr_comparison_cross_style", "TRO"),
            ("_mil_isr_comparison_cross_style", "MIL"),
            ("_svalbard_isr_comparison_cross_style", "SVL"),
            ("_tromso_isr_comparison_cross_style", "TRO"),
            ("_millstone_isr_comparison_cross_style", "MIL"),
        ]

        for suffix, station in suffix_map:
            if name.endswith(suffix):
                return station

        # Otherwise this is the primary ISR represented by the prefix.
        if name.startswith("mil_allstyles"):
            return "MIL"
        if name.startswith("tromso_allstyles") or name.startswith("tro_allstyles"):
            return "TRO"
        if name.startswith("svalbard_allstyles") or name.startswith("svl_allstyles"):
            return "SVL"

    # Generic fallback for unusual naming.
    text = str(path).lower()

    for code, tokens in (
        ("MIL", ("millstone", "mil")),
        ("TRO", ("tromso", "tro")),
        ("SVL", ("svalbard", "svl")),
    ):
        for token in tokens:
            if re.search(rf"(^|[/_\-]){token}([/_\-]|$)", text):
                return code

    return None


def canonical_style_order(styles):
    styles = list(
        dict.fromkeys(
            str(x)
            for x in styles
        )
    )

    preferred = [
        s
        for s in STYLE_ORDER_PREFERRED
        if s in styles
    ]

    extras = sorted(
        s
        for s in styles
        if s not in STYLE_ORDER_PREFERRED
    )

    return preferred + extras


def decode_strings(values):
    return np.asarray(
        [
            x.decode("utf-8")
            if isinstance(x, bytes)
            else str(x)
            for x in values
        ],
        dtype=object,
    )


def load_isr_dataset(path: Path):
    with h5py.File(path, "r") as f:
        required = [
            "altitude",
            "Ne",
            "time_utc",
            "local_time",
        ]

        for key in required:
            if key not in f:
                raise KeyError(
                    f"{key!r} not found in {path}"
                )

        altitude = np.asarray(
            f["altitude"][:],
            dtype=float,
        ).reshape(-1)

        ne = np.asarray(
            f["Ne"][:],
            dtype=float,
        )

        utc_raw = decode_strings(
            f["time_utc"][:]
        )

        lt_raw = decode_strings(
            f["local_time"][:]
        )

    if ne.ndim != 2:
        raise ValueError(
            f"Expected 2-D Ne in {path}; "
            f"got {ne.shape}"
        )

    if ne.shape[0] != altitude.size:
        if ne.shape[1] == altitude.size:
            ne = ne.T
        else:
            raise ValueError(
                f"Ne shape {ne.shape} incompatible "
                f"with altitude size {altitude.size}"
            )

    time_utc = pd.to_datetime(
        utc_raw,
        errors="coerce",
    )

    local_time = pd.to_datetime(
        lt_raw,
        errors="coerce",
    )

    valid = ~pd.isna(time_utc)

    time_utc = pd.DatetimeIndex(
        time_utc[valid]
    )

    local_time = pd.DatetimeIndex(
        local_time[valid]
    )

    ne = ne[
        :,
        np.asarray(valid),
    ]

    order = np.argsort(
        time_utc.values
    )

    return {
        "altitude_km": altitude,
        "Ne": ne[:, order],
        "time_utc": time_utc[order],
        "local_time": local_time[order],
    }


def nearest_isr_profile_in_hour(
    isr,
    start,
    end,
):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    midpoint = (
        start
        + (end - start) / 2
    )

    times = isr[
        "time_utc"
    ]

    if len(times) == 0:
        return None

    idx = int(
        np.argmin(
            np.abs(
                times.asi8
                - midpoint.value
            )
        )
    )

    ts = times[idx]

    if not (
        start
        <= ts
        < end
    ):
        return None

    return {
        "index": idx,
        "time_utc": ts,
        "local_time": isr[
            "local_time"
        ][idx],
        "Ne": np.asarray(
            isr["Ne"][:, idx],
            dtype=float,
        ),
    }


def ne_m3_to_fp_mhz(ne_m3):
    ne_m3 = np.asarray(
        ne_m3,
        dtype=float,
    )

    return (
        FP_COEFF_MHZ
        * np.sqrt(
            np.clip(
                ne_m3,
                0.0,
                None,
            )
        )
    )


def interpolate_native(
    isr_alt,
    values,
    target_alt,
):
    isr_alt = np.asarray(
        isr_alt,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    target_alt = np.asarray(
        target_alt,
        dtype=float,
    )

    good = (
        np.isfinite(isr_alt)
        & np.isfinite(values)
    )

    if np.count_nonzero(good) < 2:
        return np.full(
            target_alt.shape,
            np.nan,
        )

    x = isr_alt[good]
    y = values[good]

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    out = np.interp(
        target_alt,
        x,
        y,
    )

    out[
        (target_alt < x.min())
        | (target_alt > x.max())
    ] = np.nan

    return out


def discover_run_files(
    run_dir: Path,
):
    """
    Discover only the intended run-level CSV products.

    pooled_rmse.csv is accepted ONLY when its immediate parent folder ends
    with "_isr_comparison_cross_style". This excludes older per-style files
    such as ..._isr_comparison/ANCHOR/pooled_rmse.csv.
    """
    dual_files = sorted(
        run_dir.glob("*/dual_isr_errors.csv")
    )

    pooled_files = []

    for condition in CONDITION_ORDER:
        pooled_files.extend(
            sorted(
                run_dir.glob(
                    f"*/{condition}/*_isr_comparison_cross_style/pooled_rmse.csv"
                )
            )
        )

    return dual_files, pooled_files


def load_dual_files(
    dual_files,
):
    frames = []

    for path in dual_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(
                f"WARNING: could not read "
                f"{path}: {exc}"
            )
            continue

        missing = [
            col
            for col in DUAL_REQUIRED
            if col not in df.columns
        ]

        if missing:
            print(
                f"SKIP dual CSV {path}: "
                f"missing {missing}"
            )
            continue

        sub = df[
            DUAL_REQUIRED
        ].copy()

        sub[
            "source_file"
        ] = str(path)

        frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=(
                DUAL_REQUIRED
                + ["source_file"]
            )
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    out["style"] = (
        out["style"]
        .astype(str)
        .str.strip()
    )

    out["condition"] = (
        out["condition"]
        .astype(str)
        .str.strip()
    )

    out["station"] = (
        out["station"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for col in (
        "n_ro_profiles",
        "n_tec_rays",
        "altitude_km",
        "analysis_error_m3",
    ):
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    out["window_start_utc"] = (
        pd.to_datetime(
            out[
                "window_start_utc"
            ],
            errors="coerce",
        )
    )

    out["window_end_utc"] = (
        pd.to_datetime(
            out[
                "window_end_utc"
            ],
            errors="coerce",
        )
    )

    return out.dropna(
        subset=[
            "style",
            "condition",
            "station",
            "window_start_utc",
            "window_end_utc",
            "altitude_km",
        ]
    ).reset_index(
        drop=True
    )


def load_pooled_files(
    pooled_files,
    run_dir,
    dual_df,
):
    frames = []

    stations_by_hour = {}

    if not dual_df.empty:
        for (
            start,
            end,
        ), group in dual_df.groupby(
            [
                "window_start_utc",
                "window_end_utc",
            ]
        ):
            stations_by_hour[
                (
                    pd.Timestamp(start),
                    pd.Timestamp(end),
                )
            ] = sorted(
                set(
                    group[
                        "station"
                    ]
                    .astype(str)
                    .str.upper()
                )
            )

    for path in pooled_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(
                f"WARNING: could not read "
                f"{path}: {exc}"
            )
            continue

        missing = [
            col
            for col in POOLED_REQUIRED
            if col not in df.columns
        ]

        if missing:
            print(
                f"SKIP pooled CSV {path}: "
                f"missing {missing}"
            )
            continue

        info = hour_folder_from_path(
            path,
            run_dir,
        )

        if info is None:
            print(
                f"SKIP pooled CSV {path}: "
                "cannot parse hour folder"
            )
            continue

        (
            hour_folder,
            start,
            end,
        ) = info

        condition = (
            condition_from_path(
                path
            )
        )

        if condition is None:
            print(
                f"SKIP pooled CSV {path}: "
                "condition not found in path"
            )
            continue

        station = station_from_path(
            path
        )

        if station is None:
            candidates = (
                stations_by_hour.get(
                    (
                        start,
                        end,
                    ),
                    [],
                )
            )

            if len(candidates) == 1:
                station = (
                    candidates[0]
                )
            else:
                print(
                    f"SKIP pooled CSV {path}: "
                    "station cannot be inferred "
                    f"(hour stations={candidates})"
                )
                continue

        sub = df[
            POOLED_REQUIRED
        ].copy()

        sub["style"] = (
            sub["style"]
            .astype(str)
            .str.strip()
        )

        sub["altitude_km"] = (
            pd.to_numeric(
                sub[
                    "altitude_km"
                ],
                errors="coerce",
            )
        )

        sub[
            "rmse_forecast_m3"
        ] = pd.to_numeric(
            sub[
                "rmse_forecast_m3"
            ],
            errors="coerce",
        )

        sub[
            "station"
        ] = station

        sub[
            "condition"
        ] = condition

        sub[
            "window_start_utc"
        ] = start

        sub[
            "window_end_utc"
        ] = end

        sub[
            "hour_folder"
        ] = hour_folder

        sub[
            "source_file"
        ] = str(path)

        frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=(
                POOLED_REQUIRED
                + [
                    "station",
                    "condition",
                    "window_start_utc",
                    "window_end_utc",
                    "hour_folder",
                    "source_file",
                ]
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def add_fp_difference(
    df,
    error_col,
    isr_by_station,
):
    """
    Match every station/hour to the ISR profile in that hour and compute:

        Ne_model_proxy = Ne_ISR + |Ne_error|

        |Delta fp| =
            |fp(Ne_model_proxy) - fp(Ne_ISR)|

    No normalized-R metric is computed here.
    """
    if df.empty:
        out = df.copy()

        out[
            "matched_isr_time_utc"
        ] = pd.NaT

        out[
            "matched_isr_local_time"
        ] = pd.NaT

        out[
            "matched_isr_ne_m3"
        ] = np.nan

        out[
            "plasma_frequency_diff_mhz"
        ] = np.nan

        out[
            "hour_id"
        ] = pd.Series(
            dtype="object"
        )

        return out

    pieces = []

    for (
        station,
        start,
        end,
    ), group in df.groupby(
        [
            "station",
            "window_start_utc",
            "window_end_utc",
        ],
        sort=True,
    ):
        station = (
            str(station)
            .upper()
        )

        if station not in isr_by_station:
            print(
                f"WARNING: no ISR dataset "
                f"loaded for {station}; "
                "skipping hour"
            )
            continue

        isr = isr_by_station[
            station
        ]

        match = (
            nearest_isr_profile_in_hour(
                isr,
                start,
                end,
            )
        )

        if match is None:
            print(
                "WARNING: no in-hour ISR "
                f"profile for {station} "
                f"{start} to {end}"
            )
            continue

        target_alt = (
            group[
                "altitude_km"
            ]
            .to_numpy(
                dtype=float
            )
        )

        isr_ne = (
            interpolate_native(
                isr[
                    "altitude_km"
                ],
                match["Ne"],
                target_alt,
            )
        )

        error = np.abs(
            pd.to_numeric(
                group[
                    error_col
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
        )

        model_proxy = (
            isr_ne
            + error
        )

        fp_diff = np.full(
            error.shape,
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(
                isr_ne
            )
            & (
                isr_ne
                >= 0.0
            )
            & np.isfinite(
                model_proxy
            )
            & (
                model_proxy
                >= 0.0
            )
        )

        fp_diff[
            valid
        ] = np.abs(
            ne_m3_to_fp_mhz(
                model_proxy[
                    valid
                ]
            )
            - ne_m3_to_fp_mhz(
                isr_ne[
                    valid
                ]
            )
        )

        out = group.copy()

        out[
            "matched_isr_time_utc"
        ] = match[
            "time_utc"
        ]

        out[
            "matched_isr_local_time"
        ] = match[
            "local_time"
        ]

        out[
            "matched_isr_ne_m3"
        ] = isr_ne

        out[
            "plasma_frequency_diff_mhz"
        ] = fp_diff

        out[
            "hour_id"
        ] = (
            f"{pd.Timestamp(start):%Y-%m-%d %H:%M}-"
            f"{pd.Timestamp(end):%H:%M}"
        )

        pieces.append(out)

    if not pieces:
        return (
            df.iloc[
                0:0
            ]
            .copy()
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def fp_diff_stats(
    df_sub,
):
    rows = []

    if (
        df_sub.empty
        or (
            "plasma_frequency_diff_mhz"
            not in df_sub.columns
        )
    ):
        return pd.DataFrame(
            columns=[
                "altitude_km",
                "q25_fp_diff_mhz",
                "median_fp_diff_mhz",
                "q75_fp_diff_mhz",
                "#_of_hr",
                "#_of_hr_fp_diff_le_0p5",
                "pct_fp_diff_le_0p5",
            ]
        )

    for (
        altitude,
        group,
    ) in df_sub.groupby(
        "altitude_km"
    ):
        values = pd.to_numeric(
            group[
                "plasma_frequency_diff_mhz"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.size == 0:
            continue

        n = int(
            values.size
        )

        n_le = int(
            np.count_nonzero(
                values
                <= FP_DIFF_LIMIT_MHZ
            )
        )

        rows.append(
            {
                "altitude_km": float(
                    altitude
                ),
                "q25_fp_diff_mhz": float(
                    np.nanpercentile(
                        values,
                        25,
                    )
                ),
                "median_fp_diff_mhz": float(
                    np.nanmedian(
                        values
                    )
                ),
                "q75_fp_diff_mhz": float(
                    np.nanpercentile(
                        values,
                        75,
                    )
                ),
                "#_of_hr": n,
                "#_of_hr_fp_diff_le_0p5": n_le,
                "pct_fp_diff_le_0p5": (
                    100.0
                    * n_le
                    / n
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "altitude_km"
        )
        .reset_index(
            drop=True
        )
    )


def choose_iri_style_for_case(
    prior_case: pd.DataFrame,
):
    """
    Choose the pooled-RMSE prior style used for the middle "IRI" panel.

    Selection is local to one station/condition so missing styles in one case
    do not affect another case.

    Priority:
        1. PCA_1D
        2. PCA_1D_10ex
        3. first available non-ANCHOR style in canonical style order

    ANCHOR is never used for the middle panel.
    """
    if prior_case.empty:
        return None

    available = set(
        prior_case[
            "style"
        ]
        .astype(str)
        .str.strip()
    )

    non_anchor = {
        style
        for style in available
        if style != IRI_ANCHOR_STYLE
    }

    if not non_anchor:
        return None

    for style in IRI_PCA1D_CANDIDATES:
        if style in non_anchor:
            return style

    ordered = canonical_style_order(
        non_anchor
    )

    return (
        ordered[0]
        if ordered
        else None
    )


def build_master_fp_summary(
    prior,
    posterior_by_label,
    primary_run_label,
):
    pieces = []

    for (station, condition), case_root in prior.groupby(
        ["station", "condition"],
        sort=True,
    ):
        iri_style = choose_iri_style_for_case(case_root)
        if iri_style is not None:
            iri_sub = case_root[
                case_root["style"] == iri_style
            ].copy()
            stats = fp_diff_stats(iri_sub)
            if not stats.empty:
                stats.insert(0, "source", "prior")
                stats.insert(1, "result", "IRI")
                stats.insert(2, "run_label", primary_run_label)
                stats.insert(3, "station", station)
                stats.insert(4, "condition", condition)
                stats.insert(5, "style", iri_style)
                pieces.append(stats)

    for run_label, posterior in posterior_by_label.items():
        if posterior.empty:
            continue
        for (station, condition, style), sub in posterior.groupby(
            ["station", "condition", "style"],
            sort=True,
        ):
            stats = fp_diff_stats(sub)
            if stats.empty:
                continue
            stats.insert(0, "source", "posterior")
            stats.insert(1, "result", "Tomography")
            stats.insert(2, "run_label", run_label)
            stats.insert(3, "station", station)
            stats.insert(4, "condition", condition)
            stats.insert(5, "style", style)
            pieces.append(stats)

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)


def draw_fp_diff_panel(
    ax,
    df_sub,
    title,
    show_ylabel,
):
    ax.axvspan(
        0.0,
        FP_DIFF_LIMIT_MHZ,
        color="lightcoral",
        alpha=0.15,
        zorder=0,
    )

    ax.axvline(
        FP_DIFF_LIMIT_MHZ,
        color="red",
        linestyle="--",
        linewidth=1.7,
        zorder=4,
    )

    if (
        df_sub.empty
        or not np.any(
            np.isfinite(
                pd.to_numeric(
                    df_sub[
                        "plasma_frequency_diff_mhz"
                    ],
                    errors="coerce",
                )
            )
        )
    ):
        ax.text(
            0.5,
            0.5,
            "No Data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )

        ax.set_title(
            title
            + "\nNo Data",
            fontsize=13,
        )

    else:
        for (
            _hour_id,
            hourly,
        ) in df_sub.groupby(
            "hour_id",
            sort=True,
        ):
            hourly = (
                hourly.groupby(
                    "altitude_km",
                    as_index=False,
                )[
                    "plasma_frequency_diff_mhz"
                ]
                .median()
                .sort_values(
                    "altitude_km"
                )
            )

            ax.plot(
                hourly[
                    "plasma_frequency_diff_mhz"
                ].to_numpy(
                    dtype=float
                ),
                hourly[
                    "altitude_km"
                ].to_numpy(
                    dtype=float
                ),
                color="0.68",
                linewidth=0.9,
                alpha=0.40,
                zorder=1,
            )

        stats = fp_diff_stats(
            df_sub
        )

        if not stats.empty:
            altitude = stats[
                "altitude_km"
            ].to_numpy(
                dtype=float
            )

            ax.fill_betweenx(
                altitude,
                stats[
                    "q25_fp_diff_mhz"
                ].to_numpy(
                    dtype=float
                ),
                stats[
                    "q75_fp_diff_mhz"
                ].to_numpy(
                    dtype=float
                ),
                color="lightblue",
                alpha=0.35,
                zorder=2,
            )

            ax.plot(
                stats[
                    "median_fp_diff_mhz"
                ].to_numpy(
                    dtype=float
                ),
                altitude,
                color="blue",
                linewidth=2.5,
                zorder=3,
            )

        ax.set_title(
            title
            + "\n"
            + (
                f"Hours="
                f"{df_sub['hour_id'].nunique()}"
            ),
            fontsize=13,
        )

    ax.set_xlim(
        0.0,
        FP_XMAX_MHZ,
    )

    ax.set_ylim(
        ALT_MIN_KM,
        ALT_MAX_KM,
    )

    ax.yaxis.set_major_locator(
        MultipleLocator(
            ALT_TICK_KM
        )
    )

    ax.set_xlabel(
        r"Plasma frequency difference "
        r"$|\Delta f_p|$ (MHz)",
        fontsize=12,
    )

    if show_ylabel:
        ax.set_ylabel(
            "Altitude (km)",
            fontsize=12,
        )

    ax.grid(
        True,
        alpha=0.25,
    )


def fp_legend():
    return [
        Line2D(
            [0],
            [0],
            color="0.68",
            lw=1.0,
            alpha=0.55,
            label="Individual hours",
        ),
        Patch(
            facecolor="lightblue",
            edgecolor="none",
            alpha=0.35,
            label="25-75 percentile",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            lw=2.5,
            label=(
                "Median plasma-frequency "
                "difference"
            ),
        ),
        Line2D(
            [0],
            [0],
            color="red",
            lw=1.7,
            linestyle="--",
            label="0.5 MHz limit",
        ),
        Patch(
            facecolor="lightcoral",
            edgecolor="none",
            alpha=0.15,
            label=(
                r"Accepted region "
                r"($|\Delta f_p| \leq 0.5$ MHz)"
            ),
        ),
    ]


def plot_comparison_panels(
    iri_prior,
    posterior_cases,
    station,
    style,
    condition,
    output_file,
):
    """Plot IRI | one posterior panel per run label."""
    n_panels = 1 + len(posterior_cases)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.1 * n_panels, 8.5),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    draw_fp_diff_panel(axes[0], iri_prior, "IRI", True)

    for panel_index, (run_label, posterior_case) in enumerate(
        posterior_cases,
        start=1,
    ):
        draw_fp_diff_panel(
            axes[panel_index],
            posterior_case,
            run_label,
            False,
        )

    axes[-1].legend(
        handles=fp_legend(),
        loc="best",
        fontsize=8,
    )

    station_label = STATION_LABELS.get(station, station)
    condition_label = CONDITION_LABELS.get(condition, condition)

    fig.suptitle(
        f"{station_label} | {style} | {condition_label} | "
        r"$|\Delta f_p|$ comparison",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)



def build_per_figure_fp_csv(
    iri_prior,
    posterior_cases,
    station,
    style,
    condition,
    iri_source_style,
):
    """
    Build one wide/horizontal CSV matching the panel arrangement of the figure.

    One row = one altitude gate.

    Column blocks:
        IRI_*
        RUN1_*
        RUN2_*
        RUN3_*

    RUN#_panel stores the actual run label so long run-label strings do not
    become unwieldy CSV column names.

    Missing panels remain present, with numerical values left blank (NaN).
    """
    stat_cols = [
        "median_fp_diff_mhz",
        "q25_fp_diff_mhz",
        "q75_fp_diff_mhz",
        "#_of_hr",
        "#_of_hr_fp_diff_le_0p5",
        "pct_fp_diff_le_0p5",
    ]

    panel_inputs = [
        (
            "IRI",
            "IRI",
            iri_prior,
        ),
    ]

    for run_index, (
        run_label,
        posterior_case,
    ) in enumerate(
        posterior_cases,
        start=1,
    ):
        panel_inputs.append(
            (
                f"RUN{run_index}",
                run_label,
                posterior_case,
            )
        )

    stats_by_prefix = {}
    altitude_arrays = []

    for (
        prefix,
        panel_label,
        sub,
    ) in panel_inputs:
        stats = fp_diff_stats(
            sub
        )

        if not stats.empty:
            stats = stats[
                [
                    "altitude_km",
                    "median_fp_diff_mhz",
                    "q25_fp_diff_mhz",
                    "q75_fp_diff_mhz",
                    "#_of_hr",
                    "#_of_hr_fp_diff_le_0p5",
                    "pct_fp_diff_le_0p5",
                ]
            ].copy()

            altitude_arrays.append(
                stats[
                    "altitude_km"
                ].to_numpy(
                    dtype=float
                )
            )

        stats_by_prefix[
            prefix
        ] = (
            panel_label,
            stats,
        )

    # Use the union of altitude gates from every plotted panel.
    if altitude_arrays:
        all_altitudes = np.unique(
            np.concatenate(
                altitude_arrays
            )
        )

        wide = pd.DataFrame(
            {
                "altitude_km": np.sort(
                    all_altitudes
                )
            }
        )
    else:
        wide = pd.DataFrame(
            {
                "altitude_km": pd.Series(
                    dtype=float
                )
            }
        )

    # Build each panel horizontally, following figure order.
    for (
        prefix,
        _panel_label,
        _sub,
    ) in panel_inputs:
        panel_label, stats = (
            stats_by_prefix[
                prefix
            ]
        )

        wide[
            f"{prefix}_panel"
        ] = panel_label

        if stats.empty:
            for col in stat_cols:
                wide[
                    f"{prefix}_{col}"
                ] = np.nan

            continue

        renamed = stats.rename(
            columns={
                col: f"{prefix}_{col}"
                for col in stat_cols
            }
        )

        keep_cols = [
            "altitude_km",
            *[
                f"{prefix}_{col}"
                for col in stat_cols
            ],
        ]

        wide = wide.merge(
            renamed[
                keep_cols
            ],
            on="altitude_km",
            how="left",
            sort=True,
        )

    # Explicit horizontal column order, like the legacy reference CSV.
    ordered_cols = [
        "altitude_km",
    ]

    for (
        prefix,
        _panel_label,
        _sub,
    ) in panel_inputs:
        ordered_cols.extend(
            [
                f"{prefix}_panel",
                f"{prefix}_median_fp_diff_mhz",
                f"{prefix}_q25_fp_diff_mhz",
                f"{prefix}_q75_fp_diff_mhz",
                f"{prefix}_#_of_hr",
                f"{prefix}_#_of_hr_fp_diff_le_0p5",
                f"{prefix}_pct_fp_diff_le_0p5",
            ]
        )

    wide = wide[
        ordered_cols
    ]

    # Figure-identification columns remain at the far left.
    wide.insert(
        0,
        "condition",
        condition,
    )
    wide.insert(
        0,
        "style",
        style,
    )
    wide.insert(
        0,
        "station",
        station,
    )

    # Formatting consistent with the legacy wide CSV.
    wide[
        "altitude_km"
    ] = pd.to_numeric(
        wide[
            "altitude_km"
        ],
        errors="coerce",
    ).round(2)

    for (
        prefix,
        _panel_label,
        _sub,
    ) in panel_inputs:
        for col in (
            f"{prefix}_median_fp_diff_mhz",
            f"{prefix}_q25_fp_diff_mhz",
            f"{prefix}_q75_fp_diff_mhz",
            f"{prefix}_pct_fp_diff_le_0p5",
        ):
            wide[
                col
            ] = pd.to_numeric(
                wide[
                    col
                ],
                errors="coerce",
            ).round(2)

        for col in (
            f"{prefix}_#_of_hr",
            f"{prefix}_#_of_hr_fp_diff_le_0p5",
        ):
            wide[
                col
            ] = pd.to_numeric(
                wide[
                    col
                ],
                errors="coerce",
            ).astype(
                "Int64"
            )

    return wide


def analyze_run_label(
    run_label,
    tomography_output_root: Path = DEFAULT_TOMOGRAPHY_OUTPUT_ROOT,
    isr_data: Path = DEFAULT_ISR_DIR,
):
    """Analyze one to three run labels.

    The first run label owns the output directory and supplies the IRI prior
    panel. Every run label contributes one tomography posterior panel.
    """
    run_labels = normalize_run_labels(run_label)
    primary_run_label = run_labels[0]

    tomography_output_root = Path(tomography_output_root).expanduser().resolve()
    isr_data = Path(isr_data).expanduser().resolve()

    run_dirs = {}
    for label in run_labels:
        run_dir = tomography_output_root / label
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Run-label directory not found: {run_dir}"
            )
        run_dirs[label] = run_dir

    output_root = run_dirs[primary_run_label] / "plasmafreq_analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    dual_raw_by_label = {}
    pooled_raw_primary = None

    for label in run_labels:
        run_dir = run_dirs[label]
        dual_files, pooled_files = discover_run_files(run_dir)

        print(f"Run: {label}")
        print(f"Run directory: {run_dir}")
        print(f"dual_isr_errors.csv files: {len(dual_files)}")
        print(
            "cross-style pooled_rmse.csv files: "
            f"{len(pooled_files)}"
        )

        if not dual_files:
            raise FileNotFoundError(
                "No hour-level dual_isr_errors.csv files "
                f"found under {run_dir}"
            )

        dual_raw = load_dual_files(dual_files)
        dual_raw["run_label"] = label
        dual_raw_by_label[label] = dual_raw

        if label == primary_run_label:
            if not pooled_files:
                raise FileNotFoundError(
                    "No cross-style pooled_rmse.csv files "
                    f"found under {run_dir}"
                )
            pooled_raw_primary = load_pooled_files(
                pooled_files,
                run_dir,
                dual_raw,
            )

    if pooled_raw_primary is None:
        raise RuntimeError("Primary run pooled-RMSE data were not loaded.")

    station_codes = set(
        pooled_raw_primary["station"].astype(str).str.upper()
    )
    for dual_raw in dual_raw_by_label.values():
        station_codes.update(
            dual_raw["station"].astype(str).str.upper()
        )
    station_codes = sorted(station_codes)

    isr_by_station = {}
    for station in station_codes:
        filename = ISR_FILENAMES.get(station)
        if filename is None:
            print(f"WARNING: unknown ISR station code {station}; skipping")
            continue
        path = isr_data / filename
        if not path.is_file():
            print(f"WARNING: ISR file missing for {station}: {path}")
            continue
        isr_by_station[station] = load_isr_dataset(path)
        print(
            f"{station}: "
            f"{len(isr_by_station[station]['time_utc'])} profiles, "
            f"{len(isr_by_station[station]['altitude_km'])} gates"
        )

    prior = add_fp_difference(
        pooled_raw_primary,
        "rmse_forecast_m3",
        isr_by_station,
    )

    posterior_by_label = {}
    for label in run_labels:
        posterior_by_label[label] = add_fp_difference(
            dual_raw_by_label[label],
            "analysis_error_m3",
            isr_by_station,
        )

    master_summary = build_master_fp_summary(
        prior,
        posterior_by_label,
        primary_run_label,
    )
    master_summary.to_csv(
        output_root / "fp_diff_percentiles_vs_altitude_all_results.csv",
        index=False,
    )

    style_union = set()
    for posterior in posterior_by_label.values():
        style_union.update(posterior["style"].astype(str))
    posterior_styles = canonical_style_order(style_union)

    figure_count = 0
    per_figure_csv_count = 0

    for condition in CONDITION_ORDER:
        figure_dir = output_root / condition / "fp_diff_comparisons"
        csv_dir = output_root / condition / "fp_diff_percentiles_by_figure"
        figure_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        for station in station_codes:
            prior_case_root = prior[
                (prior["station"] == station)
                & (prior["condition"] == condition)
            ].copy()

            iri_source_style = choose_iri_style_for_case(prior_case_root)
            if iri_source_style is not None:
                iri_prior = prior_case_root[
                    prior_case_root["style"] == iri_source_style
                ].copy()
                print(
                    f"{station} | {condition}: middle IRI panel uses "
                    f"pooled-RMSE style={iri_source_style}"
                )
            else:
                iri_prior = prior.iloc[0:0].copy()
                print(
                    f"{station} | {condition}: no non-ANCHOR pooled-RMSE "
                    "prior available; middle IRI panel will show No Data"
                )

            for style in posterior_styles:
                posterior_cases = []
                for label in run_labels:
                    posterior = posterior_by_label[label]
                    posterior_case = posterior[
                        (posterior["station"] == station)
                        & (posterior["condition"] == condition)
                        & (posterior["style"] == style)
                    ].copy()
                    posterior_cases.append((label, posterior_case))

                if (
                    iri_prior.empty
                    and all(case.empty for _, case in posterior_cases)
                ):
                    continue

                stem = (
                    f"{station}_{style}_{condition}_"
                    f"compare_{len(run_labels)}run"
                )

                plot_comparison_panels(
                    iri_prior=iri_prior,
                    posterior_cases=posterior_cases,
                    station=station,
                    style=style,
                    condition=condition,
                    output_file=figure_dir / f"{stem}_fp_diff.png",
                )
                figure_count += 1

                fig_csv = build_per_figure_fp_csv(
                    iri_prior=iri_prior,
                    posterior_cases=posterior_cases,
                    station=station,
                    style=style,
                    condition=condition,
                    iri_source_style=iri_source_style,
                )
                if not fig_csv.empty:
                    fig_csv.to_csv(
                        csv_dir / f"{stem}_fp_diff_percentiles.csv",
                        index=False,
                    )
                    per_figure_csv_count += 1

    print()
    print("Complete.")
    print("Compared run labels: " + " | ".join(run_labels))
    print(f"Output owner: {primary_run_label}")
    print(f"Output: {output_root}")
    print(f"Plasma-frequency-difference figures: {figure_count}")
    print(f"Per-figure fp percentile CSVs: {per_figure_csv_count}")

    return output_root



def main():
    args = parse_args()

    analyze_run_label(
        run_label=(
            args.run_label
        ),
        tomography_output_root=(
            args.tomography_output_root
        ),
        isr_data=(
            args.isr_data
        ),
    )


if __name__ == "__main__":
    main()
