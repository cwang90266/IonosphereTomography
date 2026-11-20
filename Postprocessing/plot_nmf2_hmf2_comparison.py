#!/usr/bin/env python3
"""Visualizes the NmF2/hmF2 table from ``tabulate_nmf2_hmf2.py``, grouped
by configuration columns (parameterization style, boost condition, ROI
radius, etc. -- 2026-11-20 user request for statistical comparison across
many one-hour runs, since a single hour isn't meaningful on its own).

For each ``--group-by`` column, produces one figure with four box plots
(forecast/analysis x NmF2 relative error / hmF2 absolute error), using the
same derived-error convention as ``Run_Scoring/score_runs.py``'s
``summary_peaks.csv``: ``dNmF2_pct = 100*(NmF2_tag/NmF2_isr - 1)``,
``dhmF2_km = hmF2_tag - hmF2_isr``. Rows without a matched ISR value are
excluded from these plots (reported as a count, not silently dropped).
Style-grouped plots reuse ``Assimilation_Cycle.output._color_for_style``
so colors match every other cross-style plot in this repo.

Usage::

    python3 Postprocessing/plot_nmf2_hmf2_comparison.py \\
        --csv /local/path/nmf2_hmf2_table.csv \\
        --out-dir /local/path/nmf2_hmf2_plots \\
        [--group-by style condition roi_radius_km]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from Assimilation_Cycle.output import _color_for_style

DEFAULT_GROUP_BY = ["style", "condition", "roi_radius_km"]
_METRICS = [
    ("dNmF2_forecast_pct", "forecast", "NmF2 rel. error (%)"),
    ("dNmF2_analysis_pct", "analysis", "NmF2 rel. error (%)"),
    ("dhmF2_forecast_km", "forecast", "hmF2 abs. error (km)"),
    ("dhmF2_analysis_km", "analysis", "hmF2 abs. error (km)"),
]


def add_derived_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["dNmF2_forecast_pct"] = 100.0 * (df["NmF2_forecast"] / df["NmF2_isr"] - 1.0)
        df["dNmF2_analysis_pct"] = 100.0 * (df["NmF2_analysis"] / df["NmF2_isr"] - 1.0)
    df["dhmF2_forecast_km"] = df["hmF2_forecast"] - df["hmF2_isr"]
    df["dhmF2_analysis_km"] = df["hmF2_analysis"] - df["hmF2_isr"]
    return df


def plot_grouped_comparison(df: pd.DataFrame, group_col: str, figsize=(11, 8)):
    """One figure, four box-plot panels (forecast/analysis x NmF2%/hmF2km),
    grouped by ``group_col``'s distinct values -- mirrors the groupby
    -and-plot pattern every other cross-style plot in this repo already
    uses (e.g. ``isr_comparison.plot_pooled_isr_rmse_cross_style``)."""
    groups = sorted(df[group_col].dropna().unique(), key=str)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (col, tag, ylabel) in zip(axes.ravel(), _METRICS):
        data = [df.loc[df[group_col] == g, col].dropna().to_numpy() for g in groups]
        bp = ax.boxplot(data, tick_labels=[str(g) for g in groups], showmeans=True)
        if group_col == "style":
            for patch, g in zip(bp["boxes"], groups):
                patch.set_color(_color_for_style(str(g)))
        ax.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_title(tag)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"NmF2/hmF2 vs. ISR, grouped by {group_col} (n={len(df)})")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path, help="Output of tabulate_nmf2_hmf2.py")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--group-by", nargs="*", default=DEFAULT_GROUP_BY)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = add_derived_error_columns(df)

    n_total = len(df)
    n_with_isr = int(df["NmF2_isr"].notna().sum())
    print(f"[plot] {n_total} rows total, {n_with_isr} with a matched ISR value "
          f"({n_total - n_with_isr} excluded from these plots)")
    df_isr = df[df["NmF2_isr"].notna()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for group_col in args.group_by:
        if group_col not in df_isr.columns:
            print(f"[plot]   skip --group-by {group_col!r}: not a column in {args.csv}")
            continue
        sub = df_isr[df_isr[group_col].notna()]
        if sub.empty:
            print(f"[plot]   skip --group-by {group_col!r}: no rows with a non-null value")
            continue
        fig = plot_grouped_comparison(sub, group_col)
        out_path = args.out_dir / f"nmf2_hmf2_by_{group_col}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot]   wrote {out_path} ({sub[group_col].nunique()} groups, {len(sub)} rows)")


if __name__ == "__main__":
    main()
