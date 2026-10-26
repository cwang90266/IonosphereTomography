"""
Standalone CSV processing + plotting for the ISR data-assimilation metrics
table written by demo_isr_da_comparison.py (Data/DA_Cache/isr_metrics.csv).

Intended for interactive use in Spyder: open this file and run cell-by-cell
(Ctrl+Enter on each "# %%" block), or run the whole file (F5) to load the
CSV and pop up the summary figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% Configuration ------------------------------------------------------------

ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "Data" / "DA_Cache" / "isr_metrics.csv"
DEFAULT_SAVE_DIR = ROOT / "Figures" / "ISR_DA_Comparison" / "isr_metrics_summary"

OBS_MODE_ORDER = ["igs_only", "ro_igs", "ro_only"]
FILTER_ORDER = ["gridded_kf", "parametric_ekf"]
FILTER_COLORS = {"gridded_kf": "#2166ac", "parametric_ekf": "#d7191c"}
FILTER_LABELS = {"gridded_kf": "Gridded KF", "parametric_ekf": "Parametric EKF"}
OBS_MODE_LABELS = {"igs_only": "IGS only", "ro_igs": "RO+IGS", "ro_only": "RO only"}


# %% Loading --------------------------------------------------------------

def load_isr_metrics(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load the ISR DA-comparison metrics CSV and order the group columns."""
    df = pd.read_csv(csv_path, parse_dates=["date", "t_centre"])
    present_obs = [m for m in OBS_MODE_ORDER if m in df["obs_mode"].unique()]
    present_filt = [f for f in FILTER_ORDER if f in df["filter_type"].unique()]
    df["obs_mode"] = pd.Categorical(df["obs_mode"], categories=present_obs, ordered=True)
    df["filter_type"] = pd.Categorical(df["filter_type"], categories=present_filt, ordered=True)
    return df


# %% Plot helpers -------------------------------------------------------------

def _group_order(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    obs_modes = [m for m in OBS_MODE_ORDER if m in df["obs_mode"].unique()]
    filters = [f for f in FILTER_ORDER if f in df["filter_type"].unique()]
    return obs_modes, filters


def _paired_boxplot(ax, df: pd.DataFrame, prior_col: str, post_col: str,
                     ylabel: str, title: str) -> None:
    """One box pair (prior=faded, posterior=solid) per (obs_mode, filter_type)."""
    obs_modes, filters = _group_order(df)
    box_w, pair_gap, filt_gap, cluster_gap = 0.6, 0.12, 0.45, 1.1

    positions, data, colors, alphas = [], [], [], []
    tick_pos, tick_labels = [], []
    x = 0.0
    for obs_mode in obs_modes:
        cluster_start = x
        for filt in filters:
            sub = df[(df["obs_mode"] == obs_mode) & (df["filter_type"] == filt)]
            for col, alpha in ((prior_col, 0.35), (post_col, 0.9)):
                vals = sub[col].dropna().to_numpy()
                positions.append(x)
                data.append(vals if vals.size else np.array([np.nan]))
                colors.append(FILTER_COLORS[filt])
                alphas.append(alpha)
                x += box_w + pair_gap
            x += filt_gap - pair_gap
        tick_pos.append((cluster_start + x - filt_gap - box_w) / 2)
        tick_labels.append(OBS_MODE_LABELS[obs_mode])
        x += cluster_gap

    bp = ax.boxplot(data, positions=positions, widths=box_w, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.3))
    for patch, color, alpha in zip(bp["boxes"], colors, alphas):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor(color)
    for line, color in zip(bp["whiskers"], np.repeat(colors, 2)):
        line.set_color(color)
        line.set_alpha(0.8)
    for line, color in zip(bp["caps"], np.repeat(colors, 2)):
        line.set_color(color)
        line.set_alpha(0.8)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _single_boxplot(ax, df: pd.DataFrame, col: str, ylabel: str, title: str,
                     zero_line: bool = False) -> None:
    """One box per (obs_mode, filter_type), colored by filter_type."""
    obs_modes, filters = _group_order(df)
    box_w, filt_gap, cluster_gap = 0.55, 0.2, 1.0

    positions, data, colors = [], [], []
    tick_pos, tick_labels = [], []
    x = 0.0
    for obs_mode in obs_modes:
        cluster_start = x
        for filt in filters:
            sub = df[(df["obs_mode"] == obs_mode) & (df["filter_type"] == filt)]
            vals = sub[col].dropna().to_numpy()
            positions.append(x)
            data.append(vals if vals.size else np.array([np.nan]))
            colors.append(FILTER_COLORS[filt])
            x += box_w + filt_gap
        tick_pos.append((cluster_start + x - filt_gap - box_w) / 2)
        tick_labels.append(OBS_MODE_LABELS[obs_mode])
        x += cluster_gap - filt_gap

    bp = ax.boxplot(data, positions=positions, widths=box_w, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.3))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)

    if zero_line:
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _add_filter_legend(fig, prior_post: bool) -> None:
    handles = [
        mpatches.Patch(facecolor=FILTER_COLORS[f], alpha=0.85,
                        edgecolor=FILTER_COLORS[f], label=FILTER_LABELS[f])
        for f in FILTER_ORDER
    ]
    if prior_post:
        handles.append(mpatches.Patch(facecolor="gray", alpha=0.35, edgecolor="gray", label="Prior"))
        handles.append(mpatches.Patch(facecolor="gray", alpha=0.9, edgecolor="gray", label="Posterior"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def _rmse_vs_occultations(ax, df: pd.DataFrame, prior_col: str, post_col: str,
                           ylabel: str, title: str, x_col: str = "n_ro_occultations",
                           xlim: tuple[float, float] | None = None,
                           ylim: tuple[float, float] | None = None) -> None:
    """Scatter of RO-only prior/posterior RMSE vs occultation count, with a
    per-filter_type mean-trend scatter overlay (larger diamond markers).
    Hollow markers = prior, filled markers = posterior.

    *xlim*/*ylim* optionally clip the axis bounds, e.g. to drop outliers
    that would otherwise dominate the scale.
    """
    ro_df = df[df["obs_mode"] == "ro_only"]
    filters = [f for f in FILTER_ORDER if f in ro_df["filter_type"].unique()]
    jitter = {filt: (i - (len(filters) - 1) / 2) * 0.2 for i, filt in enumerate(filters)}

    for filt in filters:
        sub = ro_df[ro_df["filter_type"] == filt].dropna(subset=[x_col, prior_col, post_col])
        if sub.empty:
            continue
        color = FILTER_COLORS[filt]
        x = sub[x_col].to_numpy(dtype=float) + jitter[filt]

        ax.scatter(x, sub[prior_col], facecolors="none", edgecolors=color,
                   marker="o", s=30, alpha=0.55, linewidths=1.1, zorder=3)
        ax.scatter(x, sub[post_col], facecolors=color, edgecolors=color,
                   marker="o", s=30, alpha=0.85, zorder=4)

        prior_mean = sub.groupby(x_col)[prior_col].mean().sort_index()
        post_mean = sub.groupby(x_col)[post_col].mean().sort_index()
        ax.scatter(prior_mean.index + jitter[filt], prior_mean.to_numpy(),
                   facecolors="none", edgecolors=color, marker="D", s=90,
                   linewidths=1.8, alpha=0.9, zorder=5)
        ax.scatter(post_mean.index + jitter[filt], post_mean.to_numpy(),
                   facecolors=color, edgecolors="black", marker="D", s=90,
                   linewidths=0.8, alpha=0.95, zorder=6)

    ax.set_xlabel("Number of RO occultations")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _add_occultation_legend(fig) -> None:
    handles = [
        mpatches.Patch(facecolor=FILTER_COLORS[f], alpha=0.85,
                        edgecolor=FILTER_COLORS[f], label=FILTER_LABELS[f])
        for f in FILTER_ORDER
    ]
    handles.append(plt.scatter([], [], facecolors="none", edgecolors="gray",
                                marker="o", s=30, linewidths=1.1, label="Prior (raw)"))
    handles.append(plt.scatter([], [], facecolors="gray", edgecolors="gray",
                                marker="o", s=30, label="Posterior (raw)"))
    handles.append(plt.scatter([], [], facecolors="none", edgecolors="gray",
                                marker="D", s=90, linewidths=1.8, label="Prior (mean)"))
    handles.append(plt.scatter([], [], facecolors="gray", edgecolors="black",
                                marker="D", s=90, label="Posterior (mean)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))


# %% Main plotting function ----------------------------------------------------

def plot_isr_metrics_summary(df: pd.DataFrame | None = None,
                              csv_path: str | Path = DEFAULT_CSV,
                              save_dir: str | Path = DEFAULT_SAVE_DIR,
                              save: bool = True,
                              show: bool = True,
                              ro_occult_xlim: tuple[float, float] | None = None,
                              ro_occult_tec_ylim: tuple[float, float] | None = (0, 20),
                              ro_occult_edp_ylim: tuple[float, float] | None = None,
                              ) -> dict[str, plt.Figure]:
    """
    Build summary figures from the ISR DA-comparison metrics CSV, grouped by
    obs_mode (data type: igs_only / ro_igs / ro_only) and filter_type
    (gridded_kf / parametric_ekf).

    Figure 1 ("prior_vs_posterior"): 2x2 grid of prior-vs-posterior box pairs
    for TEC RMSE, EDP RMSE, NmF2 error, hmF2 error.

    Figure 2 ("pct_improvement"): TEC RMSE and EDP RMSE percent improvement.

    Figure 3 ("ro_only_vs_noccult"): RO-only TEC RMSE and EDP RMSE (prior and
    posterior) as a function of the number of RO occultations in the window.
    *ro_occult_xlim* clips the shared x-axis (occultation count) on both
    panels; *ro_occult_tec_ylim*/*ro_occult_edp_ylim* clip the y-axis of the
    TEC and EDP panels respectively. Leave as None for auto-scaling.

    Pass an already-loaded DataFrame via *df* to skip re-reading the CSV
    (e.g. when iterating on plots inside Spyder).
    """
    if df is None:
        df = load_isr_metrics(csv_path)

    figs: dict[str, plt.Figure] = {}

    # ---- Figure 1: prior vs posterior error metrics -------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
    _paired_boxplot(axes[0, 0], df, "prior_tec_rmse", "post_tec_rmse",
                     "TEC RMSE (TECU)", "TEC RMSE: Prior vs Posterior")
    _paired_boxplot(axes[0, 1], df, "prior_edp_rmse", "post_edp_rmse",
                     r"EDP RMSE (N$_e$, m$^{-3}$)", "EDP RMSE: Prior vs Posterior")
    _paired_boxplot(axes[1, 0], df, "prior_NmF2_err_pct", "post_NmF2_err_pct",
                     "NmF2 error (%)", "NmF2 Error: Prior vs Posterior")
    _paired_boxplot(axes[1, 1], df, "prior_hmF2_err_km", "post_hmF2_err_km",
                     "hmF2 error (km)", "hmF2 Error: Prior vs Posterior")
    for ax in axes.flat:
        ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.4)
    fig1.suptitle("ISR Data-Assimilation Metrics — Prior vs Posterior",
                  fontsize=13, fontweight="bold")
    _add_filter_legend(fig1, prior_post=True)
    fig1.tight_layout(rect=(0, 0.05, 1, 0.96))
    figs["prior_vs_posterior"] = fig1

    # ---- Figure 2: percent-improvement metrics -------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    _single_boxplot(axes2[0], df, "tec_rmse_pct_improvement",
                     "TEC RMSE improvement (%)", "TEC RMSE % Improvement", zero_line=True)
    _single_boxplot(axes2[1], df, "edp_rmse_pct_improvement",
                     "EDP RMSE improvement (%)", "EDP RMSE % Improvement", zero_line=True)
    fig2.suptitle("ISR Data-Assimilation Metrics — Posterior Improvement over Prior",
                  fontsize=13, fontweight="bold")
    _add_filter_legend(fig2, prior_post=False)
    fig2.tight_layout(rect=(0, 0.08, 1, 0.93))
    figs["pct_improvement"] = fig2

    # ---- Figure 3: RO-only RMSE vs number of occultations --------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    _rmse_vs_occultations(axes3[0], df, "prior_tec_rmse", "post_tec_rmse",
                           "TEC RMSE (TECU)", "RO-only TEC RMSE vs # Occultations",
                           xlim=ro_occult_xlim, ylim=ro_occult_tec_ylim)
    _rmse_vs_occultations(axes3[1], df, "prior_edp_rmse", "post_edp_rmse",
                           r"EDP RMSE (N$_e$, m$^{-3}$)", "RO-only EDP RMSE vs # Occultations",
                           xlim=ro_occult_xlim, ylim=ro_occult_edp_ylim)
    fig3.suptitle("ISR Data-Assimilation Metrics — RO-only RMSE vs Occultation Count",
                  fontsize=13, fontweight="bold")
    _add_occultation_legend(fig3)
    fig3.tight_layout(rect=(0, 0.09, 1, 0.93))
    figs["ro_only_vs_noccult"] = fig3

    if save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            out_path = save_dir / f"isr_metrics_{name}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[isr_metrics] saved {out_path}")

    if show:
        plt.show()

    return figs


# %% Run as script --------------------------------------------------------------
if __name__ == "__main__":
    metrics_df = load_isr_metrics()
    print(f"[isr_metrics] loaded {len(metrics_df)} rows from {DEFAULT_CSV}")
    plot_isr_metrics_summary(metrics_df)
