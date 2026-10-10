#!/usr/bin/env python3
"""
Generate per-window OCC sweep summaries from checkpoints without waiting for full run.

Usage:
    python generate_window_summary.py --window-key "2025-08-27_0120" --checkpoint-dir "./Data/checkpoints/" --save-dir "./Figures/test_param_iono/"
    python generate_window_summary.py --date 2025.239 --hhmm 0120
"""

import os
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

METHOD_STYLE = {
    "prior":     dict(color="gray",      label="Prior (IRI)"),
    "kf":        dict(color="steelblue", label="KF posterior"),
    "ekf_param": dict(color="seagreen",  label="EKF_Param posterior"),
}

# One observation configuration per filter run (test_param_iono.py's
# FILTER_MODES); distinguished by linestyle/marker so each figure can show
# RO-only, IGS-only, and RO+IGS side by side instead of a single "primary"
# mode picked by ro_igs > ro_only > igs_only fallback.
MODE_STYLE = {
    "ro_only":  dict(linestyle="-",  marker="o"),
    "ro_igs":   dict(linestyle="--", marker="s"),
    "igs_only": dict(linestyle=":",  marker="^"),
}
MODE_LABEL = {
    "ro_only":  "RO only",
    "ro_igs":   "RO+IGS",
    "igs_only": "IGS only",
}
MODES = ("ro_only", "ro_igs", "igs_only")

# Fallback station list/coords — mirrors test_param_iono.py's IGS_SIM_STATIONS
# / IGS_SIM_STATIONS_JSON. Used so the station-map panel can still show
# station locations from IGSNetwork.json even when a checkpoint predates the
# "igs_stations"/per-mode station_edp_errors fields (or an analysis failed
# for a given bin), instead of skipping the whole figure.
IGS_SIM_STATIONS      = ["TRO1", "WUTH", "NYA1", "KIR0", "SOD3", "ALRT", "SCOR", "HOFN", "REYK"]
IGS_SIM_STATIONS_JSON = "./Data/IGS_Stations/IGSNetwork.json"


def _load_station_coords_fallback(codes: list, json_path: str = IGS_SIM_STATIONS_JSON) -> dict:
    """Resolve {code: {"code", "lat", "lon"}} from IGSNetwork.json by 4-char
    prefix match, with no ROI gating — a lighter-weight standalone version of
    test_param_iono.py's _load_igs_sim_stations for when checkpoints don't
    carry station geometry themselves."""
    try:
        with open(json_path, "r") as fh:
            network = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    codes_upper = [c.upper() for c in codes]
    resolved: dict = {}
    for entry_key, entry in network.items():
        prefix = entry_key[:4].upper()
        if prefix not in codes_upper or prefix in resolved:
            continue
        try:
            lat = float(entry["Latitude"])
            lon = float(entry["Longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        resolved[prefix] = {"code": prefix, "lat": lat, "lon": lon}
    return resolved


def _extract_summary(data: dict) -> dict:
    """
    Pull out only the small summary fields plots need, so the multi-GB
    checkpoint dict (parsed_list, truth_arcs, full EDP/covariance arrays,
    ...) can be dropped immediately instead of held for every bin at once.
    """
    if "error" in data:
        return {"error": data["error"]}

    filter_results = data.get("filter_results") or {}
    rmse_by_mode = {}
    for mode in MODES:
        fr = filter_results.get(mode)
        if not fr:
            continue
        kf_res  = fr.get("kf_result") or {}
        ekf_res = fr.get("ekf_param") or {}
        rmse_by_mode[mode] = {
            "prior_rmse":    fr.get("prior_rmse"),
            "kf_post_rmse":  kf_res.get("post_rmse"),
            "ekf_post_rmse": ekf_res.get("post_rmse"),
        }

    return {
        "rmse_by_mode":          rmse_by_mode,
        # Nested {mode: {...}} as produced by test_param_iono.py's
        # _process_time_window_with_arc_subset; None/missing if that mode
        # had no result for this bin.
        "station_edp_errors":    data.get("station_edp_errors"),
        "hf_reflection_errors":  data.get("hf_reflection_errors"),
        "critical_frequencies":  data.get("critical_frequencies"),
        "ro_tangent_points":     data.get("ro_tangent_points"),
        "igs_stations":          data.get("igs_stations"),
        "igs_ipp_points":        data.get("igs_ipp_points"),
    }


def load_window_checkpoints(window_key: str, checkpoint_dir: str) -> dict:
    """
    Load all bin checkpoints for a single window, one file at a time —
    each multi-GB JSON is parsed, reduced to a small summary dict, and
    dropped before the next file is read, so peak RSS stays at ~1 file's
    size instead of all bins summed (which previously drove the process
    into swap).
    """
    import gc

    window_results = {}
    checkpoint_path = Path(checkpoint_dir)

    for checkpoint_file in sorted(checkpoint_path.glob(f"{window_key}_*.json")):
        if checkpoint_file.name.endswith(".meta.json"):
            continue
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            parts = checkpoint_file.stem.split("_")
            bin_count = parts[-1]
            if bin_count == "None":
                bin_count = None
            else:
                try:
                    bin_count = int(bin_count)
                except ValueError:
                    bin_count = None
            window_results[bin_count] = _extract_summary(data)
            del data
            gc.collect()
            print(f"  Loaded {checkpoint_file.name} (bin={bin_count})")
        except Exception as e:
            print(f"  [warn] Failed to load {checkpoint_file.name}: {e}")

    return window_results


def _style_axes(ax) -> None:
    ax.set_facecolor("#2b2b2b")
    ax.tick_params(colors="lightgray", labelsize=8)
    ax.grid(True, alpha=0.3, color="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


def _sorted_bins(window_results: dict) -> list:
    return sorted([k for k in window_results.keys() if k is not None])


def plot_occ_sweep_summary(window_results: dict, window_key: str, save_dir: str, hhmm: str) -> None:
    """TEC RMSE (prior/KF/EKF_Param) vs. occultation-count bin, one line per
    (method, observation mode) — RO only / RO+IGS / IGS only."""
    bin_counts = _sorted_bins(window_results)
    if not bin_counts:
        print("  [skip] tec rmse: no valid bins")
        return

    field_by_method = {"prior": "prior_rmse", "kf": "kf_post_rmse", "ekf_param": "ekf_post_rmse"}
    series = {mode: {m: [] for m in field_by_method} for mode in MODES}
    for bin_count in bin_counts:
        result = window_results[bin_count]
        rmse_by_mode = {} if "error" in result else (result.get("rmse_by_mode") or {})
        for mode in MODES:
            mode_vals = rmse_by_mode.get(mode) or {}
            for method, field in field_by_method.items():
                series[mode][method].append(mode_vals.get(field, np.nan))

    if all(np.all(np.isnan(series[mode][m])) for mode in MODES for m in field_by_method):
        print("  [skip] tec rmse: all-NaN")
        return

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    _style_axes(ax)

    for mode in MODES:
        for method in ("prior", "kf", "ekf_param"):
            vals = series[mode][method]
            if np.all(np.isnan(vals)):
                continue
            ax.plot(bin_counts, vals, lw=1.8, markersize=6,
                     color=METHOD_STYLE[method]["color"],
                     linestyle=MODE_STYLE[mode]["linestyle"],
                     marker=MODE_STYLE[mode]["marker"],
                     label=f"{METHOD_STYLE[method]['label']} — {MODE_LABEL[mode]}")

    ax.set_xlabel("Occultation Count Bin", color="lightgray", fontsize=11)
    ax.set_ylabel("TEC RMSE (TECU)", color="lightgray", fontsize=11)
    ax.set_title(f"OCC Sweep: TEC Retrieval Convergence — {window_key} ({hhmm})",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#2b2b2b", labelcolor="lightgray",
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    save_path = Path(save_dir) / "per_window" / hhmm / f"occ_sweep_tec_rmse_{window_key}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, facecolor="#1e1e1e", edgecolor="none", bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def _pick_geometry_bin(window_results: dict, bin_counts: list) -> "dict | None":
    """
    Pick one bin's result to source station/tangent-point/IPP geometry from
    for the station-map overlay — prefer the full-arc bin (bin_count=None,
    "bin_all"), else the largest occultation-count bin available, since both
    give the most complete point cloud.
    """
    if None in window_results and "error" not in window_results[None]:
        return window_results[None]
    for bc in sorted(bin_counts, reverse=True):
        result = window_results[bc]
        if "error" not in result and result.get("igs_stations"):
            return result
    return window_results[bin_counts[-1]] if bin_counts else None


def _plot_station_geometry_map(ax, geometry: "dict | None", stations: list,
                                station_by_code: dict) -> None:
    """Cartopy map: station markers + transparent RO tangent-point / IGS IPP
    scatter, matching the globe-plot convention used elsewhere in the
    pipeline (test_param_iono.py's TEC-max tangent-point markers).

    station_by_code : {code: {"lat", "lon"}}, resolved by the caller — from
    checkpoint geometry when available, else from IGSNetwork.json — so
    stations still render even when a checkpoint carries no RO/IGS point
    cloud (older checkpoints, or a bin where geometry capture failed)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ro_points      = (geometry or {}).get("ro_tangent_points") or {}
    igs_ipp_points = (geometry or {}).get("igs_ipp_points") or {}

    st_lats = [station_by_code[c]["lat"] for c in stations if c in station_by_code]
    st_lons = [station_by_code[c]["lon"] for c in stations if c in station_by_code]

    all_lats = list(st_lats) + list(ro_points.get("lat") or []) + list(igs_ipp_points.get("lat") or [])
    all_lons = list(st_lons) + list(ro_points.get("lon") or []) + list(igs_ipp_points.get("lon") or [])
    cen_lat = float(np.mean(all_lats)) if all_lats else 0.0
    cen_lon = float(np.mean(all_lons)) if all_lons else 0.0

    proj = ccrs.Orthographic(central_longitude=cen_lon, central_latitude=cen_lat)
    fig = ax.figure
    subplotspec = ax.get_subplotspec()
    ax.remove()
    ax = fig.add_subplot(subplotspec, projection=proj)
    ax.set_facecolor("#2b2b2b")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#aaaaaa")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor="#888888")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

    legend_handles = []
    if ro_points.get("lat"):
        ax.scatter(ro_points["lon"], ro_points["lat"], s=6, color="orange", alpha=0.15,
                    transform=ccrs.PlateCarree(), zorder=2)
        legend_handles.append(Line2D([0], [0], color="orange", marker="o", ms=5,
                                      linestyle="none", label="RO tangent / TEC-max pts"))
    if igs_ipp_points.get("lat"):
        ax.scatter(igs_ipp_points["lon"], igs_ipp_points["lat"], s=6, color="mediumpurple",
                    alpha=0.15, transform=ccrs.PlateCarree(), zorder=3)
        legend_handles.append(Line2D([0], [0], color="mediumpurple", marker="o", ms=5,
                                      linestyle="none", label="IGS IPP points"))
    if st_lats:
        ax.scatter(st_lons, st_lats, s=50, color="white", edgecolors="black",
                    linewidth=0.8, marker="^", transform=ccrs.PlateCarree(), zorder=5)
        for code in stations:
            st = station_by_code.get(code)
            if st is None:
                continue
            ax.text(st["lon"] + 1.0, st["lat"] + 1.0, code, color="white", fontsize=6,
                     transform=ccrs.PlateCarree(), zorder=6)
        legend_handles.append(Line2D([0], [0], color="white", marker="^", ms=6,
                                      linestyle="none", markeredgecolor="black",
                                      label="IGS station"))

    ax.set_title("Station / RO / IGS geometry", color="white", fontsize=9, fontweight="bold")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=6,
                   facecolor="#2b2b2b", labelcolor="lightgray")


def plot_station_edp_errors_vs_occ(window_results: dict, window_key: str, save_dir: str, hhmm: str) -> None:
    """
    Per-station EDP RMSE (below F2 peak) vs. occultation-count bin, one line
    per (method, observation mode) — RO only / RO+IGS / IGS only — plus a
    station/RO-tangent-point/IGS-IPP location map.
    Source: result["station_edp_errors"][mode][station][method]["rmse"].
    """
    bin_counts = _sorted_bins(window_results)
    if not bin_counts:
        print("  [skip] station edp errors: no valid bins")
        return

    # Collect the union of stations that appear in any bin/mode's error data.
    stations: list = []
    for bc in bin_counts:
        see_by_mode = window_results[bc].get("station_edp_errors") or {}
        for mode in MODES:
            for st in (see_by_mode.get(mode) or {}).keys():
                if st not in stations:
                    stations.append(st)

    have_error_data = bool(stations)
    if not stations:
        # No station_edp_errors in any checkpoint (older checkpoint format,
        # or the analysis failed for every bin) — still show the station
        # locations via the known IGS_SIM_STATIONS list so the map isn't
        # skipped just because the error-vs-bin lines have nothing to plot.
        print("  [warn] station edp errors: no station_edp_errors in checkpoints — "
              "falling back to IGS_SIM_STATIONS for the location map only")
        stations = list(IGS_SIM_STATIONS)

    # Resolve station coordinates: prefer whatever a checkpoint captured
    # (igs_stations, added alongside per-mode station_edp_errors), fall back
    # to IGSNetwork.json for any station code not covered by that.
    geometry = _pick_geometry_bin(window_results, bin_counts)
    station_by_code = {s["code"]: s for s in (geometry or {}).get("igs_stations") or []}
    missing_codes = [c for c in stations if c not in station_by_code]
    if missing_codes:
        station_by_code.update(_load_station_coords_fallback(missing_codes))

    stations = [s for s in stations if s in station_by_code] or stations
    if not station_by_code:
        print("  [skip] station edp errors: no station coordinates resolvable "
              f"(checkpoint geometry empty and {IGS_SIM_STATIONS_JSON} unavailable)")
        return

    n_panels = len(stations) + 1  # +1 for the geometry map
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.6 * n_rows),
                              facecolor="#1e1e1e", squeeze=False)
    fig.patch.set_facecolor("#1e1e1e")

    for i, station in enumerate(stations):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        _style_axes(ax)

        any_line = False
        if have_error_data:
            for mode in MODES:
                for method in ("prior", "kf", "ekf_param"):
                    vals = []
                    for bc in bin_counts:
                        see_by_mode = window_results[bc].get("station_edp_errors") or {}
                        st_entry = (see_by_mode.get(mode) or {}).get(station) or {}
                        m_entry = st_entry.get(method) or {}
                        vals.append(m_entry.get("rmse", np.nan))
                    if np.all(np.isnan(vals)):
                        continue
                    any_line = True
                    ax.plot(bin_counts, vals, lw=1.4, markersize=4,
                             color=METHOD_STYLE[method]["color"],
                             linestyle=MODE_STYLE[mode]["linestyle"],
                             marker=MODE_STYLE[mode]["marker"])
        if not any_line:
            ax.text(0.5, 0.5, "no error data\nin checkpoints", color="lightgray",
                     fontsize=8, ha="center", va="center", transform=ax.transAxes)

        ax.set_title(station, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("OCC bin", color="lightgray", fontsize=7)
        ax.set_ylabel("EDP RMSE (m⁻³)", color="lightgray", fontsize=7)

    # Geometry map in the next open slot.
    map_idx = len(stations)
    row, col = divmod(map_idx, n_cols)
    _plot_station_geometry_map(axes[row][col], geometry, stations, station_by_code)

    # Hide any remaining unused axes.
    for j in range(n_panels, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis("off")

    legend_handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], lw=2, label=METHOD_STYLE[m]["label"])
        for m in ("prior", "kf", "ekf_param")
    ] + [
        Line2D([0], [0], color="lightgray", lw=1.6, linestyle=MODE_STYLE[mode]["linestyle"],
               marker=MODE_STYLE[mode]["marker"], ms=5, label=MODE_LABEL[mode])
        for mode in MODES
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, fontsize=8,
               facecolor="#2b2b2b", labelcolor="lightgray", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Per-station EDP RMSE vs. occultation count — {window_key} ({hhmm})",
                 color="white", fontsize=13, fontweight="bold", y=1.07)
    fig.tight_layout()

    save_path = Path(save_dir) / "per_window" / hhmm / f"occ_sweep_station_edp_{window_key}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, facecolor="#1e1e1e", edgecolor="none", bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_hf_reflection_errors_vs_occ(window_results: dict, window_key: str, save_dir: str, hhmm: str) -> None:
    """
    HF reflection-height error (mean_height_error_km) vs. occultation-count bin,
    one subplot per probe frequency, one line per (method, observation mode).
    Source: result["hf_reflection_errors"][mode][freq_str][method]["mean_height_error_km"].
    """
    bin_counts = _sorted_bins(window_results)
    if not bin_counts:
        print("  [skip] hf reflection errors: no valid bins")
        return

    freqs: list = []
    for bc in bin_counts:
        hre_by_mode = window_results[bc].get("hf_reflection_errors") or {}
        for mode in MODES:
            for f in (hre_by_mode.get(mode) or {}).keys():
                if f not in freqs:
                    freqs.append(f)
    # Sort numerically (keys are strings after JSON round-trip).
    freqs = sorted(freqs, key=lambda f: float(f))

    if not freqs:
        print("  [skip] hf reflection errors: no hf_reflection_errors in checkpoints")
        return

    n_f = len(freqs)
    n_cols = 3
    n_rows = int(np.ceil(n_f / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows),
                              facecolor="#1e1e1e", squeeze=False)
    fig.patch.set_facecolor("#1e1e1e")

    for i, freq in enumerate(freqs):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        _style_axes(ax)

        for mode in MODES:
            for method in ("prior", "kf", "ekf_param"):
                vals = []
                for bc in bin_counts:
                    hre_by_mode = window_results[bc].get("hf_reflection_errors") or {}
                    f_entry = (hre_by_mode.get(mode) or {}).get(freq) or {}
                    m_entry = f_entry.get(method) or {}
                    vals.append(m_entry.get("mean_height_error_km", np.nan))
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(bin_counts, vals, lw=1.4, markersize=4,
                         color=METHOD_STYLE[method]["color"],
                         linestyle=MODE_STYLE[mode]["linestyle"],
                         marker=MODE_STYLE[mode]["marker"])

        ax.set_title(f"{float(freq):.0f} MHz", color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("OCC bin", color="lightgray", fontsize=7)
        ax.set_ylabel("Height error (km)", color="lightgray", fontsize=7)

    for j in range(n_f, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis("off")

    legend_handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], lw=2, label=METHOD_STYLE[m]["label"])
        for m in ("prior", "kf", "ekf_param")
    ] + [
        Line2D([0], [0], color="lightgray", lw=1.6, linestyle=MODE_STYLE[mode]["linestyle"],
               marker=MODE_STYLE[mode]["marker"], ms=5, label=MODE_LABEL[mode])
        for mode in MODES
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, fontsize=8,
               facecolor="#2b2b2b", labelcolor="lightgray", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"HF reflection-height error vs. occultation count — {window_key} ({hhmm})",
                 color="white", fontsize=13, fontweight="bold", y=1.07)
    fig.tight_layout()

    save_path = Path(save_dir) / "per_window" / hhmm / f"occ_sweep_hf_reflection_{window_key}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, facecolor="#1e1e1e", edgecolor="none", bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_critical_freq_errors_vs_occ(window_results: dict, window_key: str, save_dir: str, hhmm: str) -> None:
    """
    foF2 / foE RMSE (MHz) vs. occultation-count bin, one line per (method,
    observation mode).
    Source: result["critical_frequencies"][mode]["foF2"/"foE"][method]["rmse_mhz"].
    """
    bin_counts = _sorted_bins(window_results)
    if not bin_counts:
        print("  [skip] critical frequencies: no valid bins")
        return

    layers = ["foF2", "foE"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")

    any_data = False
    for i, layer in enumerate(layers):
        ax = axes[i]
        _style_axes(ax)
        for mode in MODES:
            for method in ("prior", "kf", "ekf_param"):
                vals = []
                for bc in bin_counts:
                    cf_by_mode = window_results[bc].get("critical_frequencies") or {}
                    l_entry = (cf_by_mode.get(mode) or {}).get(layer) or {}
                    m_entry = l_entry.get(method) or {}
                    v = m_entry.get("rmse_mhz", np.nan)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        any_data = True
                    vals.append(v)
                if np.all([pd.isna(v) for v in vals]):
                    continue
                ax.plot(bin_counts, vals, lw=1.6, markersize=5,
                         color=METHOD_STYLE[method]["color"],
                         linestyle=MODE_STYLE[mode]["linestyle"],
                         marker=MODE_STYLE[mode]["marker"])

        ax.set_title(f"{layer} RMSE", color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Occultation Count Bin", color="lightgray", fontsize=9)
        ax.set_ylabel("RMSE (MHz)", color="lightgray", fontsize=9)

    if not any_data:
        print("  [skip] critical frequencies: all-NaN / missing")
        plt.close(fig)
        return

    legend_handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], lw=2, label=METHOD_STYLE[m]["label"])
        for m in ("prior", "kf", "ekf_param")
    ] + [
        Line2D([0], [0], color="lightgray", lw=1.6, linestyle=MODE_STYLE[mode]["linestyle"],
               marker=MODE_STYLE[mode]["marker"], ms=5, label=MODE_LABEL[mode])
        for mode in MODES
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, fontsize=9,
               facecolor="#2b2b2b", labelcolor="lightgray", bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(f"Critical-frequency error vs. occultation count — {window_key} ({hhmm})",
                 color="white", fontsize=13, fontweight="bold", y=1.13)
    fig.tight_layout()

    save_path = Path(save_dir) / "per_window" / hhmm / f"occ_sweep_critical_freq_{window_key}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, facecolor="#1e1e1e", edgecolor="none", bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-window OCC sweep summaries from checkpoints"
    )
    parser.add_argument("--window-key", type=str, help="Window key (e.g., '2025-08-27_0120')")
    parser.add_argument("--date", type=str, help="Date as YYYY.DOY (e.g., 2025.239)")
    parser.add_argument("--hhmm", type=str, help="Window time as HHMM (e.g., 0120)")
    parser.add_argument("--checkpoint-dir", type=str, default="./Data/checkpoints/",
                        help="Checkpoint directory")
    parser.add_argument("--save-dir", type=str, default="./Figures/test_param_iono/",
                        help="Output directory for figures")

    args = parser.parse_args()

    if args.window_key:
        window_key = args.window_key
        hhmm = window_key.split("_")[1] if "_" in window_key else "0000"
    elif args.date and args.hhmm:
        parts = args.date.split(".")
        year = int(parts[0])
        doy = int(parts[1])
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        window_key = f"{date.strftime('%Y-%m-%d')}_{args.hhmm}"
        hhmm = args.hhmm
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nGenerating per-window summary for {window_key}…")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Save dir:       {args.save_dir}")
    print()

    window_results = load_window_checkpoints(window_key, args.checkpoint_dir)

    if not window_results:
        print("  [error] No checkpoints found for this window")
        sys.exit(1)

    print(f"\nLoaded {len(window_results)} bin(s)")
    print(f"Generating plots…")

    plot_occ_sweep_summary(window_results, window_key, args.save_dir, hhmm)
    plot_station_edp_errors_vs_occ(window_results, window_key, args.save_dir, hhmm)
    plot_hf_reflection_errors_vs_occ(window_results, window_key, args.save_dir, hhmm)
    plot_critical_freq_errors_vs_occ(window_results, window_key, args.save_dir, hhmm)

    print(f"\n✓ Summary complete: {args.save_dir}")


if __name__ == "__main__":
    main()
