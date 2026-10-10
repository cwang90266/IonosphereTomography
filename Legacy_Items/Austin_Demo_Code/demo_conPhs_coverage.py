#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_conPhs_coverage.py — podTc2 / conPhs pairing coverage demonstration.

Scans a day's podTc2 files, attempts to find a matching conPhs file for each,
and writes a single diagnostic figure:

  • 2×2 constellation TEC panels (GPS / Galileo / GLONASS / BeiDou):
      Solid lines  = absolute TEC from podTc2.
      Dashed lines = relative TEC from conPhs, bias-shifted so the profile
                     aligns at the top of the conPhs altitude range.
      Grey colour  = no paired conPhs found.
      Constellation colour = paired occultation.

  • Globe map (Robinson projection):
      Filled circles  = podTc2 + conPhs paired, colour by constellation.
      Grey  ×  marks  = podTc2 only (no conPhs match).

Run from the project root:
    python demo_conPhs_coverage.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import gc
import io
import contextlib

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import netCDF4
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from TEC_model.conPhs_file_processing import find_conPhs_for_podTc, parse_conPhs_nc_file

# ─────────────────────────────────────────────────────────────────────────────
# Constellation → colour family and 2×2 panel position  (matches demo_group.py)
# ─────────────────────────────────────────────────────────────────────────────
CONSTELLATION_CONFIG = {
    "G": {"name": "GPS",     "cmap": "Blues",   "panel": (0, 0),
          "title_color": "steelblue"},
    "R": {"name": "GLONASS", "cmap": "Purples", "panel": (1, 0),
          "title_color": "mediumpurple"},
    "E": {"name": "Galileo", "cmap": "Oranges", "panel": (0, 1),
          "title_color": "darkorange"},
    "C": {"name": "BeiDou",  "cmap": "Greens",  "panel": (1, 1),
          "title_color": "seagreen"},
}
_FALLBACK_CMAP = "Greys"

_ECEF_TO_LL = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# §A  Metadata scan
# ─────────────────────────────────────────────────────────────────────────────

def scan_podtc2_metadata(base_path: str, file_suffix: str = ".0001_nc") -> pd.DataFrame:
    """
    Read NC *attributes only* from every podTc2 file in base_path.
    Returns a DataFrame with one row per file.
    """
    rows  = []
    files = sorted(f for f in os.listdir(base_path) if f.endswith(file_suffix))
    print(f"  Scanning {len(files)} podTc2 files …")

    for fname in files:
        fpath = os.path.join(base_path, fname)
        try:
            with netCDF4.Dataset(fpath, "r") as nc:
                lat  = float(nc.getncattr("lat_tecmax_tangent"))
                lon  = float(nc.getncattr("lon_tecmax_tangent"))
                yr   = int(nc.getncattr("year"))
                mo   = int(nc.getncattr("month"))
                dy   = int(nc.getncattr("day"))
                hh   = int(nc.getncattr("hour"))
                mm   = int(nc.getncattr("minute"))
                ss   = int(nc.getncattr("second"))
                conid = str(nc.getncattr("conid")).strip()
                prn   = str(nc.getncattr("prn_id")).strip()
        except Exception:
            continue

        if abs(lat) > 90:
            continue

        dt = pd.Timestamp(yr, mo, dy, hh, mm, ss)
        rows.append({
            "filename":  fname,
            "full_path": fpath,
            "date":      dt,
            "lat":       lat,
            "lon":       lon,
            "spacecraft": fname.split(".")[0].replace("podTc2_", ""),
            "constellation": conid,
            "prn":        prn,
        })

    meta = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"  Found {len(meta)} valid podTc2 files.")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# §B  Per-occultation data loading
# ─────────────────────────────────────────────────────────────────────────────

def _shift_rel_tec(pod_alt: np.ndarray, pod_tec: np.ndarray,
                   con_alt: np.ndarray, con_tec: np.ndarray) -> np.ndarray:
    """
    Bias-shift the conPhs relative TEC so it aligns with the podTc2 absolute
    TEC at the top of the conPhs altitude range.

    The conPhs relative TEC is in descending height order (highest altitude
    first); the top of its range is con_alt[0].  We interpolate the podTc2
    absolute TEC at that height and compute the offset.
    """
    top_h = float(con_alt[0])

    # pod_alt is also descending; np.interp requires ascending x
    pod_alt_asc = pod_alt[::-1]
    pod_tec_asc = pod_tec[::-1]
    pod_at_top  = float(np.interp(top_h, pod_alt_asc, pod_tec_asc,
                                  left=np.nan, right=np.nan))
    if np.isnan(pod_at_top):
        return con_tec   # can't align — return as-is

    offset = pod_at_top - float(con_tec[0])
    return con_tec + offset


def load_occultation(row: pd.Series,
                     conPhs_base_dir: str | None,
                     time_window_min: float = 15.0) -> dict | None:
    """
    Parse podTc2 data and (if available) the matching conPhs data for one row
    from the metadata DataFrame.

    Returns a dict with keys:
        filename, lat, lon, date, spacecraft, constellation, prn
        pod_alt      : tangent altitude array (km), descending
        pod_tec      : absolute TEC array (TECU)
        con_alt      : conPhs tangent altitude (km) or None
        con_tec      : conPhs relative TEC, bias-shifted (TECU) or None
        has_conPhs   : bool
    """
    pod = parse_podTc2_nc_file(row["full_path"])
    if pod is None:
        return None

    # Compute tangent altitudes (parse_podTc2_nc_file already validates geometry)
    _, _, tang_raw = rayTangent(pod["LEO"], pod["GNSS"], units="km")
    pod_alt = tang_raw * 1e-3
    pod_tec = pod.get("TEC_podTc2", pod.get("TEC", np.zeros_like(pod_alt)))

    valid = np.isfinite(pod_tec) & (pod_alt > 0)
    if not np.any(valid):
        return None

    pod_alt = pod_alt[valid]
    pod_tec = pod_tec[valid]

    # ── conPhs ────────────────────────────────────────────────────────────────
    con_alt = None
    con_tec = None

    conphs_path = find_conPhs_for_podTc(
        row["full_path"], conPhs_base_dir, time_window_min
    )
    if conphs_path is not None:
        con = parse_conPhs_nc_file(conphs_path)
        if con is not None:
            con_alt = con["tangent_alt_km"]
            con_tec = _shift_rel_tec(pod_alt, pod_tec, con_alt, con["rel_TEC"])

    return {
        "filename":     row["filename"],
        "lat":          row["lat"],
        "lon":          row["lon"],
        "date":         row["date"],
        "spacecraft":   row["spacecraft"],
        "constellation": row["constellation"],
        "prn":          row["prn"],
        "pod_alt":      pod_alt,
        "pod_tec":      pod_tec,
        "con_alt":      con_alt,
        "con_tec":      con_tec,
        "has_conPhs":   con_alt is not None,
    }


def _load_worker(args: tuple) -> tuple[int, dict | None]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    idx, row_dict, conPhs_base_dir, time_window_min = args
    with contextlib.redirect_stdout(io.StringIO()):
        entry = load_occultation(pd.Series(row_dict), conPhs_base_dir, time_window_min)
    return idx, entry


# ─────────────────────────────────────────────────────────────────────────────
# §C  Coverage figure
# ─────────────────────────────────────────────────────────────────────────────

def _constellation_color(const: str, idx_within: int, n_total: int) -> tuple:
    """Return an RGBA colour from the constellation colormap."""
    cfg      = CONSTELLATION_CONFIG.get(const, {})
    cmap     = mpl.colormaps.get_cmap(cfg.get("cmap", _FALLBACK_CMAP))
    n_total  = max(n_total, 1)
    t        = 0.40 + 0.55 * (idx_within / max(n_total - 1, 1))
    return cmap(t)


def plot_coverage(
    entries:    list[dict],
    save_path:  str,
    day_label:  str = "",
    alt_max_km: float = 600.0,
) -> str:
    """
    Write the 4-panel constellation TEC + globe coverage figure.

    Parameters
    ----------
    entries     : list of dicts from load_occultation (None entries are skipped).
    save_path   : output PNG path.
    day_label   : text for the figure suptitle (e.g. "2024-05-10 DOY 131").
    alt_max_km  : upper y-limit for all TEC panels.

    Returns
    -------
    str — the path to the saved figure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    valid   = [e for e in entries if e is not None]
    paired  = [e for e in valid if e["has_conPhs"]]
    n_total = len(valid)
    n_pair  = len(paired)

    # ── Count occultations per constellation (for shade scaling) ────────────
    const_occs: dict[str, list[dict]] = defaultdict(list)
    for e in valid:
        const_occs[e["constellation"]].append(e)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9))
    fig.suptitle(
        f"podTc2 / conPhs Coverage  —  {day_label}\n"
        f"{n_total} podTc2 occultations  |  {n_pair} paired with conPhs  "
        f"({n_pair / max(n_total, 1) * 100:.0f}%)\n"
        f"Solid = podTc2 absolute TEC   |   Dashed = conPhs relative TEC "
        f"(bias-shifted at overlap top)",
        fontsize=11,
    )

    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1, 1, 1.6],
                  wspace=0.40, hspace=0.50)

    # ── TEC panel axes (fixed constellation positions) ────────────────────────
    _POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}
    ax_tec:   dict[str, plt.Axes] = {}
    first_ax = None
    for const, (row, col) in _POS.items():
        cfg = CONSTELLATION_CONFIG[const]
        ax  = fig.add_subplot(gs[row, col],
                              sharey=first_ax if first_ax is not None else None)
        ax.set_title(cfg["name"], fontsize=9,
                     color=cfg["title_color"], fontweight="bold")
        ax.grid(True, alpha=0.3, ls=":")
        ax_tec[const] = ax
        if first_ax is None:
            first_ax = ax

    # ── Globe axis ────────────────────────────────────────────────────────────
    ax_globe = fig.add_subplot(gs[:, 2],
                               projection=ccrs.Robinson())
    ax_globe.set_global()
    ax_globe.add_feature(cfeature.LAND,  facecolor="whitesmoke", zorder=0)
    ax_globe.add_feature(cfeature.OCEAN, facecolor="aliceblue",  zorder=0)
    ax_globe.add_feature(
        cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="gray"
    )
    ax_globe.gridlines(lw=0.3, alpha=0.4)

    # ── Plot TEC profiles ─────────────────────────────────────────────────────
    const_idx: dict[str, int] = defaultdict(int)

    for e in valid:
        const    = e["constellation"]
        n_con    = len(const_occs[const])
        idx      = const_idx[const]
        col_pair = _constellation_color(const, idx, n_con)
        col_pod  = col_pair if e["has_conPhs"] else (0.75, 0.75, 0.75, 0.6)

        ax = ax_tec.get(const) or ax_tec["G"]

        # podTc2 absolute TEC
        ax.plot(e["pod_tec"], e["pod_alt"],
                color=col_pod, lw=0.9, alpha=0.7, zorder=3)

        # conPhs relative TEC (bias-shifted)
        if e["has_conPhs"]:
            ax.plot(e["con_tec"], e["con_alt"],
                    color=col_pair, lw=0.9, alpha=0.85,
                    ls="--", zorder=4)

        const_idx[const] += 1

    # ── Apply axis labels, limits, and style legend ───────────────────────────
    style_handles = [
        Line2D([0], [0], color="dimgray", lw=1.6,
               label="podTc2 (absolute TEC)"),
        Line2D([0], [0], color="dimgray", lw=1.6, ls="--",
               label="conPhs (relative TEC, shifted)"),
        Line2D([0], [0], color=(0.75, 0.75, 0.75), lw=1.6,
               label="podTc2 only (no conPhs)"),
    ]
    style_placed = False

    for const, ax in ax_tec.items():
        entries_c = const_occs.get(const, [])
        n_paired_c = sum(1 for e in entries_c if e["has_conPhs"])

        if entries_c:
            if not style_placed:
                ax.legend(handles=style_handles, fontsize=7,
                          loc="upper right", framealpha=0.88)
                style_placed = True
            else:
                ax.text(0.97, 0.97,
                        f"{len(entries_c)} occs\n{n_paired_c} paired",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="black",
                        bbox=dict(fc="white", ec="none", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "No data",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color="lightgray", fontsize=11, style="italic")

        ax.set_ylim(0, alt_max_km)
        if const in ("G", "R"):
            ax.set_ylabel("Tangent Altitude (km)")
        else:
            ax.tick_params(labelleft=False)
        if const in ("R", "C"):
            ax.set_xlabel("TEC (TECU)")

    # ── Globe scatter ─────────────────────────────────────────────────────────
    # Unpaired first (grey × ), then paired on top (coloured ●)
    unpaired = [e for e in valid if not e["has_conPhs"]]
    if unpaired:
        ax_globe.scatter(
            [e["lon"] for e in unpaired],
            [e["lat"] for e in unpaired],
            marker="x", s=30, c="gray", linewidths=0.8,
            transform=ccrs.Geodetic(), zorder=4, label="podTc2 only",
        )

    for const in ("G", "R", "E", "C"):
        paired_c = [e for e in paired if e["constellation"] == const]
        if not paired_c:
            continue
        cfg   = CONSTELLATION_CONFIG[const]
        cmap  = mpl.colormaps.get_cmap(cfg["cmap"])
        n_c   = len(paired_c)
        colors = [cmap(0.40 + 0.55 * (i / max(n_c - 1, 1))) for i in range(n_c)]
        ax_globe.scatter(
            [e["lon"] for e in paired_c],
            [e["lat"] for e in paired_c],
            c=colors, s=45, marker="o",
            edgecolors="black", linewidths=0.4,
            transform=ccrs.Geodetic(), zorder=5,
            label=f"{cfg['name']} ({n_c})",
        )

    ax_globe.set_title(
        "TEC-max tangent points\n"
        "● = podTc2 + conPhs   × = podTc2 only",
        fontsize=9,
    )
    ax_globe.legend(loc="lower left", fontsize=8, framealpha=0.85,
                    markerscale=1.2)

    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCoverage figure saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# §D  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def demo_conPhs_coverage_main() -> None:
    """
    Scan a day's podTc2 files, pair each with the best-matching conPhs file
    (if one exists), and write the coverage figure.
    """
    import datetime
    
    DOY = 284
    YYYY = 2024
    podTc_base_path   = f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/{YYYY}.{DOY}/"
    conPhs_base_dir   = None    # None → auto-discover from podTc path
    time_window_min   = 45.0    # max start-time offset (minutes) for pairing
    num_workers       = 12      # parallel workers for loading
    save_dir          = "./Figures/conPhs_coverage/"
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.isdir(podTc_base_path):
        print(f"ERROR: podTc_base_path not found: {podTc_base_path}")
        return

    print("=" * 65)
    print("  demo_conPhs_coverage.py — podTc2 / conPhs pairing demo")
    print("=" * 65)

    # ── Step 1: Scan metadata ─────────────────────────────────────────────────
    meta = scan_podtc2_metadata(podTc_base_path)
    if meta.empty:
        print("No valid podTc2 files found.  Exiting.")
        return

    # ── Step 2: Derive day label from the directory name ──────────────────────
    dir_name = podTc_base_path.rstrip("/").split("/")[-1]
    # Accept both "YYYY-MM-DD" and "YYYY.DDD" formats
    try:
        batch_date = pd.Timestamp(dir_name)          # "2024-05-10"
    except Exception:
        yr, doy = map(int, dir_name.split("."))      # "2024.131"
        batch_date = pd.Timestamp(
            datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)
        )
    day_label = f"{batch_date.strftime('%Y-%m-%d')}  (DOY {batch_date.dayofyear})"
    print(f"\nBatch date: {day_label}")
    print(f"Pairing window: ±{time_window_min} min")

    # ── Step 3: Load podTc2 + attempt conPhs pairing for each file ───────────
    print(f"\nLoading occultation data ({num_workers} workers) …")
    tasks   = [
        (i, row.to_dict(), conPhs_base_dir, time_window_min)
        for i, (_, row) in enumerate(meta.iterrows())
    ]
    entries = [None] * len(tasks)
    n_paired_so_far = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_load_worker, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), unit="file", ncols=80,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for future in as_completed(futures):
                idx, entry = future.result()
                entries[idx] = entry
                if entry is not None and entry["has_conPhs"]:
                    n_paired_so_far += 1
                pbar.set_postfix(paired=n_paired_so_far, refresh=False)
                pbar.update(1)

    valid   = [e for e in entries if e is not None]
    paired  = [e for e in valid if e["has_conPhs"]]
    print(f"\n  Loaded  : {len(valid)} / {len(meta)} occultations")
    print(f"  Paired  : {len(paired)} have a conPhs match")
    print(f"  Missing : {len(valid) - len(paired)} podTc2-only")

    if not valid:
        print("No data to plot.  Exiting.")
        return

    # ── Step 4: Write coverage figure ─────────────────────────────────────────
    safe_day   = batch_date.strftime("%Y%m%d")
    save_path  = os.path.join(save_dir, f"conPhs_coverage_{safe_day}.png")

    plot_coverage(valid, save_path, day_label=day_label)

    # ── Step 5: Print summary ──────────────────────────────────────────────────
    print("\nPairing summary by constellation:")
    for const, cfg in CONSTELLATION_CONFIG.items():
        c_all    = [e for e in valid   if e["constellation"] == const]
        c_paired = [e for e in c_all   if e["has_conPhs"]]
        if c_all:
            print(f"  {cfg['name']:8s}: {len(c_paired):>3}/{len(c_all):>3} paired "
                  f"({len(c_paired)/len(c_all)*100:.0f}%)")
    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_conPhs_coverage_main()
