#!/usr/bin/env python3.11
"""
Diagnose the EKF Fibonacci-sphere ROI grid in isolation.

This reproduces EXACTLY the ROI + grid that demo_isr_da_comparison.py's EKF
uses for a given window -- built from the podTc occultation tangent points and
the IGS ground-station pierce points -- then draws every stage of
`_fibonacci_roi_grid()` so you can see WHY the footprint looks the way it does.

It answers the questions the "Horizontal Correlation" map cannot:
  * Where are the RO tangent extrema and IGS pierce points (the ROI anchors)?
  * Where does the centroid land, and how big is the enclosing great-circle disk?
  * Is a single stray anchor inflating the radius (the classic near-pole tongue)?
  * After the max_pts cap, do the kept points still surround the data, or has
    the "densest core" slid off to one side?

Nothing here writes to the DA cache or the OUTPUT/OUTPUT_FINAL folders; it only
reads the measurement data and saves one diagnostic PNG. Re-run it freely while
editing `_fibonacci_roi_grid()` -- pass --ekf-grid-km / --margin-km / --max-pts
to try parameter changes without touching the source, or --module-fn to test
your edited `_fibonacci_roi_grid` directly.

Examples
--------
  # Best (most-occultation) window of Sept 22, default 100 km / cap 600:
  python3.11 tools/diagnose_fibonacci_grid.py --date 2025-09-22

  # A specific window, sweep a couple of parameter choices:
  python3.11 tools/diagnose_fibonacci_grid.py --date 2025-09-22 --window 1115 \
      --ekf-grid-km 100 --margin-km 200 --max-pts 600

  # Compare the real module function's output against these params:
  python3.11 tools/diagnose_fibonacci_grid.py --date 2025-09-22 --module-fn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# The tool lives in tools/; the demo modules are one level up.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the demo's OWN data-loading + ROI + grid helpers so this is a faithful
# reproduction of what the EKF sees -- not a re-implementation that could drift.
from demo_isr_da_comparison import (          # noqa: E402
    select_priority_days,
    build_minima_windows_for_day,
    load_igs_for_day,
    _filter_igs_cmp,
    _collapse_igs_arc_to_central_epoch,
    _ro_extrema_points,
    _fibonacci_roi_grid,
    _gate_ro_anchors_to_footprint,
    _robust_sphere_centroid,
    _latlon_unit_vectors,
    _fibonacci_sphere_latlon,
    _EARTH_RADIUS_KM,
    EKF_ROI_GATE_KM,
    EKF_GRID_KM,
    EKF_GRID_MAX_PTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (great-circle math, mirrors _fibonacci_roi_grid internals)
# ─────────────────────────────────────────────────────────────────────────────
def _unit_to_latlon(v: np.ndarray) -> tuple[float, float]:
    """3-D unit vector -> (lat_deg, lon_deg)."""
    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[1], v[0]))
    return float(lat), float(lon)


def _great_circle_km(lat_deg, lon_deg, c_lat, c_lon) -> np.ndarray:
    """Great-circle distance (km) from each (lat,lon) to a single centre."""
    V = _latlon_unit_vectors(np.asarray(lat_deg, float), np.asarray(lon_deg, float))
    c = _latlon_unit_vectors([c_lat], [c_lon])[0]
    ang = np.arccos(np.clip(V @ c, -1.0, 1.0))
    return _EARTH_RADIUS_KM * ang


def _circle_latlon(c_lat, c_lon, radius_km, n=361):
    """Sample a great-circle circle of *radius_km* around (c_lat,c_lon)."""
    ang = radius_km / _EARTH_RADIUS_KM                    # angular radius (rad)
    brng = np.radians(np.linspace(0, 360, n))
    lat1 = np.radians(c_lat)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang) +
                     np.cos(lat1) * np.sin(ang) * np.cos(brng))
    lon2 = np.radians(c_lon) + np.arctan2(
        np.sin(brng) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 180) % 360 - 180


def _recompute_internals(roi_lats, roi_lons, spacing_km, margin_km, max_pts,
                         radius_percentile=95.0, trim_iters=3, trim_percentile=90.0):
    """Re-run _fibonacci_roi_grid's math step-by-step so we can print/plot it.

    Mirrors the robust (Fix 2) version: trimmed unit centroid + percentile
    radius, so the diagnostic stays a faithful reproduction of the module.
    """
    roi_lats = np.asarray(roi_lats, float)
    roi_lons = np.asarray(roi_lons, float)
    ok = np.isfinite(roi_lats) & np.isfinite(roi_lons)
    roi_lats, roi_lons = roi_lats[ok], roi_lons[ok]

    V = _latlon_unit_vectors(roi_lats, roi_lons)
    centroid = _robust_sphere_centroid(
        V, trim_iters=trim_iters, trim_percentile=trim_percentile)
    c_lat, c_lon = _unit_to_latlon(centroid)

    cos_to_roi = np.clip(V @ centroid, -1.0, 1.0)
    ang_to_roi = np.arccos(cos_to_roi)
    anchor_dist_km = _EARTH_RADIUS_KM * ang_to_roi
    pct_ang = float(np.percentile(ang_to_roi, float(radius_percentile)))
    max_ang = float(ang_to_roi.max())
    radius_km = _EARTH_RADIUS_KM * pct_ang + float(margin_km)

    n_global = max(int(round(4.0 * np.pi * _EARTH_RADIUS_KM ** 2 /
                             float(spacing_km) ** 2)), 60)
    flat, flon = _fibonacci_sphere_latlon(n_global)
    F = _latlon_unit_vectors(flat, flon)
    ang = np.arccos(np.clip(F @ centroid, -1.0, 1.0))
    dist_km = _EARTH_RADIUS_KM * ang
    keep = dist_km <= radius_km
    n_kept = int(keep.sum())

    klat, klon, kdist = flat[keep], flon[keep], dist_km[keep]
    capped = max_pts is not None and klat.size > int(max_pts)
    if capped:
        order = np.argsort(kdist)[:int(max_pts)]
        klat, klon = klat[order], klon[order]

    return dict(
        roi_lats=roi_lats, roi_lons=roi_lons,
        c_lat=c_lat, c_lon=c_lon,
        anchor_dist_km=anchor_dist_km,
        max_ang_deg=np.degrees(max_ang),
        pct_ang_deg=np.degrees(pct_ang), radius_percentile=radius_percentile,
        radius_km=radius_km,
        n_global=n_global, n_kept=n_kept, capped=capped,
        klat=klat, klon=klon,
        cap_radius_km=(float(np.sort(kdist)[int(max_pts) - 1]) if capped else radius_km),
    )


# ─────────────────────────────────────────────────────────────────────────────
def _select_window(windows, want):
    """Pick a window: --window matches hhmm or full key; else max-occultation."""
    if want:
        for w in windows:
            if w["window_key"] == want or w["window_key"].endswith(want):
                return w
        raise SystemExit(f"No window matching '{want}'. Available: "
                         + ", ".join(w["window_key"] for w in windows))
    return max(windows, key=lambda w: w["n_occ"])


def _build_roi(window, igs_arcs, gate_km=EKF_ROI_GATE_KM):
    """Reproduce _process_window_bin's ROI anchors for the FULL window.

    Applies the Fix 1 gate: RO anchors far from the IGS/ISR footprint are
    separated out (returned as ``ro_gated`` for distinct plotting), so the
    diagnostic mirrors exactly what seeds the Fibonacci grid.
    """
    t_centre = window["t_centre"]
    lo, hi = window["lo"], window["hi"]
    group_key = t_centre.strftime("%Y-%m-%d_%H%M")
    full_group_meta = window["group_meta"]

    window_width_min = (hi - lo).total_seconds() / 60.0
    igs_window_arcs = _filter_igs_cmp(igs_arcs, group_key,
                                      window_minutes=window_width_min)
    igs_window_arcs = [_collapse_igs_arc_to_central_epoch(a) for a in igs_window_arcs]

    igs_pts = [(float(a["lat_tecmax_tangent"]), float(a["lon_tecmax_tangent"]))
               for a in igs_window_arcs
               if np.isfinite(a.get("lat_tecmax_tangent", np.nan))
               and np.isfinite(a.get("lon_tecmax_tangent", np.nan))]
    ro_all = _ro_extrema_points(full_group_meta)

    ro_kept, n_drop = _gate_ro_anchors_to_footprint(ro_all, igs_pts, gate_km=gate_km)
    kept_keys = {(round(float(p[0]), 4), round(float(p[1]), 4)) for p in ro_kept}
    ro_gated = [p for p in ro_all
                if (round(float(p[0]), 4), round(float(p[1]), 4)) not in kept_keys]

    return ro_kept, ro_gated, igs_pts, group_key, n_drop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="ISR day, e.g. 2025-09-22")
    ap.add_argument("--window", default=None,
                    help="Window HHMM (e.g. 1115) or full key; default = the "
                         "window with the most occultations.")
    ap.add_argument("--ekf-grid-km", type=float, default=EKF_GRID_KM,
                    help=f"Fibonacci point spacing (km). Default {EKF_GRID_KM:.0f} "
                         "(matches the module default).")
    ap.add_argument("--margin-km", type=float, default=200.0,
                    help="Great-circle margin beyond the farthest anchor. Default 200.")
    ap.add_argument("--max-pts", type=int,
                    default=(0 if EKF_GRID_MAX_PTS is None else EKF_GRID_MAX_PTS),
                    help="Cap on kept grid points (densest core); 0 disables. "
                         f"Default {'0 (off, matches module default)' if EKF_GRID_MAX_PTS is None else EKF_GRID_MAX_PTS}.")
    ap.add_argument("--roi-gate-km", type=float, default=EKF_ROI_GATE_KM,
                    help="Great-circle gate (km): drop RO anchors farther than "
                         "this from the IGS/ISR footprint before they seed the "
                         f"grid. Default {EKF_ROI_GATE_KM:.0f}; 0 disables.")
    ap.add_argument("--module-fn", action="store_true",
                    help="Also call the real _fibonacci_roi_grid() and overplot "
                         "its output (verifies the reproduction matches).")
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Default tools/fib_grid_diag_<date>_<hhmm>.png")
    args = ap.parse_args()

    # ── Locate the day + window ─────────────────────────────────────────────
    days = select_priority_days()
    day = next((d for d in days if str(d.get("date")) == args.date), None)
    if day is None:
        raise SystemExit(f"{args.date} is not a priority ISR day. Available: "
                         + ", ".join(str(d.get('date')) for d in days))
    podtc_dir = day["podtc_dir"]
    windows = build_minima_windows_for_day(podtc_dir, day["date"])
    if not windows:
        raise SystemExit(f"No minima windows found for {args.date}.")
    igs_arcs = load_igs_for_day(pd.Timestamp(day["date"]))
    window = _select_window(windows, args.window)

    gate_km = None if args.roi_gate_km <= 0 else args.roi_gate_km
    ro_pts, ro_gated, igs_pts, group_key, n_drop = _build_roi(
        window, igs_arcs, gate_km=gate_km)
    roi_lats = [p[0] for p in ro_pts + igs_pts]
    roi_lons = [p[1] for p in ro_pts + igs_pts]

    max_pts = None if args.max_pts <= 0 else args.max_pts
    D = _recompute_internals(roi_lats, roi_lons,
                             args.ekf_grid_km, args.margin_km, max_pts)

    # ── Text diagnostics ────────────────────────────────────────────────────
    print("=" * 74)
    print(f"Fibonacci ROI grid diagnostic — {group_key}  (n_occ={window['n_occ']})")
    print("=" * 74)
    print(f"  ROI gate           : {'off' if gate_km is None else f'{gate_km:.0f} km'}"
          f"  → dropped {n_drop} stray RO anchor(s) far from IGS/ISR footprint")
    print(f"  ROI anchors        : {len(ro_pts)} RO extrema (gated) + {len(igs_pts)} IGS "
          f"= {len(D['roi_lats'])} finite")
    print(f"  Centroid (lat,lon) : ({D['c_lat']:.2f}, {D['c_lon']:.2f})  (robust/trimmed)")
    print(f"  Enclosing radius   : {D['radius_km']:.0f} km  "
          f"(p{D['radius_percentile']:.0f} anchor angle {D['pct_ang_deg']:.1f}° "
          f"+ {args.margin_km:.0f} km margin; max anchor {D['max_ang_deg']:.1f}°)")
    print(f"  Global Fib points  : {D['n_global']}  @ {args.ekf_grid_km:.0f} km spacing")
    print(f"  Kept in disk       : {D['n_kept']}"
          + (f"  → capped to {len(D['klat'])} (core radius "
             f"{D['cap_radius_km']:.0f} km)" if D['capped'] else "  (no cap hit)"))

    # Flag stray anchors that inflate the radius (the near-pole tongue cause).
    d = D["anchor_dist_km"]
    med = float(np.median(d)) if d.size else 0.0
    order = np.argsort(d)[::-1]
    print(f"  Anchor dist to centroid: median={med:.0f} km, "
          f"max={d.max():.0f} km" if d.size else "  (no anchors)")
    strays = [i for i in order if d[i] > 3.0 * max(med, 1.0)]
    if strays:
        print(f"  ⚠ {len(strays)} STRAY anchor(s) > 3× median distance "
              f"(these inflate the disk radius):")
        for i in strays[:8]:
            kind = "RO " if i < len(ro_pts) else "IGS"
            la, lo = D["roi_lats"][i], D["roi_lons"][i]
            print(f"       {kind} ({la:7.2f}, {lo:8.2f})   {d[i]:6.0f} km from centroid")
    else:
        print("  ✓ no stray anchors (all within 3× median distance)")
    print("=" * 74)

    # ── Map ─────────────────────────────────────────────────────────────────
    proj = ccrs.NorthPolarStereo(central_longitude=D["c_lon"])
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.92")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.gridlines(linewidth=0.3, color="0.7")
    pc = ccrs.PlateCarree()

    # Kept grid points (the actual EKF state grid)
    ax.scatter(D["klat"] * 0 + D["klon"], D["klat"], s=6, c="tab:green",
               alpha=0.55, transform=pc, label=f"Fib grid ({len(D['klat'])} pts)", zorder=3)
    # RO tangent extrema + IGS pierce points (the ROI anchors)
    if ro_pts:
        ax.scatter([p[1] for p in ro_pts], [p[0] for p in ro_pts], s=26,
                   c="tab:blue", edgecolor="k", linewidth=0.3, transform=pc,
                   label=f"RO extrema ({len(ro_pts)})", zorder=5)
    if igs_pts:
        ax.scatter([p[1] for p in igs_pts], [p[0] for p in igs_pts], s=34,
                   marker="^", c="tab:orange", edgecolor="k", linewidth=0.3,
                   transform=pc, label=f"IGS pierce ({len(igs_pts)})", zorder=5)
    # Gated-out RO anchors (Fix 1): shown but NOT used to seed the grid.
    if ro_gated:
        ax.scatter([p[1] for p in ro_gated], [p[0] for p in ro_gated], s=40,
                   marker="x", c="crimson", linewidth=1.4, transform=pc,
                   label=f"RO gated-out ({len(ro_gated)})", zorder=5)
    # Centroid + enclosing radius circle
    ax.scatter([D["c_lon"]], [D["c_lat"]], s=180, marker="*", c="red",
               edgecolor="k", transform=pc, label="centroid", zorder=6)
    clat, clon = _circle_latlon(D["c_lat"], D["c_lon"], D["radius_km"])
    ax.plot(clon, clat, color="red", lw=1.4, transform=pc,
            label=f"enclosing disk ({D['radius_km']:.0f} km)", zorder=4)
    if D["capped"]:
        clat2, clon2 = _circle_latlon(D["c_lat"], D["c_lon"], D["cap_radius_km"])
        ax.plot(clon2, clat2, color="purple", lw=1.2, ls="--", transform=pc,
                label=f"cap core ({D['cap_radius_km']:.0f} km)", zorder=4)

    # Optional: overplot the real module function's grid to verify reproduction.
    if args.module_fn:
        mlat, mlon = _fibonacci_roi_grid(roi_lats, roi_lons, args.ekf_grid_km,
                                         margin_km=args.margin_km, max_pts=max_pts)
        ax.scatter(mlon, mlat, s=2, c="black", alpha=0.7, transform=pc,
                   label=f"_fibonacci_roi_grid() ({len(mlat)})", zorder=2)
        print(f"  [--module-fn] real _fibonacci_roi_grid() returned {len(mlat)} pts "
              f"(reproduction has {len(D['klat'])}).")

    ax.set_title(f"Fibonacci ROI grid — {group_key}\n"
                 f"{args.ekf_grid_km:.0f} km spacing, margin {args.margin_km:.0f} km, "
                 f"cap {'off' if max_pts is None else max_pts}", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    hhmm = group_key.split("_")[-1]
    out = Path(args.out) if args.out else (_ROOT / "tools" /
                                           f"fib_grid_diag_{args.date}_{hhmm}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic map → {out}")


if __name__ == "__main__":
    main()
