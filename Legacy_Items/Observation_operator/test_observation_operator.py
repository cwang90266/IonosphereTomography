from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer

from .ray_operator import build_los_fibonacci_operator
from .forward_tec import abel_ne_state, forward_tec
from .grid_builder import great_circle_distance_km, geodesic_circle_latlon

CENTER_LAT = 69.6
CENTER_LON = 19.2
RADIUS_KM = 2000.0
SPACING_DEG = 5.0

_XFM = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


def clean_profile(p: pd.DataFrame):
    a = p["Abel_alt_km"].to_numpy(float)
    n = p["Abel_Ne"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(n)
    a, n = a[ok], n[ok]
    order = np.argsort(a)
    a, n = a[order], n[order]
    u, idx = np.unique(a, return_index=True)
    return u, n[idx]


def tangent_track_latlonalt(leo_xyz_km: np.ndarray, gnss_xyz_km: np.ndarray):
    """Return tangent-point lat/lon/alt for each RO ray."""
    leo = np.asarray(leo_xyz_km, float)
    gnss = np.asarray(gnss_xyz_km, float)
    if leo.ndim != 2 or gnss.ndim != 2:
        raise ValueError("LEO/GNSS must be 2-D arrays")
    if leo.shape[0] == 3:
        leo = leo.T
    if gnss.shape[0] == 3:
        gnss = gnss.T
    if leo.shape != gnss.shape or leo.shape[1] != 3:
        raise ValueError("LEO/GNSS must have shape (n,3) or (3,n)")

    d = gnss - leo
    denom = np.einsum("ij,ij->i", d, d)
    t = np.divide(-np.einsum("ij,ij->i", leo, d), denom, out=np.zeros(len(leo)), where=denom > 0)
    t = np.clip(t, 0.0, 1.0)
    tp = leo + t[:, None] * d
    lon, lat, alt_m = _XFM.transform(tp[:, 0] * 1000.0, tp[:, 1] * 1000.0, tp[:, 2] * 1000.0)
    return np.asarray(lat), np.asarray(lon), np.asarray(alt_m) / 1000.0


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def default_output_dir(ray_csv: Path) -> Path:
    stem = ray_csv.stem
    return Path("Figures") / "observation_operator_test" / stem


def plot_event_figure(
    out_path: Path,
    filename: str,
    op,
    meas: np.ndarray,
    fwd: np.ndarray,
    tangent_height_km: np.ndarray,
    tangent_lat: np.ndarray,
    tangent_lon: np.ndarray,
    abel_top_km: float,
):
    order = np.argsort(tangent_height_km)
    h = tangent_height_km[order]
    meas = meas[order]
    fwd = fwd[order]

    roi_lat, roi_lon = geodesic_circle_latlon(CENTER_LAT, CENTER_LON, RADIUS_KM)

    fig = plt.figure(figsize=(15.5, 6.8))

    # All geometry below is geographic longitude/latitude.
    data_crs = ccrs.PlateCarree()

    # Map centered on the tomography ROI.
    map_crs = ccrs.Orthographic(
        central_longitude=float(CENTER_LON),
        central_latitude=float(CENTER_LAT),
    )

    # ------------------------------------------------------------------
    # LEFT: geometry + active Fibonacci voxels
    # ------------------------------------------------------------------
    ax_map = fig.add_subplot(1, 2, 1, projection=map_crs)

    ax_map.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax_map.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.65, zorder=1)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.55, zorder=1)

    ax_map.gridlines(
        crs=data_crs,
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )

    # Exact 2000-km ROI boundary.
    ax_map.plot(
        roi_lon,
        roi_lat,
        color="green",
        linewidth=2.0,
        transform=data_crs,
        label=f"ROI = {int(RADIUS_KM)} km",
        zorder=3,
    )

    # Plot ONLY horizontal Fibonacci columns actually used by H.
    geo = op.grid.geolocation
    ax_map.scatter(
        geo[:, 0],
        geo[:, 1],
        s=32,
        c="k",
        alpha=0.95,
        transform=data_crs,
        label="Voxel columns used by H",
        zorder=5,
    )

    # RO tangent-point track, color-coded by tangent height.
    scat = ax_map.scatter(
        tangent_lon,
        tangent_lat,
        c=tangent_height_km,
        s=42,
        cmap="viridis",
        edgecolors="none",
        transform=data_crs,
        zorder=6,
    )
    ax_map.plot(
        tangent_lon,
        tangent_lat,
        color="red",
        linewidth=1.6,
        transform=data_crs,
        label="RO tangent track",
        zorder=5,
    )

    # ROI center.
    ax_map.scatter(
        [CENTER_LON],
        [CENTER_LAT],
        marker="*",
        s=180,
        c="k",
        transform=data_crs,
        label=f"Center ({CENTER_LAT:.1f}°, {CENTER_LON:.1f}°)",
        zorder=7,
    )

    # IMPORTANT: for Cartopy projected axes, use set_extent(..., crs=PlateCarree())
    # instead of set_xlim()/set_ylim() with degree coordinates.
    roi_lon_arr = np.asarray(roi_lon, dtype=float)
    roi_lat_arr = np.asarray(roi_lat, dtype=float)

    lon_min = np.nanmin(roi_lon_arr)
    lon_max = np.nanmax(roi_lon_arr)
    lat_min = np.nanmin(roi_lat_arr)
    lat_max = np.nanmax(roi_lat_arr)

    lon_pad = max(3.0, 0.08 * (lon_max - lon_min))
    lat_pad = max(2.0, 0.08 * (lat_max - lat_min))

    ax_map.set_extent(
        [
            lon_min - lon_pad,
            lon_max + lon_pad,
            lat_min - lat_pad,
            lat_max + lat_pad,
        ],
        crs=data_crs,
    )

    ax_map.set_title("Geometry: ROI + active Fibonacci voxels")
    ax_map.legend(loc="lower left", fontsize=9, framealpha=0.9)

    cbar = fig.colorbar(
        scat,
        ax=ax_map,
        orientation="horizontal",
        pad=0.08,
        fraction=0.055,
    )
    cbar.set_label("Tangent height (km)")

    # ------------------------------------------------------------------
    # RIGHT: measured TEC versus forward H @ Abel-Ne TEC.
    # ------------------------------------------------------------------
    ax_tec = fig.add_subplot(1, 2, 2)
    ax_tec.plot(meas, h, linewidth=2.0, label="Real measured TEC")
    ax_tec.plot(fwd, h, linewidth=2.0, label="Forwarded TEC from Abel Ne")
    ax_tec.axhline(float(abel_top_km), linestyle="--", linewidth=1.4,
                   label=f"Abel top = {abel_top_km:.0f} km")
    ax_tec.set_xlabel("TEC (TECU)")
    ax_tec.set_ylabel("Tangent height (km)")
    ax_tec.set_title("TEC comparison")
    ax_tec.grid(alpha=0.28)
    ax_tec.legend(loc="lower left", fontsize=10)

    rmse = float(np.sqrt(np.mean((fwd - meas) ** 2)))
    bias = float(np.mean(fwd - meas))
    ax_tec.text(
        0.03,
        0.97,
        f"RMSE = {rmse:.2f} TECU\nBias = {bias:.2f} TECU\nUsed voxel columns = {len(geo)}",
        transform=ax_tec.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9),
        fontsize=10,
    )

    fig.suptitle(filename)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Validate Observation_operator on all RO events in the selected time window. "
            "For each event, save one figure with geometry/voxel layout on the left and "
            "measured-vs-forwarded TEC on the right."
        )
    )
    ap.add_argument(
        "--ray-csv",
        type=Path,
        default=Path("Figures/preparation_test/RO/ro_lat69.6lon19.2_radius2000km_timewindow1hr_20251118_10001100.csv"),
    )
    ap.add_argument(
        "--abel-csv",
        type=Path,
        default=Path("Figures/preparation_test/RO/ro_lat69.6lon19.2_radius2000km_timewindow1hr_20251118_10001100_abel_profiles.csv"),
    )
    ap.add_argument("--segments", type=int, default=1000)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the 15 per-event figures and summary CSV will be saved.",
    )
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.ray_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rays = pd.read_csv(args.ray_csv)
    prof = pd.read_csv(args.abel_csv)
    rows = []

    grouped = list(rays.groupby("filename", sort=False))
    for idx, (fn, g) in enumerate(grouped, start=1):
        p = prof[prof["filename"].astype(str) == str(fn)]
        if len(p) == 0:
            raise ValueError(f"No Abel profile rows found for filename: {fn}")

        alt_km, ne = clean_profile(p)
        leo = g[["LEO_x", "LEO_y", "LEO_z"]].to_numpy(float).T
        gnss = g[["GNSS_x", "GNSS_y", "GNSS_z"]].to_numpy(float).T

        op = build_los_fibonacci_operator(
            leo,
            gnss,
            alt_km,
            spacing_deg=SPACING_DEG,
            num_segments=args.segments,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            radius_km=RADIUS_KM,
        )
        fwd = forward_tec(op.H, abel_ne_state(ne, len(op.grid.geolocation)))
        meas = g["TEC"].to_numpy(float)
        tangent_h = g["tangent_height_km"].to_numpy(float)

        tlat, tlon, _ = tangent_track_latlonalt(
            g[["LEO_x", "LEO_y", "LEO_z"]].to_numpy(float),
            g[["GNSS_x", "GNSS_y", "GNSS_z"]].to_numpy(float),
        )

        dist = great_circle_distance_km(
            op.grid.geolocation[:, 1],
            op.grid.geolocation[:, 0],
            CENTER_LAT,
            CENTER_LON,
        )

        out_fig = args.output_dir / f"{idx:02d}_{safe_name(fn)}_geometry_tec.png"
        plot_event_figure(
            out_fig,
            fn,
            op,
            meas,
            fwd,
            tangent_h,
            tlat,
            tlon,
            float(np.nanmax(alt_km)),
        )

        rows.append(
            dict(
                filename=fn,
                figure_path=str(out_fig),
                n_rays=len(g),
                n_active_voxel_columns=len(op.grid.geolocation),
                max_active_radius_km=float(np.max(dist)),
                n_inside_six_point_samples=int(op.n_inside_six_point_samples),
                n_outside_two_point_samples=int(op.n_outside_two_point_samples),
                n_boundary_fallback_samples=int(op.n_boundary_fallback_samples),
                rmse_tecu=float(np.sqrt(np.mean((fwd - meas) ** 2))),
                bias_tecu=float(np.mean(fwd - meas)),
                fraction_forward_below_measured=float(np.mean(fwd < meas)),
                abel_top_km=float(np.nanmax(alt_km)),
            )
        )

    out = pd.DataFrame(rows)
    summary_csv = args.output_dir / "all_ro_operator_summary.csv"
    out.to_csv(summary_csv, index=False)

    print(out[[
        "filename",
        "n_active_voxel_columns",
        "rmse_tecu",
        "bias_tecu",
        "fraction_forward_below_measured",
        "figure_path",
    ]].to_string(index=False))
    print(f"\nSaved {len(out)} figure(s) to: {args.output_dir}")
    print(f"Saved summary CSV to: {summary_csv}")

    assert np.all(out.max_active_radius_km <= RADIUS_KM + 1e-6), (
        "At least one active Fibonacci voxel lies outside the 2000-km ROI."
    )
    assert np.allclose(out.fraction_forward_below_measured, 1.0), (
        "At least one event has recomputed H@Abel TEC >= measured TEC."
    )
    assert np.all(out.n_inside_six_point_samples > 0), (
        "At least one event never used the six-point inside-ROI interpolation."
    )
    assert np.all(out.n_outside_two_point_samples > 0), (
        "At least one event never used the two-point outside-ROI fallback."
    )

    print("\nPASS:")
    print("  * generated 15 per-event geometry+TEC figures")
    print("  * all active Fibonacci columns are inside the 2000-km ROI")
    print("  * inside-ROI LOS samples use 6-point interpolation")
    print("  * outside-ROI LOS samples use nearest-column x 2-altitude interpolation")
    print("  * all 15 H@Abel TEC profiles stay below measured RO TEC")


if __name__ == "__main__":
    main()
