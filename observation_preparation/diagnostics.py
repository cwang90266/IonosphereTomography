"""Diagnostic/verification tooling for observation_preparation (Plan
Section 7, objective 4 / Section 10 step 4).

Each function works standalone against either freshly-prepared entries or
entries reloaded from netCDF (``netcdf_io.read_observations``), and takes
plain ``ObservationEntry`` objects/dicts (§6.1) rather than reaching back
into ``ro_source``/``igs_source`` internals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .schema import ObservationEntry
from .roi import circular_roi_points, geodesic_circle_latlon, DEFAULT_FIBONACCI_SPACING_DEG
from .roi_selection import haversine_km, sample_ray_geodetic

from TEC_model.podTc_file_processing import rayTangent, ECEFtolla


def _save_or_return(fig, output_path):
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return Path(output_path)
    return fig


def _draw_base_map(ax, center_lat, center_lon):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.65, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.55, zorder=1)
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.35, color="gray", alpha=0.35, linestyle="--",
    )
    if center_lat is not None and center_lon is not None:
        ax.scatter(
            [float(center_lon)], [float(center_lat)],
            transform=ccrs.PlateCarree(),
            marker="*", s=110, c="k", zorder=7,
            label=f"ROI center ({float(center_lat):.1f}°, {float(center_lon):.1f}°)",
        )


def _draw_roi_overlay(ax, center_lat, center_lon, radius_km):
    fib_lat, fib_lon = circular_roi_points(
        float(center_lat), float(center_lon), float(radius_km),
        spacing_deg=DEFAULT_FIBONACCI_SPACING_DEG,
    )
    if len(fib_lat):
        ax.scatter(
            fib_lon, fib_lat, transform=ccrs.PlateCarree(),
            s=22, color="k", alpha=0.85, zorder=2, label="voxels",
        )
    roi_lat, roi_lon = geodesic_circle_latlon(float(center_lat), float(center_lon), float(radius_km))
    ax.plot(
        roi_lon, roi_lat, transform=ccrs.Geodetic(),
        color="green", linewidth=2.0, zorder=6, label=f"ROI = {float(radius_km):.0f} km",
    )


def plot_geolocation(
    entries: list[ObservationEntry],
    roi: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
):
    """One combined geolocation map for a batch of RO+IGS entries (Plan
    Section 7.1): RO tangent-point tracks colored by altitude, IGS
    pierce-point scatter, and the ROI center/voxels/radius if ``roi`` is
    given (the same dict shape ``netcdf_io``/``roi_selection.build_roi_dict``
    use). Replaces the old per-RO-occultation PNG + separate IGS map
    (``_plot_ro_event``/``export_igs_outputs``'s inline block) with one
    figure -- see ``plot_obs_detail`` for the per-observation TEC/Abel
    panels those used to also carry.
    """
    center_lat = roi.get("roi_center_lat") if roi else None
    center_lon = roi.get("roi_center_lon") if roi else None
    radius_km = roi.get("roi_radius_km") if roi else None

    fig = plt.figure(figsize=(10, 8))
    map_crs = (
        ccrs.Orthographic(central_longitude=float(center_lon), central_latitude=float(center_lat))
        if center_lat is not None and center_lon is not None
        else ccrs.Robinson()
    )
    ax = fig.add_subplot(111, projection=map_crs)
    ax.set_title(f"Geolocation of prepared observations (n={len(entries)})")
    _draw_base_map(ax, center_lat, center_lon)

    ro_entries = [e for e in entries if e.obs_type == "RO"]
    igs_entries = [e for e in entries if e.obs_type == "IGS"]

    if ro_entries:
        lat_parts, lon_parts, alt_parts = [], [], []
        for e in ro_entries:
            tangent_xyz, _, _ = rayTangent(
                np.asarray(e.rec_ecef_km, dtype=float), np.asarray(e.gnss_ecef_km, dtype=float), units="km",
            )
            lat, lon, _ = ECEFtolla(tangent_xyz)
            lat = np.asarray(lat, dtype=float).ravel()
            lon = np.asarray(lon, dtype=float).ravel()
            alt = (
                np.asarray(e.tangent_alt_km, dtype=float).ravel()
                if e.tangent_alt_km is not None else np.full(lat.shape, np.nan)
            )
            valid = np.isfinite(lat) & np.isfinite(lon)
            if np.any(valid):
                ax.plot(
                    lon[valid], lat[valid], transform=ccrs.Geodetic(),
                    color="red", linewidth=1.0, alpha=0.6, zorder=3,
                )
            lat_parts.append(lat)
            lon_parts.append(lon)
            alt_parts.append(alt)

        cat_lat = np.concatenate(lat_parts)
        cat_lon = np.concatenate(lon_parts)
        cat_alt = np.concatenate(alt_parts)
        valid = np.isfinite(cat_lat) & np.isfinite(cat_lon) & np.isfinite(cat_alt)
        if np.any(valid):
            sc = ax.scatter(
                cat_lon[valid], cat_lat[valid], transform=ccrs.PlateCarree(),
                c=cat_alt[valid], s=10, zorder=4, label=f"RO tangent points (n={len(ro_entries)})",
            )
            cb = fig.colorbar(sc, ax=ax, pad=0.04, shrink=0.7)
            cb.set_label("RO tangent altitude (km)")

    if igs_entries:
        ipp_lat = np.concatenate([
            np.asarray(e.pierce_lat, dtype=float) if e.pierce_lat is not None else np.array([np.nan])
            for e in igs_entries
        ])
        ipp_lon = np.concatenate([
            np.asarray(e.pierce_lon, dtype=float) if e.pierce_lon is not None else np.array([np.nan])
            for e in igs_entries
        ])
        valid = np.isfinite(ipp_lat) & np.isfinite(ipp_lon)
        if np.any(valid):
            ax.scatter(
                ipp_lon[valid], ipp_lat[valid], transform=ccrs.PlateCarree(),
                s=18, color="tab:blue", marker="^", zorder=5,
                label=f"IGS pierce points (n={len(igs_entries)})",
            )

    if center_lat is not None and center_lon is not None and radius_km is not None:
        _draw_roi_overlay(ax, center_lat, center_lon, radius_km)

    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_obs_detail(entry: ObservationEntry, output_path: str | Path | None = None):
    """Per-observation detail view (Plan Section 7.1's "optional per-obs
    detail view"): TEC vs. tangent altitude (RO) or vs. epoch index (IGS),
    plus the Abel Ne profile for RO entries that ran Abel inversion."""
    has_abel = entry.obs_type == "RO" and entry.abel is not None
    fig, axes = plt.subplots(1, 2 if has_abel else 1, figsize=(10 if has_abel else 5, 4))
    ax_tec = axes[0] if has_abel else axes

    if entry.obs_type == "RO" and entry.tangent_alt_km is not None:
        ax_tec.plot(entry.tec, entry.tangent_alt_km)
        ax_tec.set(xlabel="TEC (TECU)", ylabel="Tangent height (km)")
    else:
        ax_tec.plot(entry.tec, marker="o")
        ax_tec.set(xlabel="Epoch index", ylabel="TEC (TECU)")
    ax_tec.set_title(entry.label or entry.obs_type)
    ax_tec.grid(True, alpha=0.25)

    if has_abel:
        ax_abel = axes[1]
        ne = entry.abel.get("Ne", [])
        alt = entry.abel.get("alt_km", entry.abel.get("alt", []))
        ax_abel.plot(ne, alt)
        ax_abel.set(xlabel="Abel Ne (m$^{-3}$)", ylabel="Altitude (km)")
        ax_abel.grid(True, alpha=0.25)

    fig.tight_layout()
    return _save_or_return(fig, output_path)


def pooled_fraction_inside_roi(
    entry: ObservationEntry,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    alt_limit_km: float,
    num_segments: int = 200,
) -> float | None:
    """All of one entry's rays' sub-``alt_limit_km`` sample points pooled
    together, and the fraction of that pool within ``radius_km`` -- one
    number per observation, matching §7.2's "across all obs" framing (as
    opposed to ``los_within_roi``'s per-ray threshold decision). ``None``
    if the entry has no ray reaching at or below ``alt_limit_km`` at all.
    """
    rec = np.asarray(entry.rec_ecef_km, dtype=np.float64)
    gnss = np.asarray(entry.gnss_ecef_km, dtype=np.float64)
    n_rays = rec.shape[1]

    pooled_dist = []
    for i in range(n_rays):
        lats, lons, alts_km = sample_ray_geodetic(rec[:, i], gnss[:, i], num_segments)
        dist_km = haversine_km(center_lat, center_lon, lats, lons)
        below = alts_km <= float(alt_limit_km)
        pooled_dist.append(dist_km[below])

    if not pooled_dist:
        return None
    cat = np.concatenate(pooled_dist)
    if not cat.size:
        return None
    return float(np.mean(cat <= radius_km))


def plot_los_penetration(
    entries: list[ObservationEntry],
    center_lat: float,
    center_lon: float,
    radius_km: float,
    alt_limit_km: float,
    num_segments: int = 200,
    output_path: str | Path | None = None,
):
    """Visual check of the full-LOS ROI containment logic itself (Plan
    Section 7.2), rather than just trusting ``los_within_roi``: altitude
    vs. distance-from-ROI-center for every sampled ray point (shading the
    ROI radius and marking alt_limit_km), plus a histogram of what
    fraction of each observation's own below-alt_limit_km LOS actually
    lands inside the ROI -- a sanity check on the fraction_required
    threshold choice.
    """
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    for e in entries:
        rec = np.asarray(e.rec_ecef_km, dtype=np.float64)
        gnss = np.asarray(e.gnss_ecef_km, dtype=np.float64)
        for i in range(rec.shape[1]):
            lats, lons, alts_km = sample_ray_geodetic(rec[:, i], gnss[:, i], num_segments)
            dist_km = haversine_km(center_lat, center_lon, lats, lons)
            ax_scatter.plot(dist_km, alts_km, color="0.6", linewidth=0.4, alpha=0.5, zorder=1)

    fractions = [
        f for e in entries
        if (f := pooled_fraction_inside_roi(e, center_lat, center_lon, radius_km, alt_limit_km, num_segments))
        is not None
    ]

    ax_scatter.axhline(
        alt_limit_km, color="tab:orange", linestyle="--", linewidth=1.2,
        label=f"alt_limit_km = {alt_limit_km:.0f}",
    )
    ax_scatter.axvspan(0, radius_km, color="tab:green", alpha=0.12, label=f"ROI radius = {radius_km:.0f} km")
    ax_scatter.set(xlabel="Distance from ROI center (km)", ylabel="Altitude (km)", title="LOS altitude vs. ROI distance")
    ax_scatter.legend(loc="upper right", fontsize=8)
    ax_scatter.grid(True, alpha=0.25)

    if fractions:
        ax_hist.hist(fractions, bins=np.linspace(0.0, 1.0, 21), color="tab:blue", edgecolor="k", alpha=0.85)
    ax_hist.set(
        xlabel="Fraction of sub-alt_limit_km LOS inside ROI",
        ylabel="Number of observations",
        title=f"n={len(fractions)} observations",
    )
    ax_hist.grid(True, alpha=0.25)

    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_tec_comparison(
    entries: list[ObservationEntry],
    edp_samples,
    num_segments: int = 1000,
    output_path: str | Path | None = None,
) -> tuple[Any, dict[str, dict[str, float]]]:
    """Measured vs. ``EDPSamples``-predicted TEC (Plan Section 7.3) -- the
    strongest end-to-end sanity check for the refactor, since it exercises
    scanning/filtering/ROI selection/netCDF round-trip and the real
    observation operator in one pass. "Predicted" uses the ensemble-mean
    density (``edp_samples.edps`` averaged over the sample axis) -- IRI
    climatology if ``edp_samples`` came straight from an IRI draw, or the
    current ensemble mean otherwise.

    Returns ``(figure_or_path, stats)`` where ``stats`` has a
    ``{"n", "bias", "rms"}`` entry per ``obs_type`` present.
    """
    density = np.asarray(edp_samples.edps, dtype=np.float64).mean(axis=-1)

    measured_parts, predicted_parts, alt_parts, elev_parts, type_parts = [], [], [], [], []

    for e in entries:
        H = edp_samples.get_observation_operator(e.to_operator_dict(), num_segments=num_segments)
        predicted = np.asarray(H) @ density.reshape(-1)
        n = e.n_rays

        measured_parts.append(np.asarray(e.tec, dtype=np.float64))
        predicted_parts.append(np.asarray(predicted, dtype=np.float64))
        alt_parts.append(np.asarray(e.tangent_alt_km, dtype=np.float64) if e.tangent_alt_km is not None else np.full(n, np.nan))
        elev_parts.append(np.asarray(e.elev_deg, dtype=np.float64) if e.elev_deg is not None else np.full(n, np.nan))
        type_parts.append(np.full(n, e.obs_type, dtype=object))

    measured = np.concatenate(measured_parts) if measured_parts else np.array([])
    predicted = np.concatenate(predicted_parts) if predicted_parts else np.array([])
    alt = np.concatenate(alt_parts) if alt_parts else np.array([])
    elev = np.concatenate(elev_parts) if elev_parts else np.array([])
    obs_type = np.concatenate(type_parts) if type_parts else np.array([], dtype=object)
    residual = predicted - measured

    stats: dict[str, dict[str, float]] = {}
    for label in ("RO", "IGS"):
        mask = obs_type == label
        if np.any(mask):
            stats[label] = {
                "n": int(mask.sum()),
                "bias": float(np.nanmean(residual[mask])),
                "rms": float(np.sqrt(np.nanmean(residual[mask] ** 2))),
            }

    fig, (ax_scatter, ax_alt, ax_elev) = plt.subplots(1, 3, figsize=(15, 5))

    for label, color in (("RO", "tab:red"), ("IGS", "tab:blue")):
        mask = obs_type == label
        if np.any(mask):
            ax_scatter.scatter(measured[mask], predicted[mask], s=10, alpha=0.5, color=color, label=f"{label} (n={int(mask.sum())})")
    if measured.size and predicted.size:
        lo = float(min(measured.min(), predicted.min()))
        hi = float(max(measured.max(), predicted.max()))
        ax_scatter.plot([lo, hi], [lo, hi], color="k", linewidth=1.0, linestyle="--")
    ax_scatter.set(xlabel="Measured TEC (TECU)", ylabel="Predicted TEC (TECU)", title="Measured vs. predicted")
    ax_scatter.legend(loc="upper left", fontsize=8)
    ax_scatter.grid(True, alpha=0.25)

    ro_mask = obs_type == "RO"
    if np.any(ro_mask):
        ax_alt.scatter(alt[ro_mask], residual[ro_mask], s=10, alpha=0.5, color="tab:red")
    ax_alt.axhline(0, color="k", linewidth=0.8)
    ax_alt.set(xlabel="RO tangent altitude (km)", ylabel="Predicted - measured (TECU)", title="RO residual vs. altitude")
    ax_alt.grid(True, alpha=0.25)

    igs_mask = obs_type == "IGS"
    if np.any(igs_mask):
        ax_elev.scatter(elev[igs_mask], residual[igs_mask], s=10, alpha=0.5, color="tab:blue")
    ax_elev.axhline(0, color="k", linewidth=0.8)
    ax_elev.set(xlabel="IGS elevation (deg)", ylabel="Predicted - measured (TECU)", title="IGS residual vs. elevation")
    ax_elev.grid(True, alpha=0.25)

    fig.tight_layout()
    return _save_or_return(fig, output_path), stats


__all__ = [
    "plot_geolocation",
    "plot_obs_detail",
    "pooled_fraction_inside_roi",
    "plot_los_penetration",
    "plot_tec_comparison",
]
