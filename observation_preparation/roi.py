from __future__ import annotations

import numpy as np


# Copied from the existing main-code ROI implementation so ROI geometry is
# owned by observation_preparation rather than imported from main.
_EARTH_RADIUS_KM = 6371.0
DEFAULT_FIBONACCI_SPACING_DEG = 5.0
DEFAULT_FIBONACCI_SPACING_KM = _EARTH_RADIUS_KM * np.deg2rad(DEFAULT_FIBONACCI_SPACING_DEG)


def fibonacci_sphere_latlon(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Golden-spiral (Fibonacci) lattice of n_points ~evenly spaced on a sphere.

    Returns (lat_deg, lon_deg) with lat in [-90, 90], lon in [-180, 180). The
    average nearest-neighbour spacing is ~R * sqrt(4*pi / n_points).
    """
    n = int(max(n_points, 1))
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    lon = np.degrees((golden * i) % (2.0 * np.pi))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def latlon_unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """(N,3) unit vectors on the sphere for arrays of lat/lon in degrees."""
    latr = np.radians(np.asarray(lat_deg, dtype=float))
    lonr = np.radians(np.asarray(lon_deg, dtype=float))
    cl = np.cos(latr)
    return np.column_stack([cl * np.cos(lonr), cl * np.sin(lonr), np.sin(latr)])


def robust_sphere_centroid(
    V: np.ndarray,
    trim_iters: int = 3,
    trim_percentile: float = 90.0,
) -> np.ndarray:
    """Outlier-resistant unit centroid of (N,3) unit vectors."""
    V = np.asarray(V, dtype=float)
    if V.shape[0] <= 3:
        c = V.mean(axis=0)
        n = float(np.linalg.norm(c))
        return V[0] if n < 1e-9 else c / n

    keep = np.ones(V.shape[0], dtype=bool)
    centroid = V.mean(axis=0)
    nrm = float(np.linalg.norm(centroid))
    centroid = V[0] if nrm < 1e-9 else centroid / nrm
    for _ in range(int(trim_iters)):
        cos_to = np.clip(V[keep] @ centroid, -1.0, 1.0)
        ang = np.arccos(cos_to)
        thr = np.percentile(ang, float(trim_percentile))
        idx = np.where(keep)[0]
        new_keep = keep.copy()
        new_keep[idx[ang > thr]] = False
        if new_keep.sum() < 3:
            break
        keep = new_keep
        c = V[keep].mean(axis=0)
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-9:
            break
        centroid = c / nrm
    return centroid


def fibonacci_roi_grid(
    roi_lats,
    roi_lons,
    spacing_km: float,
    margin_km: float = 200.0,
    max_pts: int | None = None,
    radius_percentile: float = 95.0,
    trim_iters: int = 3,
    trim_percentile: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fibonacci-sphere grid (~spacing_km apart) covering the RO/IGS footprint.

    This is the existing main-code implementation moved intact into the
    observation_preparation package.  A global Fibonacci sphere is generated
    and then clipped by great-circle distance to the robust footprint centroid.
    """
    roi_lats = np.asarray(roi_lats, dtype=float)
    roi_lons = np.asarray(roi_lons, dtype=float)
    ok = np.isfinite(roi_lats) & np.isfinite(roi_lons)
    roi_lats, roi_lons = roi_lats[ok], roi_lons[ok]
    if roi_lats.size == 0:
        return np.array([]), np.array([])

    V = latlon_unit_vectors(roi_lats, roi_lons)
    centroid = robust_sphere_centroid(
        V, trim_iters=trim_iters, trim_percentile=trim_percentile
    )

    cos_to_roi = np.clip(V @ centroid, -1.0, 1.0)
    ang_to_roi = np.arccos(cos_to_roi)
    pct_ang = float(np.percentile(ang_to_roi, float(radius_percentile)))
    radius_km = _EARTH_RADIUS_KM * pct_ang + float(margin_km)

    n_global = int(
        round(4.0 * np.pi * _EARTH_RADIUS_KM**2 / float(spacing_km) ** 2)
    )
    n_global = max(n_global, 60)
    flat, flon = fibonacci_sphere_latlon(n_global)
    F = latlon_unit_vectors(flat, flon)
    ang = np.arccos(np.clip(F @ centroid, -1.0, 1.0))
    dist_km = _EARTH_RADIUS_KM * ang
    keep = dist_km <= radius_km
    klat, klon, kdist = flat[keep], flon[keep], dist_km[keep]

    if max_pts is not None and klat.size > int(max_pts):
        order = np.argsort(kdist)[: int(max_pts)]
        klat, klon = klat[order], klon[order]

    return klat, klon


def circular_roi_points(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    spacing_km: float | None = None,
    spacing_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Fibonacci points inside an exact circular great-circle ROI.

    By default the point spacing is the same 5-degree great-circle target used
    by the project's equal-area global Fibonacci grid.  At Earth radius 6371 km
    this is about 556 km.  Pass ``spacing_km`` explicitly to override it.
    """
    radius_km = float(radius_km)
    if spacing_km is not None and spacing_deg is not None:
        raise ValueError("Specify only one of spacing_km or spacing_deg.")
    if spacing_km is None:
        if spacing_deg is None:
            spacing_deg = DEFAULT_FIBONACCI_SPACING_DEG
        spacing_km = _EARTH_RADIUS_KM * np.deg2rad(float(spacing_deg))
    return fibonacci_roi_grid(
        [float(center_lat)],
        [float(center_lon)],
        spacing_km=float(spacing_km),
        margin_km=radius_km,
        max_pts=None,
    )


def geodesic_circle_latlon(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    n: int = 361,
) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude/longitude samples of a great-circle-radius boundary."""
    bearings = np.linspace(0.0, 2.0 * np.pi, int(n))
    ang = float(radius_km) / _EARTH_RADIUS_KM
    lat1 = np.deg2rad(float(center_lat))
    lon1 = np.deg2rad(float(center_lon))

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(ang)
        + np.cos(lat1) * np.sin(ang) * np.cos(bearings)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearings) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * np.sin(lat2),
    )
    lon2 = (lon2 + np.pi) % (2.0 * np.pi) - np.pi
    return np.rad2deg(lat2), np.rad2deg(lon2)


def square_roi_bounds(
    center_lat: float,
    center_lon: float,
    half_width_km: float,
) -> tuple[float, float, float, float]:
    """Approximate square ROI bounds around a center using a physical half-width.

    Returns (lat_min, lat_max, lon_min, lon_max). Longitude scaling is adjusted
    by cos(latitude), matching the physical-distance intent of the existing ROI
    code near high latitudes.
    """
    dlat = float(half_width_km) / 111.32
    coslat = max(abs(np.cos(np.deg2rad(float(center_lat)))), 1e-6)
    dlon = float(half_width_km) / (111.32 * coslat)
    return (
        float(center_lat) - dlat,
        float(center_lat) + dlat,
        float(center_lon) - dlon,
        float(center_lon) + dlon,
    )


def square_roi_boundary_latlon(
    center_lat: float,
    center_lon: float,
    half_width_km: float,
    n_per_edge: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lat/lon samples around the perimeter of a square ROI."""
    lat0, lat1, lon0, lon1 = square_roi_bounds(
        center_lat, center_lon, half_width_km
    )
    n = int(max(n_per_edge, 2))
    bottom_lon = np.linspace(lon0, lon1, n)
    right_lat = np.linspace(lat0, lat1, n)
    top_lon = np.linspace(lon1, lon0, n)
    left_lat = np.linspace(lat1, lat0, n)
    lat = np.concatenate([
        np.full(n, lat0),
        right_lat,
        np.full(n, lat1),
        left_lat,
    ])
    lon = np.concatenate([
        bottom_lon,
        np.full(n, lon1),
        top_lon,
        np.full(n, lon0),
    ])
    return lat, lon


__all__ = [
    "DEFAULT_FIBONACCI_SPACING_DEG",
    "DEFAULT_FIBONACCI_SPACING_KM",
    "fibonacci_sphere_latlon",
    "latlon_unit_vectors",
    "robust_sphere_centroid",
    "fibonacci_roi_grid",
    "circular_roi_points",
    "geodesic_circle_latlon",
    "square_roi_bounds",
    "square_roi_boundary_latlon",
]
