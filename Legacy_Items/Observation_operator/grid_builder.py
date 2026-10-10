from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.tri import Triangulation
import pyproj

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
EARTH_RADIUS_KM = 6371.0

@dataclass
class GlobalFibonacciMesh:
    geolocation: np.ndarray  # [lon, lat]
    mesh: np.ndarray
    spacing_deg: float

@dataclass
class ROIFibonacciMesh:
    geolocation: np.ndarray  # [lon, lat], every point strictly inside/on ROI
    mesh: np.ndarray         # Delaunay triangles on local AEQD projection
    xy_m: np.ndarray
    spacing_deg: float
    center_lat: float
    center_lon: float
    radius_km: float

@dataclass
class LOSGridSelection:
    geolocation: np.ndarray  # only horizontal columns that H actually uses
    roi_vertex_indices: np.ndarray
    spacing_deg: float
    center_lat: float
    center_lon: float
    radius_km: float


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat = np.radians(np.asarray(lat_deg, float))
    lon = np.radians(np.asarray(lon_deg, float))
    h = np.asarray(alt_m, float)
    sl, cl = np.sin(lat), np.cos(lat)
    co, so = np.cos(lon), np.sin(lon)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sl * sl)
    return np.stack(np.broadcast_arrays(
        (N + h) * cl * co,
        (N + h) * cl * so,
        (N * (1 - WGS84_E2) + h) * sl,
    ), axis=-1)


def great_circle_distance_km(lat_deg, lon_deg, center_lat, center_lon):
    lat = np.radians(np.asarray(lat_deg, float))
    lon = np.radians(np.asarray(lon_deg, float))
    lat0 = np.radians(float(center_lat))
    lon0 = np.radians(float(center_lon))
    dlat = lat - lat0
    dlon = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def generate_global_fibonacci_mesh(spacing_deg: float = 5.0) -> GlobalFibonacciMesh:
    d = np.radians(float(spacing_deg))
    n = max(12, int(round(4 * np.pi / d**2)))
    golden = np.pi * (np.sqrt(5.0) - 1.0)
    idx = np.arange(n, dtype=float)
    sin_lat = 1.0 - 2.0 * (idx + 0.5) / n
    lat = np.degrees(np.arcsin(sin_lat))
    lon = np.degrees(golden * idx % (2 * np.pi)) - 180.0
    cos_lat = np.sqrt(np.clip(1 - sin_lat**2, 0, 1))
    lonr = np.radians(lon)
    xyz = np.column_stack([cos_lat * np.cos(lonr), cos_lat * np.sin(lonr), sin_lat])
    hull = ConvexHull(xyz)
    tri = hull.simplices.copy()
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    flip = np.einsum('ij,ij->i', np.cross(b - a, c - a), (a + b + c) / 3) < 0
    tri[flip, 1], tri[flip, 2] = tri[flip, 2].copy(), tri[flip, 1].copy()
    return GlobalFibonacciMesh(np.column_stack([lon, lat]), tri.astype(int), float(spacing_deg))


def build_roi_fibonacci_mesh(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    spacing_deg: float = 5.0,
) -> ROIFibonacciMesh:
    """Build the fixed horizontal tomography grid strictly inside the ROI.

    IMPORTANT: retain the ORIGINAL GLOBAL Fibonacci connectivity.  We do not
    re-Delaunay-triangulate the ROI subset, because doing so can create artificial
    long triangles across the clipped circular boundary.  Such long triangles can
    pull in a remote ROI vertex that is not a real neighbour of the LOS.

    Steps:
      1. Build the global equal-area Fibonacci lattice + spherical hull triangles.
      2. Keep only vertices whose great-circle radius is <= ROI radius.
      3. Keep only original global triangles whose three vertices survive.
      4. Project those retained vertices to local AEQD only for point location and
         barycentric interpolation.

    Therefore every horizontal triangle is made of genuine neighbouring Fibonacci
    vertices from the original global mesh, while every state column remains inside
    the circular ROI.
    """
    g = generate_global_fibonacci_mesh(spacing_deg)
    lon = g.geolocation[:, 0]
    lat = g.geolocation[:, 1]
    dist = great_circle_distance_km(lat, lon, center_lat, center_lon)
    keep = dist <= float(radius_km) + 1e-9
    kept_global = np.where(keep)[0]
    geo = g.geolocation[keep]
    if len(geo) < 3:
        raise ValueError('ROI contains fewer than 3 Fibonacci points; cannot triangulate.')

    # Preserve ORIGINAL global adjacency.
    old_to_new = np.full(len(g.geolocation), -1, dtype=int)
    old_to_new[kept_global] = np.arange(len(kept_global), dtype=int)
    tri_keep = np.all(keep[g.mesh], axis=1)
    mesh = old_to_new[g.mesh[tri_keep]]
    if len(mesh) == 0:
        raise ValueError('ROI contains no complete original Fibonacci triangles.')

    crs_geo = pyproj.CRS.from_epsg(4326)
    crs_aeqd = pyproj.CRS.from_proj4(
        f'+proj=aeqd +lat_0={float(center_lat)} +lon_0={float(center_lon)} '
        '+datum=WGS84 +units=m +no_defs'
    )
    xfm = pyproj.Transformer.from_crs(crs_geo, crs_aeqd, always_xy=True)
    x, y = xfm.transform(geo[:, 0], geo[:, 1])
    xy = np.column_stack([x, y])

    return ROIFibonacciMesh(
        geolocation=geo,
        mesh=mesh.astype(int),
        xy_m=xy,
        spacing_deg=float(spacing_deg),
        center_lat=float(center_lat),
        center_lon=float(center_lon),
        radius_km=float(radius_km),
    )


class ROIHorizontalLocator:
    """Locate LOS midpoint horizontally in the fixed ROI Fibonacci grid.

    * inside ROI + inside Delaunay hull -> 3 barycentric horizontal vertices
    * outside ROI (or just outside the finite Delaunay hull) -> nearest one
      horizontal vertex; vertical interpolation later makes this a 2-point Ne
      interpolation in the 3-D state.
    """
    def __init__(self, grid: ROIFibonacciMesh):
        self.grid = grid
        crs_geo = pyproj.CRS.from_epsg(4326)
        crs_aeqd = pyproj.CRS.from_proj4(
            f'+proj=aeqd +lat_0={grid.center_lat} +lon_0={grid.center_lon} '
            '+datum=WGS84 +units=m +no_defs'
        )
        self.xfm = pyproj.Transformer.from_crs(crs_geo, crs_aeqd, always_xy=True)
        # Use the retained ORIGINAL Fibonacci triangles, not a new ROI Delaunay.
        self.triangulation = Triangulation(
            grid.xy_m[:, 0], grid.xy_m[:, 1], triangles=grid.mesh
        )
        self.trifinder = self.triangulation.get_trifinder()
        # Keep this alias so the ray operator can access .simplices exactly as before.
        class _MeshView:
            pass
        self.delau = _MeshView()
        self.delau.simplices = grid.mesh
        # Unit-sphere tree gives a robust nearest horizontal column globally.
        lat = np.radians(grid.geolocation[:, 1])
        lon = np.radians(grid.geolocation[:, 0])
        unit = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        self.nearest_tree = cKDTree(unit)

    def locate(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        dist = great_circle_distance_km(lat, lon, self.grid.center_lat, self.grid.center_lon)
        inside_roi = dist <= self.grid.radius_km + 1e-9

        x, y = self.xfm.transform(lon, lat)
        qxy = np.column_stack([x, y])
        simplex = np.full(len(lat), -1, dtype=int)
        bary = np.zeros((len(lat), 3), dtype=float)

        if np.any(inside_roi):
            ii = np.where(inside_roi)[0]
            s = np.asarray(self.trifinder(qxy[ii, 0], qxy[ii, 1]), dtype=int)
            simplex[ii] = s
            good = s >= 0
            if np.any(good):
                jj = ii[good]
                ss = s[good]
                verts = self.grid.mesh[ss]
                A = self.grid.xy_m[verts[:, 0]]
                B = self.grid.xy_m[verts[:, 1]]
                C = self.grid.xy_m[verts[:, 2]]
                P = qxy[jj]
                v0 = B - A
                v1 = C - A
                v2 = P - A
                den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
                w1 = np.divide(
                    v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1],
                    den,
                    out=np.zeros_like(den),
                    where=np.abs(den) > 0,
                )
                w2 = np.divide(
                    v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1],
                    den,
                    out=np.zeros_like(den),
                    where=np.abs(den) > 0,
                )
                w0 = 1.0 - w1 - w2
                bary[jj] = np.column_stack([w0, w1, w2])

        # Explicit outside-ROI fallback.  Also covers the thin sliver between the
        # circular boundary and the convex hull of the finite ROI lattice.
        fallback = simplex < 0
        nearest = np.full(len(lat), -1, dtype=int)
        if np.any(fallback):
            lr = np.radians(lat[fallback])
            orr = np.radians(lon[fallback])
            uq = np.column_stack([np.cos(lr)*np.cos(orr), np.cos(lr)*np.sin(orr), np.sin(lr)])
            _, nv = self.nearest_tree.query(uq, k=1)
            nearest[fallback] = nv

        return simplex, bary, nearest, inside_roi


def geodesic_circle_latlon(center_lat, center_lon, radius_km, n=361):
    br = np.linspace(0, 2*np.pi, int(n))
    ang = float(radius_km) / EARTH_RADIUS_KM
    p1 = np.radians(center_lat)
    l1 = np.radians(center_lon)
    p2 = np.arcsin(np.sin(p1)*np.cos(ang) + np.cos(p1)*np.sin(ang)*np.cos(br))
    l2 = l1 + np.arctan2(
        np.sin(br)*np.sin(ang)*np.cos(p1),
        np.cos(ang) - np.sin(p1)*np.sin(p2),
    )
    l2 = (l2 + np.pi) % (2*np.pi) - np.pi
    return np.degrees(p2), np.degrees(l2)
