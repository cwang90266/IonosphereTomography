from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pyproj
from .grid_builder import build_roi_fibonacci_mesh, ROIHorizontalLocator, LOSGridSelection

@dataclass
class LOSOperatorResult:
    H: np.ndarray
    grid: LOSGridSelection
    altitude_km: np.ndarray
    n_inside_six_point_samples: int = 0
    n_outside_two_point_samples: int = 0
    n_boundary_fallback_samples: int = 0

_XFM = pyproj.Transformer.from_crs('EPSG:4978', 'EPSG:4979', always_xy=True)

def _lla(xyz):
    lon, lat, alt = _XFM.transform(xyz[...,0]*1000, xyz[...,1]*1000, xyz[...,2]*1000)
    return np.asarray(lat), np.asarray(lon), np.asarray(alt)/1000

def _interp_h(ht, h):
    i = np.searchsorted(ht, h, side='right') - 1
    i = np.clip(i, 0, len(ht)-2)
    den = ht[i+1] - ht[i]
    t = np.divide(h-ht[i], den, out=np.zeros_like(h, dtype=float), where=den!=0)
    return i, np.column_stack([1-t, t])

def build_los_fibonacci_operator(
    leo_km,
    gnss_km,
    altitude_km,
    *,
    spacing_deg=5.0,
    num_segments=1000,
    center_lat,
    center_lon,
    radius_km,
):
    """Build the common Ne->TEC operator for a circular ROI Fibonacci state.

    Horizontal rule (matching the intended tomography behaviour):
      1. Fibonacci state columns exist ONLY inside the circular ROI.
      2. LOS midpoint inside ROI and covered by a triangle:
            3 horizontal barycentric vertices x 2 altitude levels = 6 Ne points.
      3. LOS midpoint outside ROI:
            nearest ROI horizontal column x 2 altitude levels = 2 Ne points.
         The same nearest-column fallback is used in the tiny circular-edge sliver
         not covered by the finite Delaunay hull.

    Vertical interpolation is linear in altitude in both cases.  H includes only
    the Abel/state altitude interval supplied through altitude_km.
    """
    leo = np.asarray(leo_km, float)
    gnss = np.asarray(gnss_km, float)
    if leo.shape[0] != 3:
        leo = leo.T
    if gnss.shape[0] != 3:
        gnss = gnss.T
    if leo.shape != gnss.shape or leo.shape[0] != 3:
        raise ValueError('LEO and GNSS must have shape (3,n_rays) or (n_rays,3).')

    alt = np.asarray(altitude_km, float)
    if alt.ndim != 1 or len(alt) < 2 or not np.all(np.diff(alt) > 0):
        raise ValueError('altitude_km must be a strictly ascending 1-D array with >=2 levels.')

    roi = build_roi_fibonacci_mesh(center_lat, center_lon, radius_km, spacing_deg)
    locator = ROIHorizontalLocator(roi)

    edges = np.linspace(0, 1, int(num_segments)+1)
    mids = 0.5*(edges[:-1] + edges[1:])

    # First pass records geometry and which ROI horizontal columns are actually used.
    raydata = []
    used_vertices = set()
    n_inside6 = 0
    n_outside2 = 0
    n_boundary = 0

    for i in range(leo.shape[1]):
        r0 = leo[:, i]
        r1 = gnss[:, i]
        v = r1-r0
        xyz = r0[None, :] + mids[:, None]*v
        lat, lon, h = _lla(xyz)
        dl = np.linalg.norm(v)*1000.0/int(num_segments)

        valid = np.isfinite(h) & (h >= alt[0]) & (h <= alt[-1])
        latv, lonv, hv = lat[valid], lon[valid], h[valid]
        if len(hv) == 0:
            raydata.append(None)
            continue

        simplex, bary, nearest, inside_roi = locator.locate(latv, lonv)
        six = simplex >= 0
        fallback = ~six

        tri_vertices = np.empty((0,3), dtype=int)
        if np.any(six):
            tri_vertices = locator.delau.simplices[simplex[six]]
            used_vertices.update(np.unique(tri_vertices).tolist())
            n_inside6 += int(np.sum(six))

        near_vertices = nearest[fallback]
        if np.any(fallback):
            used_vertices.update(np.unique(near_vertices).tolist())
            n_outside2 += int(np.sum(fallback & ~inside_roi))
            n_boundary += int(np.sum(fallback & inside_roi))

        raydata.append(dict(
            h=hv,
            dl=np.full(len(hv), dl, dtype=float),
            simplex=simplex,
            bary=bary,
            nearest=nearest,
            inside_roi=inside_roi,
        ))

    active = np.array(sorted(used_vertices), dtype=int)
    if len(active) == 0:
        raise ValueError('No ROI Fibonacci columns were used by any LOS segment.')
    local_map = np.full(len(roi.geolocation), -1, dtype=int)
    local_map[active] = np.arange(len(active))

    ngeo = len(active)
    H = np.zeros((leo.shape[1], len(alt)*ngeo), dtype=float)

    # Second pass assembles the sparse interpolation weights into dense H.
    for ir, rd in enumerate(raydata):
        if rd is None:
            continue
        h = rd['h']
        dl = rd['dl']
        simplex = rd['simplex']
        bary = rd['bary']
        nearest = rd['nearest']
        ia, wv = _interp_h(alt, h)

        six_idx = np.where(simplex >= 0)[0]
        if len(six_idx):
            verts_roi = locator.delau.simplices[simplex[six_idx]]
            verts = local_map[verts_roi]
            for j in range(3):
                hw = bary[six_idx, j]
                np.add.at(H[ir], ia[six_idx]*ngeo + verts[:,j], dl[six_idx]*hw*wv[six_idx,0]/1e16)
                np.add.at(H[ir], (ia[six_idx]+1)*ngeo + verts[:,j], dl[six_idx]*hw*wv[six_idx,1]/1e16)

        two_idx = np.where(simplex < 0)[0]
        if len(two_idx):
            verts = local_map[nearest[two_idx]]
            np.add.at(H[ir], ia[two_idx]*ngeo + verts, dl[two_idx]*wv[two_idx,0]/1e16)
            np.add.at(H[ir], (ia[two_idx]+1)*ngeo + verts, dl[two_idx]*wv[two_idx,1]/1e16)

    grid = LOSGridSelection(
        geolocation=roi.geolocation[active],
        roi_vertex_indices=active,
        spacing_deg=float(spacing_deg),
        center_lat=float(center_lat),
        center_lon=float(center_lon),
        radius_km=float(radius_km),
    )
    return LOSOperatorResult(
        H=H,
        grid=grid,
        altitude_km=alt,
        n_inside_six_point_samples=n_inside6,
        n_outside_two_point_samples=n_outside2,
        n_boundary_fallback_samples=n_boundary,
    )
