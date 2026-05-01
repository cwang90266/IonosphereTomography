"""
locate_in_mesh.py
-----------------

Given a spherical triangular mesh defined by

    geolocation : (N, 2) array of [lat, lon] vertices
    mesh        : (T, 3) array of vertex indices (one row per triangle)

determine, for each query point in (lat, lon), which triangle it falls in.

The points are treated as lying on the unit sphere, so the test handles the
antimeridian and the poles correctly (no special-casing needed).

Algorithm
=========
1. Convert every vertex and query point to a 3D unit vector (x, y, z).
2. Compute the (normalised) centroid of every triangle in 3D.
3. Build a KD-tree on the centroids; for each query point fetch its `k`
   nearest candidate triangles.  Euclidean distance in 3D between unit
   vectors is monotonic in great-circle distance, so this is a valid
   nearest-neighbour search on the sphere.
4. For each candidate triangle (A, B, C) and query point P on the unit
   sphere, P lies inside the spherical triangle iff the three scalar
   triple products

        s1 = (A x B) . P
        s2 = (B x C) . P
        s3 = (C x A) . P

   all have the same sign as the triangle's orientation scalar

        orient = (A x B) . C       (= (B x C) . A = (C x A) . B)

   This is robust to either CCW or CW winding (the orientation scalar
   captures it) and crucially distinguishes the triangle's interior
   from its antipodal region: a naive "all-same-sign" test without the
   orientation reference accepts both, which is wrong.

If, for some query point, none of the `k` nearest triangles contains it,
`k` is automatically doubled and the search retries (up to `max_k`).  Any
query point still unresolved gets a triangle index of -1.

Optional planar barycentric weights (computed in the plane of the triangle
in 3D) are returned for use in interpolation.  These reduce to the usual
planar barycentric coordinates as the triangle shrinks, which is the
common case for geophysical meshes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def latlon_to_xyz(lat, lon, degrees: bool = True) -> np.ndarray:
    """Convert (lat, lon) to 3D unit vectors on the unit sphere.

    Accepts scalars or arrays.  Returns an array with a trailing axis of
    length 3.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if degrees:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def find_containing_triangles(
    query_latlon,
    geolocation,
    mesh,
    degrees: bool = True,
    k: int = 8,
    max_k: int | None = None,
    eps: float = 1e-10,
    return_bary: bool = False,
):
    """Locate each query point in the spherical triangular mesh.

    Parameters
    ----------
    query_latlon : (M, 2) array_like
        Query points as [lat, lon].
    geolocation : (N, 2) array_like
        Mesh vertices as [lat, lon].
    mesh : (T, 3) array_like of int
        Vertex indices of the three corners of each triangle.
    degrees : bool, default True
        Whether the lat/lon inputs are in degrees.  If False, radians.
    k : int, default 8
        Initial number of nearest-centroid candidates to test per query.
    max_k : int, optional
        Maximum `k` to try if some queries remain unresolved.  Defaults to
        the number of triangles (i.e. brute force as a last resort).
    eps : float, default 1e-10
        Tolerance for the sign test (a point exactly on an edge has a
        scalar triple product equal to zero).
    return_bary : bool, default False
        If True, also return planar barycentric weights for each resolved
        query point.

    Returns
    -------
    triangle_idx : (M,) ndarray of int
        Index into `mesh` of the containing triangle, or -1 if no
        triangle contained the point.
    bary : (M, 3) ndarray, optional
        Planar barycentric weights w_A, w_B, w_C such that
        ``w_A * A + w_B * B + w_C * C`` is the projection of the query
        point onto the plane of triangle (A, B, C).  Only returned when
        ``return_bary=True``.
    """
    query_latlon = np.asarray(query_latlon, dtype=float).reshape(-1, 2)
    geolocation = np.asarray(geolocation, dtype=float).reshape(-1, 2)
    mesh = np.asarray(mesh, dtype=np.int64).reshape(-1, 3)

    n_tri = mesh.shape[0]
    if max_k is None:
        max_k = n_tri

    # 3D unit vectors
    V = latlon_to_xyz(geolocation[:, 0], geolocation[:, 1], degrees=degrees)  # (N, 3)
    Q = latlon_to_xyz(query_latlon[:, 0], query_latlon[:, 1], degrees=degrees)  # (M, 3)
    M = Q.shape[0]

    # Triangle corners
    A = V[mesh[:, 0]]
    B = V[mesh[:, 1]]
    C = V[mesh[:, 2]]

    # Per-triangle orientation reference (positive for CCW, negative for CW
    # as seen from outside the sphere).  Used to disambiguate the triangle
    # from its antipode in the containment test.
    orient = np.einsum("ij,ij->i", np.cross(A, B), C)  # (T,)

    # Triangle centroids, projected onto the unit sphere
    centroids = (A + B + C) / 3.0
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    tree = cKDTree(centroids)

    triangle_idx = np.full(M, -1, dtype=np.int64)

    current_k = min(k, n_tri)
    while True:
        unresolved = triangle_idx == -1
        if not np.any(unresolved):
            break

        # k nearest candidate triangles for the still-unresolved queries
        _, cand = tree.query(Q[unresolved], k=current_k)
        if current_k == 1:
            cand = cand[:, None]  # cKDTree drops the trailing axis when k=1

        # Test candidates column by column.  Most points resolve in the
        # first one or two columns, so this is cheap.
        unresolved_idx = np.where(unresolved)[0]
        for col in range(cand.shape[1]):
            still = triangle_idx[unresolved_idx] == -1
            if not np.any(still):
                break
            sub_idx = unresolved_idx[still]
            tris = cand[still, col]

            a = A[tris]
            b = B[tris]
            c = C[tris]
            p = Q[sub_idx]
            r = orient[tris]

            # Each scalar triple product, multiplied by the triangle's
            # orientation reference, must be non-negative for the point
            # to be on the "inside" side of every edge.
            s1 = np.einsum("ij,ij->i", np.cross(a, b), p) * r
            s2 = np.einsum("ij,ij->i", np.cross(b, c), p) * r
            s3 = np.einsum("ij,ij->i", np.cross(c, a), p) * r

            inside = (s1 >= -eps) & (s2 >= -eps) & (s3 >= -eps)
            triangle_idx[sub_idx[inside]] = tris[inside]

        # If anything is still unresolved, widen the search.
        if np.any(triangle_idx == -1) and current_k < max_k:
            current_k = min(current_k * 2, max_k)
        else:
            break

    if not return_bary:
        return triangle_idx

    # Planar barycentric weights for the resolved queries
    bary = np.zeros((M, 3))
    found = triangle_idx != -1
    if np.any(found):
        tris = triangle_idx[found]
        a = A[tris]
        b = B[tris]
        c = C[tris]
        p = Q[found]
        # Project p onto the triangle's plane and solve a 2x2 system in
        # barycentric form.  This is the standard planar barycentric
        # computation; it gives sensible interpolation weights for the
        # small spherical triangles found in geophysical meshes.
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denom = d00 * d11 - d01 * d01
        wB = (d11 * d20 - d01 * d21) / denom
        wC = (d00 * d21 - d01 * d20) / denom
        wA = 1.0 - wB - wC
        bary[found, 0] = wA
        bary[found, 1] = wB
        bary[found, 2] = wC

    return triangle_idx, bary


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny mesh: an icosahedron's first few faces, just enough to
    # verify correctness.
    rng = np.random.default_rng(0)

    # Octahedron vertices
    geolocation = np.array(
        [
            [90.0, 0.0],     # 0: north pole
            [-90.0, 0.0],    # 1: south pole
            [0.0, 0.0],      # 2: equator, 0E
            [0.0, 90.0],     # 3: equator, 90E
            [0.0, 180.0],    # 4: equator, 180
            [0.0, -90.0],    # 5: equator, 90W
        ]
    )
    mesh = np.array(
        [
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 2],
            [1, 3, 2],
            [1, 4, 3],
            [1, 5, 4],
            [1, 2, 5],
        ]
    )

    # A few hand-picked queries
    queries = np.array(
        [
            [45.0, 45.0],     # northern hemisphere, between 0E and 90E -> tri 0
            [45.0, 135.0],    # northern, 90E to 180 -> tri 1
            [-30.0, -45.0],   # southern, 90W to 0E -> tri 7
            [10.0, 1.0],      # near the prime meridian, just north -> tri 0 or 3
            [89.9, 200.0],    # near north pole, lon 200E (= -160) -> tri 2
        ]
    )

    tri, bary = find_containing_triangles(
        queries, geolocation, mesh, return_bary=True
    )
    for q, t, w in zip(queries, tri, bary):
        print(f"query {q}  ->  triangle {t}  (bary {w.round(3)})")

    # Sanity check: the barycentric reconstruction should land on the
    # query point's direction.
    Q = latlon_to_xyz(queries[:, 0], queries[:, 1])
    V = latlon_to_xyz(geolocation[:, 0], geolocation[:, 1])
    for i, t in enumerate(tri):
        if t < 0:
            continue
        a, b, c = V[mesh[t]]
        recon = bary[i, 0] * a + bary[i, 1] * b + bary[i, 2] * c
        recon /= np.linalg.norm(recon)
        cos_err = np.dot(recon, Q[i])
        print(f"  reconstruction cos-similarity: {cos_err:.6f}")
