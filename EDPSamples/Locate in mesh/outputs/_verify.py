"""Verify the containment logic from locate_in_mesh.py without scipy.

Runs the same scalar-triple-product test in brute-force form on a known
mesh, and cross-checks against an independent ground-truth oracle (the
sign of the dot product with the analytic face-plane normals of an
octahedron).
"""

import numpy as np


def latlon_to_xyz(lat, lon, degrees=True):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if degrees:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def brute_force_locate(Q, V, mesh, eps=1e-10):
    """Brute-force version of the spherical containment test."""
    A = V[mesh[:, 0]]
    B = V[mesh[:, 1]]
    C = V[mesh[:, 2]]
    orient = np.einsum("ij,ij->i", np.cross(A, B), C)
    out = np.full(Q.shape[0], -1, dtype=int)
    for i, p in enumerate(Q):
        s1 = np.einsum("ij,j->i", np.cross(A, B), p) * orient
        s2 = np.einsum("ij,j->i", np.cross(B, C), p) * orient
        s3 = np.einsum("ij,j->i", np.cross(C, A), p) * orient
        inside = (s1 >= -eps) & (s2 >= -eps) & (s3 >= -eps)
        hits = np.where(inside)[0]
        if len(hits):
            out[i] = hits[0]
    return out


def oracle_octahedron(Q):
    """Independent ground truth for an octahedron whose 8 faces are the
    octants.  A point on the unit sphere is in the octant identified by
    (sign(x), sign(y), sign(z))."""
    sx = np.sign(Q[:, 0])
    sy = np.sign(Q[:, 1])
    sz = np.sign(Q[:, 2])
    # Map (sx,sy,sz) -> octant index 0..7
    return ((sx > 0).astype(int) * 4
            + (sy > 0).astype(int) * 2
            + (sz > 0).astype(int))


# Octahedron mesh: same as in the demo
geolocation = np.array(
    [
        [90.0, 0.0],
        [-90.0, 0.0],
        [0.0, 0.0],
        [0.0, 90.0],
        [0.0, 180.0],
        [0.0, -90.0],
    ]
)
mesh = np.array(
    [
        [0, 2, 3],   # +z, +x, +y  -> octant (+,+,+) = 7
        [0, 3, 4],   # +z, +y, -x  -> octant (-,+,+) = 3
        [0, 4, 5],   # +z, -x, -y  -> octant (-,-,+) = 1
        [0, 5, 2],   # +z, -y, +x  -> octant (+,-,+) = 5
        [1, 3, 2],   # -z, +y, +x  -> octant (+,+,-) = 6
        [1, 4, 3],   # -z, -x, +y  -> octant (-,+,-) = 2
        [1, 5, 4],   # -z, -y, -x  -> octant (-,-,-) = 0
        [1, 2, 5],   # -z, +x, -y  -> octant (+,-,-) = 4
    ]
)
# Map triangle index -> octant index (computed from mesh winding)
tri_to_octant = np.array([7, 3, 1, 5, 6, 2, 0, 4])

# Random queries strictly inside octants (avoid the coordinate planes)
rng = np.random.default_rng(42)
N = 5000
Q = rng.normal(size=(N, 3))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
# Push away from coordinate planes
mask = np.all(np.abs(Q) > 0.05, axis=1)
Q = Q[mask]

# Convert back to lat/lon to feed through the full pipeline
lat = np.rad2deg(np.arcsin(Q[:, 2]))
lon = np.rad2deg(np.arctan2(Q[:, 1], Q[:, 0]))
queries = np.column_stack([lat, lon])

V = latlon_to_xyz(geolocation[:, 0], geolocation[:, 1])
Qn = latlon_to_xyz(queries[:, 0], queries[:, 1])

tri_pred = brute_force_locate(Qn, V, mesh)
octant_pred = tri_to_octant[tri_pred]
octant_truth = oracle_octahedron(Qn)

n_ok = np.sum(octant_pred == octant_truth)
n_unresolved = np.sum(tri_pred == -1)
print(f"queries:          {len(queries)}")
print(f"resolved:         {len(queries) - n_unresolved}")
print(f"matches oracle:   {n_ok}")
print(f"mismatches:       {len(queries) - n_ok - n_unresolved}")
assert n_unresolved == 0, "some queries unresolved"
assert n_ok == len(queries), "containment disagrees with oracle"
print("OK: containment test agrees with oracle on all queries.")

# Also test the edge cases from the demo (same as in __main__ block)
demo_queries = np.array(
    [
        [45.0, 45.0],
        [45.0, 135.0],
        [-30.0, -45.0],
        [10.0, 1.0],
        [89.9, 200.0],
    ]
)
Qd = latlon_to_xyz(demo_queries[:, 0], demo_queries[:, 1])
tri_d = brute_force_locate(Qd, V, mesh)
oct_d = tri_to_octant[tri_d]
oct_d_truth = oracle_octahedron(Qd)
print("\nDemo queries:")
for q, t, op, ot in zip(demo_queries, tri_d, oct_d, oct_d_truth):
    print(f"  {q} -> tri {t} (octant {op}, truth {ot})")
assert np.all(oct_d == oct_d_truth), "demo queries: mismatch"
print("OK: all demo queries resolve to the correct octant.")
