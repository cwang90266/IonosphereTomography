# -*- coding: utf-8 -*-
"""
Pytest suite for edp_samples.py, scoped to what's testable without the
compiled IRI2020 Fortran executable or a live network connection (see the
module verification plan, section 4.2): geometry generators, mesh
containment/interpolation math, dataset construction and NetCDF round trip,
and the namelist/binary I/O contract with the (absent, here) Fortran driver.

Explicit regressions for bugs found while reading/exercising the code:
- from_xarray's "Point" case reconstructed Lat from ds.attrs["Lon"] (typo),
  silently replacing latitude with longitude on every round trip;
- plot_geolocation/interp had no "Global" case in their geo_type match,
  silently returning None for a geo_type __init__ otherwise builds fine;
- genPolarArea("south", ...) used the wrong angular span and never negated
  latitudes, producing a cap spanning most of the globe instead of a small
  region near -90;
- EDPSamples.__init__'s "Polar" case passed a signed (possibly negative)
  minLat straight through to genPolarArea, which needs a positive magnitude
  regardless of hemisphere;
- the placeholder edps/feature_edps (when evaluate_iri != 1 and no array is
  supplied) used np.ndarray(shape) -- uninitialized memory -- instead of
  np.zeros(shape);
- EDPSamples.interp() queried find_containing_triangles with (lat, lon)
  even though self.geolocation is stored (lon, lat) for every geo_type
  reachable through __init__, silently mismatching every mesh-based
  geo_type despite interp()'s own lat-first "lla" contract; its "ecef"
  path additionally fed interp_heights a raw-meters altitude against a
  km-scaled altitude table.

Note on the geolocation column order question generally: fixing interp()
above did NOT require resolving what the canonical convention *should* be
(see Module_Verification_Plan.md section 2.2/7.1/9, still open) -- only
making interp() internally consistent with what self.geolocation actually,
uniformly contains today across every geo_type __init__ can produce.

Run with:  /opt/anaconda3/bin/python3 -m pytest EDPSamples/ -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import edp_samples as M
from edp_samples import EDPSamples


# ===========================================================================
# Geodetic <-> ECEF
# ===========================================================================

class TestGeodeticEcef:
    @pytest.mark.parametrize("lat,lon,alt", [
        (0.0, 0.0, 0.0),
        (45.0, -73.0, 1000.0),
        (-33.9, 151.2, 500.0),
        (89.9, 12.0, 0.0),
        (-89.9, -170.0, 200.0),
    ])
    def test_round_trip(self, lat, lon, alt):
        xyz = M._geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = M._ecef_to_geodetic(xyz)
        assert lat2 == pytest.approx(lat, abs=1e-6)
        assert lon2 == pytest.approx(lon, abs=1e-6)
        assert alt2 == pytest.approx(alt, abs=1e-3)

    def test_vectorized_round_trip(self):
        lat = np.array([0.0, 45.0, -45.0, 89.0])
        lon = np.array([0.0, 90.0, -90.0, 180.0])
        alt = np.array([0.0, 100.0, 500.0, 1000.0])
        xyz = M._geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = M._ecef_to_geodetic(xyz)
        assert np.allclose(lat2, lat, atol=1e-6)
        assert np.allclose(lon2, lon, atol=1e-6)
        assert np.allclose(alt2, alt, atol=1e-3)


# ===========================================================================
# interp_heights
# ===========================================================================

class TestInterpHeights:
    def test_weights_sum_to_one_inside_table(self):
        table = np.array([0.0, 100.0, 200.0, 300.0])
        query = np.array([10.0, 150.0, 299.0])
        idx, w = M.interp_heights(table, query)
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_reproduces_linear_function_exactly(self):
        table = np.linspace(0.0, 700.0, 15)
        f_table = 3.0 * table + 7.0   # arbitrary linear function
        query = np.array([12.3, 250.0, 699.9, 0.0])
        idx, w = M.interp_heights(table, query)
        f_interp = w[:, 0] * f_table[idx] + w[:, 1] * f_table[idx + 1]
        f_true = 3.0 * query + 7.0
        assert np.allclose(f_interp, f_true, atol=1e-9)

    def test_extrapolation_below_and_above_table(self):
        table = np.array([100.0, 200.0, 300.0])
        idx_lo, w_lo = M.interp_heights(table, np.array([50.0]))
        idx_hi, w_hi = M.interp_heights(table, np.array([400.0]))
        # Weights still sum to 1 but one of them exceeds [0, 1] outside the table.
        assert w_lo.sum() == pytest.approx(1.0)
        assert w_hi.sum() == pytest.approx(1.0)
        assert w_lo[0, 0] > 1.0 or w_lo[0, 1] < 0.0
        assert w_hi[0, 1] > 1.0 or w_hi[0, 0] < 0.0

    def test_raises_on_degenerate_table(self):
        with pytest.raises(ValueError):
            M.interp_heights(np.array([1.0]), np.array([1.0]))


# ===========================================================================
# find_containing_triangles (tested on its own documented [lat, lon] contract)
# ===========================================================================

class TestFindContainingTriangles:
    @pytest.fixture
    def small_mesh(self):
        # A simple 2x2 grid of 4 points, 2 triangles, all as [lat, lon].
        geo = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
        mesh = np.array([[0, 1, 2], [1, 3, 2]])
        return geo, mesh

    def test_centroids_resolve_to_their_own_triangle(self, small_mesh):
        geo, mesh = small_mesh
        centroids = geo[mesh].mean(axis=1)
        idx = M.find_containing_triangles(centroids, geo, mesh)
        assert np.array_equal(idx, np.arange(len(mesh)))

    def test_point_outside_mesh_returns_minus_one(self, small_mesh):
        geo, mesh = small_mesh
        far_away = np.array([[80.0, 170.0]])
        idx = M.find_containing_triangles(far_away, geo, mesh)
        assert idx[0] == -1

    def test_barycentric_weights_reproduce_query_point(self):
        # find_containing_triangles's own docstring: barycentric weights
        # reproduce the *planar projection* of the query point onto the
        # triangle's plane, not its exact position on the sphere -- those
        # two coincide only in the small-triangle limit, so use a
        # realistic sub-degree mesh (~10 km) rather than the 10-degree
        # (~1000 km) small_mesh fixture, where the sphere-vs-plane gap
        # would dominate and swamp the comparison.
        geo = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]])
        mesh = np.array([[0, 1, 2]])
        query = np.array([[0.03, 0.03]])   # interior

        idx, bary = M.find_containing_triangles(query, geo, mesh, return_bary=True)
        assert idx[0] != -1
        tri = mesh[idx[0]]
        vertices_ecef = M._geodetic_to_ecef(geo[tri, 0], geo[tri, 1], 0.0)
        reconstructed = bary[0] @ vertices_ecef
        query_ecef = M._geodetic_to_ecef(query[0, 0], query[0, 1], 0.0)
        assert np.allclose(reconstructed, query_ecef, atol=50.0)   # meters
        assert bary[0].sum() == pytest.approx(1.0)


# ===========================================================================
# Geometry generators
# ===========================================================================

class TestGenRectangularArea:
    def test_vertex_and_triangle_shapes(self):
        vertices, triangles = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)
        assert vertices.ndim == 2 and vertices.shape[1] == 2
        assert triangles.ndim == 2 and triangles.shape[1] == 3

    def test_triangle_indices_in_range(self):
        vertices, triangles = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)
        assert triangles.min() >= 0
        assert triangles.max() < vertices.shape[0]

    def test_no_degenerate_triangles(self):
        vertices, triangles = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)
        for tri in triangles:
            a, b, c = vertices[tri]
            area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            assert area2 > 1e-9

    def test_column_order_is_lon_then_lat_as_documented(self):
        # genRectangularArea's own docstring: "Columns are longitude and latitude."
        vertices, _ = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)
        assert vertices[:, 0].min() >= -10 - 1e-9 and vertices[:, 0].max() <= 10 + 1e-9
        assert vertices[:, 1].min() >= 0 - 1e-9 and vertices[:, 1].max() <= 10 + 1e-9


class TestGenPolarArea:
    @pytest.mark.parametrize("pole,minLat", [("north", 80.0), ("south", 80.0)])
    def test_vertex_and_triangle_validity(self, pole, minLat):
        vertices, triangles = EDPSamples.genPolarArea(pole, minLat, 5.0)
        triangles = np.asarray(triangles)
        assert triangles.min() >= 0
        assert triangles.max() < len(vertices)

    def test_north_cap_latitudes_above_minlat(self):
        vertices, _ = EDPSamples.genPolarArea("north", 80.0, 5.0)
        assert vertices[:, 1].min() >= 80.0 - 1e-9

    def test_south_cap_latitudes_below_negative_minlat(self):
        vertices, _ = EDPSamples.genPolarArea("south", 80.0, 5.0)
        assert vertices[:, 1].max() <= -80.0 + 1e-9

    def test_invalid_pole_raises(self):
        with pytest.raises(ValueError):
            EDPSamples.genPolarArea("east", 80.0, 5.0)

    def test_south_cap_mirrors_north_cap_span_and_ring_count(self):
        """
        Regression: the "south" branch previously used (minLat+90) as the
        angular span and never negated latitudes, so genPolarArea("south",
        80.0, 5.0) spanned +80 down to -90 (170 degrees, most of the globe)
        with ~35 rings, instead of a small ~10-degree cap near -90 with the
        same ring count as the equivalent north cap.
        """
        north_vertices, _ = EDPSamples.genPolarArea("north", 80.0, 5.0)
        south_vertices, _ = EDPSamples.genPolarArea("south", 80.0, 5.0)
        assert south_vertices.shape[0] == north_vertices.shape[0]
        assert south_vertices[:, 1].max() <= -80.0 + 1e-9
        assert south_vertices[:, 1].min() >= -90.0 - 1e-9


class TestGenGlobalArea:
    def test_vertex_count_matches_formula(self):
        dSpace = 30.0
        vertices, triangles = EDPSamples.genGlobalArea(dSpace=dSpace)
        dSpace_rad = np.radians(dSpace)
        expected = max(12, round(4.0 * np.pi / dSpace_rad ** 2))
        assert vertices.shape[0] == expected

    def test_triangle_indices_valid_and_nondegenerate(self):
        vertices, triangles = EDPSamples.genGlobalArea(dSpace=45.0)
        assert triangles.min() >= 0
        assert triangles.max() < vertices.shape[0]
        # Every vertex should participate in at least one triangle (convex hull
        # of points on a sphere should use them all).
        assert set(np.unique(triangles)) == set(range(vertices.shape[0]))

    def test_covers_full_latitude_range(self):
        vertices, _ = EDPSamples.genGlobalArea(dSpace=20.0)
        assert vertices[:, 1].min() < -80.0
        assert vertices[:, 1].max() > 80.0


class TestGenRegionalArea:
    def test_all_vertices_within_radius(self):
        center_lat, center_lon, radius = 32.0, -178.0, 20.0
        vertices, triangles = EDPSamples.genRegionalArea(center_lat, center_lon, radius, 2.5)

        center_hat = M._geodetic_to_ecef(center_lat, center_lon, 0.0)
        center_hat = center_hat / np.linalg.norm(center_hat)
        pts = M._geodetic_to_ecef(vertices[:, 1], vertices[:, 0], 0.0)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        ang_dist = np.degrees(np.arccos(np.clip(pts @ center_hat, -1.0, 1.0)))

        assert ang_dist.max() <= radius + 1e-6

    def test_triangle_indices_valid(self):
        vertices, triangles = EDPSamples.genRegionalArea(32.0, -178.0, 20.0, 2.5)
        assert triangles.min() >= 0
        assert triangles.max() < vertices.shape[0]

    def test_handles_antimeridian_center_without_raising(self):
        """
        Regression target: a center near the antimeridian is exactly the
        case a naive lon/lat-difference-based crop would mishandle (e.g.
        treating vertices at +179 and -179 as ~358 degrees apart instead of
        ~2). genRegionalArea crops via true great-circle distance (an ECEF
        dot product), so this should produce a normal, roughly circular
        region straddling +/-180 rather than an empty or malformed one.
        """
        vertices, triangles = EDPSamples.genRegionalArea(32.0, -178.0, 20.0, 2.5)
        assert vertices.shape[0] > 50
        # A region straddling the antimeridian should have vertices on both
        # sides of it (near +180 and near -180), not clustered on one side.
        assert np.any(vertices[:, 0] > 170.0)
        assert np.any(vertices[:, 0] < -170.0)

    def test_latitude_span_matches_radius(self):
        vertices, _ = EDPSamples.genRegionalArea(32.0, -178.0, 20.0, 2.5)
        assert vertices[:, 1].min() == pytest.approx(12.0, abs=1.0)
        assert vertices[:, 1].max() == pytest.approx(52.0, abs=1.0)

    def test_too_small_radius_raises(self):
        with pytest.raises(ValueError):
            EDPSamples.genRegionalArea(0.0, 0.0, 0.01, 5.0)


class TestGenLineOfSight:
    def test_endpoints_match_inputs(self):
        vertices, segments = EDPSamples.genLineOfSight(
            0.0, 0.0, 500_000.0, 10.0, 10.0, 500_000.0, num_points=5
        )
        assert vertices[0] == pytest.approx([0.0, 0.0], abs=1e-6)
        assert vertices[-1] == pytest.approx([10.0, 10.0], abs=1e-6)

    def test_segment_count(self):
        vertices, segments = EDPSamples.genLineOfSight(
            0.0, 0.0, 500_000.0, 10.0, 10.0, 500_000.0, num_points=6
        )
        assert vertices.shape[0] == 6
        assert segments.shape[0] == 5

    def test_num_points_below_two_raises(self):
        with pytest.raises(ValueError):
            EDPSamples.genLineOfSight(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, num_points=1)


# ===========================================================================
# EDPSamples construction (no Fortran driver: evaluate_iri left at its
# default, or edps supplied explicitly)
# ===========================================================================

def _make_sampling_parameters(n_sample: int) -> pd.DataFrame:
    sp = pd.DataFrame({
        "hour": [12.0] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}
    return sp


class TestEDPSamplesConstructionPoint:
    def test_builds_with_explicit_edps(self):
        n_alt, n_sample = 10, 2
        altitude = np.linspace(100.0, 500.0, n_alt)
        sp = _make_sampling_parameters(n_sample)
        edps = np.zeros((n_alt, 1, n_sample))
        feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), 1, n_sample))

        ds = EDPSamples(DateTime="2026-01-01T00:00:00", geo_type="Point",
                         altitude=altitude, sampling_parameters=sp,
                         Lon=12.5, Lat=45.5, edps=edps, feature_edps=feature_edps)

        assert ds.geolocation.shape == (1, 2)
        # "Point" stores geolocation as [[Lon, Lat]] (see __init__).
        assert ds.geolocation[0, 0] == pytest.approx(12.5)
        assert ds.geolocation[0, 1] == pytest.approx(45.5)

    def test_missing_lon_lat_raises(self):
        altitude = np.linspace(100.0, 500.0, 5)
        sp = _make_sampling_parameters(2)
        with pytest.raises(ValueError):
            EDPSamples(DateTime="2026-01-01", geo_type="Point",
                       altitude=altitude, sampling_parameters=sp)

    def test_placeholder_edps_shape_when_not_evaluating_iri(self):
        n_alt, n_sample = 6, 3
        altitude = np.linspace(100.0, 500.0, n_alt)
        sp = _make_sampling_parameters(n_sample)
        ds = EDPSamples(DateTime="2026-01-01", geo_type="Point",
                         altitude=altitude, sampling_parameters=sp, Lon=0.0, Lat=0.0)
        assert ds.edps.shape == (n_alt, 1, n_sample)
        assert ds.feature_edps.shape == (len(EDPSamples.FEATURE_LABEL), 1, n_sample)

    def test_placeholder_edps_are_deterministically_zero(self):
        """
        Regression: the placeholder used np.ndarray(shape) (uninitialized
        memory) instead of np.zeros(shape), so un-evaluated EDPs were
        non-deterministic garbage -- occasionally NaN/Inf -- rather than a
        predictable, safe default.
        """
        n_alt, n_sample = 6, 3
        altitude = np.linspace(100.0, 500.0, n_alt)
        sp = _make_sampling_parameters(n_sample)
        ds = EDPSamples(DateTime="2026-01-01", geo_type="Point",
                         altitude=altitude, sampling_parameters=sp, Lon=0.0, Lat=0.0)
        assert np.all(ds.edps == 0.0)
        assert np.all(ds.feature_edps == 0.0)

    def test_mismatched_edps_shape_raises(self):
        altitude = np.linspace(100.0, 500.0, 6)
        sp = _make_sampling_parameters(3)
        wrong_edps = np.zeros((5, 1, 3))   # wrong leading (height) dimension
        feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), 1, 3))
        with pytest.raises(AssertionError):
            EDPSamples(DateTime="2026-01-01", geo_type="Point",
                       altitude=altitude, sampling_parameters=sp, Lon=0.0, Lat=0.0,
                       edps=wrong_edps, feature_edps=feature_edps)


class TestEDPSamplesConstructionOtherGeoTypes:
    def _build(self, geo_type, **kwargs):
        n_alt = 5
        altitude = np.linspace(100.0, 500.0, n_alt)
        sp = _make_sampling_parameters(2)
        return EDPSamples(DateTime="2026-01-01", geo_type=geo_type,
                           altitude=altitude, sampling_parameters=sp, **kwargs)

    def test_rectangle(self):
        ds = self._build("Rectangle", minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)
        assert ds.mesh is not None
        assert ds.geolocation.shape[0] > 0

    def test_rectangle_missing_args_raises(self):
        with pytest.raises(ValueError):
            self._build("Rectangle", minLon=-10)

    def test_polar(self):
        ds = self._build("Polar", minLat=80.0, dLat=5.0)
        assert ds.mesh is not None
        assert ds.attrs["pole"] == "north"

    def test_polar_south(self):
        ds = self._build("Polar", minLat=-80.0, dLat=5.0)
        assert ds.attrs["pole"] == "south"

    def test_polar_south_mesh_matches_north_equivalent_size(self):
        """
        Regression: __init__ selects "south" from a negative minLat but
        previously passed that raw negative value straight through to
        genPolarArea, which requires a positive magnitude regardless of
        pole. That produced a ~170-degree cap (most of the globe) instead
        of a small region near -90, with ~65x too many vertices.
        """
        north = self._build("Polar", minLat=80.0, dLat=5.0)
        south = self._build("Polar", minLat=-80.0, dLat=5.0)
        assert south.geolocation.shape[0] == north.geolocation.shape[0]
        assert south.geolocation[:, 1].max() <= -80.0 + 1e-9
        assert south.geolocation[:, 1].min() >= -90.0 - 1e-9

    def test_global_equal_spaced(self):
        ds = self._build("Global", equal_spaced=True, dLat=45.0)
        assert ds.mesh is not None
        assert ds.attrs["equal_spaced"] == 1

    def test_global_rectangular(self):
        ds = self._build("Global", equal_spaced=False, dLat=60.0, dLon=90.0)
        assert ds.mesh is not None
        assert ds.attrs["equal_spaced"] == 0

    def test_occultation_with_explicit_points(self):
        ds = self._build("Occultation", pt1=(0.0, 0.0), pt2=(5.0, 0.0), pt3=(0.0, 5.0),
                          dLat=5.0, dLon=5.0)
        assert ds.mesh is not None
        assert ds.attrs["pt1"] == (0.0, 0.0)

    def test_regional(self):
        ds = self._build("Regional", Lat=32.0, Lon=-178.0, radius=20.0, dLat=2.5)
        assert ds.mesh is not None
        assert ds.geolocation.shape[0] > 50
        assert ds.attrs["radius"] == 20.0

    def test_regional_missing_args_raises(self):
        with pytest.raises(ValueError):
            self._build("Regional", Lat=32.0, Lon=-178.0)   # missing radius, dLat

    def test_invalid_geo_type_raises(self):
        with pytest.raises(ValueError):
            self._build("NotAGeoType")


# ===========================================================================
# NetCDF round trip
# ===========================================================================

class TestNetCDFRoundTrip:
    def test_point_round_trip_preserves_lat_and_lon(self, tmp_path):
        """
        Regression: from_xarray's "Point" case previously reconstructed with
        Lat=ds.attrs["Lon"], silently replacing latitude with longitude on
        every round trip. Uses distinct Lon/Lat so the swap would be caught.
        """
        n_alt, n_sample = 4, 2
        altitude = np.linspace(100.0, 400.0, n_alt)
        sp = _make_sampling_parameters(n_sample)
        edps = np.arange(n_alt * n_sample, dtype=float).reshape(n_alt, 1, n_sample)
        feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), 1, n_sample))
        ds = EDPSamples(DateTime="2026-01-01T00:00:00", geo_type="Point",
                         altitude=altitude, sampling_parameters=sp,
                         Lon=12.5, Lat=45.5, edps=edps, feature_edps=feature_edps)

        path = tmp_path / "point.nc"
        ds.saveNetCDF(str(path))
        restored = EDPSamples.fromNetCDF(str(path))

        assert restored.attrs["Lon"] == pytest.approx(12.5)
        assert restored.attrs["Lat"] == pytest.approx(45.5)
        assert restored.geolocation[0, 0] == pytest.approx(12.5)
        assert restored.geolocation[0, 1] == pytest.approx(45.5)
        assert np.allclose(restored.edps, edps)

    @pytest.mark.parametrize("geo_type,kwargs", [
        ("Rectangle", dict(minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)),
        ("Polar", dict(minLat=80.0, dLat=5.0)),
        ("Global", dict(equal_spaced=True, dLat=45.0)),
        ("Regional", dict(Lat=32.0, Lon=-178.0, radius=20.0, dLat=5.0)),
    ])
    def test_mesh_geo_types_round_trip(self, tmp_path, geo_type, kwargs):
        n_alt = 4
        altitude = np.linspace(100.0, 400.0, n_alt)
        sp = _make_sampling_parameters(2)
        ds = EDPSamples(DateTime="2026-01-01", geo_type=geo_type,
                         altitude=altitude, sampling_parameters=sp, **kwargs)

        path = tmp_path / f"{geo_type}.nc"
        ds.saveNetCDF(str(path))
        restored = EDPSamples.fromNetCDF(str(path))

        assert restored.attrs["geo_type"] == geo_type
        assert np.allclose(restored.altitude, altitude)
        assert restored.geolocation.shape == ds.geolocation.shape
        assert np.allclose(restored.edps, ds.edps)
        assert np.array_equal(restored.mesh, ds.mesh)


# ===========================================================================
# Fortran driver I/O contract (namelist writer, binary reader) -- no
# subprocess call, so these run without the compiled executable.
# ===========================================================================

class TestWriteIRI2020Namelist:
    def test_namelist_contains_expected_fields(self, tmp_path):
        altitude = np.array([100.0, 200.0, 300.0])
        geolocation = np.array([[10.0, 20.0], [30.0, 40.0]])   # [lon, lat] per the writer's own indexing
        sp = pd.DataFrame({
            "hour": [12.0, np.nan], "f107": [120.0, 130.0],
            "ap": [10.0, 20.0], "ig12": [80.0, 90.0], "rz12": [70.0, 75.0],
        })
        path = tmp_path / "namelist.nml"

        flag = M.write_IRI2020_namelist("2026-03-01T06:00:00", altitude, geolocation, sp, str(path))
        assert flag == 0

        text = path.read_text()
        assert "npts  = 2" in text
        assert "nheight  = 3" in text
        assert "nSample  = 2" in text
        assert "year  = 2026" in text
        assert "month  = 3" in text
        assert "hour  = 6" in text
        assert "height_grid(1)  = 100.0" in text
        # latitude(idx) uses geolocation[:,1], longitude(idx) uses geolocation[:,0]
        assert "latitude(1)  = 20.0" in text
        assert "longitude(1)  = 10.0" in text
        # NaN sample value substituted with the fill value
        assert "phy_inputs(1,2)  = -99999.0" in text
        assert "phy_inputs(2,1)  = 120.0" in text


class TestReadIRI2020BinaryOutput:
    @staticmethod
    def _build_buffer(npts, nSample, nheight, edps, features):
        """
        edps: (nheight, npts, nSample); features: (13, npts, nSample) --
        matches read_IRI2020_binary_output's documented output layout, used
        here in reverse to construct a synthetic input buffer.
        """
        header = (
            int(npts).to_bytes(4, "little", signed=True)
            + int(nSample).to_bytes(4, "little", signed=True)
            + int(nheight).to_bytes(4, "little", signed=True)
        )
        # File layout is (sample, pts, [heights + 13 features]); build by
        # inverting read_IRI2020_binary_output's own transpose.
        edps_slice = np.transpose(edps, (2, 1, 0))        # (nSample, npts, nheight)
        features_slice = np.transpose(features, (2, 1, 0))  # (nSample, npts, 13)
        reshaped = np.concatenate([edps_slice, features_slice], axis=2)
        body = reshaped.astype(np.float32).tobytes()
        return header + body

    def test_round_trip_recovers_known_values(self, tmp_path):
        npts, nSample, nheight = 2, 3, 4
        rng = np.random.default_rng(0)
        edps = rng.normal(size=(nheight, npts, nSample)).astype(np.float32)
        features = rng.normal(size=(13, npts, nSample)).astype(np.float32)

        path = tmp_path / "iri_out.dat"
        path.write_bytes(self._build_buffer(npts, nSample, nheight, edps, features))

        edps_out, features_out = M.read_IRI2020_binary_output(str(path))
        assert edps_out.shape == (nheight, npts, nSample)
        assert features_out.shape == (13, npts, nSample)
        assert np.allclose(edps_out, edps, atol=1e-5)
        assert np.allclose(features_out, features, atol=1e-5)

    def test_truncated_file_raises_value_error(self, tmp_path):
        npts, nSample, nheight = 2, 2, 3
        edps = np.zeros((nheight, npts, nSample), dtype=np.float32)
        features = np.zeros((13, npts, nSample), dtype=np.float32)
        buf = self._build_buffer(npts, nSample, nheight, edps, features)

        path = tmp_path / "truncated.dat"
        path.write_bytes(buf[:-4])   # drop the last float

        with pytest.raises(ValueError):
            M.read_IRI2020_binary_output(str(path))


# ===========================================================================
# EDPSamples.interp()
# ===========================================================================

class TestInterp:
    def _make_point(self):
        n_alt = 5
        altitude = np.linspace(100.0, 500.0, n_alt)
        sp = _make_sampling_parameters(2)
        return EDPSamples(DateTime="2026-01-01", geo_type="Point",
                           altitude=altitude, sampling_parameters=sp, Lon=0.0, Lat=0.0)

    def _make_rectangle(self):
        n_alt = 5
        altitude = np.linspace(100.0, 500.0, n_alt)
        sp = _make_sampling_parameters(2)
        return EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                           altitude=altitude, sampling_parameters=sp,
                           minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)

    def test_point_lla_altitude_interpolation(self):
        ds = self._make_point()
        idx_alt, w_alt = ds.interp(np.array([[0.0, 0.0, 250.0]]), coordinate="lla")
        # altitude grid [100,200,300,400,500]: 250 is exactly between idx 1 and 2
        assert idx_alt[0] == 1
        assert w_alt[0] == pytest.approx([0.5, 0.5])

    def test_point_ecef_altitude_interpolation(self):
        ds = self._make_point()
        xyz_m = M._geodetic_to_ecef(0.0, 0.0, 250_000.0)
        xyz_km = (xyz_m / 1000.0).reshape(1, 3)
        idx_alt, w_alt = ds.interp(xyz_km, coordinate="ecef")
        assert idx_alt[0] == 1
        assert w_alt[0] == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_rectangle_lla_resolves_known_triangle_centroid(self):
        """
        Regression: interp() previously queried find_containing_triangles
        with (lat, lon) while self.geolocation is stored (lon, lat), so a
        genuine lat-first "lla" query (as interp()'s own contract promises)
        would mismatch and typically return -1 instead of the correct
        triangle.
        """
        ds = self._make_rectangle()
        tri0 = ds.mesh[0]
        lon_c, lat_c = ds.geolocation[tri0].mean(axis=0)

        idx_alt, w_alt, idx_mesh, w_mesh = ds.interp(
            np.array([[lat_c, lon_c, 250.0]]), coordinate="lla")

        assert idx_mesh[0] == 0
        assert w_mesh[0].sum() == pytest.approx(1.0)

    def test_rectangle_ecef_resolves_known_triangle_centroid(self):
        ds = self._make_rectangle()
        tri0 = ds.mesh[0]
        lon_c, lat_c = ds.geolocation[tri0].mean(axis=0)
        xyz_m = M._geodetic_to_ecef(lat_c, lon_c, 250_000.0)
        xyz_km = (xyz_m / 1000.0).reshape(1, 3)

        idx_alt, w_alt, idx_mesh, w_mesh = ds.interp(xyz_km, coordinate="ecef")

        assert idx_mesh[0] == 0
        assert idx_alt[0] == 1
        assert w_alt[0] == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_rectangle_point_outside_mesh_returns_minus_one(self):
        ds = self._make_rectangle()
        idx_alt, w_alt, idx_mesh, w_mesh = ds.interp(
            np.array([[80.0, 170.0, 250.0]]), coordinate="lla")
        assert idx_mesh[0] == -1


# ===========================================================================
# Line-of-sight TEC integration: get_observation_operator vs
# forward_model_mesh_tec, cross-checked against a hand-derived analytic value.
# ===========================================================================

class TestLineOfSightTEC:
    """
    Both methods integrate density along a ray via ray_points -> midpoints ->
    altitude lookup -> np.interp/dl summation. Ground truth here uses a
    purely RADIAL ray (GNSS and LEO along the same lat/lon, differing only
    in altitude), so altitude increases linearly with path length and the
    analytic TEC for a *constant* density N0 across [alt_lo, alt_hi] is
    exactly N0 * (alt_hi - alt_lo) [m] / 1e16 -- independent of exactly how
    the ray is discretized, which is what makes it a clean oracle.
    """

    ALTITUDE = np.linspace(100.0, 500.0, 5)   # [100, 200, 300, 400, 500] km
    N0 = 1e11                                  # constant density, m^-3
    NUM_SEGMENTS = 5000
    # Analytic: only the [100, 500] km portion of the 0->1000 km radial ray
    # contributes; 400 km * 1000 m/km * N0 / 1e16.
    EXPECTED_TEC = N0 * 400_000.0 / 1e16

    @staticmethod
    def _radial_ray(lat, lon, alt_top_km=1000.0):
        gnss = (M._geodetic_to_ecef(lat, lon, 0.0) / 1000.0).reshape(3, 1)
        leo = (M._geodetic_to_ecef(lat, lon, alt_top_km * 1000.0) / 1000.0).reshape(3, 1)
        return {"LEO": leo, "GNSS": gnss}

    def _make_point(self):
        sp = _make_sampling_parameters(1)
        edps = np.full((len(self.ALTITUDE), 1, 1), self.N0)
        feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), 1, 1))
        return EDPSamples(DateTime="2026-01-01", geo_type="Point",
                           altitude=self.ALTITUDE, sampling_parameters=sp,
                           Lon=0.0, Lat=0.0, edps=edps, feature_edps=feature_edps)

    def _make_rectangle(self):
        sp = _make_sampling_parameters(1)
        n_geo_probe = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)[0].shape[0]
        edps = np.full((len(self.ALTITUDE), n_geo_probe, 1), self.N0)
        feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), n_geo_probe, 1))
        return EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                           altitude=self.ALTITUDE, sampling_parameters=sp,
                           minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5,
                           edps=edps, feature_edps=feature_edps)

    def test_forward_model_point_matches_analytic_radial_integral(self):
        ds = self._make_point()
        podTc2 = self._radial_ray(0.0, 0.0)
        tec = ds.forward_model_mesh_tec(podTc2, sample_idx=0, num_segments=self.NUM_SEGMENTS)
        assert tec[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)

    def test_observation_operator_point_matches_analytic_radial_integral(self):
        ds = self._make_point()
        podTc2 = self._radial_ray(0.0, 0.0)
        H = ds.get_observation_operator(podTc2, num_segments=self.NUM_SEGMENTS)
        tec_via_H = H @ ds.edps.reshape(-1)
        assert tec_via_H[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)

    def test_forward_model_mesh_matches_analytic_radial_integral(self):
        """Spatially uniform density: horizontal interpolation should be a no-op."""
        ds = self._make_rectangle()
        podTc2 = self._radial_ray(3.0, 2.0)   # interior to a triangle, not on a grid line
        tec = ds.forward_model_mesh_tec(podTc2, sample_idx=0, num_segments=self.NUM_SEGMENTS)
        assert tec[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)

    def test_observation_operator_mesh_matches_analytic_radial_integral(self):
        ds = self._make_rectangle()
        podTc2 = self._radial_ray(3.0, 2.0)
        H = ds.get_observation_operator(podTc2, num_segments=self.NUM_SEGMENTS)
        edps_flat = ds.edps.reshape(-1, ds.edps.shape[2])[:, 0]
        tec_via_H = H @ edps_flat
        assert tec_via_H[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)

    def _make_rectangle_with_latitude_gradient(self, center_lat):
        """A field that varies smoothly (linearly) with latitude, equal to
        N0 exactly at `center_lat` -- lets a ray positioned at that latitude
        have a known analytic TEC even though the field isn't uniform."""
        sp = _make_sampling_parameters(1)
        geo, _ = EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)
        n_geo = geo.shape[0]
        field = self.N0 * (1.0 + 0.05 * (geo[:, 1] - center_lat))
        edps = np.tile(field[None, :, None], (len(self.ALTITUDE), 1, 1))
        ds = EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                         altitude=self.ALTITUDE, sampling_parameters=sp,
                         minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5,
                         edps=edps, feature_edps=np.zeros((len(EDPSamples.FEATURE_LABEL), n_geo, 1)))
        return ds

    def test_observation_operator_matches_analytic_for_smoothly_varying_field(self):
        """
        Regression: get_observation_operator queried find_containing_
        triangles (and its cKDTree fallback) with (lat, lon) against
        self.geolocation, which is stored (lon, lat) -- the same mismatch
        just fixed in interp(). A spatially uniform field can't expose this
        (any blend of one constant is that constant), so this uses a field
        that varies linearly with latitude, evaluated at a ray positioned
        exactly at that field's reference latitude/longitude (interior to a
        triangle, not on a grid line) -- interpolation should recover N0
        there regardless of which neighboring vertices contribute.
        Previously this was ~5% off; now matches to noise-level precision.
        """
        ds = self._make_rectangle_with_latitude_gradient(center_lat=3.0)
        podTc2 = self._radial_ray(3.0, 2.0)
        H = ds.get_observation_operator(podTc2, num_segments=self.NUM_SEGMENTS)
        edps_flat = ds.edps.reshape(-1, ds.edps.shape[2])[:, 0]
        tec_via_H = H @ edps_flat
        assert tec_via_H[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)

    def test_forward_model_and_observation_operator_agree_on_mesh(self):
        """
        Cross-check between the two independently-coded implementations on
        a random (non-uniform) field, rather than against the analytic
        oracle -- catches a regression in either one that happens to still
        pass the (single-value) analytic check via cancellation. Only
        meaningful at an interior, non-grid-aligned ray position: at a
        shared mesh vertex/edge, find_containing_triangles and
        LinearNDInterpolator's independently-built Delaunay triangulation
        can legitimately pick different (but each internally valid)
        triangles, which is a real property of comparing two different
        triangulations of the same points -- not a bug -- and would make
        this comparison meaningless there.
        """
        ds = self._make_rectangle()
        rng = np.random.default_rng(0)
        n_alt, n_geo, _ = ds.edps.shape
        varied_edps = rng.uniform(1e9, 1e11, size=(n_alt, n_geo, 1))
        ds2 = EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                          altitude=self.ALTITUDE, sampling_parameters=_make_sampling_parameters(1),
                          minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5,
                          edps=varied_edps, feature_edps=np.zeros((len(EDPSamples.FEATURE_LABEL), n_geo, 1)))

        podTc2 = self._radial_ray(3.0, 2.0)   # interior to a triangle, not on a grid line
        tec_direct = ds2.forward_model_mesh_tec(podTc2, sample_idx=0, num_segments=self.NUM_SEGMENTS)
        H = ds2.get_observation_operator(podTc2, num_segments=self.NUM_SEGMENTS)
        tec_via_H = H @ varied_edps.reshape(-1, 1)[:, 0]

        assert tec_via_H[0] == pytest.approx(tec_direct[0], rel=0.01)


# ===========================================================================
# subset_region / subset_union_triangles
# ===========================================================================

class TestSubsetRegion:
    def _make_rectangle_ds(self):
        n_alt = 4
        altitude = np.linspace(100.0, 400.0, n_alt)
        sp = _make_sampling_parameters(2)
        return EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                           altitude=altitude, sampling_parameters=sp,
                           minLon=-20, maxLon=20, dLon=5, minLat=-10, maxLat=10, dLat=5)

    def test_subset_keeps_only_vertices_in_bbox(self):
        ds = self._make_rectangle_ds()
        sub = ds.subset_region(lat_min=-10, lat_max=10, lon_min=-5, lon_max=5)
        lon, lat = sub.geolocation[:, 0], sub.geolocation[:, 1]
        assert np.all(lon >= -5 - 1e-9) and np.all(lon <= 5 + 1e-9)
        assert sub.geolocation.shape[0] <= ds.geolocation.shape[0]

    def test_subset_mesh_indices_stay_valid(self):
        ds = self._make_rectangle_ds()
        sub = ds.subset_region(lat_min=-10, lat_max=10, lon_min=-5, lon_max=5)
        if sub.mesh is not None and len(sub.mesh) > 0:
            assert sub.mesh.max() < sub.geolocation.shape[0]

    def test_too_few_vertices_raises(self):
        ds = self._make_rectangle_ds()
        with pytest.raises(ValueError):
            ds.subset_region(lat_min=9.99, lat_max=9.999, lon_min=19.99, lon_max=19.999)


# ===========================================================================
# Plotting: headless smoke tests only
# ===========================================================================

class TestPlotGeolocationSmoke:
    def _build(self, geo_type, **kwargs):
        n_alt = 4
        altitude = np.linspace(100.0, 400.0, n_alt)
        sp = _make_sampling_parameters(2)
        return EDPSamples(DateTime="2026-01-01", geo_type=geo_type,
                           altitude=altitude, sampling_parameters=sp, **kwargs)

    @pytest.mark.parametrize("geo_type,kwargs", [
        ("Point", dict(Lon=0.0, Lat=0.0)),
        ("Rectangle", dict(minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)),
        ("Polar", dict(minLat=80.0, dLat=5.0)),
        ("Global", dict(equal_spaced=True, dLat=45.0)),
        ("Regional", dict(Lat=32.0, Lon=-178.0, radius=20.0, dLat=5.0)),   # antimeridian-straddling
    ])
    def test_plot_geolocation_does_not_raise(self, geo_type, kwargs):
        import matplotlib.pyplot as plt
        ds = self._build(geo_type, **kwargs)
        ds.plot_geolocation()   # regression: Global previously had no case (silent no-op, not a crash either -- but this now actually draws)
        plt.close("all")


class TestPlotHorizontalField:
    def _build(self, geo_type, **kwargs):
        n_alt, n_sample = 4, 2
        altitude = np.linspace(100.0, 400.0, n_alt)
        sp = _make_sampling_parameters(n_sample)
        ds = EDPSamples(DateTime="2026-01-01", geo_type=geo_type,
                         altitude=altitude, sampling_parameters=sp, **kwargs)
        rng = np.random.default_rng(0)
        n_geo = ds.geolocation.shape[0]
        edps = rng.uniform(1e9, 1e11, size=(n_alt, n_geo, n_sample))
        return EDPSamples(DateTime="2026-01-01", geo_type=geo_type,
                           altitude=altitude, sampling_parameters=sp,
                           edps=edps, feature_edps=np.zeros((len(EDPSamples.FEATURE_LABEL), n_geo, n_sample)),
                           **kwargs)

    @pytest.mark.parametrize("geo_type,kwargs", [
        ("Point", dict(Lon=0.0, Lat=0.0)),                     # no mesh -> scatter fallback
        ("Rectangle", dict(minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)),
        ("Polar", dict(minLat=80.0, dLat=5.0)),
        ("Global", dict(equal_spaced=True, dLat=45.0)),
        ("Regional", dict(Lat=32.0, Lon=-178.0, radius=20.0, dLat=5.0)),   # antimeridian-straddling
    ])
    @pytest.mark.parametrize("scalar_kind", ["Ne", "mean", "median", "std"])
    def test_string_scalar_does_not_raise(self, geo_type, kwargs, scalar_kind):
        import matplotlib.pyplot as plt
        ds = self._build(geo_type, **kwargs)
        ds.plot_horizontal_field(scalar=scalar_kind, target_alt=250.0)
        plt.close("all")

    def test_array_scalar_does_not_raise(self):
        import matplotlib.pyplot as plt
        ds = self._build("Rectangle", minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)
        values = np.linspace(0.0, 1.0, ds.geolocation.shape[0])
        ax = ds.plot_horizontal_field(scalar=values)
        assert ax is not None
        plt.close("all")

    def test_string_scalar_without_target_alt_raises(self):
        ds = self._build("Rectangle", minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)
        with pytest.raises(ValueError):
            ds.plot_horizontal_field(scalar="Ne")

    def test_unknown_scalar_string_raises(self):
        ds = self._build("Rectangle", minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)
        with pytest.raises(ValueError):
            ds.plot_horizontal_field(scalar="not_a_field", target_alt=250.0)

    def test_wrong_length_array_raises(self):
        ds = self._build("Rectangle", minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)
        with pytest.raises(ValueError):
            ds.plot_horizontal_field(scalar=np.zeros(ds.geolocation.shape[0] + 1))

    def test_custom_scalar_label_used_on_colorbar(self):
        import matplotlib.pyplot as plt
        ds = self._build("Rectangle", minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5)
        values = np.linspace(0.0, 1.0, ds.geolocation.shape[0])
        ax = ds.plot_horizontal_field(scalar=values, scalar_label="My Custom Label")
        colorbar_axes = ax.figure.axes[-1]   # the horizontal colorbar is appended last
        assert colorbar_axes.get_xlabel() == "My Custom Label"
        plt.close("all")
