# -*- coding: utf-8 -*-
"""
Tests for ``impact_vectors.py`` (``Impact_Vector_Plan.md``) -- the core
exactness identity (Section 3/Section 9 tests 1-3) against real
``EDPSamples`` geometry (for ``build_f_edp``'s mesh/altitude
interpolation) and real ``Parameterization`` styles (for the decode
Jacobian), but a synthetic ``H_grid``/ensemble (no IRI2020 needed, no ray
geometry needed -- matching ``Ensemble_Kalman_Engine/tests/test_observation_operator_integration.py``'s
own established "real styles, synthetic grid" convention) with a couple
of hand-assigned fake "RO entries" splitting the synthetic observations
into row-ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

Parameterization = pytest.importorskip(
    "Parameterization", reason="Parameterization.py and its dependencies not importable in this environment"
)

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "EDPSamples",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import edp_samples as E  # noqa: E402

from Ensemble_Kalman_Engine.analysis_engine import AnalysisEngine, AnalysisConfig
from Ensemble_Kalman_Engine.ensemble_state import EnsembleState
from Ensemble_Kalman_Engine.observation_operator import GenericObservationOperator
from Ensemble_Kalman_Engine.osse import generate_osse_observation

from Assimilation_Cycle.impact_vectors import (
    QueryPoint, ImpactVectorAccumulator, build_f_edp,
    plot_impact_vector_heatmap, plot_impact_vector_ranking, plot_impact_vector_ranking_cross_style,
    isr_correction_direction, rank_by_isr_alignment,
    plot_impact_vector_isr_alignment, plot_impact_vector_isr_aligned_heatmap,
    save_impact_vectors, load_impact_vectors, replot_impact_vectors_from_saved, replot_impact_vectors_tree,
)


@dataclass(frozen=True)
class _FakeEntry:
    label: str


def _make_sampling_parameters(n_sample: int) -> pd.DataFrame:
    sp = pd.DataFrame({
        "hour": [12.0] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}
    return sp


def _build_edp_samples(n_alt=5, minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5):
    """A small, real (mesh-bearing) EDPSamples instance -- geometry only
    needed here (altitude/geolocation/mesh for build_f_edp), not a real
    IRI2020-drawn ensemble."""
    altitude = np.linspace(100.0, 500.0, n_alt)
    sp = _make_sampling_parameters(2)
    geom_only = E.EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                              altitude=altitude, sampling_parameters=sp,
                              minLon=minLon, maxLon=maxLon, dLon=dLon,
                              minLat=minLat, maxLat=maxLat, dLat=dLat)
    n_geo = geom_only.geolocation.shape[0]
    edps = np.ones((n_alt, n_geo, 2)) * 1e10
    return E.EDPSamples(DateTime="2026-01-01", geo_type="Rectangle",
                         altitude=altitude, sampling_parameters=sp, edps=edps,
                         feature_edps=np.zeros((len(E.EDPSamples.FEATURE_LABEL), n_geo, 2)),
                         minLon=minLon, maxLon=maxLon, dLon=dLon,
                         minLat=minLat, maxLat=maxLat, dLat=dLat)


def _query_point_at_grid_center(edp_samples, n_alt) -> QueryPoint:
    lon, lat = edp_samples.geolocation.mean(axis=0)
    return QueryPoint(name="center", lat=float(lat), lon=float(lon),
                       altitudes=np.asarray(edp_samples.altitude, dtype=float))


def _decoded_value_at_query_point(obs_op, edp_samples, qp: QueryPoint, x: np.ndarray) -> np.ndarray:
    """Independent ground truth for the exactness check: decode ``x`` to
    physical density, then interpolate at ``qp`` directly via
    ``build_f_edp`` applied to the *decoded* field -- deliberately not
    reusing any impact_vectors.py internals beyond build_f_edp itself
    (the thing both the production code and this check share, already
    covered by its own, separate construction)."""
    density = obs_op.decode(x.reshape(*obs_op.param_shape))   # (n_height, n_geo)
    F = build_f_edp(edp_samples, qp.lat, qp.lon, qp.altitudes)
    return F @ density.reshape(-1)


class TestBuildFEdp:
    def test_exact_altitude_match_gives_weight_one_on_that_level(self):
        edp_samples = _build_edp_samples()
        lon, lat = edp_samples.geolocation[0]
        F = build_f_edp(edp_samples, lat, lon, edp_samples.altitude)
        # Each requested altitude exactly matches a native grid level ->
        # each output row should be a pure (no cross-altitude blending)
        # barycentric combination of that level's own 3 mesh vertices.
        n_height = len(edp_samples.altitude)
        n_geo = edp_samples.geolocation.shape[0]
        assert F.shape == (n_height, n_height * n_geo)
        for h in range(n_height):
            row = F[h]
            other_levels = np.delete(row.reshape(n_height, n_geo), h, axis=0)
            assert np.allclose(other_levels, 0.0)
            assert row.sum() == pytest.approx(1.0, abs=1e-8)

    def test_off_grid_altitude_blends_two_neighboring_levels(self):
        edp_samples = _build_edp_samples()
        lon, lat = edp_samples.geolocation[0]
        altitude = np.asarray(edp_samples.altitude)
        mid = np.array([0.5 * (altitude[0] + altitude[1])])
        F = build_f_edp(edp_samples, lat, lon, mid)
        n_geo = edp_samples.geolocation.shape[0]
        reshaped = F[0].reshape(len(altitude), n_geo)
        nonzero_levels = np.where(np.abs(reshaped).sum(axis=1) > 1e-12)[0]
        assert set(nonzero_levels.tolist()) == {0, 1}
        assert reshaped.sum() == pytest.approx(1.0, abs=1e-8)


class TestImpactVectorExactness:
    """Two genuinely distinct properties, found to need separating while
    writing these tests (the original single-test design conflated them,
    and failed in a way that revealed why they're different):

    (A) **Partition exactness** -- splitting the observations into
        multiple ROs and summing their impacts must reproduce *exactly*
        what a single undivided "RO" covering all the same observations
        would report. Always exact, for every style including nonlinear
        ones (plan Section 3: a column/row partition identity, no
        approximation involved) -- Section 9 tests 1-3 are really this.
    (B) **Physical-change accuracy** -- does the summed impact match the
        *actual* decoded (``decode(x_final) - decode(x_mean)``) change at
        the query point? Exact for a linear decode (``raw``), but only a
        first-order (Jacobian-at-each-iterate) *approximation* for a
        genuinely nonlinear decode (``density_10ex``/``ANCHOR``/PCA
        styles) -- confirmed empirically while building this test: the
        relative error shrinks roughly linearly with the perturbation
        scale (0.3 -> 27%, 0.03 -> 2%, 0.001 -> 0.07%), the signature of a
        linearization remainder, not a bug. ``Impact_Vector_Plan.md``
        Section 3's original claim that this holds "unconditionally"
        under ``centering="mean"`` was only correct for property (A); it
        overstated (B) for nonlinear styles and has been corrected there
        to match what's actually true (plan doc updated same day this
        was found)."""

    def _run(self, style, hyper_params, linearization, entry_ray_ranges_spec,
              n_height=5, n_obs=7, spread=1.0, x_true_scale=1.5, seed=0):
        """``entry_ray_ranges_spec``: list of (label, slice) -- kept
        separate from the RNG-consuming setup above it, so two calls with
        the same ``seed`` but different ``entry_ray_ranges_spec`` (e.g.
        one split two ways, one as a single combined entry) see bit-
        identical observations/ensembles/trajectories, isolating property
        (A) from everything else."""
        edp_samples = _build_edp_samples(n_alt=n_height)
        n_geo = edp_samples.geolocation.shape[0]
        param_shape = (n_height, n_geo)

        rng = np.random.default_rng(seed)
        H_grid = rng.standard_normal((n_obs, n_height * n_geo)) * 0.05
        parameterization = Parameterization.EDP_Parameterization(style=style, hyper_params=hyper_params)
        obs_op = GenericObservationOperator(
            parameterization=parameterization, H_grid=H_grid, param_shape=param_shape,
            n_height=n_height, n_geo=n_geo, altitude=np.asarray(edp_samples.altitude),
        )

        n_state = int(np.prod(param_shape))
        n_members = 150
        X_f = 10.0 + spread * rng.standard_normal((n_state, n_members))
        ensemble = EnsembleState(X=X_f, param_shape=param_shape)

        x_true = 10.0 + x_true_scale * rng.standard_normal(n_state)
        R = 0.02 * np.eye(n_obs)
        osse = generate_osse_observation(x_true, obs_op, R, rng=np.random.default_rng(seed + 1))

        entries = [_FakeEntry(label) for label, _ in entry_ray_ranges_spec]
        entry_ray_ranges = [(e, s) for e, (_, s) in zip(entries, entry_ray_ranges_spec)]

        qp = _query_point_at_grid_center(edp_samples, n_height)
        acc = ImpactVectorAccumulator(edp_samples, obs_op, [qp], entry_ray_ranges)

        config = AnalysisConfig(linearization=linearization, centering="mean", rng=np.random.default_rng(seed + 2))
        _, diag = AnalysisEngine().analyze(
            ensemble, obs_op, osse.y_obs, R, config, on_iteration=acc.on_iteration,
        )
        impacts = acc.finalize(diag.alpha_history)

        x_mean = ensemble.X.mean(axis=1)
        value_before = _decoded_value_at_query_point(obs_op, edp_samples, qp, x_mean)
        value_after = _decoded_value_at_query_point(obs_op, edp_samples, qp, diag.x_mean_a)
        actual_change = value_after - value_before

        total_impact = sum(impacts[(e.label, qp.name)] for e in entries)
        return total_impact, actual_change, diag, impacts, qp

    @pytest.mark.parametrize("style,hyper_params,linearization,kwargs", [
        ("raw", None, "linear", {}),
        ("density_10ex", None, "iterated", dict(spread=0.3, x_true_scale=0.3)),
        ("density_10ex", None, "iterated", dict(spread=0.5, x_true_scale=1.0, n_obs=5, seed=3)),
    ], ids=["raw_linear", "nonlinear_no_damping", "nonlinear_with_damping"])
    def test_partition_exactness_two_way_split_matches_single_combined_entry(
        self, style, hyper_params, linearization, kwargs,
    ):
        """Property (A): splitting the same observations across 2 ROs and
        summing their impacts must exactly match treating them as one
        combined RO -- an always-exact identity, independent of whether
        the decode is linear."""
        n_obs = kwargs.get("n_obs", 7)
        split = n_obs // 2
        two_way = [("RO_A", slice(0, split)), ("RO_B", slice(split, n_obs))]
        combined = [("RO_ALL", slice(0, n_obs))]

        total_split, _, diag_split, _, qp = self._run(style, hyper_params, linearization, two_way, **kwargs)
        total_combined, _, diag_combined, _, _ = self._run(style, hyper_params, linearization, combined, **kwargs)

        np.testing.assert_allclose(total_split, total_combined, atol=1e-6, rtol=1e-6)
        assert diag_split.alpha_history == diag_combined.alpha_history
        if linearization == "iterated":
            assert diag_split.n_iterations >= 1
        if kwargs.get("seed") == 3:   # the "nonlinear_with_damping" case specifically
            assert any(a < 1.0 for a in diag_split.alpha_history), (
                "test setup didn't actually force damping -- adjust x_true_scale/spread"
            )

    def test_raw_style_also_matches_the_true_decoded_change_exactly(self):
        """Property (B) for 'raw' specifically: decode is literally the
        identity there, so (A)'s always-exact total *also* equals the
        real decoded forecast->analysis change -- not true in general for
        a nonlinear style (see the class docstring)."""
        total_impact, actual_change, diag, *_ = self._run(
            "raw", None, "linear", [("RO_A", slice(0, 3)), ("RO_B", slice(3, 7))],
        )
        np.testing.assert_allclose(total_impact, actual_change, atol=1e-6, rtol=1e-6)
        assert diag.n_iterations == 1

    def test_nonlinear_style_approximation_error_shrinks_with_perturbation_scale(self):
        """Property (B) for a nonlinear style: not exact, but must
        converge toward the true decoded change as the problem becomes
        more linear (smaller forecast spread / smaller true departure) --
        the actual signature confirmed while building this test. A flat
        or growing error here would mean something is really wrong, not
        just "approximate"."""
        errors = []
        for scale in (0.1, 0.01, 0.001):
            total_impact, actual_change, _, _, _ = self._run(
                "density_10ex", None, "iterated",
                [("RO_A", slice(0, 3)), ("RO_B", slice(3, 7))],
                spread=scale, x_true_scale=scale,
            )
            errors.append(np.linalg.norm(total_impact - actual_change) / max(np.linalg.norm(actual_change), 1e-12))
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.01, f"error should be small at a near-linear scale, got {errors[-1]}"

    def test_per_ro_split_is_nontrivial_not_all_on_one_entry(self):
        """Sanity check that the per-RO attribution is actually doing
        something (not, e.g., accidentally attributing everything to one
        entry due to an indexing bug)."""
        _, _, _, impacts, qp = self._run(
            "raw", None, "linear", [("RO_A", slice(0, 3)), ("RO_B", slice(3, 7))], seed=5,
        )
        a = impacts[("RO_A", qp.name)]
        b = impacts[("RO_B", qp.name)]
        assert np.any(np.abs(a) > 1e-8)
        assert np.any(np.abs(b) > 1e-8)


class TestVisualization:
    """Plan Section 8, first half -- heatmap + plain-magnitude ranking,
    per-style and cross-style. Following this project's established
    plot-test convention (test_isr_comparison.py's TestPlotsRun): does it
    run, are the basic axes/labels sane -- not pixel-level checks."""

    def _toy_impacts(self, n_alt=6, seed=0):
        rng = np.random.default_rng(seed)
        altitude = np.linspace(90.0, 600.0, n_alt)
        impacts_by_ro = {
            "RO_1": rng.normal(scale=1e10, size=n_alt),
            "RO_2": rng.normal(scale=5e9, size=n_alt),
            "RO_3": rng.normal(scale=2e10, size=n_alt),
        }
        return impacts_by_ro, altitude

    def test_heatmap_runs_and_orders_by_total_abs_impact(self):
        import matplotlib.pyplot as plt
        impacts_by_ro, altitude = self._toy_impacts()
        ax = plot_impact_vector_heatmap(impacts_by_ro, altitude)
        expected_order = sorted(impacts_by_ro, key=lambda l: -np.abs(impacts_by_ro[l]).sum())
        assert [t.get_text() for t in ax.get_yticklabels()] == expected_order
        assert ax.get_xlabel() == "Altitude (km)"
        plt.close("all")

    def test_heatmap_color_scale_is_symmetric_around_zero(self):
        import matplotlib.pyplot as plt
        impacts_by_ro, altitude = self._toy_impacts()
        ax = plot_impact_vector_heatmap(impacts_by_ro, altitude)
        mesh = ax.collections[-1]
        vmin, vmax = mesh.get_clim()
        assert vmin == -vmax
        plt.close("all")

    def test_ranking_bar_chart_runs(self):
        import matplotlib.pyplot as plt
        impacts_by_ro, _ = self._toy_impacts()
        ax = plot_impact_vector_ranking(impacts_by_ro)
        assert len(ax.patches) == len(impacts_by_ro)
        plt.close("all")

    def test_cross_style_ranking_runs_with_matching_ros(self):
        import matplotlib.pyplot as plt
        impacts_a, _ = self._toy_impacts(seed=1)
        impacts_b, _ = self._toy_impacts(seed=2)
        ax = plot_impact_vector_ranking_cross_style({"StyleA": impacts_a, "StyleB": impacts_b})
        assert len(ax.patches) == 2 * len(impacts_a)
        assert ax.get_legend() is not None
        plt.close("all")

    def test_cross_style_ranking_handles_ro_missing_from_one_style(self):
        """A style whose batch didn't include some RO (or only one
        style's run computed impact vectors) must not error -- missing
        entries contribute a zero bar instead."""
        import matplotlib.pyplot as plt
        impacts_a, _ = self._toy_impacts(seed=1)
        impacts_b = {k: v for k, v in self._toy_impacts(seed=2)[0].items() if k != "RO_2"}
        ax = plot_impact_vector_ranking_cross_style({"StyleA": impacts_a, "StyleB": impacts_b})
        assert ax is not None
        plt.close("all")


class TestIsrAlignment:
    """Section 9 test 5: a hand-constructed case with a known sign by
    inspection, AND genuine per-altitude sign variation (both within the
    ISR-correction direction itself and within one RO's own profile) --
    so the heatmap's per-cell recoloring is actually checked per-cell,
    not just per-RO."""

    # 4 altitudes; ISR coverage matches the model grid exactly (no
    # interpolation distortion) so d's values are exactly computable by
    # hand: d = M - F = [+0.2e10, -0.2e10, +0.2e10, -0.2e10].
    model_altitude = np.array([100.0, 200.0, 300.0, 400.0])
    isr_altitude = model_altitude.copy()
    forecast_profile = np.full(4, 1.0e10)
    isr_density = np.array([1.2e10, 0.8e10, 1.2e10, 0.8e10])
    d_expected = np.array([0.2e10, -0.2e10, 0.2e10, -0.2e10])

    def _impacts(self):
        d = self.d_expected
        return {
            "RO_TOWARD": 0.5 * d,     # aligned with d at every altitude -> positive projection
            "RO_AWAY": -0.5 * d,      # anti-aligned everywhere -> negative projection
            "RO_MIXED": np.array([0.1e10, 0.1e10, -0.1e10, -0.1e10]),   # not proportional to d
        }

    def test_correction_direction_matches_hand_computed_value(self):
        d, mask = isr_correction_direction(
            self.forecast_profile, self.isr_density, self.isr_altitude, self.model_altitude,
        )
        np.testing.assert_allclose(d, self.d_expected, rtol=1e-6)
        assert mask.all()

    def test_correction_direction_masks_altitudes_outside_isr_coverage(self):
        extended_model_altitude = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 900.0])
        d, mask = isr_correction_direction(
            np.full(6, 1.0e10), self.isr_density, self.isr_altitude, extended_model_altitude,
        )
        assert not mask[0] and not mask[-1]       # 50km and 900km: outside [100,400]
        assert mask[1:5].all()                     # 100..400km: inside
        assert d[0] == 0.0 and d[-1] == 0.0        # masked entries contribute zero, not nan

    def test_rank_by_isr_alignment_recovers_known_sign_and_ordering(self):
        ranked = rank_by_isr_alignment(
            self._impacts(), self.forecast_profile, self.isr_density, self.isr_altitude, self.model_altitude,
        )
        assert ranked["RO_TOWARD"]["projection"] > 0
        assert ranked["RO_AWAY"]["projection"] < 0
        np.testing.assert_allclose(ranked["RO_TOWARD"]["projection"], -ranked["RO_AWAY"]["projection"])
        # RO_MIXED: projection = sum([0.1e10*0.2e10, 0.1e10*-0.2e10, -0.1e10*0.2e10, -0.1e10*-0.2e10])
        expected_mixed = 0.1e10 * 0.2e10 - 0.1e10 * 0.2e10 - 0.1e10 * 0.2e10 + 0.1e10 * 0.2e10
        np.testing.assert_allclose(ranked["RO_MIXED"]["projection"], expected_mixed, rtol=1e-6)

    def test_cosine_is_bounded_and_exact_for_parallel_or_orthogonal_impacts(self):
        """RO_TOWARD/RO_AWAY are exactly proportional to ``d`` (``+-0.5*d``)
        -- cosine similarity for a vector with itself (or its negation)
        is exactly +-1 regardless of either vector's scale, unlike the
        raw ``projection`` whose magnitude depends on both. RO_MIXED is
        exactly orthogonal to ``d`` (its projection is 0), so its cosine
        is exactly 0 too."""
        ranked = rank_by_isr_alignment(
            self._impacts(), self.forecast_profile, self.isr_density, self.isr_altitude, self.model_altitude,
        )
        np.testing.assert_allclose(ranked["RO_TOWARD"]["cosine"], 1.0, rtol=1e-6)
        np.testing.assert_allclose(ranked["RO_AWAY"]["cosine"], -1.0, rtol=1e-6)
        np.testing.assert_allclose(ranked["RO_MIXED"]["cosine"], 0.0, atol=1e-12)
        for key in ranked:
            assert -1.0 - 1e-9 <= ranked[key]["cosine"] <= 1.0 + 1e-9

    def test_alignment_ranking_plot_orders_most_helpful_first(self):
        import matplotlib.pyplot as plt
        ax = plot_impact_vector_isr_alignment(
            self._impacts(), self.forecast_profile, self.isr_density, self.isr_altitude, self.model_altitude,
        )
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels[0] == "RO_TOWARD"
        assert labels[-1] == "RO_AWAY"
        plt.close("all")

    def test_alignment_ranking_plot_bars_are_cosine_not_raw_projection(self):
        """Bar lengths must be the bounded ``cosine`` metric (exactly
        +-1 here), not the raw ``projection`` (which would be +-0.1e20
        for these inputs) -- confirms the 2026-11-06 switch actually
        took effect in the plotted values, not just in ``rank_by_isr_alignment``'s
        returned dict."""
        import matplotlib.pyplot as plt
        ax = plot_impact_vector_isr_alignment(
            self._impacts(), self.forecast_profile, self.isr_density, self.isr_altitude, self.model_altitude,
        )
        widths = [abs(patch.get_width()) for patch in ax.patches]
        assert all(w <= 1.0 + 1e-9 for w in widths)
        assert max(widths) == pytest.approx(1.0, rel=1e-6)
        plt.close("all")

    def test_aligned_heatmap_per_cell_values_match_sign_recoloring_exactly(self):
        """The real point of this test: confirm recoloring happens
        per-(RO, altitude) cell, not per-RO -- RO_MIXED's own row must
        show genuinely mixed sign."""
        import matplotlib.pyplot as plt
        impacts = self._impacts()
        ax = plot_impact_vector_isr_aligned_heatmap(
            impacts, self.forecast_profile, self.isr_density, self.isr_altitude, self.model_altitude,
        )
        mesh = ax.collections[-1]
        M = np.asarray(mesh.get_array()).reshape(len(impacts), len(self.model_altitude))
        labels = [t.get_text() for t in ax.get_yticklabels()]

        sign_d = np.sign(self.d_expected)
        for row_idx, label in enumerate(labels):
            expected_row = impacts[label] * sign_d
            np.testing.assert_allclose(M[row_idx], expected_row, rtol=1e-6)
        # RO_MIXED's row must have both a positive and a negative cell.
        mixed_row = M[labels.index("RO_MIXED")]
        assert np.any(mixed_row > 0) and np.any(mixed_row < 0)
        plt.close("all")

    def test_aligned_heatmap_blanks_altitudes_outside_isr_coverage(self):
        import matplotlib.pyplot as plt
        extended_model_altitude = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 900.0])
        impacts = {"RO_X": np.ones(6)}
        ax = plot_impact_vector_isr_aligned_heatmap(
            impacts, np.ones(6), self.isr_density, self.isr_altitude, extended_model_altitude,
        )
        mesh = ax.collections[-1]
        M = np.ma.getmaskarray(mesh.get_array()).reshape(1, 6)
        assert M[0, 0] and M[0, -1]          # 50km, 900km: masked (outside coverage)
        assert not M[0, 1:5].any()           # 100..400km: not masked
        plt.close("all")


class TestSaveLoadReplotImpactVectors:
    """The point of persisting these at all (user request, 2026-11-06):
    a plotting-only change should regenerate every plot straight from a
    saved file, with no diagnostic pass/ensemble/pipeline involved."""

    altitude = np.array([100.0, 200.0, 300.0, 400.0])
    impacts_by_ro = {"RO_A": np.array([1.0, 2.0, -1.0, 0.5]), "RO_B": np.array([-0.5, 0.0, 1.0, 2.0])}

    def test_round_trip_without_isr_data_preserves_impacts_and_leaves_isr_fields_none(self, tmp_path):
        path = save_impact_vectors(
            tmp_path / "batch0000_data.nc", self.impacts_by_ro, self.altitude,
            style="ANCHOR", batch_index=0, query_point_name="My Site",
        )
        saved = load_impact_vectors(path)
        assert saved.style == "ANCHOR"
        assert saved.batch_index == 0
        assert saved.query_point_name == "My Site"
        assert set(saved.impacts_by_ro) == set(self.impacts_by_ro)
        for label in self.impacts_by_ro:
            np.testing.assert_allclose(saved.impacts_by_ro[label], self.impacts_by_ro[label])
        np.testing.assert_allclose(saved.altitude, self.altitude)
        assert saved.forecast_profile is None
        assert saved.isr_density is None
        assert saved.isr_altitude is None

    def test_round_trip_with_isr_data_preserves_it_exactly(self, tmp_path):
        forecast_profile = np.array([1.0e10, 1.1e10, 1.2e10, 1.0e10])
        isr_density = np.array([1.05e10, 1.25e10])
        isr_altitude = np.array([150.0, 350.0])
        path = save_impact_vectors(
            tmp_path / "batch0000_data.nc", self.impacts_by_ro, self.altitude,
            forecast_profile=forecast_profile, isr_density=isr_density, isr_altitude=isr_altitude,
        )
        saved = load_impact_vectors(path)
        np.testing.assert_allclose(saved.forecast_profile, forecast_profile)
        np.testing.assert_allclose(saved.isr_density, isr_density)
        np.testing.assert_allclose(saved.isr_altitude, isr_altitude)

    def test_replot_from_saved_without_isr_data_writes_only_the_four_non_isr_plots(self, tmp_path):
        import matplotlib.pyplot as plt
        data_path = save_impact_vectors(
            tmp_path / "batch0000_data.nc", self.impacts_by_ro, self.altitude, query_point_name="My Site",
        )
        written = replot_impact_vectors_from_saved(data_path, tmp_path, file_prefix="batch0000")
        names = {p.name for p in written}
        assert names == {
            "batch0000_heatmap.png", "batch0000_ranking.png",
            "batch0000_ranking_relative_norm.png", "batch0000_ranking_relative_local.png",
        }
        for p in written:
            assert p.exists() and p.stat().st_size > 0
        plt.close("all")

    def test_replot_from_saved_with_isr_data_also_writes_the_two_isr_plots(self, tmp_path):
        import matplotlib.pyplot as plt
        data_path = save_impact_vectors(
            tmp_path / "batch0000_data.nc", self.impacts_by_ro, self.altitude, query_point_name="My Site",
            forecast_profile=np.array([1.0e10, 1.1e10, 1.2e10, 1.0e10]),
            isr_density=np.array([1.05e10, 1.25e10]), isr_altitude=np.array([150.0, 350.0]),
        )
        written = replot_impact_vectors_from_saved(data_path, tmp_path, file_prefix="batch0000")
        names = {p.name for p in written}
        assert "batch0000_isr_alignment.png" in names
        assert "batch0000_isr_aligned_heatmap.png" in names
        plt.close("all")

    def test_replot_tree_finds_and_regenerates_every_saved_file_under_root(self, tmp_path):
        import matplotlib.pyplot as plt
        (tmp_path / "ANCHOR" / "impact_vectors" / "Site A").mkdir(parents=True)
        (tmp_path / "PCA_1D_10ex" / "impact_vectors" / "Site A").mkdir(parents=True)
        save_impact_vectors(
            tmp_path / "ANCHOR" / "impact_vectors" / "Site A" / "batch0000_data.nc",
            self.impacts_by_ro, self.altitude, style="ANCHOR", batch_index=0, query_point_name="Site A",
        )
        save_impact_vectors(
            tmp_path / "PCA_1D_10ex" / "impact_vectors" / "Site A" / "batch0000_data.nc",
            self.impacts_by_ro, self.altitude, style="PCA_1D_10ex", batch_index=0, query_point_name="Site A",
        )
        written = replot_impact_vectors_tree(tmp_path)
        assert len(written) == 8   # 4 non-ISR plots x 2 saved files
        assert (tmp_path / "ANCHOR" / "impact_vectors" / "Site A" / "batch0000_ranking.png").exists()
        assert (tmp_path / "PCA_1D_10ex" / "impact_vectors" / "Site A" / "batch0000_ranking.png").exists()
        plt.close("all")
