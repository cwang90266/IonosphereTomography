# -*- coding: utf-8 -*-
"""Phase 3 tests for knot_optimization.py."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from knot_optimization import (GapMap, KnotOptimizationConfig, SpacingProfile, WarpedGapMap,
                               basis_from_grid_file, density_initial_grid, load_grid_file,
                               optimize_knots, save_grid_file)
from spline_basis import ConstrainedSplineBasis
from spline_fit import BatchedSplineFit, prepare_targets

H0, H1 = 90.0, 900.0


def base_cfg(**kw):
    d = dict(h0_km=H0, h1_km=H1, n_points=10, delta_min_km=15.0, fit_space="linear",
             training_files=[], output_path="unused.json")
    d.update(kw)
    return KnotOptimizationConfig(**d)


class TestConfig:
    def test_spacing_threshold_is_configurable_and_validated(self):
        cfg = base_cfg(delta_min_km=20.0, delta_min_first_km=40.0)
        f = cfg.floors(10)
        assert f[0] == 40.0 and np.all(f[1:] == 20.0)
        assert base_cfg().delta_min_first_km == 15.0
        with pytest.raises(ValueError, match="infeasible"):
            base_cfg(n_points=60, delta_min_km=15.0)
        with pytest.raises(ValueError):
            base_cfg(delta_min_km=0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            base_cfg(n_points=10, delta_min_km=88.0)
            assert any("nearly forced" in str(x.message) for x in w)

    def test_other_validation(self):
        with pytest.raises(ValueError):
            base_cfg(fit_space="pow")
        with pytest.raises(ValueError):
            base_cfg(h1_km=50.0)
        with pytest.raises(ValueError):
            base_cfg(n_points=1)

    def test_json_round_trip(self, tmp_path):
        cfg = base_cfg(fit_space="pow", alpha=2.0, delta_min_first_km=30.0)
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        assert KnotOptimizationConfig.from_json(p) == cfg


class TestAltitudeDependentSpacing:
    PROFILE = [[60.0, 10.0], [150.0, 10.0], [250.0, 20.0], [900.0, 20.0]]

    def test_profile_s_and_inverse(self):
        sp = SpacingProfile(*np.array(self.PROFILE).T)
        h = np.linspace(40.0, 950.0, 400)
        np.testing.assert_allclose(sp.h_of_s(sp.s(h)), h, atol=1e-9)
        # ds/dh = 1/delta
        eps = 1e-5
        np.testing.assert_allclose((sp.s(h + eps) - sp.s(h - eps)) / (2 * eps), 1 / sp.delta(h), rtol=1e-6)
        assert sp.delta(np.array([100.0, 200.0, 600.0])) == pytest.approx([10.0, 15.0, 20.0])

    def test_gaps_respect_local_threshold(self):
        cfg = base_cfg(h0_km=60.0, n_points=20, delta_min_km=self.PROFILE)
        gm = cfg.gap_map(20)
        rng = np.random.default_rng(3)
        sp = cfg.spacing_profile()
        for _ in range(30):
            g = gm.grid(rng.normal(scale=4, size=19))
            assert g[0] == 60.0 and g[-1] == H1
            gaps = np.diff(g)
            lo = np.minimum(sp.delta(g[:-1]), sp.delta(g[1:]))
            assert np.all(gaps >= lo - 1e-9)                  # never below the local threshold
            assert np.all(np.diff(sp.s(g)) >= 1 - 1e-9)
            np.testing.assert_allclose(gm.grid(gm.theta(g, min_share=1e-15)), g, atol=1e-6)

    def test_constant_profile_matches_plain_gap_map(self):
        cfg = base_cfg(delta_min_km=15.0, delta_min_first_km=30.0)
        th = np.random.default_rng(4).normal(size=9)
        np.testing.assert_allclose(cfg.gap_map(10).grid(th), GapMap(H0, H1, cfg.floors(10)).grid(th),
                                   atol=1e-9)

    def test_chain_rule(self):
        cfg = base_cfg(h0_km=60.0, n_points=9, delta_min_km=self.PROFILE)
        gm = cfg.gap_map(9)
        rng = np.random.default_rng(5)
        w = rng.normal(size=7)
        f = lambda th: w @ gm.grid(th)[1:-1] + 1e-3 * np.sum(gm.grid(th)[1:-1] ** 2)
        th = rng.normal(size=8)
        dR_dx = w + 2e-3 * gm.grid(th)[1:-1]
        eps = 1e-6
        fd = np.array([(f(th + eps * e) - f(th - eps * e)) / (2 * eps) for e in np.eye(8)])
        np.testing.assert_allclose(gm.chain(th, dR_dx), fd, rtol=1e-5, atol=1e-7)

    def test_validation(self):
        with pytest.raises(ValueError):
            base_cfg(delta_min_km=[[100.0, 10.0], [90.0, 20.0]])
        with pytest.raises(ValueError):
            base_cfg(delta_min_km=[[100.0, -1.0]])
        with pytest.raises(ValueError, match="infeasible"):
            base_cfg(h0_km=60.0, n_points=70, delta_min_km=self.PROFILE)
        assert base_cfg(h0_km=60.0, delta_min_km=self.PROFILE).delta_min_first_km == 10.0


class TestGapMap:
    def test_round_trip_and_floors(self, rng=np.random.default_rng(1)):
        floors = np.array([30.0] + [12.0] * 8)
        gm = GapMap(H0, H1, floors)
        for _ in range(20):
            g = gm.grid(rng.normal(scale=3, size=9))
            assert g[0] == H0 and g[-1] == H1
            assert np.all(np.diff(g) >= floors - 1e-9)
            np.testing.assert_allclose(gm.grid(gm.theta(g, min_share=1e-15)), g, atol=1e-6)

    def test_chain_rule(self):
        rng = np.random.default_rng(2)
        gm = GapMap(H0, H1, np.full(7, 10.0))
        w = rng.normal(size=6)
        f = lambda th: w @ gm.grid(th)[1:-1] + 0.001 * np.sum(gm.grid(th)[1:-1] ** 2)
        th = rng.normal(size=7)
        dR_dx = w + 0.002 * gm.grid(th)[1:-1]
        eps = 1e-6
        fd = np.array([(f(th + eps * e) - f(th - eps * e)) / (2 * eps) for e in np.eye(7)])
        np.testing.assert_allclose(gm.chain(th, dR_dx), fd, rtol=1e-6, atol=1e-8)


def planted_targets(rng, planted, n_mem=40, H=260.0):
    h = np.arange(H0, H1 + 0.1, 5.0)
    cb = ConstrainedSplineBasis(planted, 3, lower_value=0.0, upper_scale_height=H)
    mean = 3e11 * np.exp(-(h - H0) / H)
    r = cb.evaluate(rng.uniform(0.3, 1.7, (n_mem, cb.n_free)), h)
    ne = (mean[None, :] * r).T[:, None, :]
    return prepare_targets(ne, h, h0=H0, h1=H1, mean=mean[:, None])


class TestOptimization:
    def test_recovers_planted_grid(self):
        rng = np.random.default_rng(3)
        planted = np.array([90, 105, 125, 160, 230, 330, 480, 680, 900.0])
        t = planted_targets(rng, planted)
        uni = BatchedSplineFit(t).fit(np.linspace(H0, H1, planted.size)).objective
        res = optimize_knots(t, planted.size, floors=np.full(planted.size - 1, 10.0),
                             n_random=6, rng=rng)
        assert res.objective < 1e-8 * uni
        np.testing.assert_allclose(res.grid, planted, atol=0.5)

    def test_beats_uniform_and_respects_threshold(self):
        rng = np.random.default_rng(4)
        planted = np.array([90, 100, 112, 130, 170, 250, 400, 600, 900.0])
        t = planted_targets(rng, planted)
        floors = np.array([25.0] + [20.0] * 7)
        res = optimize_knots(t, 9, floors=floors, n_random=2, rng=rng)
        uni = BatchedSplineFit(t).fit(np.linspace(H0, H1, 9)).objective
        assert res.objective < 0.5 * uni
        assert np.all(np.diff(res.grid) >= floors - 1e-9)

    def test_density_init_is_valid_grid(self):
        rng = np.random.default_rng(5)
        t = planted_targets(rng, np.array([90, 110, 140, 200, 300, 500, 900.0]))
        g = density_initial_grid(t, 12, 3)
        assert g[0] == H0 and g[-1] == H1 and np.all(np.diff(g) > 0)


class TestGridFile:
    def test_save_load_and_basis(self, tmp_path):
        rng = np.random.default_rng(6)
        t = planted_targets(rng, np.array([90, 120, 180, 300, 500, 900.0]), n_mem=10)
        cfg = base_cfg(n_points=6, output_path=str(tmp_path / "grid.json"))
        res = optimize_knots(t, 6, floors=cfg.floors(6), n_random=0, init=["uniform"])
        save_grid_file(cfg.output_path, res, cfg, t)
        d = load_grid_file(cfg.output_path)
        np.testing.assert_allclose(d["grid_km"], res.grid)
        assert d["spacing"]["delta_min_km"] == 15.0
        assert isinstance(d["spacing"]["gaps_at_threshold"], list)
        json.loads((tmp_path / "grid.json").read_text())
        cb = basis_from_grid_file(cfg.output_path, upper_scale_height=260.0)
        assert cb.n_free == 6
