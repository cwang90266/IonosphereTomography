# -*- coding: utf-8 -*-
"""Tests for cycle_config.py's grid-margin math (grid_margin_km ->
effective_grid_radius_deg) -- the rest of CycleConfig is exercised
implicitly throughout the other test modules."""
from __future__ import annotations

import numpy as np
import pytest

from Assimilation_Cycle.cycle_config import CycleConfig


def _make_cfg(**overrides):
    defaults = dict(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=19.2, radius_km=2000.0,
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


class TestEffectiveGridRadiusDeg:
    def test_zero_margin_matches_resolved_grid_radius(self):
        cfg = _make_cfg(grid_radius_deg=18.0, grid_margin_km=0.0)
        assert cfg.effective_grid_radius_deg == cfg.resolved_grid_radius_deg == 18.0

    def test_margin_adds_degrees_on_top(self):
        cfg = _make_cfg(grid_radius_deg=18.0, grid_margin_km=500.0)
        expected_margin_deg = np.degrees(500.0 / 6371.0)
        assert cfg.effective_grid_radius_deg == pytest.approx(18.0 + expected_margin_deg)

    def test_margin_applies_on_top_of_the_radius_km_fallback_too(self):
        # grid_radius_deg unset -> resolved_grid_radius_deg falls back to
        # radius_km's bare number; margin still adds on top of that.
        cfg = _make_cfg(radius_km=5.0, grid_radius_deg=None, grid_margin_km=500.0)
        expected_margin_deg = np.degrees(500.0 / 6371.0)
        assert cfg.effective_grid_radius_deg == pytest.approx(5.0 + expected_margin_deg)

    def test_default_margin_is_zero(self):
        cfg = _make_cfg(grid_radius_deg=10.0)
        assert cfg.grid_margin_km == 0.0
        assert cfg.effective_grid_radius_deg == 10.0


class TestInterBatchInflationFactorDefault:
    def test_defaults_to_none_no_behavior_change(self):
        cfg = _make_cfg()
        assert cfg.inter_batch_inflation_factor is None


class TestDiagonalBoostValidation:
    def test_defaults_to_none_no_behavior_change(self):
        cfg = _make_cfg()
        assert cfg.diagonal_boost_amplitude is None
        assert cfg.diagonal_boost_log_space is False
        assert cfg.diagonal_boost_vertical_scale_km == 30.0
        assert cfg.diagonal_boost_horizontal_scale_km == 200.0
        assert cfg.diagonal_boost_taper_start_km == 400.0
        assert cfg.diagonal_boost_taper_end_km == 700.0
        assert cfg.diagonal_boost_taper_floor == 0.1

    def test_negative_amplitude_rejected(self):
        with pytest.raises(ValueError, match="diagonal_boost_amplitude"):
            _make_cfg(diagonal_boost_amplitude=-0.1)

    def test_taper_end_not_after_start_rejected(self):
        with pytest.raises(ValueError, match="diagonal_boost_taper_end_km"):
            _make_cfg(diagonal_boost_taper_start_km=700.0, diagonal_boost_taper_end_km=400.0)

    def test_taper_floor_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="diagonal_boost_taper_floor"):
            _make_cfg(diagonal_boost_taper_floor=1.5)
        with pytest.raises(ValueError, match="diagonal_boost_taper_floor"):
            _make_cfg(diagonal_boost_taper_floor=-0.1)

    def test_zero_amplitude_allowed_as_explicit_noop(self):
        cfg = _make_cfg(diagonal_boost_amplitude=0.0)
        assert cfg.diagonal_boost_amplitude == 0.0

    def test_nonpositive_vertical_scale_rejected(self):
        with pytest.raises(ValueError, match="diagonal_boost_vertical_scale_km"):
            _make_cfg(diagonal_boost_vertical_scale_km=0.0)

    def test_nonpositive_horizontal_scale_rejected(self):
        with pytest.raises(ValueError, match="diagonal_boost_horizontal_scale_km"):
            _make_cfg(diagonal_boost_horizontal_scale_km=-5.0)


class TestIriSpreadKwargsDefault:
    """Confirmed with the user 2026-09-29 after directly inspecting the
    underlying IRI2020 input files' real cadences -- previously defaulted
    to `{}` (no spread at all), a real bug: a fresh-build ensemble with
    that default is 2000 identical IRI2020 runs."""

    def test_defaults_to_real_windows_not_empty(self):
        cfg = _make_cfg()
        assert cfg.iri_spread_kwargs == {
            "hour_sample_range": 3,
            "f107_sample_range": 30,
            "ap_sample_range": 30,
            "ig_sample_range": 12,
            "rz_sample_range": 12,
        }

    def test_explicit_empty_override_still_means_no_spread(self):
        """An explicit {} must still work as "no spread" (existing
        precomputed-file-mode tests rely on this) -- the new default only
        applies when the field is left unset entirely."""
        cfg = _make_cfg(iri_spread_kwargs={})
        assert cfg.iri_spread_kwargs == {}

    def test_two_separately_constructed_configs_dont_share_the_dict(self):
        """default_factory, not a bare mutable default -- mutating one
        config's dict must not leak into another's."""
        cfg1 = _make_cfg()
        cfg2 = _make_cfg()
        cfg1.iri_spread_kwargs["hour_sample_range"] = 999
        assert cfg2.iri_spread_kwargs["hour_sample_range"] == 3
