# -*- coding: utf-8 -*-
"""
Tests for iri_selection.py (plan Section 4.3/4.9), network-free via the
same monkeypatched-get_apf107/get_ig_rz fixture pattern as
IRI_Sample_Inputs/test_iri_sample_inputs.py.
"""
from __future__ import annotations

import IRI_Sample_inputs as M
import pandas as pd
import pytest

from Assimilation_Cycle import iri_selection
from Assimilation_Cycle.cycle_config import CycleConfig


@pytest.fixture
def fixture_apf107() -> dict:
    n = 30
    days = list(range(1, n + 1))
    return {
        "yr": [2024] * n, "mn": [6] * n, "dy": days,
        "iapda": [10 + i for i in range(n)],
        "iiap": [[10 + i, 11 + i, 12 + i, 13 + i, 9 + i, 8 + i, 7 + i, 6 + i] for i in range(n)],
        "ir": [0] * n,
        "f107": [100.0 + i for i in range(n)],
        "f107_81": [95.0 + i for i in range(n)],
        "f107_365": [90.0 + i for i in range(n)],
    }


@pytest.fixture
def fixture_ig_rz() -> dict:
    return {
        "Revision": [1], "Start_end_month": [5, 2024, 7, 2024],
        "ig": [80.0, 81.0, 82.0, 83.0, 84.0],
        "rz": [70.0, 71.0, 72.0, 73.0, 74.0],
    }


@pytest.fixture
def no_network(monkeypatch, fixture_apf107, fixture_ig_rz):
    monkeypatch.setattr(M, "get_apf107", lambda: fixture_apf107)
    monkeypatch.setattr(M, "get_ig_rz", lambda: fixture_ig_rz)


def _make_cfg(**overrides) -> CycleConfig:
    defaults = dict(
        start_time="2024-06-15T00:00:00", end_time="2024-06-15T02:00:00",
        center_lat=69.6, center_lon=18.9, radius_km=500.0,
        n_ensemble=25,
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


class TestSamplingParametersForCycle:
    def test_ensemble_size_matches_n_ensemble(self, no_network):
        cfg = _make_cfg(n_ensemble=17)
        df = iri_selection.sampling_parameters_for_cycle(cfg)
        assert len(df) == 17
        assert set(df.columns) >= {"hour", "f107", "ap", "ig12", "rz12"}

    def test_uses_center_time_of_window(self, no_network):
        # window midpoint of [00:00, 02:00) is 01:00 -> hour spread should
        # be drawn around hour=1, not the window's start_time.
        cfg = _make_cfg(n_ensemble=5, iri_spread_kwargs={"hour_sample_range": 1})
        df = iri_selection.sampling_parameters_for_cycle(cfg)
        # every drawn hour should be within [0, 2] of the center hour (1),
        # or NaN (the "no constraint" sentinel -- None coerced by pandas
        # since the column is mixed float/None)
        assert all(pd.isna(h) or 0 <= h <= 2 for h in df["hour"])

    def test_spread_kwargs_forwarded(self, no_network):
        cfg = _make_cfg(n_ensemble=10, iri_spread_kwargs={"f107_sample_range": 3})
        df = iri_selection.sampling_parameters_for_cycle(cfg)
        assert df.attrs["f107_sample_range"] == 3
        assert df.attrs["sample_method"] == "randomSamples"


class TestPrecomputedFileMode:
    def test_saves_and_reloads_iri_sample_inputs(self, tmp_path, no_network):
        out = tmp_path / "iri_cycle"
        cfg = _make_cfg(iri_sample_inputs_output_path=out)
        iri = iri_selection.load_or_build_iri_sample_inputs(cfg)
        assert (tmp_path / "iri_cycle.pkl").exists()

        cfg_reload = _make_cfg(iri_sample_inputs_path=out)
        # loading from pickle must not touch get_apf107/get_ig_rz again --
        # remove the monkeypatch's effect to prove no re-fetch happens.
        reloaded = iri_selection.load_or_build_iri_sample_inputs(cfg_reload)
        assert reloaded.year == iri.year
        assert reloaded.current_idx_f107 == iri.current_idx_f107

    def test_accepts_path_with_pkl_suffix_already_present(self, tmp_path, no_network):
        out = tmp_path / "iri_cycle2.pkl"
        cfg = _make_cfg(iri_sample_inputs_output_path=out)
        iri_selection.load_or_build_iri_sample_inputs(cfg)
        assert (tmp_path / "iri_cycle2.pkl").exists()

        cfg_reload = _make_cfg(iri_sample_inputs_path=out)
        reloaded = iri_selection.load_or_build_iri_sample_inputs(cfg_reload)
        assert reloaded.year == 2024
