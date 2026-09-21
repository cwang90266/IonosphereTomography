# -*- coding: utf-8 -*-
"""
Pytest suite for IRI_Sample_inputs.py.

Two layers, matching the module verification plan's section 4.1:

- TestGetApf107 / TestGetIgRz exercise the fixed-width/comma-delimited file
  parsers directly, against small local fixture files (via tmp_path +
  monkeypatch.chdir) so no network access is needed and a network.get call
  would fail the test loudly if the "use cached file" path regresses.
- Everything else monkeypatches get_apf107/get_ig_rz to return small,
  fully-controlled dicts and constructs IRI_Sample_Inputs through its real
  __init__, so the sampling logic is tested against known values rather than
  needing to hand-derive expectations from real IRI driving-index history.

Two of these tests are explicit regressions for bugs found only by tracing
what quantileSamples actually computes (not by reading it):
- ig12 was being drawn from the f107 quantile range, not the ig12 range;
- the "ap" quantile triple used min()/max() on a list of 8-value rows, which
  compares rows lexicographically rather than finding the true min/max
  across all their values.

Run with:  /opt/anaconda3/bin/python3 -m pytest IRI_Sample_Inputs/ -q
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

import IRI_Sample_inputs as M


# ===========================================================================
# Fixed-width / comma-delimited file fixtures for the raw parsers
# ===========================================================================

def _apf107_line(yy2: int, mn: int, dy: int, iiap: list[int], iapda: int,
                  ir: int, f107: float, f107_81: float, f107_365: float) -> str:
    line = f"{yy2:3d}{mn:3d}{dy:3d}"
    line += "".join(f"{v:3d}" for v in iiap)
    line += f"{iapda:3d}{ir:3d}{f107:5.1f}{f107_81:5.1f}{f107_365:5.1f}"
    assert len(line) == 54, f"fixture line has wrong width: {len(line)}"
    return line


class TestGetApf107:
    def test_parses_fixture_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lines = [
            _apf107_line(24, 6, 14, [10, 11, 12, 13, 9, 8, 7, 6], 20, 0, 120.5, 115.3, 110.0),
            _apf107_line(24, 6, 15, [11, 12, 13, 14, 10, 9, 8, 7], 21, 0, 121.0, 116.0, 111.0),
        ]
        (tmp_path / "apf107.dat").write_text("\n".join(lines) + "\n")

        result = M.get_apf107()

        assert result["yr"] == [2024, 2024]
        assert result["mn"] == [6, 6]
        assert result["dy"] == [14, 15]
        assert result["iiap"][0] == [10, 11, 12, 13, 9, 8, 7, 6]
        assert result["iapda"] == [20, 21]
        assert result["ir"] == [0, 0]
        assert result["f107"] == pytest.approx([120.5, 121.0])
        assert result["f107_81"] == pytest.approx([115.3, 116.0])
        assert result["f107_365"] == pytest.approx([110.0, 111.0])

    def test_year_rollover_2000s_vs_1900s(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lines = [
            _apf107_line(29, 1, 1, [0] * 8, 0, 0, 100.0, 100.0, 100.0),   # 29 -> 2029
            _apf107_line(30, 1, 1, [0] * 8, 0, 0, 100.0, 100.0, 100.0),   # 30 -> 1930
        ]
        (tmp_path / "apf107.dat").write_text("\n".join(lines) + "\n")

        result = M.get_apf107()
        assert result["yr"] == [2029, 1930]

    def test_uses_local_cache_without_network_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "apf107.dat").write_text(
            _apf107_line(24, 6, 14, [0] * 8, 0, 0, 100.0, 100.0, 100.0) + "\n"
        )

        def _fail(*a, **k):
            raise AssertionError("get_apf107 should not hit the network when a local cache exists")

        monkeypatch.setattr(M.requests, "get", _fail)
        M.get_apf107()   # must not raise


class TestGetIgRz:
    @staticmethod
    def _write_fixture(tmp_path, ig_values, rz_values,
                        start_month=5, start_year=2024, end_month=7, end_year=2024):
        lines = [
            "1,2024",
            f"{start_month},{start_year},{end_month},{end_year}",
        ]
        lines += [str(v) for v in ig_values]
        lines += [str(v) for v in rz_values]
        (tmp_path / "ig_rz.dat").write_text("\n".join(lines) + "\n")

    def test_parses_fixture_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fixture(tmp_path, [80.0, 81.0, 82.0], [70.0, 71.0, 72.0])

        result = M.get_ig_rz()

        assert result["Start_end_month"] == [5, 2024, 7, 2024]
        assert result["ig"] == pytest.approx([80.0, 81.0, 82.0])
        assert result["rz"] == pytest.approx([70.0, 71.0, 72.0])

    def test_raises_on_too_short_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ig_rz.dat").write_text("1,2024\n5,2024,7,2024\n80.0\n")
        with pytest.raises(ValueError):
            M.get_ig_rz()


# ===========================================================================
# IRI_Sample_Inputs: monkeypatched fixtures for the higher-level sampling logic
# ===========================================================================

@pytest.fixture
def fixture_apf107() -> dict:
    """30 days of June 2024 with strictly increasing, individually-checkable values."""
    n = 30
    days = list(range(1, n + 1))
    return {
        "yr": [2024] * n,
        "mn": [6] * n,
        "dy": days,
        "iapda": [10 + i for i in range(n)],
        "iiap": [[10 + i, 11 + i, 12 + i, 13 + i, 9 + i, 8 + i, 7 + i, 6 + i] for i in range(n)],
        "ir": [0] * n,
        "f107": [100.0 + i for i in range(n)],
        "f107_81": [95.0 + i for i in range(n)],
        "f107_365": [90.0 + i for i in range(n)],
    }


@pytest.fixture
def fixture_ig_rz() -> dict:
    """
    Start_end_month=[5,2024,7,2024] expands (see IRI_Sample_Inputs.__init__)
    to the month list [4,5,6,7,8] of 2024, so ig/rz need exactly 5 entries.
    """
    return {
        "Revision": [1],
        "Start_end_month": [5, 2024, 7, 2024],
        "ig": [80.0, 81.0, 82.0, 83.0, 84.0],
        "rz": [70.0, 71.0, 72.0, 73.0, 74.0],
    }


@pytest.fixture
def make_iri(monkeypatch, fixture_apf107, fixture_ig_rz):
    """Factory: build an IRI_Sample_Inputs via its real __init__, network-free."""
    def _make(date_str: str = "2024-06-15"):
        monkeypatch.setattr(M, "get_apf107", lambda: fixture_apf107)
        monkeypatch.setattr(M, "get_ig_rz", lambda: fixture_ig_rz)
        return M.IRI_Sample_Inputs(date_str)
    return _make


class TestIRISampleInputsInit:
    def test_date_only_string_defaults_time_to_zero(self, make_iri):
        ds = make_iri("2024-06-15")
        assert (ds.year, ds.month, ds.day) == (2024, 6, 15)
        assert (ds.hour, ds.minute, ds.second) == (0, 0, 0)

    def test_full_datetime_string_preserves_time(self, make_iri):
        ds = make_iri("2024-06-15T14:30:45")
        assert (ds.hour, ds.minute, ds.second) == (14, 30, 45)

    def test_current_idx_f107_matches_requested_day(self, make_iri):
        ds = make_iri("2024-06-15")
        assert ds.current_idx_f107 == 14   # day 15 is index 14 in a 1..30 range

    def test_current_idx_igrz_matches_requested_month(self, make_iri):
        ds = make_iri("2024-06-15")
        assert ds.current_idx_igrz == 2   # month 6 is index 2 in [4,5,6,7,8]

    def test_date_outside_apf107_coverage_raises(self, make_iri):
        with pytest.raises(ValueError):
            make_iri("2024-07-01")   # apf107 fixture only covers June 2024


class TestSaveLoadPickle:
    def test_round_trip(self, make_iri, tmp_path):
        ds = make_iri("2024-06-15")
        stem = str(tmp_path / "iri_inputs")
        ds.save_to_file(stem)
        restored = M.IRI_Sample_Inputs.fromPickle(stem)

        assert (restored.year, restored.month, restored.day) == (ds.year, ds.month, ds.day)
        assert restored.current_idx_f107 == ds.current_idx_f107
        assert restored.apf107 == ds.apf107


# ===========================================================================
# quantileSamples
# ===========================================================================

class TestQuantileSamples:
    def test_all_none_ranges_give_single_row(self, make_iri):
        ds = make_iri("2024-06-15")
        result = ds.quantileSamples()
        assert len(result) == 1
        assert result.iloc[0][["hour", "f107", "ap", "ig12", "rz12"]].isna().all()

    def test_f107_only_gives_min_mean_max_and_none(self, make_iri):
        ds = make_iri("2024-06-15")
        # current_idx_f107=14, range=3 -> slice[11:17] -> f107 values 111..116
        result = ds.quantileSamples(f107_sample_range=3)
        assert len(result) == 4   # 1 * 4 * 1 * 1 * 1
        values = set(result["f107"].dropna().round(6))
        assert values == {111.0, 113.5, 116.0}
        assert result["f107"].isna().any()

    def test_ig12_drawn_from_ig_range_not_f107_range(self, make_iri):
        """
        Regression: previously the innermost loop was `for ig12 in
        f107_range`. With f107_sample_range left unset (f107_range=[None]),
        that bug would force every 'ig12' value to None regardless of
        ig_sample_range. current_idx_igrz=2, ig=[80,81,82,83,84],
        ig_sample_range=1 -> slice[1:3] -> [81,82] -> {81, 81.5, 82, None}.
        """
        ds = make_iri("2024-06-15")
        result = ds.quantileSamples(ig_sample_range=1)
        ig12_values = set(result["ig12"].dropna().round(6))
        assert ig12_values == {81.0, 81.5, 82.0}
        assert result["ig12"].isna().any()

    def test_ap_quantile_uses_true_global_min_and_max(self, monkeypatch, fixture_ig_rz):
        """
        Regression: ap_range = [max(0, min(min(ap_range))), mean, max(max(ap_range))]
        compares 8-value rows lexicographically (list comparison), not the
        true min/max scalar across all values. Two rows are constructed so
        the lexicographically-smallest/largest row does NOT contain the true
        global min/max, which the old code would get wrong.
        """
        apf107 = {
            "yr": [2024, 2024, 2024], "mn": [6, 6, 6], "dy": [14, 15, 16],
            "iapda": [0, 0, 0], "ir": [0, 0, 0],
            "iiap": [
                [3, 50, 50, 50, 50, 50, 50, 100],   # lexicographically smallest (starts with 3)
                [4, 50, 50, 50, 50, 50, 50, 1],     # contains the true global min (1)
                [5, 50, 50, 50, 50, 50, 50, 2],
            ],
            "f107": [100.0, 100.0, 100.0],
            "f107_81": [100.0, 100.0, 100.0], "f107_365": [100.0, 100.0, 100.0],
        }
        monkeypatch.setattr(M, "get_apf107", lambda: apf107)
        monkeypatch.setattr(M, "get_ig_rz", lambda: fixture_ig_rz)
        ds = M.IRI_Sample_Inputs("2024-06-15")   # current_idx_f107 == 1

        result = ds.quantileSamples(ap_sample_range=1)   # slice[0:2] -> rows 0,1
        ap_values = result["ap"].dropna()
        assert ap_values.min() == pytest.approx(1.0), "true global min across both rows is 1"
        assert ap_values.max() == pytest.approx(100.0), "true global max across both rows is 100"

    def test_attrs_record_requested_ranges(self, make_iri):
        ds = make_iri("2024-06-15")
        result = ds.quantileSamples(f107_sample_range=2, ig_sample_range=1)
        assert result.attrs["f107_sample_range"] == 2
        assert result.attrs["ig_sample_range"] == 1
        assert result.attrs["ap_sample_range"] is None
        assert result.attrs["sample_method"] == "quantileSamples"


# ===========================================================================
# randomSamples
# ===========================================================================

class TestRandomSamples:
    def test_default_nsample_is_one_all_none(self, make_iri):
        ds = make_iri("2024-06-15")
        result = ds.randomSamples()
        assert len(result) == 1
        assert result.iloc[0][["hour", "f107", "ap", "ig12", "rz12"]].isna().all()

    def test_nsample_controls_length(self, make_iri):
        ds = make_iri("2024-06-15")
        result = ds.randomSamples(f107_sample_range=2, nSample=10)
        assert len(result) == 10

    def test_sampled_f107_within_declared_window(self, make_iri):
        ds = make_iri("2024-06-15")
        # current_idx_f107=14, range=3 -> slice[11:17) -> f107 values 111..116
        result = ds.randomSamples(f107_sample_range=3, nSample=50)
        vals = result["f107"].dropna()
        assert vals.min() >= 111.0
        assert vals.max() <= 116.0

    def test_unset_range_gives_none_column(self, make_iri):
        ds = make_iri("2024-06-15")
        result = ds.randomSamples(f107_sample_range=2, nSample=20)
        assert result["ap"].isna().all()
        assert result["ig12"].isna().all()
        assert result["rz12"].isna().all()

    def test_attrs_record_nsample_and_method(self, make_iri):
        ds = make_iri("2024-06-15")
        result = ds.randomSamples(nSample=7)
        assert result.attrs["nSample"] == 7
        assert result.attrs["sample_method"] == "randomSamples"
