# -*- coding: utf-8 -*-
"""Tests for receiver_sigma_table.py."""
from __future__ import annotations

import numpy as np
import pytest

from observation_preparation.receiver_sigma_table import (
    ReceiverSigmaTable,
    load_receiver_sigma_table,
    match_receiver,
    sigma_for_entry,
)
from observation_preparation.schema import ObservationEntry


def _make_entry(label="", rec_id="", obs_source="", tangent_alt_km=None, n_rays=3):
    return ObservationEntry(
        obs_type="RO",
        tec=np.zeros(n_rays),
        gnss_ecef_km=np.zeros((3, n_rays)),
        rec_ecef_km=np.zeros((3, n_rays)),
        tangent_alt_km=np.asarray(tangent_alt_km) if tangent_alt_km is not None else None,
        label=label, rec_id=rec_id, obs_source=obs_source,
    )


class TestLoadReceiverSigmaTable:
    def test_loads_bins_and_receiver_columns(self, tmp_path):
        path = tmp_path / "sigma.csv"
        path.write_text(
            "alt_min_km,alt_max_km,GN04,GN05,YM08\n"
            "0,100,2.0,2.5,5.0\n"
            "100,300,3.0,3.5,6.0\n"
        )
        table = load_receiver_sigma_table(path)
        assert table.receivers == ["GN04", "GN05", "YM08"]
        np.testing.assert_allclose(table.alt_min_km, [0.0, 100.0])
        np.testing.assert_allclose(table.alt_max_km, [100.0, 300.0])
        np.testing.assert_allclose(table.sigma, [[2.0, 2.5, 5.0], [3.0, 3.5, 6.0]])

    def test_blank_cell_becomes_nan(self, tmp_path):
        path = tmp_path / "sigma.csv"
        path.write_text("alt_min_km,alt_max_km,GN04\n0,100,\n")
        table = load_receiver_sigma_table(path)
        assert np.isnan(table.sigma[0, 0])

    def test_too_few_columns_raises(self, tmp_path):
        path = tmp_path / "sigma.csv"
        path.write_text("alt_min_km,alt_max_km\n0,100\n")
        with pytest.raises(ValueError, match="at least 3 columns"):
            load_receiver_sigma_table(path)


class TestReceiverSigmaTableLookup:
    def test_inclusive_bin_edges(self):
        table = ReceiverSigmaTable(
            alt_min_km=np.array([0.0, 100.0]), alt_max_km=np.array([100.0, 200.0]),
            receivers=["YM08"], sigma=np.array([[5.0], [6.0]]),
        )
        out = table.lookup("YM08", np.array([0.0, 100.0, 150.0, 200.0, 250.0]))
        np.testing.assert_allclose(out[:2], [5.0, 5.0])  # 100 hits first bin (first match wins)
        np.testing.assert_allclose(out[2:4], [6.0, 6.0])
        assert np.isnan(out[4])  # outside every bin

    def test_unknown_receiver_is_all_nan(self):
        table = ReceiverSigmaTable(
            alt_min_km=np.array([0.0]), alt_max_km=np.array([100.0]),
            receivers=["YM08"], sigma=np.array([[5.0]]),
        )
        out = table.lookup("GN04", np.array([50.0]))
        assert np.isnan(out[0])


class TestMatchReceiver:
    def test_matches_from_label(self):
        entry = _make_entry(label="podTc2_YM08.2025.322.10.06.0022.C22.01_0000.0001_nc")
        assert match_receiver(entry, ["GN04", "GN05", "YM08"]) == "YM08"

    def test_no_match_returns_none(self):
        entry = _make_entry(label="podTc2_ZZ99.2025.322.10.06.0022.C22.01_0000.0001_nc")
        assert match_receiver(entry, ["GN04", "GN05", "YM08"]) is None

    def test_falls_back_to_rec_id_when_label_empty(self):
        entry = _make_entry(label="", rec_id="GN04")
        assert match_receiver(entry, ["GN04", "GN05", "YM08"]) == "GN04"

    def test_case_and_separator_insensitive(self):
        entry = _make_entry(label="gn-0-4_occultation_file")
        assert match_receiver(entry, ["GN04"]) == "GN04"


class TestSigmaForEntry:
    def test_no_table_returns_constant(self):
        entry = _make_entry(label="podTc2_YM08...", tangent_alt_km=[10.0, 20.0, 30.0])
        sigma = sigma_for_entry(entry, None, default_sigma=3.0)
        np.testing.assert_allclose(sigma, [3.0, 3.0, 3.0])

    def test_igs_entry_without_tangent_alt_uses_constant(self):
        entry = _make_entry(label="TRO1", tangent_alt_km=None)
        table = ReceiverSigmaTable(
            alt_min_km=np.array([0.0]), alt_max_km=np.array([1000.0]),
            receivers=["TRO1"], sigma=np.array([[9.0]]),
        )
        sigma = sigma_for_entry(entry, table, default_sigma=3.0)
        np.testing.assert_allclose(sigma, [3.0, 3.0, 3.0])

    def test_ro_entry_uses_table_value_when_covered(self):
        entry = _make_entry(
            label="podTc2_YM08.2025.322.10.06.0022.C22.01_0000.0001_nc",
            tangent_alt_km=[50.0, 150.0, 500.0],
        )
        table = ReceiverSigmaTable(
            alt_min_km=np.array([0.0, 100.0]), alt_max_km=np.array([100.0, 300.0]),
            receivers=["YM08"], sigma=np.array([[5.0], [6.0]]),
        )
        sigma = sigma_for_entry(entry, table, default_sigma=3.0)
        np.testing.assert_allclose(sigma, [5.0, 6.0, 3.0])  # 500 km uncovered -> fallback

    def test_unmatched_receiver_falls_back_to_constant(self):
        entry = _make_entry(label="podTc2_ZZ99...", tangent_alt_km=[50.0], n_rays=1)
        table = ReceiverSigmaTable(
            alt_min_km=np.array([0.0]), alt_max_km=np.array([100.0]),
            receivers=["YM08"], sigma=np.array([[5.0]]),
        )
        sigma = sigma_for_entry(entry, table, default_sigma=3.0)
        np.testing.assert_allclose(sigma, [3.0])

    def test_blank_cell_falls_back_to_constant(self):
        entry = _make_entry(label="podTc2_YM08...", tangent_alt_km=[50.0], n_rays=1)
        table = ReceiverSigmaTable(
            alt_min_km=np.array([0.0]), alt_max_km=np.array([100.0]),
            receivers=["YM08"], sigma=np.array([[np.nan]]),
        )
        sigma = sigma_for_entry(entry, table, default_sigma=3.0)
        np.testing.assert_allclose(sigma, [3.0])
