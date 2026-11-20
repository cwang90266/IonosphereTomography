# -*- coding: utf-8 -*-
"""Tests for iri_collection.py (Phase 6 prerequisite: full-resolution
training-collection generation via the main package's own EDPSamples
code)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import iri_collection as ic


def _touch_hour_folders(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True)


class TestConditionsFromRunHours:
    def test_parses_hour_folders_in_sorted_order(self, tmp_path):
        _touch_hour_folders(tmp_path, [
            "2025-08-27_01000200", "2025-08-26_00000100", "2025-08-26_14001500",
        ])
        conds = ic.conditions_from_run_hours(tmp_path, 42.6195, -71.4917, 18.0, "MIL")

        assert [c.time for c in conds] == [
            "2025-08-26T00:00:00", "2025-08-26T14:00:00", "2025-08-27T01:00:00",
        ]
        assert conds[0].label == "MIL_2025-08-26_0000"
        assert all(c.center_lat == 42.6195 and c.center_lon == -71.4917 for c in conds)
        assert all(c.grid_radius_deg == 18.0 for c in conds)

    def test_ignores_non_matching_entries(self, tmp_path):
        _touch_hour_folders(tmp_path, ["2025-08-26_00000100"])
        (tmp_path / "2025-08-26_mode_timing.csv").touch()
        (tmp_path / "not_a_date_folder").mkdir()

        conds = ic.conditions_from_run_hours(tmp_path, 0.0, 0.0, 18.0, "X")
        assert len(conds) == 1

    def test_raises_on_no_matches(self, tmp_path):
        (tmp_path / "unrelated").mkdir()
        with pytest.raises(ValueError, match="no 'YYYY-MM-DD_HHMMHHMM'"):
            ic.conditions_from_run_hours(tmp_path, 0.0, 0.0, 18.0, "X")

    def test_counts_real_mil_and_tro_run_directories(self):
        """Verification step 1 from the plan: sanity-check the exact hour
        counts against the real run directories this collection is meant
        to replicate, before spending any IRI2020 time on them. Skipped
        when the shared drive isn't mounted in this environment."""
        gdrive = Path(
            "/Users/cwang/Library/CloudStorage/GoogleDrive-cwang@planetiq.com/"
            "Shared drives/science analysis/ionosphere/tomography_output"
        )
        mil_dir = gdrive / "MIL_aug2530_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"
        tro_may_dir = gdrive / "TRO_may1821_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"
        tro_oct_dir = gdrive / "TRO_oct1417_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"
        if not mil_dir.is_dir():
            pytest.skip("shared drive not mounted in this environment")

        mil = ic.conditions_from_run_hours(mil_dir, 42.6195, -71.4917, 18.0, "MIL")
        tro_may = ic.conditions_from_run_hours(tro_may_dir, 69.6, 19.2, 18.0, "TRO_may")
        tro_oct = ic.conditions_from_run_hours(tro_oct_dir, 69.6, 19.2, 18.0, "TRO_oct")

        print(f"MIL training conditions: {len(mil)}")
        print(f"TRO_may validation conditions: {len(tro_may)}")
        print(f"TRO_oct validation conditions: {len(tro_oct)}")
        assert len(mil) > 0
        assert len(tro_may) > 0
        assert len(tro_oct) > 0


class TestBuildCollection:
    def test_skip_existing_avoids_rebuild(self, tmp_path, monkeypatch):
        conds = [ic.IriCollectionCondition(label="A", time="2025-01-01T00:00:00",
                                           center_lat=0.0, center_lon=0.0, grid_radius_deg=18.0)]
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "A_edp_samples.nc").write_bytes(b"not a real file, just a cache placeholder")

        def _boom(cfg):
            raise AssertionError("load_or_build_edp_samples should not be called when cached")

        monkeypatch.setattr("Assimilation_Cycle.ensemble_init.load_or_build_edp_samples", _boom)

        paths = ic.build_collection(conds, np.arange(60.0, 901.0, 10.0), out_dir,
                                    n_ensemble=5, skip_existing=True, verbose=False)
        assert paths == [out_dir / "A_edp_samples.nc"]

    def test_calls_load_or_build_edp_samples_with_condition_fields(self, tmp_path, monkeypatch):
        conds = [ic.IriCollectionCondition(label="B", time="2025-06-01T13:00:00",
                                           center_lat=42.6195, center_lon=-71.4917, grid_radius_deg=18.0)]
        out_dir = tmp_path / "out"
        seen = {}

        def _fake_build(cfg):
            seen["cfg"] = cfg

        monkeypatch.setattr("Assimilation_Cycle.ensemble_init.load_or_build_edp_samples", _fake_build)

        altitude = np.arange(60.0, 901.0, 10.0)
        ic.build_collection(conds, altitude, out_dir, n_ensemble=7, skip_existing=True, verbose=False)

        cfg = seen["cfg"]
        assert cfg.center_lat == 42.6195
        assert cfg.center_lon == -71.4917
        assert cfg.grid_radius_deg == 18.0
        assert cfg.n_ensemble == 7
        np.testing.assert_array_equal(cfg.altitude_grid, altitude)
        assert cfg.edp_samples_output_path == out_dir / "B_edp_samples.nc"
        # default_iri_spread_kwargs() used when not overridden
        assert cfg.iri_spread_kwargs["hour_sample_range"] == 3
