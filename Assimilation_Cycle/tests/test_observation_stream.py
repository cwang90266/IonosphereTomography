# -*- coding: utf-8 -*-
"""
Tests for observation_stream.py (plan Section 4.2/4.9/4.15).

batch_entries is tested directly against synthetic ObservationEntry
objects (no real RO/IGS data needed). assemble()'s source-selection
gating and precomputed-file mode are tested by monkeypatching
prepare_ro_observations/prepare_igs_observations, so these tests don't
need real podTc2/RINEX data on disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observation_preparation import ObservationEntry, write_observations
import Assimilation_Cycle.observation_stream as observation_stream
from Assimilation_Cycle.cycle_config import CycleConfig


def _make_entry(obs_type, date, n_rays=3, label="", tangent_alt_km=None):
    rng = np.random.default_rng(abs(hash((obs_type, str(date)))) % (2**32))
    return ObservationEntry(
        obs_type=obs_type,
        tec=rng.uniform(5, 20, size=n_rays).astype(np.float32),
        gnss_ecef_km=rng.uniform(20000, 26000, size=(3, n_rays)),
        rec_ecef_km=rng.uniform(6800, 7200, size=(3, n_rays)),
        date=pd.Timestamp(date),
        label=label,
        tangent_alt_km=np.asarray(tangent_alt_km) if tangent_alt_km is not None else None,
    )


def _make_cfg(**overrides) -> CycleConfig:
    defaults = dict(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=18.9, radius_km=500.0,
        obs_sigma=2.0,
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


class TestBatchEntries:
    def test_sorts_chronologically_and_batches(self):
        cfg = _make_cfg(batch_size=2)
        entries = [
            _make_entry("IGS", "2025-11-18T10:20:00", label="c"),
            _make_entry("RO", "2025-11-18T10:05:00", label="a"),
            _make_entry("RO", "2025-11-18T10:10:00", label="b"),
            _make_entry("IGS", "2025-11-18T10:25:00", label="d"),
            _make_entry("RO", "2025-11-18T10:30:00", label="e"),
        ]
        batches = observation_stream.batch_entries(entries, cfg)

        assert len(batches) == 3  # 5 entries, batch_size=2 -> 2,2,1
        assert [b.batch_index for b in batches] == [0, 1, 2]
        assert batches[0].start_time == pd.Timestamp("2025-11-18T10:05:00")
        assert batches[1].start_time == pd.Timestamp("2025-11-18T10:20:00")
        assert batches[2].start_time == pd.Timestamp("2025-11-18T10:30:00")

    def test_concatenates_rays_and_R_is_constant_diagonal(self):
        cfg = _make_cfg(batch_size=10, obs_sigma=3.0)
        entries = [_make_entry("RO", "2025-11-18T10:00:00", n_rays=4),
                   _make_entry("IGS", "2025-11-18T10:01:00", n_rays=2)]
        batches = observation_stream.batch_entries(entries, cfg)

        assert len(batches) == 1
        batch = batches[0]
        assert batch.y_obs.shape == (6,)
        assert batch.podTc2_data["rec_ecef_km"].shape == (3, 6)
        assert batch.podTc2_data["gnss_ecef_km"].shape == (3, 6)
        assert batch.obs_type.tolist() == ["RO"] * 4 + ["IGS"] * 2
        np.testing.assert_allclose(batch.R, (3.0 ** 2) * np.eye(6))

    def test_entries_without_date_go_last_without_crashing(self):
        cfg = _make_cfg(batch_size=10)
        dated = _make_entry("RO", "2025-11-18T10:00:00")
        undated = ObservationEntry(
            obs_type="RO", tec=np.array([1.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
            date=None,
        )
        batches = observation_stream.batch_entries([undated, dated], cfg)
        assert len(batches) == 1  # doesn't raise on None-date comparison

    def test_empty_entries_gives_no_batches(self):
        cfg = _make_cfg(batch_size=10)
        assert observation_stream.batch_entries([], cfg) == []

    def test_entry_ray_ranges_track_source_entries(self):
        cfg = _make_cfg(batch_size=10)
        e1 = _make_entry("RO", "2025-11-18T10:00:00", n_rays=4, label="a")
        e2 = _make_entry("IGS", "2025-11-18T10:01:00", n_rays=2, label="b")
        batch = observation_stream.batch_entries([e1, e2], cfg)[0]

        assert len(batch.entry_ray_ranges) == 2
        (entry_a, slice_a), (entry_b, slice_b) = batch.entry_ray_ranges
        assert entry_a.label == "a" and slice_a == slice(0, 4)
        assert entry_b.label == "b" and slice_b == slice(4, 6)
        np.testing.assert_allclose(batch.y_obs[slice_a], e1.tec)
        np.testing.assert_allclose(batch.y_obs[slice_b], e2.tec)


class TestMaxTecPerBatch:
    def test_accumulates_until_cap_then_starts_new_batch(self):
        cfg = _make_cfg(max_tec_per_batch=10)
        entries = [
            _make_entry("RO", "2025-11-18T10:00:00", n_rays=4, label="a"),
            _make_entry("RO", "2025-11-18T10:01:00", n_rays=4, label="b"),
            _make_entry("RO", "2025-11-18T10:02:00", n_rays=4, label="c"),
        ]
        batches = observation_stream.batch_entries(entries, cfg)
        # a+b = 8 rays <= 10 -> batch 0; c alone -> batch 1 (8+4=12 > 10)
        assert len(batches) == 2
        assert batches[0].y_obs.shape == (8,)
        assert batches[1].y_obs.shape == (4,)

    def test_oversized_entry_gets_its_own_batch_not_split(self):
        cfg = _make_cfg(max_tec_per_batch=5)
        entries = [_make_entry("RO", "2025-11-18T10:00:00", n_rays=20, label="big")]
        batches = observation_stream.batch_entries(entries, cfg)
        assert len(batches) == 1
        assert batches[0].y_obs.shape == (20,)  # not split, even though 20 > cap

    def test_max_tec_per_batch_overrides_batch_size(self):
        # batch_size alone would give entry-count batches of 1; max_tec_per_batch
        # (set) takes priority and groups by ray count instead.
        cfg = _make_cfg(batch_size=1, max_tec_per_batch=100)
        entries = [_make_entry("RO", f"2025-11-18T10:0{i}:00", n_rays=3) for i in range(5)]
        batches = observation_stream.batch_entries(entries, cfg)
        assert len(batches) == 1
        assert batches[0].y_obs.shape == (15,)

    def test_none_max_tec_per_batch_preserves_entry_count_policy(self):
        cfg = _make_cfg(batch_size=2, max_tec_per_batch=None)
        entries = [_make_entry("RO", f"2025-11-18T10:0{i}:00", n_rays=100) for i in range(4)]
        batches = observation_stream.batch_entries(entries, cfg)
        assert len(batches) == 2  # entry-count batching, ray count irrelevant


class TestMinTangentAltFilter:
    def test_drops_low_tangent_rays_from_ro_entries(self):
        cfg = _make_cfg(batch_size=10, min_tangent_alt_km=100.0)
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=5,
                             tangent_alt_km=[10.0, 50.0, 100.0, 150.0, 200.0])
        batches = observation_stream.batch_entries([entry], cfg)
        assert len(batches) == 1
        # rays at 10, 50 dropped (< 100); 100, 150, 200 kept
        assert batches[0].y_obs.shape == (3,)
        np.testing.assert_allclose(batches[0].podTc2_data["rec_ecef_km"],
                                    entry.rec_ecef_km[:, [2, 3, 4]])

    def test_entry_dropped_entirely_when_no_rays_survive(self):
        cfg = _make_cfg(batch_size=10, min_tangent_alt_km=500.0)
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=3,
                             tangent_alt_km=[10.0, 50.0, 100.0])
        batches = observation_stream.batch_entries([entry], cfg)
        assert batches == []

    def test_igs_entries_unaffected(self):
        cfg = _make_cfg(batch_size=10, min_tangent_alt_km=100.0)
        igs_entry = _make_entry("IGS", "2025-11-18T10:00:00", n_rays=3)  # no tangent_alt_km
        batches = observation_stream.batch_entries([igs_entry], cfg)
        assert len(batches) == 1
        assert batches[0].y_obs.shape == (3,)

    def test_none_min_tangent_alt_km_disables_filter(self):
        cfg = _make_cfg(batch_size=10, min_tangent_alt_km=None)
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=3,
                             tangent_alt_km=[1.0, 2.0, 3.0])
        batches = observation_stream.batch_entries([entry], cfg)
        assert batches[0].y_obs.shape == (3,)


class TestFilterEntryRays:
    def test_all_true_mask_returns_same_entry(self):
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=3)
        mask = np.array([True, True, True])
        result = observation_stream._filter_entry_rays(entry, mask)
        assert result is entry

    def test_all_false_mask_returns_none(self):
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=3)
        mask = np.array([False, False, False])
        assert observation_stream._filter_entry_rays(entry, mask) is None

    def test_partial_mask_slices_per_ray_fields_and_keeps_abel(self):
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=4)
        entry.abel = {"Ne": np.array([1.0, 2.0]), "alt_km": np.array([100.0, 200.0])}
        mask = np.array([True, False, True, False])
        result = observation_stream._filter_entry_rays(entry, mask)

        assert result.n_rays == 2
        np.testing.assert_allclose(result.tec, entry.tec[[0, 2]])
        np.testing.assert_allclose(result.rec_ecef_km, entry.rec_ecef_km[:, [0, 2]])
        # abel is per-entry, not per-ray -- unaffected by the ray mask
        assert result.abel is entry.abel

    def test_extra_arrays_matching_ray_count_are_sliced(self):
        entry = _make_entry("RO", "2025-11-18T10:00:00", n_rays=4)
        entry.extra["ipp_distance_km"] = np.array([1.0, 2.0, 3.0, 4.0])
        entry.extra["some_scalar"] = 42
        mask = np.array([True, False, False, True])
        result = observation_stream._filter_entry_rays(entry, mask)
        np.testing.assert_allclose(result.extra["ipp_distance_km"], [1.0, 4.0])
        assert result.extra["some_scalar"] == 42


class TestPrecomputedFileMode:
    def test_ro_loads_from_precomputed_path_without_calling_prepare(self, tmp_path, monkeypatch):
        entries = [_make_entry("RO", "2025-11-18T10:00:00")]
        path = tmp_path / "ro.nc"
        write_observations(entries, path)

        def _boom(*a, **k):
            raise AssertionError("prepare_ro_observations should not be called when a precomputed path is set")
        monkeypatch.setattr(observation_stream, "prepare_ro_observations", _boom)

        cfg = _make_cfg(ro_observations_path=path)
        loaded = observation_stream._load_or_compute_ro(cfg)
        assert len(loaded) == 1
        assert loaded[0].obs_type == "RO"

    def test_ro_saves_fresh_computation_when_output_path_set(self, tmp_path, monkeypatch):
        fake_raw = [{
            "tec": np.array([1.0, 2.0]),
            "LEO": np.zeros((3, 2)),
            "GNSS": np.ones((3, 2)),
            "date": pd.Timestamp("2025-11-18T10:00:00"),
        }]
        monkeypatch.setattr(observation_stream, "prepare_ro_observations", lambda **kw: fake_raw)

        out_path = tmp_path / "ro_out.nc"
        cfg = _make_cfg(ro_observations_output_path=out_path)
        entries = observation_stream._load_or_compute_ro(cfg)

        assert len(entries) == 1
        assert out_path.exists()

    def test_igs_requires_stations_and_cache_dir(self):
        cfg = _make_cfg(igs_kwargs={})
        with pytest.raises(ValueError, match="stations|cache_dir"):
            observation_stream._load_or_compute_igs(cfg)


class TestSourceSelection:
    def _patch_both(self, monkeypatch, ro_calls, igs_calls):
        def _ro(**kw):
            ro_calls.append(kw)
            return [{
                "tec": np.array([1.0]), "LEO": np.zeros((3, 1)), "GNSS": np.ones((3, 1)),
                "date": pd.Timestamp("2025-11-18T10:00:00"),
            }]

        def _igs(**kw):
            igs_calls.append(kw)
            return [{
                "tec": np.array([2.0]), "LEO": np.zeros((3, 1)), "GNSS": np.ones((3, 1)),
                "date": pd.Timestamp("2025-11-18T10:01:00"),
            }]

        monkeypatch.setattr(observation_stream, "prepare_ro_observations", _ro)
        monkeypatch.setattr(observation_stream, "prepare_igs_observations", _igs)

    def test_ro_only_skips_igs(self, monkeypatch):
        ro_calls, igs_calls = [], []
        self._patch_both(monkeypatch, ro_calls, igs_calls)
        cfg = _make_cfg(obs_sources="RO")
        batches = observation_stream.assemble(cfg)
        assert len(ro_calls) == 1
        assert len(igs_calls) == 0
        assert all((b.obs_type == "RO").all() for b in batches)

    def test_igs_only_skips_ro(self, monkeypatch):
        ro_calls, igs_calls = [], []
        self._patch_both(monkeypatch, ro_calls, igs_calls)
        cfg = _make_cfg(obs_sources="IGS", igs_kwargs={"stations": ["TRO1"], "cache_dir": "/tmp/x"})
        batches = observation_stream.assemble(cfg)
        assert len(ro_calls) == 0
        assert len(igs_calls) == 1
        assert all((b.obs_type == "IGS").all() for b in batches)

    def test_both_calls_both_and_merges(self, monkeypatch):
        ro_calls, igs_calls = [], []
        self._patch_both(monkeypatch, ro_calls, igs_calls)
        cfg = _make_cfg(obs_sources="both", igs_kwargs={"stations": ["TRO1"], "cache_dir": "/tmp/x"}, batch_size=10)
        batches = observation_stream.assemble(cfg)
        assert len(ro_calls) == 1
        assert len(igs_calls) == 1
        assert len(batches) == 1
        assert sorted(batches[0].obs_type.tolist()) == ["IGS", "RO"]
