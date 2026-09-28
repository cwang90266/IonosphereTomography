# -*- coding: utf-8 -*-
"""Round-trip tests for ensemble_io.py (plan Section 4.10)."""
from __future__ import annotations

import numpy as np
import pytest

from Ensemble_Kalman_Engine import EnsembleState
from Assimilation_Cycle.ensemble_io import save_ensemble_netcdf, load_ensemble_netcdf


def _toy_ensemble(n_height=5, n_geo=3, n_members=7, rng=None) -> EnsembleState:
    rng = rng or np.random.default_rng(0)
    param_shape = (n_height, n_geo)
    X = rng.normal(size=(n_height * n_geo, n_members))
    return EnsembleState(X=X, param_shape=param_shape)


class TestRoundTrip:
    def test_bare_round_trip(self, tmp_path):
        ens = _toy_ensemble()
        path = save_ensemble_netcdf(ens, tmp_path / "ens.nc")
        assert path.exists()

        loaded, meta = load_ensemble_netcdf(path)
        np.testing.assert_allclose(loaded.X, ens.X)
        assert loaded.param_shape == ens.param_shape
        assert meta == {}

    def test_round_trip_with_full_metadata(self, tmp_path):
        ens = _toy_ensemble()
        roi = {"center_lat": 69.6, "center_lon": 18.9, "radius_km": 500.0}
        hyper_params = {"minlog10Density": 4.0}

        path = save_ensemble_netcdf(
            ens, tmp_path / "batch_0003.nc",
            style="density_10ex", hyper_params=hyper_params,
            cycle_start_time="2025-11-18T10:00:00", cycle_end_time="2025-11-18T11:00:00",
            roi=roi, batch_index=3,
            batch_start_time="2025-11-18T10:20:00", batch_end_time="2025-11-18T10:25:00",
        )
        loaded, meta = load_ensemble_netcdf(path)

        np.testing.assert_allclose(loaded.X, ens.X)
        assert loaded.param_shape == ens.param_shape
        assert meta["style"] == "density_10ex"
        assert meta["hyper_params"] == hyper_params
        assert meta["roi"] == roi
        assert meta["batch_index"] == 3
        assert meta["cycle_start_time"] == "2025-11-18T10:00:00"
        assert meta["batch_start_time"] == "2025-11-18T10:20:00"

    def test_single_member_ensemble(self, tmp_path):
        ens = _toy_ensemble(n_members=1)
        path = save_ensemble_netcdf(ens, tmp_path / "single.nc")
        loaded, _ = load_ensemble_netcdf(path)
        np.testing.assert_allclose(loaded.X, ens.X)

    def test_one_file_per_batch_pattern(self, tmp_path):
        """Mirrors cycle_driver's intended usage: a batch-indexed filename
        per batch, not one growing file."""
        rng = np.random.default_rng(1)
        paths = []
        for i in range(3):
            ens = _toy_ensemble(rng=rng)
            paths.append(save_ensemble_netcdf(
                ens, tmp_path / f"ensemble_batch_{i:04d}.nc", batch_index=i,
            ))
        assert len(set(paths)) == 3
        for i, path in enumerate(paths):
            _, meta = load_ensemble_netcdf(path)
            assert meta["batch_index"] == i
