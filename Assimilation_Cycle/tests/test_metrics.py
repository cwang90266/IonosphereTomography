# -*- coding: utf-8 -*-
"""Tests for metrics.py (plan Section 4.13)."""
from __future__ import annotations

import numpy as np

from Assimilation_Cycle.metrics import (
    compute_rmse_reduction,
    rank_histogram_counts,
    effective_rank,
    rmse,
)


class TestRmseReduction:
    def test_analysis_closer_to_truth_shows_positive_reduction(self):
        rng = np.random.default_rng(0)
        y_obs = np.full(50, 10.0) + rng.normal(scale=0.1, size=50)
        y_forecast = np.full(50, 8.0)   # off by 2
        y_analysis = np.full(50, 9.8)   # off by 0.2, much closer

        result = compute_rmse_reduction(y_obs, y_forecast, y_analysis)
        assert result.rmse_forecast == rmse(y_obs - y_forecast)
        assert result.rmse_analysis == rmse(y_obs - y_analysis)
        assert result.reduction > 0
        assert 0 < result.fractional_reduction < 1

    def test_no_change_gives_zero_reduction(self):
        y_obs = np.array([1.0, 2.0, 3.0])
        result = compute_rmse_reduction(y_obs, y_obs + 1.0, y_obs + 1.0)
        assert result.reduction == 0.0

    def test_by_obs_type_split(self):
        y_obs = np.array([10.0, 10.0, 20.0, 20.0])
        y_forecast = np.array([8.0, 8.0, 15.0, 15.0])
        y_analysis = np.array([9.9, 9.9, 19.9, 19.9])
        obs_type = np.array(["RO", "RO", "IGS", "IGS"])

        result = compute_rmse_reduction(y_obs, y_forecast, y_analysis, obs_type=obs_type)
        assert set(result.by_obs_type) == {"RO", "IGS"}
        assert result.by_obs_type["RO"]["n_obs"] == 2
        assert result.by_obs_type["IGS"]["n_obs"] == 2
        # RO's forecast error (2.0) is much larger than IGS's relative pattern is identical
        # here, so both types should individually show a reduction too.
        assert result.by_obs_type["RO"]["rmse_forecast"] > result.by_obs_type["RO"]["rmse_analysis"]
        assert result.by_obs_type["IGS"]["rmse_forecast"] > result.by_obs_type["IGS"]["rmse_analysis"]


class TestRankHistogram:
    def test_well_calibrated_ensemble_gives_near_uniform_histogram(self):
        """Draw both the ensemble and the 'observation' from the same
        distribution (a textbook well-calibrated setup) -- the rank
        histogram should be close to flat."""
        rng = np.random.default_rng(42)
        n_obs, n_members = 4000, 9
        Y_ensemble = rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_members))
        y_obs = rng.normal(loc=0.0, scale=1.0, size=n_obs)

        counts = rank_histogram_counts(y_obs, Y_ensemble, rng=rng)
        assert counts.shape == (n_members + 1,)
        assert counts.sum() == n_obs

        expected = n_obs / (n_members + 1)
        # Loose tolerance -- this is a statistical check, not exact.
        assert np.all(np.abs(counts - expected) < 0.25 * expected)

    def test_observation_always_above_ensemble_gives_top_rank(self):
        n_obs, n_members = 20, 5
        Y_ensemble = np.tile(np.arange(n_members, dtype=float), (n_obs, 1))
        y_obs = np.full(n_obs, 100.0)  # always above every ensemble member

        counts = rank_histogram_counts(y_obs, Y_ensemble)
        assert counts[-1] == n_obs
        assert counts[:-1].sum() == 0

    def test_observation_always_below_ensemble_gives_rank_zero(self):
        n_obs, n_members = 20, 5
        Y_ensemble = np.tile(np.arange(n_members, dtype=float), (n_obs, 1))
        y_obs = np.full(n_obs, -100.0)

        counts = rank_histogram_counts(y_obs, Y_ensemble)
        assert counts[0] == n_obs

    def test_under_dispersive_ensemble_skews_u_shaped(self):
        """Observation variance much larger than ensemble spread -> most
        observations land at the extreme ranks (U-shape), not the middle."""
        rng = np.random.default_rng(7)
        n_obs, n_members = 3000, 9
        Y_ensemble = rng.normal(loc=0.0, scale=0.1, size=(n_obs, n_members))
        y_obs = rng.normal(loc=0.0, scale=5.0, size=n_obs)

        counts = rank_histogram_counts(y_obs, Y_ensemble, rng=rng)
        middle = n_members // 2
        edge_mass = counts[0] + counts[-1]
        middle_mass = counts[middle]
        assert edge_mass > middle_mass


class TestEffectiveRank:
    def test_full_rank_random_ensemble(self):
        rng = np.random.default_rng(1)
        n_state, n_members = 50, 10
        X = rng.normal(size=(n_state, n_members))
        result = effective_rank(X)
        # anomaly matrix has at most n_members - 1 independent directions
        assert result["rank"] == n_members - 1
        assert result["singular_values"].shape[0] == n_members
        assert result["n_members"] == n_members

    def test_collapsed_ensemble_has_low_rank(self):
        """All members identical (a fully collapsed ensemble) -> the
        anomaly matrix is exactly zero, rank 0."""
        n_state, n_members = 20, 6
        member = np.arange(n_state, dtype=float)
        X = np.tile(member[:, np.newaxis], (1, n_members))
        result = effective_rank(X)
        assert result["rank"] == 0
        np.testing.assert_allclose(result["singular_values"], 0.0, atol=1e-10)

    def test_rank_deficient_by_construction(self):
        """Members drawn from a 3-dimensional subspace of a larger state
        space -> anomaly rank should be capped at 3, not n_members - 1."""
        rng = np.random.default_rng(3)
        n_state, n_members, true_rank = 100, 20, 3
        basis = rng.normal(size=(n_state, true_rank))
        coeffs = rng.normal(size=(true_rank, n_members))
        X = basis @ coeffs
        result = effective_rank(X)
        assert result["rank"] == true_rank
