#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the truncated-TEC-profile filter in _build_clean_ro_entry
(max_lowest_tangent_alt_km): an occultation whose valid rays never reach
below the threshold tangent altitude is rejected.

Run from the tomography project root:
    python -m pytest observation_preparation/test_ro_truncated_profile_filter.py
"""

import numpy as np
import pytest

from observation_preparation import ro_source


def _occultation(tangent_km: np.ndarray, tec: np.ndarray | None = None) -> dict:
    n = len(tangent_km)
    return {
        "LEO": np.zeros((3, n)),
        "GNSS": np.zeros((3, n)),
        "TEC_podTc2": np.full(n, 50.0) if tec is None else tec,
        "_tangent_km": np.asarray(tangent_km, dtype=float),
    }


@pytest.fixture(autouse=True)
def fake_ray_tangent(monkeypatch):
    """rayTangent returns geodetic altitude in metres; feed the tangent
    altitudes stored on the fake occultation straight through."""
    current = {}

    def _fake(leo, gnss, units="km"):
        alt_km = current["tangent_km"]
        return np.zeros_like(alt_km), np.zeros_like(alt_km), alt_km * 1e3

    monkeypatch.setattr(ro_source, "rayTangent", _fake)
    return current


def _build(data, fake_ray_tangent, **kwargs):
    fake_ray_tangent["tangent_km"] = data["_tangent_km"]
    return ro_source._build_clean_ro_entry(
        data, label="occ", min_valid_rays=10, max_rays_per_occultation=200, **kwargs
    )


def test_truncated_profile_rejected_by_default(fake_ray_tangent):
    data = _occultation(np.linspace(580.0, 450.0, 100))
    entry, info = _build(data, fake_ray_tangent)
    assert entry is None
    assert info["reject_reason"] == "truncated_tec_profile"
    assert info["lowest_tangent_alt_km"] == pytest.approx(450.0)


def test_full_profile_accepted(fake_ray_tangent):
    data = _occultation(np.linspace(580.0, 20.0, 100))
    entry, info = _build(data, fake_ray_tangent)
    assert entry is not None
    assert info["accepted"]
    assert info["lowest_tangent_alt_km"] == pytest.approx(20.0)


def test_threshold_is_configurable_and_inclusive(fake_ray_tangent):
    data = _occultation(np.linspace(580.0, 450.0, 100))
    entry, _ = _build(data, fake_ray_tangent, max_lowest_tangent_alt_km=450.0)
    assert entry is not None
    entry, info = _build(data, fake_ray_tangent, max_lowest_tangent_alt_km=449.0)
    assert entry is None and info["reject_reason"] == "truncated_tec_profile"


def test_none_disables_the_filter(fake_ray_tangent):
    data = _occultation(np.linspace(580.0, 560.0, 100))
    entry, _ = _build(data, fake_ray_tangent, max_lowest_tangent_alt_km=None)
    assert entry is not None


def test_only_valid_rays_count(fake_ray_tangent):
    """Low rays with unusable TEC do not make a truncated profile pass."""
    tangent = np.linspace(580.0, 20.0, 100)
    tec = np.full(100, 50.0)
    tec[tangent < 450.0] = np.nan
    entry, info = _build(_occultation(tangent, tec), fake_ray_tangent)
    assert entry is None
    assert info["reject_reason"] == "truncated_tec_profile"


def test_too_few_rays_reported_before_truncation(fake_ray_tangent):
    data = _occultation(np.linspace(580.0, 560.0, 5))
    entry, info = _build(data, fake_ray_tangent)
    assert entry is None
    assert info["reject_reason"] == "too_few_valid_rays"
