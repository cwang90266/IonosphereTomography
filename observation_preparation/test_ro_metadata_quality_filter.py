#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for PlanetiQ RO metadata-level DCB QC and duplicate selection.

Run from the tomography project root:
    python -m pytest observation_preparation/test_ro_metadata_quality_filter.py
"""

import numpy as np
import pandas as pd

from observation_preparation.ro_source import (
    filter_ro_dcb_quality,
    deduplicate_ro_metadata,
)


def _row(
    filename: str,
    start_time: float,
    leveling_err: float,
    *,
    leo_id="4",
    conid="G",
    prn_id="32",
    leodcb_flag=1,
    gpsdcb_flag=1,
):
    return {
        "filename": filename,
        "full_path": f"/tmp/{filename}",
        "date": pd.Timestamp("2025-10-16T14:04:00")
        + pd.to_timedelta(start_time, unit="s"),
        "lat": 69.6,
        "lon": 19.2,
        "leo_id": str(leo_id),
        "conid": str(conid),
        "prn_id": str(prn_id),
        "start_time": float(start_time),
        "leveling_err": float(leveling_err),
        "leodcb_flag": int(leodcb_flag),
        "gpsdcb_flag": int(gpsdcb_flag),
    }


def test_dcb_flags_require_both_one():
    df = pd.DataFrame([
        _row("good.nc", 0, 0.1, leodcb_flag=1, gpsdcb_flag=1),
        _row("bad_leo.nc", 60, 0.1, leodcb_flag=0, gpsdcb_flag=1),
        _row("bad_gps.nc", 120, 0.1, leodcb_flag=1, gpsdcb_flag=0),
    ])
    kept, rejected = filter_ro_dcb_quality(df, require_dcb_flags=True)
    assert kept["filename"].tolist() == ["good.nc"]
    assert set(rejected["filename"]) == {"bad_leo.nc", "bad_gps.nc"}
    assert set(rejected["reject_reason"]) == {"invalid_dcb_flag"}


def test_dcb_filter_can_be_disabled():
    df = pd.DataFrame([
        _row("bad.nc", 0, 0.1, leodcb_flag=0, gpsdcb_flag=0),
    ])
    kept, rejected = filter_ro_dcb_quality(df, require_dcb_flags=False)
    assert kept["filename"].tolist() == ["bad.nc"]
    assert rejected.empty


def test_16_second_duplicate_keeps_smaller_leveling_err():
    df = pd.DataFrame([
        _row("0012.nc", 12, 0.30),
        _row("0028.nc", 28, 0.05),
    ])
    kept, rejected = deduplicate_ro_metadata(df, time_tolerance_s=30.0)
    assert kept["filename"].tolist() == ["0028.nc"]
    assert rejected["filename"].tolist() == ["0012.nc"]
    assert rejected.iloc[0]["duplicate_kept_filename"] == "0028.nc"
    assert rejected.iloc[0]["time_separation_s"] == 16.0
    assert rejected.iloc[0]["reject_reason"] == "duplicate_higher_leveling_err"


def test_same_pair_more_than_30_seconds_are_not_duplicates():
    df = pd.DataFrame([
        _row("a.nc", 0, 0.30),
        _row("b.nc", 31, 0.05),
    ])
    kept, rejected = deduplicate_ro_metadata(df, time_tolerance_s=30.0)
    assert set(kept["filename"]) == {"a.nc", "b.nc"}
    assert rejected.empty


def test_time_connected_three_file_cluster_keeps_best():
    # 0 -> 18 -> 35 s: each consecutive gap is <= 30 s, so this is one
    # transitive duplicate cluster even though first-to-last separation is 35 s.
    df = pd.DataFrame([
        _row("a.nc", 0, 0.10),
        _row("b.nc", 18, 0.03),
        _row("c.nc", 35, 0.08),
    ])
    kept, rejected = deduplicate_ro_metadata(df, time_tolerance_s=30.0)
    assert kept["filename"].tolist() == ["b.nc"]
    assert set(rejected["filename"]) == {"a.nc", "c.nc"}
    assert set(rejected["duplicate_kept_filename"]) == {"b.nc"}


def test_same_time_but_different_transmitter_is_not_duplicate():
    df = pd.DataFrame([
        _row("g32.nc", 0, 0.10, conid="G", prn_id="32"),
        _row("e04.nc", 5, 0.01, conid="E", prn_id="4"),
    ])
    kept, rejected = deduplicate_ro_metadata(df, time_tolerance_s=30.0)
    assert set(kept["filename"]) == {"g32.nc", "e04.nc"}
    assert rejected.empty
