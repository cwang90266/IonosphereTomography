"""Shared time-window helpers used by both ro_source.py and igs_source.py.

Deduplicated from what was previously copy-pasted verbatim in both
prepare_ro_observations.py and prepare_igs_observations.py (Plan Section
2.4/10 step 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def format_filename_number(value) -> str:
    """Compact, filesystem-safe numeric token used in exported file names."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def time_window_tag(start_time, end_time) -> str:
    """Return e.g. ``timewindow1hr_20251118_10001100``."""
    start = normalize_timestamp(start_time)
    end = normalize_timestamp(end_time)
    if start is None or end is None:
        return "timewindow_unknown"

    minutes = (end - start).total_seconds() / 60.0
    if np.isclose(minutes % 60.0, 0.0):
        duration = f"{format_filename_number(minutes / 60.0)}hr"
    else:
        duration = f"{format_filename_number(minutes)}min"

    if start.date() == end.date():
        return f"timewindow{duration}_{start:%Y%m%d}_{start:%H%M}{end:%H%M}"
    return f"timewindow{duration}_{start:%Y%m%d_%H%M}_{end:%Y%m%d_%H%M}"


__all__ = [
    "normalize_timestamp",
    "format_filename_number",
    "time_window_tag",
]
