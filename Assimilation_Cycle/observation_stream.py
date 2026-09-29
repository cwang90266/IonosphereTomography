#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observation assembly, merge, and batching (plan Section 4.2), the RO/IGS
source toggle (4.15), and the precomputed-file load/save half of Section
4.9 that applies to observations.

``observation_preparation.prepare_ro_observations``/``prepare_igs_observations``
return raw "clean entry" dicts, not ``ObservationEntry`` objects -- the
conversion normally happens inside that package's own ``export_ro_outputs``/
``export_igs_outputs`` (which also unconditionally plots a geolocation
map). This module converts via ``ObservationEntry.from_dict`` directly and
calls ``write_observations``/``read_observations`` itself instead, so a
fresh cycle run doesn't pay for a plot it didn't ask for -- diagnostics
are a separate, opt-in concern (``output.py``, plan Section 4.14).

Batching unit: each ``CycleBatch`` bundles ``cfg.batch_size`` consecutive
(by ``ObservationEntry.date``) RO occultations / IGS arcs -- not
individual rays -- concatenating every ray across those entries into one
``podTc2_data``/``y_obs``. Per plan Section 8, decision 1, this has no
physical meaning (there is no state-transition model); it exists purely
to size each analysis-engine call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from observation_preparation import (
    ObservationEntry,
    prepare_ro_observations,
    prepare_igs_observations,
    write_observations,
    read_observations,
    build_roi_dict,
)

from .cycle_config import CycleConfig
from .cycle_driver import CycleBatch


def _load_or_compute_ro(cfg: CycleConfig) -> list[ObservationEntry]:
    if cfg.ro_observations_path is not None:
        return read_observations(cfg.ro_observations_path)

    raw = prepare_ro_observations(
        center_lat=cfg.center_lat, center_lon=cfg.center_lon, radius_km=cfg.radius_km,
        start_time=cfg.start_time, end_time=cfg.end_time,
        **cfg.ro_kwargs,
    )
    entries = [ObservationEntry.from_dict(d, obs_type="RO") for d in raw]

    if cfg.ro_observations_output_path is not None:
        roi = build_roi_dict(
            cfg.center_lat, cfg.center_lon, cfg.radius_km,
            cfg.ro_kwargs.get("roi_mode", "full_los"),
            alt_limit_km=cfg.ro_kwargs.get("alt_limit_km"),
            fraction_required=cfg.ro_kwargs.get("fraction_required"),
        )
        write_observations(entries, cfg.ro_observations_output_path, roi=roi)

    return entries


def _load_or_compute_igs(cfg: CycleConfig) -> list[ObservationEntry]:
    if cfg.igs_observations_path is not None:
        return read_observations(cfg.igs_observations_path)

    missing = [k for k in ("stations", "cache_dir") if k not in cfg.igs_kwargs]
    if missing:
        raise ValueError(
            f"observation_stream: cfg.igs_kwargs is missing required key(s) {missing} "
            "(prepare_igs_observations needs 'stations' and 'cache_dir')."
        )

    raw = prepare_igs_observations(
        center_lat=cfg.center_lat, center_lon=cfg.center_lon, radius_km=cfg.radius_km,
        start_time=cfg.start_time, end_time=cfg.end_time,
        **cfg.igs_kwargs,
    )
    entries = [ObservationEntry.from_dict(d, obs_type="IGS") for d in raw]

    if cfg.igs_observations_output_path is not None:
        roi = build_roi_dict(
            cfg.center_lat, cfg.center_lon, cfg.radius_km,
            cfg.igs_kwargs.get("roi_mode", "full_los"),
            alt_limit_km=cfg.igs_kwargs.get("alt_limit_km"),
            fraction_required=cfg.igs_kwargs.get("fraction_required"),
        )
        write_observations(entries, cfg.igs_observations_output_path, roi=roi)

    return entries


def _entries_to_batch(entries: list[ObservationEntry], obs_sigma: float, batch_index: int) -> CycleBatch:
    rec = np.concatenate([np.asarray(e.rec_ecef_km) for e in entries], axis=1)
    gnss = np.concatenate([np.asarray(e.gnss_ecef_km) for e in entries], axis=1)
    tec = np.concatenate([np.asarray(e.tec) for e in entries])
    obs_type = np.concatenate([np.full(e.n_rays, e.obs_type) for e in entries])
    n = tec.shape[0]
    R = (obs_sigma ** 2) * np.eye(n)

    dates = [e.date for e in entries if e.date is not None]
    entry_ray_ranges = []
    start = 0
    for e in entries:
        entry_ray_ranges.append((e, slice(start, start + e.n_rays)))
        start += e.n_rays

    return CycleBatch(
        podTc2_data={"rec_ecef_km": rec, "gnss_ecef_km": gnss},
        y_obs=tec, R=R, obs_type=obs_type,
        batch_index=batch_index,
        start_time=min(dates) if dates else None,
        end_time=max(dates) if dates else None,
        entry_ray_ranges=entry_ray_ranges,
    )


def _filter_entry_rays(entry: ObservationEntry, mask: np.ndarray) -> ObservationEntry | None:
    """Return a copy of ``entry`` with only the rays where ``mask`` is
    True, or None if none survive. Per-ray fields are the known scalar-
    per-ray/xyz-per-ray schema fields plus any ``extra`` array whose last
    axis matches the original ray count; ``abel`` (one retrieved profile
    per RO occultation, not per ray) and every other scalar field pass
    through unchanged."""
    if mask.all():
        return entry
    if not mask.any():
        return None

    from dataclasses import replace

    n_rays = entry.n_rays
    per_ray_1d = ("tec", "tangent_alt_km", "pierce_lat", "pierce_lon", "elev_deg", "snr_l1", "snr_l2")
    per_ray_xyz = ("gnss_ecef_km", "rec_ecef_km")

    updates: dict = {}
    for field_name in per_ray_1d:
        val = getattr(entry, field_name)
        if val is not None:
            updates[field_name] = np.asarray(val)[mask]
    for field_name in per_ray_xyz:
        val = getattr(entry, field_name)
        if val is not None:
            updates[field_name] = np.asarray(val)[:, mask]

    new_extra = dict(entry.extra)
    for key, val in entry.extra.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[-1] == n_rays:
            new_extra[key] = arr[..., mask]
    updates["extra"] = new_extra

    return replace(entry, **updates)


def _apply_min_tangent_alt_filter(entries: list[ObservationEntry], min_tangent_alt_km: float) -> list[ObservationEntry]:
    """RO-only (Final_Packaging.docx): drop rays with ``tangent_alt_km``
    below the threshold; drop an entry entirely if no rays survive. IGS
    entries pass through unchanged (no tangent-point concept the same
    way)."""
    filtered = []
    for e in entries:
        if e.obs_type != "RO" or e.tangent_alt_km is None:
            filtered.append(e)
            continue
        mask = np.asarray(e.tangent_alt_km) >= min_tangent_alt_km
        kept = _filter_entry_rays(e, mask)
        if kept is not None:
            filtered.append(kept)
    return filtered


def batch_entries(entries: list[ObservationEntry], cfg: CycleConfig) -> list[CycleBatch]:
    """Sort ``entries`` chronologically (plan Section 4.2 -- for future
    cross-cycle continuity, Section 8 decision 5; no effect on the math
    within one cycle, decision 1), apply ``cfg.min_tangent_alt_km``'s
    RO ray filter if set, and split into batches.

    Batching policy: ``cfg.max_tec_per_batch`` (TEC/ray-count cap,
    Final_Packaging.docx) if set, else ``cfg.batch_size`` (entry-count,
    the original policy -- kept as the default so existing single-style
    callers/tests are unaffected). Under the TEC-count policy, entries
    accumulate into the current batch until adding the next one would
    exceed the cap; an entry larger than the cap on its own still gets
    its own batch rather than being split, since batches are what the
    per-entry TEC/EDP comparison plots slice back out of via
    ``CycleBatch.entry_ray_ranges``.
    """
    entries = sorted(entries, key=lambda e: (e.date is None, e.date))
    if cfg.min_tangent_alt_km is not None:
        entries = _apply_min_tangent_alt_filter(entries, cfg.min_tangent_alt_km)

    batches = []
    if cfg.max_tec_per_batch is not None:
        chunk: list[ObservationEntry] = []
        chunk_rays = 0
        for e in entries:
            if chunk and chunk_rays + e.n_rays > cfg.max_tec_per_batch:
                batches.append(_entries_to_batch(chunk, cfg.obs_sigma, batch_index=len(batches)))
                chunk, chunk_rays = [], 0
            chunk.append(e)
            chunk_rays += e.n_rays
        if chunk:
            batches.append(_entries_to_batch(chunk, cfg.obs_sigma, batch_index=len(batches)))
    else:
        for i in range(0, len(entries), cfg.batch_size):
            chunk = entries[i:i + cfg.batch_size]
            batches.append(_entries_to_batch(chunk, cfg.obs_sigma, batch_index=len(batches)))
    return batches


def assemble(cfg: CycleConfig) -> list[CycleBatch]:
    """
    Assemble one cycle's observations end to end (plan Section 4.2):
    source selection (4.15) -> precomputed-file load or fresh
    RO/IGS preparation (4.9) -> chronological merge -> batching.
    """
    entries: list[ObservationEntry] = []
    if cfg.obs_sources in ("RO", "both"):
        entries.extend(_load_or_compute_ro(cfg))
    if cfg.obs_sources in ("IGS", "both"):
        entries.extend(_load_or_compute_igs(cfg))

    return batch_entries(entries, cfg)
