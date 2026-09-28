#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abel-profile TEC consistency check (plan Section 4.16): replicate one RO
occultation's Abel-retrieved density profile across every horizontal grid
point (a uniform field -- "if the whole ROI looked like this one
profile"), forward-model it through ``EDPSamples.get_observation_operator``
against real observation geometry, and compare the predicted TEC to real
measured TEC. The residual indicates how badly the Abel inversion's
implicit spherical-symmetry/horizontal-homogeneity assumption fails in
that region (plan Section 2, item 13 -- confirmed TEC-vs-TEC).

Standalone from the rest of the assimilation machinery on purpose: no
ensemble, no ``Parameterization``, no ``Ensemble_Kalman_Engine`` at all --
just ``EDPSamples``'s own forward operator applied to a plain density
array built from ``ObservationEntry.abel`` (``Ne``/``alt_km``, already
populated by ``Abel_Inverter.run_abel_inversion`` via
``observation_preparation.ro_source``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from observation_preparation import ObservationEntry


def build_uniform_field_from_abel(
    entry: ObservationEntry, altitude_grid: np.ndarray, n_geo: int,
) -> np.ndarray:
    """
    Interpolate ``entry.abel['Ne']`` (defined on ``entry.abel['alt_km']``)
    onto ``altitude_grid`` and replicate it across every one of ``n_geo``
    horizontal grid points.

    Altitudes in ``altitude_grid`` outside the Abel profile's own coverage
    are clamped to the nearest edge value (``numpy.interp``'s default) --
    a simplification appropriate for a consistency *check*, not a claim
    that the Abel profile is a good density estimate beyond where it was
    actually retrieved.

    Returns
    -------
    field : ndarray, shape (n_height, n_geo, 1)
        A single-member "ensemble" -- ready for
        ``EDPSamples.get_observation_operator``/``H @ field.reshape(-1, 1)``.
    """
    if not entry.abel or "Ne" not in entry.abel or "alt_km" not in entry.abel:
        raise ValueError(
            "build_uniform_field_from_abel: entry.abel is missing (or lacks "
            "'Ne'/'alt_km') -- only RO entries built with include_abel=True "
            "carry a retrieved profile."
        )

    ne = np.asarray(entry.abel["Ne"], dtype=np.float64)
    alt = np.asarray(entry.abel["alt_km"], dtype=np.float64)
    finite = np.isfinite(ne) & np.isfinite(alt)
    ne, alt = ne[finite], alt[finite]
    order = np.argsort(alt)
    ne, alt = ne[order], alt[order]

    profile = np.interp(np.asarray(altitude_grid, dtype=np.float64), alt, ne)
    n_height = profile.shape[0]
    return np.tile(profile[:, np.newaxis, np.newaxis], (1, n_geo, 1))


def predict_tec_from_abel(
    entry: ObservationEntry, edp_samples, podTc2_data: dict, num_segments: int = 1000,
) -> np.ndarray:
    """Predicted TEC (shape ``(n_rays,)``) from ``entry``'s Abel-uniform
    field, forward-modeled through ``edp_samples``'s grid against
    ``podTc2_data`` ray geometry (typically *other* observations' rays,
    not ``entry``'s own -- see ``abel_consistency_check``)."""
    n_geo = edp_samples.geolocation.shape[0]
    field = build_uniform_field_from_abel(entry, edp_samples.altitude, n_geo)
    H = edp_samples.get_observation_operator(podTc2_data, num_segments=num_segments)
    return (H @ field.reshape(-1, 1))[:, 0]


@dataclass
class AbelConsistencyResult:
    y_predicted: np.ndarray
    y_measured: np.ndarray
    obs_type: np.ndarray | None = None
    source_label: str = ""

    @property
    def residual(self) -> np.ndarray:
        return self.y_predicted - self.y_measured

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residual ** 2)))

    def rmse_by_obs_type(self) -> dict[str, float]:
        if self.obs_type is None:
            return {}
        out = {}
        for t in np.unique(self.obs_type):
            mask = self.obs_type == t
            out[str(t)] = float(np.sqrt(np.mean(self.residual[mask] ** 2)))
        return out


def abel_consistency_check(
    source_entry: ObservationEntry,
    target_entries: list[ObservationEntry],
    edp_samples,
    num_segments: int = 1000,
) -> AbelConsistencyResult:
    """
    Compare ``source_entry``'s Abel-retrieved profile (replicated as a
    uniform field) against real measured TEC from ``target_entries``.

    ``target_entries`` should generally exclude ``source_entry`` itself
    (checking against *other* geometries is the point -- the source
    occultation's own rays are close to a self-check already provided by
    ``Abel_Inverter``'s own ``TEC_forward`` field; this function does not
    enforce that exclusion itself, since a caller may deliberately want to
    include it for comparison).
    """
    rec = np.concatenate([np.asarray(e.rec_ecef_km) for e in target_entries], axis=1)
    gnss = np.concatenate([np.asarray(e.gnss_ecef_km) for e in target_entries], axis=1)
    y_measured = np.concatenate([np.asarray(e.tec) for e in target_entries])
    obs_type = np.concatenate([np.full(e.n_rays, e.obs_type) for e in target_entries])

    podTc2_data = {"rec_ecef_km": rec, "gnss_ecef_km": gnss}
    y_predicted = predict_tec_from_abel(source_entry, edp_samples, podTc2_data, num_segments=num_segments)

    return AbelConsistencyResult(
        y_predicted=y_predicted, y_measured=y_measured, obs_type=obs_type,
        source_label=source_entry.label,
    )
