#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble construction (plan Section 4.4) and its precomputed-file mode
(4.9): ``EDPSamples`` (IRI2020-populated) + ``Parameterized_EDPSamples`` ->
``EnsembleState``, the cycle's prior.

Deferred imports of ``edp_samples``/``Parameterization`` (top-level
modules only once ``EDPSamples/``/``Parameterization/`` are on
``sys.path`` -- see the package ``__init__.py`` docstring), matching
``Ensemble_Kalman_Engine.driver``'s own convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Ensemble_Kalman_Engine import EnsembleState

from .cycle_config import CycleConfig
from .diagonal_boost import apply_diagonal_boost
from .iri_selection import sampling_parameters_for_cycle


def load_or_build_edp_samples(cfg: CycleConfig):
    """Load a precomputed ``EDPSamples`` (plan Section 4.9), or run
    IRI2020 fresh over the cycle's grid/ROI -- the expensive step (one
    IRI2020 run per ensemble member over the full horizontal grid).

    If ``cfg.diagonal_boost_amplitude`` is set, applies it here -- after
    load/build, before any parameterization is fit (see
    ``diagonal_boost.py``) -- and before saving, so
    ``edp_samples_output_path`` (if set) reflects the ensemble actually
    used downstream. Applies uniformly whether the base ensemble was just
    built or loaded from a cache, so a shared cached base can be boosted
    differently per run without re-saving a separate file each time."""
    import edp_samples as E

    if cfg.edp_samples_path is not None:
        edp = E.EDPSamples.fromNetCDF(str(cfg.edp_samples_path))
    else:
        sampling_parameters = sampling_parameters_for_cycle(cfg)
        edp = E.EDPSamples(
            DateTime=cfg.center_time_str,
            geo_type="Regional",
            altitude=cfg.altitude_grid,
            sampling_parameters=sampling_parameters,
            evaluate_iri=1,
            Lon=cfg.center_lon, Lat=cfg.center_lat, radius=cfg.effective_grid_radius_deg,
            dLat=cfg.horizontal_resolution_deg,
        )

    if cfg.diagonal_boost_amplitude:
        rng = np.random.default_rng(cfg.diagonal_boost_rng_seed)
        edp = apply_diagonal_boost(
            edp, cfg.diagonal_boost_amplitude,
            vertical_scale_km=cfg.diagonal_boost_vertical_scale_km,
            horizontal_scale_km=cfg.diagonal_boost_horizontal_scale_km,
            rng=rng,
            log_space=cfg.diagonal_boost_log_space,
        )

    if cfg.edp_samples_output_path is not None:
        Path(cfg.edp_samples_output_path).parent.mkdir(parents=True, exist_ok=True)
        edp.saveNetCDF(cfg.edp_samples_output_path)

    return edp


def load_or_build_parameterized_edp_samples(cfg: CycleConfig, edp_samples=None):
    """Load a precomputed ``Parameterized_EDPSamples`` (plan Section 4.9),
    or encode a (possibly freshly-built) ``EDPSamples`` via ``cfg.style``.

    ``edp_samples`` is only needed when not loading
    ``parameterized_edp_samples_path`` directly -- pass the result of
    ``load_or_build_edp_samples`` (or omit to have this function build one
    itself)."""
    from Parameterization import Parameterized_EDPSamples

    if cfg.parameterized_edp_samples_path is not None:
        return Parameterized_EDPSamples.fromNetCDF(str(cfg.parameterized_edp_samples_path))

    if edp_samples is None:
        edp_samples = load_or_build_edp_samples(cfg)

    pes = Parameterized_EDPSamples(edp_samples, style=cfg.style, hyper_params=cfg.hyper_params)

    if cfg.parameterized_edp_samples_output_path is not None:
        Path(cfg.parameterized_edp_samples_output_path).parent.mkdir(parents=True, exist_ok=True)
        pes.saveNetCDF(cfg.parameterized_edp_samples_output_path)

    return pes


def build(cfg: CycleConfig) -> tuple[EnsembleState, object, object]:
    """
    Build the cycle's prior ensemble (plan Section 4.4) end to end:
    IRI selection -> ``EDPSamples`` -> ``Parameterized_EDPSamples`` ->
    ``EnsembleState``, honoring the precomputed-file mode (4.9) at each
    step.

    Returns
    -------
    ensemble : EnsembleState
    edp_samples : EDPSamples
        Needed downstream to build each batch's ``GenericObservationOperator``
        (``cycle_driver.run_cycle``).
    parameterization : Parameterization.EDP_Parameterization
        Likewise needed per batch.
    """
    edp_samples = None
    if cfg.parameterized_edp_samples_path is None:
        edp_samples = load_or_build_edp_samples(cfg)

    pes = load_or_build_parameterized_edp_samples(cfg, edp_samples=edp_samples)
    if edp_samples is None:
        edp_samples = pes.EDPSamples

    ensemble = EnsembleState.from_parameterized_edp_samples(pes)
    return ensemble, edp_samples, pes.Parameterization
