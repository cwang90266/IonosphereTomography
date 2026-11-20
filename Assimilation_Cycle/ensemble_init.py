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

import warnings
from pathlib import Path

import numpy as np

from Ensemble_Kalman_Engine import EnsembleState

from .cycle_config import CycleConfig
from .diagonal_boost import apply_diagonal_boost
from .iri_selection import sampling_parameters_for_cycle
from .isr_pca_basis import resolve_hyper_params_for_style

_NAN_DENSITY_FLOOR = 1.0e6
"""m^-3; same floor value as diagonal_boost.py's `_DENSITY_FLOOR` ("raised
2026-11-16 from a 1.0 m^-3 no-op-sized floor, still physically negligible
next to real EDP values of >= ~1e8") -- reused here for the identical
reason, not re-derived. Defined separately (not imported) since that one is
a private module constant and the two floors are conceptually independent
even though they share a value."""


def _replace_nan_density(edp) -> None:
    """IRI2020 has occasionally produced `NaN` Ne at low latitude when
    evaluated down to low altitudes (e.g. the 60 km floor the `SPLINE`
    style's training collection needs) -- a Fortran-side fix was made, but
    isn't confirmed to fully eliminate it (user, 2026-11-19). A `NaN`
    silently reaching any style's fit (`get_PCA`'s eigendecomposition,
    `Spline_Basis.spline_fit`'s per-column least squares) would corrupt it
    with no clear error, so guard here, before any parameterization is
    built, for every style -- not just `SPLINE`. Replaces in place."""
    edps = edp.edps
    nan_mask = np.isnan(edps)
    n_nan = int(nan_mask.sum())
    if n_nan:
        alt_idx, geo_idx, mem_idx = np.where(nan_mask)
        warnings.warn(
            f"load_or_build_edp_samples: replacing {n_nan} NaN Ne value(s) with "
            f"{_NAN_DENSITY_FLOOR:g} m^-3 (altitude indices {sorted(set(alt_idx.tolist()))}, "
            f"{len(set(zip(geo_idx.tolist(), mem_idx.tolist())))} distinct (geo, member) cells affected) "
            "-- likely an IRI2020 low-latitude/low-altitude evaluation issue.",
            stacklevel=3,
        )
        edps[nan_mask] = _NAN_DENSITY_FLOOR


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
        _replace_nan_density(edp)

    if cfg.diagonal_boost_amplitude:
        rng = np.random.default_rng(cfg.diagonal_boost_rng_seed)
        edp = apply_diagonal_boost(
            edp, cfg.diagonal_boost_amplitude,
            vertical_scale_km=cfg.diagonal_boost_vertical_scale_km,
            horizontal_scale_km=cfg.diagonal_boost_horizontal_scale_km,
            rng=rng,
            log_space=cfg.diagonal_boost_log_space,
            taper_start_km=cfg.diagonal_boost_taper_start_km,
            taper_end_km=cfg.diagonal_boost_taper_end_km,
            taper_floor=cfg.diagonal_boost_taper_floor,
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

    hyper_params = resolve_hyper_params_for_style(cfg, cfg.style, cfg.hyper_params)
    pes = Parameterized_EDPSamples(edp_samples, style=cfg.style, hyper_params=hyper_params)

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
