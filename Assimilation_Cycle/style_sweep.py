#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style-selection sweep (plan Section 4.12): run the identical cycle (same
observations, same IRI-drawn ``EDPSamples`` climatology) once per
parameterization style, for side-by-side comparison -- the tool's stated
initial use case (plan Section 2, item 8) is choosing a style this way
before settling into a small fixed set for regular use.

Only the (cheap) ``Parameterized_EDPSamples`` encode step and the
assimilation loop differ per style; observation assembly (4.2) and the
IRI2020-based ``EDPSamples`` draw (4.4, the expensive step: one IRI2020
run per ensemble member) happen once and are shared, not repeated per
style.
"""

from __future__ import annotations

from dataclasses import replace

from Ensemble_Kalman_Engine import EnsembleState

from .cycle_config import CycleConfig
from .cycle_driver import CycleResult, run_cycle
from . import ensemble_init
from . import observation_stream


def run_style_sweep(
    cfg: CycleConfig,
    styles: list[str],
    hyper_params_by_style: dict[str, dict] | None = None,
) -> dict[str, CycleResult]:
    """
    Run ``cfg``'s cycle once per style in ``styles``.

    Parameters
    ----------
    cfg : CycleConfig
        ``cfg.style``/``cfg.hyper_params`` are ignored (overridden per
        style below); everything else (time window, ROI, grid,
        observation sources, batching, OSSE mode, ...) is shared across
        every style in the sweep. ``cfg.parameterized_edp_samples_path``
        is also ignored here even if set -- a precomputed *parameterized*
        file is style-specific by construction, so it can't be reused
        across a sweep of different styles; ``cfg.edp_samples_path``
        (the un-parameterized IRI draw) still applies normally.
    styles : list[str]
        Any registered ``Parameterization.Parameterization_Style``.
    hyper_params_by_style : dict[str, dict], optional
        Per-style ``hyper_params`` override (e.g. a different
        ``retaining_threshold`` per PCA style); styles not present here
        fall back to ``cfg.hyper_params``.

    Returns
    -------
    dict[str, CycleResult]
        Keyed by style, for side-by-side comparison (e.g. via
        ``metrics``/``output``'s RMSE-reduction and rank diagnostics).
    """
    from Parameterization import Parameterized_EDPSamples

    hyper_params_by_style = hyper_params_by_style or {}

    batches = observation_stream.assemble(cfg)
    edp_samples = ensemble_init.load_or_build_edp_samples(cfg)

    results: dict[str, CycleResult] = {}
    for style in styles:
        hyper_params = hyper_params_by_style.get(style, cfg.hyper_params)
        style_cfg = replace(cfg, style=style, hyper_params=hyper_params)

        pes = Parameterized_EDPSamples(edp_samples, style=style, hyper_params=hyper_params)
        ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)

        results[style] = run_cycle(style_cfg, edp_samples, pes.Parameterization, batches, ensemble_prior)

    return results
