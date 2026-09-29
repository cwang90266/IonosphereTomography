#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-batch cycle loop (plan Section 4.6), plus intermediate-ensemble
persistence (4.10) and OSSE substitution (4.11).

Split in two layers, mirroring how ``Ensemble_Kalman_Engine`` itself
separates pure math (``kalman_core``/``analysis_engine``, toy-operator
testable) from wiring (``driver.py``, real-data tested):

- ``run_batch_loop`` -- the actual sequential-batch assimilation loop.
  Takes an already-built ``GenericObservationOperator`` per batch, so it
  has no dependency on ``EDPSamples``/real ray geometry and is fully
  unit-testable with toy operators (see
  ``Ensemble_Kalman_Engine/tests/test_analysis_engine.py``'s pattern).
- ``run_cycle`` -- the real entry point: builds the operators from a real
  ``EDPSamples``/``parameterization`` + each batch's ``podTc2_data``, then
  calls ``run_batch_loop``. This needs the heavier real-data path and is
  exercised separately (integration/real-data tests).

Per plan Section 8, decision 3: the ensemble persists between batches with
no inflation/perturbation by default (there is no state-transition model
in this filter -- decision 1). This is now an opt-in choice, not an
absolute rule: real-data review (2026-09-26) found real cases where pure
persistence may let ensemble spread shrink batch over batch with nothing
replacing it, limiting the filter's ability to respond to new
observations later in a cycle -- set ``CycleConfig.inter_batch_inflation_factor``
to re-inflate the analysis ensemble at the end of every batch (via
``Ensemble_Kalman_Engine.perturbation.SelectiveInflation``, applied
through the analysis engine's own existing perturbation hook). Leaving it
``None`` (the default) keeps exactly the original decision-8.3 behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from Ensemble_Kalman_Engine import EnsembleState, AnalysisDiagnostics
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver
from Ensemble_Kalman_Engine.osse import generate_osse_observation
from Ensemble_Kalman_Engine.perturbation import SelectiveInflation

from .cycle_config import CycleConfig
from .ensemble_io import save_ensemble_netcdf
from .metrics import RmseReduction, compute_rmse_reduction, rank_histogram_counts, effective_rank


@dataclass
class CycleBatch:
    """One chunk of chronologically-ordered observations (plan Section
    4.2), ready to assimilate."""

    podTc2_data: dict[str, np.ndarray]   # {"rec_ecef_km": (3, n_ray), "gnss_ecef_km": (3, n_ray)}
    y_obs: np.ndarray                     # (n_ray,) TEC
    R: np.ndarray                         # (n_ray, n_ray)
    obs_type: np.ndarray | None = None    # (n_ray,) e.g. "RO"/"IGS", for metrics splits
    batch_index: int = 0
    start_time: Any | None = None
    end_time: Any | None = None
    entry_ray_ranges: list[tuple[Any, slice]] = field(default_factory=list)
    """(source ObservationEntry, slice into this batch's ray axis) for
    each entry contributing rays to this batch (Final_Packaging.docx's
    per-RO TEC/EDP comparison plots need to slice y_obs/H@density back
    out per source entry) -- empty when not populated by the caller
    (observation_stream.py's assemble()/batch_entries() populate it;
    hand-built batches in tests may leave it empty)."""


@dataclass
class BatchOutcome:
    batch_index: int
    diagnostics: AnalysisDiagnostics
    rmse_reduction: RmseReduction
    rank_histogram: np.ndarray
    effective_rank_forecast: dict
    effective_rank_analysis: dict
    y_obs_used: np.ndarray   # real or OSSE-substituted, whichever was actually assimilated
    y_forecast: np.ndarray = None    # H @ forecast ensemble mean, shape (n_obs,)
    y_analysis: np.ndarray = None    # H @ analysis ensemble mean, shape (n_obs,)


@dataclass
class CycleResult:
    final_ensemble: EnsembleState
    batch_outcomes: list[BatchOutcome] = field(default_factory=list)
    osse_x_true: np.ndarray | None = None
    """The held-out truth state (plan Section 4.11), only set when
    ``osse_mode`` was used; fixed across all batches per the constant-state
    model (Section 8, decision 1)."""


def _roi_dict(cfg: CycleConfig) -> dict[str, Any]:
    return {"center_lat": cfg.center_lat, "center_lon": cfg.center_lon, "radius_km": cfg.radius_km}


def run_batch_loop(
    ensemble_prior: EnsembleState,
    driver: GeneralEnKFDriver,
    obs_operators: list,
    batches: list[CycleBatch],
    cfg: CycleConfig | None = None,
    on_batch=None,
    rng: np.random.Generator | None = None,
) -> CycleResult:
    """
    Run the sequential-batch assimilation loop (plan Section 4.6).

    Parameters
    ----------
    ensemble_prior : EnsembleState
        The cycle's prior (Section 4.4) -- if ``cfg.osse_mode`` is set, one
        member is held out from this as the OSSE truth (Section 4.11)
        before the loop starts, and the remaining members seed the
        forecast.
    driver : GeneralEnKFDriver
        Configured for the cycle's style (``driver.config`` supplies the
        default ``AnalysisConfig`` used for every batch).
    obs_operators : list[GenericObservationOperator]
        One per batch, same order/length as ``batches`` -- built by the
        caller (``run_cycle`` for real data, or directly by a test with
        toy operators).
    batches : list[CycleBatch]
        Chronologically ordered (plan Section 4.2); order has no effect on
        the math itself (Section 8, decision 1) but is what makes a later
        cycling system possible (Section 8, decision 5).
    cfg : CycleConfig, optional
        Used for ``osse_mode``/``osse_rng_seed`` and
        ``save_intermediate_ensembles``/``ensemble_output_dir``. Omit for
        a bare toy-operator test that doesn't need either.
    on_batch : callable, optional
        ``on_batch(batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome)``,
        invoked once per batch right after that batch's outcome is built
        (forecast = the prior going into this batch, analysis = the
        posterior coming out). For callers that need to decode density
        and plot/save per batch (e.g. ``package_run.py``'s per-RO
        TEC/EDP comparison plots, plan/docx steps 7-8) without this loop
        itself retaining every batch's ensemble in memory -- real runs
        can have ``n_state x n_members`` large enough that holding all of
        them at once is a real cost. ``BatchOutcome.y_forecast``/
        ``y_analysis`` already cover the common case that doesn't need
        the full ensemble (predicted TEC, not density).
    rng : numpy.random.Generator, optional
        Overrides ``cfg.osse_rng_seed`` if both are given; used for OSSE
        noise synthesis and rank-histogram tie-breaking.
    """
    if len(obs_operators) != len(batches):
        raise ValueError(
            f"run_batch_loop: {len(obs_operators)} obs_operators for {len(batches)} batches"
        )

    osse_mode = bool(cfg is not None and cfg.osse_mode)
    if rng is None:
        seed = cfg.osse_rng_seed if cfg is not None else None
        rng = np.random.default_rng(seed)

    ensemble = ensemble_prior
    x_true = None
    if osse_mode:
        if ensemble.n_members < 2:
            raise ValueError("run_batch_loop: osse_mode needs at least 2 ensemble members "
                              "(one held out as truth, the rest seed the forecast).")
        x_true = ensemble.X[:, 0].copy()
        ensemble = EnsembleState(X=ensemble.X[:, 1:], param_shape=ensemble.param_shape)

    save_intermediate = bool(
        cfg is not None and cfg.save_intermediate_ensembles and cfg.ensemble_output_dir is not None
    )

    # Opt-in inter-batch inflation (see module docstring): None (default)
    # keeps exact decision-8.3 persistence; a set factor re-inflates the
    # analysis ensemble at the end of every batch via the analysis
    # engine's own existing perturbation hook.
    inter_batch_perturbation = None
    if cfg is not None and cfg.inter_batch_inflation_factor is not None:
        inter_batch_perturbation = SelectiveInflation(factor=cfg.inter_batch_inflation_factor)

    outcomes: list[BatchOutcome] = []
    for batch, obs_operator in zip(batches, obs_operators):
        y_obs = np.asarray(batch.y_obs)
        if osse_mode:
            osse_obs = generate_osse_observation(x_true, obs_operator, batch.R, rng=rng)
            y_obs = osse_obs.y_obs

        y_forecast_mean = obs_operator.forward_single(ensemble.ensemble_mean())
        Y_forecast_ensemble = obs_operator.forward_ensemble(ensemble.X)
        rank_forecast = effective_rank(ensemble.X)

        ensemble_a, diag = driver.assimilate_one_cycle(
            ensemble, obs_operator, y_obs, batch.R, perturbation=inter_batch_perturbation,
        )

        y_analysis_mean = obs_operator.forward_single(ensemble_a.ensemble_mean())
        rank_analysis = effective_rank(ensemble_a.X)
        rmse_red = compute_rmse_reduction(y_obs, y_forecast_mean, y_analysis_mean, obs_type=batch.obs_type)
        rank_hist = rank_histogram_counts(y_obs, Y_forecast_ensemble, rng=rng)

        if save_intermediate:
            out_path = Path(cfg.ensemble_output_dir) / f"ensemble_batch_{batch.batch_index:04d}.nc"
            save_ensemble_netcdf(
                ensemble_a, out_path,
                style=driver.style, hyper_params=driver.hyper_params,
                cycle_start_time=str(cfg.start_time), cycle_end_time=str(cfg.end_time),
                roi=_roi_dict(cfg), batch_index=batch.batch_index,
                batch_start_time=str(batch.start_time) if batch.start_time is not None else None,
                batch_end_time=str(batch.end_time) if batch.end_time is not None else None,
            )

        outcome = BatchOutcome(
            batch_index=batch.batch_index,
            diagnostics=diag,
            rmse_reduction=rmse_red,
            rank_histogram=rank_hist,
            effective_rank_forecast=rank_forecast,
            effective_rank_analysis=rank_analysis,
            y_obs_used=y_obs,
            y_forecast=y_forecast_mean,
            y_analysis=y_analysis_mean,
        )
        outcomes.append(outcome)

        if on_batch is not None:
            on_batch(batch, obs_operator, ensemble, ensemble_a, outcome)

        # Persistence (plan Section 8, decision 3): no inflation/perturbation
        # between batches -- the analysis ensemble is simply next batch's prior.
        ensemble = ensemble_a

    return CycleResult(final_ensemble=ensemble, batch_outcomes=outcomes, osse_x_true=x_true)


def run_cycle(cfg: CycleConfig, edp_samples, parameterization, batches: list[CycleBatch],
              ensemble_prior: EnsembleState, on_batch=None) -> CycleResult:
    """
    Real entry point (plan Section 4.6): build one ``GenericObservationOperator``
    per batch from real geometry, then run ``run_batch_loop``.

    ``edp_samples``/``parameterization``/``ensemble_prior`` come from
    ``ensemble_init.build`` (Section 4.4); ``batches`` from
    ``observation_stream.assemble`` (Section 4.2). ``on_batch``: see
    ``run_batch_loop``'s docstring.
    """
    driver = GeneralEnKFDriver(style=cfg.style, hyper_params=cfg.hyper_params)
    obs_operators = [
        driver.build_observation_operator(
            edp_samples, parameterization, ensemble_prior.param_shape, batch.podTc2_data,
        )
        for batch in batches
    ]
    return run_batch_loop(ensemble_prior, driver, obs_operators, batches, cfg=cfg, on_batch=on_batch)
