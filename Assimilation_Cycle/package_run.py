#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final packaging orchestrator (``Final_Packaging.docx``): one function,
``run_package``, that runs the complete assimilation-cycle workflow --
IRI sampling, ``EDPSamples`` construction, per-style parameterization,
observation preparation, per-style assimilation, and cross-style
comparison -- against one ``CycleConfig``, writing every intermediate and
final artifact the docx specifies under ``cfg.output_dir``.

This module is wiring only: every real step is an already-built,
already-tested function from ``iri_selection``/``ensemble_init``/
``observation_stream``/``cycle_driver``/``output``. The one piece of new
control flow is per-batch, per-RO decoding for the TEC/EDP comparison
plots (steps 7-8), done via ``cycle_driver.run_batch_loop``'s ``on_batch``
callback so a real run's per-batch density fields are decoded, plotted,
and discarded one batch at a time rather than all held in memory at once.

Deferred imports of ``Parameterization``/``edp_samples`` (only importable
once their own directories are on ``sys.path`` -- see the package
``__init__.py`` docstring), matching every other module in this package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver

from .cycle_config import CycleConfig
from . import iri_selection, ensemble_init, observation_stream, output
from .cycle_driver import CycleResult, run_cycle


@dataclass
class PackageResult:
    output_dir: Path
    edp_samples: Any
    batches: list
    results_by_style: dict[str, tuple[CycleResult, int, float]]
    """style -> (CycleResult, n_state, wall_time_seconds)."""


def _savefig(fig_or_ax_or_axes, path: Path) -> None:
    """Accepts a bare ``Figure``, a single ``Axes`` (``.figure`` used), or
    an array/list of ``Axes`` sharing one figure (e.g.
    ``Parameterized_EDPSamples.plot_reconstruction_error_statistics``'s
    ``plt.subplots(1, 2, ...)`` return) -- every plotting function used
    in this module returns one of these three shapes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = fig_or_ax_or_axes
    if isinstance(obj, (list, tuple, np.ndarray)):
        obj = obj.flat[0] if isinstance(obj, np.ndarray) else obj[0]
    fig = obj.figure if hasattr(obj, "figure") else obj
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _collect_unique_entries(batches: list) -> list:
    """Every distinct ``ObservationEntry`` referenced across all batches'
    ``entry_ray_ranges``, in first-seen order. Uses ``id()`` for
    de-duplication, not ``entry in seen``/hashing the entry itself --
    ``ObservationEntry``'s auto-generated ``__eq__`` would compare numpy
    array fields with ``==``, which raises rather than giving a bool."""
    seen_ids: set[int] = set()
    entries = []
    for batch in batches:
        for entry, _ray_slice in batch.entry_ray_ranges:
            if id(entry) not in seen_ids:
                seen_ids.add(id(entry))
                entries.append(entry)
    return entries


def run_package(cfg: CycleConfig) -> PackageResult:
    """Run the complete packaged workflow for one assimilation cycle.

    Requires ``cfg.output_dir`` and ``cfg.label`` (both used to name every
    artifact written). Every other field's existing default applies --
    in particular ``cfg.styles``/``cfg.hyper_params_by_style`` (default
    ANCHOR/PCA_3D_10ex/PCA_1D_10ex), ``cfg.n_ensemble`` (2000),
    ``cfg.altitude_grid`` (90-900km/10km), ``cfg.max_tec_per_batch``
    (unset by default -- pass 300 explicitly for the docx's stated
    default batching policy) and ``cfg.min_tangent_alt_km`` (unset by
    default -- no RO filtering unless set).
    """
    if cfg.output_dir is None:
        raise ValueError("run_package: cfg.output_dir is required")
    if not cfg.label:
        raise ValueError("run_package: cfg.label is required")

    # -- Step 1: output folder -------------------------------------------
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Step 2: IRI sample inputs (.pkl artifact) -------------------------
    # Saved regardless of edp_samples_path, so this artifact always exists
    # even in precomputed-EDPSamples mode (where it otherwise wouldn't get
    # built/saved as a side effect of anything below).
    cfg = cfg if cfg.iri_sample_inputs_output_path is not None else replace(
        cfg, iri_sample_inputs_output_path=out / f"{cfg.label}_iri_sample_inputs",
    )
    iri_selection.load_or_build_iri_sample_inputs(cfg)

    # -- Step 4 (moved ahead of step 3): EDPSamples ------------------------
    # edp_samples.sampling_parameters below is step 3's input for the
    # distribution plot -- moved here (rather than calling
    # iri_selection.sampling_parameters_for_cycle a second, independent
    # time, as this used to) so the plot always reflects the *actual*
    # ensemble in use. That separate call was a real bug found 2026-09-29:
    # in precomputed-edp_samples_path mode it silently redrew a fresh,
    # unrelated (and, with iri_spread_kwargs left at its default {}, fully
    # degenerate/zero-spread) sample never used for anything but the plot,
    # completely disconnected from the real (non-degenerate) driving
    # indices of the ensemble actually loaded and assimilated -- and even
    # in the fresh-build case it was a *second*, differently-seeded random
    # draw from the one that actually built the ensemble, not the literal
    # values used. Reading the real values back off edp_samples itself
    # fixes both.
    cfg = cfg if cfg.edp_samples_output_path is not None else replace(
        cfg, edp_samples_output_path=out / f"{cfg.label}_edp_samples.nc",
    )
    edp_samples = ensemble_init.load_or_build_edp_samples(cfg)

    # -- Step 3: input-distribution plot, from the real ensemble's own
    #    sampling_parameters (see note above) -----------------------------
    _savefig(output.plot_iri_input_distributions(edp_samples.sampling_parameters),
             out / f"{cfg.label}_iri_input_distributions.png")

    _savefig(edp_samples.plot_geolocation(), out / f"{cfg.label}_horizontal_grid.png")

    raw_edps = np.asarray(edp_samples.edps)
    for target_alt in (100, 200, 300, 400, 500):
        alt_idx = int(np.argmin(np.abs(edp_samples.altitude - target_alt)))
        actual_alt = float(edp_samples.altitude[alt_idx])
        mean_density = raw_edps[alt_idx, :, :].mean(axis=1)
        _savefig(
            output.plot_horizontal(edp_samples, mean_density, target_alt=actual_alt,
                                    scalar_label=f"mean Ne (m$^{{-3}}$) [{actual_alt:.0f} km]"),
            out / f"{cfg.label}_mean_density_{int(actual_alt)}km.png",
        )

    from Parameterization import Parameterized_EDPSamples

    parameterizations: dict[str, Any] = {}
    for style in cfg.styles:
        hyper_params = cfg.hyper_params_by_style.get(style)
        pes = Parameterized_EDPSamples(edp_samples, style=style, hyper_params=hyper_params)
        pes.saveNetCDF(out / f"{cfg.label}_{style}_parameterized.nc")
        parameterizations[style] = pes

        _savefig(pes.plot_reconstruction_error_statistics(),
                 out / f"{cfg.label}_{style}_reconstruction_error.png")

    # -- Step 5: observation preparation + geolocation/obs-operator plots -
    if cfg.ro_observations_output_path is None and cfg.obs_sources in ("RO", "both"):
        cfg = replace(cfg, ro_observations_output_path=out / f"{cfg.label}_ro_observations.nc")
    if cfg.igs_observations_output_path is None and cfg.obs_sources in ("IGS", "both"):
        cfg = replace(cfg, igs_observations_output_path=out / f"{cfg.label}_igs_observations.nc")
    batches = observation_stream.assemble(cfg)

    all_entries = _collect_unique_entries(batches)

    _savefig(output.plot_observations_geolocation(edp_samples, all_entries),
             out / f"{cfg.label}_observations_geolocation.png")

    ro_entries = [e for e in all_entries if e.obs_type == "RO"]
    for entry in ro_entries:
        fig = output.plot_observation_operator_sum(edp_samples, entry)
        _savefig(fig, out / f"{cfg.label}_obs_operator" / f"{entry.label}.png")
    if ro_entries:
        _savefig(output.plot_observation_operator_sum_combined(edp_samples, ro_entries),
                 out / f"{cfg.label}_obs_operator_sum_all_ro.png")

    # -- Steps 6-8: per-style assimilation, RMSE/rank plots, per-RO -------
    #    TEC/EDP comparison plots (via the on_batch callback -- decodes
    #    and plots one batch's forecast/analysis density at a time). -----
    # cross_style_data collects, per (batch_index, entry.label), enough to
    # build a same-RO comparison across every style once the per-style
    # loop below finishes -- keyed by batch_index too (not just entry
    # label) since an entry could in principle appear in different
    # batches across runs, and each occurrence should get its own
    # cross-style figure rather than conflating them.
    cross_style_data: dict[tuple[int, str], dict[str, Any]] = {}

    results_by_style: dict[str, tuple[CycleResult, int, float]] = {}
    for style, pes in parameterizations.items():
        ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)
        style_cfg = replace(
            cfg, style=style, hyper_params=cfg.hyper_params_by_style.get(style),
            save_intermediate_ensembles=True,
            ensemble_output_dir=out / f"{cfg.label}_{style}_ensembles",
        )
        style_dir = out / f"{cfg.label}_{style}"

        def _on_batch(batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome,
                      _style_dir=style_dir, _style=style):
            ro_here = [(e, s) for e, s in batch.entry_ray_ranges if e.obs_type == "RO"]
            if not ro_here:
                return
            decoded_forecast = obs_operator.decode(ensemble_forecast.to_param_shape())
            decoded_analysis = obs_operator.decode(ensemble_analysis.to_param_shape())
            for entry, ray_slice in ro_here:
                y_forecast = outcome.y_forecast[ray_slice]
                y_analysis = outcome.y_analysis[ray_slice]
                y_measured = outcome.y_obs_used[ray_slice]
                _savefig(
                    output.plot_tec_edp_profile_comparison(
                        entry, y_forecast, y_analysis, y_measured,
                        edp_samples, decoded_forecast, decoded_analysis,
                    ),
                    _style_dir / "profiles" / f"batch{batch.batch_index:04d}_{entry.label}.png",
                )

                key = (batch.batch_index, entry.label)
                slot = cross_style_data.setdefault(key, {"entry": entry, "tec": {}, "edp": {}})
                slot["tec"][_style] = (y_forecast, y_analysis, y_measured)
                try:
                    _, _, forecast_profile, analysis_profile = output.resolve_edp_query_point_and_profiles(
                        edp_samples, entry, decoded_forecast, decoded_analysis,
                    )
                    slot["edp"][_style] = (forecast_profile, analysis_profile)
                except ValueError:
                    slot["edp"][_style] = None   # no ray in the default 250-350km window for this entry/style

        t0 = time.time()
        result = run_cycle(style_cfg, edp_samples, pes.Parameterization, batches, ensemble_prior,
                            on_batch=_on_batch)
        wall_time_s = time.time() - t0

        _savefig(output.plot_rmse_reduction(result), style_dir / "rmse_reduction.png")
        _savefig(output.plot_rank_histogram(result), style_dir / "rank_histogram.png")
        _savefig(output.plot_effective_rank_series(result), style_dir / "effective_rank.png")
        _savefig(output.plot_igs_tec_scatter(batches, result, style_label=style),
                 style_dir / "igs_tec_scatter.png")

        # Spatial distribution of the optimal (final analysis) EDP field,
        # at the same altitudes as Step 4's forecast/prior mean-density
        # plots -- lets a reviewer compare prior vs. analysis side by side
        # for the same style. Decode needs a real (any) obs_operator for
        # this style (decode() is geometry-independent -- see
        # observation_operator.py -- so any batch's operator works).
        decode_driver = GeneralEnKFDriver(style=style, hyper_params=cfg.hyper_params_by_style.get(style))
        decode_op = decode_driver.build_observation_operator(
            edp_samples, pes.Parameterization, ensemble_prior.param_shape, batches[0].podTc2_data,
        )
        analysis_mean_density = decode_op.decode(result.final_ensemble.to_param_shape()).mean(axis=2)

        # Forecast (prior) mean density, decoded through this SAME style's
        # decode_op -- not the raw IRI ensemble mean already shown in Step
        # 4's mean_density_*.png. Using the same decode path on both sides
        # of the difference below isolates what assimilation actually
        # changed; comparing against the raw (un-encoded) mean would also
        # pick up each style's own encode/decode reconstruction error as
        # spurious "difference" (confirmed with the user 2026-10-01).
        forecast_mean_density = decode_op.decode(ensemble_prior.to_param_shape()).mean(axis=2)
        diff_density = analysis_mean_density - forecast_mean_density

        for target_alt in (100, 200, 300, 400, 500):
            alt_idx = int(np.argmin(np.abs(edp_samples.altitude - target_alt)))
            actual_alt = float(edp_samples.altitude[alt_idx])
            _savefig(
                output.plot_horizontal(
                    edp_samples, analysis_mean_density[alt_idx, :], target_alt=actual_alt,
                    scalar_label=f"analysis mean Ne (m$^{{-3}}$) [{actual_alt:.0f} km]",
                ),
                style_dir / f"{cfg.label}_{style}_analysis_mean_density_{int(actual_alt)}km.png",
            )
            abs_max = float(np.abs(diff_density[alt_idx, :]).max()) or 1.0
            _savefig(
                output.plot_horizontal(
                    edp_samples, diff_density[alt_idx, :], target_alt=actual_alt,
                    scalar_label=f"analysis - forecast mean Ne (m$^{{-3}}$) [{actual_alt:.0f} km]",
                    cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                ),
                style_dir / f"{cfg.label}_{style}_analysis_minus_forecast_density_{int(actual_alt)}km.png",
            )

        results_by_style[style] = (result, ensemble_prior.n_state, wall_time_s)

    # -- Cross-style per-RO TEC/EDP comparison (2026-09-29 user request) --
    # One two-panel figure per RO, all styles overlaid in each panel --
    # complements (not part of) the per-style profiles/ saved above.
    altitude = np.asarray(edp_samples.altitude)
    for (batch_index, entry_label), slot in cross_style_data.items():
        _savefig(
            output.plot_cross_style_tec_edp_comparison(
                slot["entry"], slot["tec"], slot["edp"], altitude,
            ),
            out / f"{cfg.label}_cross_style_profiles" / f"batch{batch_index:04d}_{entry_label}.png",
        )

    # -- Step 9: cross-style comparison ------------------------------------
    _savefig(output.plot_style_comparison_summary(results_by_style),
             out / f"{cfg.label}_style_comparison.png")

    return PackageResult(
        output_dir=out, edp_samples=edp_samples, batches=batches, results_by_style=results_by_style,
    )
