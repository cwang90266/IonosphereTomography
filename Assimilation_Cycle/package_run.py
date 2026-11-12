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
from Ensemble_Kalman_Engine.analysis_engine import AnalysisConfig
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver

from .cycle_config import CycleConfig
from . import iri_selection, ensemble_init, observation_stream, output, isr_comparison, impact_vectors
from . import abel_consistency
from .isr_pca_basis import resolve_hyper_params_for_style
from .cycle_driver import CycleResult, run_cycle


@dataclass
class PackageResult:
    output_dir: Path
    edp_samples: Any
    batches: list
    results_by_style: dict[str, tuple[CycleResult, int, float]]
    """style -> (CycleResult, n_state, wall_time_seconds)."""
    isr_matches_by_style: dict[str, list] = field(default_factory=dict)
    """style -> list[isr_comparison.BatchIsrMatch], populated only when
    ``cfg.isr_file_path`` was set (ISR_Integration_Plan.md Section 4.3);
    empty otherwise."""
    isr_matches_by_tag_and_style: dict[str, dict[str, list]] = field(default_factory=dict)
    """ISR output tag -> style -> list[BatchIsrMatch].

    The empty-string tag is the backward-compatible primary
    ``cfg.isr_file_path``; named tags come from
    ``cfg.additional_isr_file_paths``.
    """


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


def _save_csv(df, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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

    # Run-level boost status (2026-10-17 user request): stamped onto the
    # TEC/EDP profile plot titles below so a saved/shared image is
    # self-identifying without needing its file path/folder name.
    boost_label = (
        f"boosted (amplitude={cfg.diagonal_boost_amplitude:g})"
        if cfg.diagonal_boost_amplitude is not None else "no boost"
    )

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

    # -- ISR comparison inputs (ISR_Integration_Plan.md Section 4.3), optional --
    # The primary ISR keeps the historical untagged output naming. Additional
    # stations use a caller-supplied short tag (e.g. "svl") but otherwise go
    # through the exact same isr_comparison routines.
    isr_by_tag: dict[str, Any] = {}
    if cfg.isr_file_path is not None:
        isr_by_tag[""] = isr_comparison.load_isr_dataset(cfg.isr_file_path)
    for isr_tag, isr_path in cfg.additional_isr_file_paths.items():
        isr_by_tag[str(isr_tag)] = isr_comparison.load_isr_dataset(isr_path)

    # -- Impact-vector query points (Impact_Vector_Plan.md Section 7/10),
    #    optional -- explicit points plus, when a primary ISR file is
    #    configured, that station's point added automatically. ----------
    impact_vector_points: list[impact_vectors.QueryPoint] = []
    if cfg.compute_impact_vectors:
        model_altitude_for_iv = np.asarray(edp_samples.altitude, dtype=float)
        for name, lat, lon in (cfg.impact_vector_query_points or []):
            impact_vector_points.append(
                impact_vectors.QueryPoint(name=str(name), lat=float(lat), lon=float(lon),
                                           altitudes=model_altitude_for_iv)
            )
        if "" in isr_by_tag:
            impact_vector_points.append(
                impact_vectors.query_point_from_isr_station(isr_by_tag[""].station, edp_samples)
            )

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
        hyper_params = resolve_hyper_params_for_style(cfg, style, cfg.hyper_params_by_style.get(style))
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
        # overlay_los defaults True here (per-entry plot); the combined
        # plot below defaults it off, since pooling every entry's track
        # gets visually dense fast -- pass overlay_los explicitly to turn
        # it on for a one-off diagnostic.
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

    # Same (batch_index, ...)-keyed cross-style collection, for the
    # impact-vector ranking plot below -- keyed by query point name
    # instead of entry label, since one figure compares every RO/IGS
    # entry's impact at a given point, across styles.
    impact_vector_cross_style_data: dict[tuple[int, str], dict[str, dict[str, np.ndarray]]] = {}

    def _run_impact_vectors_for_batch(batch, obs_operator, ensemble_forecast, outcome,
                                       style, style_dir, pes) -> None:
        """Impact_Vector_Plan.md Section 7's diagnostic pass: a second,
        ``centering="mean"``-forced ``analyze()`` call over this batch's
        exact forecast ensemble/observations (``outcome.y_obs_used``,
        matching whatever the production call actually assimilated,
        including under OSSE substitution)/``R`` -- purely read-only,
        never feeds back into the production analysis above, whose own
        result this never sees."""
        accumulator = impact_vectors.ImpactVectorAccumulator(
            edp_samples, obs_operator, impact_vector_points, batch.entry_ray_ranges,
        )
        diag_driver = GeneralEnKFDriver(
            style=style, hyper_params=pes.Parameterization.hyper_params,
            config=AnalysisConfig.default_for_style(style, centering="mean"),
        )
        _, iv_diag = diag_driver.assimilate_one_cycle(
            ensemble_forecast, obs_operator, outcome.y_obs_used, batch.R,
            on_iteration=accumulator.on_iteration,
        )
        impacts = accumulator.finalize(iv_diag.alpha_history)

        decoded_forecast_iv = None   # built lazily, only if an ISR-alignment plot below needs it
        isr_dataset_primary = isr_by_tag.get("")

        for qp in impact_vector_points:
            impacts_by_ro = {
                entry_label: profile
                for (entry_label, point_name), profile in impacts.items()
                if point_name == qp.name
            }
            if not impacts_by_ro:
                continue   # this point had no matching entries in this particular batch

            iv_dir = style_dir / "impact_vectors" / qp.name
            batch_tag = f"batch{batch.batch_index:04d}"
            _savefig(
                impact_vectors.plot_impact_vector_heatmap(
                    impacts_by_ro, qp.altitudes,
                    title=f"{qp.name} impact vectors ({style}, batch {batch.batch_index})",
                ),
                iv_dir / f"{batch_tag}_heatmap.png",
            )
            _savefig(
                impact_vectors.plot_impact_vector_ranking(
                    impacts_by_ro,
                    title=f"{qp.name} impact ranking ({style}, batch {batch.batch_index})",
                ),
                iv_dir / f"{batch_tag}_ranking.png",
            )
            _savefig(
                impact_vectors.plot_impact_vector_ranking_relative_to_norm(
                    impacts_by_ro,
                    title=f"{qp.name} impact ranking, relative to total change ({style}, batch {batch.batch_index})",
                ),
                iv_dir / f"{batch_tag}_ranking_relative_norm.png",
            )
            _savefig(
                impact_vectors.plot_impact_vector_ranking_relative_to_local_change(
                    impacts_by_ro, qp.altitudes,
                    title=f"{qp.name} impact ranking, relative to local change ({style}, batch {batch.batch_index})",
                ),
                iv_dir / f"{batch_tag}_ranking_relative_local.png",
            )

            impact_vector_cross_style_data.setdefault((batch.batch_index, qp.name), {})[style] = impacts_by_ro

            forecast_profile_for_save = None
            isr_density_for_save = None
            isr_altitude_for_save = None

            if isr_dataset_primary is not None and qp.name == isr_dataset_primary.station.name:
                mid = isr_comparison.batch_midpoint(batch)
                nearest = isr_comparison.nearest_isr_profile(isr_dataset_primary, mid) if mid is not None else None
                if nearest is not None:
                    _, isr_density_nearest = nearest
                    if decoded_forecast_iv is None:
                        decoded_forecast_iv = obs_operator.decode(ensemble_forecast.to_param_shape())
                    _, _, forecast_profile_pt, _ = output.resolve_edp_query_point_and_profiles(
                        edp_samples, None, decoded_forecast_iv, decoded_forecast_iv,
                        lat=qp.lat, lon=qp.lon,
                    )
                    forecast_profile_mean = forecast_profile_pt.mean(axis=1)
                    _savefig(
                        impact_vectors.plot_impact_vector_isr_alignment(
                            impacts_by_ro, forecast_profile_mean, isr_density_nearest,
                            isr_dataset_primary.altitude, qp.altitudes,
                            title=f"{qp.name} ISR-alignment ranking ({style}, batch {batch.batch_index})",
                        ),
                        iv_dir / f"{batch_tag}_isr_alignment.png",
                    )
                    _savefig(
                        impact_vectors.plot_impact_vector_isr_aligned_heatmap(
                            impacts_by_ro, forecast_profile_mean, isr_density_nearest,
                            isr_dataset_primary.altitude, qp.altitudes,
                            title=f"{qp.name} ISR-aligned heatmap ({style}, batch {batch.batch_index})",
                        ),
                        iv_dir / f"{batch_tag}_isr_aligned_heatmap.png",
                    )
                    forecast_profile_for_save = forecast_profile_mean
                    isr_density_for_save = isr_density_nearest
                    isr_altitude_for_save = isr_dataset_primary.altitude

            # Persist the raw per-entry profiles (plan: user request
            # 2026-11-06) -- a later plotting-only change can regenerate
            # every plot above straight from this file, via
            # impact_vectors.replot_impact_vectors_from_saved /
            # replot_impact_vectors_tree, with no pipeline rerun at all.
            impact_vectors.save_impact_vectors(
                iv_dir / f"{batch_tag}_data.nc", impacts_by_ro, qp.altitudes,
                style=style, batch_index=batch.batch_index, query_point_name=qp.name,
                forecast_profile=forecast_profile_for_save,
                isr_density=isr_density_for_save, isr_altitude=isr_altitude_for_save,
            )

    results_by_style: dict[str, tuple[CycleResult, int, float]] = {}
    # Backward-compatible primary-ISR matches plus the full multi-ISR mapping.
    isr_matches_by_style: dict[str, list] = {}
    isr_matches_by_tag_and_style: dict[str, dict[str, list]] = {
        tag: {} for tag in isr_by_tag
    }
    for style, pes in parameterizations.items():
        ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)
        style_cfg = replace(
            # The actual resolved hyper_params (incl. a loaded ISR PCA basis,
            # if any) -- pes.Parameterization.hyper_params, not
            # cfg.hyper_params_by_style.get(style) directly, which for
            # 'PCA_1D_10ex_ISR' would be missing the loaded 'PCA'/'PCA_mean'.
            # Functionally inert for run_cycle itself (it receives pes's
            # already-built parameterization object separately), but kept
            # consistent for provenance.
            cfg, style=style, hyper_params=pes.Parameterization.hyper_params,
            save_intermediate_ensembles=True,
            ensemble_output_dir=out / f"{cfg.label}_{style}_ensembles",
        )
        style_dir = out / f"{cfg.label}_{style}"

        isr_collectors = {
            tag: isr_comparison.IsrComparisonCollector(edp_samples, isr_dataset)
            for tag, isr_dataset in isr_by_tag.items()
        }

        def _on_batch(batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome,
                      _style_dir=style_dir, _style=style, _isr_collectors=isr_collectors, _pes=pes):
            for _collector in _isr_collectors.values():
                _collector.on_batch(
                    batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome
                )

            if cfg.compute_impact_vectors and impact_vector_points and batch.entry_ray_ranges:
                _run_impact_vectors_for_batch(
                    batch, obs_operator, ensemble_forecast, outcome, _style, _style_dir, _pes,
                )

            ro_here = [(e, s) for e, s in batch.entry_ray_ranges if e.obs_type == "RO"]
            if not ro_here:
                return
            decoded_forecast = obs_operator.decode(ensemble_forecast.to_param_shape())
            decoded_analysis = obs_operator.decode(ensemble_analysis.to_param_shape())
            # Full forecast-ensemble TEC (2026-11-12 user request #1: the TEC
            # panel showed only the forecast *mean*, with no indication of
            # the prior ensemble's spread, unlike the EDP panel) -- computed
            # once per batch (not per entry) since it covers every ray in
            # the batch; sliced per entry below like y_forecast/y_analysis.
            y_forecast_ensemble_all = obs_operator.forward_ensemble(ensemble_forecast.X)
            for entry, ray_slice in ro_here:
                y_forecast = outcome.y_forecast[ray_slice]
                y_analysis = outcome.y_analysis[ray_slice]
                y_measured = outcome.y_obs_used[ray_slice]
                y_forecast_ensemble = y_forecast_ensemble_all[ray_slice]

                # Spherically-symmetric Abel-consistency check (2026-11-12
                # user request #2): replicate this RO's own Abel-retrieved
                # profile across every geo vertex -- the same
                # horizontal-homogeneity assumption the Abel inversion
                # itself makes -- and forward-model that uniform field
                # through this entry's own ray geometry, so the TEC panel
                # can show whether the Abel profile alone (no tomography)
                # already reproduces the measured TEC.
                y_abel_spherical = None
                if entry.abel and "Ne" in entry.abel and "alt_km" in entry.abel:
                    abel_field = abel_consistency.build_uniform_field_from_abel(
                        entry, edp_samples.altitude, edp_samples.geolocation.shape[0],
                    )
                    y_abel_spherical = obs_operator.forward_from_density(abel_field[..., 0])[ray_slice]

                _savefig(
                    output.plot_tec_edp_profile_comparison(
                        entry, y_forecast, y_analysis, y_measured,
                        edp_samples, decoded_forecast, decoded_analysis,
                        style=_style, boost_label=boost_label,
                        y_forecast_ensemble=y_forecast_ensemble,
                        y_abel_spherical=y_abel_spherical,
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

        # -- Per-style ISR comparison for every configured ISR station. --
        for isr_tag, isr_collector in isr_collectors.items():
            isr_dataset = isr_by_tag[isr_tag]
            isr_matches_by_tag_and_style[isr_tag][style] = isr_collector.matches
            if isr_tag == "":
                # Preserve PackageResult's historical primary-ISR field.
                isr_matches_by_style[style] = isr_collector.matches

            comparison_name = (
                f"{cfg.label}_isr_comparison"
                if isr_tag == ""
                else f"{cfg.label}_{isr_tag}_isr_comparison"
            )
            isr_dir = out / comparison_name / style
            model_altitude = np.asarray(edp_samples.altitude)

            for match in isr_collector.matches:
                _savefig(
                    isr_comparison.plot_isr_nearest_point_comparison(
                        match,
                        model_altitude,
                        isr_dataset.altitude,
                        station_name=isr_dataset.station.name,
                        style=style,
                    ),
                    isr_dir / "nearest_point" / f"batch{match.batch_index:04d}.png",
                )

            _savefig(
                isr_comparison.plot_isr_range_comparison(
                    isr_collector.matches,
                    isr_dataset,
                    cfg.start_time,
                    cfg.end_time,
                    model_altitude,
                    percentile=cfg.isr_range_percentile,
                    style=style,
                ),
                isr_dir / "range_comparison.png",
            )

            rmse_df = isr_comparison.pooled_isr_rmse_by_altitude(
                isr_collector.matches,
                model_altitude,
                isr_dataset.altitude,
            )
            _save_csv(rmse_df, isr_dir / "pooled_rmse.csv")
            _savefig(
                isr_comparison.plot_pooled_isr_rmse(rmse_df, style=style),
                isr_dir / "pooled_rmse.png",
            )

        # Spatial distribution of the optimal (final analysis) EDP field,
        # at the same altitudes as Step 4's forecast/prior mean-density
        # plots -- lets a reviewer compare prior vs. analysis side by side
        # for the same style. Decode needs a real (any) obs_operator for
        # this style (decode() is geometry-independent -- see
        # observation_operator.py -- so any batch's operator works).
        decode_driver = GeneralEnKFDriver(style=style, hyper_params=pes.Parameterization.hyper_params)
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
                boost_label=boost_label,
            ),
            out / f"{cfg.label}_cross_style_profiles" / f"batch{batch_index:04d}_{entry_label}.png",
        )

    # -- Cross-style impact-vector ranking (Impact_Vector_Plan.md Section 8),
    #    only populated when cfg.compute_impact_vectors was set. -----------
    for (batch_index, point_name), impacts_by_ro_by_style in impact_vector_cross_style_data.items():
        _savefig(
            impact_vectors.plot_impact_vector_ranking_cross_style(
                impacts_by_ro_by_style,
                title=f"{point_name} impact ranking, batch {batch_index} (all styles)",
            ),
            out / f"{cfg.label}_impact_vectors_cross_style" / point_name / f"batch{batch_index:04d}_ranking.png",
        )

    # -- Cross-style ISR comparison for every configured ISR station. -----
    for isr_tag, isr_dataset in isr_by_tag.items():
        matches_by_style = isr_matches_by_tag_and_style.get(isr_tag, {})
        if not matches_by_style:
            continue

        comparison_name = (
            f"{cfg.label}_isr_comparison_cross_style"
            if isr_tag == ""
            else f"{cfg.label}_{isr_tag}_isr_comparison_cross_style"
        )
        isr_cross_dir = out / comparison_name
        model_altitude = np.asarray(edp_samples.altitude)

        batch_indices = sorted({
            m.batch_index
            for matches in matches_by_style.values()
            for m in matches
        })
        for batch_index in batch_indices:
            matches_for_batch = {
                style: next(
                    (m for m in matches if m.batch_index == batch_index),
                    None,
                )
                for style, matches in matches_by_style.items()
            }
            matches_for_batch = {
                s: m for s, m in matches_for_batch.items() if m is not None
            }
            if not matches_for_batch:
                continue

            _savefig(
                isr_comparison.plot_isr_nearest_point_comparison_cross_style(
                    matches_for_batch,
                    model_altitude,
                    isr_dataset.altitude,
                    station_name=isr_dataset.station.name,
                ),
                isr_cross_dir / "nearest_point" / f"batch{batch_index:04d}.png",
            )

        _savefig(
            isr_comparison.plot_isr_range_comparison_cross_style(
                matches_by_style,
                isr_dataset,
                cfg.start_time,
                cfg.end_time,
                model_altitude,
                percentile=cfg.isr_range_percentile,
            ),
            isr_cross_dir / "range_comparison.png",
        )

        rmse_cross_df = isr_comparison.pooled_isr_rmse_by_altitude_cross_style(
            matches_by_style,
            model_altitude,
            isr_dataset.altitude,
        )
        _save_csv(rmse_cross_df, isr_cross_dir / "pooled_rmse.csv")
        _savefig(
            isr_comparison.plot_pooled_isr_rmse_cross_style(rmse_cross_df),
            isr_cross_dir / "pooled_rmse.png",
        )

    # -- Step 9: cross-style comparison ------------------------------------
    _savefig(output.plot_style_comparison_summary(results_by_style),
             out / f"{cfg.label}_style_comparison.png")

    return PackageResult(
        output_dir=out,
        edp_samples=edp_samples,
        batches=batches,
        results_by_style=results_by_style,
        isr_matches_by_style=isr_matches_by_style,
        isr_matches_by_tag_and_style=isr_matches_by_tag_and_style,
    )
