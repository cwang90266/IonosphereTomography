#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``CycleConfig`` (Assimilation_Cycle_Integration_Plan.md Section 4.1):
everything needed to reproduce one assimilation cycle -- time window, ROI,
grid, style, batching, observation sources, precomputed-file paths, and
the OSSE toggle.

Design note not spelled out in the plan doc: ``IRI_Sample_Inputs`` has two
ensemble-generation methods, ``quantileSamples`` (spread specified via
range parameters, ensemble size an emergent combinatorial product -- no
direct size control) and ``randomSamples`` (spread via the same range
parameters, but ``nSample`` picks the ensemble size directly). Since the
plan's ``n_ensemble`` field implies direct control, ``iri_selection.py``
uses ``randomSamples``, not ``quantileSamples`` -- flagged here since nothing
in the plan doc named this choice explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

ObsSource = Literal["RO", "IGS", "both"]
_EARTH_RADIUS_KM = 6371.0


def default_altitude_grid() -> np.ndarray:
    """90-900 km at 10 km spacing (Final_Packaging.docx's default; 82
    levels) -- the real production standard confirmed this session,
    superseding the plan doc's original 0-900km placeholder."""
    return np.arange(90.0, 901.0, 10.0)


def default_styles() -> list[str]:
    return ["ANCHOR", "PCA_3D_10ex", "PCA_1D_10ex"]


def default_hyper_params_by_style() -> dict:
    """Removal thresholds 1e-4/1e-5 (Final_Packaging.docx) ->
    retaining_threshold = 1 - removal_threshold, same reading confirmed
    with the user for the earlier real-data style comparison. ANCHOR
    needs no hyper_params."""
    return {
        "PCA_3D_10ex": {"retaining_threshold": 1 - 1e-4},
        "PCA_1D_10ex": {"retaining_threshold": 1 - 1e-5},
    }


def default_iri_spread_kwargs() -> dict:
    """Real-window defaults, confirmed with the user 2026-09-29 after
    directly inspecting the underlying IRI2020 input files (previously
    defaulted to `{}` -- no spread at all on any index, a real bug found
    the same day: a fresh-build ensemble with this left unset is
    2000 identical IRI2020 runs, see
    `Assimilation_Cycle_Integration_Plan.md` Section 21/22).

    Each `*_sample_range` is a window of *table rows*, not a fixed real
    time span -- the two input files have different native cadences, so
    the same integer means a different real span depending on which
    index it's applied to:
      - `apf107.dat` (`f107`/`ap`) is indexed one row per *calendar day*
        (confirmed via its `yr`/`mn`/`dy` fields) -- `ap`'s value within
        each row is itself 8 genuinely 3-hourly sub-values, but the
        *window* steps in whole days. `f107_sample_range=30`/
        `ap_sample_range=30` -> +/-30 real days.
      - `ig_rz.dat` (`ig12`/`rz12`) is indexed one row per *calendar
        month* (confirmed via its own `Start_end_month` header) --
        `ig_sample_range=12`/`rz_sample_range=12` -> +/-12 real months.
        These are 12-month *smoothed* indices by construction, so even a
        wide window changes them slowly -- 12 is a real month-scale
        window, not a narrow one, despite the smaller number.
      - `hour` indexes local time-of-day directly in whole hours (no
        separate file/cadence question) -- `hour_sample_range=3` ->
        +/-3 real hours, exactly as stated.
    """
    return {
        "hour_sample_range": 3,
        "f107_sample_range": 30,
        "ap_sample_range": 30,
        "ig_sample_range": 12,
        "rz_sample_range": 12,
    }


@dataclass
class CycleConfig:
    # --- Time window and ROI (Section 4.1) ---
    start_time: str
    end_time: str
    center_lat: float
    center_lon: float
    radius_km: float
    """Observation ROI capture radius, in km -- forwarded as-is to
    ``prepare_ro_observations``/``prepare_igs_observations`` (Section 4.2),
    whose own ``radius_km`` convention this matches. Deliberately NOT
    reused for the assimilation grid's radius below -- a wide observation
    catchment (e.g. 2000 km, generous enough to catch occultations whose
    LOS passes near the ROI even when the tangent point itself is
    farther out) is a different physical quantity from how large a region
    you actually want to model/assimilate over."""

    # --- Grid (Section 4.1, 8.4): vertical grid is always an explicit
    # array (supports a non-uniform grid later); horizontal resolution is
    # an angular spacing in degrees. ---
    altitude_grid: np.ndarray = field(default_factory=default_altitude_grid)
    horizontal_resolution_deg: float = 2.5
    grid_radius_deg: float | None = None
    """Angular radius (degrees) of the ``EDPSamples`` ``geo_type="Regional"``
    grid (Section 4.4) -- ``EDPSamples.genRegionalArea``'s own native unit,
    intentionally not km, so no earth-radius conversion constant needs to
    live here. Distinct from ``radius_km`` above (see its docstring); a
    real cycle should set this explicitly, sized to the region you
    actually want to assimilate over (typically much smaller than the
    observation ROI's capture radius). Defaults to ``radius_km``'s bare
    numeric value (i.e. re-read as degrees) only for backward-compatible
    convenience when both happen to be small test-grid numbers -- not a
    real unit conversion, don't rely on this default for a real cycle."""
    grid_margin_km: float = 0.0
    """Extra radius (km) added on top of ``grid_radius_deg`` when building
    the assimilation grid (see ``effective_grid_radius_deg`` below),
    without changing ``grid_radius_deg`` itself or the observation
    capture radius (``radius_km``). Motivated by a real finding: RO
    occultations whose line of sight stays close to the grid boundary fit
    noticeably worse than ones that reach near the center, plausibly
    because the state has less room/support to adjust near the edge
    (mesh-boundary/nearest-neighbor effects in ray integration). Default
    ``0.0`` -- no margin, unchanged behavior."""

    # --- Parameterization (Section 4.1) --- single-style fields, used by
    # cycle_driver.run_cycle/style_sweep.run_style_sweep directly. A
    # packaged run (package_run.py) instead uses `styles`/
    # `hyper_params_by_style` below to run + compare several at once. ---
    style: str = "density_10ex"
    hyper_params: dict | None = None

    label: str = ""
    """Assimilation cycle label (Final_Packaging.docx) -- used in output
    filenames (e.g. the saved IRI_Sample_Inputs pickle) so artifacts from
    different cycles in the same output_dir don't collide."""
    styles: list[str] = field(default_factory=default_styles)
    """Styles run and cross-compared by a packaged run (`package_run.py`)
    -- distinct from the single `style` field above, which existing
    single-style callers (`cycle_driver.run_cycle`, tests) keep using
    unchanged."""
    hyper_params_by_style: dict = field(default_factory=default_hyper_params_by_style)
    """Per-style hyper_params for `styles` above (keyed by style name;
    a style absent here gets `hyper_params=None`) -- the packaged run's
    analogue of `style_sweep.run_style_sweep`'s own parameter of the same
    name."""

    # --- Ensemble size / IRI spread (Section 4.3-4.4) ---
    n_ensemble: int = 2000
    iri_spread_kwargs: dict = field(default_factory=default_iri_spread_kwargs)
    """Forwarded to ``IRI_Sample_Inputs.randomSamples`` as
    ``hour_sample_range``/``f107_sample_range``/``ap_sample_range``/
    ``ig_sample_range``/``rz_sample_range`` (all optional; a key left out
    of an explicit override means no spread on that specific index, not a
    fallback to the default below -- see ``IRI_Sample_Inputs``). Default
    is ``default_iri_spread_kwargs()`` (+/-3 hours, +/-30 days for
    `f107`/`ap`, +/-12 months for `ig12`/`rz12`) -- see that function's
    docstring for why those numbers, and note they're table-row windows
    at two different native cadences, not a single uniform unit. Only
    matters for a *fresh* IRI2020 build (``edp_samples_path`` unset);
    has no effect in precomputed-``edp_samples_path`` mode, since the
    real ensemble there comes from the cached file, not a fresh draw."""

    # --- Batching and observation error (Section 4.2, 8.1, 8.2) ---
    batch_size: int = 200
    """Pure compute-chunking parameter (Section 8, decision 1) -- there is
    no state-transition model in this filter, so batch boundaries have no
    physical meaning. Entry-count based; used by `observation_stream.batch_entries`
    when `max_tec_per_batch` is not the active policy (see below)."""
    max_tec_per_batch: int | None = None
    """TEC/ray-count batching cap (Final_Packaging.docx default 300),
    replacing `batch_size`'s entry-count semantics for a packaged run:
    entries accumulate into the current batch until adding the next one
    would exceed this many total rays, then a new batch starts. An entry
    larger than the cap on its own still gets its own batch (never split
    -- keeps "one entry = one physical occultation/arc" intact for the
    per-entry TEC/EDP comparison plots). `None` (default) keeps
    `batch_size`'s entry-count behavior; set this to switch policies."""
    min_tangent_alt_km: float | None = None
    """RO-only per-ray filter (Final_Packaging.docx): rays with
    `tangent_alt_km` below this are dropped before batching; an RO entry
    with no rays left is dropped entirely. IGS entries are unaffected.
    `None` (default) disables the filter."""
    obs_sigma: float = 1.0
    """Constant observation-error standard deviation (TECU), same for RO
    and IGS, observations independent (Section 8, decision 2)."""

    # --- Observation source selection (Section 4.15) ---
    obs_sources: ObsSource = "both"
    ro_kwargs: dict = field(default_factory=dict)
    """Source-specific pass-through to ``prepare_ro_observations`` (e.g.
    ``podtc_dir``, ``roi_mode``, ``alt_limit_km``) -- kept out of
    ``CycleConfig``'s own fields since RO/IGS take mostly-disjoint
    parameter sets; only the cycle-level concepts (time window, ROI,
    source selection) are promoted to top-level fields."""
    igs_kwargs: dict = field(default_factory=dict)
    """Source-specific pass-through to ``prepare_igs_observations`` (e.g.
    ``stations``, ``cache_dir``, ``local_obs_by_station``)."""

    # --- Precomputed-file mode (Section 4.9): None means compute fresh;
    # a path means load from there instead. The matching *_output_path
    # (below) says where to save a freshly computed result, if anywhere. ---
    ro_observations_path: str | Path | None = None
    igs_observations_path: str | Path | None = None
    iri_sample_inputs_path: str | Path | None = None
    edp_samples_path: str | Path | None = None
    parameterized_edp_samples_path: str | Path | None = None

    ro_observations_output_path: str | Path | None = None
    igs_observations_output_path: str | Path | None = None
    iri_sample_inputs_output_path: str | Path | None = None
    edp_samples_output_path: str | Path | None = None
    parameterized_edp_samples_output_path: str | Path | None = None

    # --- Intermediate ensemble persistence (Section 4.10) ---
    save_intermediate_ensembles: bool = False
    ensemble_output_dir: str | Path | None = None

    # --- OSSE mode (Section 4.11) ---
    osse_mode: bool = False
    osse_rng_seed: int | None = None

    # --- Inter-batch covariance inflation (opt-in; Section 8.3's
    # "persistence, no inflation" default is unchanged unless this is set) ---
    inter_batch_inflation_factor: float | None = None
    """Multiplicative ensemble-spread inflation applied to the analysis
    ensemble at the end of every batch (via
    ``Ensemble_Kalman_Engine.perturbation.SelectiveInflation``, the same
    hook the analysis engine already exposes -- see ``cycle_driver.py``),
    before it's carried forward as the next batch's prior. Motivated by a
    real finding: pure persistence (the original Section 8.3 default) may
    let ensemble spread shrink batch over batch with nothing replacing it,
    progressively limiting the filter's ability to respond to new/
    boundary observations. `None` (default) keeps pure persistence, no
    behavior change; `1.0` is also a no-op (explicit but pointless);
    values `> 1.0` inflate, e.g. `1.05` re-inflates spread by 5% after
    every batch."""

    # --- Diagonal covariance boosting (opt-in; user-proposed 2026-09-27) ---
    diagonal_boost_amplitude: float | None = None
    """Adds a smoothed, per-point-std-scaled random field to the raw
    IRI2020 ensemble (see ``diagonal_boost.py``), applied once at
    ensemble-construction time, *before* any style's parameterization is
    fit -- manufactures less-correlated spread directions the
    climatological draw's own low effective rank otherwise lacks.
    `None`/`0` (default): no change, the original ensemble is used as-is.
    A value like `0.3` injects, at every (height, geo) point, independent
    spread with standard deviation 30% of that point's own ensemble std
    (after vertical/horizontal smoothing per
    ``diagonal_boost_vertical_scale_km``/``diagonal_boost_horizontal_scale_km``
    -- purely white/uncorrelated noise would be both unphysical and
    numerically risky for the observation operator's ray integration).
    Real-data held-out sweep found this is only safe with `log_space=True`
    for `log10`-based styles (`density_10ex`/`PCA_*_10ex`) -- see
    `diagonal_boost_log_space` and the README."""
    diagonal_boost_log_space: bool = False
    """`False` (default): perturb linear density directly (validated to
    genuinely help for the `raw` style, real held-out RMSE improvement
    around amplitude~0.2 with no pathology). `True`: perturb
    `log10(density)` instead (see `diagonal_boost.py`) -- required for
    `density_10ex`/`PCA_*_10ex`, since those styles fit `log10(density)`
    and linear-space perturbation's positivity clip creates an 8-12
    order-of-magnitude discontinuity in the space they actually fit,
    which caused catastrophic/non-convergent real-data results. Has no
    effect when `diagonal_boost_amplitude` is unset."""
    diagonal_boost_vertical_scale_km: float = 30.0
    """Gaussian smoothing length (km) applied to the raw noise field along
    the altitude coordinate before it's added -- only used when
    ``diagonal_boost_amplitude`` is set."""
    diagonal_boost_horizontal_scale_km: float = 200.0
    """Gaussian smoothing length (km, great-circle) applied to the raw
    noise field over the horizontal point cloud before it's added -- only
    used when ``diagonal_boost_amplitude`` is set. Works directly on the
    unstructured/mesh-based horizontal grid (no regular-grid assumption)."""
    diagonal_boost_rng_seed: int | None = None
    """Seeds the random field draw for reproducible comparisons. `None`
    (default): a fresh unseeded generator every call -- fine for a single
    production run, but any *comparison* across boost settings needs this
    set and held fixed, the same standing lesson as
    ``AnalysisConfig.rng`` elsewhere in this project."""

    # --- Output (Section 4.7/4.14) ---
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.altitude_grid = np.asarray(self.altitude_grid, dtype=float)
        if self.altitude_grid.ndim != 1:
            raise ValueError("altitude_grid must be 1-D")
        if self.obs_sources not in ("RO", "IGS", "both"):
            raise ValueError(f"obs_sources must be 'RO', 'IGS', or 'both', got {self.obs_sources!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_tec_per_batch is not None and self.max_tec_per_batch < 1:
            raise ValueError("max_tec_per_batch must be >= 1")
        if self.diagonal_boost_amplitude is not None and self.diagonal_boost_amplitude < 0:
            raise ValueError("diagonal_boost_amplitude must be >= 0")
        if self.diagonal_boost_vertical_scale_km <= 0:
            raise ValueError("diagonal_boost_vertical_scale_km must be > 0")
        if self.diagonal_boost_horizontal_scale_km <= 0:
            raise ValueError("diagonal_boost_horizontal_scale_km must be > 0")

    @property
    def resolved_grid_radius_deg(self) -> float:
        """``grid_radius_deg`` if set, else ``radius_km``'s bare number
        reinterpreted as degrees (see ``grid_radius_deg``'s docstring --
        convenience fallback only, not a unit conversion)."""
        return self.grid_radius_deg if self.grid_radius_deg is not None else self.radius_km

    @property
    def effective_grid_radius_deg(self) -> float:
        """``resolved_grid_radius_deg`` plus ``grid_margin_km`` (converted
        to degrees) -- what ``ensemble_init.py`` actually builds the grid
        at. Equal to ``resolved_grid_radius_deg`` when ``grid_margin_km``
        is 0 (the default)."""
        margin_deg = np.degrees(self.grid_margin_km / _EARTH_RADIUS_KM)
        return self.resolved_grid_radius_deg + margin_deg

    @property
    def center_time_str(self) -> str:
        """Midpoint of [start_time, end_time), for IRI driving-index
        selection (Section 4.3) and the ``EDPSamples``/``Parameterized_EDPSamples``
        ``DateTime`` argument."""
        import pandas as pd

        t0 = pd.Timestamp(self.start_time)
        t1 = pd.Timestamp(self.end_time)
        mid = t0 + (t1 - t0) / 2
        return mid.isoformat()
