#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assimilation_Cycle -- orchestrates one ionosphere data-assimilation cycle
by wiring together the five already-built modules
(``observation_preparation``, ``IRI_Sample_Inputs``, ``EDPSamples``,
``Parameterization``, ``Ensemble_Kalman_Engine``): pick a time window and
ROI, assemble real RO/IGS observations, draw a matching IRI2020 ensemble,
and run the observations through ``GeneralEnKFDriver`` in sequential
batches.

See ``Assimilation_Cycle_Integration_Plan.md`` (repository root) for the
full design. Per that plan's Section 3, this package imports the five
survivor modules but adds no logic to them, with one additive exception
(``Ensemble_Kalman_Engine.driver.GeneralEnKFDriver``'s ``build_ensemble``/
``build_observation_operator`` split, Section 4.5).

Heavier, cartopy/xarray/IRI2020-dependent imports (``EDPSamples``,
``Parameterization``, ``IRI_Sample_inputs``) are deferred inside the
functions that need them, matching ``Ensemble_Kalman_Engine.driver``'s own
convention, so this package stays importable (e.g. for ``metrics``/
``ensemble_io`` unit tests) without those on the path.
"""

from .cycle_config import CycleConfig, default_altitude_grid
from .ensemble_io import save_ensemble_netcdf, load_ensemble_netcdf
from .cycle_driver import CycleBatch, BatchOutcome, CycleResult, run_batch_loop, run_cycle
from . import observation_stream
from . import iri_selection
from . import ensemble_init
from . import metrics
from . import output
from . import style_sweep
from . import abel_consistency
from .diagonal_boost import apply_diagonal_boost
from .package_run import PackageResult, run_package

__all__ = [
    "CycleConfig",
    "default_altitude_grid",
    "save_ensemble_netcdf",
    "load_ensemble_netcdf",
    "CycleBatch",
    "BatchOutcome",
    "CycleResult",
    "run_batch_loop",
    "run_cycle",
    "observation_stream",
    "iri_selection",
    "ensemble_init",
    "metrics",
    "output",
    "style_sweep",
    "abel_consistency",
    "apply_diagonal_boost",
    "PackageResult",
    "run_package",
]
