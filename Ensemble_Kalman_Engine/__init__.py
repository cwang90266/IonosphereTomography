#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble_Kalman_Engine -- a general-purpose Ensemble Kalman Filter module
for parameterized ionospheric tomography.

See ``General_EnKF_Implementation_Plan.md`` (repository root) for the full
design. In short: the state representation (raw grid, log10, ANCHOR, PCA,
...) is a pluggable ``Parameterization.EDP_Parameterization`` style, not
something this engine hardcodes; the analysis step supports both a
one-shot linear update and an iterated (Gauss-Newton) update for strongly
nonlinear parameterizations, with an optional nearest-analog centering
strategy on top of either.

Per the plan's Section 5.0 code-reuse policy, this package's analysis
math is a from-scratch replication of
``Ionosphere_Tomography_Inverter/enkf_update.py`` (no import dependency on
that package), while ``Parameterization.py`` and ``EDPSamples`` remain
imported as shared, general-purpose infrastructure.
"""

from .kalman_core import (
    ensemble_anomalies,
    kalman_gain,
    apply_inflation,
    stochastic_update,
    ensrf_update,
    AnalysisResult,
)
from .perturbation import (
    Perturbation,
    GaussianDither,
    SelectiveInflation,
    BoundsClamp,
    CustomPerturbation,
    PerturbationChain,
)
from .ensemble_state import EnsembleState, flatten_param_vec, unflatten_to_param_shape
from .observation_operator import GenericObservationOperator
from .jacobian_utils import compose_observation_jacobian
from .analysis_engine import AnalysisEngine, AnalysisConfig, AnalysisDiagnostics
from .osse import generate_osse_observation, recovery_error, OSSEObservation
from .driver import GeneralEnKFDriver

__all__ = [
    "ensemble_anomalies",
    "kalman_gain",
    "apply_inflation",
    "stochastic_update",
    "ensrf_update",
    "AnalysisResult",
    "Perturbation",
    "GaussianDither",
    "SelectiveInflation",
    "BoundsClamp",
    "CustomPerturbation",
    "PerturbationChain",
    "EnsembleState",
    "flatten_param_vec",
    "unflatten_to_param_shape",
    "GenericObservationOperator",
    "compose_observation_jacobian",
    "AnalysisEngine",
    "AnalysisConfig",
    "AnalysisDiagnostics",
    "generate_osse_observation",
    "recovery_error",
    "OSSEObservation",
    "GeneralEnKFDriver",
]
