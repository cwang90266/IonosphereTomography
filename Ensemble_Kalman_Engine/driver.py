#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestration driver (plan Section 5.6): plays the same role
``Ionophy_Tomography_Inverter_EnKF.py`` plays for the old, ANCHOR-only
pipeline, but parameterized by ``style`` -- the same assimilation loop
runs unchanged for ``raw``, ``density_10ex``, ``ANCHOR``, or any ``PCA_*``
style (plan Section 4).

This module is the wiring layer only: it has no analysis math of its own
(that's ``analysis_engine.py``/``kalman_core.py``), and it imports
``Parameterization``/``EDPSamples`` as the shared infrastructure they are
(plan Section 5.0) -- deferred imports inside ``GeneralEnKFDriver`` so the
rest of this package stays importable without those (heavier,
cartopy/xarray-dependent) modules on the path.

Status: exercised end-to-end in ``tests/test_end_to_end_real_data.py``
against a real, existing EDPSamples file (``TestCode/EDPSam_Point.nc``,
IRI-drawn) with a synthesized (vertical-path) ray geometry -- real orbit
ephemeris was not available in this session, so that geometry is a stand-in
for genuine LEO/GNSS occultation paths, not a claim that RO-realistic
geometry has been tested. Swap in a real ``podTc2_data`` dict (from
``observation_preparation/``) to run this against actual occultation
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis_engine import AnalysisEngine, AnalysisConfig
from .ensemble_state import EnsembleState
from .observation_operator import GenericObservationOperator
from .perturbation import Perturbation


@dataclass
class GeneralEnKFDriver:
    """
    Parameters
    ----------
    style : str
        Any registered ``Parameterization.Parameterization_Style``
        ('raw', 'density_10ex', 'ANCHOR', 'PCA_1D', 'PCA_3D', ...).
    hyper_params : dict, optional
        Passed through to ``EDP_Parameterization``/``Parameterized_EDPSamples``
        (e.g. a pre-fit PCA basis, or ``retaining_threshold`` to fit one).
    config : AnalysisConfig, optional
        Defaults to ``AnalysisConfig.default_for_style(style)`` if omitted
        (confirmed per-style linearization defaults, plan Section 6.2/6.4).
    """

    style: str
    hyper_params: dict | None = None
    config: AnalysisConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = AnalysisConfig.default_for_style(self.style)

    def build_ensemble_and_operator(
        self, edp_samples, podTc2_data: dict, num_segments: int = 1000
    ) -> tuple[EnsembleState, GenericObservationOperator]:
        """
        Wire a real ``EDPSamples`` draw + ray geometry into an
        ``EnsembleState`` (5.2) and a ``GenericObservationOperator`` (5.4),
        for the configured ``style``.
        """
        from Parameterization import Parameterized_EDPSamples   # shared infra, per 5.0

        pes = Parameterized_EDPSamples(edp_samples, style=self.style, hyper_params=self.hyper_params)
        ensemble = EnsembleState.from_parameterized_edp_samples(pes)

        obs_operator = GenericObservationOperator.from_edp_samples(
            edp_samples,
            pes.Parameterization,
            param_shape=ensemble.param_shape,
            podTc2_data=podTc2_data,
            num_segments=num_segments,
        )
        return ensemble, obs_operator

    def assimilate_one_cycle(
        self,
        ensemble: EnsembleState,
        obs_operator: GenericObservationOperator,
        y_obs: np.ndarray,
        R: np.ndarray,
        perturbation: Perturbation | None = None,
    ):
        """
        Run one assimilation cycle (plan Section 4's full diagram): the
        analysis engine handles the linear/iterated x mean/nearest_analog
        choice per ``self.config``; ``perturbation`` (5.3) is applied
        after, if given.

        Returns
        -------
        ensemble_a : EnsembleState
            The analysis ensemble -- the forecast seed for the next cycle.
        diagnostics : AnalysisDiagnostics
        """
        config = self.config
        if perturbation is not None:
            from dataclasses import replace
            config = replace(config, perturbation=perturbation)
        return AnalysisEngine().analyze(ensemble, obs_operator, y_obs, R, config)

    def decode_to_edps(self, ensemble: EnsembleState, obs_operator: GenericObservationOperator) -> np.ndarray:
        """
        Decode the (filter-space) ensemble back to physical electron
        density, shape ``(n_height, n_geo, n_members)`` -- for diagnostics,
        plotting, or seeding the next cycle's real-space forward model.
        """
        return obs_operator.decode(ensemble.to_param_shape())
