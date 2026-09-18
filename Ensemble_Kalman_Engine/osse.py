#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSSE (Observation System Simulation Experiment) support (plan Section 5.8).

A small, separate utility living alongside the engine -- not a hook inside
the analysis engine's control flow. It reuses exactly the interface the
engine already requires for real assimilation (an observation operator +
``R``), applied once to a chosen truth state instead of once per ensemble
member:

    y_obs = observation_operator.forward_single(x_true) + noise(R)

The resulting ``(y_obs, R)`` is handed to ``AnalysisEngine.analyze`` through
its normal entry point -- the engine has no "truth-known" code path, so an
OSSE run is scored against a known answer while exercising the exact same
assimilation code a real run would use.

GNSS/LEO ray-geometry construction is *not* this module's concern -- it
stays wherever it already lives (``EDPSamples.get_observation_operator``,
``observation_preparation/``). This module only needs an already-built
observation operator instance, real-geometry-based or a toy one for early
engine tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la


@dataclass
class OSSEObservation:
    """The synthetic observation plus everything needed to score recovery."""

    y_obs: np.ndarray
    x_true: np.ndarray
    y_true_noiseless: np.ndarray
    R: np.ndarray


def generate_osse_observation(
    x_true: np.ndarray,
    obs_operator,
    R: np.ndarray,
    rng: np.random.Generator | None = None,
) -> OSSEObservation:
    """
    Synthesize an observation from a known truth state.

    Parameters
    ----------
    x_true : ndarray, shape (n_state,)
        The truth state, in the same filter-space layout as the ensemble
        (e.g. one held-out draw from the same climatology used to seed
        the forecast, or a hand-specified profile for a fully controlled
        test).
    obs_operator : GenericObservationOperator (or any duck-typed object
        with a ``forward_single(x) -> y`` method)
        The *same* operator instance the engine will use for the real
        assimilation call.
    R : ndarray, shape (n_obs, n_obs)
        Observation error covariance -- noise is drawn consistent with it.
    rng : numpy.random.Generator, optional

    Returns
    -------
    OSSEObservation
    """
    if rng is None:
        rng = np.random.default_rng()

    y_true = obs_operator.forward_single(x_true)
    n_obs = y_true.shape[0]
    try:
        L_chol = la.cholesky(R, lower=True)
    except la.LinAlgError:
        L_chol = la.cholesky(R + 1e-8 * np.eye(n_obs), lower=True)
    noise = L_chol @ rng.standard_normal(n_obs)

    return OSSEObservation(
        y_obs=y_true + noise, x_true=x_true, y_true_noiseless=y_true, R=R
    )


def recovery_error(x_true: np.ndarray, x_analysis_mean: np.ndarray) -> dict:
    """
    Basic OSSE scoring: how far the analysis mean is from the known truth,
    in absolute and (state-norm-)relative terms. Callers with a covariance
    estimate on hand (e.g. ``S_a S_a^T`` from the returned ensemble) should
    additionally check that the truth error is consistent with the
    reported spread -- that check is problem-specific and left to the
    caller/test, not baked in here.
    """
    err = x_analysis_mean - x_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "relative_l2_error": float(np.linalg.norm(err) / max(np.linalg.norm(x_true), 1e-12)),
    }
