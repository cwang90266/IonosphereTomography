#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The analysis engine (plan Section 4 / Section 6): two orthogonal, composable
choices, not a single mode switch.

1. **Linearization strategy** -- ``"linear"`` (one-shot, plan Section 1) vs.
   ``"iterated"`` (Remedy A, Section 6.2): a damped Gauss-Newton loop that
   re-linearizes the observation operator via
   ``EDP_Parameterization.get_Jacobian`` at each step, holding the
   ensemble-estimated ``Q ~ S S^T`` fixed throughout. Confirmed default for
   every nonlinear style.
2. **Centering strategy** -- ``"mean"`` vs. ``"nearest_analog"`` (Remedy B,
   Section 6.3): which point the analysis *mean* correction is anchored to.
   Confirmed default across the board, orthogonal to the choice above.

Design note on how the two combine, spelled out here because the source
deck leaves the exact mechanics of "re-select the nearest analog every
iteration" (plan Section 6.4's confirmed policy) as a heuristic rather
than a fully specified algorithm: at each Gauss-Newton step the engine
compares the freshly-stepped point against the best-matching *original*
ensemble member (found once, cheaply, from the already-computed forecast
ensemble ``Y_f`` -- no repeated full-ensemble nonlinear evaluation) and
keeps whichever has the smaller observation residual as the center for the
next iteration's re-linearization. This is a specific, reviewable
implementation of the confirmed policy, not a re-litigation of it -- flag
if a different mechanics is intended.

Centering only ever changes which point the analysis *mean* is anchored
to; the ensemble *spread* (``S_a``) is always built from the standard
mean-centered ``S``/linearized-``Y`` pair, so an unlucky centering choice
cannot bias the reported covariance -- only the mean correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ensemble_state import EnsembleState
from .kalman_core import ensemble_anomalies, kalman_gain, apply_inflation
from .perturbation import Perturbation


Linearization = str   # "linear" | "iterated"
Centering = str       # "mean" | "nearest_analog"

# Confirmed styles/defaults (plan Sections 6.2, 6.4): `iterated` for every
# nonlinear parameterization, `linear` only for `raw` (genuinely linear
# forward map); `nearest_analog` centering everywhere.
DEFAULT_LINEARIZATION_BY_STYLE: dict[str, Linearization] = {
    "raw": "linear",
    "density_10ex": "iterated",
    "ANCHOR": "iterated",
    "PCA_1D": "iterated",
    "PCA_1D_10ex": "iterated",
    "PCA_3D": "iterated",
    "PCA_3D_10ex": "iterated",
}


@dataclass
class AnalysisConfig:
    """
    Confirmed defaults (see the "Confirmed" notes in the plan doc, Section
    6.2/6.4): ``alpha0=1.0`` with backtracking, a 5-10 iteration cap
    (parameterizations in use are smooth), relative-tolerance convergence,
    ``nearest_analog`` centering with the required escape hatch to fall
    back to ``mean`` if it proves too costly.
    """

    linearization: Linearization = "iterated"
    centering: Centering = "nearest_analog"
    max_iterations: int = 8                 # within the confirmed 5-10 range
    alpha0: float = 1.0
    alpha_backtrack_factor: float = 0.5
    max_backtracks: int = 5
    convergence_tol: float = 1e-3
    localization: np.ndarray | None = None
    inflation: float | np.ndarray | None = None
    perturbation: Perturbation | None = None
    rng: np.random.Generator | None = None

    @classmethod
    def default_for_style(cls, style: str, **overrides) -> "AnalysisConfig":
        """Confirmed per-style default linearization; centering stays
        ``nearest_analog`` (the plan's across-the-board default) unless
        overridden."""
        linearization = DEFAULT_LINEARIZATION_BY_STYLE.get(style, "iterated")
        return cls(linearization=linearization, **overrides)


@dataclass
class AnalysisDiagnostics:
    n_iterations: int
    converged: bool
    residual_norm_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    center_source_history: list[str] = field(default_factory=list)   # "mean" | "ensemble_member" | "gauss_newton"
    K_mean: np.ndarray | None = None
    x_mean_a: np.ndarray | None = None
    """The analysis mean from the (deterministic) gain-based correction,
    *before* the stochastic perturbed-obs ensemble is built. The returned
    ensemble's own ``ensemble_mean()`` will differ from this by
    O(1/sqrt(n_members)) sampling noise from the observation
    perturbations (expected perturbed-obs EnKF behaviour, not bias) --
    use this field when a noise-free point estimate is wanted."""


def _residual_norm(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_obs - y_pred))


def _best_ensemble_analog(y_obs: np.ndarray, Y_f: np.ndarray) -> tuple[int, float]:
    dist = np.linalg.norm(y_obs[:, np.newaxis] - Y_f, axis=0)
    i_star = int(np.argmin(dist))
    return i_star, float(dist[i_star])


class AnalysisEngine:
    """
    Stateless engine: call ``analyze`` with an ``EnsembleState``, an
    observation operator (``GenericObservationOperator`` duck-typed:
    needs ``forward_ensemble``, ``forward_single``, ``linearized``), the
    observation, ``R``, and a config. Returns a new ``EnsembleState`` plus
    diagnostics.
    """

    def analyze(
        self,
        ensemble: EnsembleState,
        obs_operator,
        y_obs: np.ndarray,
        R: np.ndarray,
        config: AnalysisConfig | None = None,
    ) -> tuple[EnsembleState, AnalysisDiagnostics]:
        if config is None:
            config = AnalysisConfig()
        rng = config.rng or np.random.default_rng()

        X_f = apply_inflation(ensemble.X, config.inflation)
        n_state, n_members = X_f.shape

        Y_f = obs_operator.forward_ensemble(X_f)          # (n_obs, n_members), full nonlinear, once
        x_mean, S = ensemble_anomalies(X_f)                # Q ~ S S^T, fixed for the whole call

        diag = AnalysisDiagnostics(n_iterations=0, converged=False)

        # -- initial center -----------------------------------------------
        if config.centering == "nearest_analog":
            i_star, _ = _best_ensemble_analog(y_obs, Y_f)
            x_i = X_f[:, i_star].copy()
            y_i = Y_f[:, i_star].copy()
            diag.center_source_history.append("ensemble_member")
        elif config.centering == "mean":
            x_i = x_mean.copy()
            y_i = Y_f.mean(axis=1)
            diag.center_source_history.append("mean")
        else:
            raise ValueError(f"Unknown centering strategy: {config.centering!r}")

        diag.residual_norm_history.append(_residual_norm(y_obs, y_i))

        # -- linearization loop ---------------------------------------------
        if config.linearization == "linear":
            Y_mean, Y_anom = ensemble_anomalies(Y_f)
            K_final = kalman_gain(S, Y_anom, R, localization=config.localization)[0]
            x_final = x_i + K_final @ (y_obs - y_i)
            diag.n_iterations = 1
            diag.converged = True

        elif config.linearization == "iterated":
            # NOTE on a correction to the deck's slide-6 formula: as
            # literally transcribed, `P_{i+1} = P_i + G_i(ybar - W.E(P_i))`
            # is only the correct MAP/Kalman step on the *first* iteration
            # (P_0 == the prior mean). Iterated as-is past that, it loses
            # the prior's pull once P_i wanders from the prior mean, and
            # degenerates into an unregularized nonlinear least-squares
            # fit to (possibly noisy/underdetermined) data -- confirmed
            # empirically: a toy-problem unit test using this literal
            # formula converged to a *worse* truth-recovery error than
            # the one-shot `linear` update, overfitting observation noise.
            # The standard fix (Gauss-Newton for a MAP estimate with a
            # Gaussian prior, e.g. Bell & Cathey 1993) keeps every
            # correction anchored to the *original* prior point `x_p`,
            # adding back the linearized prior-consistency term
            # `W_hat_i (x_i - x_p)`:
            #
            #   x_{i+1} = x_p + K_i [ (y_obs - y(x_i)) + W_hat_i (x_i - x_p) ]
            #
            # which reduces to the deck's own formula when x_i == x_p
            # (iteration 0) and correctly balances prior vs. data at every
            # subsequent step. `x_p` is fixed for the whole loop; `x_i` is
            # the moving linearization point.
            x_p, y_p = x_i, y_i
            x_i_current, y_i_current = x_i, y_i
            W_hat = None
            for it in range(config.max_iterations):
                W_hat = obs_operator.linearized(x_i_current)
                Y_lin = W_hat @ S                            # linearized ensemble anomalies
                K_i, _ = kalman_gain(S, Y_lin, R, localization=config.localization)

                pseudo_residual = (y_obs - y_i_current) + W_hat @ (x_i_current - x_p)

                alpha = config.alpha0
                accepted = False
                for _ in range(config.max_backtracks + 1):
                    x_candidate = x_p + alpha * (K_i @ pseudo_residual)
                    y_candidate = obs_operator.forward_single(x_candidate)
                    if _residual_norm(y_obs, y_candidate) <= diag.residual_norm_history[-1]:
                        accepted = True
                        break
                    alpha *= config.alpha_backtrack_factor
                if not accepted:
                    # No damped step improved the residual; stop iterating
                    # at the best point found so far rather than diverge.
                    break

                diag.alpha_history.append(alpha)

                # Confirmed policy (6.4): re-select the nearest analog at
                # every iteration -- compare the fresh Gauss-Newton point
                # against the best original ensemble member and keep
                # whichever is closer to the observation as the *next
                # linearization point* (x_p, the prior anchor, does not
                # move -- see note above).
                if config.centering == "nearest_analog":
                    i_star, dist_star = _best_ensemble_analog(y_obs, Y_f)
                    if dist_star < _residual_norm(y_obs, y_candidate):
                        x_candidate = X_f[:, i_star].copy()
                        y_candidate = Y_f[:, i_star].copy()
                        diag.center_source_history.append("ensemble_member")
                    else:
                        diag.center_source_history.append("gauss_newton")
                else:
                    diag.center_source_history.append("gauss_newton")

                step_size = np.linalg.norm(x_candidate - x_i_current)
                scale = max(np.linalg.norm(x_i_current), 1e-12)
                x_i_current, y_i_current = x_candidate, y_candidate
                diag.residual_norm_history.append(_residual_norm(y_obs, y_i_current))
                diag.n_iterations = it + 1

                if step_size / scale < config.convergence_tol:
                    diag.converged = True
                    break

            x_final = x_i_current
            if W_hat is None:
                W_hat = obs_operator.linearized(x_final)
            Y_lin = W_hat @ S
            K_final, _ = kalman_gain(S, Y_lin, R, localization=config.localization)

        else:
            raise ValueError(f"Unknown linearization strategy: {config.linearization!r}")

        diag.K_mean = K_final
        diag.x_mean_a = x_final

        # -- build the analysis ensemble (plan Section 1's perturbed-obs
        #    formula, spread always from the standard mean-centered S/Y,
        #    mean anchored at x_final regardless of centering choice) ------
        n_obs = y_obs.shape[0]
        try:
            import scipy.linalg as la
            L_chol = la.cholesky(R, lower=True)
        except Exception:
            L_chol = np.linalg.cholesky(R + 1e-8 * np.eye(n_obs))
        eps = L_chol @ rng.standard_normal((n_obs, n_members))

        if config.linearization == "linear":
            Y_centered = Y_f - Y_f.mean(axis=1, keepdims=True)
        else:
            Y_centered = W_hat @ (X_f - x_mean[:, np.newaxis])

        S_a = (X_f - x_mean[:, np.newaxis]) - K_final @ (Y_centered + eps)
        X_a = x_final[:, np.newaxis] + S_a

        if config.perturbation is not None:
            X_a = config.perturbation(X_a, rng=rng)

        return EnsembleState(X=X_a, param_shape=ensemble.param_shape), diag
