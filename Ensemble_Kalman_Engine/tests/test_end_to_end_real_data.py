# -*- coding: utf-8 -*-
"""
End-to-end OSSE recovery check against *real* IRI-sampled data (plan
Section 5.7's "per-style sanity cycle" + "OSSE-based recovery check",
run for real rather than on a toy operator).

Uses ``TestCode/EDPSam_Point.nc`` -- an existing, already-generated
EDPSamples file (90 heights x 1 geo column x 2000 IRI-drawn samples) --
so this needs no IRI model run, only the repo's existing test data.
Ray geometry (``podTc2_data``) is synthesized as a handful of vertical
integration paths at the sample's own (lat, lon): since this file has a
single geo column, only the *altitude* extent of each path matters to
``EDPSamples.get_observation_operator``, so a vertical path is a
legitimate way to exercise the real ray-integration code without needing
real LEO/GNSS ephemeris.

One ensemble member is held out as the OSSE truth (plan Section 5.8);
the rest seed the forecast ensemble. This is the strongest check in the
suite: real data, the real (not toy) observation operator, and a known
answer to score against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

Parameterization = pytest.importorskip(
    "Parameterization", reason="Parameterization.py and its dependencies not importable in this environment"
)
edp_samples_module = pytest.importorskip(
    "edp_samples", reason="edp_samples.py and its dependencies (pyproj, cartopy, ...) not importable"
)

from Ensemble_Kalman_Engine.observation_operator import GenericObservationOperator
from Ensemble_Kalman_Engine.ensemble_state import EnsembleState
from Ensemble_Kalman_Engine.analysis_engine import AnalysisEngine, AnalysisConfig
from Ensemble_Kalman_Engine.osse import generate_osse_observation, recovery_error
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver

_DATA_FILE = Path(__file__).resolve().parents[2] / "TestCode" / "EDPSam_Point.nc"

pytestmark = pytest.mark.skipif(
    not _DATA_FILE.exists(), reason=f"real test data not found: {_DATA_FILE}"
)


@pytest.fixture(scope="module")
def real_edp_samples():
    EDPSamples = edp_samples_module.EDPSamples
    full = EDPSamples.fromNetCDF(str(_DATA_FILE))
    # Subsample for test speed (150 of 2000 real IRI-drawn members is
    # plenty to demonstrate recovery, and keeps ANCHOR fitting fast).
    return EDPSamples.from_xarray(full.isel(sample=slice(0, 150)))


@pytest.fixture(scope="module")
def vertical_ray_geometry(real_edp_samples):
    """Six vertical integration paths (whole-column and several partial
    sub-columns) at the sample's own (lat, lon) -- enough distinct linear
    functionals of the profile to make the analysis well-posed."""
    lon, lat = 20.0, 30.0
    alt_bounds_km = [(100, 990), (100, 400), (400, 700), (700, 990), (150, 990), (100, 250)]
    LEO = np.zeros((3, len(alt_bounds_km)))
    GNSS = np.zeros((3, len(alt_bounds_km)))
    for i, (a0, a1) in enumerate(alt_bounds_km):
        GNSS[:, i] = edp_samples_module._geodetic_to_ecef(lat, lon, a0 * 1000.0) / 1000.0
        LEO[:, i] = edp_samples_module._geodetic_to_ecef(lat, lon, a1 * 1000.0) / 1000.0
    return {"LEO": LEO, "GNSS": GNSS}


_STYLE_HYPERPARAMS = {
    "PCA_1D": {"retaining_threshold": 0.999},
    "PCA_1D_10ex": {"retaining_threshold": 0.999},
    "PCA_3D_10ex": {"retaining_threshold": 0.999},
}


@pytest.mark.parametrize(
    "style", ["raw", "density_10ex", "ANCHOR", "PCA_1D", "PCA_1D_10ex", "PCA_3D_10ex"]
)
def test_osse_recovery_real_data_per_style(real_edp_samples, vertical_ray_geometry, style):
    pes = Parameterization.Parameterized_EDPSamples(
        real_edp_samples, style=style, hyper_params=_STYLE_HYPERPARAMS.get(style)
    )
    ensemble = EnsembleState.from_parameterized_edp_samples(pes)

    obs_op = GenericObservationOperator.from_edp_samples(
        real_edp_samples, pes.Parameterization, ensemble.param_shape,
        vertical_ray_geometry, num_segments=200,
    )

    # Hold out one real IRI-drawn member as the OSSE truth; the rest seed
    # the forecast ensemble (plan Section 5.8's suggested practical use).
    x_true = ensemble.X[:, 0].copy()
    forecast = EnsembleState(X=ensemble.X[:, 1:], param_shape=ensemble.param_shape)

    n_obs = obs_op.n_obs
    y_true = obs_op.forward_single(x_true)
    R = (0.05 * np.abs(y_true).mean()) ** 2 * np.eye(n_obs)   # ~5% relative obs error
    osse = generate_osse_observation(x_true, obs_op, R, rng=np.random.default_rng(0))

    config = AnalysisConfig.default_for_style(style, rng=np.random.default_rng(1))
    _, diag = AnalysisEngine().analyze(forecast, obs_op, osse.y_obs, R, config)

    prior_err = recovery_error(x_true, forecast.ensemble_mean())["rmse"]
    posterior_err = recovery_error(x_true, diag.x_mean_a)["rmse"]

    assert posterior_err < 0.8 * prior_err, (
        f"[{style}] analysis did not measurably recover the OSSE truth: "
        f"prior_rmse={prior_err}, posterior_rmse={posterior_err}"
    )


def test_general_enkf_driver_wiring_end_to_end(real_edp_samples, vertical_ray_geometry):
    """Exercises GeneralEnKFDriver itself (plan Section 5.6), not just the
    pieces it composes -- build_ensemble_and_operator, assimilate_one_cycle,
    and decode_to_edps, all against the real data used above."""
    driver = GeneralEnKFDriver(style="ANCHOR")
    ensemble, obs_op = driver.build_ensemble_and_operator(
        real_edp_samples, vertical_ray_geometry, num_segments=200
    )

    x_true = ensemble.X[:, 0].copy()
    forecast = EnsembleState(X=ensemble.X[:, 1:], param_shape=ensemble.param_shape)

    y_true = obs_op.forward_single(x_true)
    R = (0.05 * np.abs(y_true).mean()) ** 2 * np.eye(obs_op.n_obs)
    osse = generate_osse_observation(x_true, obs_op, R, rng=np.random.default_rng(0))

    result_ensemble, diag = driver.assimilate_one_cycle(forecast, obs_op, osse.y_obs, R)
    assert result_ensemble.X.shape == forecast.X.shape

    prior_err = recovery_error(x_true, forecast.ensemble_mean())["rmse"]
    posterior_err = recovery_error(x_true, diag.x_mean_a)["rmse"]
    assert posterior_err < 0.8 * prior_err

    decoded = driver.decode_to_edps(result_ensemble, obs_op)
    assert decoded.shape == (obs_op.n_height, obs_op.n_geo, result_ensemble.n_members)
    assert np.all(np.isfinite(decoded))
