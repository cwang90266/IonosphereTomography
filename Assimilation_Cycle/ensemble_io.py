#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
netCDF I/O for ``Ensemble_Kalman_Engine.EnsembleState`` (plan Section
4.10) -- this has no I/O of its own today. Used for (a) intermediate
per-batch ensemble persistence during a cycle, and (b) the precomputed-file
mode's ensemble-adjacent artifacts.

Uses ``xarray``, matching the convention already used by
``EDPSamples``/``Parameterized_EDPSamples`` netCDF I/O (plain ``netCDF4``
is used elsewhere in the repo for ``observation_preparation``'s
per-observation padded-ray schema, a different shape of problem).

A saved file is deliberately self-describing beyond the bare ``(X,
param_shape)`` needed to reconstruct an ``EnsembleState`` (plan Section 2,
item 11 / Section 4.10): ``style``/``hyper_params`` (to ``decode()`` later
without separately remembering the run's configuration) and cycle/batch
identity (time window, ROI, grid, batch index/time span) when the caller
supplies them, so a file can be matched against an external validation
source later without the original ``CycleConfig`` on hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from Ensemble_Kalman_Engine import EnsembleState

_JSON_ATTRS = ("hyper_params", "roi", "extra_attrs")


def save_ensemble_netcdf(
    ensemble: EnsembleState,
    path: str | Path,
    *,
    style: str | None = None,
    hyper_params: dict | None = None,
    cycle_start_time: str | None = None,
    cycle_end_time: str | None = None,
    roi: dict[str, Any] | None = None,
    batch_index: int | None = None,
    batch_start_time: str | None = None,
    batch_end_time: str | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Save one ``EnsembleState`` to netCDF, self-describing per the
    module docstring above. All keyword arguments beyond ``ensemble``/
    ``path`` are optional metadata -- omit what isn't known/applicable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_state, n_members = ensemble.X.shape
    ds = xr.Dataset(
        data_vars={"X": (("state", "member"), ensemble.X)},
        coords={"state": np.arange(n_state), "member": np.arange(n_members)},
        attrs={"param_shape": json.dumps(list(ensemble.param_shape))},
    )
    if style is not None:
        ds.attrs["style"] = style
    if hyper_params is not None:
        ds.attrs["hyper_params"] = json.dumps(hyper_params, default=str)
    if cycle_start_time is not None:
        ds.attrs["cycle_start_time"] = cycle_start_time
    if cycle_end_time is not None:
        ds.attrs["cycle_end_time"] = cycle_end_time
    if roi is not None:
        ds.attrs["roi"] = json.dumps(roi, default=str)
    if batch_index is not None:
        ds.attrs["batch_index"] = int(batch_index)
    if batch_start_time is not None:
        ds.attrs["batch_start_time"] = batch_start_time
    if batch_end_time is not None:
        ds.attrs["batch_end_time"] = batch_end_time
    if extra_attrs is not None:
        ds.attrs["extra_attrs"] = json.dumps(extra_attrs, default=str)

    ds.to_netcdf(path)
    return path


def load_ensemble_netcdf(path: str | Path) -> tuple[EnsembleState, dict[str, Any]]:
    """Load an ``EnsembleState`` back from netCDF.

    Returns
    -------
    ensemble : EnsembleState
    metadata : dict
        Every attr saved by ``save_ensemble_netcdf`` beyond ``X``/
        ``param_shape`` (``style``, ``hyper_params``, cycle/batch identity,
        ...), with the JSON-encoded ones (``hyper_params``, ``roi``,
        ``extra_attrs``) decoded back to dicts. Keys not present in the
        file are simply absent from this dict.
    """
    with xr.open_dataset(path) as ds:
        ds.load()
        X = np.asarray(ds["X"].values)
        param_shape = tuple(json.loads(ds.attrs["param_shape"]))
        ensemble = EnsembleState(X=X, param_shape=param_shape)

        metadata: dict[str, Any] = {}
        for key, value in ds.attrs.items():
            if key == "param_shape":
                continue
            if key in _JSON_ATTRS:
                metadata[key] = json.loads(value)
            else:
                metadata[key] = value

    return ensemble, metadata
