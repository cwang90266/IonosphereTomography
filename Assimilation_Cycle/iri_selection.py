#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRI driving-index selection (plan Section 4.3) and its precomputed-file
mode (4.9).

Design note (also recorded in ``cycle_config.py``): ``IRI_Sample_Inputs``
has two ensemble-generation methods --
``quantileSamples`` (spread via range parameters, ensemble size an
emergent combinatorial product, no direct size control) and
``randomSamples`` (same spread parameters, but ``nSample`` picks the
ensemble size directly). This module uses ``randomSamples`` so
``cfg.n_ensemble`` has a direct effect, matching how the plan doc
describes ``n_ensemble`` as a configured cycle parameter.

Deferred import of ``IRI_Sample_inputs`` (the module is a top-level import
only once ``IRI_Sample_Inputs/`` is on ``sys.path`` -- see the package
``__init__.py`` docstring) so the rest of ``Assimilation_Cycle`` stays
importable without it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cycle_config import CycleConfig


def _strip_pkl_suffix(path: str | Path) -> str:
    """``IRI_Sample_Inputs.save_to_file``/``fromPickle`` append/expect a
    literal ``.pkl`` suffix themselves -- accept a path with or without
    it already attached."""
    s = str(path)
    return s[: -len(".pkl")] if s.endswith(".pkl") else s


def load_or_build_iri_sample_inputs(cfg: CycleConfig):
    """Load a precomputed ``IRI_Sample_Inputs`` (plan Section 4.9), or
    build one fresh for the cycle's center time and optionally save it."""
    from IRI_Sample_inputs import IRI_Sample_Inputs

    if cfg.iri_sample_inputs_path is not None:
        return IRI_Sample_Inputs.fromPickle(_strip_pkl_suffix(cfg.iri_sample_inputs_path))

    # apf107.dat/ig_rz.dat are fetched-artifact files (IRI_Sample_Inputs
    # downloads/caches them, not source data shipped with the repo) --
    # keep them under this cycle's own output directory rather than
    # wherever the process's cwd happens to be (separating source from
    # execution artifacts). Falls back to the original cwd-relative
    # behavior when no output_dir is set (e.g. a bare cycle_driver.run_cycle/
    # style_sweep call with no packaged-run output directory).
    data_dir = str(cfg.output_dir) if cfg.output_dir is not None else None
    iri = IRI_Sample_Inputs(cfg.center_time_str, data_dir=data_dir)
    if cfg.iri_sample_inputs_output_path is not None:
        Path(_strip_pkl_suffix(cfg.iri_sample_inputs_output_path)).parent.mkdir(parents=True, exist_ok=True)
        iri.save_to_file(_strip_pkl_suffix(cfg.iri_sample_inputs_output_path))
    return iri


def sampling_parameters_for_cycle(cfg: CycleConfig) -> pd.DataFrame:
    """
    Produce the ``sampling_parameters`` DataFrame ``EDPSamples.__init__``
    expects (plan Section 4.3/4.4): ``cfg.n_ensemble`` random draws around
    the cycle's driving indices, spread controlled by
    ``cfg.iri_spread_kwargs`` (``hour_sample_range``/``f107_sample_range``/
    ``ap_sample_range``/``ig_sample_range``/``rz_sample_range``, all
    optional).
    """
    iri = load_or_build_iri_sample_inputs(cfg)
    return iri.randomSamples(nSample=cfg.n_ensemble, **cfg.iri_spread_kwargs)
