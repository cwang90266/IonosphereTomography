#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISR-derived PCA basis (``ISR_Integration_Plan.md`` Section 4.1): fits a
1-D, log10-space PCA basis from real ISR-measured electron-density
profiles collected over an extended period, instead of an IRI2020
climatological ensemble -- for use with the ``PCA_1D_10ex_ISR``
parameterization style (``Parameterization.Parameterized_EDPSamples``
already accepts a pre-fit ``PCA``/``PCA_mean`` for that style; this module
is what builds one from ISR data).

**Why only 1-D:** the ISR instrument is at one fixed site -- there is no
horizontal/geolocation dimension to compute a joint vertical+horizontal
(``PCA_3D``-style) basis from (plan Section 3).

**Input must be the IRI2020-topside-extended ISR file**
(``extend_isr_edp_with_iri2020.py``'s output, e.g.
``TROISR2025_nonan_extended.nc``), not the raw preprocessed file
``isr_comparison.py`` reads directly -- a PCA basis needs to be defined on
the same altitude range the rest of the system uses (typically 90-900km),
but native ISR gates only reach ~650km. ``isr_comparison.py``'s direct
model-to-ISR-grid comparison has no such requirement and intentionally
does not need this extension.

**Altitude grid**: profiles are resampled (log10-linear interpolation, no
extrapolation) from the extended file's own (native + IRI-extended)
altitude axis onto the caller-supplied ``altitude_grid`` -- normally
``CycleConfig.altitude_grid``, the regular IRI2020 production grid.
Confirmed with the user (2026-10-06): use the regular IRI2020 grid for
now, not ISR's own native (irregular) grid -- a possible future direction,
not built here.

**Station-agnostic**: this module never reads or assumes a station
identity -- it only operates on altitude/density arrays already extracted
by the caller (typically via ``isr_comparison.load_isr_dataset`` or a
direct ``xarray`` read of the extended file), consistent with the rest of
this package's ISR tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


def _interp_log10_no_extrap(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``np.interp`` in log10 space; points in ``x_new`` outside
    ``[x.min(), x.max()]`` come back ``nan`` (no extrapolation)."""
    log_y = np.log10(np.clip(y, 1.0, None))
    return np.interp(x_new, x, log_y, left=np.nan, right=np.nan)


@dataclass
class PCADiagnosticsLite:
    """Thin, netCDF-round-trippable stand-in for
    ``Parameterization.PCADiagnostics`` (that one holds a dataclass that
    isn't itself saved/loaded anywhere in this codebase yet -- kept
    separate rather than extending the shared one, since this module
    doesn't want to add an ISR-specific save/load contract to a
    survivor-module type)."""

    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    n_retained: int
    retaining_threshold: float
    n_profiles_used: int
    n_profiles_total: int


@dataclass
class IsrPcaBasis:
    altitude: np.ndarray            # (n_height,) -- the model grid this basis is defined on
    PCA: np.ndarray                  # (n_height, n_retained)
    PCA_mean: np.ndarray             # (n_height,) log10-space mean subtracted before PCA
    diagnostics: PCADiagnosticsLite


def resample_isr_profiles_to_grid(
    isr_altitude: np.ndarray, isr_density: np.ndarray, altitude_grid: np.ndarray,
) -> np.ndarray:
    """Resample ``(n_isr_alt, n_profile)`` ISR density profiles (on the
    extended file's native+IRI-extended altitude axis) onto
    ``altitude_grid`` via log10-linear interpolation, column by column.
    Returns ``(n_height, n_profile)`` log10(density); a grid point outside
    ``isr_altitude``'s range comes back ``nan`` for every profile (no
    extrapolation -- see module docstring)."""
    n_profile = isr_density.shape[1]
    n_height = len(altitude_grid)
    out = np.empty((n_height, n_profile), dtype=float)
    for j in range(n_profile):
        out[:, j] = _interp_log10_no_extrap(altitude_grid, isr_altitude, isr_density[:, j])
    return out


def build_isr_pca_basis(
    extended_isr_path: str | Path, altitude_grid: np.ndarray, retaining_threshold: float,
    min_valid_fraction: float = 0.5,
) -> IsrPcaBasis:
    """
    Fit a 1-D, log10-space PCA basis from an IRI2020-topside-extended ISR
    file's full profile record.

    Parameters
    ----------
    extended_isr_path : str or Path
        ``extend_isr_edp_with_iri2020.py``'s output (``altitude``/``Ne``
        data variables on the combined native+extended altitude axis).
    altitude_grid : ndarray, shape (n_height,)
        The model grid to define the basis on (e.g.
        ``CycleConfig.altitude_grid``) -- must lie within the extended
        file's own altitude range, or the out-of-range levels contribute
        no real information (see below).
    retaining_threshold : float in (0, 1]
        Forwarded to ``Parameterization.get_PCA`` -- same semantics as
        every other PCA style in this codebase (minimum cumulative
        explained-variance fraction to retain).
    min_valid_fraction : float
        A resampled profile is dropped from the fit if fewer than this
        fraction of ``altitude_grid`` points resolve to a finite value
        (e.g. a profile whose own extension fit failed, see
        ``extend_isr_edp_with_iri2020.py``'s ``n_valid_fit`` -- that
        script already leaves such a profile's extended rows ``nan``
        rather than fabricating a value). Any grid point that's still
        ``nan`` in a kept profile is treated as a zero perturbation from
        the mean by ``get_PCA`` itself (its own documented NaN handling),
        not dropped point-by-point.

    Returns
    -------
    IsrPcaBasis
    """
    from Parameterization import get_PCA

    with xr.open_dataset(extended_isr_path, decode_timedelta=False) as ds:
        isr_altitude = np.asarray(ds["altitude"].values, dtype=float)
        isr_density = np.asarray(ds["Ne"].values, dtype=float)

    altitude_grid = np.asarray(altitude_grid, dtype=float)
    log_density = resample_isr_profiles_to_grid(isr_altitude, isr_density, altitude_grid)

    n_profiles_total = log_density.shape[1]
    valid_fraction = np.isfinite(log_density).mean(axis=0)
    keep = valid_fraction >= min_valid_fraction
    log_density = log_density[:, keep]
    n_profiles_used = log_density.shape[1]
    if n_profiles_used == 0:
        raise ValueError(
            "build_isr_pca_basis: no ISR profile has enough valid resampled altitude "
            f"points (min_valid_fraction={min_valid_fraction}) -- check that altitude_grid "
            "overlaps the extended file's own altitude range."
        )

    PCA, mean, diag = get_PCA(log_density, retaining_threshold)

    diagnostics = PCADiagnosticsLite(
        singular_values=diag.singular_values,
        explained_variance_ratio=diag.explained_variance_ratio,
        cumulative_variance_ratio=diag.cumulative_variance_ratio,
        n_retained=diag.n_retained,
        retaining_threshold=diag.retaining_threshold,
        n_profiles_used=n_profiles_used,
        n_profiles_total=n_profiles_total,
    )
    return IsrPcaBasis(altitude=altitude_grid, PCA=PCA, PCA_mean=mean, diagnostics=diagnostics)


def save_isr_pca_basis(basis: IsrPcaBasis, path: str | Path) -> None:
    """Save an :class:`IsrPcaBasis` to netCDF, for
    ``CycleConfig.isr_pca_basis_path``'s precomputed-file mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = basis.diagnostics
    ds = xr.Dataset(
        data_vars=dict(
            altitude=(("altitude",), basis.altitude),
            PCA=(("altitude", "nPCA"), basis.PCA),
            PCA_mean=(("altitude",), basis.PCA_mean),
            singular_values=(("mode",), d.singular_values),
            explained_variance_ratio=(("mode",), d.explained_variance_ratio),
            cumulative_variance_ratio=(("mode",), d.cumulative_variance_ratio),
        ),
        attrs=dict(
            n_retained=d.n_retained,
            retaining_threshold=d.retaining_threshold,
            n_profiles_used=d.n_profiles_used,
            n_profiles_total=d.n_profiles_total,
        ),
    )
    ds.to_netcdf(path)


def load_isr_pca_basis(path: str | Path) -> IsrPcaBasis:
    """Inverse of :func:`save_isr_pca_basis`."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        diagnostics = PCADiagnosticsLite(
            singular_values=np.asarray(ds["singular_values"].values),
            explained_variance_ratio=np.asarray(ds["explained_variance_ratio"].values),
            cumulative_variance_ratio=np.asarray(ds["cumulative_variance_ratio"].values),
            n_retained=int(ds.attrs["n_retained"]),
            retaining_threshold=float(ds.attrs["retaining_threshold"]),
            n_profiles_used=int(ds.attrs["n_profiles_used"]),
            n_profiles_total=int(ds.attrs["n_profiles_total"]),
        )
        return IsrPcaBasis(
            altitude=np.asarray(ds["altitude"].values, dtype=float),
            PCA=np.asarray(ds["PCA"].values, dtype=float),
            PCA_mean=np.asarray(ds["PCA_mean"].values, dtype=float),
            diagnostics=diagnostics,
        )


def resolve_hyper_params_for_style(cfg, style: str, hyper_params: dict | None) -> dict | None:
    """The one piece of per-style wiring ``'PCA_1D_10ex_ISR'`` needs beyond
    what every other style already gets from ``hyper_params``/
    ``hyper_params_by_style``: if ``style`` is this one, ``hyper_params``
    doesn't already carry a ``'PCA'``, and ``cfg.isr_pca_basis_path`` is
    set, load that basis and merge it in. A no-op (returns
    ``hyper_params`` unchanged) for every other style, or when ``'PCA'``
    is already supplied explicitly (that takes priority over the
    configured path). Called by ``ensemble_init.py``, ``package_run.py``,
    and ``style_sweep.py`` wherever they resolve a style's hyper_params
    before building a ``Parameterized_EDPSamples``."""
    if style != "PCA_1D_10ex_ISR":
        return hyper_params
    hp = dict(hyper_params) if hyper_params else {}
    if "PCA" not in hp and getattr(cfg, "isr_pca_basis_path", None) is not None:
        basis = load_isr_pca_basis(cfg.isr_pca_basis_path)
        hp.update(hyper_params_from_basis(basis))
    return hp


def hyper_params_from_basis(basis: IsrPcaBasis) -> dict:
    """``{'PCA': ..., 'PCA_mean': ...}``, ready to pass as
    ``hyper_params`` to ``Parameterized_EDPSamples(..., style='PCA_1D_10ex_ISR',
    hyper_params=...)`` (or via ``CycleConfig.hyper_params``/
    ``hyper_params_by_style``)."""
    return {"PCA": basis.PCA, "PCA_mean": basis.PCA_mean}
