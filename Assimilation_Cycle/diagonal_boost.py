#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal covariance boosting (user-proposed, 2026-09-27): the raw
IRI2020-driven ensemble is a climatological draw whose member-to-member
variability is dominated by a handful of smooth, broadly-correlated modes
(driving indices like f107/ap/ig varying together across the whole
column/region) -- confirmed independently by how few PCA modes are needed
to explain nearly all of a real ensemble's variance (`PCA_3D_10ex`, ~110
of 13202 nominal dimensions). That low *effective* rank limits the EnKF's
reachable correction subspace to directions the climatological ensemble
happens to already vary along, which can leave genuine small-scale
structure (seen as large TEC/EDP residuals on specific real occultations)
unreachable no matter how much data is assimilated.

This module manufactures new, less-correlated directions of spread by
adding a smoothed random field to every ensemble member's raw density
profile, *before* any parameterization (PCA or otherwise) is fit --
critical, since fitting PCA to the un-boosted ensemble and adding noise
afterward would just have `retaining_threshold` discard most of it.

Design (as specified by the user):
  1. Draw one i.i.d. standard-normal field per ensemble member, on the
     same (height, geo) grid as the raw ``EDPs`` array.
  2. Smooth each field vertically and horizontally (Gaussian-kernel
     weighted local averaging -- a distance-based kernel along the real
     altitude coordinate, and a haversine-distance-based kernel over the
     (generally unstructured) horizontal point cloud) to introduce a
     configurable local correlation length, rather than leaving them pure
     white noise.
  3. Renormalize each (height, geo) point's smoothed field to exactly
     zero mean / unit standard deviation across members (smoothing both
     shifts the sample mean at finite ensemble size and shrinks the
     variance; the renormalization guarantees the *injected* perturbation
     has precisely the requested scale regardless of the smoothing
     kernel's own variance-reduction, and never shifts the forecast
     mean).
  4. Scale by ``amplitude * per-point ensemble std`` and add to the
     original samples (linear space) or to ``log10`` of the original
     samples (log space -- see below).

**Linear vs. log10 space (``log_space`` parameter), found the hard way
(2026-09-27 real-data sweep):** adding noise directly to linear-space
density, scaled by that point's own linear-space std, can push individual
members' density below zero at points/members where the local
std/mean ratio is high (observed up to >2 in a real ensemble) --
requiring a clip to a small positive floor. That's fine for the ``raw``
style (confirmed on real data: a genuine, modest, robust held-out RMSE
improvement, no pathology) since ``raw`` consumes density directly. But
``density_10ex``/``PCA_*_10ex`` take ``log10`` of density before fitting
anything -- a value clipped to the floor becomes ``log10(1.0) = 0`` next
to typical surrounding values around ``log10 ~ 9-12``, an 8-12
order-of-magnitude discontinuity in the space those styles actually fit,
which severely corrupts the ensemble's sample statistics at that point
and caused catastrophic (RMSE into the millions) or non-convergent
real-data results for those styles at exactly this amplitude range.
``log_space=True`` avoids the mechanism entirely: the perturbation is
added to ``log10(density)`` instead, which has no positivity constraint
(so no clipping is ever needed), and converted back via ``10**(...)`` --
matches what ``density_10ex``/``PCA_*_10ex`` themselves operate on, and
is also a more physically apt (multiplicative-in-linear-space) way to
perturb a field spanning multiple orders of magnitude. Not yet validated
for ``PCA_3D_10ex`` at the time this was written -- see
``Assimilation_Cycle_Integration_Plan.md`` §17 for the latest result.
"""

from __future__ import annotations

import numpy as np

_EARTH_RADIUS_KM = 6371.0
_DENSITY_FLOOR = 1.0   # m^-3; real EDP values are >= ~1e8, this is a no-op safety clip (linear space only)


def _haversine_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(n, n) great-circle distance matrix in km, from degree-valued
    ``lon``/``lat`` arrays of shape (n,)."""
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _gaussian_smoothing_matrix(distance: np.ndarray, scale: float) -> np.ndarray:
    """Row-normalized Gaussian-kernel smoothing matrix from a pairwise
    ``distance`` matrix and a length ``scale`` (same units)."""
    if scale <= 0:
        return np.eye(distance.shape[0])
    K = np.exp(-0.5 * (distance / scale) ** 2)
    return K / K.sum(axis=1, keepdims=True)


def _smoothed_normalized_noise(
    shape: tuple[int, int, int],
    altitude: np.ndarray,
    geo: np.ndarray,
    vertical_scale_km: float,
    horizontal_scale_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """i.i.d. per-member noise, vertically+horizontally Gaussian-smoothed,
    then renormalized per (height, geo) point to exact zero-mean/unit-std
    across members (see module docstring, point 3)."""
    W = rng.standard_normal(size=shape)

    dh = np.abs(altitude[:, None] - altitude[None, :])
    K_v = _gaussian_smoothing_matrix(dh, vertical_scale_km)
    W_v = np.einsum('ij,jgm->igm', K_v, W)

    dist_h = _haversine_km(geo[:, 0], geo[:, 1])
    K_h = _gaussian_smoothing_matrix(dist_h, horizontal_scale_km)
    W_hv = np.einsum('ij,hjm->him', K_h, W_v)

    mean_pt = W_hv.mean(axis=2, keepdims=True)
    centered = W_hv - mean_pt
    std_pt = centered.std(axis=2, keepdims=True)
    std_pt = np.where(std_pt < 1e-12, 1.0, std_pt)
    return centered / std_pt


def apply_diagonal_boost(
    edp_samples,
    amplitude: float | None,
    vertical_scale_km: float = 30.0,
    horizontal_scale_km: float = 200.0,
    rng: np.random.Generator | None = None,
    log_space: bool = False,
):
    """
    Return a copy of ``edp_samples`` with a smoothed, per-point-std-scaled
    random field added to every member's raw ``EDPs``, before any
    parameterization is fit.

    Parameters
    ----------
    edp_samples : EDPSamples
        The raw (un-parameterized) ensemble -- must be applied before
        ``Parameterized_EDPSamples`` is built (see module docstring).
    amplitude : float or None
        The injected perturbation's per-point standard deviation, as a
        fraction of that point's own ensemble standard deviation in
        whichever space it's applied (linear density, or ``log10``
        density if ``log_space=True``) (``0``/``None`` -- returns
        ``edp_samples`` unchanged, a no-op).
    vertical_scale_km : float
        Gaussian smoothing length along the altitude coordinate (km).
    horizontal_scale_km : float
        Gaussian smoothing length over the horizontal point cloud (km,
        great-circle distance) -- handles the unstructured/mesh-based
        horizontal grid directly, no regular-grid assumption.
    rng : numpy.random.Generator, optional
        Defaults to an unseeded generator; pass one explicitly for
        reproducible comparisons (this codebase's standing lesson from
        the inter-batch-inflation and batch-order investigations).
    log_space : bool
        ``False`` (default): perturb linear density directly, clipped to
        stay positive -- validated to genuinely help for ``raw``, but
        causes severe corruption for ``log10``-based styles (see module
        docstring). ``True``: perturb ``log10(density)`` instead and
        convert back via ``10**(...)`` -- no clipping ever needed, matches
        what ``density_10ex``/``PCA_*_10ex`` actually fit.

    Returns
    -------
    EDPSamples
        A deep copy with boosted ``EDPs``; ``edp_samples`` itself is left
        untouched. Returned as-is (no copy) when ``amplitude`` is falsy.
    """
    if not amplitude:
        return edp_samples

    rng = rng or np.random.default_rng()

    X = edp_samples.edps                      # (n_height, n_geo, n_members)
    altitude = edp_samples.altitude            # (n_height,)
    geo = edp_samples.geolocation              # (n_geo, 2) == [lon, lat] degrees

    base = np.log10(X) if log_space else X
    std_hg = base.std(axis=2)                  # (n_height, n_geo)

    normalized = _smoothed_normalized_noise(
        X.shape, altitude, geo, vertical_scale_km, horizontal_scale_km, rng,
    )

    perturbation = amplitude * std_hg[:, :, None] * normalized
    base_boosted = base + perturbation
    X_boosted = (10.0 ** base_boosted) if log_space else np.clip(base_boosted, a_min=_DENSITY_FLOOR, a_max=None)

    boosted = edp_samples.copy(deep=True)
    var_name = boosted.VAR_EDPS
    boosted[var_name] = (boosted[var_name].dims, X_boosted.astype(boosted[var_name].dtype))
    return boosted
