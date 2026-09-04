# fit isr 8-parameter


import pandas as pd
import numpy as np

def _fit_esr_8params(
    edps: list[dict],
    t_centre: pd.Timestamp,
    alt_grid: np.ndarray,
) -> tuple[np.ndarray | None, dict | None]:
    """
    Find the ESR ISR profile closest to t_centre and fit it with the same
    8-parameter model used by the EKF.

    TRO is explicitly ignored.

    Returns
    -------
    esr_params : ndarray, shape (N_STATE,)
        [log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE]
        in PARAM_NAMES ordering.

    esr_edp : dict
        Original ESR profile used in the fit.
    """

    t_centre = pd.Timestamp(t_centre)
    if t_centre.tzinfo is not None:
        t_centre = t_centre.tz_localize(None)

    esr_inst = INSTRUMENTS["ESR"]

    candidates = []

    for e in edps:

        # ------------------------------------------------------------
        # ESR ONLY
        # ------------------------------------------------------------
        if (
            abs(float(e["lat"]) - esr_inst["lat"]) > ISR_SITE_MATCH_DEG
            or abs(float(e["lon"]) - esr_inst["lon"]) > ISR_SITE_MATCH_DEG
        ):
            continue

        e_time = pd.Timestamp(e["time"])
        if e_time.tzinfo is not None:
            e_time = e_time.tz_localize(None)

        dt_min = abs((e_time - t_centre).total_seconds()) / 60.0

        if dt_min <= ISR_WINDOW_HALF_MINUTES:
            candidates.append((dt_min, e))

    if not candidates:
        print(
            f"  [ESR-COV] No ESR profile within "
            f"±{ISR_WINDOW_HALF_MINUTES} min of {t_centre}"
        )
        return None, None

    # Closest ESR scan to the EKF window centre
    candidates.sort(key=lambda x: x[0])
    dt_min, esr_edp = candidates[0]

    esr_alt = np.asarray(esr_edp["alt_km"], dtype=float)
    esr_ne = np.asarray(esr_edp["ne_m3"], dtype=float)

    valid = (
        np.isfinite(esr_alt)
        & np.isfinite(esr_ne)
        & (esr_ne > 1e7)
    )

    if valid.sum() < ISR_MIN_VALID_GATES:
        print(
            f"  [ESR-COV] ESR profile has only {valid.sum()} valid gates; "
            "skipping covariance tuning."
        )
        return None, None

    esr_alt = esr_alt[valid]
    esr_ne = esr_ne[valid]

    # Sort by altitude before interpolation
    order = np.argsort(esr_alt)
    esr_alt = esr_alt[order]
    esr_ne = esr_ne[order]

    # ------------------------------------------------------------
    # Put ESR onto the same altitude grid used by the EKF.
    # Do NOT extrapolate outside the measured ESR altitude range.
    # ------------------------------------------------------------
    esr_on_grid = np.interp(
        alt_grid,
        esr_alt,
        esr_ne,
        left=np.nan,
        right=np.nan,
    )

    valid_grid = np.isfinite(esr_on_grid) & (esr_on_grid > 1e7)

    if valid_grid.sum() < ISR_MIN_VALID_GATES:
        print(
            "  [ESR-COV] Too little ESR coverage after interpolation; "
            "skipping covariance tuning."
        )
        return None, None

    # _fit_iri_params expects a complete profile.
    # Restrict the fitting grid to the part actually measured by ESR.
    fit_alt = np.asarray(alt_grid)[valid_grid]
    fit_ne = esr_on_grid[valid_grid]

    try:
        esr_params = np.asarray(
            _fit_iri_params(fit_ne, fit_alt),
            dtype=float,
        )
    except Exception as exc:
        print(f"  [ESR-COV] ESR 8-param fit failed: {exc}")
        return None, None

    if esr_params.size != N_STATE or not np.all(np.isfinite(esr_params)):
        print(
            f"  [ESR-COV] Invalid ESR parameter fit: {esr_params}"
        )
        return None, None

    print(
        f"\n  [ESR-COV] ESR profile selected:"
        f"\n            time = {esr_edp['time']}"
        f"\n            Δt   = {dt_min:.2f} min"
    )

    print("  [ESR-COV] ESR fitted parameters:")
    for name, value in zip(PARAM_NAMES, esr_params):
        print(f"            {name:<12s} = {value:.6g}")

    return esr_params, esr_edp


def _tune_param_covariance_with_esr(
    P_b: np.ndarray,
    mean_state: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    esr_params: np.ndarray,
    weight: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Tune the 8x8 EKF parameter background covariance using ESR truth.

    The ESR-derived parameter vector is compared against the IRI prior
    parameter vector at the horizontal grid point nearest ESR.

    A single ESR profile gives one parameter-error sample:

        e = p_ESR - p_IRI

    so its covariance contribution is

        P_ESR = e e^T

    and the tuned covariance is

        P_b,tuned = P_b + weight * P_ESR

    TRO is not involved.
    """

    P_b = np.asarray(P_b, dtype=float)
    mean_state = np.asarray(mean_state, dtype=float)
    esr_params = np.asarray(esr_params, dtype=float)

    if P_b.shape != (N_STATE, N_STATE):
        raise ValueError(
            f"P_b has shape {P_b.shape}; "
            f"expected ({N_STATE}, {N_STATE})"
        )

    esr_inst = INSTRUMENTS["ESR"]

    # ------------------------------------------------------------
    # Find EKF grid point nearest ESR
    # ------------------------------------------------------------
    distances = _haversine_km(
        esr_inst["lat"],
        esr_inst["lon"],
        grid_lats,
        grid_lons,
    )

    esr_grid_idx = int(np.argmin(distances))
    nearest_km = float(distances[esr_grid_idx])

    # IRI 8-param background at ESR
    iri_params_esr = mean_state[:, esr_grid_idx].copy()

    # ------------------------------------------------------------
    # Parameter error against ESR truth
    # ------------------------------------------------------------
    error = esr_params - iri_params_esr

    # One-sample covariance contribution.
    #
    # This is rank-1, which is expected because we only have one ESR
    # truth profile in this experiment.
    P_esr = np.outer(error, error)

    # ------------------------------------------------------------
    # Update the ORIGINAL background covariance rather than replacing it.
    # ------------------------------------------------------------
    P_tuned = P_b + float(weight) * P_esr

    # Numerical symmetry
    P_tuned = 0.5 * (P_tuned + P_tuned.T)

    print(
        f"\n  [ESR-COV] Nearest EKF grid point to ESR:"
        f"\n            index    = {esr_grid_idx}"
        f"\n            lat/lon  = "
        f"{grid_lats[esr_grid_idx]:.3f}, "
        f"{grid_lons[esr_grid_idx]:.3f}"
        f"\n            distance = {nearest_km:.1f} km"
    )

    print("\n  [ESR-COV] IRI prior vs ESR parameter truth:")

    for i, name in enumerate(PARAM_NAMES):
        print(
            f"            {name:<12s} "
            f"IRI={iri_params_esr[i]:12.5g}  "
            f"ESR={esr_params[i]:12.5g}  "
            f"error={error[i]:+12.5g}"
        )

    print("\n  [ESR-COV] Parameter standard deviations:")
    for i, name in enumerate(PARAM_NAMES):

        std_old = np.sqrt(max(P_b[i, i], 0.0))
        std_esr = abs(error[i])
        std_new = np.sqrt(max(P_tuned[i, i], 0.0))

        print(
            f"            {name:<12s} "
            f"old={std_old:10.4g}  "
            f"ESR-error={std_esr:10.4g}  "
            f"new={std_new:10.4g}"
        )

    diagnostics = {
        "esr_params": esr_params.copy(),
        "iri_params_esr": iri_params_esr,
        "esr_param_error": error,
        "P_b_original": P_b.copy(),
        "P_esr": P_esr,
        "P_b_tuned": P_tuned.copy(),
        "esr_grid_idx": esr_grid_idx,
        "esr_grid_distance_km": nearest_km,
    }

    return P_tuned, diagnostics