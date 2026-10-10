import numpy as np


# ============================================================
# EKF pairwise parameter-correlation tuning
# ============================================================
EKF_PARAM_CORR_TUNING = {
    ("log10(NmF2)", "hmF2"):       -0.0125,
    ("log10(NmF2)", "H0"):         -0.0165,
    ("log10(NmF2)", "gamma"):      -0.0047,
    ("log10(NmF2)", "B0"):         +0.0018,
    ("log10(NmF2)", "B1"):         +0.0135,
    ("log10(NmF2)", "log10(NmE)"): +0.0072,
    ("log10(NmF2)", "hmE"):        +0.0000,

    ("hmF2", "H0"):                +0.0633,
    ("hmF2", "gamma"):             +0.0181,
    ("hmF2", "B0"):                +0.0157,
    ("hmF2", "B1"):                -0.0382,
    ("hmF2", "log10(NmE)"):        +0.0316,
    ("hmF2", "hmE"):               +0.0000,

    ("H0", "gamma"):               +0.0165,
    ("H0", "B0"):                  +0.0157,
    ("H0", "B1"):                  -0.0435,
    ("H0", "log10(NmE)"):          +0.0262,
    ("H0", "hmE"):                 +0.0000,

    ("gamma", "B0"):               +0.0047,
    ("gamma", "B1"):               -0.0118,
    ("gamma", "log10(NmE)"):       +0.0073,
    ("gamma", "hmE"):              +0.0000,

    ("B0", "B1"):                  -0.0044,
    ("B0", "log10(NmE)"):          +0.0107,
    ("B0", "hmE"):                 +0.0000,

    ("B1", "log10(NmE)"):          -0.0273,
    ("B1", "hmE"):                 +0.0000,

    ("log10(NmE)", "hmE"):         +0.0000,
}


# ============================================================
# Apply manually specified correlations to an existing
# state-estimate error covariance matrix.
#
# IMPORTANT:
#   - diagonal variances are preserved
#   - only off-diagonal covariances are replaced
# ============================================================

def tune_all_parameter_covariances(
    P_b,
    param_names,
    tuning_dict=EKF_PARAM_CORR_TUNING,
):
    """
    Replace the off-diagonal parameter covariances using
    user-defined correlations.

    Cov(i,j) = rho(i,j) * sigma_i * sigma_j

    Parameters
    ----------
    P_b : ndarray, shape (8, 8)
        Original IRI-generated parameter covariance.

    param_names : sequence of str
        EKF parameter names in state order.

    tuning_dict : dict
        Pairwise correlations.

    Returns
    -------
    P_new : ndarray, shape (8, 8)
        Covariance with original diagonal variances and
        manually specified off-diagonal covariances.
    """

    P_b = np.asarray(P_b, dtype=float)

    if P_b.shape != (len(param_names), len(param_names)):
        raise ValueError(
            f"P_b shape {P_b.shape} does not match "
            f"{len(param_names)} parameters."
        )

    P_new = P_b.copy()

    # ------------------------------------------------------------
    # Original state-estimate error variance / STD
    # ------------------------------------------------------------

    variance = np.maximum(
        np.diag(P_b),
        0.0,
    )

    sigma = np.sqrt(variance)

    # ------------------------------------------------------------
    # Start with ZERO covariance between different parameters
    #
    # Keep original diagonal variance exactly.
    # ------------------------------------------------------------

    P_new[:] = 0.0
    np.fill_diagonal(
        P_new,
        variance,
    )

    name_to_idx = {
        name: i
        for i, name in enumerate(param_names)
    }

    # ------------------------------------------------------------
    # Set all manually defined cross-parameter correlations
    # ------------------------------------------------------------

    for (name_i, name_j), rho in tuning_dict.items():

        if name_i not in name_to_idx:
            raise ValueError(
                f"Unknown parameter: {name_i}"
            )

        if name_j not in name_to_idx:
            raise ValueError(
                f"Unknown parameter: {name_j}"
            )

        rho = float(rho)

        if not -1.0 <= rho <= 1.0:
            raise ValueError(
                f"Correlation must be in [-1,1]: "
                f"{name_i} <-> {name_j} = {rho}"
            )

        i = name_to_idx[name_i]
        j = name_to_idx[name_j]

        cov_ij = (
            rho
            * sigma[i]
            * sigma[j]
        )

        P_new[i, j] = cov_ij
        P_new[j, i] = cov_ij

    # ------------------------------------------------------------
    # Numerical symmetry
    # ------------------------------------------------------------

    P_new = 0.5 * (
        P_new + P_new.T
    )

    # ------------------------------------------------------------
    # Check covariance validity
    # ------------------------------------------------------------

    eigvals = np.linalg.eigvalsh(P_new)

    min_eig = float(
        np.min(eigvals)
    )

    if min_eig < -1e-10:
        raise ValueError(
            "Manual parameter correlations produce an invalid "
            "covariance matrix (not positive semidefinite).\n"
            f"Minimum eigenvalue = {min_eig:.6e}"
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print("\n[EKF covariance] Manual parameter correlations applied:")

    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):

            denom = sigma[i] * sigma[j]

            rho = (
                P_new[i, j] / denom
                if denom > 0
                else 0.0
            )

            print(
                f"  {param_names[i]:<12s} <-> "
                f"{param_names[j]:<12s} "
                f"rho={rho:+.4f}   "
                f"cov={P_new[i,j]:+.6g}"
            )

    return P_new