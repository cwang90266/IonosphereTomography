import numpy as np
from scipy import sparse


def haversine_matrix(lats, lons):
    """Pairwise great-circle distances in km between (lat, lon) points."""
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_gaussian_smoother(heights, lats, lons, sigma_h, sigma_latlon, truncate=3.0):
    """
    Gaussian smoothing matrix for a function on a heights x (lat, lon) grid.

    The function vector f is indexed as f[k * n_pts + i], where k is the height
    index and i is the horizontal point index. Apply smoothing as f_smooth = S @ f.

    Parameters
    ----------
    heights     : array (nh,)       Height levels (any units; sigma_h must match)
    lats        : array (n_pts,)    Point latitudes  (degrees)
    lons        : array (n_pts,)    Point longitudes (degrees)
    sigma_h     : float             Vertical Gaussian std dev (same units as heights)
    sigma_latlon: float             Horizontal Gaussian std dev (km)
    truncate    : float             Zero weights beyond this many sigma (sparsity)

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape (nh*n_pts, nh*n_pts)
        Row-normalized smoothing matrix.
    """
    heights = np.asarray(heights, dtype=float)
    lats    = np.asarray(lats,    dtype=float)
    lons    = np.asarray(lons,    dtype=float)

    # Vertical weight matrix (nh x nh)
    dh  = heights[:, None] - heights[None, :]
    W_h = np.where(np.abs(dh) <= truncate * sigma_h,
                   np.exp(-0.5 * (dh / sigma_h) ** 2), 0.0)

    # Horizontal weight matrix (n_pts x n_pts)
    d_ll = haversine_matrix(lats, lons)
    W_ll = np.where(d_ll <= truncate * sigma_latlon,
                    np.exp(-0.5 * (d_ll / sigma_latlon) ** 2), 0.0)

    # Full weight matrix via Kronecker product, then row-normalize
    # kron(W_h, W_ll)[k_out*n + i_out, k_in*n + i_in] = W_h[k_out,k_in] * W_ll[i_out,i_in]
    W = sparse.kron(sparse.csr_matrix(W_h), sparse.csr_matrix(W_ll), format="csr")

    row_sums = np.array(W.sum(axis=1)).ravel()
    S = sparse.diags(1.0 / row_sums) @ W

    return S.tocsr()