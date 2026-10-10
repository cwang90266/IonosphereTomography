import numpy as np
from scipy.optimize import minimize

def closest_point_on_ellipsoid(P_LEO, P_GNSS, R_E, R_P):
    """
    Finds the closest point on the ellipsoid defined by
    x^2/R_E^2 + y^2/R_E^2 + z^2/R_P^2 = 1
    to the segment between P_LEO and P_GNSS (both are 3D Cartesian coordinates).
    
    Args:
        P_LEO: np.ndarray of shape (3,) - [x, y, z] of LEO satellite
        P_GNSS: np.ndarray of shape (3,) - [x, y, z] of GNSS satellite
        R_E: float - Equatorial radius
        R_P: float - Polar radius
    
    Returns:
        np.ndarray: (3,) Closest point on ellipsoid.
    """

    def point_on_segment(t):
        return P_LEO + t * (P_GNSS - P_LEO)

    def ellipsoid_constraint(x):
        return (x[0]/R_E)**2 + (x[1]/R_E)**2 + (x[2]/R_P)**2 - 1
    
    def objective(t):
        # Project onto ellipsoid
        P= point_on_segment(t[0])
        # Find closest point Q on ellipsoid to P
        def dist(x):
            return np.sum((x - P)**2)

        x0 = P * min(1.0, R_E / (np.linalg.norm(P) + 1e-9))
        cons = {'type':'eq', 'fun': ellipsoid_constraint}
        res = minimize(dist, x0, constraints=cons, method='SLSQP')
        if not res.success:
            raise RuntimeError("Could not find closest point on ellipsoid.")
        Q = res.x
        return np.linalg.norm(P - Q)
    
    # Minimize distance over segment param t in [-1, 1]
    seg_res = minimize(objective, [0.5], bounds=[(-1, 1)], method='L-BFGS-B')
    t_best = seg_res.x[0]
    P_star = point_on_segment(t_best)
    def dist(x):
        return np.sum((x - P_star)**2)
    # Find closest point Q_star on ellipsoid to P_star. Q_star has been found
    # in the inner optimization loop of previous optimization. But Q_star was not
    # output in the previous optimization step.
    x0 = P_star * min(1.0, R_E / (np.linalg.norm(P_star) + 1e-9))
    cons = {'type':'eq', 'fun': ellipsoid_constraint}
    res = minimize(dist, x0, constraints=cons, method='SLSQP')
    if not res.success:
        raise RuntimeError("Could not find closest point on ellipsoid.")
    Q_star = res.x
    return Q_star