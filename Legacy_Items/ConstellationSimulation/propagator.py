"""
propagator.py

Orbital mechanics engine for the GNSS-RO availability simulation.

Advances TX/RX satellite state vectors (as produced by tx_constellation.py /
rx_constellation.py) forward in time with an RK4 numerical integrator. The
total acceleration acting on each satellite is

    a_total = a_2body + a_J2 + a_J3 + a_SRP

All internal integration is done in km / km*s^-1 / km*s^-2 (the natural
units for the constants below); Satellite.r_eci_m / v_eci_m_s are metres,
so the km<->m conversion happens only at the module boundary.

Note on reference frames: this module treats whatever frame a Satellite's
r_eci_m/v_eci_m_s are expressed in as a non-rotating inertial frame (the
two-body/J2/J3/SRP formulas below all assume that). Walker-generated
satellites satisfy this. Satellites instantiated from RINEX broadcast
ephemerides (TXConstellation.from_rinex_nav) are in ECEF instead -- call
ecef_to_eci() / satellite_ecef_to_eci() / constellation_ecef_to_eci() on
them first to get a proper (GMST-only, pseudo-inertial) ECI state before
propagating.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tx_constellation import Satellite

# ── Physical constants ────────────────────────────────────────────────────
MU = 398600.4418        # km^3 s^-2, Earth's gravitational parameter
R_E = 6378.137          # km, Earth's equatorial radius
J2 = 1.08263e-3         # Earth oblateness zonal harmonic
J3 = -2.532e-6          # Earth pear-shape zonal harmonic
P_SR = 4.56e-6          # N/m^2, solar radiation pressure at 1 AU
AU_KM = 149597870.7     # km, 1 astronomical unit
OMEGA_E = 7.2921151467e-5   # rad/s, WGS-84 Earth rotation rate

# Default cannonball SRP parameters for a small LEO satellite.
DEFAULT_MASS_KG = 30.0
DEFAULT_AREA_M2 = 0.4
DEFAULT_CR = 1.3

_J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────────────────────
# Sun position (low-precision solar ephemeris, Vallado/Astronomical Almanac)
# ──────────────────────────────────────────────────────────────────────────

def _days_since_j2000(epoch: datetime) -> float:
    """Fractional days elapsed between *epoch* and the J2000.0 reference."""
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return (epoch - _J2000_EPOCH).total_seconds() / 86400.0


def sun_position_km(epoch: datetime) -> np.ndarray:
    """Geocentric position of the Sun (km) at *epoch*, via the low-precision
    solar ephemeris (accurate to ~0.01 deg through 2050; Vallado, "Fundamentals
    of Astrodynamics and Applications"). Frame matches whatever inertial
    frame the propagated satellites are expressed in (see module docstring).
    """
    T = _days_since_j2000(epoch) / 36525.0

    lam_mean = math.radians((280.460 + 36000.771 * T) % 360.0)
    m_anom = math.radians((357.5291092 + 35999.05034 * T) % 360.0)

    lam_ecl = (lam_mean
               + math.radians(1.914666471) * math.sin(m_anom)
               + math.radians(0.019994643) * math.sin(2.0 * m_anom))
    r_au = (1.000140612
            - 0.016708617 * math.cos(m_anom)
            - 0.000139589 * math.cos(2.0 * m_anom))
    eps = math.radians(23.439291 - 0.0130042 * T)

    x = r_au * math.cos(lam_ecl)
    y = r_au * math.cos(eps) * math.sin(lam_ecl)
    z = r_au * math.sin(eps) * math.sin(lam_ecl)
    return np.array([x, y, z]) * AU_KM


def shadow_factor(r_sat_km: np.ndarray, r_sun_km: np.ndarray,
                   r_earth_km: float = R_E) -> float:
    """Cylindrical Earth-shadow eclipse fraction nu: 1.0 in sunlight, 0.0 in
    (umbra-only, no penumbra) shadow.

    The satellite is considered eclipsed when it lies on the anti-sun side
    of Earth's centre *and* its perpendicular distance from the Earth-Sun
    line is smaller than Earth's radius (i.e. inside the shadow cylinder).
    """
    sun_hat = r_sun_km / np.linalg.norm(r_sun_km)
    along = np.dot(r_sat_km, sun_hat)
    if along > 0.0:
        return 1.0  # sunward side of Earth's centre -> always lit
    perp_dist = np.linalg.norm(r_sat_km - along * sun_hat)
    return 0.0 if perp_dist < r_earth_km else 1.0


# ──────────────────────────────────────────────────────────────────────────
# ECEF <-> ECI frame conversion (GMST-only pseudo-inertial ECI, i.e. no
# precession/nutation/polar motion -- consistent with this module's overall
# precision level).
# ──────────────────────────────────────────────────────────────────────────

def _gmst_rad(epoch: datetime) -> float:
    """Greenwich Mean Sidereal Time (rad) at *epoch*, IAU-82 formula
    (Vallado, "Fundamentals of Astrodynamics and Applications"). UT1 is
    approximated by UTC (sub-second UT1-UTC differences are neglected).
    """
    T = _days_since_j2000(epoch) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    gmst_deg = (gmst_sec % 86400.0) * (360.0 / 86400.0)
    return math.radians(gmst_deg % 360.0)


def ecef_to_eci(r_ecef_km: np.ndarray, v_ecef_km_s: np.ndarray, epoch: datetime,
                omega_earth: float = OMEGA_E):
    """Convert an ECEF (Earth-fixed, rotating) position/velocity to a
    GMST-only pseudo-inertial ECI frame at *epoch*.

        r_eci = Rz(theta) @ r_ecef
        v_eci = Rz(theta) @ (v_ecef + omega_earth_vec x r_ecef)

    where theta is Greenwich Mean Sidereal Time and omega_earth_vec =
    [0, 0, omega_earth] is Earth's rotation vector -- the omega x r term
    restores the rotational velocity that is "invisible" in the co-rotating
    ECEF frame (an ECEF-stationary point has v_ecef = 0 but is actually
    moving in inertial space).

    Parameters
    ----------
    r_ecef_km, v_ecef_km_s : (3,) array-like
        ECEF position (km) and velocity (km/s).
    epoch : datetime
        UTC epoch the state vector is valid at.
    omega_earth : float
        Earth's rotation rate (rad/s).

    Returns
    -------
    (r_eci_km, v_eci_km_s) : tuple of (3,) np.ndarray
    """
    theta = _gmst_rad(epoch)
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    r_ecef_km = np.asarray(r_ecef_km, dtype=float)
    v_ecef_km_s = np.asarray(v_ecef_km_s, dtype=float)
    omega_cross_r = np.array([
        -omega_earth * r_ecef_km[1],
         omega_earth * r_ecef_km[0],
         0.0,
    ])

    r_eci_km = Rz @ r_ecef_km
    v_eci_km_s = Rz @ (v_ecef_km_s + omega_cross_r)
    return r_eci_km, v_eci_km_s


def eci_to_ecef(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray, epoch: datetime,
                omega_earth: float = OMEGA_E):
    """Inverse of ecef_to_eci(): convert a GMST-only pseudo-inertial ECI
    position/velocity back to ECEF (Earth-fixed, rotating) at *epoch*.

        r_ecef = Rz(-theta) @ r_eci
        v_ecef = Rz(-theta) @ v_eci - omega_earth_vec x r_ecef

    Used to bring propagated/Walker-generated satellites (pseudo-inertial
    ECI, see module docstring) back to the ECEF convention that
    test_param_iono.py's occultation-arc pipeline expects.

    Parameters
    ----------
    r_eci_km, v_eci_km_s : (3,) array-like
        ECI position (km) and velocity (km/s).
    epoch : datetime
        UTC epoch the state vector is valid at.
    omega_earth : float
        Earth's rotation rate (rad/s).

    Returns
    -------
    (r_ecef_km, v_ecef_km_s) : tuple of (3,) np.ndarray
    """
    theta = _gmst_rad(epoch)
    c, s = math.cos(-theta), math.sin(-theta)
    Rz_inv = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    r_eci_km = np.asarray(r_eci_km, dtype=float)
    v_eci_km_s = np.asarray(v_eci_km_s, dtype=float)

    r_ecef_km = Rz_inv @ r_eci_km
    omega_cross_r = np.array([
        -omega_earth * r_ecef_km[1],
         omega_earth * r_ecef_km[0],
         0.0,
    ])
    v_ecef_km_s = Rz_inv @ v_eci_km_s - omega_cross_r
    return r_ecef_km, v_ecef_km_s


def satellite_ecef_to_eci(sat: "Satellite") -> "Satellite":
    """Convert *sat* in place from ECEF to pseudo-inertial ECI at its
    current epoch. Intended to be called right after
    TXConstellation.from_rinex_nav(), whose broadcast-ephemeris state
    vectors are computed in ECEF/PZ-90 even though they are stored on
    Satellite.r_eci_m/v_eci_m_s (see the module docstring).
    """
    if sat.epoch is None:
        raise ValueError(f"Satellite {sat.name!r} has no epoch to convert at")
    r_ecef_km = np.asarray(sat.r_eci_m, dtype=float) * 1e-3
    v_ecef_km_s = np.asarray(sat.v_eci_m_s, dtype=float) * 1e-3
    r_eci_km, v_eci_km_s = ecef_to_eci(r_ecef_km, v_ecef_km_s, sat.epoch)
    sat.r_eci_m = r_eci_km * 1e3
    sat.v_eci_m_s = v_eci_km_s * 1e3
    return sat


def constellation_ecef_to_eci(constellation):
    """Convert every satellite in a TX/RXConstellation (anything exposing a
    .satellites list of Satellite) from ECEF to pseudo-inertial ECI in place.
    """
    for sat in constellation.satellites:
        satellite_ecef_to_eci(sat)
    return constellation


# ──────────────────────────────────────────────────────────────────────────
# Perturbation accelerations (all km / km*s^-2)
# ──────────────────────────────────────────────────────────────────────────

def acceleration_two_body(r_km: np.ndarray, mu: float = MU) -> np.ndarray:
    """a_2body = -mu / r^3 * r_vec."""
    r = np.linalg.norm(r_km)
    return -mu / r ** 3 * r_km


def acceleration_j2(r_km: np.ndarray, mu: float = MU, r_e: float = R_E,
                    j2: float = J2) -> np.ndarray:
    """J2 (Earth oblateness) Cartesian perturbing acceleration."""
    x, y, z = r_km
    r = np.linalg.norm(r_km)
    factor = -1.5 * j2 * (mu * r_e ** 2 / r ** 5)
    z2_r2 = (z ** 2) / (r ** 2)
    ax = factor * x * (1.0 - 5.0 * z2_r2)
    ay = factor * y * (1.0 - 5.0 * z2_r2)
    az = factor * z * (3.0 - 5.0 * z2_r2)
    return np.array([ax, ay, az])


def acceleration_j3(r_km: np.ndarray, mu: float = MU, r_e: float = R_E,
                    j3: float = J3) -> np.ndarray:
    """J3 (Earth pear-shape) Cartesian perturbing acceleration."""
    x, y, z = r_km
    r = np.linalg.norm(r_km)
    factor = -0.5 * j3 * (mu * r_e ** 3 / r ** 7)
    z3_r2 = z ** 3 / r ** 2
    z4_r2 = z ** 4 / r ** 2
    ax = factor * 5.0 * x * (3.0 * z - 7.0 * z3_r2)
    ay = factor * 5.0 * y * (3.0 * z - 7.0 * z3_r2)
    az = factor * (3.0 * r ** 2 - 30.0 * z ** 2 + 35.0 * z4_r2)
    return np.array([ax, ay, az])


def acceleration_srp(r_km: np.ndarray, epoch: datetime,
                     mass_kg: float = DEFAULT_MASS_KG,
                     area_m2: float = DEFAULT_AREA_M2,
                     cr: float = DEFAULT_CR, p_sr: float = P_SR) -> np.ndarray:
    """Cannonball solar radiation pressure acceleration:

        a_SRP = -P_SR * C_R * (A/m) * nu * u_hat_sun

    where u_hat_sun points from the satellite toward the Sun (so the minus
    sign gives the correct anti-sunward push), and nu is the cylindrical
    shadow factor. P_SR is in N/m^2 = kg/(m*s^2); (A/m) is m^2/kg, so
    P_SR * C_R * (A/m) * nu comes out in m/s^2 and must be converted to
    km/s^2 to match the rest of the integrator.
    """
    r_sun_km = sun_position_km(epoch)
    vec_to_sun = r_sun_km - r_km
    u_sun_hat = vec_to_sun / np.linalg.norm(vec_to_sun)
    nu = shadow_factor(r_km, r_sun_km)

    a_mag_m_s2 = p_sr * cr * (area_m2 / mass_kg) * nu   # m/s^2
    a_mag_km_s2 = a_mag_m_s2 * 1e-3                      # m/s^2 -> km/s^2
    return -a_mag_km_s2 * u_sun_hat


def total_acceleration(r_km: np.ndarray, epoch: datetime,
                       mass_kg: float = DEFAULT_MASS_KG,
                       area_m2: float = DEFAULT_AREA_M2,
                       cr: float = DEFAULT_CR) -> np.ndarray:
    """a_total = a_2body + a_J2 + a_J3 + a_SRP, all km/s^2."""
    return (acceleration_two_body(r_km)
            + acceleration_j2(r_km)
            + acceleration_j3(r_km)
            + acceleration_srp(r_km, epoch, mass_kg, area_m2, cr))


# ──────────────────────────────────────────────────────────────────────────
# RK4 numerical integrator
# ──────────────────────────────────────────────────────────────────────────

def _state_derivative(state_km: np.ndarray, epoch: datetime, mass_kg: float,
                       area_m2: float, cr: float) -> np.ndarray:
    """d/dt [r; v] = [v; a_total(r)] for state = concatenate([r_km, v_km_s])."""
    r_km = state_km[:3]
    v_km_s = state_km[3:]
    a_km_s2 = total_acceleration(r_km, epoch, mass_kg, area_m2, cr)
    return np.concatenate([v_km_s, a_km_s2])


def rk4_step(r_km: np.ndarray, v_km_s: np.ndarray, epoch: datetime, dt_s: float,
             mass_kg: float = DEFAULT_MASS_KG, area_m2: float = DEFAULT_AREA_M2,
             cr: float = DEFAULT_CR):
    """Advance one (r_km, v_km_s) state by *dt_s* seconds with classical RK4.

    Returns
    -------
    (r_km_new, v_km_s_new, epoch_new)
    """
    state = np.concatenate([np.asarray(r_km, dtype=float),
                             np.asarray(v_km_s, dtype=float)])
    half_dt = dt_s / 2.0
    half_epoch = epoch + timedelta(seconds=half_dt)
    full_epoch = epoch + timedelta(seconds=dt_s)

    k1 = _state_derivative(state, epoch, mass_kg, area_m2, cr)
    k2 = _state_derivative(state + half_dt * k1, half_epoch, mass_kg, area_m2, cr)
    k3 = _state_derivative(state + half_dt * k2, half_epoch, mass_kg, area_m2, cr)
    k4 = _state_derivative(state + dt_s * k3, full_epoch, mass_kg, area_m2, cr)

    new_state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new_state[:3], new_state[3:], full_epoch


# ──────────────────────────────────────────────────────────────────────────
# Satellite / constellation convenience wrappers (metres <-> km at boundary)
# ──────────────────────────────────────────────────────────────────────────

def propagate_satellite(sat: "Satellite", dt_s: float, n_steps: int = 1,
                        mass_kg: float = DEFAULT_MASS_KG,
                        area_m2: float = DEFAULT_AREA_M2,
                        cr: float = DEFAULT_CR) -> "Satellite":
    """Advance *sat* in place by n_steps * dt_s seconds of RK4 integration.

    sat.r_eci_m / sat.v_eci_m_s (metres, metres/second) are converted to
    km / km*s^-1 for integration and converted back on return.
    """
    if sat.epoch is None:
        raise ValueError(f"Satellite {sat.name!r} has no epoch to propagate from")

    r_km = np.asarray(sat.r_eci_m, dtype=float) * 1e-3
    v_km_s = np.asarray(sat.v_eci_m_s, dtype=float) * 1e-3
    epoch = sat.epoch

    for _ in range(n_steps):
        r_km, v_km_s, epoch = rk4_step(r_km, v_km_s, epoch, dt_s, mass_kg, area_m2, cr)

    sat.r_eci_m = r_km * 1e3
    sat.v_eci_m_s = v_km_s * 1e3
    sat.epoch = epoch
    return sat


def propagate_constellation(constellation, dt_s: float, n_steps: int = 1,
                            mass_kg: float = DEFAULT_MASS_KG,
                            area_m2: float = DEFAULT_AREA_M2,
                            cr: float = DEFAULT_CR):
    """Advance every satellite in a TXConstellation/RXConstellation (anything
    exposing a .satellites list of Satellite) by n_steps * dt_s seconds.
    """
    for sat in constellation.satellites:
        propagate_satellite(sat, dt_s, n_steps, mass_kg, area_m2, cr)
    return constellation
