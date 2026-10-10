"""
tx_constellation.py

GNSS transmitter constellations for the GNSS-RO availability simulation.

Two ways to populate a TXConstellation:
  1. A synthetic Walker Star pattern (arbitrary sats/plane, planes,
     inclination, altitude) -- useful for "what if" constellation studies.
  2. Real broadcast-ephemeris state vectors read from a RINEX mixed
     navigation file (GPS + GLONASS + Galileo + BeiDou in one file), via
     the existing TEC_model.igs_tec_pipeline.BroadcastEphemeris parser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# WGS-84 / IS-GPS-200 constants (shared convention with TEC_model.igs_tec_pipeline)
MU_EARTH = 3.986004418e14   # m^3 s^-2, Earth gravitational parameter
R_EARTH = 6378137.0          # m, WGS-84 semi-major axis


@dataclass
class Satellite:
    """A single transmitter state vector at one epoch (Earth-centered inertial)."""

    name: str
    r_eci_m: np.ndarray            # (3,) ECI position, metres
    v_eci_m_s: np.ndarray          # (3,) ECI velocity, metres/second
    epoch: Optional[datetime] = None
    plane: Optional[int] = None
    slot: Optional[int] = None
    constellation: Optional[str] = None


def circular_state_vector(a_m: float, inclination_deg: float, raan_deg: float,
                           arg_lat_deg: float, mu: float = MU_EARTH):
    """ECI position/velocity of a circular orbit given semi-major axis *a_m*,
    inclination, RAAN, and argument of latitude (argument of perigee folded
    in, since e = 0 makes it undefined on its own).

    Returns
    -------
    (r_eci_m, v_eci_m_s) : tuple of (3,) np.ndarray
    """
    i = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    u = math.radians(arg_lat_deg)

    r_pf = np.array([a_m * math.cos(u), a_m * math.sin(u), 0.0])
    v_mag = math.sqrt(mu / a_m)
    v_pf = np.array([-v_mag * math.sin(u), v_mag * math.cos(u), 0.0])

    cO, sO = math.cos(raan), math.sin(raan)
    cI, sI = math.cos(i), math.sin(i)
    # Perifocal -> ECI rotation R = Rz(raan) @ Rx(inclination); argument of
    # perigee is folded into u already, so no third rotation is needed.
    R = np.array([
        [cO, -sO * cI,  sO * sI],
        [sO,  cO * cI, -cO * sI],
        [0.0,      sI,       cI],
    ])
    return R @ r_pf, R @ v_pf


def generate_walker_star(sats_per_plane: int, planes: int, inclination_deg: float,
                          altitude_km: float, phase_factor: int = 1,
                          raan_span_deg: float = 180.0, raan_offset_deg: float = 0.0,
                          epoch: Optional[datetime] = None,
                          name_prefix: str = "TX") -> List[Satellite]:
    """Generate a Walker Star constellation (i:sats_per_plane*planes/planes/phase_factor).

    Walker Star spreads orbital planes over *raan_span_deg* = 180 degrees
    (rather than the 360 degrees used by a Walker Delta pattern), which is
    the standard convention for near-polar constellations such as Iridium.

    Parameters
    ----------
    sats_per_plane : int
        Number of satellites evenly spaced within each orbital plane.
    planes : int
        Number of orbital planes, evenly spaced in RAAN across *raan_span_deg*.
    inclination_deg : float
        Orbital inclination shared by every plane (degrees).
    altitude_km : float
        Circular orbit altitude above the WGS-84 mean radius (km).
    phase_factor : int
        Walker phasing parameter F; shifts the in-plane satellite spacing
        of each successive plane by F * 360 / (planes * sats_per_plane) deg.
    raan_span_deg : float
        Total RAAN spread across all planes (180 for Walker Star, 360 for
        Walker Delta).
    raan_offset_deg : float
        Constant RAAN offset applied to every plane (lets a single-plane
        preset, e.g. Walker(n, 1, ...), be placed at an arbitrary RAAN).
    epoch : datetime, optional
        Epoch to tag each generated state vector with.
    name_prefix : str
        Prefix used to name each satellite, e.g. "TX_P00S03".

    Returns
    -------
    list[Satellite]
    """
    if sats_per_plane < 1 or planes < 1:
        raise ValueError("sats_per_plane and planes must each be >= 1")

    a_m = R_EARTH + altitude_km * 1e3
    raan_step = raan_span_deg / planes
    anomaly_step = 360.0 / sats_per_plane
    phase_step = phase_factor * 360.0 / (planes * sats_per_plane)

    satellites: List[Satellite] = []
    for p in range(planes):
        raan = raan_offset_deg + p * raan_step
        for s in range(sats_per_plane):
            u = (s * anomaly_step + p * phase_step) % 360.0
            r, v = circular_state_vector(a_m, inclination_deg, raan, u)
            satellites.append(Satellite(
                name=f"{name_prefix}_P{p:02d}S{s:02d}",
                r_eci_m=r, v_eci_m_s=v, epoch=epoch, plane=p, slot=s,
                constellation="walker_star",
            ))
    return satellites


class TXConstellation:
    """A collection of GNSS transmitter state vectors."""

    def __init__(self) -> None:
        self.satellites: List[Satellite] = []

    # ------------------------------------------------------------------
    def from_walker_star(self, sats_per_plane: int, planes: int,
                          inclination_deg: float, altitude_km: float,
                          phase_factor: int = 1, raan_span_deg: float = 180.0,
                          raan_offset_deg: float = 0.0,
                          epoch: Optional[datetime] = None,
                          name_prefix: str = "TX") -> "TXConstellation":
        """Populate this constellation with a synthetic Walker Star pattern.

        See :func:`generate_walker_star` for parameter definitions.
        """
        self.satellites = generate_walker_star(
            sats_per_plane=sats_per_plane, planes=planes,
            inclination_deg=inclination_deg, altitude_km=altitude_km,
            phase_factor=phase_factor, raan_span_deg=raan_span_deg,
            raan_offset_deg=raan_offset_deg, epoch=epoch, name_prefix=name_prefix,
        )
        return self

    # ------------------------------------------------------------------
    def from_rinex_nav(self, nav_path: str | Path, epoch: Optional[datetime] = None,
                        constellations: str = "GREC",
                        velocity_dt_s: float = 1.0) -> "TXConstellation":
        """Instantiate TX constellation state vectors from a RINEX mixed
        navigation file (RINEX-3 files that combine GPS/GLONASS/Galileo/
        BeiDou broadcast messages in a single file are commonly labelled
        "MN" -- mixed nav).

        Parameters
        ----------
        nav_path : str or Path
            Path to the RINEX 2/3 navigation file (may be .gz compressed).
        epoch : datetime, optional
            UTC epoch at which to evaluate every satellite's state vector.
            Defaults to the current UTC time.
        constellations : str
            RINEX single-letter system codes to keep, e.g. "GREC" for
            GPS/Galileo/BeiDou/GLONASS ("G"=GPS, "R"=GLONASS, "E"=Galileo,
            "C"=BeiDou).
        velocity_dt_s : float
            Half-width (seconds) of the central-difference step used to
            derive ECEF velocity from the broadcast position propagator
            (which only yields position directly).

        Returns
        -------
        self, with self.satellites populated (positions/velocities are in
        ECEF at *epoch*, not ECI -- broadcast ephemerides are computed in
        the ECEF/PZ-90 frames used by TEC_model.igs_tec_pipeline).
        """
        # Reuse the broadcast-ephemeris Kepler/PZ-90 propagators already
        # validated by the IGS ground-station TEC pipeline instead of
        # duplicating GPS/GLONASS/Galileo/BeiDou orbit mechanics here.
        from TEC_model.igs_tec_pipeline import BroadcastEphemeris, _utc_to_gps_sow

        if epoch is None:
            epoch = datetime.now(timezone.utc)
        t_ep = pd.Timestamp(epoch)
        if t_ep.tzinfo is None:
            t_ep = t_ep.tz_localize("UTC")

        ephem = BroadcastEphemeris(nav_path)

        def _t_arg(sv: str, t: pd.Timestamp) -> float:
            # GLONASS ephemerides are referenced to UTC seconds-of-day;
            # every other constellation uses GPS seconds-of-week.
            if sv.startswith("R"):
                return float(t.hour * 3600 + t.minute * 60 + t.second)
            return _utc_to_gps_sow(t)

        satellites: List[Satellite] = []
        for sv in sorted(ephem._cache.keys()):
            if sv[:1] not in constellations:
                continue

            dt = pd.Timedelta(seconds=velocity_dt_s)
            r0 = ephem.sv_position_km(sv, _t_arg(sv, t_ep - dt))
            r1 = ephem.sv_position_km(sv, _t_arg(sv, t_ep + dt))
            r_mid = ephem.sv_position_km(sv, _t_arg(sv, t_ep))
            if r0 is None or r1 is None or r_mid is None:
                continue

            r_ecef_m = r_mid * 1e3
            v_ecef_m_s = (r1 - r0) * 1e3 / (2.0 * velocity_dt_s)

            satellites.append(Satellite(
                name=sv, r_eci_m=r_ecef_m, v_eci_m_s=v_ecef_m_s, epoch=epoch,
                constellation={"G": "GPS", "R": "GLONASS", "E": "Galileo",
                               "C": "BeiDou", "J": "QZSS", "S": "SBAS"}.get(sv[0], sv[0]),
            ))

        self.satellites = satellites
        return self
