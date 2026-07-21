"""
rx_constellation.py

LEO receiver (occultation-observing) constellations for the GNSS-RO
availability simulation: the PlanetiQ baseline preset, arbitrary
one-off LEO additions, and synthetic Walker LEO shells.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from tx_constellation import (
    MU_EARTH, R_EARTH, Satellite, circular_state_vector, generate_walker_star,
)

ScalarOrSequence = Union[float, Sequence[float]]


def _broadcast(value: ScalarOrSequence, n: int, name: str) -> np.ndarray:
    """Return a length-n float array: a scalar is repeated, a sequence is
    validated to already have length n."""
    if np.isscalar(value):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or a length-{n} sequence")
    return arr


class RXConstellation:
    """A collection of LEO receiver state vectors."""

    def __init__(self) -> None:
        self.satellites: List[Satellite] = []

    # ------------------------------------------------------------------
    def add_planetiq_baseline(self, raan_deg: Sequence[float] = (0.0, 5.0, 180.0),
                               reverse: Sequence[bool] = (False, False, True),
                               inclination_deg: float = 97.86,
                               altitude_km: float = 620.0,
                               arg_lat_deg: float = 90.0,
                               epoch: Optional[datetime] = None,
                               name_prefix: str = "PlanetiQ") -> "RXConstellation":
        """Add the baseline PlanetiQ constellation: 3 operational satellites
        in a 620 km, 97.86 deg sun-synchronous near-circular orbit -- but,
        unlike a single shared plane phased 120 deg apart, each satellite
        gets its own orbital plane (one RAAN each, default 0/5/180 deg) and
        all three share the same *arg_lat_deg* (default 90 deg, i.e. near
        the pole) at *epoch*, so they pass over the poles simultaneously
        rather than being spread out in time.

        One satellite per *reverse* (default: only the RAAN=180 one) has
        its velocity vector negated so it travels the opposite way around
        its orbital plane. This is a physically valid state vector -- a
        circular 2-body orbit is time-reversible, so simply flipping v
        yields a satellite retracing the same great circle the other
        direction -- giving genuinely counter-rotating geometry instead of
        merely a relabelled ascending node.
        """
        n = len(raan_deg)
        if len(reverse) != n:
            raise ValueError("reverse must have the same length as raan_deg")

        a_m = R_EARTH + altitude_km * 1e3
        sats: List[Satellite] = []
        for k, (raan, rev) in enumerate(zip(raan_deg, reverse)):
            r, v = circular_state_vector(a_m, inclination_deg, raan, arg_lat_deg)
            if rev:
                v = -v
            sats.append(Satellite(
                name=f"{name_prefix}_{k:02d}", r_eci_m=r, v_eci_m_s=v,
                epoch=epoch, plane=k, slot=0, constellation="PlanetiQ",
            ))
        self.satellites.extend(sats)
        return self

    # ------------------------------------------------------------------
    def add_arbitrary_leo(self, n: int, altitude_km: ScalarOrSequence,
                           inclination_deg: ScalarOrSequence,
                           raan_deg: ScalarOrSequence = 0.0,
                           mean_anomaly_deg: Optional[ScalarOrSequence] = None,
                           epoch: Optional[datetime] = None,
                           name_prefix: str = "LEO") -> "RXConstellation":
        """Add *n* arbitrary additional LEO satellites.

        Each orbital element (*altitude_km*, *inclination_deg*, *raan_deg*,
        *mean_anomaly_deg*) may be a single scalar shared by all *n*
        satellites, or a length-*n* sequence giving each satellite its own
        value. If *mean_anomaly_deg* is omitted, satellites are evenly
        phased (360/n degrees apart).
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        alt = _broadcast(altitude_km, n, "altitude_km")
        inc = _broadcast(inclination_deg, n, "inclination_deg")
        raan = _broadcast(raan_deg, n, "raan_deg")
        if mean_anomaly_deg is None:
            anomaly = np.linspace(0.0, 360.0, n, endpoint=False)
        else:
            anomaly = _broadcast(mean_anomaly_deg, n, "mean_anomaly_deg")

        sats: List[Satellite] = []
        for k in range(n):
            a_m = R_EARTH + alt[k] * 1e3
            r, v = circular_state_vector(a_m, inc[k], raan[k], anomaly[k])
            sats.append(Satellite(
                name=f"{name_prefix}_{k:02d}", r_eci_m=r, v_eci_m_s=v,
                epoch=epoch, slot=k, constellation="arbitrary_leo",
            ))
        self.satellites.extend(sats)
        return self

    # ------------------------------------------------------------------
    def add_walker_constellation(self, sats_per_plane: int, planes: int,
                                  inclination_deg: float, altitude_km: float,
                                  phase_factor: int = 1, walker_type: str = "star",
                                  raan_offset_deg: float = 0.0,
                                  epoch: Optional[datetime] = None,
                                  name_prefix: str = "LEOW") -> "RXConstellation":
        """Add one synthetic Walker LEO shell to this constellation.

        Parameters
        ----------
        walker_type : {"star", "delta"}
            "star" spreads orbital planes over 180 deg RAAN (e.g. polar/
            near-polar LEO shells); "delta" spreads them over the full
            360 deg (e.g. inclined-plane shells such as Starlink).
        See :func:`tx_constellation.generate_walker_star` for the remaining
        parameters.
        """
        if walker_type not in ("star", "delta"):
            raise ValueError("walker_type must be 'star' or 'delta'")
        raan_span_deg = 180.0 if walker_type == "star" else 360.0

        sats = generate_walker_star(
            sats_per_plane=sats_per_plane, planes=planes,
            inclination_deg=inclination_deg, altitude_km=altitude_km,
            phase_factor=phase_factor, raan_span_deg=raan_span_deg,
            raan_offset_deg=raan_offset_deg, epoch=epoch, name_prefix=name_prefix,
        )
        for s in sats:
            s.constellation = f"walker_{walker_type}"
        self.satellites.extend(sats)
        return self

    # ------------------------------------------------------------------
    def add_walker_constellations(self, configs: Sequence[Dict]) -> "RXConstellation":
        """Add multiple Walker LEO shells at once (e.g. a multi-shell LEO
        constellation study). Each dict in *configs* is forwarded as
        keyword arguments to :meth:`add_walker_constellation`.
        """
        for cfg in configs:
            self.add_walker_constellation(**cfg)
        return self
