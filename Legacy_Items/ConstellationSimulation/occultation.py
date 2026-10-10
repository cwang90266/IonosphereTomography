"""
occultation.py

Radio-occultation (RO) link geometry for the GNSS-RO availability simulation.

At every time step, every TX (GNSS)-RX (LEO) pair is tested against the
Straight-Line Tangent Altitude (SLTA) geometric mask: the straight-line path
between the two satellites must have its point of closest approach to
Earth's centre fall strictly between the two satellites, and that point's
altitude above R_E must lie inside the ionospheric band [h_min, h_max].

Geometrically valid links are grouped across time into per TX-RX-pair
"arcs" using exactly the LEO/GNSS/tangent_alt_km/prn_id/leo_id/conid schema
that test_param_iono.py's forward-modeling pipeline
(_build_arc_rays / forward_model_arc / run_forward_models) already consumes
-- so a configuration flag on compute_occultation_availability() can hand
these arcs straight to run_forward_models() to simulate actual TEC
measurements along each SLTA-validated ray path.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from propagator import DEFAULT_AREA_M2, DEFAULT_CR, DEFAULT_MASS_KG, eci_to_ecef

# ── SLTA geometric-mask constants ─────────────────────────────────────────
R_E = 6378.137    # km, Earth's spherical radius for the geometric mask
H_MIN = 0.0       # km, solid-Earth boundary
H_MAX = 800.0     # km, top of the ionosphere


# ──────────────────────────────────────────────────────────────────────────
# SLTA line-of-sight geometry
# ──────────────────────────────────────────────────────────────────────────

def los_geometry(r_tx_km: np.ndarray, r_rx_km: np.ndarray):
    """Straight-Line Tangent Altitude (SLTA) line-of-sight geometry for one
    TX-RX pair, both position vectors expressed in the same frame (km),
    with Earth's centre at the origin.

        L      = r_rx - r_tx        line-of-sight chord vector (TX -> RX)
        u_hat  = L / |L|            unit vector along the line of sight
        d      = -(r_tx . u_hat)    distance from TX to closest approach
        r_t    = r_tx + d * u_hat   tangent-point position vector
        h_t    = |r_t| - R_E        tangent altitude

    Returns
    -------
    (L, u_hat, d, r_t, h_t) : (ndarray, ndarray, float, ndarray, float)
    """
    r_tx_km = np.asarray(r_tx_km, dtype=float)
    r_rx_km = np.asarray(r_rx_km, dtype=float)

    L = r_rx_km - r_tx_km
    L_norm = float(np.linalg.norm(L))
    u_hat = L / L_norm
    d = -float(np.dot(r_tx_km, u_hat))
    r_t = r_tx_km + d * u_hat
    h_t = float(np.linalg.norm(r_t)) - R_E
    return L, u_hat, d, r_t, h_t


def is_valid_occultation(L: np.ndarray, d: float, h_t: float,
                          h_min: float = H_MIN, h_max: float = H_MAX) -> bool:
    """SLTA validity mask: a link is a valid occultation candidate at this
    time step iff ALL of the following hold:

        0 < d < |L|
        h_t > h_min
        h_t < h_max
    """
    L_norm = float(np.linalg.norm(L))
    return bool(0.0 < d < L_norm and h_t > h_min and h_t < h_max)


def find_valid_links(tx_satellites: Sequence, rx_satellites: Sequence,
                      epoch: datetime) -> List[dict]:
    """Test every TX-RX satellite pair at one time step against the SLTA
    validity mask.

    Parameters
    ----------
    tx_satellites : sequence of tx_constellation.Satellite (e.g.
        TXConstellation.satellites)
    rx_satellites : sequence of rx_constellation.Satellite (e.g.
        RXConstellation.satellites)
    epoch : datetime
        Tag attached to every returned link (must match the epoch the
        satellites' state vectors are valid at).

    Returns
    -------
    list of dict, one per geometrically valid link, each with keys
    tx, rx, epoch, r_tx_km, r_rx_km, L, u_hat, d, r_t, h_t.
    """
    links: List[dict] = []
    for tx in tx_satellites:
        r_tx_km = np.asarray(tx.r_eci_m, dtype=float) * 1e-3
        for rx in rx_satellites:
            r_rx_km = np.asarray(rx.r_eci_m, dtype=float) * 1e-3
            L, u_hat, d, r_t, h_t = los_geometry(r_tx_km, r_rx_km)
            if is_valid_occultation(L, d, h_t):
                links.append(dict(
                    tx=tx, rx=rx, epoch=epoch,
                    r_tx_km=r_tx_km, r_rx_km=r_rx_km,
                    L=L, u_hat=u_hat, d=d, r_t=r_t, h_t=h_t,
                ))
    return links


# ──────────────────────────────────────────────────────────────────────────
# Time-stepped availability scan
# ──────────────────────────────────────────────────────────────────────────

def scan_availability(tx_constellation, rx_constellation,
                       dt_s: float, n_steps: int,
                       mass_kg: float = DEFAULT_MASS_KG,
                       area_m2: float = DEFAULT_AREA_M2,
                       cr: float = DEFAULT_CR) -> List[List[dict]]:
    """Propagate *tx_constellation* and *rx_constellation* forward by
    n_steps * dt_s seconds (RK4, see propagator.py), applying the SLTA
    validity mask at every epoch (including the starting epoch).

    Both constellations are advanced in place; call this once per scan.

    Returns
    -------
    list of length (n_steps + 1), each element the find_valid_links() result
    for that epoch.
    """
    from propagator import propagate_constellation

    epoch0 = tx_constellation.satellites[0].epoch
    if epoch0 is None:
        raise ValueError("TX constellation satellites have no epoch set")

    history: List[List[dict]] = [
        find_valid_links(tx_constellation.satellites, rx_constellation.satellites, epoch0)
    ]
    for _ in range(n_steps):
        propagate_constellation(tx_constellation, dt_s, 1, mass_kg, area_m2, cr)
        propagate_constellation(rx_constellation, dt_s, 1, mass_kg, area_m2, cr)
        epoch = tx_constellation.satellites[0].epoch
        history.append(
            find_valid_links(tx_constellation.satellites, rx_constellation.satellites, epoch)
        )
    return history


# ──────────────────────────────────────────────────────────────────────────
# Link -> arc aggregation (test_param_iono.py-compatible schema)
# ──────────────────────────────────────────────────────────────────────────

def _group_links_by_pair(
        link_history: Sequence[Sequence[dict]]) -> Dict[Tuple[str, str], List[dict]]:
    """Flatten scan_availability()'s per-epoch link_history into one
    time-ordered link list per (tx.name, rx.name) pair."""
    by_pair: Dict[Tuple[str, str], List[dict]] = {}
    for links in link_history:
        for link in links:
            key = (link["tx"].name, link["rx"].name)
            by_pair.setdefault(key, []).append(link)
    return by_pair


def segment_occultation_passes(
        link_history: Sequence[Sequence[dict]], dt_s: float,
        gap_tol: float = 1.5) -> List[List[dict]]:
    """Split each TX-RX pair's per-epoch valid-link list into separate
    continuous occultation passes.

    scan_availability() reports SLTA validity independently at every epoch,
    so a single physical occultation (satellites rise/set through the SLTA
    window over many consecutive epochs) shows up as many consecutive valid
    links for the same pair -- these must collapse to ONE event, not one per
    epoch. Conversely the same TX-RX pair can occult multiple, unrelated
    times over a simulation (successive orbital revolutions), and those
    separate passes must NOT be merged into a single arc spanning the gap
    between them. This splits on both: within a pair's time-ordered link
    list, consecutive links whose epochs are more than gap_tol * dt_s apart
    start a new segment.

    Returns
    -------
    list of list of dict -- each inner list is one continuous pass's link
    dicts, in epoch order.
    """
    by_pair = _group_links_by_pair(link_history)
    gap_s = gap_tol * dt_s

    segments: List[List[dict]] = []
    for link_list in by_pair.values():
        current = [link_list[0]]
        for prev, link in zip(link_list, link_list[1:]):
            if (link["epoch"] - prev["epoch"]).total_seconds() <= gap_s:
                current.append(link)
            else:
                segments.append(current)
                current = [link]
        segments.append(current)
    return segments


def build_occultation_arcs(link_history: Sequence[Sequence[dict]], dt_s: float,
                            gap_tol: float = 1.5) -> List[dict]:
    """Group per-epoch SLTA-valid links (from scan_availability) into
    per-continuous-pass arcs (see segment_occultation_passes), using exactly
    the arc-dict schema that test_param_iono.py's
    _build_arc_rays()/run_forward_models() expect:

        arc["GNSS"]           (3, n_epochs) ECEF km  -- TX positions
        arc["LEO"]            (3, n_epochs) ECEF km  -- RX positions
        arc["tangent_alt_km"] (n_epochs,)            -- SLTA h_t per epoch
        arc["prn_id"], arc["leo_id"], arc["conid"]    -- string identifiers

    A TX-RX pair that occults more than once over the simulation (separate
    orbital passes) yields one arc per pass, not one arc spanning the whole
    simulation.

    TX/RX state vectors are stored in whatever frame the constellation
    simulation propagates them in (pseudo-inertial ECI for Walker-generated
    satellites, see propagator.py); each link's positions are converted to
    ECEF at its own epoch via propagator.eci_to_ecef() since
    test_param_iono.py's pipeline is ECEF throughout.
    """
    segments = segment_occultation_passes(link_history, dt_s, gap_tol)

    arcs: List[dict] = []
    for link_list in segments:
        n = len(link_list)
        gnss_ecef = np.empty((3, n))
        leo_ecef = np.empty((3, n))
        tangent_alt_km = np.empty(n)

        for i, link in enumerate(link_list):
            r_tx_ecef, _ = eci_to_ecef(link["r_tx_km"], np.zeros(3), link["epoch"])
            r_rx_ecef, _ = eci_to_ecef(link["r_rx_km"], np.zeros(3), link["epoch"])
            gnss_ecef[:, i] = r_tx_ecef
            leo_ecef[:, i] = r_rx_ecef
            tangent_alt_km[i] = link["h_t"]

        prn_id = link_list[0]["tx"].name
        leo_id = link_list[0]["rx"].name
        conid = prn_id[0].upper() if prn_id[:1].upper() in "GREC" else "?"

        arcs.append(dict(
            GNSS=gnss_ecef, LEO=leo_ecef, tangent_alt_km=tangent_alt_km,
            prn_id=prn_id, leo_id=leo_id, conid=conid,
        ))
    return arcs


# ──────────────────────────────────────────────────────────────────────────
# Configuration-flag integration with test_param_iono.py
# ──────────────────────────────────────────────────────────────────────────

def _import_test_param_iono():
    """Import test_param_iono.py, which lives one directory above this
    module (the repository root, /home/austinhunter/IonosphereTomography),
    inserting the repo root onto sys.path first if needed."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import test_param_iono
    return test_param_iono


def simulate_measurements(arcs: List[dict], truth_state, model_state,
                           grid_lats_1deg: np.ndarray, grid_lons_1deg: np.ndarray,
                           grid_lats_5deg: np.ndarray, grid_lons_5deg: np.ndarray,
                           alt_grid: np.ndarray) -> Tuple[List[dict], List[dict]]:
    """Feed SLTA-valid occultation arcs (see build_occultation_arcs) into
    test_param_iono.run_forward_models() to forward-model simulated TEC
    measurements along each ray path.

    Parameters
    ----------
    arcs : list of dict, as returned by build_occultation_arcs().
    truth_state, model_state : Ionosphere_Tomography_Inverter.ionospheric_state.
        IonosphericState instances (see test_param_iono.build_truth_state /
        build_model_ensemble) the two ensembles were built on.
    grid_lats_1deg, grid_lons_1deg : truth (1-deg) grid *truth_state* is defined on.
    grid_lats_5deg, grid_lons_5deg : model (5-deg) grid *model_state* is defined on.
    alt_grid : altitude levels (km) both ensembles are defined on.

    Returns
    -------
    (truth_arcs, model_arcs) -- test_param_iono.run_forward_models()'s return
    value verbatim.
    """
    tpi = _import_test_param_iono()
    return tpi.run_forward_models(
        arcs, truth_state, model_state,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg, alt_grid,
    )


def compute_occultation_availability(
        tx_constellation, rx_constellation, dt_s: float, n_steps: int,
        simulate_measurements_flag: bool = False,
        truth_state=None, model_state=None,
        grid_lats_1deg: Optional[np.ndarray] = None,
        grid_lons_1deg: Optional[np.ndarray] = None,
        grid_lats_5deg: Optional[np.ndarray] = None,
        grid_lons_5deg: Optional[np.ndarray] = None,
        alt_grid: Optional[np.ndarray] = None,
        mass_kg: float = DEFAULT_MASS_KG, area_m2: float = DEFAULT_AREA_M2,
        cr: float = DEFAULT_CR,
):
    """End-to-end SLTA availability run: propagate both constellations for
    n_steps * dt_s seconds, apply the SLTA validity mask at every epoch, and
    group the resulting valid links into test_param_iono.py-compatible arcs.

    *simulate_measurements_flag* is the configuration flag gating the
    integration with test_param_iono.py: when True, the arcs are additionally
    passed to simulate_measurements() (-> run_forward_models()) to forward-
    model simulated TEC along each SLTA-validated link. truth_state,
    model_state and all of grid_lats_1deg/grid_lons_1deg/grid_lats_5deg/
    grid_lons_5deg/alt_grid are then required (see simulate_measurements()
    for what they are).

    Returns
    -------
    arcs                             if simulate_measurements_flag is False
    (arcs, truth_arcs, model_arcs)    if simulate_measurements_flag is True
    """
    link_history = scan_availability(
        tx_constellation, rx_constellation, dt_s, n_steps, mass_kg, area_m2, cr,
    )
    arcs = build_occultation_arcs(link_history, dt_s)

    if not simulate_measurements_flag:
        return arcs

    required = dict(
        truth_state=truth_state, model_state=model_state,
        grid_lats_1deg=grid_lats_1deg, grid_lons_1deg=grid_lons_1deg,
        grid_lats_5deg=grid_lats_5deg, grid_lons_5deg=grid_lons_5deg,
        alt_grid=alt_grid,
    )
    missing = [name for name, val in required.items() if val is None]
    if missing:
        raise ValueError(
            "simulate_measurements_flag=True requires: " + ", ".join(missing)
        )

    truth_arcs, model_arcs = simulate_measurements(
        arcs, truth_state, model_state,
        grid_lats_1deg, grid_lons_1deg, grid_lats_5deg, grid_lons_5deg, alt_grid,
    )
    return arcs, truth_arcs, model_arcs
