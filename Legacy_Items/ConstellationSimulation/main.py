"""
main.py

Top-level orchestrator for the GNSS-RO data-availability simulation. Wires
together every other module in this package into one time-stepped run:

    tx_constellation / rx_constellation  -- constellation initialisation at t=0
    propagator                           -- RK4 orbit propagation (2-body + J2 + J3 + SRP)
    occultation                          -- per-epoch SLTA geometric mask + arc aggregation
                                             (optionally: test_param_iono.py measurement physics)
    availability                         -- translation to the
                                             demo_occultation_availability.py analysis/plotting
                                             pipeline

Usage
-----
    python main.py                                    # uses RINEX defaults (2025-06-03)
    python main.py --tx-mode walker --rx-mode planetiq --time-step 30 \
        --start-time 2026-01-01T00:00:00 --end-time 2026-01-02T00:00:00  # synthetic TX
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from tx_constellation import TXConstellation
from rx_constellation import RXConstellation
from propagator import (
    DEFAULT_AREA_M2, DEFAULT_CR, DEFAULT_MASS_KG, constellation_ecef_to_eci,
)
import occultation as occ
import availability as av


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Top-level configuration for one end-to-end availability run. Every
    field here is also exposed as a CLI flag by build_arg_parser()/
    config_from_args() below.
    """

    # Time loop
    start_time: datetime = datetime(2025, 6, 3, tzinfo=timezone.utc)
    end_time: datetime = datetime(2025, 6, 3, 23, 59, tzinfo=timezone.utc)
    time_step_s: float = 60.0

    # TX constellation
    tx_mode: str = "rinex"          # "walker" | "rinex"
    tx_sats_per_plane: int = 6
    tx_planes: int = 6
    tx_inclination_deg: float = 55.0
    tx_altitude_km: float = 20200.0
    tx_rinex_path: Optional[str] = None  # Auto-download if not provided
    tx_rinex_constellations: str = "GREC"

    # RX constellation
    rx_mode: str = "planetiq"        # "planetiq" | "custom"
    rx_custom_n: int = 6
    rx_custom_altitude_km: float = 550.0
    rx_custom_inclination_deg: float = 53.0

    # Optional single extra RX satellite (on top of the PlanetiQ baseline,
    # independent of rx_mode) -- e.g. an ISS-inclination cubesat.
    rx_add_extra_sat: bool = False
    rx_extra_altitude_km: float = 590.0
    rx_extra_inclination_deg: float = 53.0
    rx_extra_raan_deg: float = 0.0

    # Ground-track panel (compute_rx_ground_tracks / _draw_ground_tracks)
    ground_track_hours: float = 2.0

    # Ionosphere measurement simulation (configuration flag -> test_param_iono.py)
    simulate_iono: bool = False

    # Propagator (SRP/cannonball) parameters, shared by TX and RX
    mass_kg: float = DEFAULT_MASS_KG
    area_m2: float = DEFAULT_AREA_M2
    cr: float = DEFAULT_CR

    # Output
    output_dir: Path = (
        Path(__file__).resolve().parent.parent
        / "Figures" / "Simulated_Occultation_Availability"
    )

    @property
    def n_steps(self) -> int:
        """Number of RK4 propagation steps of time_step_s covering
        [start_time, end_time]."""
        total_s = (self.end_time - self.start_time).total_seconds()
        return max(int(round(total_s / self.time_step_s)), 0)


def build_arg_parser() -> argparse.ArgumentParser:
    d = SimulationConfig()
    p = argparse.ArgumentParser(
        description="Orchestrate the GNSS-RO constellation / occultation "
                    "availability simulation.")

    g_time = p.add_argument_group("time loop")
    g_time.add_argument("--start-time", type=str, default=d.start_time.isoformat(),
                        help="ISO-8601 UTC simulation start (default: %(default)s)")
    g_time.add_argument("--end-time", type=str, default=d.end_time.isoformat(),
                        help="ISO-8601 UTC simulation end (default: %(default)s)")
    g_time.add_argument("--time-step", type=float, default=d.time_step_s,
                        dest="time_step_s",
                        help="Orbit-propagation time step in seconds "
                             "(default: %(default)s)")

    g_tx = p.add_argument_group("TX constellation")
    g_tx.add_argument("--tx-mode", choices=["walker", "rinex"], default=d.tx_mode)
    g_tx.add_argument("--tx-sats-per-plane", type=int, default=d.tx_sats_per_plane)
    g_tx.add_argument("--tx-planes", type=int, default=d.tx_planes)
    g_tx.add_argument("--tx-inclination-deg", type=float, default=d.tx_inclination_deg)
    g_tx.add_argument("--tx-altitude-km", type=float, default=d.tx_altitude_km)
    g_tx.add_argument("--tx-rinex-path", type=str, default=d.tx_rinex_path,
                      help="RINEX mixed-nav file path (optional; auto-downloads from "
                           "CDDIS if not provided when --tx-mode rinex)")
    g_tx.add_argument("--tx-rinex-constellations", type=str,
                      default=d.tx_rinex_constellations)

    g_rx = p.add_argument_group("RX constellation")
    g_rx.add_argument("--rx-mode", choices=["planetiq", "custom"], default=d.rx_mode,
                      help="'planetiq': the 3-satellite baseline preset only; "
                           "'custom': PlanetiQ baseline + n additional LEOs")
    g_rx.add_argument("--rx-custom-n", type=int, default=d.rx_custom_n,
                      help="Number of additional arbitrary LEO satellites "
                           "added on top of the PlanetiQ baseline when "
                           "--rx-mode custom")
    g_rx.add_argument("--rx-custom-altitude-km", type=float,
                      default=d.rx_custom_altitude_km)
    g_rx.add_argument("--rx-custom-inclination-deg", type=float,
                      default=d.rx_custom_inclination_deg)
    g_rx.add_argument("--rx-add-extra-sat", action="store_true",
                      default=d.rx_add_extra_sat,
                      help="Add one extra RX satellite on top of the "
                           "PlanetiQ baseline, independent of --rx-mode "
                           "(orbital elements set via --rx-extra-*)")
    g_rx.add_argument("--rx-extra-altitude-km", type=float,
                      default=d.rx_extra_altitude_km)
    g_rx.add_argument("--rx-extra-inclination-deg", type=float,
                      default=d.rx_extra_inclination_deg)
    g_rx.add_argument("--rx-extra-raan-deg", type=float,
                      default=d.rx_extra_raan_deg,
                      help="RAAN (deg) of the optional extra RX satellite "
                           "(default: %(default)s)")

    g_track = p.add_argument_group("ground-track panel")
    g_track.add_argument("--ground-track-hours", type=float,
                         default=d.ground_track_hours,
                         help="Length (hours) of the RX ground-track panel "
                              "appended to the global/regional figures "
                              "(default: %(default)s)")

    g_iono = p.add_argument_group("ionosphere measurement simulation")
    g_iono.add_argument("--simulate-iono", action="store_true", default=d.simulate_iono,
                        help="Run every valid occultation arc through "
                             "test_param_iono.py's IRI-based forward model "
                             "(builds IRI truth/model ensembles -- slow).")

    g_srp = p.add_argument_group("propagator (SRP) parameters")
    g_srp.add_argument("--mass-kg", type=float, default=d.mass_kg)
    g_srp.add_argument("--area-m2", type=float, default=d.area_m2)
    g_srp.add_argument("--cr", type=float, default=d.cr)

    p.add_argument("--output-dir", type=str, default=str(d.output_dir))
    return p


def config_from_args(argv: Optional[List[str]] = None) -> SimulationConfig:
    args = build_arg_parser().parse_args(argv)

    def _parse_dt(s: str) -> datetime:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    return SimulationConfig(
        start_time=_parse_dt(args.start_time),
        end_time=_parse_dt(args.end_time),
        time_step_s=args.time_step_s,
        tx_mode=args.tx_mode,
        tx_sats_per_plane=args.tx_sats_per_plane,
        tx_planes=args.tx_planes,
        tx_inclination_deg=args.tx_inclination_deg,
        tx_altitude_km=args.tx_altitude_km,
        tx_rinex_path=args.tx_rinex_path,
        tx_rinex_constellations=args.tx_rinex_constellations,
        rx_mode=args.rx_mode,
        rx_custom_n=args.rx_custom_n,
        rx_custom_altitude_km=args.rx_custom_altitude_km,
        rx_custom_inclination_deg=args.rx_custom_inclination_deg,
        rx_add_extra_sat=args.rx_add_extra_sat,
        rx_extra_altitude_km=args.rx_extra_altitude_km,
        rx_extra_inclination_deg=args.rx_extra_inclination_deg,
        rx_extra_raan_deg=args.rx_extra_raan_deg,
        ground_track_hours=args.ground_track_hours,
        simulate_iono=args.simulate_iono,
        mass_kg=args.mass_kg,
        area_m2=args.area_m2,
        cr=args.cr,
        output_dir=Path(args.output_dir),
    )


# ──────────────────────────────────────────────────────────────────────────
# Constellation initialisation (t = start_time)
# ──────────────────────────────────────────────────────────────────────────

def build_tx_constellation(cfg: SimulationConfig) -> TXConstellation:
    tx = TXConstellation()
    if cfg.tx_mode == "walker":
        tx.from_walker_star(
            sats_per_plane=cfg.tx_sats_per_plane, planes=cfg.tx_planes,
            inclination_deg=cfg.tx_inclination_deg, altitude_km=cfg.tx_altitude_km,
            epoch=cfg.start_time, name_prefix="TX",
        )
    elif cfg.tx_mode == "rinex":
        # Ensure repo root is on sys.path for TEC_model imports.
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # Auto-download BRDC file if path not provided.
        if cfg.tx_rinex_path:
            rinex_path = cfg.tx_rinex_path
        else:
            brdc_cache = repo_root / "Data" / "BRDC_Daily"
            rinex_path = str(_download_brdc_file(cfg.start_time, brdc_cache))

        tx.from_rinex_nav(
            rinex_path, epoch=cfg.start_time,
            constellations=cfg.tx_rinex_constellations,
        )
        # Broadcast-ephemeris state vectors come out in ECEF; the propagator
        # (2-body/J2/J3/SRP) requires a pseudo-inertial frame.
        constellation_ecef_to_eci(tx)
    else:
        raise ValueError(f"Unknown tx_mode: {cfg.tx_mode!r}")
    return tx


def build_rx_constellation(cfg: SimulationConfig) -> RXConstellation:
    rx = RXConstellation()
    rx.add_planetiq_baseline(epoch=cfg.start_time)
    if cfg.rx_mode == "custom":
        rx.add_arbitrary_leo(
            n=cfg.rx_custom_n, altitude_km=cfg.rx_custom_altitude_km,
            inclination_deg=cfg.rx_custom_inclination_deg,
            epoch=cfg.start_time, name_prefix="LEO",
        )
    elif cfg.rx_mode != "planetiq":
        raise ValueError(f"Unknown rx_mode: {cfg.rx_mode!r}")

    if cfg.rx_add_extra_sat:
        rx.add_arbitrary_leo(
            n=1, altitude_km=cfg.rx_extra_altitude_km,
            inclination_deg=cfg.rx_extra_inclination_deg,
            raan_deg=cfg.rx_extra_raan_deg,
            epoch=cfg.start_time, name_prefix="ExtraLEO",
        )
    return rx


# ──────────────────────────────────────────────────────────────────────────
# BRDC (broadcast ephemeris) download from CDDIS
# ──────────────────────────────────────────────────────────────────────────

def _download_brdc_file(epoch: datetime, cache_dir: Path) -> Path:
    """Download the combined multi-GNSS BRDC (broadcast ephemeris) file for
    *epoch*'s day from CDDIS, reusing the same authenticated
    TEC_model.igs_tec_pipeline.RinexDownloader every other demo script in
    this repo already uses (RinexDownloader.nav_file()) -- it reads NASA
    Earthdata Login credentials from ~/.netrc (machine
    urs.earthdata.nasa.gov), tries the multi-GNSS mixed-nav product in
    CDDIS's /{yy}p/ directory first (BRDM/BRDC..._MN.rnx.gz), then falls
    back to the GPS-only IGS consolidated brdc{doy}0.{yy}n.gz in /{yy}n/.

    Falls back to any BRDC file already present under Data/ if the CDDIS
    download itself fails (e.g. no network, no ~/.netrc entry yet).

    Parameters
    ----------
    epoch : datetime
        Simulation start time (used to determine which day's BRDC file).
    cache_dir : Path
        Directory to cache downloaded (and decompressed) files.

    Returns
    -------
    Path to the (decompressed) BRDC RINEX nav file.

    Raises
    ------
    RuntimeError if both download fails and no fallback local file is found.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from TEC_model.igs_tec_pipeline import RinexDownloader

    downloader = RinexDownloader(cache_dir=cache_dir)
    try:
        # `station` only affects the GPS-only /{yy}n/ fallback pattern -- the
        # preferred /{yy}p/ mixed-nav product is station-independent.
        return downloader.nav_file(station="BRDC", date=epoch)
    except Exception as e:
        print(f"  CDDIS download failed: {e}")
        print(f"  Looking for fallback BRDC files in Data/...")
        data_root = repo_root / "Data"
        brdc_files = sorted(
            list(data_root.glob("**/brdc*.[0-9][0-9]n"))
            + list(data_root.glob("**/BRD[CM]*.rnx*"))
        )
        if brdc_files:
            fallback = brdc_files[0]
            print(f"    Using fallback: {fallback}")
            return fallback

        raise RuntimeError(
            f"Failed to download BRDC for {epoch.date()} and no local fallback found"
        ) from e


# ──────────────────────────────────────────────────────────────────────────
# Ionosphere measurement simulation (test_param_iono.py bridge)
# ──────────────────────────────────────────────────────────────────────────

def _simulate_iono_measurements(cfg: SimulationConfig,
                                 arcs: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Build the IRI truth/model ensembles test_param_iono.run_forward_models()
    needs, then forward-model simulated TEC along every SLTA-validated arc.

    Mirrors test_param_iono.py's own window driver exactly: Fibonacci grids
    are built from the arcs' own tangent-point tracks (via
    test_param_iono._arc_tangent_tracks / _make_fibonacci_grid), a 1-deg IRI
    grid becomes the "truth" ensemble, a 5-deg IRI grid becomes the
    stochastic "model" ensemble, and both feed
    occultation.simulate_measurements() -> run_forward_models().
    """
    print("\nsimulate_iono=True: building IRI truth/model ensembles ...")
    tpi = occ._import_test_param_iono()

    tp_lats_all, tp_lons_all = tpi._arc_tangent_tracks(arcs)
    time_dt = pd.Timestamp(cfg.start_time)

    grid_lats_1deg, grid_lons_1deg = tpi._make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=1.0)
    grid_lats_5deg, grid_lons_5deg = tpi._make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=5.0)
    print(f"  1-deg grid: {len(grid_lats_1deg)} nodes, "
          f"5-deg grid: {len(grid_lats_5deg)} nodes")

    mean_1deg, _ = tpi.build_iri_state_grid_cached(
        time_dt, grid_lats_1deg, grid_lons_1deg, tpi.ALT_GRID,
        spacing_deg=1.0,
        lat_min=float(grid_lats_1deg.min()), lat_max=float(grid_lats_1deg.max()),
        lon_min=float(grid_lons_1deg.min()), lon_max=float(grid_lons_1deg.max()),
    )
    truth_state = tpi.build_truth_state(mean_1deg)
    print(f"  Truth ensemble: {truth_state.n_members} members x "
          f"{truth_state.n_grid_points} grid points")

    mean_5deg, _ = tpi.build_iri_state_grid_cached(
        time_dt, grid_lats_5deg, grid_lons_5deg, tpi.ALT_GRID,
        spacing_deg=5.0,
        lat_min=float(grid_lats_5deg.min()), lat_max=float(grid_lats_5deg.max()),
        lon_min=float(grid_lons_5deg.min()), lon_max=float(grid_lons_5deg.max()),
    )
    model_state = tpi.build_model_ensemble(
        mean_5deg, grid_lats_5deg, grid_lons_5deg,
        n_members=tpi.N_MEMBERS, corr_length_km=tpi.CORR_LENGTH_KM,
    )
    print(f"  Model ensemble: {model_state.n_members} members x "
          f"{model_state.n_grid_points} grid points")

    print("  Running test_param_iono.run_forward_models() ...")
    truth_arcs, model_arcs = occ.simulate_measurements(
        arcs, truth_state, model_state,
        grid_lats_1deg, grid_lons_1deg, grid_lats_5deg, grid_lons_5deg, tpi.ALT_GRID,
    )
    print(f"  Forward-modeled TEC for {len(truth_arcs)} arcs")
    return truth_arcs, model_arcs


# ──────────────────────────────────────────────────────────────────────────
# Orchestration loop
# ──────────────────────────────────────────────────────────────────────────

def run_simulation(cfg: SimulationConfig):
    """Initialise both constellations, propagate them across
    [cfg.start_time, cfg.end_time] in cfg.time_step_s steps, apply the SLTA
    geometric mask at every step, and (optionally) run the resulting arcs
    through test_param_iono.py's measurement physics.

    Returns
    -------
    (events, arcs, truth_arcs, model_arcs) -- events is a flat list of
    availability.OccultationEvent (one per continuous occultation pass, not
    per epoch); arcs is the test_param_iono.py-compatible per-pass list; truth_arcs /
    model_arcs are None unless cfg.simulate_iono is True.
    """
    print(f"Building TX constellation (mode={cfg.tx_mode!r}) ...")
    tx = build_tx_constellation(cfg)
    print(f"  {len(tx.satellites)} TX satellites")

    print(f"Building RX constellation (mode={cfg.rx_mode!r}) ...")
    rx = build_rx_constellation(cfg)
    print(f"  {len(rx.satellites)} RX satellites")

    n_steps = cfg.n_steps
    print(f"Propagating {cfg.start_time.isoformat()} -> {cfg.end_time.isoformat()} "
          f"({n_steps} steps @ {cfg.time_step_s:g} s) and scanning SLTA geometry ...")
    link_history = occ.scan_availability(
        tx, rx, dt_s=cfg.time_step_s, n_steps=n_steps,
        mass_kg=cfg.mass_kg, area_m2=cfg.area_m2, cr=cfg.cr,
    )
    n_links = sum(len(links) for links in link_history)
    print(f"  {n_links} valid occultation links across {len(link_history)} epochs")

    events = av.events_from_occultation_links(link_history, dt_s=cfg.time_step_s)
    arcs = occ.build_occultation_arcs(link_history, dt_s=cfg.time_step_s)
    print(f"  Grouped into {len(arcs)} continuous occultation passes")

    truth_arcs = model_arcs = None
    if cfg.simulate_iono:
        if arcs:
            truth_arcs, model_arcs = _simulate_iono_measurements(cfg, arcs)
        else:
            print("\nsimulate_iono=True but no valid arcs were found -- skipping.")

    return events, arcs, truth_arcs, model_arcs


# ──────────────────────────────────────────────────────────────────────────
# Output generation
# ──────────────────────────────────────────────────────────────────────────

def generate_output(cfg: SimulationConfig, events: List["av.OccultationEvent"],
                    vmax_by_radius: Optional[dict] = None):
    """Translate the recorded occultation events into the
    demo_occultation_availability.py DataFrame schema, save the 2x2 regional
    summary figure, and print a console availability summary.

    Parameters
    ----------
    vmax_by_radius : dict, optional
        Forwarded to av.plot_global_occultation_density() so the global
        density figure's colorbars can be pinned to a fixed scale shared
        across separate runs (e.g. comparing two RX constellations) instead
        of each auto-scaling to its own peak.

    Returns
    -------
    (df, fig) -- the built DataFrame and the matplotlib Figure
    (fig is None if there were no events to plot).
    """
    df = av.build_occultation_dataframe(events)

    print("\n" + "=" * 70)
    print("Occultation availability summary")
    print("=" * 70)
    print(f"  Total occultation events detected  : {len(df)}")

    if df.empty:
        print("  No occultation events -- nothing to plot.")
        print("=" * 70)
        return df, None

    n_in_roi = int(df["in_roi"].sum())
    print(f"  Total events within ROI             : {n_in_roi}")

    day = pd.Timestamp(cfg.start_time).normalize()

    # Fresh, independent RX constellation copy purely for ground-track
    # visualization -- scan_availability() (run_simulation) hides its own
    # propagation loop and does not retain per-epoch RX positions, so the
    # ground track is computed here as a separate propagation instead.
    rx_for_tracks = build_rx_constellation(cfg)
    ground_tracks = av.compute_rx_ground_tracks(
        rx_for_tracks, start_time=cfg.start_time,
        duration_hours=cfg.ground_track_hours,
        mass_kg=cfg.mass_kg, area_m2=cfg.area_m2, cr=cfg.cr,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    global_save_path = cfg.output_dir / f"global_occultation_density_{day.date()}.png"
    av.plot_global_occultation_density(
        df, day, save_path=global_save_path, ground_tracks=ground_tracks,
        vmax_by_radius=vmax_by_radius,
    )
    print(f"  Saved global density figure → {global_save_path}")

    doa = av._import_demo_occultation_availability()
    if n_in_roi:
        _, counts = doa.rolling_window_count(
            df.loc[df["in_roi"], "tecmax_time"], day, window_hours=1.0,
        )
        peak_per_hour = int(counts.max()) if counts.size else 0
    else:
        peak_per_hour = 0
    print(f"  Peak in-ROI occultations / 1 h window: {peak_per_hour}")
    print("=" * 70)

    if n_in_roi == 0:
        # plot_occultation_availability's per-satellite ROI histogram panel
        # assumes at least one in-ROI occultation; nothing to plot otherwise.
        print("  No in-ROI occultations -- skipping regional availability figure.")
        return df, None

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = cfg.output_dir / f"simulated_occultation_availability_{day.date()}.png"
    fig = av.plot_simulated_availability(df, day, save_path=save_path)

    return df, fig


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    cfg = config_from_args(argv)
    events, arcs, truth_arcs, model_arcs = run_simulation(cfg)
    generate_output(cfg, events)


if __name__ == "__main__":
    main()
