#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
compare_polar_vs_inclined_availability.py

Run ConstellationSimulation twice -- once with just the 3-satellite PlanetiQ
polar (97.86 deg) baseline, once with that baseline plus one extra satellite
on the 53 deg-inclined orbit (--rx-add-extra-sat) -- and save each run's
global occultation-density figure with the SAME colorbar scale per radius
panel, so the two are visually comparable.

demo_occultation_availability.py itself works from real podTc2 mission data
and has no notion of "3 polar satellites" / "53 deg inclined orbit" -- those
are ConstellationSimulation/main.py RX-constellation options, so that is
what this drives (main.generate_output(), which produces the
demo_occultation_availability-style figures from simulated data).

Usage
-----
  tools/compare_polar_vs_inclined_availability.py \
      --start-time 2025-06-03T00:00:00 --end-time 2025-06-03T23:59:00
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ConstellationSimulation"))

import pandas as pd

import availability as av
import main as sim_main
import occultation as occ


def _run(cfg: sim_main.SimulationConfig, tx):
    """Like sim_main.run_simulation(), but takes an already-built TX
    constellation (deep-copied by the caller) instead of re-parsing the
    RINEX broadcast ephemeris -- that parse alone costs ~6 min/~1.7 GB for
    the full "GREC" multi-GNSS constellation, and it is identical across
    runs that only vary the RX constellation, so paying it twice for a
    two-run comparison is wasted time/memory.
    """
    rx = sim_main.build_rx_constellation(cfg)
    link_history = occ.scan_availability(
        tx, rx, dt_s=cfg.time_step_s, n_steps=cfg.n_steps,
        mass_kg=cfg.mass_kg, area_m2=cfg.area_m2, cr=cfg.cr,
    )
    events = av.events_from_occultation_links(link_history, dt_s=cfg.time_step_s)
    df = av.build_occultation_dataframe(events)
    return df


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-time", type=str, default="2025-06-03T00:00:00")
    p.add_argument("--end-time", type=str, default="2025-06-03T23:59:00")
    p.add_argument("--time-step", type=float, default=60.0)
    p.add_argument("--output-dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent
                               / "Figures" / "Simulated_Occultation_Availability"
                               / "polar_vs_inclined"))
    args = p.parse_args(argv)

    cfg_polar = sim_main.config_from_args(
        [f"--start-time={args.start_time}", f"--end-time={args.end_time}",
         f"--time-step={args.time_step}",
         "--output-dir", str(Path(args.output_dir) / "3_polar")])

    cfg_polar_plus_inclined = sim_main.config_from_args(
        [f"--start-time={args.start_time}", f"--end-time={args.end_time}",
         f"--time-step={args.time_step}",
         "--rx-add-extra-sat",
         "--output-dir", str(Path(args.output_dir) / "3_polar_plus_53deg")])

    print("Building shared TX constellation once (both runs share start_time/tx settings) ...")
    tx_base = sim_main.build_tx_constellation(cfg_polar)
    print(f"  {len(tx_base.satellites)} TX satellites")

    print("=" * 70)
    print("Run 1/2: 3 polar (97.86 deg) satellites only")
    print("=" * 70)
    df_polar = _run(cfg_polar, copy.deepcopy(tx_base))

    print("\n" + "=" * 70)
    print("Run 2/2: 3 polar satellites + 1 extra 53 deg-inclined satellite")
    print("=" * 70)
    df_inclined = _run(cfg_polar_plus_inclined, copy.deepcopy(tx_base))

    radii_km = (500.0, 1500.0, 2500.0)
    vmax_by_radius = {}
    for df in (df_polar, df_inclined):
        if df.empty:
            continue
        _glat, _glon, peaks = av.compute_global_density_peaks(df, pd.Timestamp(cfg_polar.start_time),
                                                                radii_km=radii_km)
        for r, peak in peaks.items():
            vmax_by_radius[r] = max(vmax_by_radius.get(r, 0.0), float(peak.max()))
    print(f"\nShared colorbar vmax per radius: {vmax_by_radius}")

    print("\n--- Run 1/2 output (3 polar) ---")
    _save(cfg_polar, df_polar, vmax_by_radius)

    print("\n--- Run 2/2 output (3 polar + 53 deg) ---")
    _save(cfg_polar_plus_inclined, df_inclined, vmax_by_radius)


def _save(cfg: sim_main.SimulationConfig, df: pd.DataFrame, vmax_by_radius: dict) -> None:
    if df.empty:
        print("  No occultation events -- nothing to plot.")
        return

    day = pd.Timestamp(cfg.start_time).normalize()
    n_in_roi = int(df["in_roi"].sum())
    print(f"  Total occultation events detected  : {len(df)}")
    print(f"  Total events within ROI             : {n_in_roi}")

    rx_for_tracks = sim_main.build_rx_constellation(cfg)
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
    print(f"  Saved global density figure -> {global_save_path}")

    if n_in_roi:
        save_path = cfg.output_dir / f"simulated_occultation_availability_{day.date()}.png"
        av.plot_simulated_availability(df, day, save_path=save_path)


if __name__ == "__main__":
    main()
