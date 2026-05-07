#!/usr/bin/env python3
"""
demo.py — End-to-end demonstration of the IonosphereTomography codebase.

Region of interest: 60–65 °N, 150–140 °W (high-latitude Alaska)

Setup (one time):
    pipenv install        # creates venv and installs from Pipfile
    pipenv run python demo.py

Or, with any venv that has the packages installed:
    python demo.py

Sections
--------
 §1  Single IRI altitude profile at the region centre (default solar indices)
 §2  Three solar-activity scenarios compared on one altitude profile
 §3  Sensitivity to ionospheric shape parameters (foF2, hmF2, B0, B1)
 §4  24-hour time profile at a fixed location
 §5  Latitude sweep 60→65 °N across the Alaska region
 §6  Rectangular triangular mesh: generate and plot
 §7  Polar-cap triangular mesh (60°N → North Pole): generate and plot
 §8  EDPSamples: populate a rectangular Alaska mesh with IRI across 3 solar scenarios
 §9  Bundled solar-index data: read and visualise apf107 / ig_rz
§10  IRI_Sample_Inputs: download live data and build a quantile-sample DataFrame
     (requires internet access; skipped gracefully if unavailable)
ADDITIONAL SECTIONS NOT UP TO DATE
"""

import sys
import importlib.resources as impr
from pathlib import Path

ROOT = Path(__file__).parent
# Make EDPSamples and IRI_Sample_Inputs importable (they are not installed packages)
sys.path.insert(0, str(ROOT))
# locate_in_mesh.py lives inside a directory with a space in its name
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.parser import parse

# ── iri2020 package (editable install from iri2020_new/src) ─────────────────
from iri2020 import IRI, timeprofile, geoprofile
import iri2020.plots as iri_plots
from iri2020.get_iri_inputs import read_apf107, read_ig_rz

# ── EDPSamples utilities ─────────────────────────────────────────────────────
from EDPSamples.edp_samples import (
    EDPSamples,
    plot_tri_mesh,
    plot_polar_mesh,
    interp_heights,
)
from EDPSamples.generate_rect_tri_mesh import generate_rect_tri_mesh
from EDPSamples.generate_polar_mesh import generate_ploar_mesh   # note: typo in source

# Standalone spherical point-in-triangle algorithm (no WGS84 altitude bug)
from locate_in_mesh import find_containing_triangles as find_triangle_sphere

# ── IRI_Sample_Inputs ────────────────────────────────────────────────────────
from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs



# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

TIME     = "2025-06-15 04:00"      # simulation UTC time
LAT_C    = 62.5                    # centre latitude  (°N)
LON_C    = -145.0                  # centre longitude (°E, = 145 °W)
ALT_KM   = [80, 700, 1]         # altitude grid: [start, stop, step] km

# Alaska region bounds (used for sweeps and mesh generation)
LAT_MIN, LAT_MAX, DLAT = 60.0, 65.0, 1.5
LON_MIN, LON_MAX, DLON = -150.0, -140.0, 1.5

# Three solar-activity scenarios spanning the realistic range for Alaska
SCENARIOS = [
    {"label": "Low activity",  "f107D":  70, "ap":  3, "IG12":  20, "Rz12":  15},
    {"label": "Nominal",       "f107D": 120, "ap": 10, "IG12":  60, "Rz12":  50},
    {"label": "High activity", "f107D": 200, "ap": 30, "IG12": 100, "Rz12":  90},
]
COLORS = ["royalblue", "forestgreen", "firebrick"]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# =============================================================================
# def _iri_time_sweep(times, altkmrange, glat, glon, **solar_kw) -> xr.Dataset:
#     """
#     Run IRI for each datetime in `times` and concatenate along the time
#     dimension. Implements the core of vprofile.timeprofile without the
#     iri.f107 attribute bug present in that function.
#     """
#     iono = xr.Dataset()
#     for t in times:
#         iri = IRI(t, altkmrange, glat, glon, **solar_kw)
#         iono = iri if not iono else xr.concat([iono, iri], dim="time")
#     return iono
# 
# 
# def _iri_lat_sweep(lats, altkm, glon, time, **solar_kw) -> xr.Dataset:
#     """
#     Run IRI at a single altitude for each latitude and concatenate along glat.
#     Implements the core of vprofile.geoprofile without the iri.f107 bug.
#     """
#     iono = xr.Dataset()
#     for lat in lats:
#         iri = IRI(time, [altkm] * 3, lat, glon, **solar_kw)
#         iono = iri if not iono else xr.concat([iono, iri], dim="glat")
#     return iono
# 
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  §1  Single IRI altitude profile at the region centre
# ─────────────────────────────────────────────────────────────────────────────

def section1() -> xr.Dataset:
    _banner("§1  Single IRI altitude profile (default solar indices)")

    iono = IRI(TIME, ALT_KM, LAT_C, LON_C)

    peak_idx = int(iono["ne"].argmax())
    print(f"  ne peak  : {float(iono['ne'][peak_idx]):.3e} m⁻³"
          f"  at {float(iono.alt_km[peak_idx]):.0f} km")
    print(f"  NmF2     : {float(iono['NmF2']):.3e} m⁻³")
    print(f"  hmF2     : {float(iono['hmF2']):.1f} km")
    print(f"  foF2     : {float(iono['foF2']):.2f} MHz")
    print(f"  TEC      : {float(iono['TEC']):.3e} m⁻²")
    print(f"  B0 / B1  : {float(iono['B0']):.1f} km / {float(iono['B1']):.3f}")
    print(f"  f107D    : {iono.attrs['f107D']:.1f}   ap : {iono.attrs['ap']:.1f}")
    print(f"  IG12     : {iono.attrs['IG12']:.1f}   Rz12: {iono.attrs['Rz12']:.1f}")

    iri_plots.altprofile(iono)
    plt.gcf().suptitle(
        f"§1  Single altitude profile — {TIME}\n"
        f"lat {LAT_C} °N   lon {LON_C} °E   (IRI default solar indices)",
        fontsize=9,
    )
    return iono


# ─────────────────────────────────────────────────────────────────────────────
#  §2  Three solar-activity scenarios compared on one altitude profile
# ─────────────────────────────────────────────────────────────────────────────

def section2() -> list[xr.Dataset]:
    _banner("§2  Three solar-activity scenarios (3 IRI calls)")

    results = []
    for sc in SCENARIOS:
        print(f"  {sc['label']:16s}  f107D={sc['f107D']:3d}  ap={sc['ap']:2d}"
              f"  IG12={sc['IG12']:3d}  Rz12={sc['Rz12']:2d}", end="  →  ")
        iono = IRI(TIME, ALT_KM, LAT_C, LON_C,
                   f107D=sc["f107D"], ap=sc["ap"],
                   IG12=sc["IG12"], Rz12=sc["Rz12"])
        print(f"NmF2={float(iono['NmF2']):.2e} m⁻³  "
              f"hmF2={float(iono['hmF2']):.0f} km  "
              f"TEC={float(iono['TEC']):.2e} m⁻²")
        results.append(iono)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"§2  Solar scenario comparison\n"
        f"lat {LAT_C} °N   lon {LON_C} °E   {TIME}"
    )

    for iono, sc, color in zip(results, SCENARIOS, COLORS):
        alt = iono.alt_km.values
        axes[0].plot(iono["ne"].values,  alt, color=color, label=sc["label"])
        axes[1].plot(iono["Ti"].values,  alt, color=color, label=sc["label"])
        axes[1].plot(iono["Te"].values,  alt, color=color, ls="--", lw=1)
        axes[2].plot(iono["nO+"].values, alt, color=color, label=sc["label"])

    axes[0].set_xlabel("Electron density ne (m⁻³)")
    axes[1].set_xlabel("Ion temp Ti  /  Electron temp Te (K)  [dashed]")
    axes[2].set_xlabel("O⁺ density (m⁻³)")
    for ax in axes:
        ax.set_ylabel("Altitude (km)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    axes[1].set_xscale("linear")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  §3  Sensitivity to ionospheric shape parameters (foF2, hmF2, B0, B1)
# ─────────────────────────────────────────────────────────────────────────────

def section3() -> None:
    _banner("§3  Sensitivity to shape parameters (foF2, hmF2, B0, B1)")
    print("  1 nominal + 4 × 10 perturbed IRI calls …")

    iono_nom = IRI(TIME, ALT_KM, LAT_C, LON_C)
    nom = {
        "foF2": float(iono_nom["foF2"]),
        "hmF2": float(iono_nom["hmF2"]),
        "B0":   float(iono_nom["B0"]),
        "B1":   float(iono_nom["B1"]),
    }
    print(f"  Nominal: foF2={nom['foF2']:.2f} MHz  hmF2={nom['hmF2']:.0f} km"
          f"  B0={nom['B0']:.1f} km  B1={nom['B1']:.3f}")

    factors = np.linspace(0.5, 2.0, 10)

    def sweep(vary_key: str) -> list[xr.Dataset]:
        out = []
        for f in factors:
            kw = dict(nom)
            kw[vary_key] = kw[vary_key] * f
            out.append(IRI(TIME, ALT_KM, LAT_C, LON_C, **kw))
        return out

    results_foF2 = sweep("foF2")
    results_hmF2 = sweep("hmF2")
    results_B0   = sweep("B0")
    results_B1   = sweep("B1")

    # iri_plots.altprofile_sensitivity expects iParam_Switch=1 for shape params
    iri_plots.altprofile_sensitivity(
        1, iono_nom,
        results_foF2, results_hmF2,
        results_B0, results_B1,
    )
    plt.gcf().suptitle(
        f"§3  Shape-parameter sensitivity\n"
        f"lat {LAT_C} °N   lon {LON_C} °E   {TIME}",
        fontsize=9,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  §4  24-hour time profile at a fixed location
# ─────────────────────────────────────────────────────────────────────────────

def section4() -> xr.Dataset:
    _banner("§4  24-hour time profile  (lat 62.5 °N  lon -145 °E)")
    print("  Running 12 IRI calls (one per 2 h) …")

    iono = timeprofile(
        ("2022-06-15 00:00", "2022-06-16 00:00"),
        timedelta(hours=0.5),
        ALT_KM, LAT_C, LON_C,)
    print(f"  Dataset dims: {dict(iono.sizes)}")

    iri_plots.timeprofile(iono)
    # timeprofile creates two figures; label the most recently opened pair
    open_figs = plt.get_fignums()
    for fn in open_figs[-2:]:
        plt.figure(fn).suptitle(
            f"§4  Time profile   lat {LAT_C} °N   lon {LON_C} °E",
            fontsize=9,
        )
    return iono


# ─────────────────────────────────────────────────────────────────────────────
#  §5  Latitude sweep 60→65 °N across the Alaska region
# ─────────────────────────────────────────────────────────────────────────────

def section5() -> xr.Dataset:
    _banner("§5  Latitude sweep  60→65 °N  at 300 km,  lon -145 °E")
    print("  Running IRI calls at lats = [60.0, 62.5, 65.0] …")

    iono = geoprofile(
        latrange=(LAT_MIN, LAT_MAX + 0.1, DLAT),
        glon=LON_C,
        altkm=300.0,
        time=TIME,)

    iri_plots.latprofile(iono)
    plt.gcf().suptitle(
        f"§5  Latitude profile   lon {LON_C} °E   {TIME}",
        fontsize=9,
    )
    return iono


# ─────────────────────────────────────────────────────────────────────────────
#  §6  Rectangular triangular mesh: generate and plot
# ─────────────────────────────────────────────────────────────────────────────

def section6() -> tuple[np.ndarray, np.ndarray]:
    _banner("§6  Rectangular triangular mesh  (Alaska region)")
    from EDPSamples.plot_mesh_globe import plot_globe_occultation_mesh
    # generate_rect_tri_mesh(minLat, maxLat, dLat, minLon, maxLon, dLon)
    # Returns vertices as (N, 2) columns = (longitude, latitude)
    vertices, triangles = generate_rect_tri_mesh(
        LAT_MIN, LAT_MAX, DLAT,
        LON_MIN, LON_MAX, DLON,
    )
    print(f"  Vertices : {vertices.shape[0]}")
    print(f"  Triangles: {triangles.shape[0]}")
    print(f"  First 3 vertices (lon, lat):\n{vertices[:3]}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tri_mesh(vertices, triangles, ax=ax)
    ax.set_title(
        f"§6  Rectangular mesh   lat {LAT_MIN}–{LAT_MAX} °N   "
        f"lon {LON_MIN}–{LON_MAX} °E\n"
        f"dLat={DLAT}°   dLon={DLON}°   →   "
        f"{vertices.shape[0]} vertices,  {triangles.shape[0]} triangles"
    )
    save_path = "./Figures/Examples/Alaska_region.png"
    
    print("Running plotting code...")
    plot_globe_occultation_mesh(vertices, triangles, LAT_C, LON_C, save_path)
    print("Complete\n")
    return vertices, triangles


# ─────────────────────────────────────────────────────────────────────────────
#  §7  Polar-cap triangular mesh: generate and plot
# ─────────────────────────────────────────────────────────────────────────────

def section7() -> tuple[np.ndarray, np.ndarray]:
    _banner("§7  Polar-cap mesh   60 °N → North Pole")

    # generate_ploar_mesh is the name in the source (typo: "ploar")
    vertices, triangles = generate_ploar_mesh("north", minLat=60.0, dLat=1.0)
    vertices  = np.array(vertices)
    triangles = np.array(triangles)
    print(f"  Vertices : {vertices.shape[0]}")
    print(f"  Triangles: {len(triangles)}")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    plot_polar_mesh(vertices, triangles, pole="north", ax=ax)
    ax.set_title("§7  Polar-cap mesh   60 °N → North Pole", va="bottom")
    return vertices, triangles


# ─────────────────────────────────────────────────────────────────────────────
#  §8  EDPSamples: fill the Alaska rectangle with IRI for 3 solar scenarios
# ─────────────────────────────────────────────────────────────────────────────

def section8(rect_vertices: np.ndarray, rect_triangles: np.ndarray) -> EDPSamples:
    _banner("§8  EDPSamples  —  Alaska rect mesh × 3 solar scenarios")

    # Altitude grid (1-D numpy array)
    alt_grid = np.arange(ALT_KM[0], ALT_KM[1] + 1, ALT_KM[2], dtype=float)
    n_height = len(alt_grid)

    # sampling_parameters DataFrame: one row per solar scenario
    # Columns must be: hour, f107, ap, ig12, rz12
    sampling_df = pd.DataFrame([
        {
            "hour": 12.0,
            "f107": float(sc["f107D"]),
            "ap":   float(sc["ap"]),
            "ig12": float(sc["IG12"]),
            "rz12": float(sc["Rz12"]),
        }
        for sc in SCENARIOS
    ])
    n_sample = len(sampling_df)
    n_geo    = rect_vertices.shape[0]

    print(f"  Altitude levels : {n_height}  ({alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km)")
    print(f"  Mesh vertices   : {n_geo}")
    print(f"  Solar scenarios : {n_sample}")
    print(f"  Total IRI calls : {n_geo * n_sample}")

    # ── Run IRI for every vertex × scenario, fill edps array ─────────────────
    # edps shape: (height, geo_vertex, sample)
    edps = np.full((n_height, n_geo, n_sample), np.nan)

    for i_s, sc in enumerate(SCENARIOS):
        for i_g in range(n_geo):
            lon_v, lat_v = rect_vertices[i_g]      # vertices are (lon, lat)
            # print(f"  scenario {i_s+1}/{n_sample}  "
            #       f"vertex {i_g+1:2d}/{n_geo}  "
            #       f"({lat_v:.1f} °N, {lon_v:.1f} °E)",
            #       end="\r", flush=True)
            iono = IRI(
                TIME, ALT_KM, lat_v, lon_v,
                f107D=sc["f107D"], ap=sc["ap"],
                IG12=sc["IG12"], Rz12=sc["Rz12"],
            )
            edps[:, i_g, i_s] = iono["ne"].values
    print()  # clear the \r progress line

    # ── Construct EDPSamples ─────────────────────────────────────────────────
    # Passing a pre-computed edps array bypasses the evaluate_iri / hardcoded-
    # path code path in __init__.  The DIM_VERTEX patch (applied at import
    # time above) prevents the AttributeError from the missing class attribute.
    eds = EDPSamples(
        DateTime=TIME,
        geo_type="Rectangle",
        altitude=alt_grid,
        sampling_parameters=sampling_df,
        edps=edps,
        minLon=LON_MIN, maxLon=LON_MAX, dLon=DLON,
        minLat=LAT_MIN, maxLat=LAT_MAX, dLat=DLAT,
    )

    print(f"  EDPSamples dims  : {dict(eds.sizes)}")
    print(f"  EDPs shape       : {eds.edps.shape}   (height, geo_vertex, sample)")
    print(f"  Sampling params  :\n{sampling_df.to_string(index=False)}")

    # ── Plot: ne altitude profiles by solar scenario ──────────────────────────
    fig, axes = plt.subplots(1, n_sample, figsize=(5 * n_sample, 7), sharey=True)
    fig.suptitle(
        f"§8  EDPSamples — ne profiles per solar scenario\n"
        f"Alaska rect mesh  ({n_geo} vertices)   {TIME}"
    )

    # Vertex closest to the centre of the region
    dists    = np.hypot(rect_vertices[:, 0] - LON_C, rect_vertices[:, 1] - LAT_C)
    i_centre = int(np.argmin(dists))

    for i_s, (sc, ax, color) in enumerate(zip(SCENARIOS, axes, COLORS)):
        for i_g in range(n_geo):
            ax.plot(edps[:, i_g, i_s], alt_grid, color="lightgray", lw=0.7)
        ax.plot(
            edps[:, i_centre, i_s], alt_grid,
            color=color, lw=2.5,
            label=(f"centre  ({rect_vertices[i_centre,1]:.1f} °N, "
                   f"{rect_vertices[i_centre,0]:.1f} °E)"),
        )
        ax.set_xscale("log")
        ax.set_xlabel("ne (m⁻³)")
        ax.set_title(sc["label"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("Altitude (km)")

    # ── Save / load roundtrip ─────────────────────────────────────────────────
    ncpath = ROOT / "alaska_edp_demo.nc"
    try:
        eds.saveNetCDF(ncpath)
        print(f"\n  Saved  → {ncpath.name}  ({ncpath.stat().st_size // 1024} KB)")
        eds_loaded = EDPSamples.fromNetCDF(ncpath)
        print(f"  Loaded   EDPSamples dims: {dict(eds_loaded.sizes)}")
    except Exception as exc:
        print(f"\n  NetCDF roundtrip skipped: {exc}")

    # ── Demonstrate interp_heights ────────────────────────────────────────────
    query_alts = np.array([120.0, 250.0, 350.0, 450.0])
    idx, w = interp_heights(alt_grid, query_alts)
    print(f"\n  interp_heights — query altitudes: {query_alts}")
    print(f"    bracket indices : {idx}")
    print(f"    lower weights   : {w[:, 0].round(3)}")
    print(f"    upper weights   : {w[:, 1].round(3)}")

    # ── Demonstrate find_containing_triangles (spherical version) ─────────────
    # geolocation passed as (lat, lon) — the convention of locate_in_mesh.py
    geoloc_latlon = rect_vertices[:, [1, 0]]   # swap (lon,lat) → (lat,lon)
    query_pts = np.array([
        [62.5, -145.0],   # region centre
        [61.0, -148.0],   # SW
        [64.0, -142.0],   # NE
    ])
    tri_idx, bary = find_triangle_sphere(
        query_pts, geoloc_latlon, rect_triangles, return_bary=True
    )
    print(f"\n  find_containing_triangles ({len(query_pts)} queries):")
    for q, ti, b in zip(query_pts, tri_idx, bary):
        print(f"    ({q[0]:.1f} °N, {q[1]:.1f} °E)  →  triangle {ti}"
              f"   bary = {b.round(3)}")

    return eds


# ─────────────────────────────────────────────────────────────────────────────
#  §9  Bundled solar-index data files: read_apf107, read_ig_rz, show_iri_inputs
# ─────────────────────────────────────────────────────────────────────────────

def section9() -> None:
    _banner("§9  Bundled solar-index data  (read_apf107, read_ig_rz)")

    with impr.as_file(impr.files("iri2020").joinpath("data")) as data_path:
        apf107 = read_apf107(str(data_path))
        ig_rz  = read_ig_rz(str(data_path))

    n_days = len(apf107["yr"])
    print(f"  apf107 : {n_days} daily records  "
          f"({apf107['yr'][0]}–{apf107['yr'][-1]})")
    print(f"  F10.7  range : {min(apf107['f107']):.1f} – {max(apf107['f107']):.1f} sfu")
    print(f"  Ap     range : {min(apf107['iapda'])} – {max(apf107['iapda'])}")
    print(f"  ig_rz  : {len(ig_rz['ig'])} monthly IG records  /  "
          f"{len(ig_rz['rz'])} Rz records")
    print(f"  IG12   range : {min(ig_rz['ig']):.1f} – {max(ig_rz['ig']):.1f}")
    print(f"  Rz12   range : {min(ig_rz['rz']):.1f} – {max(ig_rz['rz']):.1f}")

    from iri2020.get_iri_inputs import show_iri_inputs
    show_iri_inputs(apf107, ig_rz)
    plt.gcf().suptitle("§9  Bundled IRI solar-index time histories", fontsize=16)


# ─────────────────────────────────────────────────────────────────────────────
# §10  IRI_Sample_Inputs: download live data + quantile-sample DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def section10() -> None:
    _banner("§10  IRI_Sample_Inputs  (downloads live apf107 + ig_rz)")
    print("  Connecting to chain-new.chain-project.net …")
    try:
        inp = IRI_Sample_Inputs(2022, 6, 15, 12, 0, 0)
        print(f"  apf107  : {len(inp.apf107['yr'])} records  "
              f"({inp.apf107['yr'][0]}–{inp.apf107['yr'][-1]})")
        print(f"  ig_rz   : {len(inp.ig_rz['ig'])} IG values")

        # Build a sample plan: ±81 days of f107, ±27 days of ap
        samples = inp.quantileSamples(
            f107_sample_range=81,
            ap_sample_range=27,
        )
        print(f"  Sample plan: {len(samples)} rows × {len(samples.columns)} params")
        print(f"  Columns: {list(samples.columns)}")
        print(samples.describe().round(2).to_string())

        # Show the live time-history plot
        from IRI_Sample_Inputs.IRI_Sample_inputs import show_iri_inputs
        show_iri_inputs(inp.apf107, inp.ig_rz)
        plt.gcf().suptitle("§10  Live IRI input parameter time histories", fontsize=16)

    except Exception as exc:
        print(f"  Skipped — {type(exc).__name__}: {exc}")
        print("  (No network, server unreachable, or date not in downloaded data)")


# ─────────────────────────────────────────────────────────────────────────────
#  §11  Point-mode EDPSamples: quantile ensemble + NetCDF roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def section11() -> None:
    _banner("§11  Point-mode EDPSamples: quantile ensemble + NetCDF roundtrip")

    # 2003-11-21 is four days after the Halloween geomagnetic storm peak
    # (solar cycle 23 maximum), which makes the wide ig/rz sample ranges
    # physically meaningful.
    POINT_TIME = "2003-11-21T12"

    print("  Fetching live apf107 / ig_rz for 2003-11-21 …")
    try:
        inp = IRI_Sample_Inputs(2003, 11, 21, 12, 0, 0)
    except Exception as exc:
        print(f"  Skipped — {type(exc).__name__}: {exc}")
        print("  (No network or server unreachable)")
        return

    # Build quantile sample plan across all five axes.
    # NOTE: quantileSamples() has a known bug — it uses f107_range instead of
    # ig12_range in the IG12 loop, so the ig12 column values may be incorrect
    # until that bug is fixed in IRI_Sample_inputs.py.
    samples = inp.quantileSamples(
        hour_sample_range=6,
        ap_sample_range=31,
        f107_sample_range=31,
        ig_sample_range=48,
        rz_sample_range=48,
    )
    print(f"  Sample plan: {len(samples)} rows × {len(samples.columns)} columns")
    print(f"  Columns: {list(samples.columns)}")
    print(f"  → {len(samples)} IRI calls will be made internally")

    alt_grid = np.arange(80, 501, 10, dtype=float)   # 80–500 km, 43 levels
    print(f"  Altitude grid: {alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km "
          f"({len(alt_grid)} levels)")

    # evaluate_iri=1 tells EDPSamples to call IRI internally for each sample
    # row. This code path uses internal path construction that may fail outside
    # the original developer's environment; fall back to manual filling if so.
    eds = None
    try:
        eds = EDPSamples(
            DateTime=POINT_TIME,
            geo_type="Point",
            altitude=alt_grid,
            sampling_parameters=samples,
            evaluate_iri=1,
            Lon=LON_C,
            Lat=LAT_C,
        )
        print(f"  EDPSamples (internal IRI) dims: {dict(eds.sizes)}")
    except Exception as exc:
        print(f"  evaluate_iri=1 path failed ({type(exc).__name__}: {exc})")
        print("  Falling back to manual IRI loop …")

    if eds is None:
        # Manual fallback: loop over sample rows and call IRI() directly,
        # mirroring the approach used in §8.
        edps = np.full((len(alt_grid), 1, len(samples)), np.nan)
        for i_s, row in samples.iterrows():
            print(f"  sample {i_s+1}/{len(samples)}", end="\r", flush=True)
            iono = IRI(
                POINT_TIME,
                [float(alt_grid[0]), float(alt_grid[-1]), float(alt_grid[1] - alt_grid[0])],
                LAT_C, LON_C,
                f107D=float(row["f107"]),
                ap=float(row["ap"]),
                IG12=float(row["ig12"]),
                Rz12=float(row["rz12"]),
            )
            edps[:, 0, i_s] = iono["ne"].values
        print()
        eds = EDPSamples(
            DateTime=POINT_TIME,
            geo_type="Point",
            altitude=alt_grid,
            sampling_parameters=samples,
            edps=edps,
            Lon=LON_C,
            Lat=LAT_C,
        )
        print(f"  EDPSamples (manual fallback) dims: {dict(eds.sizes)}")

    # ── NetCDF roundtrip ──────────────────────────────────────────────────────
    ncpath = ROOT / "point_ensemble_demo.nc"
    eds.saveNetCDF(ncpath)
    print(f"\n  Saved  → {ncpath.name}  ({ncpath.stat().st_size // 1024} KB)")

    eds_loaded = EDPSamples.fromNetCDF(ncpath)
    print(f"  Loaded   EDPSamples dims: {dict(eds_loaded.sizes)}")

    match = np.allclose(
        eds.edps.values,
        eds_loaded.edps.values,
        equal_nan=True,
    )
    print(f"  Roundtrip data integrity : {'PASS ✓' if match else 'FAIL ✗'}")

    # ── Plot: ensemble spread of ne profiles at the single point ──────────────
    edps_vals = eds.edps.values[:, 0, :]   # shape (height, sample)

    fig, ax = plt.subplots(figsize=(7, 8))
    for i_s in range(edps_vals.shape[1]):
        ax.plot(edps_vals[:, i_s], alt_grid, color="lightsteelblue", lw=0.5, alpha=0.6)

    ne_median = np.nanmedian(edps_vals, axis=1)
    ne_p10    = np.nanpercentile(edps_vals, 10, axis=1)
    ne_p90    = np.nanpercentile(edps_vals, 90, axis=1)

    ax.fill_betweenx(alt_grid, ne_p10, ne_p90, alpha=0.25,
                     color="steelblue", label="10th–90th percentile")
    ax.plot(ne_median, alt_grid, color="steelblue", lw=2.5, label="Median")

    ax.set_xscale("log")
    ax.set_xlabel("Electron density ne (m⁻³)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        f"§11  ne ensemble spread at ({LAT_C} °N, {LON_C} °E)\n"
        f"{POINT_TIME}   {edps_vals.shape[1]} quantile samples"
    )
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    
# ─────────────────────────────────────────────────────────────────────────────
# §12  Occultation Defined Grid Points
# ─────────────────────────────────────────────────────────────────────────────
def section12() -> tuple[np.ndarray, np.ndarray]:
    from EDPSamples.generate_occultation_tri_mesh import generate_occultation_mesh
    from EDPSamples.plot_mesh_globe import plot_globe_occultation_mesh
    from TEC_model.podTc_file_processing import parse_podTc2_nc_file
    _banner("§12 Occultation Defined Grid Points")
    
    
    # podTc2_string = "podTc2_GN05.2025.152.06.07.0026.C33.00_0000.0001_nc"
    podTc2_string = "podTc2_GN05.2025.152.06.09.0026.C21.01_0000.0001_nc"
    podTc2_file = f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.152/{podTc2_string}"
    
    
    save_path = f"./Figures/Examples/{podTc2_string}_mesh_geometry.png"
    podTc_data = parse_podTc2_nc_file(podTc2_file)
    
    print("Generating Occultation Mesh...")
    vertices_podTc, triangles_podTc, pt1, pt2, pt3 = generate_occultation_mesh(filename=podTc2_file,dLat=5,dLon=5)
    print("Complete\n")
    
    print("Running plotting code...")
    plot_globe_occultation_mesh(vertices_podTc, triangles_podTc, podTc_data['lat_tecmax_tangent'], podTc_data['lon_tecmax_tangent'], save_path)
    print("Complete\n")
    
    return vertices_podTc, triangles_podTc, pt1, pt2, pt3
    
def section13(rect_vertices: np.ndarray, rect_triangles: np.ndarray, pt1: tuple, pt2: tuple, pt3: tuple) -> EDPSamples:
    _banner("§8  EDPSamples  —  Alaska rect mesh × 3 solar scenarios")

    # Altitude grid (1-D numpy array)
    alt_grid = np.arange(ALT_KM[0], ALT_KM[1] + 1, ALT_KM[2], dtype=float)
    n_height = len(alt_grid)

    # sampling_parameters DataFrame: one row per solar scenario
    # Columns must be: hour, f107, ap, ig12, rz12
    sampling_df = pd.DataFrame([
        {
            "hour": 12.0,
            "f107": float(sc["f107D"]),
            "ap":   float(sc["ap"]),
            "ig12": float(sc["IG12"]),
            "rz12": float(sc["Rz12"]),
        }
        for sc in SCENARIOS
    ])
    n_sample = len(sampling_df)
    n_geo    = rect_vertices.shape[0]

    print(f"  Altitude levels : {n_height}  ({alt_grid[0]:.0f}–{alt_grid[-1]:.0f} km)")
    print(f"  Mesh vertices   : {n_geo}")
    print(f"  Solar scenarios : {n_sample}")
    print(f"  Total IRI calls : {n_geo * n_sample}")

    # ── Run IRI for every vertex × scenario, fill edps array ─────────────────
    # edps shape: (height, geo_vertex, sample)
    edps = np.full((n_height, n_geo, n_sample), np.nan)

    for i_s, sc in enumerate(SCENARIOS):
        for i_g in range(n_geo):
            lon_v, lat_v = rect_vertices[i_g]      # vertices are (lon, lat)
            # print(f"  scenario {i_s+1}/{n_sample}  "
            #       f"vertex {i_g+1:2d}/{n_geo}  "
            #       f"({lat_v:.1f} °N, {lon_v:.1f} °E)",
            #       end="\r", flush=True)
            iono = IRI(
                TIME, ALT_KM, lat_v, lon_v,
                f107D=sc["f107D"], ap=sc["ap"],
                IG12=sc["IG12"], Rz12=sc["Rz12"],
            )
            edps[:, i_g, i_s] = iono["ne"].values
    print()  # clear the \r progress line

    # ── Construct EDPSamples ─────────────────────────────────────────────────
    # Passing a pre-computed edps array bypasses the evaluate_iri / hardcoded-
    # path code path in __init__.  The DIM_VERTEX patch (applied at import
    # time above) prevents the AttributeError from the missing class attribute.
    eds = EDPSamples(
        DateTime=TIME,
        geo_type="Occultation",
        altitude=alt_grid,
        sampling_parameters=sampling_df,
        edps=edps,
        pt1 = pt1,
        pt2 = pt2,
        pt3 = pt3
    )

    print(f"  EDPSamples dims  : {dict(eds.sizes)}")
    print(f"  EDPs shape       : {eds.edps.shape}   (height, geo_vertex, sample)")
    print(f"  Sampling params  :\n{sampling_df.to_string(index=False)}")

    # ── Plot: ne altitude profiles by solar scenario ──────────────────────────
    fig, axes = plt.subplots(1, n_sample, figsize=(5 * n_sample, 7), sharey=True)
    fig.suptitle(
        f"§8  EDPSamples — ne profiles per solar scenario\n"
        f"Alaska rect mesh  ({n_geo} vertices)   {TIME}"
    )

    # Vertex closest to the centre of the region
    dists    = np.hypot(rect_vertices[:, 0] - LON_C, rect_vertices[:, 1] - LAT_C)
    i_centre = int(np.argmin(dists))

    for i_s, (sc, ax, color) in enumerate(zip(SCENARIOS, axes, COLORS)):
        for i_g in range(n_geo):
            ax.plot(edps[:, i_g, i_s], alt_grid, color="lightgray", lw=0.7)
        ax.plot(
            edps[:, i_centre, i_s], alt_grid,
            color=color, lw=2.5,
            label=(f"centre  ({rect_vertices[i_centre,1]:.1f} °N, "
                   f"{rect_vertices[i_centre,0]:.1f} °E)"),
        )
        ax.set_xscale("log")
        ax.set_xlabel("ne (m⁻³)")
        ax.set_title(sc["label"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("Altitude (km)")

    # ── Save / load roundtrip ─────────────────────────────────────────────────
    ncpath = ROOT / "alaska_edp_demo.nc"
    try:
        eds.saveNetCDF(ncpath)
        print(f"\n  Saved  → {ncpath.name}  ({ncpath.stat().st_size // 1024} KB)")
        eds_loaded = EDPSamples.fromNetCDF(ncpath)
        print(f"  Loaded   EDPSamples dims: {dict(eds_loaded.sizes)}")
    except Exception as exc:
        print(f"\n  NetCDF roundtrip skipped: {exc}")

    # ── Demonstrate interp_heights ────────────────────────────────────────────
    query_alts = np.array([120.0, 250.0, 350.0, 450.0])
    idx, w = interp_heights(alt_grid, query_alts)
    print(f"\n  interp_heights — query altitudes: {query_alts}")
    print(f"    bracket indices : {idx}")
    print(f"    lower weights   : {w[:, 0].round(3)}")
    print(f"    upper weights   : {w[:, 1].round(3)}")

    # ── Demonstrate find_containing_triangles (spherical version) ─────────────
    # geolocation passed as (lat, lon) — the convention of locate_in_mesh.py
    geoloc_latlon = rect_vertices[:, [1, 0]]   # swap (lon,lat) → (lat,lon)
    query_pts = np.array([
        [62.5, -145.0],   # region centre
        [61.0, -148.0],   # SW
        [64.0, -142.0],   # NE
    ])
    tri_idx, bary = find_triangle_sphere(
        query_pts, geoloc_latlon, rect_triangles, return_bary=True
    )
    print(f"\n  find_containing_triangles ({len(query_pts)} queries):")
    for q, ti, b in zip(query_pts, tri_idx, bary):
        print(f"    ({q[0]:.1f} °N, {q[1]:.1f} °E)  →  triangle {ti}"
              f"   bary = {b.round(3)}")

    return eds

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("IonosphereTomography demo")
    print("Alaska region:  60–65 °N,  150–140 °W")
    print("=" * 60)

    # section1()
    # section2()
    # section3()
    # section4()
    # section5()
    v_rect, t_rect = section6()
    # section7()
    section8(v_rect, t_rect)
    # section9()
    # section10()
    # section11()
    v_occ, t_occ, pt1, pt2, pt3 = section12()
    section13(v_occ, t_occ, pt1, pt2, pt3)

    print("\n" + "=" * 60)
    print("All sections complete — displaying figures.")
    print("=" * 60)
    plt.tight_layout()
    # plt.show()
    


if __name__ == "__main__":
    main()
