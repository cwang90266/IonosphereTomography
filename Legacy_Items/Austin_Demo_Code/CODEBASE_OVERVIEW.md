# IonosphereTomography Codebase Overview

This repository contains two major subsystems:

1. **`iri2020_new/`** — A Python wrapper around the IRI2020 (International Reference Ionosphere) Fortran model. It builds a Fortran executable, calls it as a subprocess, and parses the results into `xarray.Dataset` objects.
2. **`EDPSamples/`** — A library for generating, storing, and querying Electron Density Profile (EDP) sample collections over geographic meshes.
3. **`IRI_Sample_Inputs/`** — Utilities for fetching and sampling the solar/geomagnetic input parameters that IRI2020 requires.

---

## Key Object: `xarray.Dataset` (the `iono` object)

Most functions return or accept an `xarray.Dataset` (often called `iono` in the code). Think of it as a labelled multi-dimensional array container — like a dictionary of NumPy arrays where each array has named dimensions and coordinates.

**Fields on the altitude dimension (`alt_km`):**
| Variable | Meaning |
|---|---|
| `ne` | Electron number density (m⁻³) |
| `Tn` | Neutral temperature (K) |
| `Ti` | Ion temperature (K) |
| `Te` | Electron temperature (K) |
| `nO+`, `nH+`, `nHe+`, `nO2+`, `nNO+`, `nCI`, `nN+` | Ion species densities |

**Scalar fields (on the `time` dimension):**
| Variable | Meaning |
|---|---|
| `NmF2`, `NmF1`, `NmE` | Peak electron density of F2/F1/E ionospheric layers (m⁻³) |
| `hmF2`, `hmF1`, `hmE` | Height of those peaks (km) |
| `TEC` | Total Electron Content (m⁻²) |
| `foF2` | F2 layer plasma frequency (MHz) |
| `B0`, `B1` | IRI bottom-side profile shape parameters |
| `EqVertIonDrift` | Equatorial vertical ion drift (m/s) |

**Dataset attributes (metadata):**
| Attribute | Meaning |
|---|---|
| `f107D` | Daily solar flux index F10.7 |
| `ap` | Geomagnetic Ap index |
| `f107_81` | 81-day smoothed F10.7 |
| `IG12` | Ionospheric Global index (12-month running mean) |
| `Rz12` | 12-month smoothed sunspot number |

---

## Module 1: `iri2020_new/src/iri2020/`

### `base.py` — Core IRI function

#### `IRI(time, altkmrange, glat, glon, foF2=None, hmF2=None, B0=None, B1=None, f107D=None, ap=None, IG12=None, Rz12=None) → xarray.Dataset`

**The central function of the entire codebase.** Runs the IRI2020 model for a single point and time.

**Parameters:**
- `time` — A datetime string (e.g., `"2022-01-01 12:00"`) or a Python `datetime` object.
- `altkmrange` — A 3-element list `[start_km, stop_km, step_km]` defining the altitude grid. Example: `[80, 1000, 10]` gives altitudes 80, 90, 100, ..., 1000 km.
- `glat` — Geodetic latitude (degrees, scalar).
- `glon` — Geodetic longitude (degrees, scalar).
- **Optional group 1 (ionospheric parameters):** `foF2`, `hmF2`, `B0`, `B1` — if *all four* are provided, they override the IRI model's internal computation of the F2 layer shape. Used for data assimilation.
- **Optional group 2 (solar/geomagnetic indices):** `f107D`, `ap`, `IG12`, `Rz12` — if *all four* are provided, they replace IRI's internal index lookup from the data files.
- If neither group is fully provided, IRI uses its built-in index lookup from the bundled data files.

**How it works:**
1. Locates the compiled Fortran executable `iri2020_driver` (builds it via CMake if missing).
2. Constructs a command-line argument list and runs the executable as a subprocess via `subprocess.check_output`.
3. Parses the stdout text into two sections: altitude-profile data (one row per altitude) and a 100-element parameter array.
4. Assembles and returns an `xarray.Dataset`.

**Example call:**
```python
from iri2020 import IRI
iono = IRI("2022-06-15 14:00", [80, 500, 10], glat=45.0, glon=-75.0)
print(iono["ne"])  # electron density profile
```

---

### `vprofile.py` — Profile sweeps over time or space

#### `datetimerange(start, end, step) → list[datetime]`

A utility analogous to Python's `range()` but for `datetime` objects. Generates a list of evenly spaced datetimes from `start` to `end` (exclusive) with step `step` (a `timedelta`).

```python
from datetime import timedelta
times = datetimerange("2022-01-01", "2022-01-02", timedelta(hours=6))
# → [2022-01-01 00:00, 2022-01-01 06:00, 2022-01-01 12:00, 2022-01-01 18:00]
```

#### `timeprofile(tlim, dt, altkmrange, glat, glon) → xarray.Dataset`

Runs `IRI` repeatedly over a time range at a fixed location and stacks the results along the `time` dimension.

**Parameters:**
- `tlim` — A 2-tuple of `(start, end)` as strings or datetimes.
- `dt` — A `timedelta` for the time step.
- `altkmrange` — Same `[start, stop, step]` list as `IRI`.
- `glat`, `glon` — Fixed location.

**Returns:** An `xarray.Dataset` concatenated along `time`, with all the same variables as a single `IRI` call.

**Example:**
```python
from datetime import timedelta
from iri2020 import timeprofile
iono = timeprofile(("2022-01-01", "2022-01-02"), timedelta(hours=1), [100, 500, 10], 45.0, -75.0)
iono["ne"]  # shape: (time, alt_km)
```

#### `geoprofile(latrange, glon, altkm, time) → xarray.Dataset`

Runs `IRI` repeatedly over a latitude sweep at a fixed longitude, altitude, and time, stacking results along the `glat` dimension.

**Parameters:**
- `latrange` — A 3-element sequence `[start, stop, step]` for `np.arange`. Example: `(-60, 61, 2.0)`.
- `glon` — Fixed longitude.
- `altkm` — Single altitude. Note: internally passed as `altkmrange=[altkm, altkm, altkm]`, so IRI returns a single-altitude profile.
- `time` — Datetime string or object.

**Example:**
```python
from iri2020 import geoprofile
iono = geoprofile((-60, 61, 5.0), glon=-75.0, altkm=300.0, time="2022-06-15 12:00")
iono["NmF2"]  # peak electron density vs latitude
```

---

### `altitude.py` — Altitude profile CLI wrapper

Wraps a single `IRI` call with a command-line interface.

#### `main(time, alt_km, glat, glon) → xarray.Dataset`
Thin wrapper around `IRI`. Takes the same arguments and returns the dataset.

#### `cli()`
Parses command-line arguments (`time`, `latlon`, optional `-alt_km`) and calls `main`, then optionally plots via `piri.altprofile(iono)`.

---

### `latitude.py` — Latitude profile CLI wrapper

Wraps `geoprofile` with a command-line interface.

#### `main(time, alt_km, glat, glon, outfn=None) → xarray.Dataset`
Calls `geoprofile(latrange=glat, glon=glon, altkm=alt_km, time=time)`. If `outfn` is provided, saves the result to a NetCDF file via `iono.to_netcdf(outfn)`.

#### `cli()`
Parses command-line arguments and calls `main`, then optionally plots via `plot_lat(iono)`.

---

### `times.py` — Time profile CLI wrapper

Wraps `timeprofile` with a command-line interface.

#### `main(time, alt_km, glat, glon) → xarray.Dataset`
Calls `timeprofile((time[0], time[1]), timedelta(hours=float(time[2])), alt_km, glat, glon)`. The `time` argument is a 3-element list: `[start_date, end_date, step_hours]`.

#### `cli()`
Parses command-line arguments and calls `main`, then optionally plots.

---

### `sensitivity_profile.py` — Sensitivity analysis

Performs a parameter sensitivity study by running IRI many times with perturbed inputs.

#### `main(time, alt_km, glat, glon, foF2=None, hmF2=None, B0=None, B1=None) → xarray.Dataset`
A thin wrapper that calls `IRI` (optionally with the four ionospheric shape parameters) and returns the dataset.

#### `cli()`

The CLI's behavior is controlled by `iParam_Switch`:

**`iParam_Switch == 0` (or None) — Ionospheric shape parameter sensitivity:**
1. Runs IRI once to get a nominal result and extract `foF2`, `hmF2`, `B0`, `B1`.
2. Creates 10 perturbed runs for each of the 4 parameters (scaling from 0.5× to 2× the nominal value), varying one parameter at a time.
3. Calls `piri.altprofile_sensitivity(1, ...)` to plot the results.

**`iParam_Switch != 0` — Solar/geomagnetic index sensitivity:**
1. Reads `apf107.dat` and `ig_rz.dat` using functions from `get_iri_inputs`.
2. For each of `f107D`, `ap`, `IG12`, `Rz12`, computes `[min, mean, median, max]` values over a window around the simulation date.
3. Runs IRI for each combination, varying one index at a time.
4. Calls `piri.altprofile_sensitivity(2, ...)` to plot.

---

### `plots.py` — Visualization

All plot functions accept an `xarray.Dataset` (the `iono` object) and produce matplotlib figures.

#### `altprofile(iono)`
2-panel plot: electron density (`ne`) vs altitude (log scale) on the left, ion and electron temperatures (`Ti`, `Te`) vs altitude on the right.

#### `timeprofile(iono)`
3 plots:
1. Peak densities `NmF2`, `NmF1`, `NmE` vs time.
2. Peak heights `hmF2`, `hmF1`, `hmE` vs time.
3. F2 plasma frequency `foF2` vs time.
4. A separate figure with TEC vs time.
5. A colour-mesh `ne` vs altitude and time.

#### `latprofile(iono)`
2-panel plot: peak densities and peak heights vs geographic latitude.

#### `altprofile_sensitivity(iParam_Switch, iono_initial, iono_param1, iono_param2, iono_param3, iono_param4)`
A 2×2 subplot figure showing how `ne` altitude profiles change as each of four parameters is varied. The nominal profile is drawn in black, the minimum in red, the maximum in blue. When `iParam_Switch==1` the four parameters are `foF2`, `hmF2`, `B0`, `B1`; when `iParam_Switch==2` they are `f107D`, `ap`, `IG12`, `Rz12`.

---

### `get_iri_inputs.py` — Parsing IRI input data files

These functions parse the two solar/geomagnetic index data files bundled with IRI2020.

#### `read_apf107(file_path) → dict`

Parses `apf107.dat` line by line using fixed-width column positions. Returns a dictionary:
- `"yr"`, `"mn"`, `"dy"` — lists of year, month, day integers.
- `"iapda"` — daily Ap index (scalar per day).
- `"iiap"` — list of 8 3-hourly Ap values per day.
- `"ir"` — sunspot number.
- `"f107"` — daily solar flux F10.7.
- `"f107_81"` — 81-day centred mean F10.7.
- `"f107_365"` — 365-day centred mean F10.7.

**Call:**
```python
apf107 = read_apf107("/path/to/data/")
```

#### `read_ig_rz(file_path) → dict`

Parses `ig_rz.dat`, which contains monthly mean IG (Ionospheric Global index) and Rz (sunspot number) values. Returns:
- `"Revision"` — file revision date integers.
- `"Start_end_month"` — `[start_month, start_year, end_month, end_year]`.
- `"ig"` — flat list of monthly IG values.
- `"rz"` — flat list of monthly Rz values.

#### `show_iri_inputs(apf107, ig_rz)`
Plots the time histories of Ap, F10.7 (three variants), IG, and Rz as a 4-panel figure.

---

### `build.py` — Fortran compilation

#### `build()`
Uses CMake to compile the IRI2020 Fortran source code (`src/iri2020/src/`) into the executable `iri2020_driver`. Called automatically by `IRI()` if the executable is not found. On Windows it uses the MinGW generator.

---

## Module 2: `EDPSamples/`

The EDPSamples subsystem manages collections of electron density profiles (EDPs) sampled at many geographic locations and for many different sets of IRI2020 input parameters simultaneously.

---

### `edp_samples.py` — Main EDP data container

#### Key object: `EDPSamples` (subclass of `xarray.Dataset`)

`EDPSamples` extends `xarray.Dataset` to hold EDP data in a structured way. Its three core dimensions are:
- **`height`** — the altitude grid.
- **`geo_vertex`** — geographic locations (each is a lat/lon point).
- **`sample`** — different sets of IRI2020 input parameters (e.g., different solar activity levels).

**Variables:**
| Name | Shape | Description |
|---|---|---|
| `geolocation` | `(geo_vertex, lat_lon)` | Latitude and longitude of each grid point |
| `EDPs` | `(height, geo_vertex, sample)` | Electron density at each altitude, location, and sample |
| `sampling_parameters` | `(sample, param)` | The IRI2020 input parameter values for each sample |
| `Mesh` (optional) | `(mesh_triangle, 3)` | Triangle connectivity for area grids |

**Coordinate:**
| Name | Description |
|---|---|
| `altitude` | 1D altitude values along the `height` dimension |

#### `EDPSamples.__init__(DateTime, geo_type, altitude, sampling_parameters, ...)`

Constructs the dataset. The `geo_type` argument (a string) controls which geographic grid is built:

| `geo_type` | Required args | What it builds |
|---|---|---|
| `"Point"` | `Lon`, `Lat` | A single geographic point |
| `"LOS"` | `LOS_LEO`, `LOS_GNSS`, `LOS_nb_point` | Points along a satellite line-of-sight chord |
| `"Rectangle"` | `minLon`, `maxLon`, `dLon`, `minLat`, `maxLat`, `dLat` | A rectangular lat/lon mesh with triangulation |
| `"Polar"` | `minLat`, `dLat` | A polar cap mesh (north if `minLat > 0`, south otherwise) |

The `sampling_parameters` argument is a `pandas.DataFrame` where each row is one set of IRI2020 inputs (columns: `hour`, `f107`, `ap`, `ig12`, `rz12`). The number of rows becomes the `sample` dimension size.

If `evaluate_iri=1` is passed, the constructor automatically calls `get_IRI2020_EDP` to fill the `EDPs` array via the Fortran `iri2020_namelist_driver` executable.

**Example:**
```python
import pandas as pd
import numpy as np
from EDPSamples.edp_samples import EDPSamples

samples = pd.DataFrame({
    "hour": [12.0, 12.0],
    "f107": [100.0, 150.0],
    "ap": [5.0, 20.0],
    "ig12": [50.0, 80.0],
    "rz12": [40.0, 70.0],
})
altitude = np.arange(100, 600, 10, dtype=float)  # km

eds = EDPSamples(
    DateTime="2022-06-15 12:00",
    geo_type="Rectangle",
    altitude=altitude,
    sampling_parameters=samples,
    minLon=-80, maxLon=-70, dLon=2.0,
    minLat=40, maxLat=50, dLat=2.0,
)
```

#### `EDPSamples.fromNetCDF(path, **kwargs) → EDPSamples` *(classmethod)*

Loads a previously saved `EDPSamples` from a NetCDF file. Reads the `geo_type` attribute to reconstruct the correct constructor call.

#### `EDPSamples.from_xarray(ds) → EDPSamples` *(classmethod)*

Converts a plain `xarray.Dataset` (already in memory) into an `EDPSamples` object, checking that all required fields are present.

#### `EDPSamples.saveNetCDF(path, **kwargs)`

Saves the dataset to a NetCDF file via `xarray.Dataset.to_netcdf`.

#### `EDPSamples.plot_geolocation()`

Visualises the geographic grid:
- `"Point"` or `"LOS"`: scatter plot of locations.
- `"Rectangle"`: calls `plot_tri_mesh`.
- `"Polar"`: calls `plot_polar_mesh`.

#### `EDPSamples.interp(positions, coordinate="lla") → tuple`

Computes interpolation indices and weights for a set of query positions.
- `positions` — shape `(M, 3)`, columns are lat/lon/altitude (in `"lla"` mode) or ECEF x/y/z metres (in `"ecef"` mode).
- Returns `(idx_alt, weight_alt)` for `"Point"` and `"LOS"` types, or `(idx_alt, weight_alt, idx_mesh, weight_mesh)` for `"Rectangle"` and `"Polar"` types.

**Properties:** `altitude`, `geolocation`, `sampling_parameters`, `edps`, `mesh` — all return the corresponding `xarray.DataArray` (or `None` for `mesh` if not present).

---

#### Module-level functions in `edp_samples.py`

#### `_geodetic_to_ecef(lat_deg, lon_deg, alt_m) → np.ndarray`
Converts geodetic coordinates (degrees, degrees, metres above WGS84 ellipsoid) to ECEF (Earth-Centred Earth-Fixed) Cartesian coordinates in metres. Output shape has a trailing axis of length 3 (x, y, z). Uses the WGS84 ellipsoid constants.

#### `_ecef_to_geodetic(xyz) → (lat_deg, lon_deg, h_m)`
The inverse: converts ECEF Cartesian (metres) back to geodetic latitude (degrees), longitude (degrees), and height (metres). Uses a Bowring-style iterative solution, vectorised with NumPy.

#### `get_IRI2020_EDP(DateTime, altitude, geolocation, sampling_parameters, ...) → np.ndarray`
The function that actually runs IRI2020 in batch mode using the Fortran `iri2020_namelist_driver` executable. It:
1. Calls `write_IRI2020_namelist` to create a Fortran namelist file.
2. Creates symlinks to the required IRI data files in the working directory.
3. Executes `iri2020_namelist_driver` as a shell command.
4. Cleans up the symlinks.
5. Calls `read_IRI2020_binary_output` to parse the binary result.

Returns a 3D NumPy array of shape `(nheight, npts, nSample)`.

**Note:** The path to the IRI executable is currently hard-coded to a specific developer path — this would need to be updated for use on another machine.

#### `write_IRI2020_namelist(DateTime, altitude, geolocation, sampling_parameters, namelist_filename) → int`
Writes a Fortran namelist file (`.nml`) that the `iri2020_namelist_driver` executable reads. The namelist format specifies:
- Grid sizes: `npts`, `nheight`, `nSample`.
- Column indices telling the driver which column of `phy_inputs` holds which parameter (`idxHour=1`, `idxF107D=2`, etc.).
- The date/time.
- Altitude grid values (`height_grid(i)`).
- Geographic coordinates (`latitude(i)`, `longitude(i)`).
- A 2D array `phy_inputs(param, sample)` with all the parameter values (NaN→fill value).

Returns `0` on success.

#### `read_IRI2020_binary_output(IRI_output_filename) → np.ndarray`
Reads the binary file written by `iri2020_namelist_driver`. The format is:
- First 12 bytes: three little-endian 32-bit integers: `npts`, `nSample`, `nheight`.
- Remaining bytes: `float32` values in loop order `(sample, pts, height)`.

Returns a 3D NumPy array shaped `(nheight, npts, nSample)`.

#### `interp_heights(height_table, height_vector) → (idx, w)`
Computes linear interpolation weights for looking up values in a 1D altitude grid.
- `height_table` — the known altitude grid (strictly increasing).
- `height_vector` — the query altitudes.
- Returns `idx` (left-bracket index) and `w` (shape `(M, 2)` weights summing to 1).

#### `find_containing_triangles(query_latlon, geolocation, mesh, ...) → triangle_idx [, bary]`
Locates each query (lat, lon) point within a spherical triangular mesh. See the full description in the [Locate in mesh](#locate-in-mesh) section below. This version uses `_geodetic_to_ecef` (WGS84) rather than the unit-sphere `latlon_to_xyz` used in the standalone `locate_in_mesh.py`.

#### `plot_polar_mesh(vertices, triangles, pole="north", ax=None)` and `plot_tri_mesh(vertices, triangles, ax=None)`
Same functions as the standalone files (see below), embedded here for convenience.

---

### `generate_rect_tri_mesh.py`

#### `generate_rect_tri_mesh(minLat, maxLat, dLat, minLon, maxLon, dLon) → (vertices, triangles)`

Generates a rectangular lat/lon grid and triangulates it. The algorithm ensures the number of lat and lon grid points is always odd, creating a staggered grid where alternate columns are offset by half a latitude step — this produces well-shaped triangles.

**Returns:**
- `vertices` — shape `(N, 2)`, columns are `(longitude, latitude)`.
- `triangles` — shape `(M, 3)`, integer indices into `vertices` for each triangle corner. Each rectangular cell is subdivided into 4 triangles.

This is also exposed as `EDPSamples.genRectangularArea` (a `@staticmethod`).

---

### `generate_polar_mesh.py`

#### `generate_ploar_mesh(pole, minLat, dLat) → (vertices, triangles)`

*(Note: "polar" is misspelled in the function name as `ploar` in this file.)*

Generates a triangular mesh for a polar cap region. The algorithm:
1. Starts at the pole with 6 triangles (longitude step 60°).
2. Moves equatorward ring by ring.
3. At each ring, checks whether the arc length between adjacent vertices has grown by more than 10%. If yes, it doubles the number of longitude points and creates 3-triangle fan transitions; if no, it keeps the same count and creates 4-triangle rectangular transitions.

This adaptive refinement keeps triangles roughly equilateral across the cap.

This is also exposed as `EDPSamples.genPolarArea`.

---

### `EDPSamples.genLineOfSight(lat1, lon1, alt1_m, lat2, lon2, alt2_m, num_points) → (vertice, segment)` *(staticmethod)*

Computes `num_points` evenly spaced points along the straight ECEF chord (not great circle arc) between two satellite positions. Steps:
1. Convert both endpoints from geodetic to ECEF using `_geodetic_to_ecef`.
2. Linear interpolation along the chord: `r(t) = (1-t)*r0 + t*r1`.
3. Convert each intermediate point back to geodetic using `_ecef_to_geodetic`.

Returns:
- `vertice` — shape `(num_points, 2)`, columns are `(latitude, longitude)`.
- `segment` — shape `(num_points-1, 2)`, pairs of consecutive point indices (the "mesh" for a line).

---

### `plot_polar_mesh.py`

#### `plot_polar_mesh(vertices, triangles, pole="north", ax=None) → ax`

Creates a polar projection plot. Converts longitude to radians and maps latitude to a radial coordinate: `r = 90 - lat` for north, `r = 90 + lat` for south. Draws each triangle's edges in black and fills with a light blue, then scatters the vertices in red.

---

### `plot_tri_mesh.py`

#### `plot_tri_mesh(vertices, triangles, ax=None) → ax`

Creates a standard Cartesian (longitude vs latitude) plot. Draws each triangle as a black-outlined, light-blue-filled polygon, and scatters vertices in red.

---

### `Locate in mesh/outputs/locate_in_mesh.py`

#### `latlon_to_xyz(lat, lon, degrees=True) → np.ndarray`
Converts lat/lon to 3D unit vectors on the unit sphere: `x = cos(lat)*cos(lon)`, `y = cos(lat)*sin(lon)`, `z = sin(lat)`. Output has a trailing axis of length 3.

#### `find_containing_triangles(query_latlon, geolocation, mesh, degrees=True, k=8, max_k=None, eps=1e-10, return_bary=False)`

The core spherical point-in-triangle algorithm. For each of M query points, finds which triangle in the mesh contains it. Uses:

1. **Convert to 3D unit vectors** — works on the unit sphere, so the antimeridian and poles need no special handling.
2. **KD-tree on triangle centroids** — finds the `k` nearest triangles by Euclidean distance in 3D (which is monotonic in great-circle distance).
3. **Scalar triple product containment test** — for triangle corners A, B, C and query point P:
   - `s1 = (A × B) · P`, `s2 = (B × C) · P`, `s3 = (C × A) · P`
   - P is inside if all three products have the same sign as the triangle's orientation: `orient = (A × B) · C`
   - The orientation check prevents falsely matching the antipodal triangle.
4. **Adaptive k widening** — if any points are unresolved after testing `k` candidates, `k` is doubled and the search retries.

**Returns:**
- `triangle_idx` — shape `(M,)` integer array; `-1` if a point was not found in any triangle.
- `bary` (optional) — shape `(M, 3)` planar barycentric weights for interpolation.

---

### `Locate in mesh/outputs/_verify.py`

A standalone test script (no external dependencies beyond NumPy). Verifies the scalar-triple-product containment logic using a brute-force implementation against an analytical ground truth (octahedron octants). Not part of the main library.

---

## Module 3: `IRI_Sample_Inputs/IRI_Sample_inputs.py`

### Stand-alone data fetching functions

#### `get_apf107() → dict`

Downloads `apf107.dat` from the CHAIN network server, saves it locally as `apf107_data.dat`, and parses it into the same dictionary structure as `read_apf107` (keys: `yr`, `mn`, `dy`, `iapda`, `iiap`, `ir`, `f107`, `f107_81`, `f107_365`).

#### `get_ig_rz() → dict`

Downloads `ig_rz.dat` from the CHAIN server, saves it locally as `ig_rz.dat`, and parses it into the same structure as `read_ig_rz` (keys: `Revision`, `Start_end_month`, `ig`, `rz`).

#### `show_iri_inputs(apf107, ig_rz)`

Same visualisation function as in `get_iri_inputs.py` — plots Ap, F10.7, IG, Rz time histories.

---

### `IRI_Sample_Inputs` class

A higher-level object that fetches the input data on construction and provides a method to generate structured sampling plans.

#### `__init__(year, month, day, hour, minute, second)`

On creation:
1. Calls `get_apf107()` and `get_ig_rz()` to download and parse the data files.
2. Finds the index into `apf107` that matches the given date (`current_idx_f107`).
3. Finds the index into `ig_rz` that matches the given year/month (`current_idx_igrz`), anchored to the 15th of each month (which is how `ig_rz.dat` is structured).

#### `quantileSamples(hour_sample_range=None, ap_sample_range=None, f107_sample_range=None, ig_sample_range=None, rz_sample_range=None) → pd.DataFrame`

Generates a full combinatorial grid of IRI2020 input samples. For each parameter:
- If the range argument is `None`, only `None` (= "use IRI default") is included for that parameter.
- If a range integer `N` is provided, it extracts a window of ±N records around the current time index and computes `[min, mean, max]` from that window, plus `None`.

The result is the Cartesian product of all parameter ranges — so if you specify ranges for all 5 parameters each with 3 values + None, you get `4^5 = 1024` rows. Each row is one sample set.

**Returns:** A `pandas.DataFrame` with columns `hour`, `f107`, `ap`, `ig12`, `rz12`. This is the `sampling_parameters` argument for `EDPSamples.__init__`.

**Example:**
```python
from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs
inp = IRI_Sample_Inputs(2022, 6, 15, 12, 0, 0)
samples = inp.quantileSamples(f107_sample_range=81, ap_sample_range=27)
print(samples.shape)  # (N, 5)
```

---

## Typical Workflow

```
IRI_Sample_Inputs          EDPSamples                   iri2020_new
─────────────────          ──────────                   ───────────
IRI_Sample_Inputs()    →   EDPSamples(              →   IRI() called
 .quantileSamples()         geo_type, altitude,          repeatedly by
                            sampling_parameters,         iri2020_namelist_driver
                            evaluate_iri=1)              (Fortran)

                       ←   EDPSamples.saveNetCDF()
                           EDPSamples.fromNetCDF()
                           EDPSamples.interp()
```

The typical use case is:
1. Use `IRI_Sample_Inputs` to generate a `pandas.DataFrame` of solar/geomagnetic parameter combinations.
2. Create an `EDPSamples` object that defines the geographic grid and altitude range.
3. Fill it with IRI2020 EDPs (either during construction or later).
4. Save to NetCDF for persistence.
5. Use `EDPSamples.interp()` to interpolate the EDP ensemble at arbitrary 3D positions (e.g., along a satellite ray path).
