# EDPSamples class structure
_Exported on 3/30/2026 at 10:12:31 PDT from Cursor (2.6.20)_

---

**User**

I need to create a class of objects called EDPSamples that inherits all properties of the dataset in xarray. An object of EDPSamples has several coordinates: altitude, a one-dimensional vector; geolocation, an array of two columns (latitude and longitude); and sampling parameters, an array of several columns (e.g., f107, f107_81, ig, ir, etc.). The main dataset, EDPs is an array indexed by height, geolocation, and sampling parameters. It also has an integer array, Mesh, with 3 columns that specify the 3 vertex of a 2-D triangle in terms of indices geolocations points.

---

**Cursor**


Exploring the codebase for xarray usage and a suitable place to add `EDPSamples`.




Adding `EDPSamples` as an `xarray.Dataset` subclass with a factory constructor, schema validation, and convenience accessors. The repo doesn't list `xarray`; we'll add it to `requirements.txt`.




Fixing coordinate setup: xarray requires dimension coordinates for `param` and `geo_component`; removing the incorrect attrs dict.



---

**User**

We will define several methods for generating geolocations. The method genRectangularArea creates an evenly spaced grid in latitude and longitude over a region specified by minLat, maxLat, minLon, and MaxLon. The method genPolarArea generates a network of grid points covering either north pole or south pole region with latitude higher than a specified minLat value.

---

**Cursor**


Reading `edp_samples.py` to add `genRectangularArea` and `genPolarArea`.




Adding `genRectangularArea` and `genPolarArea` as static methods that return `(n, 2)` latitude/longitude arrays, matching the user's naming and parameter style.




Here’s what was added to `framework/edp_samples.py`.

### `genRectangularArea(minLat, maxLat, minLon, maxLon, numLat, numLon)`
- Builds a regular `numLat × numLon` grid with `np.linspace` on each axis and `meshgrid(..., indexing="ij")`.
- Returns shape `(numLat * numLon, 2)` with columns **latitude, longitude** (longitude changes fastest along the flattened order).
- Checks: `numLat`, `numLon` ≥ 1; `minLat ≤ maxLat`.

### `genPolarArea(pole, minLat, numLat, numLon)`
- **`pole="north"`:** lat from `minLat` to `90` (requires `minLat < 90`).  
- **`pole="south"`:** lat from `-90` to `-minLat`, with **positive** `minLat` as the magnitude of the equatorward edge (e.g. `minLat=66.5` → cap at and poleward of 66.5°S).
- Longitude: `numLon` points on **`[-180, 180)`** (`endpoint=False` when `numLon > 1` so -180 and +180 are not both included). If `numLon == 1`, lon is `0`.
- Same flattening order and `(n, 2)` layout as above.

Both are **`@staticmethod`**s on `EDPSamples`, so you can call them without an instance:

```python
geo = EDPSamples.genRectangularArea(-10.0, 10.0, -5.0, 5.0, 11, 11)
geo_n = EDPSamples.genPolarArea("north", 60.0, 8, 24)
geo_s = EDPSamples.genPolarArea("south", 60.0, 8, 24)
```

Pass the result into `EDPSamples.from_arrays(..., geolocation=geo, ...)` as before. Parameter names use the usual Python `minLat` style (not `MaxLon`); semantics match your min/max bounds.

---

**User**

A third method of generating geolocation is by generating locations along a line-of-sight between two satellites.

---

**Cursor**


Checking `edp_samples.py` — it may have been truncated.




Restoring the emptied `edp_samples.py` and adding `genLineOfSight` (ECEF chord between two geodetic positions, converted back to lat/lon).



---

**User**

Regenerate the code, use all inputs given previously.

---

**Cursor**


Regenerating the full `edp_samples.py` with EDPSamples, `from_arrays`,.Mesh/EDPs layout, and geolocation generators: `genRectangularArea`, `genPolarArea`, and `genLineOfSight`.




Summary of what was regenerated:

### `EDPSamples` (`xarray.Dataset` subclass)
- **1D** coordinate `altitude` on dimension `height`.
- **Data variable** `geolocation`: shape `(geo, 2)` — latitude, longitude (with `geo_component` coordinate).
- **Data variable** `sampling_parameters`: shape `(sample, param)` with a `param` coordinate for names (e.g. f107, f107_81, ig, ir).
- **Data variable** `EDPs`: shape `(height, geo, sample)`.
- **Optional** `Mesh`: shape `(triangle, 3)`, integer indices into `geo`.

### Construction & access
- **`from_arrays(...)`** — builds the dataset; optional `mesh` and `attrs`.
- **Properties:** `altitude`, `geolocation`, `sampling_parameters`, `edps`, `mesh`.

### Geolocation generators (all return `(n, 2)` lat/lon)
- **`genRectangularArea(minLat, maxLat, minLon, maxLon, numLat, numLon)`** — regular grid on a lat/lon box.
- **`genPolarArea(pole, minLat, numLat, numLon)`** — polar cap (`"north"` / `"south"`), longitudes on `[-180, 180)`.
- **`genLineOfSight(lat1, lon1, alt1_m, lat2, lon2, alt2_m, num_points)`** — samples on the WGS84 **ECEF chord** between two satellites; `num_points ≥ 2`, includes endpoints.

### Internals
- **`_geodetic_to_ecef` / `_ecef_to_geodetic`** — WGS84 helpers for LOS.

`requirements.txt` again lists **`xarray`** next to **`numpy`** so imports resolve. The full implementation lives in `framework/edp_samples.py`.

---

**User**

I also need a method to plot the geolocations for an object of EDFSamples class.

---

**Cursor**


Adding `plotGeolocations` to match existing camelCase generator methods, using matplotlib (already a project dependency).




Here’s what was added.

### `plotGeolocations(self, ax=None, *, show_mesh=False, **scatter_kwargs)`

- **Scatter** of all `geolocation` rows: **x = longitude**, **y = latitude** (degrees).
- **Axis labels**, light **grid**, title `"Geolocations"`.
- **`show_mesh=True`**: if `Mesh` exists, draws each triangle by connecting the three indexed points in lon/lat (closed rings, gray lines under the scatter).
- **`scatter_kwargs`**: forwarded to `ax.scatter` (e.g. `s`, `c`, `marker`, `alpha`).
- If **`ax` is omitted**, it creates a figure with `plt.subplots(figsize=(8, 5))` and calls **`plt.tight_layout()`** before returning the axes. It does **not** call `plt.show()` so you stay in control of display/saving.

Example:

```python
ds = EDPSamples.from_arrays(...)
ds.plotGeolocations()
plt.show()

ds.plotGeolocations(show_mesh=True, s=8, c="steelblue")
```

Imports: `matplotlib.pyplot` and `matplotlib.axes.Axes` at the top of `framework/edp_samples.py`.

---

**User**

The data array EDPs is filled by a method IRI2020_EDP that calls the Fortran driver for iri2020. The inputs to the driver are the altitude, geolocation and the sample parameters

---

**Cursor**


Searching the codebase for IRI2020 or Fortran bindings to integrate with `EDPSamples`.




Adding `IRI2020_EDP` to fill `EDPs` by calling a pluggable Fortran driver hook, plus `framework/iri2020_driver.py` with a documented stub to bind the real IRI2020 library.




Here is what was added.

### `EDPSamples.IRI2020_EDP(self, driver=None, *, altitude_unit="km", **driver_kwargs)`

- Walks every **(height, geo, sample)** index.
- For each triple it passes to the driver:
  - **altitude** – from coordinate `altitude`, converted to **km** when `altitude_unit="m"`.
  - **geolocation** – `latitude = geolocation[geo, 0]`, `longitude = geolocation[geo, 1]`.
  - **sample row** – `sampling_parameters[sample, :]` plus the **`param`** coordinate names (f107, f107_81, ig, ir, …).
- Writes results into **`EDPs`**, then sets `EDPs.attrs["iri_model"] = "IRI-2020"`.
- **`driver`** signature:  
  `driver(altitude_km, latitude_deg, longitude_deg, sample_params, param_names, **driver_kwargs) -> float`  
  If `driver` is omitted, it uses **`iri2020_fortran_edp_point`** from `framework/iri2020_driver.py`.

### `framework/iri2020_driver.py`

- Defines **`iri2020_fortran_edp_point(...)`**, which currently raises **`NotImplementedError`** with a short note to connect your real **Fortran IRI2020** entry point (f2py, `ctypes`, etc.).
- **`driver_kwargs`** from `IRI2020_EDP` are forwarded so you can pass UT, `jf` switches, or anything else your Fortran wrapper needs without changing the outer API.

You replace the body of `iri2020_fortran_edp_point` with calls into your compiled driver (or pass a custom `driver=` to `IRI2020_EDP` for experimentation). The loop stays in Python; if you need speed, implement batching or a vectorized call inside that function.

---

**User**

I also need methods to save the dataset into a netcdf file and load the dataset from a netcdf file

---

**Cursor**


Here’s what was added.

### `saveNetCDF(self, path, **kwargs)`
- Calls **`xarray.Dataset.to_netcdf`**, so you can pass anything that API accepts (`mode`, `format`, `engine`, `encoding`, etc.).
- **`path`** can be a `str` or **`pathlib.Path`**.

### `fromNetCDF(path, **kwargs)` *(classmethod)*
- Opens the file with **`xarray.open_dataset(path, **kwargs)`**, **`load()`**s it so everything is in memory, then builds an **`EDPSamples`** with the same `data_vars`, `coords`, and `attrs`.
- Uses a context manager so the file is closed after the read.

Example:

```python
ds.saveNetCDF("run_edps.nc")
restored = EDPSamples.fromNetCDF("run_edps.nc")
```

`requirements.txt` now includes **`netcdf4`** so NetCDF4 read/write is usually available to xarray. If you prefer another engine (e.g. `h5netcdf`), you can pass `engine=...` into both methods.

