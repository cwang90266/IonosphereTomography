# IonosphereTomography Virtual Environment Dependency Analysis

## Core Dependencies (Essential for Project)

### Numerical Computing & Data Processing
- `numpy` — fundamental numerical operations
- `scipy` — scientific computing (optimization, interpolation, filtering)
- `pandas` — data manipulation and analysis
- `xarray` — labeled multidimensional arrays and datasets
- `netCDF4` — reading/writing netCDF scientific data files
- `h5py` — HDF5 file format support
- `cdflib` — CDF file format support

### Ionosphere Models & Geophysics
- `iri2020` — IRI 2020 ionosphere forward model (CRITICAL)
- `pyfiri` — Python IRI wrapper
- `PyIRI` — additional IRI variant
- `georinex` — GPS RINEX file parsing (required for TEC pipeline)
- `hatanaka` — RINEX data compression/decompression
- `igrf` — International Geomagnetic Reference Field
- `PPigrf` — alternative IGRF implementation
- `aacgmv2` — Altitude Adjusted Corrected Geomagnetic coordinates

### Coordinate & Projection Handling
- `pyproj` — map projections and coordinate transformations
- `astropy` — astronomy library for coordinates and time handling
- `astropy-iers-data` — high-precision Earth orientation data for astropy

### Data Assimilation
- `filterpy` — Kalman filter implementations (EnKF support)

### Visualization & Mapping
- `matplotlib` — 2D plotting
- `plotly` — interactive plotting
- `Cartopy` — cartographic mapping library
- `pyshp` — shapefile format support
- `Shapely` — geometric operations

### Time & Calendar Handling
- `cftime` — calendar date/time handling for scientific data
- `python-dateutil` — date utilities
- `pytz` — timezone handling
- `DateTimeTools` — additional date/time utilities

### Data Acquisition & Processing Utilities
- `requests` — HTTP library for downloading data
- `beautifulsoup4` — HTML parsing for web scraping
- `tqdm` — progress bar visualization
- `joblib` — parallel processing and job management
- `kpindex` — geomagnetic activity indices

### File I/O & Serialization
- `PyFileIO` — file I/O utilities
- `RecarrayTools` — structured array tools
- `fortranformat` — Fortran binary format parsing (if needed)

---

## Optional / Likely Not Core (Recommend Removal)

### Machine Learning / Deep Learning Stack
⚠️ **Remove unless actively using neural networks:**
- `tensorflow` — deep learning framework
- `keras` — deep learning API
- `tensorboard` — TensorFlow visualization tool
- `google-pasta` — TensorFlow utility
- `grpcio` — gRPC protocol (TensorFlow dependency)
- `opt_einsum` — Einstein summation optimization (ML utility)
- `ml_dtypes` — machine learning data types
- `flatbuffers` — serialization format (TensorFlow dependency)
- `protobuf` — protocol buffers (TensorFlow dependency)

### Jupyter / Interactive Development (Dev-Only)
⚠️ **Remove if using editors instead of Jupyter notebooks:**
- `ipykernel` — Jupyter kernel
- `jupyter_client` — Jupyter client
- `jupyter_core` — Jupyter core
- `ipython` — IPython shell
- `spyder-kernels` — Spyder IDE kernels

### Development & Debugging Tools (Dev-Only)
- `pytest` — testing framework
- `debugpy` — Python debugger
- `py-spy` — Python profiler

### Space Science (Potentially Unused)
⚠️ **Verify usage before keeping:**
- `spacepy` — space physics library
- `PyGeopack` — magnetosphere models
- `pyomnidata` — space weather data

### Utilities & Infrastructure (Low Priority)
- `certifi` — SSL certificates
- `charset-normalizer` — text encoding
- `idna` — domain name utilities
- `urllib3` — HTTP client
- `cloudpickle` — enhanced pickling
- `decorator` — decorator utilities
- `executing` — source code execution
- `stack-data` — stack inspection
- `Pillow` — image processing
- `six` — Python 2/3 compatibility
- `Markdown`, `markdown-it-py`, `mdurl` — Markdown parsing
- `Rich` — rich terminal text
- `qrcode` — QR code generation
- `PyYAML` — YAML parsing
- `openpyxl` — Excel file handling
- `ncompress`, `unlzw3` — compression utilities
- `packaging`, `setuptools`, `wheel`, `pip` — packaging infrastructure
- `psutil` — system utilities
- `platformdirs` — platform-specific paths
- `kiwisolver` — constraint solver (matplotlib dependency)
- `cycler` — color cycling (matplotlib dependency)
- `fonttools` — font utilities (matplotlib dependency)
- `contourpy` — contour plotting (matplotlib dependency)
- `astunparse` — AST utilities
- `comm` — communication utilities
- `asttokens` — AST token utilities
- `parso` — Python parser
- `jedi` — code completion
- `pexpect` — expect-like functionality
- `pickleshare` — pickle utilities
- `prompt_toolkit` — terminal prompt toolkit
- `ptyprocess` — pseudo-terminal utilities
- `pure_eval` — Python evaluation
- `tornado` — web framework
- `Werkzeug` — WSGI utilities
- `termcolor` — colored terminal output
- `wrapt` — decorator wrapper
- `namex` — naming utilities
- `narwhals` — dataframe abstraction
- `importlib_resources` — import utilities
- `iniconfig` — INI config parsing
- `pluggy` — plugin system

---

## Recommended Actions

1. **Immediate removal** (if not actively using):
   - Remove entire TensorFlow/Keras stack (~500MB+)
   - Remove Jupyter tools if using an IDE

2. **Verify before removal**:
   - Check codebase for imports of `spacepy`, `PyGeopack`, `pyomnidata`
   - Confirm `fortranformat` usage in data pipelines
   - Confirm `openpyxl` usage in data export

3. **Consider minimal environment**:
   - Create separate `requirements-dev.txt` for jupyter/testing tools
   - Keep only essential packages in production/analysis environment

---

## Summary Statistics

- **Essential packages**: ~45
- **Optional/removable packages**: ~80
- **Potential to reduce**: Significant (remove TensorFlow stack and dev tools for ~60% size reduction)
