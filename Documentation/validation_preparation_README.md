# Real ISR and ionosonde validation

Place this folder in `/home/pin/Desktop/tomography_project/` and run from
that project root. Files are read directly; no pickle, cache, or load_edps call.

```bash
python -m validation_preparation.test_isr_ionosonde_validation \
  --centre "2025-12-15T06:00:00Z" --height-bin-km 10
```

Paths live in the validators, relative to the project root:
* `isr_profile_validation.ISR_DATA_DIR`: `Data/ISR_Data/`
* `ionosonde_parameter_validation.IONOSONDE_DATA_DIR`: `Data/Ionosonde_Data/`

Dependencies: numpy, pandas, matplotlib; netCDF4 for NC files; h5py for HDF5.
The test reuses `ALT_GRID` and `INSTRUMENTS` from `demo_isr_initial_conditions`.
The ISR default valid-gate count comes from `plotIonosphereTomography`.
Supply `sites=` and `min_valid_gates=` to the ISR function to use it without
those project imports. There is no data-path configuration in the test.

## ISR comparison

The supplied NetCDF has `gdalt` AND `ne` dimensioned by range and timestamps;
the reader uses the named time dimension, not array-position assumptions.
Each scan uses its own geodetic altitude, never slant range as altitude.
The `dne` uncertainty is not subtracted from the observed Ne.
Only fitted kindat 6400 observations are retained, matching the main code.

Per-scan RMSE interpolates input Ne onto that scan's ISR heights. Altitude RMSE
interpolates ISR Ne onto the input voxel grid, then pools squared residuals over
scans AND input levels within fixed [lower, upper) height bins. The default bin
width is 10 km; this is not resampling onto a uniformly spaced model grid.
Empty bins remain NaN. Uneven voxel spacing means some bins may be empty.
All overlapping heights are considered; there is no F2/topside cutoff or
extrapolation. Finite overlapping samples and Ne > 1e7 m^-3 are required by
default. Each accepted scan must meet the configured minimum valid-gate count.
The one input column is held fixed throughout the chosen time window.

UTC window dictionaries use `lo`, `hi`, `t_centre`, and optional `window_key`.
Both endpoints are included, consistent with the main-code ISR selector.
Availability ranges refer to successfully decoded records and may contain gaps.
The provided ISR ncdisp reports 2025-12-15 04:00:40--07:59:15 UTC; use 06:00
for this example, not 20:00 unless other files cover that time.

## Ionosonde comparison

Only product parameters are read; the large `pf` profile array is not loaded.
NmF2/NmE use direct density variables when provided, otherwise the product's
fof2/foe are converted to density with (f_Hz/8.98)^2. No observation peaks are
computed from a profile. One closest usable record inside the window supplies
all four comparisons; missing values are not borrowed from another scan.

The supplied product has no explicit hmF2/hmE peak-height variables.
kmztrf/kmztre are described as mean real layer heights. To compare those as
explicitly labeled proxies, add `--allow-mean-height-proxy`. This does not turn
them into true peak heights or repair missing critical-frequency measurements.

The four bar panels use blue for observed product values, red for input.
Missing observations are annotated and remain NaN in the comparison CSV.

## Formats and limits

Recursive discovery supports .nc, .nc4, .h5, .hdf5, .hdf. Readers support NetCDF
timestamp groups, ordinary HDF5 timestamp groups, Madrigal Data/Array Layout
(including split groups with 1D/2D Parameters), and Madrigal Data/Table Layout.
For tables with ut1_unix/ut2_unix the record midpoint is used as observation time.
Station latitude/longitude must be present in attributes or Madrigal Experiment
Parameters metadata; station identity is not guessed from data values.
Unknown layouts/units and ambiguous HDF5 time axes are reported per file.
Exactly matching duplicate NC/HDF5 exports are counted once.

The real-data test never generates observation files. Numerical regression
checks were run separately during preparation; your real NC/HDF5 files were
not provided, so full real-file validation must be run on your machine.
