# Spline_Basis / the 'SPLINE' parameterization style

`Spline_Basis/` implements a per-geo-column cubic B-spline vertical basis,
and `Parameterization.py`'s `'SPLINE'` style uses it as an alternative to
the PCA/ANCHOR parameterizations already in `Assimilation_Cycle`. Both the
basis/fitting code and the knot-grid optimizer already exist and are
tested; this document is about *using* them, not building them.

## 1. What SPLINE actually fits

For each geo column `g`, SPLINE represents density as a ratio to that
column's own ensemble-mean profile, `mean_g(h)`, through a cubic B-spline
`y_g(h)` evaluated on a fixed knot grid:

```
Ne_g(h) = mean_g(h) * g_space(y_g(h))
```

where `g_space` depends on the grid file's own `fit_space`:

| `fit_space` | `g_space(y)` | Notes |
| --- | --- | --- |
| `linear` | `y` | Ratio itself is the spline value; can go negative, same caveat as other linear-ratio styles (e.g. `PCA_3D`). |
| `pow` | `\|y\|^alpha` | Always >= 0 for any `alpha > 0`; requires `alpha` set in the grid file. |
| `log10` | `10^y` | Always > 0; the spline fits `log10(density/mean)`. |

**This is the answer to "can SPLINE be combined with log10 or pow": yes —
`fit_space` is not a separate style name, it's a property recorded inside
the grid file `'SPLINE'` is built from.** Pick it when you *generate* the
grid file (Section 3), not when you select the style string.

The basis also enforces `Ne(h0) = 0` at the grid's bottom boundary (so the
whole package can safely sample down to `h0`, e.g. 60 km, without forcing
unrealistic Ne values there) and a configurable upper boundary condition
(exponential falloff, i.e. a topside scale height, fit from the ensemble's
own topside slope). See `Documentation/Spline_Basis_Knot_Optimization_Plan.md`
for the full design background.

## 2. Package contents (already in the repo, nothing here needs rebuilding)

- `Spline_Basis/spline_basis.py` -- `ConstrainedSplineBasis` (the per
  -column basis + boundary conditions).
- `Spline_Basis/spline_fit.py` -- `prepare_targets`/`BatchedSplineFit`
  (the tested fitting machinery SPLINE's encode step reuses).
- `Spline_Basis/knot_optimization.py` -- `KnotOptimizationConfig`,
  `optimize_from_config`, `save_grid_file`/`load_grid_file` (the offline
  knot-grid optimizer; **not** part of the runtime package -- only its
  output grid file is a runtime input, see `Spline_Basis_Knot_Optimization_Plan.md`'s
  Goal #4).
- `Spline_Basis/iri_collection.py` -- builds a full-resolution training/
  validation collection for the optimizer via the main package's own
  `ensemble_init.load_or_build_edp_samples` (so the optimizer's training
  ensemble is generated exactly the way a real cycle's ensemble is).
- `Driver_Scripts/optimize_vertical_grid.py` -- the driver script that
  calls the two pieces above to produce a grid file end to end.
- `Parameterization.Parameterized_EDPSamples(..., style='SPLINE',
  hyper_params={'grid_file_path': ...})` -- the runtime consumer.

## 3. Producing a grid file in a given fit_space

A grid file is a small, self-describing JSON artifact (`grid_km`,
`degree`, `fit_space`, `alpha`, boundary-condition settings, train/
validation metrics, a `grid_file_version` guard). `fit_space`/`alpha` are
exactly what gets read back by `Parameterized_EDPSamples` at runtime.

### Option A -- edit `Driver_Scripts/optimize_vertical_grid.py`'s constants

The script currently hardcodes `FIT_SPACE = "linear"` (no `ALPHA`, since
linear doesn't need one). Change that constant (and add e.g. `ALPHA =
2.0` + pass it into the `KnotOptimizationConfig(...)` call) for `pow` or
`log10`, then rerun with a new `--output-name` so the existing grid file
isn't overwritten:

```bash
python3 Driver_Scripts/optimize_vertical_grid.py \
    --output-name grid_h60_dvar_pow_n14_full.json
```

`build_collection`'s `skip_existing=True` means this does **not** re-run
IRI2020 if the training/validation ensemble files from a previous run are
already on disk -- only the (much cheaper) knot-optimization fit itself
reruns.

### Option B -- call `KnotOptimizationConfig`/`optimize_from_config` directly

Useful when you already have the training/validation `*_edp_samples.nc`
files (e.g. from a previous `optimize_vertical_grid.py` run) and just want
a different `fit_space` without going through the driver script:

```python
from knot_optimization import KnotOptimizationConfig, optimize_from_config

cfg = KnotOptimizationConfig(
    h0_km=60.0, h1_km=900.0, n_points=14,
    delta_min_km=[[60.0, 10.0], [150.0, 10.0], [250.0, 20.0]],
    fit_space="pow", alpha=2.0,             # or fit_space="log10" (no alpha)
    representation="ratio", degree=3,
    training_files=[...],                   # existing *_edp_samples.nc paths
    validation_files=[...],
    output_path="grid_h60_dvar_pow_n14_full.json",
    members_per_column=None, metric_members=None, geo_stride=1,  # full resolution
)
result = optimize_from_config(cfg, save=True, verbose=True)
```

`fit_space="pow"` requires `alpha > 0`; `fit_space="log10"` takes no
`alpha`. Combining `fit_space="log10"` with `lower_bc=True` (the default)
emits a warning, not an error -- the lower value becomes the log10 floor,
which makes the spline overshoot near `h0`; this is a known, already
-tested characteristic (`Parameterization/test_parameterization.py` gives
log10 a looser reconstruction-error tolerance for exactly this reason),
not a bug.

## 4. Using a grid file in a run

`'SPLINE'` is a single style name -- `fit_space` only changes which grid
file you point it at, nothing else about how you declare the style:

```python
cfg.spline_grid_file_path = "/path/to/grid_h60_dvar_pow_n14_full.json"
cfg.styles = ["ANCHOR", "SPLINE"]   # or just ["SPLINE"], etc.
```

or, per-style (takes priority over `spline_grid_file_path` if both are
set):

```python
cfg.hyper_params_by_style["SPLINE"] = {"grid_file_path": "/path/to/grid_h60_dvar_pow_n14_full.json"}
```

**Altitude grid gotcha:** `CycleConfig.altitude_grid` defaults to
`default_altitude_grid()`'s 90-900 km floor. A grid file's `h0` is
typically 60 km (specifically so Ne isn't forced to 0 at 90 km), so a
`'SPLINE'` run needs `cfg.altitude_grid` extended down to match:

```python
import numpy as np
cfg.altitude_grid = np.arange(60.0, 901.0, 10.0)
```

**One grid file per style name.** Since every `'SPLINE'` entry in
`cfg.styles` shares the same style name, you cannot compare `linear` vs.
`pow` vs. `log10` SPLINE side by side within one `cfg.styles` list the way
you compare, say, `PCA_3D_10ex` vs. `PCA_3D_pow`. Run separate
`CycleConfig`s (or separate `package_run` calls), one per grid file, the
same way you'd compare any other per-style hyperparameter choice today.

## 5. Example script

`Driver_Scripts/Examples/spline_style_fit_space_example.py` loads a real,
already-built EDPSamples ensemble, builds a `'SPLINE'`
`Parameterized_EDPSamples` from a chosen grid file, prints that grid
file's `fit_space`/`alpha`, and reports the reconstruction RMSE (excluding
the structural `h0` boundary sample). Run it as-is to see the existing,
already-validated `linear` grid file in action, or point `--grid-file` at
a `pow`/`log10` grid file you produce via Section 3 to see SPLINE pick up
that fit_space with no other code change.

## 6. Verification status

- 761 repo-wide tests pass, including 12 SPLINE-specific round-trip/
  Jacobian/NetCDF tests covering all three `fit_space` values
  (`Parameterization/test_parameterization.py`).
- `jacobian_utils.py` needs no changes for any `fit_space` -- SPLINE's
  per-column Jacobian already matches its existing block-diagonal-by-geo
  -column dispatch case.
- The real-data reconstruction RMSE from the runtime path
  (`Parameterized_EDPSamples`) matches the offline knot optimizer's own
  recorded training metric for the committed `linear` grid file to many
  significant figures, confirmed against a full 2000-member, 161-geo
  -column Millstone Hill ensemble.
