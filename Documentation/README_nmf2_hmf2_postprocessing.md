# NmF2/hmF2 cross-run tabulation and visualization

Two post-processing scripts under `Postprocessing/` for comparing F2-layer
peak density (NmF2) and height (hmF2) across **many** one-hour runs --
forecast, analysis, ISR, and (when available) ionosonde -- since a single
hour's comparison isn't statistically meaningful on its own:

1. **`tabulate_nmf2_hmf2.py`** walks a results tree and writes one wide
   CSV, one row per (run, hour, condition, style, batch).
2. **`plot_nmf2_hmf2_comparison.py`** reads that CSV and produces grouped
   box plots of forecast/analysis error against ISR (by style, boost
   condition, ROI radius, or any other column in the table).

Neither script reruns the EnKF pipeline. Both work entirely from files
`package_run.py` already saved (`*_edp_samples.nc`,
`*_<style>_parameterized.nc`, `*_<style>_ensembles/ensemble_batch_*.nc`),
reusing the same forecast/analysis reconstruction and ISR/ionosonde
matching as `Assimilation_Cycle/isr_comparison.py`, and the same
`retrieve_peak` 3-point log10-parabola retrieval `Run_Scoring/score_runs.py`
already uses for its own `summary_peaks.csv`.

## What it does

### `tabulate_nmf2_hmf2.py`

1. Scans `--results-root` (via `rglob`) for every `*_edp_samples.nc` --
   works whether that root is a single run tree or a common parent
   holding many run trees side by side.
2. For each hour/condition folder found, matches the run's own ROI center
   (`Lat`/`Lon` attrs on its `*_edp_samples.nc`) against each provided
   `--isr`/`--ionosonde` station, within `--center-tolerance-deg`. A run
   with no matching station still gets a row (forecast/analysis decoded
   at the run's own ROI center); its ISR/ionosonde columns are left blank.
3. For each style present in that folder with both a `*_parameterized.nc`
   and a non-empty `*_ensembles/ensemble_batch_*.nc`, reconstructs the
   forecast/analysis ensemble at the matched (or ROI-center) location,
   and retrieves NmF2/hmF2 for forecast, analysis, and (if matched) ISR,
   all on the same altitude gates.
4. Fills in identifying/configuration columns from three independent
   sources, each column filled if its source has it and left blank
   otherwise: the hour's own `*_edp_samples.nc` attrs (most
   authoritative); the run tree's own `each_run_setting.csv` (written by
   `run_hourly_all4styles_boost_compare.py`); and, as a last resort,
   tokens parsed out of the run directory's own name (two different
   naming schemes exist on real disk; both are tried). Anything left over
   in the name is kept verbatim in `extra_tags` rather than dropped.
5. Incomplete hours/styles are skipped with a printed reason, never
   raised -- a multi-day batch over many run trees finishes and reports
   what it could.

### `plot_nmf2_hmf2_comparison.py`

For each `--group-by` column (default `style`, `condition`,
`roi_radius_km`), produces one PNG with four box-plot panels (forecast and
analysis, x NmF2 relative error (%) and hmF2 absolute error (km)), using
the same derived-error convention as `score_runs.py`'s own
`summary_peaks.csv`: `dNmF2_pct = 100*(NmF2_tag/NmF2_isr - 1)`,
`dhmF2_km = hmF2_tag - hmF2_isr`. Rows with no matched ISR value are
excluded from these plots (the console reports how many, never silently).
Style-grouped plots reuse `Assimilation_Cycle/output.py`'s
`_color_for_style`, matching every other cross-style plot in this repo.

## Prerequisites

- One or more local run trees produced by `package_run.py` /
  `run_hourly_all4styles_boost_compare.py` -- just the folders under each
  run's top-level directory (the saved `*_edp_samples.nc`,
  `*_<style>_parameterized.nc`, `*_<style>_ensembles/` files).
- A local, preprocessed ISR netCDF file per station you want matched
  (e.g. `MILISR2025_nonan.nc`), and optionally a local, preprocessed
  ionosonde netCDF file per station.
- The same Python environment used for the rest of this package
  (`numpy`, `pandas`, `xarray`, `matplotlib`).
- No display/`$DISPLAY` is required -- `plot_nmf2_hmf2_comparison.py`
  already forces matplotlib's `Agg` backend before importing `pyplot`,
  so it runs unmodified on a headless server.

## Usage

### 1. Tabulate

```bash
python3 Postprocessing/tabulate_nmf2_hmf2.py \
    --results-root /local/path/to/tomography_output \
    --isr MIL=/local/path/MILISR2025_nonan.nc \
    --isr TRO=/local/path/TROISR2025_nonan.nc \
    --ionosonde MIL=/local/path/MILIONO_Aug2530_2025.nc \
    --out /local/path/nmf2_hmf2_table.csv
```

`--results-root` may be a single run tree, or a directory holding several
run trees side by side (the same convention as
`regenerate_all_local_mil_isr_comparison_plots.py`). Add one `--isr`/
`--ionosonde` entry per station you have a file for; stations with no
matching run are simply never used, and runs with no matching station
still get a row.

### 2. Visualize

```bash
python3 Postprocessing/plot_nmf2_hmf2_comparison.py \
    --csv /local/path/nmf2_hmf2_table.csv \
    --out-dir /local/path/nmf2_hmf2_plots
```

Add `--group-by <col> [<col> ...]` to control which columns get their own
figure (any column in the CSV works, not just the three defaults -- e.g.
`--group-by style obs_sigma_tecu`).

### Options -- `tabulate_nmf2_hmf2.py`

| Flag | Default | Meaning |
|---|---|---|
| `--results-root` | *(required)* | Local directory holding one or more run trees. |
| `--isr` | none | Repeatable `CODE=PATH` to a local, preprocessed ISR netCDF. |
| `--ionosonde` | none | Repeatable `CODE=PATH` to a local, preprocessed ionosonde netCDF. |
| `--out` | *(required)* | Output CSV path. |
| `--e-top-km` | `150.0` | Lower bound of the F2 search window (same default as `score_runs.py`). |
| `--f-top-km` | `700.0` | Upper bound of the F2 search window (same default as `score_runs.py`). |
| `--center-tolerance-deg` | `0.5` | How close (degrees, lat and lon) a run's ROI center must be to a provided station to match it. |
| `--hours` | all | Restrict to specific hour-folder names (e.g. `2025-08-26_00000100`). |
| `--conditions` | all | Restrict to specific boost conditions (`no_boost`, `boosted`). |
| `--styles` | all | Restrict to specific parameterization styles. |

### Options -- `plot_nmf2_hmf2_comparison.py`

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | *(required)* | Output of `tabulate_nmf2_hmf2.py`. |
| `--out-dir` | *(required)* | Directory to write PNGs into (created if missing). |
| `--group-by` | `style condition roi_radius_km` | One figure per column named here. |

## Output

- `tabulate_nmf2_hmf2.py` writes one CSV with identifying columns
  (`run_label`, `run_dir`, `hour_label`, `window_start_utc`,
  `window_end_utc`, `batch_index`, `condition`, `style`, `station_code`),
  configuration columns (`center_lat`, `center_lon`, `roi_radius_km`,
  `grid_radius_deg`, `n_ensemble`, `obs_sigma_tecu`, `boost_amplitude`,
  `iri_downsampling_factor`, `ro_los_fraction_required`,
  `pca_discard_scale`, `extra_tags`), and metric columns
  (`NmF2_forecast`, `hmF2_forecast`, `NmF2_analysis`, `hmF2_analysis`,
  `NmF2_isr`, `hmF2_isr`, `NmF2_ionosonde`, `hmF2_ionosonde`,
  `isr_time_delta_min`, `ionosonde_time_delta_min`).
- `plot_nmf2_hmf2_comparison.py` writes one `nmf2_hmf2_by_<group_col>.png`
  per `--group-by` column.

Note: `roi_radius_km` (the physical ROI radius used for RO observation
selection, in km) and `grid_radius_deg` (the EDP grid's own extent, in
degrees) are different quantities -- both are kept because either may be
the one you want to group/filter by.

## Running on a server

Both scripts are plain CLI tools with no interactive steps, so the
straightforward way to run this on a server is the two commands above
under `nohup`/`tmux`/a batch scheduler, e.g.:

```bash
nohup python3 Postprocessing/tabulate_nmf2_hmf2.py \
    --results-root /data/tomography_output \
    --isr MIL=/data/isr/MILISR2025_nonan.nc \
    --out /data/nmf2_hmf2_table.csv \
    > tabulate_nmf2_hmf2.log 2>&1 &
```

`Driver_Scripts/Examples/nmf2_hmf2_postprocessing_example.py` wraps both
steps into one invocation (tabulate, then plot, failing loudly if either
step errors) with server-friendly defaults -- see that script for a
ready-to-edit starting point. A full tabulation over many run trees can
take a while (every batch's ensemble is reconstructed from disk); progress
is printed every 25 hour-folders processed, so redirect to a log file and
tail it rather than running with no output at all.

## Troubleshooting

- **A run tree's rows all have blank `NmF2_isr`/`hmF2_isr`/`station_code`**:
  no provided `--isr`/`--ionosonde` station is within
  `--center-tolerance-deg` of that run's own ROI center -- check the
  run's real center (its `*_edp_samples.nc` `Lat`/`Lon` attrs) against the
  station file's own attrs, or widen `--center-tolerance-deg`.
- **`roi_radius_km`/`obs_sigma_tecu`/`boost_amplitude` are blank for a
  run**: that run tree has no `each_run_setting.csv` and its directory
  name doesn't match either known naming scheme -- check `extra_tags` for
  whatever was left over; a genuinely new naming scheme needs a new case
  added to `parse_run_label_tokens`.
- **A style is skipped with "no saved ensemble batches"**: that
  style's run was incomplete on disk (e.g. still copying from Drive, or
  the run itself failed) -- re-copy/rerun before tabulating.
- **Plot step reports rows "excluded from these plots"**: those are rows
  with no matched ISR value (see the first point above); this is expected
  whenever some runs have no nearby station, not a bug.
