# Regenerating ISR-comparison plots for local Millstone Hill runs

`regenerate_all_local_mil_isr_comparison_plots.py` regenerates the
ISR-comparison plots (per-style and cross-style) for every Millstone
Hill-centered run tree it finds under a local directory, using only local
files -- no Google Drive access. It does **not** rerun the EnKF pipeline;
it only reconstructs each style's density at the ISR station from files
`package_run.py` already saved, and rewrites the derived PNG/CSV plots.

Use this instead of pointing the plots at the shared Drive directly:
bulk regeneration against Drive has been unreliable (writes have been
observed silently reverting after a delay), so copy the relevant run
folders and the ISR/ionosonde data files to local disk first, then run
this script against those local copies.

## What it does

1. Scans `--results-root` for immediate subdirectories (each one run
   tree, e.g. `MIL_aug2530_2025_R2k_..., TRO_may1821_2025_R2k_..., ...`).
2. Keeps only the subdirectories whose runs are centered on Millstone
   Hill (42.6195, -71.4917) -- checked from each run's own saved
   `Lat`/`Lon` attrs on its first `*_edp_samples.nc` file, not from
   directory names.
3. For each kept run tree, regenerates its ISR-comparison plots in place
   against the local `--isr-file` (and optionally `--ionosonde-file`),
   overwriting whatever plots were already there.

## Prerequisites

- A local copy of each run tree you want regenerated (just the folders
  under the run's top-level directory -- the script needs the saved
  `*_edp_samples.nc`, `*_<style>_parameterized.nc`, and
  `*_<style>_ensembles/` files `package_run.py` already wrote).
- A local, preprocessed ISR netCDF file (e.g. `MILISR2025_nonan.nc`).
- A local, preprocessed ionosonde netCDF file, if you want the ionosonde
  overlay (optional).
- The same Python environment used for the rest of this package
  (`xarray`, `numpy`, `pandas`, `matplotlib`).

## Usage

Check which run trees would be touched first, without writing anything:

```bash
python3 Postprocessing/regenerate_all_local_mil_isr_comparison_plots.py \
    --results-root /local/path/to/tomography_output \
    --isr-file /local/path/to/MILISR2025_nonan.nc \
    --ionosonde-file /local/path/to/MILIONO_Aug2530_2025.nc \
    --dry-run
```

Then run it for real (drop `--dry-run`):

```bash
python3 Postprocessing/regenerate_all_local_mil_isr_comparison_plots.py \
    --results-root /local/path/to/tomography_output \
    --isr-file /local/path/to/MILISR2025_nonan.nc \
    --ionosonde-file /local/path/to/MILIONO_Aug2530_2025.nc
```

Omit `--ionosonde-file` entirely to skip the ionosonde overlay.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--results-root` | *(required)* | Local directory holding one or more run trees side by side. |
| `--isr-file` | *(required)* | Local, preprocessed ISR netCDF. |
| `--ionosonde-file` | none | Local, preprocessed ionosonde netCDF. Omit to skip the overlay. |
| `--center-tolerance-deg` | `0.5` | How close (degrees, lat and lon) a run's center must be to Millstone Hill to be included. |
| `--isr-range-percentile` | `0.0` | Only affects the cycle-wide range plot's old-schema fallback; has no effect on files that already carry `Ne_min`/`Ne_max`. |
| `--dry-run` | off | List the run trees that would be regenerated; write nothing. |

## Output

For each matching run tree, under every hour/condition folder:

- `{label}_isr_comparison/{style}/` -- per-style ISR-comparison plots.
- `{label}_isr_comparison_cross_style/` -- the cross-style comparison
  plots (only written when a folder has more than one style).

Existing files at those paths are overwritten; nothing else in the run
tree is touched.

## Troubleshooting

- **"No Millstone Hill-centered run trees found under --results-root"**:
  either `--results-root` doesn't contain any run trees yet (check the
  local copy finished), or none of them are Millstone Hill-centered --
  rerun with `--dry-run` to see each subdirectory's detected center and
  why it was skipped.
- **A run tree you expected is skipped with "no \*\_edp_samples.nc
  found"**: that local copy is incomplete -- re-copy that run tree's
  folders from the source.
