# Run_Scoring User Manual

Scores hourly assimilation runs after the fact. Two scripts under
`Run_Scoring/`:

1. `extract_run_diagnostics.py` rebuilds each run's forecast and analysis
   from the files it saved: the TEC predicted for every ray, the full
   electron-density profile at each ISR site, and the overlap between the
   rays and the ISR column. It writes one `<hour>.npz` per hour folder.
2. `score_runs.py` turns those `.npz` files into per-hour metrics and
   summary tables: TEC fit, E/F relative error against ISR, NmF2/hmF2 and
   NmE/hmE, joint TEC + ISR success, and skill predictors.

Both scripts work on any number of hour folders, so new runs are scored by
pointing them at the new output folder.

## 1. Why a rebuild is needed

The hourly driver (`run_hourly_all4styles_boost_compare.py`) stores only the
analysis error at the ISR gates (`dual_isr_errors.csv`). It does not store:

- the forecast (prior) at any site other than Tromsø;
- the model TEC for each ray, so the fit to the RO data cannot be checked;
- full profiles, so NmF2/hmF2/NmE/hmE cannot be compared.

Runs made with the driver after the change of 2026-10-25 also write
`forecast_error_m3`, `isr_time_utc`, `isr_ne_m3`, `forecast_ne_m3` and
`analysis_ne_m3` to `dual_isr_errors.csv` for every station. This covers
forecast-vs-analysis at Svalbard without a rebuild. The TEC fit, the
overlap and full-column profiles still need the rebuild.

Everything above is recomputed exactly from the saved ensembles. On the
October 2025 runs (30 hours × 8 runs), the rebuilt analysis reproduces
`dual_isr_errors.csv` to a relative difference of about 1e-12 at both
Tromsø and Svalbard.

## 2. Required inputs

### 2.1 Run outputs (per hour folder)

```
<output_root>/<YYYY-MM-DD>_<HHMM><HHMM>/      e.g. 2025-10-16_14001500
    accepted_ro_observations.nc               RO rays used in the hour
    dual_isr_errors.csv                       optional; used only for the QC check
    <condition>/                              e.g. no_boost, boosted
        *_edp_samples.nc                      IRI grid (altitude, geolocation, mesh)
        *_<style>_parameterized.nc            prior ensemble + decoder
        *_<style>_ensembles/ensemble_batch_NNNN.nc   analysis ensemble(s)
```

Conditions are found automatically: any subfolder holding an
`*_edp_samples.nc`. Styles are found automatically: any style with both a
`*_parameterized.nc` and at least one `ensemble_batch_*.nc`. The last batch
file is taken as the final analysis of the hour.

> **Output mode of the driver.** `run_hourly_all4styles_boost_compare.py`
> writes these three file types by default (`--output scoring`), at about
> 1.5 GB per hour folder. A run made with `--output minimal` does not write
> them and cannot be scored by `extract_run_diagnostics.py`. `--output all`
> writes them as well as the IRI cache files (`apf107.dat`, `ig_rz.dat`,
> `*_iri_sample_inputs.pkl`).

### 2.2 ISR files

One preprocessed ISR netCDF per station, in the `*_nonan.nc` schema:
`altitude`, `Ne(altitude_gate, time)` and `time_utc`, with
`station_latitude`/`station_longitude` attributes. Each is passed as
`CODE=PATH`. The code must match the `station` column of
`dual_isr_errors.csv` (`TRO`, `SVL`) for the QC check to run. Any other
station can be added the same way.

### 2.3 Python environment

The same environment as `Assimilation_Cycle` (see
`Assimilation_Cycle_README.md` §1.1): `numpy`, `pandas`, `scipy`,
`xarray`, `netCDF4`. IRI2020 is **not** needed. The extraction script puts
the repository folders on `sys.path` itself, so run it from anywhere.

## 3. Running

### 3.1 Extract

```bash
cd /path/to/IonosphereTomography
python Run_Scoring/extract_run_diagnostics.py \
    --output-root "/path/to/tomography_output" \
    --out-dir /path/to/scoring/extract \
    --isr TRO=/path/to/TROISR2025_nonan.nc \
    --isr SVL=/path/to/SVLISR2025_nonan.nc \
    --workers 3
```

It prints one line per hour, with the QC result:

```
2025-10-16_14001500: 8 runs, QC max rel diff = 7.85e-13
```

A QC value much larger than about 1e-9 means the rebuild does not match
the run. Usual causes are a different ISR file, a different grid or ROI,
or a code version that decodes differently. Don't trust that hour's scores
until the cause is found.

Useful options:

| Option | Purpose |
| --- | --- |
| `--hours H1 H2 ...` | Process only these hour folders |
| `--conditions no_boost` / `--styles PCA_3D_10ex` | Restrict what is rebuilt |
| `--overwrite` | Redo hours whose `.npz` already exists (by default they are skipped, so an interrupted batch can simply be restarted) |
| `--workers N` | Parallel hours. Each worker needs about 2–3 GB of RAM; 3–4 fit in 16 GB. With 8 on a 16 GB machine, the batch stalls on memory swapping. |
| `--center-lat`, `--center-lon`, `--radius-km`, `--grid-radius-deg`, `--max-tec-per-batch`, `--obs-sigma` | Must match the run configuration. The defaults match `run_hourly_all4styles_boost_compare.py`. |
| `--e-top-km` | Top of the E region for the model-TEC E share (default 150 km) |

**Run time:** about 13 minutes per hour folder for 8 runs. Building the ray
operator takes about 1 minute; decoding 2 × 8 ensembles of 2,000 members
takes the rest. Reading from the Google Drive mount adds about 1.5 GB of
download per hour folder, at roughly 8 MB/s. With 4 workers, 30 hours took
about 2.5–3 hours.

### 3.2 Score

```bash
python Run_Scoring/score_runs.py \
    --npz-dir /path/to/scoring/extract \
    --out-dir /path/to/scoring/results \
    --isr TRO=/path/to/TROISR2025_nonan.nc \
    --isr SVL=/path/to/SVLISR2025_nonan.nc
```

This takes seconds. Rerun it with different thresholds or regions as often
as needed; no new extraction is required.

| Option | Default | Meaning |
| --- | --- | --- |
| `--obs-sigma` | 3.0 TECU | TEC observation error |
| `--tec-ok-sigma` | 1.5 | TEC fit counts as OK if RMS ≤ this × σ (4.5 TECU) |
| `--overfit-sigma` | 2/3 | Flags hours with RMS below this × σ (2 TECU) |
| `--e-tolerance` | 0.10 | Allowed rise in E-region relative error for joint success |
| `--e-bottom-km`, `--e-top-km`, `--f-top-km` | 90, 150, 700 | E region = [90, 150) km, F region = [150, 700) km |
| `--f-split-km` | 400 | Bottomside/topside split for the ray classes |
| `--e-valley-top-km`, `--e-valley-ratio` | 200, 1.1 | E-peak acceptance test (§4.4) |
| `--min-departure-dex` | 0.02 | Correction direction skips gates where the forecast is within this of ISR in log10 Ne (§4.7) |

## 4. Metric definitions

### 4.1 ISR matching

For each hour, the ISR profile nearest the hour's midpoint is used, and
only if it falls inside the hour. This is the rule the hourly driver uses
for `dual_isr_errors.csv`. Model profiles are interpolated in log10 Ne onto
the native ISR gates, with no extrapolation. The model profile is taken at
the station location with the grid's barycentric interpolation, the same
as in the driver.

### 4.2 TEC fit

For each run-hour: RMS of (ensemble-mean model TEC − observed TEC), for all
rays and by tangent-altitude class:

- `Eray`: tangent point below the E top (150 km).
- `Fbot`: tangent point from 150 to 400 km.
- `Ftop`: tangent point at or above 400 km.

`Eshare` is the median fraction of model TEC that comes from grid levels
below the E top, on E rays.

RO limb rays with low tangent points still cross the F region on both sides
of the tangent point. On the October 2025 runs, the E region contributes
only about 4–6% of their TEC, so the E-ray fit is a weak test of the E
region. Use `Eshare_a` vs `Eshare_f` to spot analyses that push TEC into
the E region.

### 4.3 Relative error

Mean over the ISR gates in each region of |model − ISR| / ISR, for the
forecast (`rel_f_*`) and the analysis (`rel_a_*`).

### 4.4 Peak parameters

ISR, forecast and analysis all use one retrieval, on the ISR gates:

- **NmF2 / hmF2:** the gate of maximum Ne at or above the E top, refined by
  a 3-point parabola in log10 Ne. A maximum on the first or last gate of
  the range means "no peak" (NaN).
- **NmE / hmE:** the same within the E region. The peak is accepted only if
  NmE ≥ 1.1 × the minimum Ne between the peak and 200 km, i.e. there is a
  real valley above it.

Near the F2 peak the ISR gates are 15–20 km apart, so hmF2 differences
below about 10 km are at the resolution limit. E peaks are rare: in October
2025 the ISR showed an E peak in only 3–4 hours per site. The summary
therefore reports detection counts alongside NmE/hmE errors.

### 4.5 Joint success

A run-hour at a station is a success when all three conditions hold:

1. TEC RMS ≤ 1.5σ.
2. The analysis F-region relative error is below the forecast's.
3. The analysis E-region relative error is no more than 10 percentage
   points above the forecast's.

The other three outcomes are counted too: fits TEC but misses ISR, misses
TEC but agrees with ISR, and misses both.

### 4.6 Skill predictors

The skill measure is the log of (analysis relative error ÷ forecast
relative error) in each region. The table gives its Spearman ρ against:

- RO profiles in the hour.
- TEC rays in the hour.
- **Geometric overlap:** the sum over rays and region levels of
  `H_grid @ G^T`, in TECU per 10¹¹ m⁻³. `G` maps the grid onto the ISR
  column using barycentric weights. The F-region sum takes every model
  level above the E top.
- **Rays crossing the ISR column:** rays with any weight on the column's
  grid nodes.
- **Spread reduction:** 1 − mean over region levels of (analysis ÷ forecast
  std of log10 Ne) at the site.

Negative ρ means more of the predictor goes with a larger improvement. With
about 25 hours, only |ρ| ≥ 0.4 is significant at the 5% level.

### 4.7 Correction direction

Relative error says how far the analysis is from ISR, not whether the
assimilation moved it the right way. At each ISR gate, in log10 Ne,

    k = (analysis − forecast) / (ISR − forecast)

is the fraction of the needed correction that the analysis delivers:

- `k < 0`: wrong direction. The analysis moved away from ISR.
- `0 ≤ k ≤ 1`: under-correction. The analysis lies between the forecast
  and ISR. A smaller σ may help.
- `k > 1`: over-correction. ISR lies between the forecast and the
  analysis. A larger σ may help.

Gates where |log10(ISR / forecast)| < 0.02 (about 5%) are skipped, since
there is little to correct there. `frac_{wrong,under,over}_{E,F}` are the
shares of gates in each class. `gain_{E,F}` is the least-squares k over the
region, Σ(a−f)(i−f) / Σ(i−f)², and each hour is classed by its gain.

σ changes how far the analysis moves, not which way. Wrong-direction hours
therefore point to the prior, the basis or the data, not to σ.

## 5. Outputs

### 5.1 `<hour>.npz` (from the extraction)

| Key | Shape | Content |
| --- | --- | --- |
| `y`, `tang`, `occ` | (n_rays,) | Observed TEC (TECU), tangent altitude (km), occultation index |
| `alt` | (n_height,) | Model altitude grid (km) |
| `HG_<CODE>` | (n_rays, n_height) | Ray sensitivity to the ISR column at each level |
| `<cond>\|<style>\|Yf`, `...\|Ya` | (n_rays,) | Forecast / analysis ensemble-mean model TEC |
| `<cond>\|<style>\|YEf`, `...\|YEa` | (n_rays,) | The part of that TEC from below the E top |
| `<cond>\|<style>\|<CODE>\|{f,a}{mean,std,logstd}` | (n_height,) | Site profiles: ensemble mean, std, std of log10 Ne |
| `runs`, `stations`, `window_start`, `window_end`, `obs_sigma`, `qc` | — | Bookkeeping and the QC result |

### 5.2 CSV files (from scoring)

| File | One row per | Main columns |
| --- | --- | --- |
| `hourly_metrics.csv` | hour × run × station | Everything below, unaggregated |
| `summary_tec_fit.csv` | run | Median `tec_rms_{f,a}_{all,Eray,Fbot,Ftop}`, `Eshare_{f,a}`, hours within 1.5σ, hours below 2/3σ |
| `summary_isr_relative.csv` | station × run | Median `rel_{f,a}_{E,F}_pct`, hours improved in E and F |
| `summary_peaks.csv` | station × run | Median absolute NmF2 (%) and hmF2 (km) errors for forecast and analysis, biases, hours improved, E-peak detection counts, NmE/hmE errors where both ISR and analysis have a peak |
| `summary_joint_success.csv` | station × run | `success`, `fits_tec_misses_isr`, `misses_tec_agrees_isr`, `misses_both` |
| `summary_correction.csv` | station × run × region | Median and quartiles of `gain`, hours wrong / under / over, median `frac_{wrong,under,over}` |
| `summary_predictors.csv` | station × run × region | `rho_n_ro`, `rho_n_rays`, `rho_overlap`, `rho_rays_crossing_column`, `rho_spread_reduction` |

## 6. Verification

- **Extraction:** the `qc` key and printed line compare the rebuilt
  analysis with `dual_isr_errors.csv` (§3.1).
- **Scoring:** `score_runs.py` was checked against the metrics behind the
  October 2025 summary report ("Ionosphere Tomography vs ISR: Assimilation
  Results Summary"). The match was exact for every hour and run: TEC RMS,
  relative errors, peak parameters, overlap and spread reduction.
