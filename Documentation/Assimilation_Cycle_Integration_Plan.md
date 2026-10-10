# Plan: Integrating the Ionosphere Data Assimilation Cycle

Author: scoping pass by Claude, 2026-09-23 (last major update 2026-09-24
-- see §11 for the real-data parameterization comparison across grid
resolution and ensemble size, the most consequential findings so far)
Status: **implementation complete through step 11 (all of §9 steps 1-12
built and tested against real data, 2026-09-23) — see §10 for status and
open tuning questions**

## 1. Purpose and scope

Four building blocks are now individually built and tested:

| Module | Path | Role | Status |
|---|---|---|---|
| `observation_preparation` | `observation_preparation/` | Filters RO (`ro_source.py`) and IGS (`igs_source.py`) raw data for a time window + ROI, writes netCDF (`ObservationEntry`/`netcdf_io.py`) | Done — 168 tests, verified against real Tromsø data |
| `IRI_Sample_Inputs` | `IRI_Sample_Inputs/IRI_Sample_inputs.py` | Draws driving-index (`ap`, `f107`, `ig12`, `rz12`) samples around a target date-time (`quantileSamples`) | Done |
| `EDPSamples` | `EDPSamples/edp_samples.py` | Builds the altitude/geolocation/mesh grid, runs IRI2020 to fill electron-density profiles, provides the ray-integration observation operator | Done |
| `Parameterization` | `Parameterization/Parameterization.py` | Maps EDPs ↔ parameter vectors (`raw`, `density_10ex`, `ANCHOR`, `PCA_1D`/`PCA_3D`) with Jacobians | Done |
| `Ensemble_Kalman_Engine` | `Ensemble_Kalman_Engine/` | Generic EnKF core (`EnsembleState`, `GenericObservationOperator`, `AnalysisEngine`, `GeneralEnKFDriver`) — style-agnostic, `linear`/`iterated` × `mean`/`nearest_analog` | Done — 46+ tests, incl. an OSSE recovery check against real IRI-sampled data |

Per-module wiring has been validated pairwise (`GeneralEnKFDriver` against
a real `EDPSamples` file; `observation_preparation` against real RO/IGS
data). **What does not exist yet is the layer above all five that runs an
actual assimilation cycle**: pick a time window and ROI, pull real
observations, pull a matching IRI climatology, build the ensemble, and
feed observations through `GeneralEnKFDriver` in the batched, time-ordered
way the user described. This plan scopes that layer only — none of the
five modules above need to change in their public contracts, with one
small exception noted in §5.6.

## 2. What's genuinely new here (informs prioritization)

Re-reading the two upstream plan docs' own "not yet done" sections
(`Observation_Preparation_Refactoring_Plan.md`,
`General_EnKF_Implementation_Plan.md` §8) against the user's description
surfaces four gaps that are new work, not integration glue:

1. **No chronological, source-tagged observation stream.** `observation_preparation` writes one netCDF for RO and one for IGS, each a flat `(obs, ray)` array — there is no step that merges the two, sorts by time, and hands out **batches** for sequential assimilation, while keeping a pointer back to which file/index each observation came from.
2. **No observation-error model (`R`).** Every existing `Ensemble_Kalman_Engine` test either supplies a toy `R` or is OSSE-synthetic. Nothing today derives `R` from real RO/IGS data (e.g. from SNR, arc geometry, or a fixed per-`obs_type` sigma).
3. **`GeneralEnKFDriver` has never run against real ray geometry.** `Ensemble_Kalman_Engine`'s own plan doc flags this explicitly (§8, "not yet done"): the end-to-end test uses synthesized vertical rays at `EDPSamples`'s own (lat, lon), not real `rec_ecef_km`/`gnss_ecef_km` from `observation_preparation`. This integration is the first real exercise of that seam.
4. **No multi-batch cycle loop.** `GeneralEnKFDriver.assimilate_one_cycle` runs exactly one analysis step. Running "several batches" per the user's description means calling it repeatedly, carrying the analysis ensemble forward as the next batch's prior — that loop, and the policy for what (if anything) happens to the ensemble *between* batches, doesn't exist.

Everything else in the user's description (grid/resolution choice, style
selection, IRI population, decode-to-EDP) is already exposed by existing
APIs and just needs wiring.

**Update (2026-09-23) — three more requirements, checked against what
already exists:**

5. **Run-from-precomputed-files mode.** Each stage should be able to load
   its input from a file instead of recomputing it, so the core
   assimilation loop can be re-run without repeating expensive upstream
   steps. The per-module save/load primitives mostly **already exist** —
   `IRI_Sample_Inputs.save_to_file`/`.fromPickle` (pickle),
   `EDPSamples.saveNetCDF`/`.fromNetCDF`,
   `Parameterized_EDPSamples.saveNetCDF`/`.fromNetCDF`, and
   `observation_preparation.write_observations`/`.read_observations` — so
   this is mainly plumbing in `Assimilation_Cycle/` to call them
   conditionally, not new I/O code (§4.9).
6. **Intermediate ensemble persistence.** Save the posterior `EnsembleState`
   after each batch (not just the final one) to netCDF for later analysis.
   Unlike the above, **this has no existing I/O** — `EnsembleState` is a
   bare `(X, param_shape)` dataclass with no `saveNetCDF`/`fromNetCDF`.
   New, small addition needed (§4.10).
7. **OSSE toggle for the real cycle.** An option to replace real TEC values
   with simulated ones while preserving the real GNSS/LEO geometry —
   i.e. run `observation_preparation`'s real ray geometry through
   `Ensemble_Kalman_Engine/osse.py`'s existing `generate_osse_observation`
   instead of using the real `tec` values read back from the netCDF. This
   composes two already-built pieces (real geometry, OSSE observation
   synthesis) that have never been connected — no new math, but a new
   wiring path (§4.11). Confirmed default truth source: hold out one
   ensemble member from the forecast pool, matching the pattern
   `Ensemble_Kalman_Engine`'s own OSSE tests already use.

**Update (2026-09-23) — three more requirements, checked against what
already exists:**

8. **Style selection, then a fixed small set going forward.** Initially the
   tool is used to *choose* a parameterization style by comparing several
   side-by-side on the same cycle; once chosen, only a small number of
   styles get used regularly. Nothing today runs the same cycle across
   multiple styles for comparison — `CycleConfig` as scoped so far assumes
   one `style` per run (§4.12, new).
9. **Regular DA metrics.** RMSE reduction (observation-minus-forecast vs.
   observation-minus-analysis), ensemble rank diagnostics, and profile
   visualization, run routinely rather than ad hoc. Checked what exists:
   `AnalysisDiagnostics` (§4.6) tracks convergence/iteration internals, not
   forecast/analysis skill against observations; nothing computes RMSE
   reduction, rank histograms, or effective ensemble rank anywhere in the
   codebase today. Confirmed with the user: "ranks of new ensembles" means
   **both** rank histograms (observation-vs-ensemble-spread calibration)
   and effective ensemble rank (anomaly-matrix rank/singular-value
   spectrum, monitoring collapse/degeneracy over batches) — genuinely new,
   §4.13.
10. **Vertical + horizontal EDP visualization.** `EDPSamples.plot_horizontal_field`
    already exists and covers the horizontal case. Checked for a vertical
    (density-vs-altitude) equivalent: `Parameterized_EDPSamples.plot_reconstruction_profile`
    exists but plots *parameterization reconstruction accuracy*
    (encode-then-decode error at one profile), not a forecast-vs-analysis
    or ensemble-mean-vs-spread profile view — a genuinely new plot is
    needed for that (§4.14).
11. **Intermediate results for future independent validation.** Reinforces
    §4.10: the saved per-batch ensembles (and §4.13's per-batch metrics)
    need to be self-describing enough (time window, ROI, grid, style,
    batch composition) to be matched up against an external data source
    later without needing this run's `CycleConfig` on hand — an addition
    to §4.10's netCDF attrs, not a new component.

**Update (2026-09-23) — two more additions:**

12. **RO/IGS source selection.** A config option to assimilate RO only,
    IGS only, or both. Checked `observation_stream.py` (§4.2) as scoped so
    far — it always calls both `prepare_ro_observations` and
    `prepare_igs_observations`; needs a source toggle (§4.15). The user
    also flagged a **future** refinement, not built now: filtering RO rays
    by comparing the tangent-point altitude against the altitude of the
    electron-density peak in that occultation's own Abel-retrieved
    profile (rays tangent below the peak are where the Abel inversion's
    spherical-symmetry assumption is weakest). Checked what this would
    need: `ObservationEntry.abel` already carries `Ne`/`alt_km`
    (`observation_preparation/schema.py`, populated by
    `Abel_Inverter.run_abel_inversion` via `ro_source.py`'s
    `include_abel=True`), `tangent_alt_km` is already a field, and
    `Parameterization.extract_robust_f2_peak(profile, alt_grid, ...)`
    already exists as a peak-finder — so when this is built, it's a new
    filter predicate composing three already-existing pieces, not new
    infrastructure. Recorded as a non-goal for now (§5).
13. **Abel-profile TEC consistency check.** Build a uniform 3D field by
    replicating one occultation's Abel-retrieved density profile across
    every horizontal grid point, forward-model it through the observation
    operator against real observation geometry, and compare the predicted
    TEC to the real measured TEC — the residual indicates how badly the
    Abel inversion's implicit spherical-symmetry/horizontal-homogeneity
    assumption fails in that region. New, standalone from the EnKF/ensemble
    machinery entirely (§4.16). **Confirmed (2026-09-23):** the comparison
    is TEC-vs-TEC — measured TEC vs. TEC predicted by applying the forward
    operator to the Abel-uniform density field (the earlier "measured EDP
    profile" wording was a misspeak); §4.16 below is written to that
    reading.

## 3. Proposed architecture

A new top-level package, `Assimilation_Cycle/`, standalone like
`Ensemble_Kalman_Engine/` — it imports the five survivor modules but adds
no logic to them:

```
Assimilation_Cycle/
    cycle_config.py       # CycleConfig dataclass: time window, ROI, grid, style, batching policy,
                           # per-stage precomputed-file paths, OSSE toggle (§4.1)
    observation_stream.py # RO+IGS (or RO-only/IGS-only, §4.15) -> merged, time-sorted, batched
                           # observations + R; or load from precomputed netCDF (§4.9);
                           # OSSE substitution (§4.11)
    iri_selection.py       # IRI_Sample_Inputs -> sampling_parameters DataFrame for the cycle;
                           # or load from a precomputed pickle (§4.9)
    ensemble_init.py       # EDPSamples + Parameterized_EDPSamples -> EnsembleState (the cycle prior);
                           # or load a precomputed EDPSamples/Parameterized_EDPSamples netCDF (§4.9)
    ensemble_io.py         # EnsembleState <-> netCDF (new; §4.10)
    cycle_driver.py         # the sequential batch loop; thin wrapper around GeneralEnKFDriver;
                           # saves each batch's posterior via ensemble_io if configured (§4.10)
    style_sweep.py         # run one cycle across several styles for comparison (new; §4.12)
    metrics.py              # RMSE reduction, rank histograms, effective ensemble rank (new; §4.13)
    output.py              # decode, save, diagnostics; vertical profile plots (new; §4.14)
    abel_consistency.py    # Abel-profile-as-uniform-field TEC check (new, standalone; §4.16)
    tests/
```

Data flow for one cycle:

```
CycleConfig (time window, ROI, grid res, style)
        |
        +--> observation_stream.assemble(cfg)
        |        -> observation_preparation.prepare_ro_observations / prepare_igs_observations
        |        -> read_observations() on both netCDFs
        |        -> merge + sort by `date`, tag each entry with (source_file, source_index)
        |        -> split into ordered batches -> list[(podTc2_data, y_obs, R, batch_meta)]
        |
        +--> iri_selection.sampling_parameters_for_cycle(cfg)
        |        -> IRI_Sample_Inputs(cfg.center_time).quantileSamples(...)
        |
        +--> ensemble_init.build(cfg, sampling_parameters)
                 -> EDPSamples(geo_type=..., altitude=cfg.altitude_grid,
                                sampling_parameters=..., evaluate_iri=1)
                 -> Parameterized_EDPSamples(edp_samples, style=cfg.style, hyper_params=...)
                 -> EnsembleState.from_parameterized_edp_samples(...)   # the cycle's prior

cycle_driver.run(ensemble_prior, batches, cfg)
    for each batch in chronological order:
        obs_operator = GenericObservationOperator.from_edp_samples(edp_samples, parameterization,
                                                                     ensemble.param_shape, podTc2_data)
        ensemble, diagnostics = AnalysisEngine().analyze(ensemble, obs_operator, y_obs, R, config)
        [optional inter-batch perturbation/inflation -- open question, §6]
        record diagnostics
    return final ensemble, per-batch diagnostics

output.finalize(final_ensemble, obs_operator, edp_samples, cfg)
    -> obs_operator.decode(...) -> electron density field
    -> save netCDF, plot_tec_comparison against held-out or all observations
```

## 4. Work items

### 4.1 `CycleConfig`
A dataclass capturing everything needed to reproduce a cycle: `start_time`,
`end_time`, ROI (reuse `observation_preparation.roi`'s existing
`circular_roi_points`/ROI-dict convention so the same ROI drives both
observation filtering and the `EDPSamples` `Regional` grid), `altitude_grid`
(a full 1-D array, not min/max/step — §8.4 confirms the vertical grid may
be non-uniform in the future, so the config always carries an explicit
array; default `np.arange(0, 901, 10)`, i.e. 0-900 km at 10 km spacing),
horizontal grid spacing (`dLat`/`dLon`, default 2.5° angular resolution on
the Earth's surface, §8.4), `style` + `hyper_params`, `n_ensemble`,
`batch_size` (§8.1 — a pure compute-chunking parameter, see §4.2), a
constant `obs_sigma` (§8.2), and an output directory (per the
[[tomography-test-output-folder]] convention for any diagnostic-run
artifacts this produces during development).

Additional fields for §4.9-§4.11: optional precomputed-input paths
(`ro_observations_path`, `igs_observations_path`, `iri_sample_inputs_path`,
`edp_samples_path`/`parameterized_edp_samples_path`) — each `None` by
default (compute fresh) or a path to load from instead; matching
`*_output_path` fields so a fresh computation can be saved for reuse next
time; `save_intermediate_ensembles: bool` + `ensemble_output_dir` (§4.10);
`osse_mode: bool` + `osse_rng_seed` (§4.11, for reproducible synthetic
noise). `style`/`hyper_params` stay single-valued on `CycleConfig` itself
(§4.12's sweep varies them externally, passing one modified `CycleConfig`
copy per style into `cycle_driver.run` rather than `CycleConfig` carrying
a list). `obs_sources: Literal["RO", "IGS", "both"]`, default `"both"`
(§4.15).

### 4.2 Observation assembly, merge, and batching (`observation_stream.py`)
- Call `prepare_ro_observations`/`prepare_igs_observations` for the
  cycle's time window + ROI (reusing `full_los` ROI mode, the confirmed
  default), gated by `cfg.obs_sources` (§4.15) — skip the RO or IGS call
  entirely when the corresponding source is excluded, rather than calling
  both and discarding one afterward (avoids the wasted parse/Abel-inversion
  cost when RO is excluded).
- `read_observations()` both resulting netCDFs back into `ObservationEntry`
  lists; each entry already carries `date`, `obs_type`, and enough identity
  (`label`, `rec_id`) to serve as the "pointer back to the original
  dataset" the user asked for — no new field needed, just a merge step.
- Merge + sort by `date` into one chronological sequence. There is no
  state-transition/dynamical model in this filter (§8.1) — the state is
  constant over the cycle — so this ordering has no effect on the math
  *within* one cycle today; it exists to make cross-cycle continuity
  (§8.5) possible later without restructuring this step.
- Split into batches of `cfg.batch_size` consecutive (in time) observations.
  Since there's no state transition, batch boundaries are purely a
  compute-chunking device (§8.1) — sized for `AnalysisEngine`/`iterated`
  mode's per-batch Gauss-Newton cost and available memory, not for any
  physical or temporal meaning. Each batch becomes one `podTc2_data` dict
  (`rec_ecef_km`/`gnss_ecef_km` stacked across the batch's rays) + `y_obs`
  (`tec`) + `R`.
- Build `R = cfg.obs_sigma**2 * I`: a single constant scalar variance,
  same across `obs_type` and independent between observations (§8.2 —
  representation error dominates the physical measurement error here, so
  a constant is the right level of fidelity for now; no SNR/geometry-based
  refinement planned).

### 4.3 IRI driving-index selection (`iri_selection.py`)
Thin wrapper: `IRI_Sample_Inputs(cfg.center_time_str).quantileSamples(...)`
to produce the `sampling_parameters` DataFrame `EDPSamples.__init__`
expects. `quantileSamples`'s `hour_sample_range`/`f107_sample_range`/etc.
control ensemble spread around the cycle's actual conditions — these map
onto `cfg.n_ensemble`/spread settings.

### 4.4 Ensemble construction (`ensemble_init.py`)
`EDPSamples(DateTime=cfg.center_time_str, geo_type="Regional", altitude=cfg.altitude_grid,
sampling_parameters=..., evaluate_iri=1, Lon=cfg.roi_lon, Lat=cfg.roi_lat, radius=cfg.roi_radius,
dLat=cfg.horizontal_resolution)` → `Parameterized_EDPSamples(edp_samples, style=cfg.style,
hyper_params=cfg.hyper_params)` → `EnsembleState.from_parameterized_edp_samples(...)`.

This runs IRI2020 once per ensemble member over the full horizontal ×
vertical grid — cost scales with `n_ensemble × n_horizontal_grid`; worth a
timing check at realistic cycle grid sizes before committing to a default
`n_ensemble` (flagged in §6).

### 4.5 Driver split (small addition to `Ensemble_Kalman_Engine`)
`GeneralEnKFDriver.build_ensemble_and_operator` currently bundles ensemble
construction with a *single* observation operator — fine for the
one-shot OSSE test it was built for, but a multi-batch cycle needs the
ensemble built once (§4.4) and a fresh `GenericObservationOperator` built
per batch. Proposed: split it into
`GeneralEnKFDriver.build_ensemble(edp_samples)` and
`GeneralEnKFDriver.build_observation_operator(edp_samples, ensemble, podTc2_data, num_segments)`,
same bodies, no behavior change — `build_ensemble_and_operator` can stay
as a thin convenience wrapper calling both, so existing tests keep passing
unmodified.

### 4.6 Multi-batch cycle loop (`cycle_driver.py`)
For each batch in chronological order: build the batch's observation
operator (§4.5), call `AnalysisEngine().analyze(...)` (via the split
driver), carry the analysis ensemble forward as the next batch's prior —
plain persistence between batches, no inflation or perturbation (§8.3;
consistent with §8.1's constant-state model). Record `AnalysisDiagnostics`
per batch.

### 4.7 Output and diagnostics (`output.py`)
`obs_operator.decode(ensemble.to_param_shape())` → save as netCDF matching
`EDPSamples`'s own schema (so it round-trips through `plot_horizontal_field`
etc.); run `observation_preparation.plot_tec_comparison` (already built,
already validated against `EDPSamples`) as the primary QC check —
measured vs. `H @ (posterior mean density)` residuals, by `obs_type`.

### 4.8 Validation
- Extend the existing OSSE utility (`Ensemble_Kalman_Engine/osse.py`,
  currently single-batch) to a multi-batch cycle-level recovery check:
  known truth field, synthetic observations split into batches in time
  order, verify the posterior converges toward truth batch-by-batch.
- Real-data smoke test using the already-verified Tromsø RO/IGS sample
  data (`/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/`)
  — the first true end-to-end run: real geometry, real observations, real
  IRI climatology, through the full cycle loop.

### 4.9 Run-from-precomputed-files mode
Each of §4.2/§4.3/§4.4's stages gets an early-return: if the corresponding
`CycleConfig` path is set and the file exists, load instead of compute.

| Stage | Compute path | Load path (existing primitive) |
|---|---|---|
| Observations | `prepare_ro_observations`/`prepare_igs_observations` | `read_observations(cfg.ro_observations_path)` / `(cfg.igs_observations_path)` |
| IRI indices | `IRI_Sample_Inputs(cfg.center_time_str)` | `IRI_Sample_Inputs.fromPickle(cfg.iri_sample_inputs_path)` |
| EDPSamples ensemble | `EDPSamples(..., evaluate_iri=1)` (the expensive one — one IRI2020 run per member) | `EDPSamples.fromNetCDF(cfg.edp_samples_path)` |
| Parameterized ensemble | `Parameterized_EDPSamples(edp_samples, style=...)` | `Parameterized_EDPSamples.fromNetCDF(cfg.parameterized_edp_samples_path)` |

Symmetrically, whichever stages *do* compute fresh should save their
result to the configured output path (when given) before continuing, so a
later run can reuse it. This makes the EDPSamples/IRI2020 step in
particular — the dominant cost per §4.4 — a one-time expense per
(time window, ROI, grid) combination rather than a per-experiment one,
which is the main practical motivation: iterating on the assimilation
math (batch size, style, `R`) without re-running IRI2020 every time.

No new I/O code required — this item is entirely about `Assimilation_Cycle/`
calling the four modules' existing save/load methods conditionally.

### 4.10 Intermediate ensemble persistence (`ensemble_io.py`)
New: `EnsembleState` has no netCDF I/O today. Add
`save_ensemble_netcdf(ensemble: EnsembleState, path, *, style=None, hyper_params=None)`
and `load_ensemble_netcdf(path) -> EnsembleState`, storing `X` (dims
`state, member`), `param_shape` (as an attr, since it's just a tuple of
ints), and — to make a saved file self-describing enough to `decode()`
later without separately remembering the run's configuration — `style`/
`hyper_params` as attrs when available. Per requirement 11 (§2), also
carry enough of the cycle's identity for later independent validation
without the original `CycleConfig`: `cycle_start_time`/`cycle_end_time`,
ROI (center/radius or bbox), grid (`altitude_grid`, horizontal resolution),
`batch_index`, and the batch's time span — cheap to attach (already all in
`CycleConfig`/batch metadata by this point), and the whole reason this
item exists is to let a saved file stand on its own.

`cycle_driver.py`'s batch loop (§4.6) calls this after every batch when
`cfg.save_intermediate_ensembles` is set, writing to
`cfg.ensemble_output_dir` with a batch-indexed filename (e.g.
`ensemble_batch_0003.nc`) — one file per batch, not one growing file, so a
partial/crashed run still leaves earlier batches usable.

### 4.11 OSSE mode (real geometry, simulated TEC)
A `cfg.osse_mode` toggle in `observation_stream.py`/`cycle_driver.py`:
after building each batch's `podTc2_data` from real `rec_ecef_km`/
`gnss_ecef_km` (unchanged — geometry stays real) and constructing that
batch's `GenericObservationOperator` (§4.5), replace the real `y_obs`
(`tec` read from `observation_preparation`) with
`generate_osse_observation(x_true, obs_operator, R, rng).y_obs` from
`Ensemble_Kalman_Engine/osse.py`.

**Truth source (confirmed):** hold out one member of the forecast
ensemble built in §4.4 as `x_true` before assimilation starts, removing it
from the forecast pool — the same convention `Ensemble_Kalman_Engine`'s
own OSSE tests already use, so no new truth-field input is required. A
fixed `cfg.osse_rng_seed` makes a given OSSE run's synthetic noise
reproducible. `x_true` is held fixed across all batches in the cycle
(consistent with §8.1's constant-state model) and scored against the
final posterior via `osse.py`'s existing `recovery_error`.

### 4.12 Style-selection sweep (`style_sweep.py`)
`run_style_sweep(cfg, styles: list[str]) -> dict[str, CycleResult]`: runs
the identical cycle (same observations, same IRI-drawn driving indices,
same grid) once per style in `styles`, varying only `style`/`hyper_params`
per §4.4/§4.5. Reuses §4.9's precomputed-file mode so the (expensive)
observation assembly and IRI2020-based EDP draw happen once and are
shared across styles rather than repeated per style — only the
`Parameterized_EDPSamples` encode step and the assimilation loop differ
per style. Returns one `CycleResult` (final ensemble + §4.13 metrics) per
style, for side-by-side comparison; once a small set of styles is settled
on, later regular runs just call `cycle_driver.run` directly with a fixed
`style` (or run the settled set with this same sweep function, ongoing,
as a routine cross-check).

### 4.13 Regular DA metrics (`metrics.py`)
Computed per batch (using the batch's `GenericObservationOperator`, the
forecast ensemble going in, and the analysis ensemble coming out) and
accumulated over the cycle:
- **RMSE reduction**: `rmse(y_obs - H @ forecast_mean)` vs.
  `rmse(y_obs - H @ analysis_mean)`, overall and split by `obs_type`
  (RO/IGS) — the basic "did assimilation help" number.
- **Rank histograms**: for each observation, the rank of `y_obs` within
  the sorted `H @ forecast_ensemble` values (ties broken randomly, the
  standard convention) — accumulated across all observations in a batch,
  and across batches in a cycle, into a histogram checking ensemble
  calibration (uniform = well-calibrated; U-shaped = under-dispersive;
  peaked = over-dispersive).
- **Effective ensemble rank**: numerical rank (or full singular-value
  spectrum, via SVD of the ensemble anomaly matrix `X - mean`) of the
  forecast and analysis ensembles at each batch — a time series across
  the cycle, to catch rank collapse/degeneracy as batches accumulate
  (ties into the existing N-vs-n rank-deficiency discussion,
  `General_EnKF_Implementation_Plan.md` §6.6).

Metrics accumulate into a `CycleMetrics` object returned alongside the
final ensemble; `output.py` (§4.14) is responsible for saving/plotting
them, `metrics.py` only computes.

### 4.14 Visualization (extends `output.py`)
- **Horizontal**: reuse `EDPSamples.plot_horizontal_field` directly —
  already built, already handles a scalar field over the grid at a target
  altitude/sample.
- **Vertical (new)**: a profile plot (density vs. altitude) at a chosen
  geo location, distinct from `Parameterized_EDPSamples.plot_reconstruction_profile`
  (which shows encode/decode error, not forecast-vs-analysis) — this one
  shows forecast ensemble mean ± spread vs. analysis ensemble mean ±
  spread at one or more grid points, the natural companion to §4.13's
  RMSE-reduction numbers.
- **Metrics plots**: RMSE-reduction bar/line chart (by `obs_type`, by
  batch), the rank histogram, and the effective-rank time series from
  §4.13.
All figures follow the [[tomography-test-output-folder]] convention for
development runs; `cfg.output_dir` for real ones.

### 4.15 RO/IGS source selection
`cfg.obs_sources` (§4.1) gates §4.2's assembly: `"RO"` skips the IGS call,
`"IGS"` skips the RO call, `"both"` (default) runs both as already scoped.
Small, additive change to §4.2 only.

### 4.16 Abel-profile TEC consistency check (`abel_consistency.py`)
A validation tool, independent of the rest of the assimilation machinery
(no ensemble, no `Parameterization`, no `Ensemble_Kalman_Engine` at all —
just `EDPSamples`'s own forward operator applied to a plain density
array):

1. Take one RO `ObservationEntry`'s `abel` dict (`Ne`, `alt_km` —
   already populated by `Abel_Inverter.run_abel_inversion` via
   `ro_source.py`).
2. Interpolate `Ne(alt_km)` onto the cycle's `cfg.altitude_grid` and
   replicate it across every horizontal grid point, producing a
   `(n_height, n_geo, 1)` uniform-field array — "if the whole ROI looked
   like this one profile."
3. Forward-model it through `EDPSamples.get_observation_operator`/
   `forward_model_mesh_tec` against real observation geometry — the rest
   of the cycle's assembled `podTc2_data` (§4.2), not just the source
   occultation's own rays (checking against *other* geometries is the
   point; the source occultation's own rays are close to a self-check
   already provided by `Abel_Inverter`'s own `TEC_forward` field).
4. Compare predicted vs. real measured TEC (see the wording note in §2,
   item 13) — reuses the same "`H @ density` vs. `y_obs`" comparison
   pattern as §4.13's RMSE metric / `observation_preparation.plot_tec_comparison`,
   just with the Abel-uniform field standing in for an ensemble mean.

Since this needs no ensemble or parameterization, it can be built and
tested early, independent of §4.4-§4.6, using any cycle that has both RO
observations (for the source Abel profile) and other observations (RO or
IGS) to check against.

## 5. Non-goals (for this pass)

- No propagation/dynamical model between *cycles* (i.e. this plan is for
  one assimilation cycle's internal batch loop). §8.5 confirms a future
  multi-cycle system is planned, where each cycle's prior is the previous
  cycle's posterior **augmented** with fresh IRI2020 draws for the new
  time — an approximate proxy for state transition, not a literal dynamical
  model. That's out of scope for this plan, but it means `ensemble_init.py`
  (§4.4) should keep "draw a fresh IRI-based ensemble" as a separable step
  from "construct an `EnsembleState`" rather than fusing them, so a later
  cycling driver can call it in an augmentation role (mix fresh IRI draws
  into an existing posterior) without restructuring this module — noted as
  a design constraint now, not built now.
- No change to `observation_preparation`, `EDPSamples`, `Parameterization`,
  or `Ensemble_Kalman_Engine`'s existing public contracts beyond the
  additive, backward-compatible split in §4.5.
- No UI/CLI beyond a `CycleConfig` + a `run_cycle(cfg)` entry point — a
  command-line wrapper can come later once the shape is validated.
- No tangent-point-vs-Abel-peak-altitude filtering yet (§2, item 12) — a
  future refinement to §4.2/§4.15's RO selection, filtering out rays whose
  tangent point sits below that occultation's own Abel-retrieved F2 peak.
  Not built now, but the pieces it would need already exist:
  `ObservationEntry.abel["Ne"]`/`["alt_km"]`, `ObservationEntry.tangent_alt_km`,
  and `Parameterization.extract_robust_f2_peak` as the peak-finder — so
  this is a new filter predicate to add later, not new infrastructure.

## 8. Decisions (resolved 2026-09-23)

1. **Batching policy.** There is no state-transition model in this filter
   — the state is constant over the cycle by design (the "Kalman filter"
   name notwithstanding). Sorting observations by time exists solely to
   enable smooth cycle-to-cycle transitions later (§8.5), not to give
   batch order any effect within a cycle today. `batch_size` is therefore
   chosen purely to optimize computational efficiency against available
   resources — a performance tuning parameter, not a physical one.
2. **Observation-error model (`R`).** A single constant scalar `obs_sigma`
   (same for RO and IGS), accounting for physical measurement error and
   (dominantly) representation error; observations are independent, so
   `R = obs_sigma**2 * I`. No per-`obs_type` or SNR-based refinement.
3. **Inter-batch ensemble treatment.** Plain persistence, no inflation —
   consistent with decision 1's constant-state model. Explicitly flagged
   for re-evaluation later, not a permanent design commitment.
4. **Grid/resolution defaults.** Both vertical and horizontal resolution
   are user-specified `CycleConfig` fields. Vertical grid is a full array
   (supporting a non-uniform grid in the future, not just min/max/step),
   defaulting to `np.arange(0, 901, 10)` (0-900 km, 10 km steps).
   Horizontal grid defaults to 2.5° angular resolution on the Earth's
   surface.
5. **Single cycle vs. cycling system.** Confirmed: a sequence of cycles is
   planned for the future, each inheriting the previous cycle's state and
   augmenting it with fresh IRI2020 values for the new time (the
   approximate state-transition proxy noted in §5). This plan still scopes
   only the single-cycle batch loop; the cycling driver is a follow-up.

## 9. Suggested sequencing

1. §4.1 `CycleConfig` + §4.5 driver split (small, no new behavior, unblocks everything else).
2. §4.2 + §4.15 observation assembly/merge/batching with the constant-`R` model (§8.2) and the RO/IGS/both source toggle (cheap to include from the start, same call site).
3. §4.3 + §4.4 IRI selection and ensemble construction, at the §8.4 default grid (10 km/0-900 km vertical, 2.5° horizontal) — timed to check IRI2020 cost at that resolution before committing it as the real default.
4. §4.9 precomputed-file load/save, right after each stage it applies to exists — cheap to add immediately and pays for itself during the timing check in step 3 (no need to re-run IRI2020 while iterating on later steps).
5. §4.6 cycle loop with persistence (§8.3).
6. §4.10 intermediate ensemble persistence, added to the loop from step 5.
7. §4.11 OSSE mode — needs §4.6's loop and §4.4's ensemble (for the held-out truth member) in place first.
8. §4.13 metrics (RMSE reduction, rank histograms, effective rank) — needs step 5's loop (forecast/analysis ensembles per batch) and benefits from step 7 (OSSE gives a known-truth check that the metrics themselves are computed correctly before trusting them on real data).
9. §4.7 + §4.14 output/diagnostics and visualization (horizontal reuses existing `plot_horizontal_field`; vertical profile plot and metrics plots are new, built alongside step 8's numbers).
10. §4.12 style sweep — thin orchestration layer once steps 1-9 work for a single style; this is also the point the user's stated initial use case (comparing styles to choose one) becomes usable end-to-end.
11. §4.8 validation — OSSE multi-batch recovery check (now straightforward given step 7-8), then the real Tromsø-data smoke test (can use step 4's precomputed-file mode to avoid re-running observation prep/IRI2020 between iterations), then a real style sweep (step 10) as the first actual use of the tool for its stated initial purpose.
12. §4.16 Abel-profile TEC consistency check — independent of the rest (needs only §4.2's real RO/IGS observations, no ensemble), so it can actually be pulled forward and done any time after step 2 if useful sooner; listed last only because it's not on the critical path to a working assimilation cycle.

## 10. Implementation status (2026-09-23)

**Built and tested (steps 1-10, 12 above; 60 new tests in
`Assimilation_Cycle/tests/`, full suite incl. `Ensemble_Kalman_Engine`/
`EDPSamples`/`Parameterization`/`IRI_Sample_Inputs` at 311 passed):**

- `cycle_config.py` (`CycleConfig`, step 1's config half).
- `Ensemble_Kalman_Engine/driver.py`'s `build_ensemble`/`build_observation_operator`
  split (step 1's driver half) — additive, `build_ensemble_and_operator` kept
  as a thin wrapper, all 46 pre-existing `Ensemble_Kalman_Engine` tests still pass.
- `observation_stream.py` (steps 2, 4, 15): RO/IGS assembly, chronological
  merge, batching, source-selection toggle, precomputed-file load/save —
  tested against synthetic `ObservationEntry`s and monkeypatched
  `prepare_ro_observations`/`prepare_igs_observations` (no real RO/IGS
  data files needed for these tests).
- `iri_selection.py` (steps 3, 4): uses `IRI_Sample_Inputs.randomSamples`,
  not `quantileSamples` (design note recorded in the module itself — see
  §4.3's docstring cross-reference), so `cfg.n_ensemble` has a direct
  effect; precomputed-file mode via the module's own existing
  `save_to_file`/`fromPickle`.
- `ensemble_init.py` (steps 3, 4): `EDPSamples` + `Parameterized_EDPSamples`
  + `EnsembleState`, all three stages precomputed-file-aware. Tested
  against **real IRI2020** (this session set `IRI2020_PATH` to the
  repo's own compiled `iri2020_new/src/iri2020/iri2020_namelist_driver`
  and confirmed it runs), not just mocked.
- `ensemble_io.py` (step 6): new `EnsembleState` netCDF I/O, self-describing
  per plan Section 2 item 11 (cycle/batch/ROI/grid identity as attrs).
- `cycle_driver.py` (steps 5, 6, 7): `run_batch_loop` (toy-operator tested,
  same pattern as `Ensemble_Kalman_Engine`'s own tests) + `run_cycle` (the
  real entry point, building operators from real `EDPSamples` geometry).
  Persistence (no inter-batch inflation), intermediate-ensemble saving,
  and OSSE substitution (held-out member, fixed across all batches) all
  verified.
- `metrics.py` (step 8): RMSE reduction, rank histograms (verified against
  a synthetic well-calibrated case giving a near-uniform histogram, and an
  under-dispersive case giving the expected U-shape), effective ensemble
  rank (verified full-rank/collapsed/rank-deficient-by-construction cases).
- `output.py` (step 9): horizontal plotting reuses `EDPSamples.plot_horizontal_field`
  directly (it already accepts a raw array, no wrapper object needed);
  vertical profile plot and the three metrics plots are new; decoded-field
  netCDF save reconstructs a real `EDPSamples` object via the same
  `geo_type`-specific attrs the original was built with (`feature_edps`
  zero-filled and flagged `feature_edps_computed=0`, since those come from
  IRI2020's own Fortran output and aren't derivable from an assimilated
  field).
- `style_sweep.py` (step 10): shares one `EDPSamples` IRI draw across
  styles, verified the shared build happens exactly once per sweep, not
  once per style.
- `abel_consistency.py` (step 12): verified against the same analytic
  radial-ray TEC oracle `EDPSamples/test_edp_samples.py`'s own
  `TestLineOfSightTEC` uses (constant density on a purely radial ray has
  an exact closed-form TEC) — both the "consistent" case (residual ~0)
  and a deliberately mismatched case (large residual) are checked.

**Two real, reproducible bugs found and fixed in `EDPSamples/edp_samples.py`
while wiring real IRI2020 runs through `ensemble_init.py`** (both
surfaced only by driving actual IRI2020 calls, not by reading the code):
1. `write_IRI2020_namelist` used `np.isnan(...)` on
   `sampling_parameters` columns that can legitimately be all-`None`
   (pandas leaves an all-`None` column as `object` dtype, not `float64`+NaN,
   whenever a driving index has zero spread requested) — `np.isnan(None)`
   raises `TypeError`. Fixed by switching to `pd.isna(...)`, which handles
   both a real NaN float and a bare `None` correctly. Regression-tested
   (`test_ensemble_init.py::test_zero_spread_on_every_index_no_longer_crashes`).
2. `EDPSamples.__init__` built `sample_param_value` via a bare
   `sampling_parameters.to_numpy()`; when driving-index columns have mixed
   native types (a real `apf107`/`ig_rz` source mixes int-valued `ap` with
   float-valued `f107`/`ig12`/`rz12`), pandas can only represent that as a
   `dtype=object` 2-D array, which xarray's netCDF writer cannot infer a
   dtype for and rejects outright (`saveNetCDF` raised `ValueError`).
   Fixed by requesting `dtype=np.float64` explicitly in `to_numpy(...)`
   (also converts any remaining `None` to `np.nan`, consistent with fix 1).

**A real, end-to-end demo run** (`density_10ex` style, real IRI2020 over a
17-height x 14-geo grid, 100 members, 6 batches of synthetic-geometry/
real-forward-model OSSE-style observations) reduced batch RMSE from
0.06-0.89 TECU (forecast) to 0.03-0.08 TECU (analysis) in every batch.
Artifacts (decoded field netCDF, intermediate per-batch ensembles,
vertical/horizontal/RMSE-reduction/rank-histogram/effective-rank plots)
saved to
`/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/`.

**A third real bug found while wiring step 11 (unit mismatch, not a
crash):** `ensemble_init.load_or_build_edp_samples` passed `cfg.radius_km`
straight through as `EDPSamples`'s `radius=` for `geo_type="Regional"` —
but `radius_km` is kilometers (matching `observation_preparation`'s
convention) while `EDPSamples.genRegionalArea`'s `radius` is **degrees**.
Invisible in every test/demo so far because the small test values (5-8)
happened to double as plausible degree numbers; would have silently built
a nonsensical multi-thousand-degree grid (or crashed) against the real RO
ROI's `radius_km=2000`. Fixed by adding a separate `CycleConfig.grid_radius_deg`
field (native degrees, no conversion constant needed) with a documented,
clearly-labeled fallback to `radius_km`'s bare number for backward
compatibility with the existing small-grid tests — `resolved_grid_radius_deg`
is what `ensemble_init.py` now actually reads. All 60 `Assimilation_Cycle`
tests still pass unchanged after this fix (they exercise the fallback
path); a real cycle should set `grid_radius_deg` explicitly, as step 11
below now does.

**Step 11, part 1 (RO only) — done, real data, real geometry:** first
diagnosed why this had stalled (see below), then ran the real thing: real
RO occultation geometry from `/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/RO_Data`
(the Tromsø sample set, 15 occultations, same parameters as
`observation_preparation/test_ro_preparation.py`'s known-good real-data
run) through `observation_stream.assemble` -> real IRI2020 via
`ensemble_init.build` (17 heights x 12 geo points, 100 members,
`density_10ex`) -> `cycle_driver.run_cycle`, 3 batches of ~840-920 real
rays each. RMSE dropped every batch: 46.3->11.6, 52.6->32.0, 29.4->18.8
TECU (forecast->analysis). This is the first exercise of `run_cycle`
against genuine occultation geometry rather than the synthesized vertical
rays used everywhere else so far (the caveat `Ensemble_Kalman_Engine`'s
own end-to-end test already carried). Diagnostics did flag
`converged=False` for the iterated Gauss-Newton loop in all 3 batches —
plausibly because `obs_sigma=1.0` TECU (fine for small synthetic tests)
is unrealistically tight against real forecast-observation residuals of
30-50 TECU; widened to `obs_sigma=3.0` for part 2 below, a real tuning
question to revisit with the user once both parts are in. Artifacts in
`.../Data_Assimilation_Cycle/step11_ro/`.

**What was actually blocking step 11 (corrected from the earlier,
imprecise "collection hangs" characterization):** not a `pytest`
collection issue and not network/CDDIS access — confirmed via `lsof`
(zero open internet sockets throughout either run). RO against the local
data completes in ~15s. IGS against the local data takes several minutes
per run, not because of a hang: `TEC_model.igs_tec_pipeline.process_igs_station`
has no way to pass a time window in — it computes satellite ephemeris and
carrier-phase-leveled TEC for an entire day (30-second cadence, ~2880
epochs x several satellites, including GLONASS's iterative numerical-
integration ephemeris) before `filter_igs_by_time` narrows to the
requested window afterward. This is a pre-existing, deliberate
characteristic from the earlier `observation_preparation` refactor
session (documented in that session's own memory: `tlim` was dropped
from `process_igs_station`'s call path on purpose), not a regression from
anything in this session, and not something to fix without asking first.

**Step 11, part 2 (RO+IGS together) — done**, using the exact local-file
parameters from `observation_preparation/test_igs_preparation.py`'s
known-good real-data run (`local_obs_by_station`/`local_nav`/`local_dcb`,
no CDDIS access), `obs_sigma=3.0` (widened per part 1's finding). Real
RO+IGS merged into 5 chronological batches (2640 rays total) — RO
dominates the first three batches (685-1211 rays each) and simply runs
out after 10:35:30 in this particular hour's local data, leaving the last
two batches IGS-only (6-8 rays each); `observation_stream`'s merge/batch
logic handled this mixed-source, uneven-coverage real case correctly with
no special-casing needed. Results:

| batch | n_obs | RO/IGS | forecast RMSE | analysis RMSE | converged |
|---|---|---|---|---|---|
| 0 | 1212 | 1211/1 | 56.99 | 28.62 | False |
| 1 | 689 | 685/4 | 36.67 | 20.30 | False |
| 2 | 725 | 721/4 | 19.64 | 17.41 | False |
| 3 | 8 | 0/8 | 14.57 | 14.54 | **True** |
| 4 | 6 | 0/6 | 29.08 | 29.05 | **True** |

RMSE improved in every batch except 4 (essentially flat, 29.08->29.05) —
consistent with batch 4's very sparse information content (6 IGS rays
against a ~1200-dimensional state) giving the analysis little to work
with, not a wiring problem. The `converged=True`/`False` split lines up
exactly with batch size (small IGS-only batches converge within the
Gauss-Newton cap, the large RO-heavy batches don't) — reinforcing part
1's `obs_sigma` hypothesis rather than pointing at a new issue. Effective
ensemble rank stayed near-full (96-97 of a possible 99) throughout, no
sign of collapse. `by_obs_type` splits (in the table's underlying data)
show RO and IGS residuals moving independently within a batch, e.g. batch
1's IGS component alone got slightly *worse* (21.2->22.9) while RO in the
same batch improved sharply (36.7->20.3) — the net batch RMSE still fell
because RO dominates that batch's observation count; worth watching if a
future run has an IGS-heavy batch instead. Artifacts (5 intermediate
ensembles, decoded field, all 4 plot types) in
`.../Data_Assimilation_Cycle/step11_both/`.

Step 11 is now complete — both parts ran against real local Tromsø data,
real ray geometry, real IRI2020, through the full cycle.

**Step 11 follow-up — Gauss-Newton step-size investigation (2026-09-23):**
user asked whether the non-convergence was a step-size (line-search
`alpha`) issue, correctly noting `obs_sigma` and step size are different
parameters (confirmed: `obs_sigma` only enters via `R`, which sizes the
Kalman gain `K_i` and therefore the *proposed* correction's magnitude;
the actual step size is the backtracking line search's `alpha`
(`alpha0`/`alpha_backtrack_factor`/`max_backtracks`), already adaptive).
Reproduced a real non-converged batch (RO batch 0, 861 rays) with full
diagnostics and tried the user's hypothesis directly: more backtracks and
more max_iterations both gave an *identical* result (proving neither cap
was the limiter — the loop was hitting its "no damped step improved the
residual across the whole backtracking range" safety break on the 3rd
Gauss-Newton iteration, not exhausting either budget); a smaller
`alpha0=0.3` did let it take one more step but landed at a *worse*
residual. Conclusion: not simply a mistuned step size — the local
linearization appears to stop being a useful guide at that point for
this real, nonlinear problem, which is what the *all-styles* run below
partially explains (see the convergence-by-style pattern).

**All 7 parameterization styles run against real data (2026-09-23),
using the precomputed-file mode end to end:** user asked (1) whether all
registered styles could be tested against real data, and (2) whether new
tests could reuse saved observations instead of repeating observation
preparation. Both done in two steps:

1. `observation_stream.assemble(cfg)` run once with
   `ro_observations_output_path`/`igs_observations_output_path` set
   (Section 4.9) -- real RO+IGS from the local Tromsø data (5 batches,
   2640 rays, same parameters as step 11 part 2) saved to
   `step12_all_styles/{ro,igs}_observations.nc`.
2. A second, separate run with `ro_observations_path`/`igs_observations_path`
   pointing at those files loaded them in **~2.3s** (vs. several minutes
   to recompute IGS) -- confirmed no RO/IGS reprocessing happened. Then
   all 7 styles (`raw`, `density_10ex`, `ANCHOR`, `PCA_1D`,
   `PCA_1D_10ex`, `PCA_3D`, `PCA_3D_10ex`) ran against the identical
   loaded batches and one shared real-IRI2020 `EDPSamples` draw (built
   once, reused across all 7 -- the same sharing `style_sweep.py` does
   internally, reimplemented as an explicit per-style try/except loop
   here so one style's failure wouldn't abort the rest).

**Result: all 7 styles ran successfully, zero crashes, zero decode-field
save failures.** State size by style (17 heights x 12 geo real grid):
`raw`/`density_10ex` N=204 (full grid), `ANCHOR` N=96 (8 params x 12
geo), `PCA_1D`/`PCA_1D_10ex` N=48 (4 retained components x 12 geo, at
0.999 retained variance), `PCA_3D`/`PCA_3D_10ex` N=5-6 (whole-field
compression, 0.999 retained variance) -- confirms real spatial
correlation across this 12-point grid is high enough for extreme
compression. RMSE improved in every style's first three (large, RO-heavy)
batches; the two small IGS-only batches stayed roughly flat in every
style (consistent with step 11's sparse-information finding, not
style-specific).

**A real, style-correlated convergence pattern emerged, directly
relevant to the step-size question above:** convergence rate split
cleanly by whether the style's forward map has a `log10` nonlinearity —
`raw`/`PCA_1D`/`PCA_3D` (linear density) converged in 4/5 or 5/5 batches;
`density_10ex`/`PCA_1D_10ex`/`PCA_3D_10ex` (log10-density) and `ANCHOR`
(Chapman-layer fit, also intrinsically nonlinear) all converged in only
2/5 batches -- the same 3 large batches every time. This is evidence
(not yet root-caused further) that the Gauss-Newton linearization
specifically struggles with the log10/Chapman nonlinearity on real,
large (700-1200 ray) batches, rather than a generic step-size or
`obs_sigma` mistuning affecting every style equally.

Artifacts: `step12_all_styles/{ro,igs}_observations.nc` (the reusable
precomputed observations) and `step12_all_styles/<style>/final_decoded_field.nc`
for each of the 7 styles.

**Convergence root-cause investigation (2026-09-23), three real findings
in sequence:**

1. **Jacobian correctness — ruled out.** Finite-difference-checked
   `GenericObservationOperator.linearized()` against `forward_single()`
   at all 5 real linearization points visited during an actual
   non-converging `density_10ex` batch (real IRI2020 ensemble, real ray
   geometry) with an eps-scan (`1e-6` to `1e-2`) rather than a single
   fixed step, so a badly-chosen step size couldn't manufacture a false
   positive/negative. Relative error scaled exactly as expected for a
   correct analytic gradient (~2e-7 at `eps=1e-6` up to ~2e-3 at
   `eps=1e-2`) at every point — the composed Jacobian
   (`H_grid` (linear) times `Parameterization.get_Jacobian`'s analytic
   `density_10ex` derivative) is correct.

2. **Anchor-relative backtracking — a real bug, fixed.** Hand-reproduced
   a failing Gauss-Newton iteration and swept `alpha` from `1e0` to
   `1e-11`: only `alpha=1.0` ever improved the residual, and every
   smaller value converged toward the *first* iteration's residual, not
   the current one. Root cause: `analysis_engine.py`'s backtracking
   candidate was `x_candidate = x_p + alpha * (K_i @ pseudo_residual)` —
   damped relative to the fixed prior anchor `x_p`, not the current
   iterate `x_i_current`, so once `x_i_current` had moved away from `x_p`
   over a few accepted steps, any `alpha<1` retreated back toward the
   stale anchor instead of taking a genuinely smaller step near the
   current point. User confirmed this was the originally-intended
   formula and had been changed at some point; fixed to standard damped
   Gauss-Newton: `x_candidate = x_i_current + alpha * (x_full_next - x_i_current)`,
   where `x_full_next = x_p + K_i @ pseudo_residual` (the undamped
   solution, unchanged). All 106 `Ensemble_Kalman_Engine` +
   `Assimilation_Cycle` tests still pass. A controlled A/B on the
   identical saved real ensemble + real observations (old vs. new
   formula, otherwise nothing else changed) showed the fix **only
   changes `ANCHOR`'s trajectory** (more iterations, lower final residual
   in every batch, e.g. batch 2: 3 iter/316.5 -> 8 iter/290.3) —
   `density_10ex`/`PCA_1D_10ex`/`PCA_3D_10ex` were bit-for-bit identical
   old vs. new, because backtracking never actually used `alpha<1` on
   those batches (only the always-identical `alpha=1` full step was ever
   accepted before stalling), so the two formulas coincide there. Kept
   the fix (strictly more correct, measurably helps `ANCHOR`, provably
   can't hurt), but it does not explain the broader pattern.

3. **`obs_sigma` sweep — does not explain it either.** Systematic sweep
   (`raw`, `density_10ex`, `ANCHOR` x `obs_sigma` in
   `{1,2,3,5,8,12,20,30,50}` TECU x the 3 large real batches, all against
   the identical saved real ensemble/observations so only `R` changes):
   - `raw` (linear/one-shot, no Gauss-Newton) converges trivially at
     every sigma, as expected — analysis RMSE drifts only mildly with
     sigma (e.g. batch 0: 23.4 at sigma=1 -> 29.7 at sigma=50).
   - `density_10ex` convergence is **non-monotonic in sigma** — e.g.
     batch 0 converges at sigma=5/8/12/20/30 but *not* at 1/2/3/50; no
     sigma value converges all 3 batches simultaneously.
   - `ANCHOR` **never converges at any sigma tested**, 1 through 50, in
     any of the 3 batches.
   - In every case, regardless of `converged` status, **analysis RMSE
     stays in a narrow, sensible band across the whole 50x sigma range**
     (e.g. `density_10ex` batch 0: 26.5-30.0 TECU; `ANCHOR` batch 0:
     25.3-32.2 TECU) and is always a large, real improvement over the
     forecast RMSE.

   **Conclusion:** `obs_sigma` is not the lever that fixes convergence —
   there is no value in this wide range that reliably resolves it, and
   `ANCHOR` is unaffected by it entirely. But the practical output
   (RMSE) is stable and good throughout, which points at the strict
   `converged` flag itself (`step_size / scale < 1e-3`, a relative-step
   criterion) being a stricter test than what real, large, noisy
   observation batches can reliably satisfy, rather than the filter
   producing bad answers when it reports `converged=False`. Not changed
   this session — a criterion change is another core-math decision that
   needs the user's sign-off, like the line-search fix above; flagged as
   the most promising next thing to look at instead of further
   `obs_sigma` tuning.

**Residual-based convergence criterion — implemented (2026-09-24), user
approved.** Added `AnalysisConfig.residual_rel_tol` (default `1e-2`):
convergence is now declared when *either* the existing relative-step-size
test passes *or* an accepted step's relative residual improvement falls
below this tolerance. All 106 tests still pass. Re-ran the two ANCHOR
cases that had visibly flattened (finding 3 above): both now correctly
report `converged=True` (batch 1: stops at the step whose improvement was
0.99%; batch 2: stops at 0.18%), while the two cases still making real
progress (`ANCHOR` batch 0, still improving 1.4-1.6%/step at the
`max_iterations` cap; `density_10ex` batch 2, still improving 12.6% when
it stops) correctly remain `False` — the new criterion only fires where
the residual has actually flattened, not indiscriminately.

**Grid domain size — the dominant factor, found while answering "what is
the horizontal grid size" (2026-09-24):** the grid used for every step
11/12 run so far had `grid_radius_deg=5` (~556 km), chosen for test speed
— but real RO/IGS ray tangent/pierce points for this ROI mostly lie
**hundreds to thousands of km further out** (median 1300-1500 km, some
past 3000 km) than that. Measured directly: 0%, 11.5%, and 18.6% of real
rays in batches 0-2 fell *inside* the 556 km grid at all. Rebuilt the
grid at `grid_radius_deg=18` (~2000 km, matching the observation ROI,
same 2.5 deg resolution, real IRI2020, `n_geo` 12 -> 161): coverage
jumped to 100%/84.5%/100%, and analysis RMSE dropped by roughly 4-5x
across every style/batch tested (e.g. `density_10ex` batch 0: ~26-30 ->
6.6 TECU; `ANCHOR` batch 0: ~25-32 -> 5.4 TECU). This was a domain-size
problem, not a resolution problem — the 12-point grid was simply too
small an area, not too coarsely spaced. **Every real-data result in this
document above this point used the too-small grid**; the numbers were
still useful for finding the Jacobian/line-search/convergence-criterion
issues (those are independent of grid size), but should not be read as
representative of the pipeline's real achievable accuracy — 5-7 TECU on
the properly-sized grid is the more representative number.

**Investigated: why does `ANCHOR` (8 x n_geo state) beat `density_10ex`
(~17 x n_geo state) on real-data RMSE? (2026-09-24, user-directed).**
Resolved with real numbers, on the properly-sized grid, isolating the
*forecast* (pre-assimilation) fit alone: `ANCHOR`'s reconstruction
(fit-then-decode each ensemble member's Chapman-layer parameters) differs
from the raw IRI2020-drawn profile by 31% RMS — a large reconstruction
error — yet its reconstructed-ensemble-mean forecast fits real TEC
substantially *better* than `density_10ex`'s (43.3/30.2/26.2 vs.
50.5/37.2/32.9 TECU forecast RMSE across the 3 batches), and this gap
exists **before any assimilation runs** -- it's a prior/forecast-quality
effect, not an EnKF behavior. A secondary, much smaller effect was ruled
in but shown minor: `density_10ex`'s ensemble mean is taken in log10
space, i.e. it decodes to the *geometric* mean of density (2% below the
raw arithmetic mean here) rather than the arithmetic mean — real, same
direction, but far too small to explain the gap (a literal raw
arithmetic-mean field's forecast RMSE, 53.2/39.1/34.7, is barely worse
than `density_10ex`'s). Conclusion: the EnKF correction is always
rank-limited by ensemble size (~100 members here) regardless of nominal
state dimension, so classical "more DOF fits better" reasoning doesn't
apply directly; more fundamentally, TEC is a line integral dominated by
the F2-layer peak, and `ANCHOR`'s 8 parameters are exactly the physically
meaningful quantities (`NmF2`, `hmF2`, `B0`, `B1`, ...) that determine
it, so fitting a smooth Chapman shape discards mostly height-to-height
roughness in the raw IRI2020 ensemble draw that is either in TEC's null
space or physically incoherent -- i.e. closer to noise than signal for
this observation type -- while `density_10ex`/`raw`'s extra nominal
flexibility is largely spent representing exactly that discarded
roughness.

**Not yet done:**
- No CLI/script entry point beyond direct Python calls (`CycleConfig` +
  `ensemble_init.build` + `observation_stream.assemble` +
  `cycle_driver.run_cycle`), per the plan's own §5 non-goal.
- The production default for `grid_radius_deg`/domain sizing given the
  finding above — every example/test config used this session picked a
  small grid for speed; a real cycle needs the grid domain sized to (or
  larger than) the observation ROI, not chosen independently. Worth a
  documented default or a validation check (e.g. warn if a configured
  grid's radius is much smaller than `radius_km`) rather than leaving
  this to be rediscovered.

## 11. Real-data parameterization comparison: ensemble size, vertical
    resolution, and a 4-style comparison (2026-09-24)

Everything in §10 above used `n_ensemble=100`. User asked for `n_ensemble=2000`
instead, and separately asked whether the analysis engine's Remedy B
(`centering="nearest_analog"`) had actually been in use throughout —
confirmed yes: `AnalysisConfig`'s dataclass default is
`centering="nearest_analog"`, and `AnalysisConfig.default_for_style`
(what every real-data script here used, with no explicit override) only
changes `linearization` per style, never `centering`. So Remedy B was
active in every real-data result in §10 and everywhere below.

### 11.1 n_ensemble=100 -> 2000 (still at the §10 17-level/50km grid)

Built a real 2000-member IRI2020 ensemble on the same properly-sized
(18deg/~2000km) grid used at the end of §10 (425s build time). Result:
`ANCHOR` now converges in **all 3** large batches (was 2/3 at n=100) —
the larger ensemble gives a better-conditioned Kalman gain, letting
Gauss-Newton actually settle instead of stalling. `density_10ex` improved
similarly (2/3 converged, up from ~1/3). Analysis RMSE stayed in the same
good range as n=100 (`ANCHOR`: 4.6-4.8 vs 4.1-5.4 before; `density_10ex`:
4.8-6.9, similar) — so the grid-size fix from §10 remained the dominant
accuracy factor; the larger ensemble's main benefit here was convergence
*reliability*. Notable side effect: `raw`'s one-shot linear update, which
had been mediocre, now nearly matched the iterated nonlinear styles
(5.5/6.4/4.1 TECU) -- a larger ensemble gives a much better-conditioned
covariance estimate that a simple linear update benefits from directly.
`ANCHOR`'s reconstruction RMSE vs. raw stayed essentially unchanged
(31.49% vs 31.29% at n=100) and its forecast advantage over `density_10ex`
persisted (43.1/29.7/25.7 vs 49.8/36.2/31.9) — confirming that finding
was a real structural effect, not n=100 sampling noise.

### 11.2 The DOF puzzle: why did ANCHOR (8 x n_geo) beat density_10ex
    (~n_height x n_geo) despite fewer parameters?

User pushed back precisely: if ANCHOR's representable profiles are a
strict subset of what a general per-height (density_10ex) profile can
represent, optimizing over the larger space should never do *worse* than
optimizing over the subset -- so where does that reasoning break?

**Resolution:** the EnKF correction is not unconstrained optimization
over the nominal state space -- `K` is built entirely from the ensemble's
own sample perturbations `S` (shape `n_state x n_members`), so the
correction is confined to a subspace of rank <= `min(n_state, n_members-1)`.
At `n_members=2000` and the §10 17-level grid: `density_10ex`
(n_state=2737) can reach at most 1999/2737 ~ 73% of its own nominal
space; `ANCHOR` (n_state=1288) can reach 100% (1288 < 1999) -- so in
*reachable*, EnKF-realizable terms, `ANCHOR`'s smaller nominal space was
actually less restricted, not more. At `n_members=100` (§10's original
tests) this was far starker: `density_10ex` ~3.6% reachable, `ANCHOR`
~7.7%. A second, independent mechanism: `ANCHOR`'s forecast (before any
assimilation) already beat `density_10ex`'s, which the rank argument
alone doesn't explain -- attributed at the time to `ANCHOR`'s Chapman fit
discarding unphysical per-height roughness in the raw IRI2020 draw
(31% RMS reconstruction "error") that doesn't help fit a TEC integral
anyway.

### 11.3 The resolution mismatch that partly overturned 11.2's explanation

User then asked about the actual vertical grid: **every script through
11.2 used `altitude_grid=np.arange(100, 901, 50)` -- 17 levels at 50km
spacing** -- a speed shortcut of this session's own, never flagged
clearly, and far coarser than the user's module-level standard of
90-900km at 10km (82 levels). Rebuilt the 2000-member ensemble at the
correct 82-level grid (1059s build time) and re-ran the same comparison.
**Result: the picture partly reverses.**

- `ANCHOR`'s forecast advantage nearly vanishes: 49.3/36.5/32.0 vs.
  `density_10ex`'s 49.6/36.3/32.0 -- essentially tied, `ANCHOR` no longer
  consistently wins. Its reconstruction error vs. the raw profile dropped
  from 31% to **7.2%**.
- Explanation: at 50km spacing, the raw 17-point "ground truth" was
  itself a crude, aliased discretization of the true IRI2020 profile --
  much of what 11.2 attributed to "ANCHOR filtering unphysical ensemble
  noise" was actually the raw grid being too coarse to represent the
  profile at all, not genuine per-member roughness in the true continuous
  profile. At 10km the raw representation is faithful, so there's much
  less for the Chapman fit to usefully discard.
- The assimilation outcome reversed further: `density_10ex` (now
  n_state=13202) **converged cleanly in all 3 batches** with clean
  monotonic residual reduction, despite a state size well past what 2000
  members can fully span (13202 >> 1999) -- the rank-restriction argument
  from 11.2, while still mathematically valid as a general principle, was
  evidently not the dominant effect in practice here. `ANCHOR` got
  *worse* at the finer resolution: converged in only 1/3 batches, higher
  analysis RMSE than `density_10ex` in 2/3 (8.73 vs 5.46 in batch 0).

**Corrected bottom line:** at the user's actual 10km resolution,
`density_10ex` performs as well as or better than `ANCHOR`, in both
forecast and assimilation outcome. The §10/11.2 "ANCHOR wins" finding was
real but specific to (and substantially an artifact of) the coarse 50km
grid, not a general property of the two parameterizations. This
correction is the main reason to treat every real-data number timestamped
before 2026-09-24 in this document as illustrative of the pipeline's
*mechanics*, not representative of its real achievable accuracy.

### 11.4 Four-style comparison at 10km/2000 members: adding PCA_1D_10ex
    and PCA_3D_10ex

User asked for a better dimension/accuracy trade-off than `ANCHOR`
(8 x n_geo, too coarse a physical model) vs. `density_10ex` (n_height x
n_geo, huge and barely-resolved by any practical ensemble) --
`PCA_1D_10ex` and `PCA_3D_10ex`, with "removal_threshold" 1e-3 and 1e-4
respectively (read as `1 - retaining_threshold`, i.e.
`retaining_threshold=0.999`/`0.9999` -- `get_PCA`'s own documented
parameter, "minimum fraction of total ensemble variance the retained
components must jointly explain"; flagged this reading to the user).
Reused the saved 10km/2000-member ensemble, no IRI2020 rebuild needed.

| style | n_state | reconstruction error | batch 0 | batch 1 | batch 2 |
|---|---|---|---|---|---|
| `density_10ex` | 13202 | 0.00% (exact) | 5.46 (conv.) | 6.73 (conv.) | 4.51 (conv.) |
| `ANCHOR` | 1288 | 7.21% | 8.73 (not conv.) | 6.69 (not conv.) | 4.72 (conv.) |
| `PCA_1D_10ex` (removal 1e-3) | 1610 | 1.08% | 5.70 (conv.) | 7.07 (conv.) | 5.08 (not conv.) |
| `PCA_3D_10ex` (removal 1e-4) | **110** | 0.13% | **5.47** (conv.) | **6.73** (conv.) | **4.50** (conv.) |

`PCA_3D_10ex` is the clear winner: matches `density_10ex`'s RMSE almost
exactly (6.731 vs 6.731 in batch 1) at 120x fewer state dimensions, and
converges reliably in all 3 batches. `PCA_1D_10ex` underperformed both
`density_10ex` and `PCA_3D_10ex` despite more nominal dimensions than
`PCA_3D_10ex`, and failed to converge in batch 2.

### 11.5 Was PCA_1D_10ex's underperformance structural or under-retention?

User asked to check with a tighter removal threshold. Swept
`retaining_threshold` from 0.999 (10 components/geo-column) through
0.9999 (13/column) to 0.99999 (17/column):

| removal threshold | components/column | n_state | reconstruction error | batch 0 | batch 1 | batch 2 |
|---|---|---|---|---|---|---|
| 1e-3 | 10 | 1610 | 1.08% | 5.70 (conv.) | 7.07 (conv.) | 5.08 (not conv.) |
| 1e-4 | 13 | 2093 | 0.74% | 6.12 (conv.) | 7.02 (conv.) | 4.91 (not conv.) |
| 1e-5 | 17 | 2737 | 0.20% | **5.36** (conv.) | **6.71** (conv.) | **4.45** (conv.) |

At `1e-5`, `PCA_1D_10ex` matches -- marginally beats -- `density_10ex`'s
full-fidelity RMSE and converges in all 3 batches. So the earlier
underperformance was under-retention, not a structural flaw in
per-column compression per se. But the magnitude of what it takes to get
there is itself the answer to "why is PCA_3D better": `PCA_1D_10ex`
needs 17 components x 161 columns = **2737** dimensions (a 4.8x
reduction from 13202) to match `density_10ex`, because per-column PCA
can only exploit *vertical* (height-to-height) correlation within each
of the 161 columns independently. `PCA_3D_10ex` reaches the same quality
with **110** total dimensions (120x reduction) at a *looser* threshold
(1e-4, not 1e-5) because joint compression also exploits *horizontal*
(geo-to-geo) correlation across the whole field -- a handful of shared
spatial-vertical modes cover what per-column compression needs thousands
of dimensions to represent.

**Conclusion for style selection:** `PCA_3D_10ex` (removal_threshold
~1e-4) is the best-evidenced trade-off of the styles tested against real
Tromsø data at the user's actual grid resolution -- matches full-grid
fidelity at a small fraction of the state size, with reliable
convergence. `ANCHOR` is the weakest of the four at proper resolution.
`density_10ex` is a safe but expensive full-fidelity baseline.
`PCA_1D_10ex` is dominated by `PCA_3D_10ex` (needs far more state to
reach the same quality) and is not recommended over it for this grid.

All real-ensemble artifacts from this section are saved under
`Runs/Tomography_Test/Data_Assimilation_Cycle/step14_n2000/` (17-level,
2000-member) and `step15_n2000_10km/` (82-level/10km, 2000-member,
`edp_samples_n2000_10km.nc`) for reuse without rebuilding IRI2020.

### 11.6 Why does ANCHOR specifically degrade at finer vertical resolution?

User asked to investigate. **Ruled out first:** float32 precision/
cancellation in `compose_observation_jacobian`'s block-diagonal-by-geo-
column einsum (the only reduction/summation step unique to ANCHOR-shaped
Jacobians, vs. `density_10ex`'s non-reducing diagonal composition) --
directly measured the cancellation ratio (L1 sum of terms / |net sum|)
and the net signal vs. the float32 noise floor at both resolutions:
cancellation ratio ~1 (no meaningful cancellation) and net signal ~2e6x
the noise floor at *both* 17 and 82 levels. Not the cause.

**Found instead, via `error_summary()['rmse_by_region']` and per-parameter
ensemble stats:**
1. `hmE` (E-layer height) has exactly zero ensemble variance at *both*
   resolutions -- every member gets `hmE=110`, evidently a fixed/default
   value in the ANCHOR fitting code rather than something actually fit.
   Pre-existing, not resolution-specific; wastes one of ANCHOR's 8
   nominal DOF entirely. User said no need to chase this now (math review
   planned later) -- left as-is, flagged for whenever that review happens.
2. **The resolution-specific mechanism:** `B0`/`B1` (bottomside shape
   parameters) shift sharply between resolutions. At 17 levels, `B1`
   mean/std/cv = 1.51/0.13/0.09 and bottomside/intermediate RMS error =
   78.2%/68.7% -- the coarse grid barely resolves the bottomside region,
   so the per-member fit can't detect real profile-to-profile shape
   differences there and converges to a similarly-wrong `B0`/`B1` for
   almost every member (artificially tight spread, badly wrong fit). At
   82 levels, the fit becomes far more accurate (bottomside/intermediate
   RMS error drops to 12.6%/1.5%) -- but `B1`'s ensemble spread roughly
   *triples* (cv 0.09 -> 0.26), now genuinely capturing real bottomside
   variability the coarse grid couldn't see. Since `B1` is a shape
   *exponent* (strongly nonlinear effect on the reconstructed density,
   not additive), a 3x wider spread in a highly nonlinear parameter is
   exactly what breaks Gauss-Newton's local-linearization assumption --
   the Jacobian at one point is a much worse approximation across a
   wider, more curved spread, so the line search finds no improving
   damped step sooner (matches the observed symptom directly: batch 1
   stalled after just 1 iteration at 82 levels, vs. running further at
   17). **So finer resolution didn't make ANCHOR's fit worse -- it made
   it better, and that improvement revealed real bottomside variability
   that the Chapman parameterization's own nonlinear structure handles
   poorly under a single linearized correction.** A real structural
   weakness of ANCHOR at proper resolution, not a numerical bug --
   reinforces `PCA_3D_10ex` as the safer choice, since it isn't tied to a
   fixed physical model that can become locally ill-linearized this way.

## 12. Final packaging (2026-09-25)

User provided `Final_Packaging.docx` (config surface + 9-step workflow +
visualization list) and asked for it to be built out as the last piece of
work. Full requirements, work items, and verification plan are in
`~/.claude/plans/vast-bouncing-haven.md` (written via `EnterPlanMode`,
approved before implementation) -- summarized here for the permanent
record.

**Built:**
- `cycle_config.py`: new fields `label`, `styles` (default `["ANCHOR",
  "PCA_3D_10ex", "PCA_1D_10ex"]`), `hyper_params_by_style` (removal
  thresholds 1e-4/1e-5 -> `retaining_threshold` 0.9999/0.99999),
  `max_tec_per_batch` (default-`None`, opt-in TEC-count batching so
  existing entry-count `batch_size` behavior is unaffected unless set),
  `min_tangent_alt_km`; `n_ensemble` default 100->2000, `altitude_grid`
  default factory -> 90-900km/10km (82 levels, the confirmed real
  standard from Section 11).
- `observation_stream.py`: `_apply_min_tangent_alt_filter` (RO-only,
  per-ray, via the new `_filter_entry_rays` general per-ray-field
  slicer -- handles the known schema fields plus any `extra` array
  matching the ray count; `abel` passes through unsliced, being per-entry
  not per-ray) and `batch_entries`'s TEC-count accumulation policy
  (entries never split across batches, so per-entry visualizations stay
  well-defined).
- `cycle_driver.py`: additive-only changes -- `CycleBatch.entry_ray_ranges`
  (source entry + ray slice per batch, populated by
  `observation_stream._entries_to_batch`), `BatchOutcome.y_forecast`/
  `y_analysis` (already-computed predicted TEC, just not previously
  stored), and `run_batch_loop`/`run_cycle`'s new `on_batch` callback
  (`(batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome)`,
  default `None` = no behavior change) -- lets a caller decode density
  and plot/save per batch without the loop itself retaining every
  batch's ensemble in memory (real runs can have `n_state x n_members`
  large enough that this matters).
- `output.py`: five new functions --
  `plot_iri_input_distributions`, `plot_observation_operator_sum` (sums
  one RO/IGS entry's `H_grid` rows, shown at several altitudes),
  `plot_tec_profile_comparison` (forecast/analysis/measured TEC by
  tangent altitude, one RO), `plot_edp_profile_comparison` (interpolates
  the gridded forecast/analysis density at a location via
  `EDPSamples.interp`'s existing mesh machinery, overlays the
  Abel-retrieved profile; location defaults to the tangent point of
  max-TEC-within-250-350km for an RO entry, or an explicit `lat`/`lon` --
  the same function the docx's planned future external-validation reuse
  needs, already built in), `plot_style_comparison_summary` (final RMSE,
  state dimension, wall-clock time, convergence rate, cross-style).
- `package_run.py` (new): `run_package(cfg) -> PackageResult`, the single
  orchestrator running all 9 workflow steps end to end -- wiring only, no
  new analysis math; reuses `style_sweep.py`'s already-established
  pattern of sharing the observation assembly and raw `EDPSamples` IRI
  draw across styles.
- Reconstruction error (step 4, "absolute and relative"): already fully
  covered by the existing `Parameterized_EDPSamples.plot_reconstruction_error_statistics`
  (two-panel absolute/relative-percent plot) -- no new plotting code
  needed there, just wiring.

**Verified:**
- 73 new/updated unit tests (`test_observation_stream.py`'s TEC-count
  batching/altitude-filter/`_filter_entry_rays` classes,
  `test_cycle_driver.py`'s `on_batch`/`y_forecast`/`y_analysis` classes,
  `test_output.py`'s five new-function classes), all passing alongside
  the full pre-existing suite: 129 total
  (`Assimilation_Cycle`+`Ensemble_Kalman_Engine`).
- One real end-to-end smoke run (`run_package` against the local Tromsø
  data, small `n_ensemble=20`/coarse grid for speed, all 3 default
  styles, `max_tec_per_batch=300`, `min_tangent_alt_km=50`) -- every one
  of the 9 steps produced its artifact: input-distribution histograms,
  horizontal grid + 5 mean-density altitude maps, 3 reconstruction-error
  plots, RO+IGS geolocation, 15 per-RO observation-operator-sum figures,
  13 batches x 3 styles of RMSE/rank-histogram/effective-rank plots, a
  TEC-profile and EDP-profile comparison PNG per RO per batch per style
  (correctly skipping the EDP plot only where no ray fell in the
  250-350km window), all netCDF/pkl saves, and the final cross-style
  summary. One real bug found and fixed during this run: `_savefig`
  didn't handle `plot_reconstruction_error_statistics`'s return type (an
  array of 2 `Axes` sharing one figure, not a single `Axes`/`Figure`) --
  fixed to unwrap array/list returns generically.
- A pre-existing `RuntimeWarning: overflow encountered in power` from
  `Parameterization.py:227` (ANCHOR's bottomside `B1` exponent term)
  surfaced during the smoke run -- not a new bug (nothing in this
  packaging work touches that code), consistent with Section 11.6's
  finding that `B1`'s wider, more nonlinear spread at proper resolution
  stresses the Chapman fit; noted but not chased further, out of scope
  for this packaging pass.

**Not yet done:** a full-scale run (`n_ensemble=2000`, production grid
resolution/spacing, `max_tec_per_batch=300`) has not been executed --
only the small/coarse smoke run above. Given the smoke run's per-style
wall-clock times (20-43s at `n_ensemble=20` on a coarse grid) and
Section 11's real-scale timings (2000-member IRI2020 builds took
7-18 minutes depending on grid), a full production run is expected to
take substantially longer; worth running when the user wants production
artifacts rather than a correctness check.

### 12.1 Full-scale run (2026-09-25/26)

Ran `run_package` at true production scale: `n_ensemble=2000`,
90-900km/10km (82 levels), 2.5deg horizontal (`n_geo=161`),
`max_tec_per_batch=300`, `min_tangent_alt_km=50`, real Tromsø RO+IGS, all
3 default styles. Reused the already-validated 2000-member `EDPSamples`
(`step15_n2000_10km`) and prepared RO/IGS observations
(`step12_all_styles`) via `edp_samples_path`/`ro_observations_path`/
`igs_observations_path` to skip re-running the ~18min IRI2020 build and
IGS RINEX processing -- every downstream step (per-style parameterization,
the full 13-batch assimilation loop, and all ~169 artifact files) still
ran for real at full scale.

**Result -- decisively confirms and quantifies Section 11's finding at
production scale:**

| style | n_state | wall time | converged | final RMSE |
|---|---|---|---|---|
| `ANCHOR` | 1288 | 7711.4s (**128.5 min**) | 12/13 | 15.99 |
| `PCA_3D_10ex` | 110 | 117.5s (**~2 min**) | 8/13 | **9.83** |
| `PCA_1D_10ex` | 2737 | 270.5s (~4.5 min) | 6/13 | 10.06 |

`ANCHOR` is **65x slower** than `PCA_3D_10ex` (dominated by per-profile
Chapman fitting across 2000 x 161 = 322,000 profiles) and, despite
converging in more batches, has the *worst* final RMSE of the three --
not just weakest on accuracy (Section 11.4/11.6) but now shown
impractically expensive at real operational scale. **`PCA_3D_10ex` wins
on every axis that matters in practice** (best accuracy, ~65x faster,
smallest state) and is the confirmed recommendation for production use.
169 artifacts (every plot type from the smoke run, at full scale) saved
under `Runs/Tomography_Test/Data_Assimilation_Cycle/step17_full_scale/`.

This closes out the `Final_Packaging.docx` scope: the packaged pipeline
runs end to end at both smoke and full production scale, against real
data, producing every specified artifact.

### 12.2 User manual (2026-09-26)

User confirmed `ANCHOR` stays in the default `styles` list deliberately
(heritage/legacy-comparison value across many prior projects, not an
oversight that the full-scale timing/accuracy result should override),
and asked for a user manual to repeat these runs on other data. Written
to `Assimilation_Cycle/README.md`: environment setup (`IRI2020_PATH`,
the `sys.path` shim every script in this session needed, since none of
`EDPSamples`/`Parameterization`/`IRI_Sample_Inputs`/`Assimilation_Cycle`
are installed packages), a minimal runnable example, the full
`CycleConfig` field reference (grouped, with the `radius_km` vs.
`grid_radius_deg` unit-mismatch pitfall called out explicitly since it
was a real bug this session), the precomputed-file-mode workflow, the
exact output file layout `run_package` produces, and a performance-
expectations table built from this session's actual measured timings
(§12.1's numbers) rather than estimates, plus a troubleshooting section
covering the IGS-credentials/local-files point and the two-criterion
`converged` flag (§11.6-adjacent) so a non-converged batch isn't
mistaken for a bad result.

## 13. Grid margin and inter-batch inflation: two real-data experiments (2026-09-26)

User reviewed the full-scale artifacts and noticed some RO occultations'
forecast and analysis TEC were both far from measured, and specifically
that occultations whose line of sight mostly stays near the modeling
domain's boundary seemed to fit worse than ones crossing the center.
Proposed two ideas to test: (1) automatically pad the assimilation grid
with an extra margin (e.g. 500km) beyond the user-specified domain, for
more state DOF near the edge; (2) inflate ensemble covariance after each
batch, since pure persistence may let spread shrink progressively and
limit the filter's ability to fit later/boundary observations.

**Built (both opt-in, default off, no behavior change unless set):**
- `CycleConfig.grid_margin_km` (default `0.0`) + a new
  `effective_grid_radius_deg` property (`resolved_grid_radius_deg` +
  margin converted to degrees) -- `ensemble_init.py` now builds the grid
  at `effective_grid_radius_deg`, not `resolved_grid_radius_deg` directly.
- `CycleConfig.inter_batch_inflation_factor` (default `None`) -- wired
  into `cycle_driver.run_batch_loop` via
  `Ensemble_Kalman_Engine.perturbation.SelectiveInflation`, passed as
  `driver.assimilate_one_cycle(..., perturbation=...)` for every batch
  when set. Reuses the analysis engine's own existing perturbation hook
  (Section 5.3) rather than adding new inflation math.
- 8 new tests (`test_cycle_config.py`'s margin-math class,
  `test_cycle_driver.py`'s `TestInterBatchInflation`) -- the inflation
  tests needed `driver.config.rng` fixed explicitly across compared runs,
  not `run_batch_loop`'s own `rng` parameter (which only covers OSSE
  noise/rank-histogram tie-breaking, not the stochastic perturbed-obs
  noise inside `assimilate_one_cycle`/`analyze()` itself) -- a real,
  previously-undocumented distinction, worth remembering for any future
  reproducibility-focused test in this area.

**Diagnosed the boundary hypothesis directly before testing the fix,**
using the cached real 2000-member grid + observations, `PCA_3D_10ex`
(fast): three independent per-RO metrics against real analysis RMSE --
closest approach to grid center, mean distance from center restricted to
each ray's altitude-valid segment, and (most direct) the literal fraction
of altitude-valid ray points falling *outside* every mesh triangle
(triggering `get_observation_operator`'s crude nearest-vertex fallback
instead of proper barycentric interpolation). **All three show ~zero
correlation with per-RO analysis RMSE** (r = -0.03, 0.04/-0.10, 0.01).
The mesh-outside-fraction number is itself a real, separate finding worth
knowing regardless of the correlation result: **10-84% of real RO ray
points fall outside the mesh entirely** in this cycle's geometry --
genuine reduced-fidelity coverage, just not one that happened to explain
this dataset's RMSE variation.

**Margin experiment (real data, controlled):** built two grids at the
*same* reduced ensemble (`n_ensemble=300`, for speed -- a fair A/B needs
matched ensemble size) -- baseline (`grid_margin_km=0`, `n_geo=161`) and
`+500km` margin (`n_geo=250`) -- ran `PCA_3D_10ex` on identical real
observations against each. **Result: the margin made things worse, not
better** -- every RO's analysis RMSE increased or stayed flat (mean
delta +2.41 TECU across 15 ROs; worst case +7.73), and overall final
RMSE went from 9.80 to 12.00. Convergence count went up (7/13 -> 9/13)
but RMSE clearly degraded, so that alone isn't evidence of a better fit
(consistent with Section 11's finding that `converged` and fit quality
don't always track together). **Working explanation:** the margin adds
~55% more grid area (and state dimension) while the observation set is
completely unchanged -- no new data constrains the added region, so a
fixed-size ensemble's effective resolution over the original (already
well-observed) core region is diluted rather than improved. Same family
of effect as Section 11.2's DOF finding: more state without more
information to constrain it doesn't help, and here it measurably hurts.

**Inflation experiment (real data):** swept
`inter_batch_inflation_factor` in `{None, 1.02, 1.05, 1.1, 1.2}` on the
real 2000-member grid/observations, `PCA_3D_10ex`. **Result: monotonic
improvement at every factor tested** -- final analysis RMSE 9.56 (None)
-> 9.41 -> 8.90 -> 8.15 -> **7.03** (1.2, the largest tested), with the
state's mean std growing 0.105 -> 0.707 as expected. Improvement
concentrates in later batches (e.g. batch 8: 6.06 -> 2.24 TECU at
factor=1.2), exactly matching the hypothesis that spread shrinkage under
pure persistence increasingly limits the filter's ability to fit later
observations. No factor tested yet made things worse -- 1.2 was simply
the largest value tried, not a confirmed optimum; a real tuning pass
(sweeping further, and checking whether very large factors eventually
hurt) is still open.

**Bottom line (revised below, 2026-09-26):** the two ideas resolved very
differently under direct testing -- inter-batch inflation looked like a
real, confirmed improvement by the per-batch RMSE metric (`1.1-1.2` a
reasonable starting point pending further tuning); the grid-margin idea
is not supported by this session's real-data test and measurably hurt
results at matched ensemble size, though it remains available (opt-in,
default off) since a different region/observation density or a much
larger ensemble could plausibly behave differently -- not re-tested at
full 2000-member scale given the clear, consistent negative signal at
300.

### 13.1 Correction (2026-09-26): the inflation "improvement" above does not survive a pooled/retrospective check

User asked a precise question about what "final RMSE" actually measures
given the constant-true-state assumption: is it the final state re-scored
against all TEC collected over the cycle, or the intermediate state right
after each batch? Tracing `run_batch_loop` exactly answered it: **neither
in the pure sense asked** -- every `BatchOutcome.rmse_reduction` scores
*that batch's own just-updated state* against *that batch's own
observations only*, computed once and never revisited. So every "final
RMSE" number used in this section (and throughout the whole project) is
really "the last batch's own analysis fit to the last batch's own data" --
not a check of whether the final state stays consistent with everything
assimilated over the cycle.

Since the true state is assumed time-invariant, that whole-cycle
self-consistency check is meaningful here (unlike in a genuinely
time-varying system) and was missing. Built it ad hoc: take the one truly
final state (after all 13 batches), forward-model it through *every*
batch's own real observation operator, score against that batch's own
real TEC, pool across all batches. Reran inflation=None vs 1.2 on the
identical real cached 2000-member grid/observations used for the sweep
above.

**Result: the earlier conclusion reverses.** Baseline (no inflation)
pooled RMSE = 8.94 overall (RO 8.78, IGS 19.84) -- consistent with the
per-batch trend, no pathology, every individual batch in a sane 3.75-38
range under the retrospective scoring too (modulo the expected extra
error from applying a later state to earlier data). Inflation=1.2 pooled
RMSE = **4042** overall (RO 4061, IGS 19.7) -- catastrophic, dominated by
batch 1 scoring **14993** TECU when checked with the state as it stood 12
batches later; every other batch individually still looked reasonable
(3.75-38). The per-batch metric used for the sweep above cannot see this
at all, since it only ever checks a batch's state against that same
batch's own data, immediately after the update.

**Likely mechanism:** `apply_inflation` (`kalman_core.py`) inflates
ensemble spread about the mean without moving the mean itself, so
inflation cannot corrupt the state directly. But 13x compounding
multiplicative inflation with no localization widens ensemble spread
unconstrained in directions no single batch's data observes -- classic
setup for spurious sample cross-correlations, which the Kalman gain then
uses to inject large, non-physical corrections into the mean on later
batches. Those corrections look fine when checked against whichever
batch produced them (what the per-batch metric measures, almost by
construction) but are wildly wrong checked against a different batch's
geometry -- exactly the observed pattern, worst for batch 1 (most
temporally/informationally distant from the final state).

**Corrected recommendation:** do **not** adopt `inter_batch_inflation_factor=1.2`
(or the earlier "1.1-1.2 as a starting point" guidance) as validated --
it most likely causes real filter divergence that the per-batch metric is
structurally blind to. If inflation is worth pursuing further, it needs
either localization (`SelectiveInflation` already supports a
per-component factor array -- never exercised that way yet), a
substantially smaller factor, or a non-compounding application (inflate
only immediately before the next batch, not accumulated through the
whole history). The grid-margin negative result is unaffected by this
correction (it never involved inflation).

**General lesson:** any future "improvement" claim resting on the
per-batch RMSE metric in this codebase should be cross-checked with a
pooled/retrospective metric (final state vs. all batches' own data)
before being trusted -- the per-batch number alone cannot distinguish a
real improvement from a method that is simply overfitting its own most
recent batch at the expense of consistency with earlier ones. This
pooled check was done ad hoc in a scratch script this session, not yet
promoted to `metrics.py`/`cycle_driver.py` as reusable code -- worth
doing if this kind of check becomes routine.

## 14. Batch size and assimilation order: does the nonlinear filter's final state depend on them? (2026-09-26)

User's follow-on question, directly motivated by §13.1: for a linear
Kalman filter, batch size/order can't change the final state estimate
(exact superposition); for the nonlinear iterated-Gauss-Newton filter
actually used here, each batch re-linearizes at the *current* mean while
holding the ensemble sensitivity `S` fixed for that batch's iterations
(`analysis_engine.py`'s docstring), so different orderings walk different
sequences of local linearizations, and different batch sizes ask a fixed
linearization to explain more or less data per step. Asked: (1) what's
the largest batch size this system can handle for the real test problem,
(2) is there a strategy for choosing assimilation order.

**First pass had a confound, caught and fixed.** Initial order-sweep and
batch-size-sweep scripts left `GeneralEnKFDriver`'s `AnalysisConfig.rng`
unseeded (defaults to a fresh `np.random.default_rng()` per call --
same root issue as the `TestInterBatchInflation` gotcha in §13). The
*same* `cap=300`/chronological config scored 8.78 in the size-sweep
script and 14.64 in the order-sweep script purely from different random
perturbed-obs draws -- caught by noticing two nominally-identical runs
disagreed by ~2x. Reran everything with `AnalysisConfig.rng` explicitly
seeded and held fixed across every compared condition within each sweep,
after first confirming (same seed run twice) that seeding gives exact
reproducibility, and (4 different seeds, same config) that the pure noise
floor is small: pooled RMSE 8.83-9.03, ~2% spread.

All results below use the same fixed 38-entry pooled/retrospective
scoring yardstick as §13.1 (final state forward-modeled through every
real entry's own observation operator, scored against that entry's own
real TEC, pooled) -- independent of whatever batching produced the tested
final state, so results are comparable across different `cap` values.

**Batch-size sweep** (one fixed seed, chronological order, real 2000-member
grid/observations, `PCA_3D_10ex`):

| cap | n_batches | wall time | converged | pooled RMSE |
|---|---|---|---|---|
| 150 | 18 | 18.9s | 9/18 | 9.60 |
| 300 (production default) | 13 | 15.2s | 7/13 | 8.99 |
| 500 | 7 | 9.7s | 4/7 | 8.79 |
| 1000 | 3 | 5.7s | 3/3 | 8.64 |
| 3000 (whole hour, 1 batch, 2545 rays/38 entries) | 1 | 4.2s | 1/1 | 8.67 |

Monotonic and clean: fewer/larger batches are both more accurate and
faster. The single-batch case -- the *entire* real test window's
observations assimilated in one Gauss-Newton solve -- converged, ran in
4.2s (fastest of anything tested), and scored statistically tied with the
best result found. **No upper batch-size limit was found within this
dataset's scale** (2545 rays, n_state=110 for `PCA_3D_10ex`); each
additional batch boundary pays a fresh-linearization-restart cost a
single joint solve never incurs, which is why more/smaller batches
uniformly did worse here, not better. Untested: whether a much larger
dataset (many hours, thousands of occultations) or a higher-dimensional
style (`ANCHOR`, `density_10ex`) hits a real compute or conditioning
limit -- not assumed to extrapolate from this result.

**Order sweep** (one fixed seed, `cap=300`/13 batches, same real data):

| order | converged | pooled RMSE |
|---|---|---|
| chronological | 7/13 | 8.99 |
| small_first (ascending batch size) | 8/13 | 8.95 |
| large_first (descending batch size) | 5/13 | 11.21 |
| reverse_chronological | 5/13 | **40.68** |

Three of four orders cluster in a fairly narrow band (8.95-11.21);
reverse-chronological is a dramatic ~4.5x outlier, far outside the ~2%
noise floor -- a real effect, not sampling noise. No clean mechanistic
explanation found for *why* that specific sequence is so much worse
(`large_first` also front-loads the two biggest batches and is only
moderately worse, so it isn't simply "biggest batches first") -- recorded
as an empirical finding on this dataset/config, not generalized into a
universal "never process backward in time" rule without more test cases.

**Answers given to the user:**
1. Largest batch size handled: the entire test window in a single batch,
   with no failure found -- and it's the best-or-tied-best, fastest
   option of everything tested at this scale.
2. Ordering strategy: if forced into small sequential batches, order by
   ascending size (ties for best here) and avoid reverse-chronological;
   the cleaner, better-evidenced strategy is to sidestep the ordering
   question by using fewer/larger batches (up to one joint batch), which
   this data shows is both safer and more accurate.

**Standing lesson, now confirmed twice this session:** any comparison
across nonlinear-EnKF runs in this codebase requires
`AnalysisConfig.rng` explicitly seeded and held fixed across every
compared condition -- an unseeded run is not a valid basis for
comparison. Neither this section's controlled results nor §13.1's pooled
metric are yet promoted into reusable `Assimilation_Cycle` code (both
done ad hoc in scratch scripts this session) -- worth building in
(`metrics.py` for the pooled score, a documented seeded-driver helper in
`cycle_driver.py`/`README.md` for controlled comparisons) if this kind of
investigation becomes routine rather than one-off.

### 14.1 Ordering by forecast-observation misfit, not chronology/size (2026-09-26)

User proposed a more principled ordering signal than batch size/chronology:
sort batches by `|forecast - measured|` (the pre-assimilation misfit
against the current state) and asked whether increasing or decreasing
order should be preferred. Flagged before testing that there's a genuine
mechanistic tension: small-misfit-first keeps early corrections small
(favors linearization validity, matching §14's `small_first` result), but
also means ensemble spread gets partly spent on the least-informative
data before the large, spread-hungry corrections arrive -- so it wasn't
obvious a priori which effect dominates.

Tested directly: same `cap=300`/13 batches, same fixed seed=42, same real
data/`PCA_3D_10ex`, same pooled 38-entry yardstick as §14, but ranked by
each batch's own pre-assimilation forecast RMSE against the *original,
unassimilated* prior mean (a one-time static ranking) instead of size or
time. Combined with §14's orders in one script/session for a single
directly-comparable table:

| order | converged | pooled RMSE |
|---|---|---|
| descending_misfit (biggest disagreement first) | **9/13** | **9.03** |
| chronological | 7/13 | 8.99 |
| small_first | 8/13 | 8.95 |
| large_first | 5/13 | 11.21 |
| ascending_misfit (smallest disagreement first) | 4/13 | **38.45** |
| reverse_chronological | 5/13 | 40.68 |

**Decisive result, resolves the tension:** descending-misfit order (biggest
disagreement first) ties for the best pooled RMSE found so far *and* gets
the best convergence rate of every order tested (9/13, beating even
`small_first`'s 8/13). Ascending-misfit order (smallest disagreement
first) is nearly as bad as the worst ordering found (38.45 vs
reverse_chronological's 40.68) -- so the "small correction first" intuition
that made `small_first` work for *size* does **not** transfer to misfit;
for misfit specifically, biggest-first wins. Mechanistic account:
processing the largest, most-informative disagreements first uses the
ensemble's full, unspent spread to make a large, well-powered correction
early; small-residual data then arrives later purely as fine refinement on
an already-good state, needing only a small correction regardless of
timing. Doing it backwards spends spread on unremarkable, already-
consistent data first, then asks an already-shrunk ensemble to make the
hardest, largest correction last -- a plausible reason ascending-misfit
failed almost as badly as reverse-chronological.

**Ruled out that this is just a restatement of the size result:** checked
whether misfit tracks batch size -- it doesn't (batch 1 and batch 10 are
nearly identical size, 175 vs 174 rays, but almost opposite misfit, 58.6
vs 23.2 TECU); `large_first` (11.21) is clearly worse than
`descending_misfit` (9.03) despite both starting with large batches, so
misfit is a genuinely more informative ranking signal than size, not a
proxy for it.

**Recommendation (best-evidenced ordering strategy to date):** if forced
into sequential batches, rank by each batch's forecast-vs-observed misfit
against the *current* state and assimilate in decreasing order, recomputed
adaptively as data arrives incrementally (a batch's true misfit shifts as
the state improves, so a one-time static ranking against the original
prior, as tested here, is an approximation of the ideal adaptive/greedy
version -- not yet tested against the static version, worth doing if this
becomes a real operational strategy rather than a one-off finding).

## 15. Iterative ensemble smoother (IES/ES-MDA-style) evaluation (2026-09-26)

Follow-on from §14's "does batch-size/order sensitivity mean smoothing
would help" discussion. Clarified first that *temporal* Kalman smoothing
(using later observations to retroactively correct an earlier-time state
via dynamics) adds nothing here, since the state is defined constant over
the whole cycle -- a single joint batch already computes the full-
information estimate a smoother would recover from a forward-only filter.
The other thing sometimes also called "(iterative ensemble) smoothing" in
the DA literature is different: resweeping the *same* static batch
multiple times, refreshing the ensemble sensitivity `S` between sweeps
(classic ES-MDA, Emerick & Reynolds 2013, tempering each sweep with
`R_k = N*R` for `N` equal sweeps so `sum(1/alpha_k)=1` and the same data
isn't double-counted as if independent). This is a real, distinct
question from temporal smoothing -- our single-batch call already holds
`S` fixed for its entire inner Gauss-Newton loop (`analysis_engine.py`'s
`analyze()`), refreshing `S` only across genuinely different batches, so
resweeping the *same* batch with a refreshed `S` was untested. User asked
to evaluate it.

**First pass had a scoring-design flaw, caught before drawing a
conclusion.** Ran the single-batch (whole 2545-ray real dataset) case
with (a) one sweep (baseline), (b) naive untempered repeats (`R`
unchanged -- included as a statistically-invalid control, expected to
double-count data), (c) tempered ES-MDA (`R_k=N*R`, `N` in {2,4,8}),
scored on the same fixed 38-entry pooled yardstick used in §13-14. Result
looked like a clean win for more sweeps (baseline 8.67 -> tempered N=4:
8.51 -> naive N=4: 7.86) -- **but flagged immediately as suspect**,
because the fixed 38-entry scoring set *is* the same 2545 rays the single
batch was fit to. Scoring in-sample cannot distinguish a real
signal-recovery improvement from the state simply absorbing more of the
observation noise on repeated sweeps -- the same category of mistake as
§13's per-batch-metric blind spot, caught before being reported as a
result rather than after.

**Corrected test: genuine held-out validation.** Split the 38 real
entries into a fixed 30-entry fit set (1680 rays) and an 8-entry held-out
set (865 rays, 5 RO + 3 IGS, never assimilated by any sweep), built one
combined observation operator from the fit entries' concatenated real
geometry, ran the same (a)/(b)/(c) sweep configs on the fit set only, and
scored exclusively against the held-out entries' real TEC -- a genuine
generalization test immune to the in-sample-noise-fitting confound.

**Result: the tempered smoother's effect is not reproducible across
seeds, in either direction.**

| config | seed base=42 | seed base=100 |
|---|---|---|
| baseline (1 sweep) | 33.44 | 33.42 |
| N=4 tempered (Rx4) | 23.58 (looks ~30% better) | 73.49 (looks ~120% worse) |
| N=8 tempered (Rx8) | 28.50 (looks better) | 58.50 (looks worse) |

The baseline is stable across seeds (33.44 vs 33.42); the multi-sweep
tempered variants swing far more than any average effect either seed
shows relative to baseline -- the *sign* of the apparent effect isn't
reproducible, so it cannot be attributed to the method rather than to
which particular perturbed-obs noise sequence happened to land well or
badly across the compounded sweeps.

**Conclusion given to the user:** on this real test problem, the
iterative ensemble smoother (properly tempered ES-MDA form) does **not**
show a reliable benefit over the existing single-sweep baseline -- the
single sweep is both simpler and measurably more robust (tiny seed-to-
seed variation vs. the multi-sweep variants' wild swings). Not
recommended as currently evaluated. **Open loose end:** the untempered
naive-repeat control was only checked at one seed (looked best of
everything, 26.51 at N=4) and was not re-verified at a second seed the
way the tempered version was -- given the tempered version's
instability, a single-seed number for the naive control shouldn't be
trusted either without the same check; not yet done, worth doing before
treating any multi-sweep variant as validated one way or the other.

## 16. Single-batch, full production resolution, all default styles (2026-09-26)

User asked to test the single-batch configuration at full production
scale (n_ensemble=2000, 82 levels/10km, real cached Tromso RO+IGS) for
all three default styles (`ANCHOR`, `PCA_3D_10ex`, `PCA_1D_10ex`).
Reused the cached `step15_n2000_10km` grid/`step12_all_styles`
observations; built each style's `Parameterized_EDPSamples` once, timed
separately from the EnKF step (ANCHOR's Chapman-profile fit is
independent of batching and was expected, correctly, to still dominate
total wall time).

| style | n_state | parameterization time | in-sample RMSE (whole 2545 rays) |
|---|---|---|---|
| ANCHOR | 1288 | 2393.6s (~40 min) | 11.07 |
| PCA_3D_10ex | 110 | 2.8s | 8.67 |
| PCA_1D_10ex | 2737 | 0.8s | 8.62 |

The single-batch EnKF step itself is fast for all three (95-253s) --
consistent with §14's finding that single-batch mode is cheap regardless
of style; ANCHOR's total cost is entirely the parameterization step, not
the assimilation.

**In-sample scoring flagged the same in-sample vs. held-out concern as
§15 before drawing any conclusion.** A first held-out check (single
80/20 fit/holdout split of the 38 real entries, split seed=0) showed a
dramatic ranking *reversal*: ANCHOR held-out RMSE 19.13 vs. PCA_3D_10ex
33.44 / PCA_1D_10ex 29.29 -- ANCHOR winning by a wide margin, opposite
the in-sample ranking and opposite §12's full-scale "PCA_3D_10ex
confirmed production recommendation." Flagged as too consequential to
report from one split, given §15's lesson that single-seed/single-split
results in this investigation chain have repeatedly failed to replicate.

**Robustness check across 3 independent fit/holdout splits (seeds
0,1,2), reusing each style's already-fitted parameterization to avoid
re-paying ANCHOR's ~40min cost per split:**

| split | ANCHOR | PCA_3D_10ex | PCA_1D_10ex |
|---|---|---|---|
| 0 | 19.13 | 33.44 | 29.29 |
| 1 | 12.86 | 10.81 | 10.73 |
| 2 | 11.16 | 7.02 | 6.72 |
| mean | **14.38** | 17.09 | 15.58 |

**The dramatic split-0 reversal does not replicate.** ANCHOR is clearly
*worse* than both PCA styles at splits 1 and 2; only at split 0 does it
win, and by far more than the average difference. Averaged across all
three, ANCHOR comes out only modestly ahead (14.38 vs 15.58/17.09) --
nowhere near the ~43% margin split 0 alone suggested. Split 0 was an
outlier, not a real generalization advantage -- same "one split looked
decisive, more splits reversed it" pattern as §15's seed sensitivity.

**More decision-relevant than the accuracy numbers: ANCHOR failed to
converge (`n_iterations=1`, stalled immediately) on splits 1 and 2**;
`PCA_3D_10ex`/`PCA_1D_10ex` converged cleanly on all 6 runs (3 splits x 2
styles). This reconfirms §11.6's finding that ANCHOR's Chapman
`B0`/`B1` exponents are sensitive enough at full vertical resolution that
Gauss-Newton often stalls -- ANCHOR's erratic held-out numbers plausibly
reflect that fragility (sometimes stalling near a good starting point by
luck, sometimes not) rather than a genuine hidden generalization edge.

**Verdict: does not overturn the §12 production recommendation.**
`PCA_3D_10ex`/`PCA_1D_10ex` remain the better default choice -- reliable
convergence (unlike ANCHOR at this resolution), ~40 minutes cheaper (no
Chapman fit), and comparable-or-better held-out accuracy once the split-0
outlier is set aside. ANCHOR's continued inclusion in the default
`styles` list for heritage/legacy reasons (user's explicit instruction,
§12) is unaffected either way. **Methodological lesson added to the
running list this session:** single-split held-out validation needs the
same seed/split-robustness discipline as RNG seeding and ordering did --
one split can look as decisive and be as wrong as one unseeded run.

## 17. Diagonal covariance boosting (2026-09-27)

User reviewed the `step19_single_batch_full_res` TEC/EDP profile plots
and noticed some real occultations (e.g.
`podTc2_YM08.2025.322.10.19.0028.E34.00_0000.0001_nc`) show large
residuals in *both* forecast and analysis -- a plausible symptom of the
raw IRI2020 climatological ensemble having strongly correlated
vertical/horizontal structure and thus insufficient effective range to
reproduce a real, localized feature (consistent with the independent
finding that `PCA_3D_10ex` needs only ~110 of ~13202 nominal dimensions
to explain nearly all of a real ensemble's variance -- the ensemble's
*effective* rank is far below its nominal size). Proposed "diagonal
boosting": generate one random field per ensemble member on the
vertical/horizontal grid, with per-point standard deviation matching the
ensemble's own IRI2020-generated std but otherwise independent, apply
vertical + horizontal smoothing for local correlation, and add these
fields (scaled by a small amplitude) to the original samples -- augmenting
the diagonal of the sample covariance matrix with new, less-correlated
spread. Asked for an opinion before implementing.

**Response:** agreed this targets the right root cause and is
mechanistically safer than the inter-batch multiplicative inflation that
failed (§13.1) -- applied once at ensemble construction, not compounded
across many sequential batches. Flagged two things before implementing:
(1) must be added to the raw ensemble *before* any parameterization is
fit, or `retaining_threshold` would discard most of the injected variance
for the PCA styles; (2) the amplitude/smoothing-scale choice needs the
same held-out validation discipline as the inflation experiment, not an
in-sample check. User confirmed both points and asked for implementation
as a `CycleConfig` option with user-selectable perturbation level.

**Built:** `Assimilation_Cycle/diagonal_boost.py`
(`apply_diagonal_boost(edp_samples, amplitude, vertical_scale_km=30.0,
horizontal_scale_km=200.0, rng=None)`): draws i.i.d. standard-normal
fields per member on the raw `EDPs` grid, smooths vertically via a
Gaussian kernel over the real altitude coordinate and horizontally via a
Gaussian kernel over haversine (great-circle) distance -- correct for
this project's unstructured/mesh-based horizontal grid and for the high
latitude (Tromso, ~70N) where naive degree-based distance would be
badly wrong -- renormalizes each `(height, geo)` point's smoothed field
to exact zero-mean/unit-std across members (so achieved perturbation
scale is exactly `amplitude x original std` regardless of how much the
smoothing kernel reduced the raw noise's variance, and the ensemble mean
is never shifted), scales by `amplitude * per-point std`, adds to the raw
samples, and clips to stay physically non-negative. New `CycleConfig`
fields: `diagonal_boost_amplitude` (`None` default, no behavior change),
`diagonal_boost_vertical_scale_km` (30.0), `diagonal_boost_horizontal_scale_km`
(200.0), `diagonal_boost_rng_seed`. Wired into
`ensemble_init.load_or_build_edp_samples` -- applied after either the
fresh-build or `edp_samples_path`-load branch, before any save, so a
shared cached base ensemble can be boosted differently per run, and a
saved `edp_samples_output_path` reflects whatever was actually used
downstream.

One real implementation wrinkle found while writing the test fixture:
`EDPSamples`'s own `Rectangle` geo_type generator produced a malformed
mesh at the tiny synthetic grid sizes convenient for unit tests (a
pre-existing characteristic of that code path at small sizes, not
something this work touches) -- switched the test fixture to `Regional`
(the same geo_type every real grid in this project already uses),
which handled a small synthetic case cleanly.

**Verified:** 21 new tests (`test_diagonal_boost.py`: no-op at
`amplitude=0`/`None`, exact mean preservation, variance increases by the
closed-form `sqrt(std^2 + (amplitude*std)^2)` within 15% at n=3000,
cross-point correlation measurably drops with a short horizontal scale
and stays higher with a long one -- confirming the smoothing scale
actually controls injected correlation length as intended -- stays
non-negative even at an extreme amplitude, exactly reproducible under a
fixed seed; `test_cycle_config.py`: validation of the new fields;
`test_ensemble_init.py`: wiring -- disabled by default is a true no-op,
enabled changes the loaded ensemble, reproducible through the config,
and the *boosted* ensemble is what gets saved). Full suite: 158 passed.

**Update (2026-09-27) -- amplitude sweep run, real negative result found:
diagonal boosting is currently incompatible with `PCA_3D_10ex`.** Swept
`diagonal_boost_amplitude` in `{0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0}` on
the real 2000-member grid, held-out scoring (fit=30 entries, score=8
held-out, same split/methodology as §16), fixed analysis-engine seed
across all amplitudes:

| amplitude | n_state | converged | holdout RMSE (overall) |
|---|---|---|---|
| 0.0 (baseline) | 110 | True | 33.44 |
| 0.1 | 1417 | True | 46,074 |
| 0.2 | 1634 | True | 6,441,958 |
| 0.3 | 1737 | True | 327,152 |
| 0.5 | 1845 | True | 30.37 |
| 0.8 | 1942 | **False** | 15.81 |
| 1.2 | 1971 | **False** | 25.71 |
| 2.0 | 1987 | True | 56.94 |

**Root cause: `n_state` explodes from 110 to 1400-2000 at any nonzero
amplitude.** `PCA_3D_10ex` sizes its own state via a variance-retention
threshold (`retaining_threshold`) -- any genuinely new, spatially
decorrelated variance added to the raw ensemble forces PCA to retain far
more components to still explain 99.99% of the now noise-dominated
spectrum. With `n_state` approaching the 2000-member ensemble's own rank
ceiling, the EnKF correction loses nearly all the regularization that
made the compact 110-dim representation reliable, and at small-to-moderate
amplitude (0.1-0.3) this produces catastrophic, non-physical RO
predictions (holdout RMSE into the millions) -- not subtle mistuning, a
real numerical failure mode. This was not anticipated when reasoning the
technique would be "safer than inflation" (correct that it doesn't
compound across batches, but wrong about a different failure mode this
introduces for adaptively-sized styles specifically).

At amplitude=0.8 (best single point, holdout=15.81, better than
baseline), a robustness check across 2 more boost-RNG seeds (11, 12) gave
19.63 and 24.51 -- consistently better than baseline's 33.44 in
*direction*, unlike the IES test's sign-flipping instability, but **none
of the three converged** (Gauss-Newton stopped at an arbitrary iterate,
not a well-defined solution) -- not enough to call validated.

**Important nuance: this failure mode is specific to adaptively-sized
styles.** `PCA_3D_10ex`/`PCA_1D_10ex` size themselves from the ensemble's
own variance spectrum; `ANCHOR` (fixed 8 params/geo-point) and
`density_10ex` (fixed full-grid size) have no such mechanism, so their
state dimension would not blow up this way -- this is evidence the
*combination* (diagonal boosting + adaptive-threshold PCA sizing) doesn't
currently work, not necessarily that the technique itself is unusable.

**Not recommending any amplitude for `PCA_3D_10ex`/`PCA_1D_10ex` as
currently implemented.** Two directions to fix this, not yet chosen
between: (a) test the technique against a fixed-dimension style
(`ANCHOR`/`density_10ex`) instead, where this specific failure mode
cannot occur; (b) redesign the boost to stay low-rank itself (e.g. a much
larger smoothing scale, so it injects only a handful of new large-scale
modes PCA can absorb cheaply, rather than near-per-point independent
structure that forces massive state growth). Flagged for the user to
weigh in before building either.

### 17.1 Root cause found (density_10ex test), fix built and validated (2026-09-27)

User asked to test `density_10ex` too, expecting it would fare better
than `PCA_3D_10ex`. **It didn't -- and that result was the key
diagnostic.** `density_10ex` has a *fixed* state dimension (13202,
always -- no variance-threshold sizing), yet it failed with almost
*identical* numbers to `PCA_3D_10ex` at every amplitude (e.g. both
exactly 15.814 at amplitude=0.8). Since `n_state` never changed for
`density_10ex`, the earlier "`n_state` explosion" theory could not be the
cause -- it was a coincidental correlate for `PCA_3D_10ex`, not the
mechanism.

**Actual root cause, confirmed by direct inspection:** at amplitude=0.1,
1516 `(height, geo, member)` values were already clipped to the
linear-space floor (1.0 m^-3); the count grows monotonically with
amplitude (6615 at 0.2, ..., 1.27M at 2.0). The raw ensemble's own
per-point std/mean ratio reaches 2.04 at some points, so even a modest
fraction of local std, added linearly, pushes density below zero and
triggers the clip. `density_10ex`/`PCA_*_10ex` both take `log10` of
density before fitting anything -- a clipped value becomes
`log10(1.0)=0` next to typical surrounding values around
`log10~9-12`, an 8-12 order-of-magnitude discontinuity in the space
those styles actually fit. A handful of such extreme leverage outliers
among 2000 members is enough to severely distort that point's sample
mean/std/Jacobian, explaining both the catastrophic RMSE at low-moderate
amplitude (rare, extreme outliers are worst when sparse) and the
non-convergence at 0.8-1.2 (as clipping becomes common enough to shift
from rare-leverage-outlier to systemic bimodal corruption).

**User's follow-on hypothesis, tested and confirmed: `raw` (no log10
transform, one-shot linear update) should be immune.** Same held-out
sweep against `raw`: **no catastrophic values anywhere, every amplitude
converged** (guaranteed, since `raw` is one-shot linear, `n_iter=1`
always), and a real, smooth optimum at amplitude~0.2 (holdout RMSE
11.56 -> 9.98, ~14% improvement), degrading gracefully past that --
exactly the well-behaved bias-variance shape a real effect should have,
not a numerical artifact. Robustness check at amplitude=0.2 across 2 more
boost seeds: 10.06, 10.21 -- tightly clustered.

**Fix built:** `apply_diagonal_boost` gained a `log_space: bool = False`
parameter. When `True`, the perturbation is added to `log10(density)`
instead of linear density, and converted back via `10**(...)` -- no
positivity constraint, so no clip is ever needed, and it matches what
`density_10ex`/`PCA_*_10ex` actually fit. New `CycleConfig.diagonal_boost_log_space`
field (default `False`, no behavior change), wired through
`ensemble_init.py`. 7 new unit tests (no clip-floor values even at
extreme amplitude, geometric- not arithmetic-mean preserved -- correctly
distinguishing the two given Jensen's inequality for a log-space
perturbation, log-space variance increases by the expected closed form,
log-space cross-point correlation still decreases with a short smoothing
scale, reproducible, differs from the linear-space result). Full suite:
164 passed.

**Re-ran the `PCA_3D_10ex` sweep with `log_space=True`: clean, large,
robust win.**

| amplitude | n_state | converged | holdout RMSE (overall) |
|---|---|---|---|
| 0.0 (baseline) | 110 | True | 33.44 |
| 0.1 | 1362 | True | 21.69 |
| 0.3 | 1762 | True | 16.89 |
| 0.5 | 1860 | True | 12.22 |
| **0.8** | 1910 | True | **11.59** |
| 1.2 | 1933 | True | 16.14 |
| 2.0 | 1946 | True | 26.79 |

No catastrophic values, every amplitude converges (unlike the
linear-space run, which stopped converging at 0.8-1.2), and a real
optimum at amplitude~0.8 giving a **~65% held-out RMSE reduction**
(33.44 -> 11.59). Robustness check across 3 boost seeds: 11.591, 11.597,
11.503 -- remarkably tight. `n_state` still grows substantially
(110 -> ~1900, confirming growth itself was never the problem -- only the
linear-space clip-discontinuity artifact was) yet the fit converges
cleanly and reliably despite the much larger state.

**Updated recommendation:** `diagonal_boost_log_space=True` should be
used whenever `diagonal_boost_amplitude` is set alongside
`density_10ex`/`PCA_*_10ex` (linear-space, i.e. the default `log_space=False`,
remains validated and fine for `raw` specifically, but is actively
dangerous for the log10-based styles). Amplitude~0.8 was a well-evidenced
starting point on one split -- see §17.2 for the multi-split cross-check,
which revises this down to ~0.5.

### 17.2 Multi-split cross-check (2026-09-27): amplitude~0.8 revised down to ~0.5

Re-ran the `log_space=True` sweep against 3 independent fit/holdout
splits (same split seeds 0/1/2 as Section 16's style-comparison
robustness check) instead of the single split=0 used above, fixed
analysis/boost seeds throughout:

| amplitude | n_state | split0 | split1 | split2 | mean |
|---|---|---|---|---|---|
| 0.0 (baseline) | 110 | 33.44 | 10.81 | 7.02 | 17.09 |
| 0.3 | 1762 | 16.89 | 2.03 | 5.26 | 8.06 |
| 0.5 | 1860 | 12.22 | 1.82 | 5.83 | **6.63** |
| 0.8 | 1910 | 11.59 | 1.83 | 7.50 | 6.97 |
| 1.2 | 1933 | 16.14 | 1.84 | **51.75** | 23.24 |
| 2.0 | 1946 | 26.79 | 1.68 | **54.93** | 27.80 |

**Real, replicated improvement at moderate amplitude (0.3-0.8):** every
split improves over its own baseline consistently across this range, not
just split0 -- the effect is real and largest on split0 (the "hardest"
split), smaller but still present on the easier splits 1/2.

**But amplitude=0.8's earlier single-split "win" sat closer to an
instability boundary than that result alone revealed.** At 1.2 and 2.0,
split2 blows up catastrophically (51.75, 54.93) while split0/split1 still
look reasonable -- instability completely invisible when only split0 was
checked, since split0 happened to degrade gracefully rather than
catastrophically at high amplitude. Exactly the failure mode Section 16
established multi-split checking is needed to catch.

**Revised recommendation: amplitude~0.5, not 0.8.** Best across-split
mean (6.63, essentially tied with 0.8's 6.97) while keeping every
individual split well-bounded (max 12.22) and sitting further from where
1.2+ clearly breaks. Treat 0.8 as within the useful range but closer to
the edge than a default should sit; 1.2+ is not safe based on this
evidence.

## 18. Repeat full-scale run with boosting; new analysis-density artifact (2026-09-27)

User asked to repeat the full-resolution/full-scale three-default-style
run, with diagonal boosting (`amplitude=0.5`, `log_space=True`) applied
uniformly to the shared raw ensemble before any style's parameterization
(the "boosting on for all three" choice -- ANCHOR/PCA_1D_10ex see
boosting for the first time here, untested until now).

**Result:** clean run, 134 artifacts, no errors.

| style | n_state | wall time | converged | final_rmse |
|---|---|---|---|---|
| ANCHOR | 1288 | 698.3s | **0/1** | 2.312 |
| PCA_3D_10ex | 1860 | 108.0s | 1/1 | 2.194 |
| PCA_1D_10ex | 5152 | 114.2s | 1/1 | 2.193 |

**Flagged immediately, before this could be misread:** `final_rmse` here
is in-sample (single batch = fit and score on the same data), the exact
metric category the IES investigation (Section 15) showed cannot
distinguish real improvement from a more flexible state absorbing its
own residual. All three styles show a suspiciously uniform ~4x
"improvement" over the unboosted single-batch run (ANCHOR 11.07->2.31,
PCA_3D_10ex 8.67->2.19, PCA_1D_10ex 8.62->2.19) -- including ANCHOR,
which **did not converge**. Only `PCA_3D_10ex`'s result is corroborated
by real held-out, multi-split validation (Sections 17.1-17.2); ANCHOR and
PCA_1D_10ex have not been held-out validated with boosting at all, and
ANCHOR's non-convergence here is consistent with (not contradicted by)
its known B0/B1 fragility at full resolution (Section 11.6) potentially
being made worse, not better, by the extra spread boosting introduces.

**New standard artifact added, per user request:** the *optimal* (final
analysis) EDP field's horizontal spatial distribution, at the same 5
altitudes (100/200/300/400/500 km) as the existing step-4 forecast/prior
mean-density plots -- lets a reviewer directly compare prior vs. analysis
side by side for the same style. Implemented in `package_run.py`'s
per-style loop, right after `run_cycle` returns: decodes
`result.final_ensemble.to_param_shape()` via a freshly-built
observation operator for that style (`decode()` is geometry-independent,
confirmed by reading `observation_operator.py` -- any batch's operator
works, so no extra ray-geometry machinery needed), takes the mean across
members, and reuses `output.plot_horizontal` exactly as step 4 does.
Saved as `{label}_{style}_analysis_mean_density_{alt}km.png` per style,
documented in the README's output-layout table. Verified with a real
(small, fast) IRI2020 smoke run following this project's established
package_run.py-testing convention (real small-scale run rather than a
synthetic unit test, since `decode()`/`plot_horizontal` are already
independently unit-tested) -- all 5 per-style plots produced, non-trivial
file sizes, correctly named by actual nearest grid altitude. Full
existing suite (164 tests) still passes -- purely additive change.

## 19. Three artifact-review fixes/additions (2026-09-27/28)

User reviewed the boosted full-scale run's artifacts and requested three
changes.

**1. Bad map projection on the observation-geolocation plot.** Root
cause: `package_run.py`'s Step 5 called
`observation_preparation.diagnostics.plot_geolocation`, which uses
`ccrs.Orthographic` centered on the ROI but never calls `set_extent` --
so the plot shows the whole visible hemisphere, and this project's real
data (all within ~2000km of one point) ends up squeezed into a small
part of an otherwise-empty disc (user: "all the interesting items are
plotted near the top"). Per this package's own architectural rule (the
five survivor modules get no logic added), fixed without touching
`observation_preparation`: new `output.plot_observations_geolocation(edp_samples, entries)`
reuses `edp_samples.plot_geolocation()` itself (identical
projection/extent-framing logic already used for `foo_horizontal_grid.png`)
and overlays RO tangent tracks (colored by altitude) + IGS pierce points
on top of that same axes, replicating the old function's geometry/plot
calls but on correctly-framed axes. `package_run.py` Step 5 now calls
this instead. Verified visually on a real smoke run: RO/IGS content now
fills a properly-framed regional map matching the grid extent (was
squeezed into ~10% of the frame near the top before).

**2. Combined observation-operator-sum plot for all RO.** New
`output.plot_observation_operator_sum_combined(edp_samples, entries, ...)`
-- same sum(H)-by-altitude visualization as the existing per-RO
`plot_observation_operator_sum`, but concatenates every RO entry's ray
geometry into one `get_observation_operator` call first, showing the
*cumulative* sensitivity pattern across the whole RO dataset rather than
one occultation at a time. `package_run.py` Step 5 now also calls this
once (after the existing per-RO loop), saved as
`foo_obs_operator_sum_all_ro.png`. Unit-tested that the combined sum
exactly equals the sum of the per-entry sums (catches a concatenation
bug that would silently drop an entry).

**3. IGS performance visualization (previously entirely missing).** New
`output.plot_igs_tec_scatter(batches, result, style_label=None)`: 2-panel
scatter (forecast vs. measured TEC, analysis vs. measured TEC), IGS rays
only, pooled across every batch via `CycleBatch.obs_type` (zipped with
`result.batch_outcomes`, since `BatchOutcome` itself doesn't carry an
obs-type back-reference), with a 1:1 reference line on each panel.
`package_run.py`'s per-style loop now calls this once per style, saved as
`foo_{style}/igs_tec_scatter.png`. Handles the no-IGS-data case
gracefully (placeholder text, not a crash).

**Verified:** 8 new unit tests (`test_output.py`) covering all three --
projection/extent match, combined-sum correctness, IGS-ray pooling across
batches, empty-entries/no-IGS-data edge cases. Full suite: 172 passed.
Real end-to-end smoke run (small/fast IRI2020, matching this project's
established `package_run.py`-testing convention) confirmed all three
artifacts render correctly and look sensible against real geometry --
the geolocation fix was visually inspected and confirmed to actually
solve the reported framing problem, not just run without error.

## 20. Automatic polar projection for high-latitude grids (2026-09-28)

User reviewed the (freshly Plate-Carree-framed) geolocation plot and
pointed out the projection itself was still wrong for this project's real
region: Plate Carree ("mercator-like") badly stretches shape/area near
the poles (longitude lines that actually converge get drawn as parallel)
-- asked for automatic switching to a polar projection when the grid is
close to a pole, Plate Carree otherwise.

**Implemented in `EDPSamples._horizontal_map_axes`** (the one shared
method underlying every horizontal plot in the whole pipeline --
`plot_geolocation`, `plot_horizontal_field`, and everything in
`Assimilation_Cycle/output.py` that calls through them, including the new
`plot_observations_geolocation`/analysis-density plots from Section 19) --
a deliberate, flagged exception to this package's "no added logic to the
five survivor modules" rule (matching the one prior precedent, the
`GeneralEnKFDriver` split), chosen because fixing it in one shared place
is far better than duplicating divergent projection logic per-plot.

`geo_type=="Polar"` already used north/south polar stereographic; now any
*other* geo_type whose grid's **mean** latitude exceeds 60 degrees (either
hemisphere) also automatically switches to polar stereographic, framed
down to the grid's own actual minimum |latitude| (same extent-setting
pattern the existing `"Polar"` branch already used, generalized off the
real data instead of `geo_type`-specific attrs). Mean rather than
"every point" beyond the threshold -- a first attempt using "every point"
failed its own test against this project's real production grid
(Regional, Lat=69.6/radius=18deg): that grid's southern edge dips to
~52N, well below any reasonable polar threshold, even though the grid is
clearly polar-region on the whole (mean ~69.5N). Caught by writing the
test against the real grid parameters *first*, before assuming the
initial criterion was right.

**Verified:** 6 new tests (`EDPSamples/test_edp_samples.py`): the real
production grid correctly switches to `NorthPolarStereo` despite its
low-latitude edge, a southern high-latitude grid switches to
`SouthPolarStereo`, a mid-latitude grid stays `PlateCarree`, a grid whose
*mean* sits below threshold stays `PlateCarree` even with a high-latitude
edge, `geo_type="Polar"` behavior is unchanged, `"Global"` never
auto-switches. Full suite: EDPSamples 108 passed, combined with
Assimilation_Cycle/Ensemble_Kalman_Engine/Parameterization 254 passed.
Visually verified on a real smoke run: the actual grid/RO/IGS content
now renders in a proper circumpolar view with correct great-circle
geometry, no more horizontal stretching.

### 20.1 Follow-on bug: observation-operator plots missed the polar switch (2026-09-28)

User reviewed the regenerated `step20` artifacts and reported the
observation-operator plots (per-RO and the new combined-all-RO plot)
were *still* narrow, stretched Plate Carree rectangles, despite the
report above -- correctly asked why, since that report was wrong for
these two specific plots (only the geolocation/mean-density plots were
actually checked visually).

**Root cause:** `output.plot_observation_operator_sum`/
`plot_observation_operator_sum_combined` build their multi-panel figure
via `plt.subplots(nrows, ncols, subplot_kw={"projection": _cartopy_projection()})`
-- and `_cartopy_projection()` was a *separate*, hardcoded
`return ccrs.PlateCarree()` in `output.py`, never consulting
`EDPSamples`'s own (now auto-polar-aware) projection logic at all. Each
panel then called `plot_horizontal_field(ax=ax, ...)` with that
already-built axes; the shared axes-builder only *selects* a projection
when it creates a fresh axes itself (`ax is None`) -- handed an existing
`ax`, it just uses whatever projection that axes already has. So the
auto-polar switch (Section 20) never had a chance to run for these two
plots specifically, even though it was correctly wired into every other
horizontal plot in the pipeline.

**Fix -- eliminated the duplication risk that caused this, not just the
symptom:** extracted the projection-selection logic out of
`_horizontal_map_axes` into a new public method,
`EDPSamples.select_map_projection()` (returns
`(proj, is_polar, is_north_polar, center_lon)`), which `_horizontal_map_axes`
now calls internally, and which `output.py`'s `_cartopy_projection`
now also calls (taking `edp_samples` as an argument) instead of
hardcoding a choice. Single source of truth -- this exact class of bug
(a second call site quietly re-deriving/hardcoding the same decision and
drifting out of sync) can no longer happen for this decision.

**Verified:** 2 new regression tests (one per affected function), using
a dedicated high-latitude fixture (the existing mid-latitude fixture
would never have caught this) -- both check the actual panel axes'
`.projection` type is `NorthPolarStereo`, not just "does not raise" (the
original tests' weakness, which is exactly why this bug shipped past
them). Caught a test-writing mistake immediately after writing these:
`fig.axes` includes the colorbar axes `plot_horizontal_field` adds per
panel (plain `Axes`, no `.projection` attribute) alongside the actual
`GeoAxes` panels -- fixed by filtering to `hasattr(ax, "projection")`
before asserting. Full suite: 364 passed. Visually confirmed on a real
smoke run: the observation-operator-sum plots (both per-RO and combined)
now render in the same proper circumpolar view as every other horizontal
plot.

**Process note:** the "report" that started this sub-section was
based on visually checking only two of the several plots the fix should
have affected -- a reminder to check every affected artifact, not a
representative sample, before reporting a visual fix as verified.

## 21. IRI input-distribution plot showed no spread (2026-09-29)

User noticed the `iri_input_distributions.png` histograms had no spread
(each driving index a single spike) and asked if that was correct.
**It wasn't -- a real bug, confirmed by tracing the exact mechanism.**

**Immediate cause:** the script that produced `step19`/`step20` never set
`iri_spread_kwargs` (defaults to `{}`). With every `*_sample_range` left
`None`, `IRI_Sample_Inputs.randomSamples` sets each driving index's
sampling range to `[None]` -- every one of the 2000 draws picks the same
"use nominal value" placeholder for every index, a genuinely degenerate
draw, not a rendering artifact.

**But the real, assimilated ensemble was never affected.** Loaded the
actual cached ensemble those runs used (`step15_n2000_10km`, via
`edp_samples_path`) and checked its own `.sampling_parameters` directly:
real, non-degenerate spread (hour 9-11, f107 117.4-129.4, ap 0-22, ig12/
rz12 modest ranges) -- confirming the assimilation results themselves
were correct throughout; only this one diagnostic plot was wrong.

**Root cause is structural, not just a missing kwarg in one script:**
`package_run.py`'s steps 2-3 *always* called
`iri_selection.sampling_parameters_for_cycle(cfg)` fresh for the
distribution plot -- completely disconnected from whatever ensemble step
4 actually used. In precomputed-`edp_samples_path` mode this silently
plots an unrelated, differently-configured (here, degenerate) redraw
instead of the real ensemble's real driving indices. Even in the
*fresh-build* case it was a second, independently-seeded `randomSamples`
call from the one that actually built the ensemble (`ensemble_init.py`
calls the same function again internally) -- not literally the values
used, just a statistically similar redraw.

**Fix:** reordered `package_run.py` so step 4 (load/build `EDPSamples`)
happens before the distribution plot, and the plot now reads
`edp_samples.sampling_parameters` -- the real, authoritative values for
whatever ensemble is actually in use, whether freshly drawn or loaded
from a cache -- instead of an independent re-derivation. Step 2's
`load_or_build_iri_sample_inputs` call (saving the `.pkl` artifact) is
kept as its own step, since that artifact should exist regardless of
`edp_samples_path`.

**Verified:** full suite still 364 passed (pure reordering + a different
data source for one plot call, no new logic to unit-test beyond what
`plot_iri_input_distributions` and `EDPSamples.sampling_parameters`
already cover independently). Directly confirmed the fix against the
exact bug scenario (`edp_samples_path` set, `iri_spread_kwargs` left
default): the plot now shows real multi-bar spread for every index
matching the cached ensemble's true driving indices. Regenerated just
this one artifact in `step20` (the only thing affected -- no change to
the assimilation results, ensemble, or any other plot, so no need to
re-run the full ~45-60min pipeline).

## 22. `iri_spread_kwargs` default changed from "no spread" to real windows (2026-09-29)

After the §21 fix, user looked at the *now-real* distribution plot and
noticed the spread still looked surprisingly sparse -- `ig12`/`rz12`
showing only 2 distinct values, `f107` showing a gap in the middle. Asked
whether the sample generation itself was working correctly.

**Investigated directly against the real underlying tables (not
speculation).** Loaded the actual `IRI_Sample_Inputs` object for the real
event date and printed both `apf107["f107"]` and `ig_rz["ig"]`/`["rz"]`
around their `current_idx`:

- `f107` window (this cycle used `f107_sample_range=3`) landed exactly on
  a real ~10-unit day-to-day jump in the historical F10.7 record (129.3
  -> 119.5 between two adjacent real days) -- the "gap in the middle" is
  genuine solar activity in the real historical data for this specific
  event window, not a sampling defect.
- `ig`/`rz` window (`ig_sample_range=1`/`rz_sample_range=1`) mechanically
  can only ever include the 2 adjacent table rows -- exactly the 2
  spikes observed, reproduced precisely from the real table values
  (88.0/90.7 for ig, 106.9/108.5 for rz).

**No bug in the sampling code** -- the sparseness was the deterministic,
correct consequence of narrow `*_sample_range` windows applied to real
data, independent of ensemble size (2000 vs 20000 samples would show the
same number of discrete bars, since the window -- not the sample count --
bounds how many distinct table rows are reachable).

**But this exposed a real units mismatch worth getting right before
setting new defaults.** User proposed default windows (ig12/rz12: 12;
f107/ap: 30; hour: already-correct +/-3) based on stating "ig12 and
rz12 are daily values" and "f107 and ap are 3 hour values." Checked the
actual file-reading code before accepting this at face value:
- `apf107.dat` (`get_apf107()`) parses one row per `yr`/`mn`/`dy` --
  **daily-indexed**, not 3-hourly. `ap`'s *value* within a row is 8
  genuinely 3-hourly sub-values, but the sample-range *window* itself
  steps in whole days.
- `ig_rz.dat` (`get_ig_rz()`) header literally reads `Start_end_month` --
  **monthly-indexed**, not daily. Matches the indices' own physical
  definition (12-month smoothed running means).

Presented this correction back to the user (window=60 would have meant
+/-60 *days* for f107/ap given day-indexing, and ig/rz are month- not
day-indexed) before implementing anything -- user confirmed: **use window
12 for ig12/rz12 (+/-12 months) and window 30 for f107/ap (+/-30 days)**.

**Implemented as the new default**, replacing the old `{}` (no spread)
default entirely: `default_iri_spread_kwargs()` in `cycle_config.py`
returns `{"hour_sample_range": 3, "f107_sample_range": 30,
"ap_sample_range": 30, "ig_sample_range": 12, "rz_sample_range": 12}`,
with a docstring recording the exact cadence evidence above so the
reasoning doesn't have to be re-derived later. Only affects fresh IRI2020
builds (`edp_samples_path` unset) -- precomputed-file-mode runs are
unaffected, and no existing cached ensemble (`step15`/`step17`/`step19`/
`step20`) needs rebuilding, since none of them relied on this default
(all used an explicit `iri_spread_kwargs` or loaded a precomputed file).

**Verified:** 3 new tests (`test_cycle_config.py`: default value is the
real-window dict not `{}`, an explicit `{}` override still means no
spread, two configs don't share the same mutable dict via
`default_factory`). 367 total passed. Directly re-ran
`sampling_parameters_for_cycle` with the new default against the real
event date: 7 distinct `hour` values, 58 distinct `f107` values, 25
distinct `ap` values (0-300, physically plausible up to a severe-storm
level), 24 distinct `ig12`/`rz12` values each -- a dramatic, confirmed
improvement over the old default's 2-6 distinct values per index.

## 23. Combined TEC/EDP profile plots, per-style and cross-style (2026-09-29)

User requested two changes to the per-RO TEC/EDP artifacts: (1) combine
the separately-saved TEC and EDP plots into one two-panel figure per RO
per style, replacing the two files; (2) additionally generate a
cross-style two-panel figure per RO, overlaying all three styles in each
panel, for direct style-to-style comparison.

**Implementation.** Refactored `plot_edp_profile_comparison`'s lat/lon-
derivation + horizontal-interpolation logic out into a new public
`resolve_edp_query_point_and_profiles` (returns the raw `(n_height,
n_members)` profiles, not yet reduced to mean/std) -- needed so
`package_run.py` can collect one style's profile at a time for the
cross-style plot without duplicating that geometry logic a second time;
`plot_edp_profile_comparison` itself now calls this helper internally,
identical behavior/signature, confirmed by the full existing test suite
passing unchanged.

New `output.plot_tec_edp_profile_comparison(...)`: builds the TEC panel
via the existing `plot_tec_profile_comparison` and the EDP panel via
`plot_edp_profile_comparison` (each already accepted an `ax=` parameter,
so this composes them rather than reimplementing either) -- catches the
EDP panel's `ValueError` (no ray in the 250-350km window) and shows a
placeholder message there instead of failing the whole figure, since the
TEC panel is always independently plottable. Found and fixed a real
cosmetic bug while visually reviewing the first version: both sub-
functions bake the entry label into their own per-axes title (sensible
standalone), which visually collided with the new figure-level
`suptitle` carrying the same label a second time -- stripped the
redundant suffix from each panel's title after calling the sub-function,
confirmed clean via a second visual check.

New `output.plot_cross_style_tec_edp_comparison(entry, tec_by_style,
edp_by_style, altitude)`: same two-panel layout, but every style overlaid
in each panel -- a stable color per style across every RO's figure
(`_color_for_style`, a small module-level registry so the same style
never gets a different color in a different figure), solid line for
forecast, dashed for analysis, so the encoding stays legible with
multiple styles in one panel. A style with no ray in the EDP window gets
named in the panel's own subtitle rather than silently dropped.

`package_run.py` wiring: `_on_batch` now calls
`plot_tec_edp_profile_comparison` once per RO per style (saved under
`{label}_{style}/profiles/`, replacing the old separate `tec_profiles/`/
`edp_profiles/` folders), and additionally collects each style's TEC
arrays + resolved EDP profiles into a `cross_style_data` dict keyed by
`(batch_index, entry.label)` (not just entry label, so an entry
appearing in more than one batch across a run still gets its own
figure). After the per-style loop finishes, one
`plot_cross_style_tec_edp_comparison` call per collected entry saves to
the new `{label}_cross_style_profiles/` folder.

**Verified:** 4 new tests (`test_output.py`) -- combined per-style plot
produces 2 populated panels titled without duplication, EDP-window
failure shows a placeholder without dropping the TEC panel; cross-style
plot overlays every style with a color stable across both panels, a
missing style is named rather than silently dropped. Full suite: 371
passed (up from 367 -- 4 new, 0 broken; the refactored
`resolve_edp_query_point_and_profiles`/`plot_edp_profile_comparison`
pair kept every existing test passing unchanged). Real smoke run with
all 3 default styles (not just one, to genuinely exercise the cross-
style overlay) confirmed both new artifact types render correctly
end-to-end -- visually reviewed both, including the title-collision fix.

## 24. Altitude taper on diagonal-boost amplitude (2026-09-29)

**Problem.** After §22 widened `iri_spread_kwargs` to real physical
driving-index windows, the user reviewed a fresh full-resolution run
(`step21_fresh_iri_single_batch_boosted/`, still using the old flat-
amplitude boost from §17) and noticed the analysis EDP became visibly
wavy at high altitude, without any corresponding improvement in TEC
residual. Their hypothesis: the constant-fraction (0.5) diagonal boost
(§17) is "overdone" at high altitude, and a taper on the boost amplitude
by altitude might eliminate the unnecessary oscillation.

**Diagnosis.** TEC is a line integral dominated by the F2-peak region
(~250-350km, already this project's reference window for EDP-profile
plots) -- a density perturbation well above that has very little effect
on TEC, so the EnKF has almost no observational leverage to correct or
constrain whatever the boost injects up there. Topside density is also
more sensitive to the now-wider driving-index spread (§22), plausibly
making the injected spread itself larger at high altitude. Both point at
the same fix: scale the boost amplitude down with altitude rather than
applying it uniformly everywhere.

**Design choice, confirmed with the user:** taper above the F2 peak
(not, e.g., a taper centered on the peak, or one keyed to a per-column
computed peak altitude) -- full boost amplitude at/below
`taper_start_km`, linearly down to `taper_floor x amplitude` by
`taper_end_km`, held at that floor above (not zero, since some spread
above the peak is still physically plausible and the floor keeps the
mechanism smooth rather than a hard cutoff). Defaults confirmed:
`taper_start_km=400.0`, `taper_end_km=700.0`, `taper_floor=0.1`.

**Implementation** (`diagonal_boost.py`): new `_altitude_taper(altitude,
start_km, end_km, floor)` helper, returning a per-height multiplier
(`1.0` at/below `start_km`, `floor` at/above `end_km`, linear ramp
between); `apply_diagonal_boost` gains `taper_start_km`/`taper_end_km`/
`taper_floor` parameters (defaults as above) and multiplies `amplitude`
by this per-height array before scaling the smoothed noise field --
`effective_amplitude = amplitude * taper`, broadcast over the height
axis only, applied identically in linear and log-space modes. Three new
`CycleConfig` fields (`diagonal_boost_taper_start_km`/`_end_km`/`_floor`,
same defaults) with `__post_init__` validation (`end_km > start_km`,
`floor` in `[0, 1]`), wired through `ensemble_init.py`'s
`apply_diagonal_boost` call. Setting `taper_start_km` far beyond the
grid's max altitude (e.g. together with a correspondingly large
`taper_end_km`, since `_altitude_taper` itself requires `end_km >
start_km`) reproduces the old flat-amplitude behavior exactly.

**Verified:** hardened two pre-existing exact-formula tests
(`test_increases_per_point_variance_by_expected_amount`,
`test_increases_log_space_variance_by_expected_amount`) to explicitly
disable tapering, after noticing they passed even with the new default
taper active purely by coincidence (generous `rtol=0.15` tolerance +
modest test amplitude) -- not genuine verification of the core
mechanism in isolation from the new one. Added a dedicated
`TestAltitudeTaper` class (6 tests: below-start full amplitude,
at/above-end floor amplitude, monotonic ramp through the transition,
floor=0 means no boost above `taper_end_km`, the confirmed 400/700/0.1
default matches by direct computation, `end_km <= start_km` rejected)
plus two new `CycleConfig` validation tests
(`test_taper_end_not_after_start_rejected`,
`test_taper_floor_out_of_range_rejected`) and an extended defaults check.
Full suite: 379 passed (up from 371). Not yet re-verified against real
data -- the user's original observation was from a real run using the
old flat-amplitude boost; the taper itself has only been unit-tested on
synthetic fixtures so far. A follow-up real run (ideally reusing
`step21`'s setup) is needed to confirm the high-altitude EDP waviness is
actually reduced without hurting TEC residual, before treating this as
fully validated.
