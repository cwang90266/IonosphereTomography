# Assimilation_Cycle User Manual

Runs one ionosphere data-assimilation cycle end to end: draws an IRI2020
ensemble, prepares real RO/IGS TEC observations, encodes the ensemble
into one or more parameterizations, assimilates the observations in
batches, and produces every plot/netCDF artifact `Final_Packaging.docx`
specifies. This document is about *running* it on your own data/region;
for the design history and the real-data findings that shaped the
defaults below, see `Assimilation_Cycle_Integration_Plan.md` in the
repository root (particularly §11 for why `PCA_3D_10ex` is recommended
and §12 for the full-scale timing numbers quoted here).

## 1. One-time environment setup

Two things must be true in whatever Python process runs this, every time
(these folders are not installed packages, so nothing here works via a
normal `pip install`):

```bash
# 1. IRI2020 executable -- needed any time you're not loading a
#    precomputed EDPSamples file (see Section 5).
source init_iri2020_env.sh   # from the repo root; sets IRI2020_PATH

# 2. Use the project's own interpreter, not a bare `python3` -- the
#    system Python has none of numpy/scipy/xarray/cartopy/etc.
/opt/anaconda3/bin/python3 your_script.py
```

Inside the script itself, before importing `Assimilation_Cycle`:

```python
import sys
REPO = "/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography"
for p in (REPO, f"{REPO}/Parameterization", f"{REPO}/EDPSamples", f"{REPO}/IRI_Sample_Inputs"):
    sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")   # headless -- do this before any plotting import

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle.package_run import run_package
```

## 2. Minimal example: one cycle, one new region/time window

```python
import numpy as np

cfg = CycleConfig(
    label="my_cycle",                       # names every output file
    output_dir="/path/to/some/output/dir",  # created if missing

    # --- time window and region ---
    start_time="2025-11-18T10:00:00",
    end_time="2025-11-18T11:00:00",
    center_lat=69.6, center_lon=19.2,        # cycle/ROI center
    radius_km=2000.0,                        # observation CAPTURE radius (see Section 4)
    grid_radius_deg=18.0,                    # assimilation GRID radius, in degrees (see Section 4)
    max_tec_per_batch=100_000,               # bigger than any real ray count -> single batch (recommended, see Section 3)

    # --- observation sources: real data on disk ---
    obs_sources="both",                      # "RO", "IGS", or "both"
    ro_kwargs=dict(
        podtc_dir="/path/to/RO_Data",        # directory of podTc2*.nc files
        roi_mode="tangent_point", alt_limit_km=None,
        min_valid_rays=50, max_rays_per_occultation=200, include_abel=True,
    ),
    igs_kwargs=dict(
        stations=["TRO1", "WUTH"],           # 4-char station IDs
        cache_dir="/path/to/RINEX_Cache",
        roi_mode="pierce_point", alt_limit_km=None,
        # No CDDIS/Earthdata credentials configured anywhere in this repo
        # -- always point at pre-fetched local files, one per station:
        local_obs_by_station={
            "TRO1": "/path/to/RINEX_Cache/TRO100NOR_S_..._MO.crx",
            "WUTH": "/path/to/RINEX_Cache/WUTH00NOR_R_..._MO.crx",
        },
        local_nav="/path/to/RINEX_Cache/BRDC00IGS_R_..._MN.rnx",
        local_dcb="/path/to/RINEX_Cache/CAS0OPSRAP_..._DCB.BIA",
        rinex_version=3, use_iri=False,
        min_valid_epochs=50, max_rays_per_arc=200, epoch_mode="center",
    ),
)

result = run_package(cfg)
```

That's it -- `run_package` runs the whole 9-step workflow (see Section 6)
and returns a `PackageResult` (`result.edp_samples`, `result.batches`,
`result.results_by_style`). Everything else below is what you'll want to
adjust for different data, different scale, or faster iteration.

## 3. `CycleConfig` field reference

Only fields you're likely to actually touch are described here in depth;
every field has a docstring in `cycle_config.py` if you need more.

### Time window and region

| field | meaning |
|---|---|
| `start_time`, `end_time` | ISO datetime strings, the assimilation window |
| `center_lat`, `center_lon` | cycle/ROI center, degrees |
| `radius_km` | **observation capture radius** in km -- how far out `prepare_ro_observations`/`prepare_igs_observations` will look for occultations/arcs |
| `grid_radius_deg` | **assimilation grid radius** in *degrees* -- how large an area `EDPSamples` actually builds a model grid over |

`radius_km` and `grid_radius_deg` are deliberately two different
quantities in two different units, and this is the single most common
way to misconfigure a new cycle: an RO capture net (`radius_km`, often
1000-2000 km, generous so occultations whose *line of sight* passes near
the region are caught even if the tangent point itself is farther out)
should be **much larger** than the region you actually want a model grid
over (`grid_radius_deg`, degrees -- e.g. `18.0` degrees is ~2000 km). If
you set `grid_radius_deg` too small relative to where your real
observations actually sit, most rays will fall entirely outside the
assimilation grid and every style's fit will be badly biased -- this was
a real bug found and fixed mid-session (`Assimilation_Cycle_Integration_Plan.md`
§10, the grid-domain finding): a 556 km-radius grid against a 2000 km
observation capture radius left **0-19%** of real rays inside the grid at
all. If you don't set `grid_radius_deg` explicitly, it silently falls
back to `radius_km`'s bare number reinterpreted as degrees -- convenient
for a tiny synthetic test, actively wrong for real data. **Always set
`grid_radius_deg` explicitly for a real cycle**, sized to the actual
region you want to assimilate over (not the observation capture radius).

### Grid resolution

| field | default | meaning |
|---|---|---|
| `altitude_grid` | 90-900 km @ 10 km (82 levels) | vertical grid; pass any 1-D array, including non-uniform spacing |
| `horizontal_resolution_deg` | 2.5 | target horizontal point spacing |
| `n_ensemble` | 2000 | IRI2020 ensemble size |

These are the confirmed real-data production defaults (verified against
real Tromsø data this session, §11/§12). Reducing any of them speeds up
every downstream step but changes results -- see Section 7 for what each
actually costs.

### Styles

| field | default | meaning |
|---|---|---|
| `styles` | `["ANCHOR", "PCA_3D_10ex", "PCA_1D_10ex"]` | styles a **packaged run** (`run_package`) runs and cross-compares |
| `hyper_params_by_style` | `{"PCA_3D_10ex": {"retaining_threshold": 0.9999}, "PCA_1D_10ex": {"retaining_threshold": 0.99999}}` | PCA styles' retained-variance threshold (`1 - removal_threshold`); `ANCHOR` needs none |
| `style` / `hyper_params` | `"density_10ex"` / `None` | single-style fields, used only by the lower-level `cycle_driver.run_cycle`/`style_sweep.run_style_sweep`, not by `run_package` |

**Recommendation from real-data testing** (`Assimilation_Cycle_Integration_Plan.md`
§11.4/§12.1): `PCA_3D_10ex` gave the best accuracy of every style tested,
at a fraction of the state size, and ran ~65x faster than `ANCHOR` at
full production scale (2 minutes vs. 128 minutes). `ANCHOR` is kept in
the default `styles` list deliberately (not a bug, not an oversight) --
per the user, it's retained for its role as a heritage/legacy comparison
point used across many prior projects, not because it's the recommended
choice for new work. If you only care about the best result fastest, run
with `styles=["PCA_3D_10ex"]` alone; keep `ANCHOR` in the list when you
specifically want that comparison, and budget real time for it (Section 7).

### Batching and RO filtering

| field | default | meaning |
|---|---|---|
| `max_tec_per_batch` | `None` (disabled) | cap on total rays per assimilation batch; set `300` for `Final_Packaging.docx`'s stated default. Entries are never split across batches. |
| `batch_size` | 200 | fallback entry-count batching policy, used only when `max_tec_per_batch` is `None` |
| `min_tangent_alt_km` | `None` (disabled) | drop RO rays with `tangent_alt_km` below this before batching; IGS unaffected |
| `obs_sigma` | 1.0 | constant TEC observation-error std (TECU), same for RO and IGS |

### Grid margin and inter-batch inflation (both opt-in, default off)

| field | default | meaning |
|---|---|---|
| `grid_margin_km` | 0.0 | extra radius (km) padded onto `effective_grid_radius_deg` beyond `grid_radius_deg`, for more state DOF near the domain edge |
| `inter_batch_inflation_factor` | `None` | multiplicative ensemble-spread inflation applied after every batch (`> 1.0` inflates; `None`/`1.0` = pure persistence, the original default) |

Tested against real data (`Assimilation_Cycle_Integration_Plan.md` §13):
a first pass looked like `inter_batch_inflation_factor` was a real
improvement (sweeping `{1.02, 1.05, 1.1, 1.2}`, final per-batch RMSE fell
9.56 -> 7.03 at `1.2`) -- **but §13.1 found this reverses under a
pooled/retrospective check.** The per-batch metric only ever scores a
batch's state against that same batch's own data, immediately after the
update, so it cannot see whether the *final* state stays consistent with
*earlier* batches. Re-scoring the truly final state against every batch's
own real observations (pooled) showed baseline (no inflation) stays sane
(pooled RMSE 8.94) while `factor=1.2` blows up to **4042** (dominated by
one earlier batch scoring 14993 when checked against the much-later
final state) -- consistent with classic EnKF divergence from repeated,
non-localized covariance inflation amplifying spurious sample
correlations over many batches. **Do not use `inter_batch_inflation_factor`
as currently implemented without localization or a much smaller
factor** -- the `1.1`-`1.2` recommendation from the first pass is
withdrawn. `grid_margin_km` is unaffected by this correction and remains
**not recommended**: a controlled A/B at matched ensemble size
(`n_ensemble=300`, baseline `n_geo=161` vs. `+500km` margin `n_geo=250`)
made *every* RO's fit worse (mean +2.4 TECU, overall final RMSE 9.80 ->
12.00) -- the margin adds grid area/state dimension without any new
observations to constrain it. Both fields stay available (opt-in, default
off) but neither has a validated case for turning on yet.

### Diagonal covariance boosting (opt-in, default off)

| field | default | meaning |
|---|---|---|
| `diagonal_boost_amplitude` | `None` | injected perturbation's per-point std, as a fraction of that point's own ensemble std (`None`/`0` = disabled, original ensemble used as-is) |
| `diagonal_boost_vertical_scale_km` | 30.0 | Gaussian smoothing length (km) along altitude for the injected noise field |
| `diagonal_boost_horizontal_scale_km` | 200.0 | Gaussian smoothing length (km, great-circle) over the horizontal point cloud for the injected noise field |
| `diagonal_boost_rng_seed` | `None` | seeds the noise draw -- set and hold fixed for any comparison across boost settings, same standing lesson as `AnalysisConfig.rng` (§14) |

User-proposed (2026-09-27), motivated by real occultations (e.g.
`podTc2_...E34.00...`) showing large forecast *and* analysis residuals in
the full-scale run's TEC/EDP profile plots -- a plausible symptom of the
raw IRI2020 climatological ensemble's low *effective* rank (the same
finding behind PCA_3D_10ex needing only ~110 of ~13202 nominal
dimensions): member-to-member variation is dominated by a few smooth,
broadly-correlated driving-index modes, which can leave genuine
small-scale structure outside the EnKF's reachable correction subspace no
matter how much data is assimilated.

**What it does** (`diagonal_boost.py`): draws one independent
standard-normal field per ensemble member on the raw `EDPs` grid,
smooths each field vertically (Gaussian kernel over the real altitude
coordinate) and horizontally (Gaussian kernel over great-circle distance,
so it works directly on the unstructured/mesh-based horizontal grid),
renormalizes each `(height, geo)` point's smoothed field to exactly
zero-mean/unit-std across members (so the smoothing kernel's own
variance reduction doesn't matter -- the achieved perturbation always has
precisely `amplitude x that point's original std`, and the ensemble mean
is never shifted), and adds the result to the raw ensemble before any
parameterization is fit -- applied inside
`ensemble_init.load_or_build_edp_samples`, so it works identically
whether the base ensemble was just built or loaded via
`edp_samples_path`, and (if `edp_samples_output_path` is set) the *boosted*
ensemble is what gets saved.

**Must be applied before parameterization, not after** -- this is why
it's wired into `load_or_build_edp_samples` rather than as a later,
separate step: fitting a PCA basis to the *un-boosted* ensemble and
adding noise afterward would just have `retaining_threshold` discard most
of the injected variance as below-threshold noise, defeating the purpose
for `PCA_3D_10ex`/`PCA_1D_10ex` specifically.

**Root cause found and fixed -- `diagonal_boost_log_space` is required
for `density_10ex`/`PCA_*_10ex`.** An initial held-out sweep (linear
space, the default) found catastrophic real-data results for
`PCA_3D_10ex` at amplitude 0.1-0.3 (held-out RMSE into the millions) and
non-convergence at 0.8-1.2. Testing `density_10ex` (fixed 13202-dim
state, no adaptive sizing) found near-identical failure numbers despite
`n_state` never changing -- ruling out state-dimension growth as the
cause. The real mechanism: linear-space perturbation, scaled by each
point's own std (which can exceed the local mean by up to 2x in this real
ensemble), pushes some members' density below zero, triggering a clip to
a small positive floor; `density_10ex`/`PCA_*_10ex` then take `log10` of
that clipped value, turning it into an 8-12 order-of-magnitude
discontinuity next to typical surrounding values -- severely corrupting
the ensemble statistics those styles actually fit. Confirmed by testing
`raw` (no `log10` transform, one-shot linear update): no catastrophic
values at any amplitude, a real ~14% held-out RMSE improvement at
amplitude~0.2, robust across seeds.

**Fix:** `diagonal_boost_log_space=True` perturbs `log10(density)`
instead of linear density (converted back via `10**(...)`) -- no
positivity constraint, so no clip is ever needed. Re-running the
`PCA_3D_10ex` sweep with this setting gave a clean, large, robust win: no
catastrophic values, every amplitude converges, and a real optimum
around amplitude~0.8 on this first (single-split) pass, giving a ~65%
held-out RMSE reduction (33.44 -> 11.59), tightly reproducible across 3
boost seeds (11.50-11.60) -- **revised down to amplitude~0.5 below**
once checked against multiple splits, so treat 0.8 here as a
now-superseded data point, not the recommendation. `n_state` still grows
substantially under boosting (110 -> ~1900) -- confirming state-dimension
growth itself was never the problem, only the linear-space clip artifact
was.

**Recommendation:** always set `diagonal_boost_log_space=True` when using
`diagonal_boost_amplitude` with `density_10ex`/`PCA_*_10ex` -- the
default `log_space=False` remains correct and validated only for `raw`.
A multi-split cross-check (3 independent fit/holdout splits, not just
the one used above) confirmed the improvement is real and replicated at
moderate amplitude (0.3-0.8 -- every split improves over its own
baseline), but revised the recommended value **down to amplitude~0.5**:
at amplitude=0.8, one of the 3 splits (not the one originally tested)
still looked fine, but at 1.2-2.0 that same split blew up catastrophically
(RMSE ~52-55) while the others didn't -- instability invisible from a
single split. Amplitude~0.5 gives essentially the same average
improvement (~6.6 vs 0.8's ~7.0 mean held-out RMSE across splits, both
down from baseline's ~17.1) while sitting further from where things break.

### Batch size and assimilation order (nonlinear styles only)

Unlike a linear Kalman filter, this filter's iterated-Gauss-Newton
nonlinear styles (`density_10ex`, `ANCHOR`, all `PCA_*_10ex`) are **not**
invariant to batch size or assimilation order -- each batch re-linearizes
at the current mean while holding the ensemble sensitivity fixed for that
batch, so different batch boundaries/orderings walk different sequences
of local linearizations and can land on different final states. Tested on
real data (`Assimilation_Cycle_Integration_Plan.md` §14), with
`AnalysisConfig.rng` explicitly seeded and held fixed across every
compared condition (**do this for any such comparison** -- unseeded, the
perturbed-obs noise alone differs run to run and confounds the result;
confirmed the noise floor here is only ~2%):

- **Batch size:** fewer/larger batches were monotonically better *and*
  faster on the real 1-hour test window (`cap=150`: pooled RMSE 9.60,
  18.9s -- `cap=1000`: 8.64, 5.7s -- the *entire* window as one batch,
  2545 rays/38 entries: 8.67, 4.2s, converged). No upper batch-size limit
  was found within this dataset's scale; each additional batch boundary
  pays a fresh-linearization-restart cost a single joint solve avoids.
  Recommendation: prefer fewer/larger batches (`max_tec_per_batch` large
  or unset, letting entry-count `batch_size` dominate, or explicitly one
  giant batch) over many small ones, unless a real operational constraint
  (streaming/incremental data arrival, memory) forces smaller batches.
- **Order:** with `cap=300` (13 batches) held fixed, most size/chronology
  orderings clustered in a narrow band (chronological 8.99, ascending-size
  8.95, descending-size 11.21) but strict reverse-chronological was a
  dramatic outlier (40.68, ~4.5x worse) -- a real effect, not noise.
  **Better signal than size or chronology: order by forecast-observation
  misfit, decreasing** -- ranking each batch by `|forecast - measured|`
  against the current state and assimilating biggest-disagreement-first
  tied for the best pooled RMSE found (9.03) *and* had the best
  convergence rate of every order tested (9/13). The reverse (smallest
  disagreement first) was nearly as bad as reverse-chronological (38.45).
  Misfit is not just a proxy for batch size -- two nearly-identical-size
  batches had almost opposite misfit (58.6 vs 23.2 TECU), and
  descending-misfit clearly beat descending-size (9.03 vs 11.21).
  Mechanistic account: processing the biggest disagreements first uses
  the ensemble's full, unspent spread for the large corrections that need
  it; small-residual data then arrives later as cheap fine refinement.
  **If batches must be processed sequentially, rank by decreasing
  forecast-observation misfit** (recomputed against the current state as
  data arrives, since a batch's true misfit shifts as the state improves)
  -- the best-evidenced strategy so far, ahead of both size heuristics and
  chronology; avoid ascending-misfit and reverse-chronological. The
  cleanest fix remains avoiding the ordering question entirely via
  fewer/larger batches per the point above.

### Observation sources

| field | meaning |
|---|---|
| `obs_sources` | `"RO"`, `"IGS"`, or `"both"` |
| `ro_kwargs` | forwarded to `prepare_ro_observations` -- see the minimal example above for the keys you'll actually set |
| `igs_kwargs` | forwarded to `prepare_igs_observations` -- same |

`prepare_igs_observations` has no way to request a time window from the
underlying RINEX processing -- it always processes the **entire day**
before filtering down to `start_time`/`end_time` afterward. This is a
real, pre-existing cost (not something this package controls), and is
the main reason IGS takes minutes even for a one-hour cycle (Section 7).

### Precomputed-file mode (the fast-iteration path)

Every expensive step -- IRI2020's ensemble draw, RO parsing + Abel
inversion, IGS's full-day RINEX processing -- can be **saved once and
reloaded** instead of recomputed on every run. Two matching fields per
artifact:

| load (skip recompute if set) | save (write after computing, if set) |
|---|---|
| `edp_samples_path` | `edp_samples_output_path` |
| `parameterized_edp_samples_path` | `parameterized_edp_samples_output_path` |
| `ro_observations_path` | `ro_observations_output_path` |
| `igs_observations_path` | `igs_observations_output_path` |
| `iri_sample_inputs_path` | `iri_sample_inputs_output_path` |

**Workflow**: run once with the `*_output_path` fields set (or just let
`run_package` default them under `output_dir`, which it does
automatically); on every later run against the *same* time window/region/
grid, pass the same paths back in via the `*_path` fields instead, and
those steps load from disk in seconds rather than recomputing. This is
how the full-scale run in Section 7 avoided repeating an 18-minute
IRI2020 build and several minutes of IGS processing. **These paths are
only valid for the exact time window/region/grid/ensemble-size they were
computed for** -- reusing `edp_samples_path` across a different region or
grid resolution will silently give you the wrong grid.

### Output

| field | meaning |
|---|---|
| `output_dir` | required for `run_package`; every artifact is written under here |
| `label` | required for `run_package`; prefixes every output filename |
| `save_intermediate_ensembles` / `ensemble_output_dir` | `run_package` always turns this on per style (`{label}_{style}_ensembles/`) |

## 4. What `run_package` produces

Given `label="foo"`, `output_dir=OUT`, under `OUT` you get (see
`Final_Packaging.docx`'s 9-step workflow, or `package_run.py`'s docstring
for the exact mapping):

```
foo_iri_sample_inputs.pkl              # step 2
foo_iri_input_distributions.png        # step 3
foo_edp_samples.nc                     # step 4
foo_horizontal_grid.png                # step 4
foo_mean_density_{100,200,300,400,500}km.png   # step 4
foo_{style}_parameterized.nc           # step 4, one per style
foo_{style}_reconstruction_error.png   # step 4, one per style
foo_ro_observations.nc, foo_igs_observations.nc   # step 5
foo_observations_geolocation.png       # step 5 -- RO tangent tracks + IGS pierce points, on the SAME projection/extent as foo_horizontal_grid.png (not a separate Orthographic/global view)
foo_obs_operator/{ro_label}.png        # step 5, one per RO occultation
foo_obs_operator_sum_all_ro.png        # step 5 -- same sum(H)-by-altitude view as the per-RO plots, but summed over every RO occultation combined
foo_{style}_ensembles/ensemble_batch_NNNN.nc   # step 6, one per batch per style
foo_{style}/rmse_reduction.png, rank_histogram.png, effective_rank.png   # step 6
foo_{style}/igs_tec_scatter.png        # step 6, one per style -- 2-panel scatter, forecast/analysis TEC vs. measured TEC, IGS rays only, pooled across all batches
foo_{style}_analysis_mean_density_{alt}km.png   # step 6, one per style, same 5 altitudes as the step-4 forecast plots -- the optimal (final analysis) EDP field's spatial distribution, directly comparable to the step-4 prior/forecast mean-density plots at the same altitude
foo_{style}/tec_profiles/batchNNNN_{ro_label}.png   # step 7, one per RO per batch
foo_{style}/edp_profiles/batchNNNN_{ro_label}.png   # step 8, one per RO per batch (skipped if no ray fell in the 250-350km window)
foo_style_comparison.png               # step 9
```

## 5. Reusing this for a validation profile (future use, already supported)

`output.plot_edp_profile_comparison` -- the function behind step 8's
per-RO plots -- was built to be reused directly for the docx's stated
future need (comparing against an independently-specified validation
profile, not one derived from an RO occultation): call it with `lat=`/
`lon=` given explicitly instead of an `entry` whose tangent point gets
used automatically. No new code is needed when that requirement arrives.

## 6. Performance expectations (real numbers, not estimates)

All measured this session against the local Tromsø sample data
(15 RO occultations, 2 IGS stations), on the production grid
(90-900km/10km, 2.5deg horizontal, `n_geo=161`):

| step | n_ensemble=20, coarse grid (smoke test) | n_ensemble=2000, production grid |
|---|---|---|
| IRI2020 ensemble build | a few seconds | **~18 minutes** |
| RO preparation (15 occultations) | ~15s | ~15s (independent of `n_ensemble`) |
| IGS preparation (2 stations) | a few minutes | a few minutes (full-day processing, independent of `n_ensemble`) |
| assimilation, `ANCHOR` | ~43s | **~128 minutes** |
| assimilation, `PCA_3D_10ex` | ~20s | **~2 minutes** |
| assimilation, `PCA_1D_10ex` | ~20s | ~4.5 minutes |

Takeaways:
- **IRI2020's ensemble build and RO/IGS preparation are the same cost
  regardless of which styles you run** -- use precomputed-file mode
  (Section 3) so you only pay for these once per region/time window,
  then iterate freely on styles/batching/filtering.
- **`ANCHOR`'s cost scales with `n_ensemble x n_geo`** (one nonlinear fit
  per ensemble member per grid point -- 322,000 individual fits at
  production scale) and dominates total wall-clock time by nearly two
  orders of magnitude versus the PCA styles. If you're iterating on
  anything other than `ANCHOR` itself, drop it from `styles` for that
  iteration and add it back for a final comparison run.
- For a first real run against new data, do a smoke test at small
  `n_ensemble` (10-50) and a coarse grid first (as above) to catch
  configuration mistakes (wrong `podtc_dir`, wrong `grid_radius_deg`,
  missing IGS local files) before committing to a multi-hour run.

**The table above uses `max_tec_per_batch=300` (13 batches), the
`Final_Packaging.docx`-stated default -- but Section 3's batch-size
findings recommend a single batch instead** (`max_tec_per_batch` set
above the real total ray count), which is both more accurate and
substantially faster for the assimilation step specifically (the
parameterization costs above -- IRI2020 build, `ANCHOR`'s Chapman fit --
are unaffected by batching, since they happen once regardless). Measured
on the same real production-scale data, single-batch: `ANCHOR` assimilation
~12 minutes (vs. ~88 minutes across 13 batches -- ~52 total with the
Chapman fit, vs. ~128 total); `PCA_3D_10ex`/`PCA_1D_10ex` assimilation
~2 minutes each (both were already fast in 13-batch mode; single-batch
mainly helps `ANCHOR`, which pays the most per-batch linearization-restart
overhead). Recommended default going forward: set `max_tec_per_batch`
large enough to force one batch, not `300`.

## 7. Troubleshooting

- **`ValueError` about missing `stations`/`cache_dir`** -- `igs_kwargs`
  needs at least these two keys; `prepare_igs_observations` requires
  them.
- **Every ray from a real cycle looks like it's missing from the grid /
  suspiciously bad fits everywhere** -- check `grid_radius_deg` against
  `radius_km` (Section 3's region fields); this is the single most
  common misconfiguration.
- **`prepare_igs_observations` trying to hit CDDIS / asking for
  credentials** -- pass `local_obs_by_station`/`local_nav`/`local_dcb` in
  `igs_kwargs` pointing at files you already have locally; no `.netrc` is
  configured anywhere in this environment.
- **A style fails to converge in some batches** -- expected and, per the
  real-data investigation (`Assimilation_Cycle_Integration_Plan.md`
  §11), not necessarily a sign of a bad result: `AnalysisConfig`'s
  `converged` flag combines a relative-step-size test and a relative-
  residual-improvement test (added this session specifically because the
  step-size test alone was too strict for real, large, noisy batches).
  Check the actual RMSE reduction (`{style}/rmse_reduction.png`) before
  assuming a non-converged batch produced a bad result.
