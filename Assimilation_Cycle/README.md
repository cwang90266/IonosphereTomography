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

As of `ISR_Integration_Plan.md`, a run can also compare its analysis EDP
directly against real incoherent-scatter-radar (ISR) ground truth, and a
4th parameterization style (`PCA_1D_10ex_ISR`) can use a basis fit from
real ISR profiles instead of the IRI2020 climatology. See Section 8
(input data, including ISR), Section 9 (the ISR comparison/4th-style
config and setup), and Section 10 (a complete end-to-end example with
all 4 styles and ISR comparison) for everything specific to that.

## 1. Installation and one-time environment setup

### 1.1 Python dependencies

Nothing under this repository is an installed package -- it only works
by putting the right folders on `sys.path` (below), so there is no
`pip install .` step. What *does* need to be present in whichever Python
environment you run scripts with: `numpy`, `pandas`, `scipy`, `xarray`,
`netCDF4`, `matplotlib`, `cartopy`, `pyproj`, `tqdm`. In this project's
own environment these are all already installed under
`/opt/anaconda3` (see below) -- on a new machine, `pip install numpy
pandas scipy xarray netCDF4 matplotlib cartopy pyproj tqdm` (or the
equivalent `conda install`) covers everything `Assimilation_Cycle` and
the modules it imports actually use. `cartopy` in particular is
frequently the fussiest to install from a bare `pip` (it needs the
system PROJ/GEOS libraries) -- `conda`/`mamba` is the easier path if you
have a choice.

### 1.2 The IRI2020 Fortran driver

Any step that runs IRI2020 itself (anything *not* loaded via a
precomputed-file path, Section 3) needs a compiled executable,
`iri2020_namelist_driver`, built from the Fortran sources under
`iri2020_new/src/iri2020/src/` (`irifun.for`, `irisub.for`, and friends,
plus the `iri2020_namelist_driver.f90` entry point). Build it once with:

```bash
cd iri2020_new/src/iri2020   # from the repo root
make                          # configures a CMake build/ tree on first run, then builds
```

**Rebuild whenever any of the `.for`/`.f90` sources change** -- `make`
only recompiles what it detects changed via normal file-timestamp
dependency tracking, but if you're not sure a change was actually picked
up, force a clean rebuild:

```bash
make clean && make
```

This is not a hypothetical caveat: a real bug in `irifun.for` was found
and fixed mid-project (co-worker + Claude, 2026-10-09) after two people
running what looked like the identical code got different results --
the actual cause was one of them running against a *stale, pre-fix*
compiled executable. After pulling in any Fortran source change, confirm
the rebuild actually happened before trusting results against it:

```bash
ls -la iri2020_new/src/iri2020/iri2020_namelist_driver   # mtime should be newer than the .for/.f90 you just changed
```

A quick end-to-end smoke test (a handful of samples, one location) is
cheap insurance before committing to a multi-hour real run against a
freshly (re)built executable:

```python
import sys
sys.path.insert(0, "IRI_Sample_Inputs"); sys.path.insert(0, "EDPSamples")
from IRI_Sample_inputs import IRI_Sample_Inputs
from edp_samples import EDPSamples
import numpy as np

sp = IRI_Sample_Inputs("2025-11-18T10:30:00").randomSamples(
    hour_sample_range=1, f107_sample_range=2, ap_sample_range=1,
    ig_sample_range=1, rz_sample_range=1, nSample=5)
edp = EDPSamples(DateTime="2025-11-18T10:30:00", geo_type="Point",
                  altitude=np.arange(100, 301, 50), sampling_parameters=sp,
                  evaluate_iri=1, Lon=19.2, Lat=69.6)
print(edp.edps.shape)   # (5, 1, 5); values should be physically sane (~1e9-1e12 m^-3)
```

### 1.3 Every run: two things must be true

In whatever Python process runs this, every time:

```bash
# 1. IRI2020 executable -- needed any time you're not loading a
#    precomputed EDPSamples file (see Section 3's "Precomputed-file mode").
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

### IRI2020 ensemble spread

| field | default | meaning |
|---|---|---|
| `iri_spread_kwargs` | see below | forwarded to `IRI_Sample_Inputs.randomSamples` as `hour_sample_range`/`f107_sample_range`/`ap_sample_range`/`ig_sample_range`/`rz_sample_range` -- how far each IRI2020 driving index is allowed to vary member-to-member |

Default (`default_iri_spread_kwargs()` in `cycle_config.py`, confirmed
with the user 2026-09-29 after directly inspecting the real underlying
IRI2020 input files):

```python
{"hour_sample_range": 3, "f107_sample_range": 30, "ap_sample_range": 30,
 "ig_sample_range": 12, "rz_sample_range": 12}
```

**Each number is a window of table *rows*, not a uniform time unit** --
the two underlying files have different native cadences, confirmed by
reading their actual structure, not assumed:
- `apf107.dat` (`f107`/`ap`) is indexed **one row per calendar day** --
  `f107_sample_range=30`/`ap_sample_range=30` means **+/-30 real days**.
  (`ap`'s value within a day is itself 8 genuinely 3-hourly sub-values,
  but the window steps in whole days, not 3-hour steps.)
- `ig_rz.dat` (`ig12`/`rz12`) is indexed **one row per calendar month**
  (confirmed via its own `Start_end_month` header) --
  `ig_sample_range=12`/`rz_sample_range=12` means **+/-12 real months**.
  These are 12-month *smoothed* indices by construction, so they move
  slowly regardless of window width -- 12 is a real month-scale window
  despite the smaller number, not a narrow one.
- `hour` indexes local time-of-day directly in whole hours --
  `hour_sample_range=3` means **+/-3 real hours**, no file/cadence
  question.

This replaced an earlier default of `{}` (no spread at all) -- a real bug
found this way: every `*_sample_range` left unset means every one of the
`n_ensemble` members draws the exact same (nominal) driving indices, a
fully degenerate ensemble with zero climatological variability
(`Assimilation_Cycle_Integration_Plan.md` §21-22: the
`foo_iri_input_distributions.png` histograms showed single spikes because
of this). Only matters for a **fresh** IRI2020 build (`edp_samples_path`
unset) -- has no effect in precomputed-file mode, since the real
ensemble there comes from the cached file, and the distribution plot now
reads that file's own real `sampling_parameters` regardless (Section 4).
Verified against the real event-date tables: the new default gives 7
distinct `hour` values, 58 distinct `f107` values, 25 distinct `ap`
values, 24 distinct `ig12`/`rz12` values each (was 2-3 distinct values
per index under the old default). Check `foo_iri_input_distributions.png`
after any fresh-build run to confirm real spread, the same way you'd
check `rmse_reduction.png` before trusting a result.

### Styles

| field | default | meaning |
|---|---|---|
| `styles` | `["ANCHOR", "PCA_3D_10ex", "PCA_1D_10ex", "PCA_1D_10ex_ISR"]` | styles a **packaged run** (`run_package`) runs and cross-compares -- 4 as of `ISR_Integration_Plan.md`, see below |
| `hyper_params_by_style` | `{"PCA_3D_10ex": {"retaining_threshold": 0.9999}, "PCA_1D_10ex": {"retaining_threshold": 0.99999}}` | PCA styles' retained-variance threshold (`1 - removal_threshold`); `ANCHOR` needs none; `PCA_1D_10ex_ISR` needs none *here* either (its basis comes from `isr_pca_basis_path`, Section 9, not a threshold) |
| `style` / `hyper_params` | `"density_10ex"` / `None` | single-style fields, used only by the lower-level `cycle_driver.run_cycle`/`style_sweep.run_style_sweep`, not by `run_package` |

**`PCA_1D_10ex_ISR`** (added `ISR_Integration_Plan.md`) is mechanically
identical to `PCA_1D_10ex` -- same log10-space PCA decode -- except its
basis is fit from real ISR-measured profiles instead of this cycle's own
IRI2020 ensemble, and it *never* fits a basis from the ensemble it's
given (unlike every other PCA style, supplying only
`retaining_threshold` for this style raises rather than silently
fitting one). Requires `isr_pca_basis_path` (or an explicit `'PCA'` in
its `hyper_params_by_style` entry) -- see Section 9 for how to build
that basis file. Real full-scale results (`ISR_Integration_Plan.md`'s
evaluation): best or tied-best of the 4 default styles against real
RO+IGS TEC, second-best against real ISR ground truth, converges
reliably (unlike `ANCHOR`).

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

### ISR ground-truth comparison (optional, default off)

| field | default | meaning |
|---|---|---|
| `isr_file_path` | `None` | path to a preprocessed ISR netCDF (station-agnostic -- station name/lat/lon are read from the file's own attrs, see Section 8.3). `None`: no ISR comparison is run, no other field below matters. |
| `isr_pca_basis_path` | `None` | path to a precomputed `isr_pca_basis.IsrPcaBasis` netCDF (Section 9.2) -- required for the `PCA_1D_10ex_ISR` style; unused by every other style. |
| `isr_range_percentile` | `0.0` | `0.0`: the ISR "range" plot shades the literal min-max of every ISR profile in the cycle's window; a value in `(0, 50)` shades the `[p, 100-p]` percentile band instead (less sensitive to one outlier scan). |

Setting `isr_file_path` alone (no `isr_pca_basis_path`) is enough to get
the full ISR comparison tooling (Section 9.1) for whichever styles you
already run -- it has no dependency on the 4th style. See Section 9 for
the full picture (both ISR use cases, the basis-build procedure, and a
worked example).

### Reproducibility across styles/conditions

| field | default | meaning |
|---|---|---|
| `analysis_rng_seed` | `None` | seeds `GeneralEnKFDriver`'s `AnalysisConfig.rng` (via `cycle_driver.run_cycle`) -- the generator actually used for the stochastic perturbed-obs noise inside the analysis step. `None` (default): a fresh, unseeded generator every call. |

**Set this explicitly whenever comparing two or more runs** (styles,
boost settings, anything) -- a real, previously-undocumented gap this
project's own history ran into twice before this field existed
(`Assimilation_Cycle_Integration_Plan.md` §13/§14): without it, the
stochastic analysis noise differs run to run and confounds whatever
you're actually trying to compare. This is a *different* generator from
`diagonal_boost_rng_seed` (controls the injected boost noise, not the
analysis step) -- set both, independently, for a fully reproducible
comparison.

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
| `diagonal_boost_horizontal_scale_km` | 500.0 (changed from 200.0, see below) | Gaussian smoothing length (km, great-circle) over the horizontal point cloud for the injected noise field |
| `diagonal_boost_rng_seed` | `None` | seeds the noise draw -- set and hold fixed for any comparison across boost settings, same standing lesson as `AnalysisConfig.rng` (§14) |
| `diagonal_boost_taper_start_km` | 400.0 | altitude below which the boost applies at full `diagonal_boost_amplitude` |
| `diagonal_boost_taper_end_km` | 700.0 | altitude at/above which the boost is held at `diagonal_boost_amplitude x diagonal_boost_taper_floor` (linear ramp between start/end) |
| `diagonal_boost_taper_floor` | 0.1 | boost amplitude fraction retained above `diagonal_boost_taper_end_km` |

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

**Altitude taper (2026-09-29), motivated by a real-data finding.** After
`iri_spread_kwargs` was widened to physically-real driving-index windows
(next section), the user noticed analysis EDPs in a fresh full-resolution
run became visibly wavy at high altitude, without any corresponding
improvement in TEC residual. Root cause: TEC is a line integral dominated
by the F2-peak region (~250-350km, already this project's reference
window for EDP-profile plots) -- a density perturbation well above that
has very little effect on TEC, so the EnKF has almost no observational
leverage to correct or constrain whatever the boost injects up there,
while topside density is also more sensitive to the now-wider
driving-index spread. `apply_diagonal_boost` now scales `amplitude` by an
altitude-dependent taper: full strength at/below
`diagonal_boost_taper_start_km`, linearly down to
`diagonal_boost_taper_floor x amplitude` by `diagonal_boost_taper_end_km`,
held at that floor above. Defaults (400km/700km/0.1) confirmed with the
user. Pass `diagonal_boost_taper_floor=0.0` for no boosting at all above
`diagonal_boost_taper_end_km`, or `diagonal_boost_taper_start_km` beyond
the grid's own max altitude for effectively no taper (full amplitude
everywhere, the old behavior). Not yet re-validated against real data
(only unit-tested on synthetic fixtures so far) -- the next full
end-to-end run should confirm the high-altitude waviness is actually
reduced without hurting TEC residual.

**Reversal, then correction (2026-10-07/08) -- the amplitude~0.5
recommendation above was validated against the wrong thing.** Every
sweep up to this point scored boosting only against held-out real RO+IGS
TEC. Once this project built independent real-ISR-ground-truth
comparison tooling (`isr_comparison.py`, see `ISR_Integration_Plan.md`)
and ran a full-scale end-to-end comparison, amplitude=0.5 at the
*original* `diagonal_boost_horizontal_scale_km=200` default turned out to
be a 1-3 order-of-magnitude **regression** against the real ISR-measured
density profile in the ~95-440km band (E-region through F2-peak/lower-
topside) -- for every parameterization style tested, including
`PCA_3D_10ex`. The TEC fit looked great because TEC is a vertically-
integrated, shape-degenerate observable: boosting handed the filter extra
spread directions with no independent information to constrain them, and
it exploited that freedom to fit TEC almost exactly by distorting the
vertical shape in ways TEC itself cannot detect -- only a direct density
measurement (real ISR) exposed this. A follow-up real 3-parameter sweep
(amplitude x `diagonal_boost_vertical_scale_km` x
`diagonal_boost_horizontal_scale_km`, scored directly against real ISR)
found the *horizontal* smoothing scale, not amplitude, was the actual
cause: at `horizontal_scale_km=500` (now the default, changed from 200),
every amplitude in 0.1-0.5 improves *both* the TEC fit and the real ISR
match simultaneously (best found: amplitude=0.1/vertical=30/horizontal=500,
ISR RMSE *below* the no-boost baseline). A wider horizontal correlation
length makes the injected perturbation look like a genuine broad-scale
density anomaly rather than small-scale, geographically-incoherent noise
the filter can exploit. **Current recommendation**: use the new default
`horizontal_scale_km=500` with amplitude in 0.1-0.5 (lower favors the ISR
match, higher favors the TEC fit -- a real but mild tradeoff at this
scale, unlike the catastrophic one at 200); `amplitude` itself is still
`None` (off) by default. ANCHOR was excluded from this sweep (too
expensive, and the same full-scale run found it does best *without*
boosting anyway). Full tables in
`Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis/evaluation/`.

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
foo_iri_input_distributions.png        # step 3 -- the REAL driving indices of the ensemble actually built/loaded in step 4 (edp_samples.sampling_parameters), not an independent redraw; a single spike per panel means iri_spread_kwargs was left at its default {} (no spread requested), not a plotting bug
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
foo_{style}/profiles/batchNNNN_{ro_label}.png   # steps 7-8, one per RO per batch per style -- two-panel figure, TEC comparison (left) + EDP comparison (right); EDP panel shows a placeholder message instead of the figure failing if no ray fell in the 250-350km window
foo_cross_style_profiles/batchNNNN_{ro_label}.png   # one per RO per batch, all styles overlaid in each of the two panels (same color per style across every RO's figure, solid=forecast/dashed=analysis) -- for comparing styles directly, not per-style output
foo_style_comparison.png               # step 9
```

**If `isr_file_path` is set** (Section 9), each of the following also
appears, per style under `foo_isr_comparison/{style}/` and once more,
all styles overlaid, under `foo_isr_comparison_cross_style/`:

```
{style}/nearest_point/batchNNNN.png    # Plot A -- that batch's forecast/analysis mean +/- std at the ISR site, overlaid with the single real ISR profile nearest that batch's midpoint
{style}/range_comparison.png           # Plot B -- the real ISR range (shaded band) across the cycle's whole window, overlaid with every batch's analysis profile -- one per cycle, not per batch
{style}/pooled_rmse.csv, pooled_rmse.png   # forecast/analysis RMSE vs. the matched nearest ISR profile, pooled across every batch, by native ISR altitude gate
```

The `_cross_style` versions are the same three artifact types with every
style overlaid in one figure/table instead of one style's own -- directly
comparable to the per-style ones, same filenames, one directory up.

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
| assimilation, `PCA_1D_10ex_ISR` | ~20s | ~2 minutes (similar to the other PCA styles -- its basis is precomputed, not fit per run) |

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
- **ISR comparison (`isr_file_path`) adds negligible time** -- reading
  the preprocessed ISR file and the per-batch nearest-profile match are
  both fast; it does not depend on `n_ensemble` or grid resolution the
  way the steps above do.

**Real full-scale pair timings** (both conditions, all 4 styles, single
batch, real 2025-11-18 Tromsø RO+IGS data, `ISR_Integration_Plan.md`'s
evaluation): a full `run_package` call across `ANCHOR`/`PCA_3D_10ex`/
`PCA_1D_10ex`/`PCA_1D_10ex_ISR` took **~100 minutes** (no boost) or
**~85 minutes** (boosted -- `ANCHOR`'s Chapman fit dominates either way;
boosting does not meaningfully change its own cost). Running the
no-boost **and** boosted conditions back to back, as a before/after
comparison (Section 10's example), is realistically a **~3-3.5 hour**
commitment at this scale -- plan accordingly, and strongly prefer
precomputed-file mode (Section 3) for the shared base ensemble between
the two conditions (the boosted run reuses the no-boost run's saved
`edp_samples_path` rather than rebuilding IRI2020 a second time).

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
- **`ValueError: Parameterization style 'PCA_1D_10ex_ISR' always
  requires a pre-fit 'PCA'...`** -- `isr_pca_basis_path` isn't set (or
  the style's `hyper_params_by_style` entry doesn't carry an explicit
  `'PCA'`). Unlike every other PCA style, this one never fits a basis
  from the ensemble it's given -- see Section 9.2 to build the basis
  file first.
- **`ValueError: density and PCA dimensions are inconsistent` when using
  `PCA_1D_10ex_ISR`** -- the ISR basis (Section 9.2) was built on a
  different altitude grid than this cycle's actual `EDPSamples`. This
  happens most often in precomputed-`edp_samples_path` mode: that file's
  *own* altitude grid is what's actually used, silently overriding
  `cfg.altitude_grid` if they differ (a real bug found exactly this way,
  `ISR_Integration_Plan.md`). Rebuild the ISR basis on the grid the
  cached `EDPSamples` file actually uses (read its `.altitude` directly
  if unsure), not `cfg.altitude_grid`'s nominal default.
- **IRI2020 results differ between two people/machines running what
  looks like identical code** -- check whether `iri2020_namelist_driver`
  was actually rebuilt after the last Fortran source change on both
  sides (Section 1.2); a stale compiled executable silently running
  against old physics is a real failure mode this project hit once.

## 8. Input data: RO, IGS, and ISR

Three independent real-data sources feed a cycle: RO and IGS (the
observations actually assimilated) and, optionally, ISR (ground truth to
validate against, Section 9 -- never assimilated itself).

### 8.1 RO (radio occultation)

`ro_kwargs["podtc_dir"]` points at a directory of `podTc2*.nc` files, one
per occultation (POD-based TEC profile product). Real example from this
project's own local data:

```
/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/RO_Data/
    podTc2_GN04.2025.322.10.15.0031.E34.01_0000.0001_nc
    podTc2_GN04.2025.322.10.20.0027.E02.01_0000.0001_nc
    ...
```

`prepare_ro_observations` (wrapped by `observation_stream.assemble`) scans
every file in the directory, keeps the ones falling inside `start_time`/
`end_time` and the ROI (`center_lat`/`center_lon`/`radius_km`,
`ro_kwargs["roi_mode"]`), runs the Abel inversion on each, and discards
occultations with too few valid rays (`min_valid_rays`). No network
access or credentials are needed -- this is a directory of files you
already have.

### 8.2 IGS (ground-based GNSS TEC)

`igs_kwargs` points at RINEX files for one or more ground stations,
plus the broadcast navigation and DCB (differential code bias) files
every station's processing needs. Real example:

```
/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/RINEX_Cache/
    TRO100NOR_S_20253220000_01D_30S_MO.crx    # one per station (local_obs_by_station)
    WUTH00NOR_R_20253220000_01D_30S_MO.crx
    BRDC00IGS_R_20253220000_01D_MN.rnx        # shared navigation file (local_nav)
    CAS0OPSRAP_20253220000_01D_01D_DCB.BIA    # shared DCB file (local_dcb)
```

No `.netrc`/CDDIS/Earthdata credentials are configured anywhere in this
environment -- always pass `local_obs_by_station`/`local_nav`/`local_dcb`
in `igs_kwargs` pointing at files you already have (the minimal example
in Section 2 shows the exact keys). `prepare_igs_observations` has no way
to request a time window from the underlying RINEX processing itself --
it always processes each station's **entire day** before filtering down
to `start_time`/`end_time` afterward (Section 3's "Observation sources"),
which is why IGS takes minutes even for a one-hour cycle regardless of
`n_ensemble`.

### 8.3 ISR (incoherent scatter radar) -- optional, for ground-truth comparison and/or the 4th style

A preprocessed ISR netCDF, **station-agnostic by schema** -- the file's
own global attributes carry its identity, never hardcoded anywhere in
this code:

| attr/variable | meaning |
|---|---|
| `station_name`, `station_latitude`, `station_longitude` (global attrs) | which ISR site this file is from |
| `altitude` (data var, dim `altitude_gate`) | native altitude gates, km (typically irregular spacing, denser at low altitude) |
| `Ne` (data var, dims `altitude_gate` x `time`) | electron density, m^-3 |
| `time_utc` (data var, dim `time`) | ISO timestamp strings |

The real file used throughout `ISR_Integration_Plan.md`'s development
and evaluation:

```
/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc
```
(EISCAT Tromsø UHF ISR, year 2025, 4-minute cadence, 39 native altitude
gates 82-647km, adaptively median-filtered with gaps already removed/
filled -- see the file's own attrs for the exact filtering policy
applied upstream of this project.) This **raw** file is all that's
needed for direct ISR comparison (Section 9.1). Building the 4th style's
basis (Section 9.2) needs one additional preprocessing step first, since
the basis is defined on the full 90-900km production grid but native ISR
gates only reach ~650km.

## 9. ISR ground-truth comparison and the 4th parameterization style

Two independent uses of ISR data, from `ISR_Integration_Plan.md`:

1. **Direct comparison** (Section 9.1): interpolate the analysis/forecast
   EDP to the ISR site and compare against the real measured profile --
   validates the general RO+IGS retrieval approach against ground truth,
   for any style, with no new parameterization needed.
2. **A new style, `PCA_1D_10ex_ISR`** (Section 9.2): a PCA basis fit from
   real ISR-measured profiles over an extended period, used the same way
   `PCA_1D_10ex`'s IRI2020-fitted basis is, on the hypothesis that real
   ISR data spans a wider range of EDP shapes than the climatology.

### 9.1 Direct comparison (works with any style, no basis needed)

Set `isr_file_path` (Section 3's "ISR ground-truth comparison" table) to
the raw ISR file (Section 8.3) and run `run_package` as usual -- every
style in `cfg.styles` automatically gets the full comparison tooling
(Section 4's artifact list: per-batch nearest-point overlays, the
cycle-wide range plot, pooled RMSE by altitude gate, each per-style and
once more with all styles overlaid). No other setup is required; this
has no dependency on Section 9.2 at all.

### 9.2 Building the ISR-derived PCA basis (needed only for `PCA_1D_10ex_ISR`)

Two steps, each run once per ISR station/region (not per cycle -- reuse
the resulting basis file across runs the same way a precomputed
`edp_samples_path` is reused):

**Step 1 -- extend ISR profiles to the production altitude grid's full
range.** Native ISR gates stop around 650km; the production grid goes to
900km. `extend_isr_edp_with_iri2020.py` (repo root) fits each profile's
`log10(Ne)` above 200km against the leading few modes of a real IRI2020
library, then extends it to 900km with a smoothly-tapered join to the
last real measurement:

```bash
source init_iri2020_env.sh
/opt/anaconda3/bin/python3 extend_isr_edp_with_iri2020.py \
    --isr-file /path/to/your_isr_file.nc \
    --out /path/to/output_dir
# -> /path/to/output_dir/your_isr_file_extended.nc
```

Station latitude/longitude/name are read from the input file's own attrs
(Section 8.3) -- this works unmodified for any ISR station's file with
that schema, not just Tromsø.

**Step 2 -- fit the PCA basis** on the production altitude grid, from
the extended file's full profile record:

```python
import sys
sys.path.insert(0, "/path/to/IonosphereTomography")
from Assimilation_Cycle import isr_pca_basis as ipb
from Assimilation_Cycle.cycle_config import default_altitude_grid

basis = ipb.build_isr_pca_basis(
    "/path/to/output_dir/your_isr_file_extended.nc",
    default_altitude_grid(),              # MUST match the grid your cycle actually uses -- see the troubleshooting entry in Section 7
    retaining_threshold=1 - 1e-4,         # matches this project's PCA_3D_10ex convention
)
ipb.save_isr_pca_basis(basis, "/path/to/output_dir/isr_pca_basis.nc")
print(basis.diagnostics.n_retained, "components retained of",
      basis.diagnostics.n_profiles_used, "profiles used")
```

Real result from the Tromsø file above: 26 components retained from all
8284 profiles (0 dropped) at `retaining_threshold=1-1e-4` -- a real,
substantive basis, not a degenerate one.

### 9.3 Using the style

Point `isr_pca_basis_path` at the saved basis file and include
`'PCA_1D_10ex_ISR'` in `cfg.styles` (it's in `default_styles()` already)
-- `resolve_hyper_params_for_style` loads the basis automatically for
that style only, with no effect on any other style:

```python
cfg = CycleConfig(
    ...,
    isr_file_path="/path/to/your_isr_file.nc",            # Section 9.1, the RAW file
    isr_pca_basis_path="/path/to/output_dir/isr_pca_basis.nc",   # Section 9.2's output
    # styles=default_styles() already includes 'PCA_1D_10ex_ISR'
)
```

## 10. End-to-end example: all 4 styles with real ISR comparison

Complete, runnable script mirroring the real full-scale runs
`ISR_Integration_Plan.md` was evaluated against -- adjust the paths for
your own region/data. Produces two complete artifact sets (no-boost and
boosted), each with all 4 styles and full ISR comparison tooling, under
`output_dir`.

```python
#!/usr/bin/env python3
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path("/path/to/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs"):
    sys.path.insert(0, str(p))

from Assimilation_Cycle.cycle_config import CycleConfig, default_styles, default_hyper_params_by_style
from Assimilation_Cycle.package_run import run_package

OUT = Path("/path/to/output_dir")

cfg_base = CycleConfig(
    # --- time window and region (Section 3) ---
    start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
    center_lat=69.6, center_lon=19.2,
    radius_km=2000.0, grid_radius_deg=18.0,
    max_tec_per_batch=100_000,   # single batch, recommended (Section 3)

    # --- real RO/IGS input data (Section 8.1/8.2) ---
    obs_sources="both",
    ro_kwargs=dict(podtc_dir="/path/to/RO_Data", roi_mode="tangent_point"),
    igs_kwargs=dict(
        stations=["TRO1", "WUTH"], cache_dir="/path/to/RINEX_Cache",
        local_obs_by_station={
            "TRO1": "/path/to/RINEX_Cache/TRO100NOR_S_..._MO.crx",
            "WUTH": "/path/to/RINEX_Cache/WUTH00NOR_R_..._MO.crx",
        },
        local_nav="/path/to/RINEX_Cache/BRDC00IGS_R_..._MN.rnx",
        local_dcb="/path/to/RINEX_Cache/CAS0OPSRAP_..._DCB.BIA",
    ),
    obs_sigma=3.0,   # this project's tuned value for real TEC data at this scale (Section 3)

    # --- real ISR input data (Section 8.3/9) ---
    isr_file_path="/path/to/your_isr_file.nc",
    isr_pca_basis_path="/path/to/isr_pca_basis.nc",   # Section 9.2 -- build once, reuse

    # --- all 4 default styles, production scale (Section 3) ---
    n_ensemble=2000,
    styles=default_styles(),
    hyper_params_by_style=default_hyper_params_by_style(),
    analysis_rng_seed=777,   # reproducible across the two conditions below (Section 3)
)

# Run 1: no boost -- also builds and saves the real base ensemble.
cfg_noboost = replace(cfg_base, output_dir=OUT / "no_boost", label="no_boost")
result_noboost = run_package(cfg_noboost)

# Run 2: boosted -- reuses Run 1's saved base ensemble (no second IRI2020
# build) and applies diagonal boosting on top of it.
base_edp_path = OUT / "no_boost" / "no_boost_edp_samples.nc"
cfg_boosted = replace(
    cfg_base, output_dir=OUT / "boosted", label="boosted",
    edp_samples_path=base_edp_path,
    diagonal_boost_amplitude=0.5, diagonal_boost_log_space=True,
    diagonal_boost_rng_seed=888,
    # diagonal_boost_horizontal_scale_km left unset -> picks up the
    # current default (500km, see Section 3) automatically.
)
result_boosted = run_package(cfg_boosted)

for label, result in (("no_boost", result_noboost), ("boosted", result_boosted)):
    print(f"=== {label} ===")
    for style, (cycle_result, n_state, wall_time) in result.results_by_style.items():
        outcome = cycle_result.batch_outcomes[-1]
        print(f"  {style}: n_state={n_state}, wall_time={wall_time:.1f}s, "
              f"converged={outcome.diagnostics.converged}, "
              f"RMSE analysis={outcome.rmse_reduction.rmse_analysis:.3f} TECU")
```

Run with `source init_iri2020_env.sh && /opt/anaconda3/bin/python3
this_script.py` (Section 1). Expect ~100 minutes for the no-boost run
and ~85 minutes for the boosted run at this scale (Section 6) -- do a
smoke test at small `n_ensemble` first if this is your first run against
new data or a new region.
