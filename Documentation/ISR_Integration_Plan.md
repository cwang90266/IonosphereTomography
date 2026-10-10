# Plan: Incoherent Scatter Radar (ISR) Integration

## 1. Purpose and scope

Add two new uses of ISR (incoherent scatter radar) electron-density data to
the existing data assimilation system (`IRI_Sample_Inputs` + `EDPSamples` +
`Parameterization` + `Ensemble_Kalman_Engine`, wired together by
`Assimilation_Cycle`):

1. **Independent validation**: compare the assimilation's analysis EDP, at
   the center of the modeling region, against the ISR-measured EDP at the
   same time. The real purpose of this tool is broader than one site: it's
   how the project validates the *general* RO+IGS EDP-retrieval approach
   against ground truth, for a region of interest centered on whichever
   ISR station is available there. The real Tromsø data on hand now is the
   first test case, not the only one — **both new pieces of work in this
   plan must be station-agnostic**, not hardcoded to Tromsø, since the
   package will be pointed at other ISR stations' regions in the future.
2. **A new parameterization style**: instead of deriving PCA basis vectors
   from an IRI2020-driven climatological ensemble (today's `PCA_1D`/
   `PCA_3D` styles), derive them from real ISR-measured profiles collected
   over an extended period, then use that basis both to represent the prior
   ensemble and as the observation operator's EDP/PCA-to-TEC map. The
   motivating hypothesis: real ISR profiles may span a *wider* range of EDP
   shapes than IRI2020's climatology, so an ISR-derived basis could capture
   ensemble spread/structure the climatological approach misses. Confirmed
   (2026-10-06): modeled directly after `PCA_1D_10ex` — 1-D (vertical-only,
   Section 3), log10-space (the `_10ex` convention already used throughout
   this codebase) — with little to no change expected to the system's
   overall structure.

Per the user's stated priority, the implementation sequence is: (A) build
the ISR-based PCA parameterization style, including diagonal-boost
compatibility; (B) evaluate it against the three established styles
(`ANCHOR`, `PCA_1D_10ex`, `PCA_3D_10ex`), with and without boosting; (C)
build the ISR-vs-analysis comparison tooling. Sections 4.1-4.3 below follow
that order, though (C) has no actual dependency on (A)/(B) and could start
in parallel (see Section 7). This document is for review before any
implementation starts.

## 2. Inputs on hand (checked directly, not assumed)

- **New ISR data file**:
  `/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc`
  — Tromsø UHF ISR, adaptively median-filtered, year 2025 (67 distinct days,
  Feb 26 - Dec 20, gaps between campaigns), 4-minute native cadence, 8284
  time samples, **39 fixed altitude gates from 82.2 to 646.7 km** (uneven
  spacing, denser at low altitude), electron density `Ne` in m⁻³
  (2.2e8-1.3e12 range), **no NaNs** (the "nonan" processing already drops
  profiles with >=2 invalid gates and fills single-gate gaps from the
  nearest valid gate — see the file's own `profile_completeness_policy`
  attribute). Station: 69.5864N, 19.2272E. Confirmed (2026-10-06): this
  file is the **raw input** to this plan's pipeline (i.e. Section 4.1.0's
  IRI2020 topside extension still needs to run against it; it has not
  been extended yet). Also confirmed: besides the altitude *ceiling*
  (646.7 km vs. the production grid's 900 km), the native ISR gates are
  on an **irregular** grid, different from the regular-spacing grid used
  for IRI2020 runs — for now, ISR profiles are resampled onto the regular
  IRI2020 grid (Section 4.1.1); using ISR's native grid instead is a
  possible future direction, not built now (Section 5).
- **Station identity is already in the file, not just known out-of-band**:
  `TROISR2025_nonan.nc`'s global attributes include `station_name`,
  `station_latitude`, `station_longitude`. This is the mechanism this plan
  uses for station-agnostic code (Section 4.1/4.3): read the station's
  identity from whichever ISR file is given, rather than hardcoding
  Tromsø's coordinates anywhere in new code.
- **The existing RO+IGS real-data test window** (`2025-11-18 10:00-11:00
  UTC`, used throughout the `Assimilation_Cycle` validation work) falls
  **inside** this ISR file's coverage — confirmed directly: 20 ISR profiles
  fall in a 2-hour window centered on it. This means use case (A) can be
  tested against a cycle run the project has already validated end-to-end,
  with no new real-data acquisition needed.
- **`extend_isr_edp_with_iri2020.py`** (repo root): an already-built,
  already-validated script that extends ISR profiles above their ~650 km
  ceiling to 900 km, by fitting each profile's `log10(Ne)` (at altitudes
  >=200 km) as a combination of the leading 3 left-singular-vectors of a
  2000-member IRI2020 library, then applying that fit to the library's
  650-900 km rows and splicing it onto the measured profile with a
  smoothstep-tapered join correction. It currently defaults to reading
  `.../TROISR2025.nc` (the pre-"nonan" file) and writing
  `TROISR2025_extended.nc`. This script is the natural source of the >650 km
  portion of any ISR-based altitude profile used below — see the dependency
  called out in Section 4.1. It also currently hardcodes `STATION_LAT`/
  `STATION_LON` as module constants (69.583/19.21) rather than reading them
  from the input file — given the station-agnostic requirement above, this
  plan proposes a small, additive fix (read `station_latitude`/
  `station_longitude` from the input file's attrs, fall back to the
  existing hardcoded values only if absent) as part of Section 4.1.0,
  rather than a separate future cleanup.
- **`validation_preparation/isr_profile_validation.py`**: an existing
  `validate_isr_profile()` that compares one model Ne column against *raw*
  ISR files (Madrigal-format, read via `netCDF4` directly, scanning a
  directory for a time window, station-matched by `kindat`/lat-lon). This
  is a different data path from the preprocessed `TROISR2025_nonan.nc` (no
  adaptive filtering, no extension) — useful prior art for the RMSE-by-
  altitude-bin methodology, but not something use case (A)'s tooling should
  call directly (see Section 4.3's reasoning for reading the preprocessed
  file instead).
- **`Assimilation_Cycle/output.py`** already has almost exactly the
  machinery use case (A) needs: `resolve_edp_query_point_and_profiles()`
  interpolates the forecast/analysis ensemble to an explicit `(lat, lon)`
  via `EDPSamples`'s own barycentric mesh interpolation, and
  `plot_edp_profile_comparison()` overlays the result against the Abel-
  retrieved profile from an RO occultation. Passing the ISR site's `(lat,
  lon)` explicitly instead of letting it derive from an RO entry is already
  a documented, supported call pattern (the docstring explicitly flags this
  as "the same function the docx's planned future external-validation use
  reuses" — this *is* that future use).
- **`Parameterization.Parameterized_EDPSamples`** already supports
  supplying a pre-fit PCA basis instead of fitting one from the ensemble
  it's given: `hyper_params={'PCA': ..., 'PCA_mean': ...}` bypasses the
  `retaining_threshold`-driven fit entirely (`Parameterization.py` lines
  ~1334-1351). This means the ISR-PCA style needs **no new math** in
  `Parameterization.py` — `PCA2EDP_1D`/`EDP2PCA_1D`/`PCA2EDP_1D_map` are
  already basis-agnostic. The real new work is building the ISR-sourced
  basis itself and deciding how it's labeled/selected (Section 4.1.2).
- **`diagonal_boost.py`** operates on the raw IRI2020-drawn density
  ensemble, strictly *before* parameterization — it has no dependency on
  which style/basis is used downstream. Diagonal boosting should therefore
  need no code changes to support the new style, only re-validation of the
  amplitude recommendation (Section 4.2).

## 3. A key structural finding: ISR gives a 1-D (vertical-only) basis

The ISR instrument is at one fixed site. `TROISR2025_nonan.nc` has **no
horizontal/geolocation dimension at all** — every profile is a single
altitude column. `get_PCA`'s 3-D variant (`PCA_3D`/`PCA_3D_10ex`, basis
shape `(nAlt, nGeo, nPCA)`) jointly captures vertical *and* horizontal
structure, fit today from an IRI2020 ensemble that *does* vary over a
horizontal grid. ISR data cannot inform a joint vertical-horizontal basis —
there is nothing to compute horizontal correlation from.

**Confirmed (2026-10-06): build only a 1-D (vertical-shape) ISR-PCA
style**, analogous to today's `PCA_1D`/`PCA_1D_10ex` (basis shape `(nAlt,
nPCA)`, applied independently at every horizontal grid point), log10-space
(the `_10ex` convention). This is a real scope reduction from "parallel to
PCA_3D_10ex" in the comparison set of Section 4.2 — the new style is
compared *against* `PCA_3D_10ex` (and `ANCHOR`/`PCA_1D_10ex`), not built as
a 3-D analog of it.

## 4. Work items

### 4.1 ISR-derived PCA basis + parameterization style

**4.1.0 Prerequisite: extend ISR profiles to the production altitude grid.**
The production altitude grid is 90-900 km / 10 km (82 levels) by default
(`CycleConfig.altitude_grid`); native ISR gates only reach 646.7 km. Before
any PCA basis can be fit on a grid consistent with the rest of the system,
each ISR profile needs:
  - Interpolation (in `log10(Ne)` space, matching every other log-space
    operation in this codebase) from the 39 native gates onto the
    production grid's sub-650 km levels.
  - The already-built IRI2020-library extension (`extend_isr_edp_with_iri2020.py`)
    for the 650-900 km levels.

This means Work Item 1 depends on **re-running the extension script against
the new `TROISR2025_nonan.nc`** (today it defaults to the older,
non-"nonan" file) to produce an extended, "nonan" analog of
`TROISR2025_extended.nc`. Confirmed (2026-10-06): this rerun is in scope
for Work Item 1 (`TROISR2025_nonan.nc` is confirmed raw input, not
pre-extended) and should include the station-agnostic fix from Section 2
(read `station_latitude`/`station_longitude` from the input file instead
of the script's hardcoded constants) at the same time, since it's the same
file being touched. `EXT_ALT_STEP` stays 10 km for now, matching the
production grid — the user separately noted that a coarser topside
(650-900 km) resolution might be worth revisiting in the future, recorded
as a non-goal for this pass (Section 5), not acted on now.

**4.1.1 New module: `isr_pca_basis.py`.** Proposed home:
`Assimilation_Cycle/isr_pca_basis.py`, alongside `diagonal_boost.py`/
`iri_selection.py` — utilities that prepare inputs for a cycle, not generic
parameterization math (keeps `Parameterization.py` untouched, consistent
with this project's standing "no new logic in survivor modules unless
there's no reasonable alternative" policy from the observation-preparation
refactor). Responsibilities:
  - Load the extended ISR netCDF (Section 4.1.0's output); resample onto
    `cfg.altitude_grid` (log10-linear interpolation below the native
    ceiling, direct read above it from the IRI-extended rows). Confirmed
    (2026-10-06): resample onto the regular IRI2020 grid for now, even
    though it's a different (and irregular) grid from ISR's own native
    gates — using ISR's native grid is a possible future direction, not
    built now (Section 5).
  - Build the `(n_height, n_profile)` log10-density matrix across however
    many ISR time samples are selected (see Open Question 3 on whether to
    use the full-year 8284 profiles or a subset) and call
    `Parameterization.get_PCA(flat, retaining_threshold)` directly — same
    function the existing PCA styles already use, so the truncation
    semantics (`retaining_threshold`, `PCADiagnostics`) are identical and
    comparable.
  - Return `(PCA, PCA_mean, diagnostics)` in exactly the shape
    `Parameterized_EDPSamples` expects for a pre-fit `PCA_1D`-family basis,
    plus a save/load pair (netCDF or `.npz`) so the basis is a reusable,
    precomputed artifact — matching the project's established
    precomputed-file-mode convention (`edp_samples_path`,
    `parameterized_edp_samples_path`, etc.).

**4.1.2 Registering the new style.** Since `Parameterized_EDPSamples`
already accepts a pre-fit `PCA`/`PCA_mean` for `PCA_1D`/`PCA_1D_10ex`, the
new style is mechanically identical to `PCA_1D_10ex` — the only difference
is *where the basis came from*. Consistent with the user's expectation of
little to no structural change, the plan is: **register a new style name**
(proposed `'PCA_1D_10ex_ISR'`, open to a different name — Open Question 5)
in `Parameterization_Style`'s `Literal` and `_init_hyper_params`/
`_build_spec`, whose behavior is byte-for-byte identical to
`'PCA_1D_10ex'` (always requires a pre-fit `PCA`, never fits one from the
ensemble it's given) — a small, mechanical registry addition, not new
math. The alternative (reusing the `'PCA_1D_10ex'` name as-is with an
ISR-sourced `PCA` in `hyper_params`) would work identically at the single-
cycle level but collides with `CycleConfig.styles`/`hyper_params_by_style`'s
one-entry-per-style-name sweep pattern, making Section 4.2's side-by-side
comparison ambiguous (two different bases under one label) — a distinct
name avoids that for the cost of one registry entry.

**4.1.3 `CycleConfig` wiring.** Add `isr_pca_basis_path` (precomputed-file
mode, same pattern as `edp_samples_path`), and ensure
`load_or_build_parameterized_edp_samples` can load that file and populate
`hyper_params['PCA']`/`['PCA_mean']` for the new style before encoding —
small, additive change to `ensemble_init.py`, no change to its existing
styles' code paths. Confirmed (2026-10-06): `default_styles()` becomes a
**4-style set** — `ANCHOR`, `PCA_1D_10ex`, `PCA_1D_10ex_ISR`,
`PCA_3D_10ex` (today's default is the first/last two plus `PCA_1D_10ex`,
i.e. this adds the new style as the fourth default, not a replacement for
any existing one) — this is the set both Section 4.2's comparison and
Section 4.3's per-style/cross-style ISR plots are run against.

**4.1.4 Diagonal boosting.** No code change expected (Section 2's finding)
— but add a wiring test confirming `diagonal_boost_log_space=True` +
`'PCA_1D_10ex_ISR'` round-trips correctly (same assertion style as the
existing `diagonal_boost.py` test suite), since this combination has never
been exercised.

**4.1.5 Tests.** Unit tests for `isr_pca_basis.py` (resampled grid shape,
PCA orthonormality, diagnostics sanity, save/load round trip) on a small
synthetic ISR-like fixture; one real-data smoke test building a tiny cycle
with `'PCA_1D_10ex_ISR'` against the actual Tromsø file, following this
project's established "unit tests first, then one real-data smoke run"
practice.

### 4.2 Evaluate the new style against the other 3 default styles

- Reuse `style_sweep.py`/`package_run.py` as-is (they already run several
  styles against shared observations and a shared IRI draw) — run the full
  4-style default set (`ANCHOR`, `PCA_1D_10ex`, `PCA_1D_10ex_ISR`,
  `PCA_3D_10ex`, per Section 4.1.3) for the real 2025-11-18 Tromsø RO+IGS
  window, which Section 2 confirmed has matching real ISR coverage.
- Reuse `metrics.py` (RMSE O-F/O-A by `obs_type`, rank histograms,
  effective ensemble rank) and the **pooled/retrospective RMSE check**
  this project adopted after discovering the per-batch metric alone can't
  distinguish real improvement from a method gaming its own most recent
  batch's data (see the inter-batch-inflation false-positive finding) —
  any accuracy claim about the new style should pass through that same
  pooled check, not just per-batch numbers.
- Follow the multi-seed/multi-split validation discipline this project
  settled on after repeated single-split reversals (ANCHOR-vs-PCA_3D_10ex,
  the IES evaluation, the diagonal-boost amplitude sweep) — no new
  single-run conclusion should be reported as decisive.
- **New diagnostic, independent of the EnKF**: a basis-quality comparison
  that projects a held-out set of real IRI2020-generated profiles (and,
  symmetrically, held-out real ISR profiles) onto (i) the IRI-fitted PCA
  basis and (ii) the ISR-fitted PCA basis, and reports reconstruction error
  for each — this directly tests the user's stated hypothesis ("ISR data
  provides a wider range of EDPs than the climatological model") on its
  own terms, before any filtering/assimilation is involved. Natural home:
  alongside `isr_pca_basis.py`, since it only needs
  `Parameterization.get_PCA`/`PCA2EDP_1D`/`EDP2PCA_1D`, no cycle machinery.
- Repeat the diagonal-boost amplitude/`log_space` sweep methodology already
  validated for `PCA_3D_10ex` (Section 17 of
  `Assimilation_Cycle_Integration_Plan.md`) for the new style, same
  held-out discipline — do not assume the same amplitude (~0.5) transfers,
  since the basis and its effective rank differ.
- Extend the existing style-comparison table (`n_state`, wall time,
  converged-batch count, final/pooled RMSE — as already produced for
  `ANCHOR`/`PCA_3D_10ex`/`PCA_1D_10ex` in the full-scale run) with the new
  style's row.

### 4.3 ISR-vs-analysis comparison tooling

Confirmed (2026-10-06): the direct comparison itself (interpolating the
analysis to the ISR site and differencing against the ISR profile) has no
dependency on Section 4.1/4.2's new parameterization style — it works
against any cycle result today — so it can be built and run right away,
in parallel with (or ahead of) Section 4.1. What's still missing is the
**visualization tooling** itself (the plots and pooled summary below don't
exist yet); the underlying interpolation machinery
(`resolve_edp_query_point_and_profiles`) already does.

- New module, proposed `Assimilation_Cycle/isr_comparison.py` (keeps
  `output.py` focused on plotting primitives; this module owns ISR-specific
  I/O + time-matching, analogous to how `observation_stream.py` owns
  RO/IGS-specific assembly).
- **Why read the preprocessed `TROISR2025_nonan.nc` directly, not
  `validate_isr_profile()`'s raw-file path**: the preprocessed file is
  already time-sorted, site-fixed, gap-filled, and (once Section 4.1.0's
  extension is applied) altitude-complete to 900 km — a direct
  nearest-time lookup is simpler and faster than
  `validate_isr_profile()`'s raw-Madrigal-directory scan, which exists to
  handle multiple candidate sites/files and doesn't know about this
  project's grid/interpolation conventions. `validate_isr_profile()`'s
  RMSE-by-altitude-bin *methodology* is still worth reusing conceptually
  (binned RMSE vs. altitude, qualitative profile overlay).
- Core function: for a given cycle result (one or more `CycleBatch`/
  `BatchOutcome` pairs) and an ISR netCDF file, with the station's
  `(lat, lon, name)` **read from that file's attrs** (Section 2) rather
  than passed in or hardcoded — the same function works unmodified for a
  future non-Tromsø region as long as its ISR file carries the same
  `station_name`/`station_latitude`/`station_longitude` attributes
  `TROISR2025_nonan.nc` does:
  1. Resolve each batch's ISR match — **confirmed (2026-10-06): the nearest
     single ISR profile to the batch's midpoint** (not an average over the
     batch window).
  2. Call `resolve_edp_query_point_and_profiles(edp_samples, entry=None,
     decoded_forecast, decoded_analysis, lat=station_lat, lon=station_lon)`
     — already supports explicit `lat`/`lon`, bypassing its RO-derived
     default — to get the forecast/analysis ensemble profile at the ISR
     site.
  3. **Plot type A — per-batch nearest-point overlay**: a qualitative
     overlay plot extending `plot_edp_profile_comparison`'s visual pattern
     (forecast/analysis mean +/- std vs. a measured curve) with the
     nearest ISR profile (step 1) in place of the Abel-retrieved one —
     distinguishing the native-measured altitude range from the
     IRI-extended range visually (same convention
     `extend_isr_edp_with_iri2020.py`'s own plots already use: solid vs. a
     distinct marker/line style). One per batch.
  4. **Plot type B — cycle-wide ISR range, confirmed new requirement
     (2026-10-06)**: the user wants this presented two ways, not just the
     single-nearest-point comparison above. For the *entire* time span
     covered by all RO+IGS observations assimilated over the cycle (i.e.
     the cycle's full start-to-end window, not one batch), collect every
     ISR profile whose timestamp falls inside that window and plot their
     **range** (e.g. a shaded min-max or percentile band across all of
     them, at each altitude) as context for how much the real EDP
     naturally varies over the cycle's duration. Overlay every batch's
     analysis profile (mean, one line per batch, consistent coloring with
     this project's existing per-batch/per-style conventions) on top of
     that band, so it's visible whether the sequence of analyses tracks
     within the real observed variability or departs from it. One plot
     per cycle (not per batch).
  5. Produce a **pooled**, RMSE-by-altitude-bin quantitative summary across
     every batch in the cycle (nearest-profile matches from step 1, not a
     single-time snapshot) — the same pooled-rather-than-per-batch
     discipline from Section 4.2, since a single qualitative plot cannot
     support a general accuracy claim. This is the quantitative companion
     to Plot A; Plot B is deliberately qualitative/visual (a range isn't a
     point estimate to score against).

- **Per-style and cross-style, confirmed new requirement (2026-10-06):**
  every plot above (A, B, and the pooled RMSE summary) is run **per
  individual style first**, then **once more as a cross-style comparison**
  across the same 4-style default set introduced in Section 4.1.3
  (`ANCHOR`, `PCA_1D_10ex`, `PCA_1D_10ex_ISR`, `PCA_3D_10ex`) — mirrors
  this project's already-established per-style-plot-plus-cross-style-plot
  convention (`plot_tec_edp_profile_comparison` vs.
  `plot_cross_style_tec_edp_comparison`, Section 23 of
  `Assimilation_Cycle_Integration_Plan.md`). Concretely:
  - **Plot A, per style**: one nearest-point overlay per batch per style
    (today's single-style `plot_edp_profile_comparison` pattern, ISR
    profile in place of Abel).
  - **Plot A, cross-style**: one plot per batch overlaying all 4 styles'
    forecast/analysis mean against the same nearest ISR profile — new,
    analogous to `plot_cross_style_tec_edp_comparison`'s overlay-by-style
    convention (consistent per-style color, solid=forecast/dashed=analysis).
  - **Plot B, per style**: one cycle-wide ISR-range plot per style, with
    that style's own per-batch analysis lines over the band.
  - **Plot B, cross-style**: one cycle-wide ISR-range plot with **all 4
    styles'** per-batch analysis lines (color-coded by style, not by
    batch) over the same band — the most direct visual answer to "which
    style's analysis best tracks the real ISR-observed range over the
    whole cycle."
  - **Pooled RMSE-by-altitude-bin, per style and cross-style**: the
    per-style numeric table extended with one additional row/column set
    comparing all 4 styles' pooled RMSE side by side (extends the
    existing style-comparison table from Section 4.2, now scored against
    ISR directly rather than only RO/IGS O-A).
  This requires `package_run.py`'s existing shared-observations/shared-IRI-
  draw sweep pattern to also share the ISR comparison inputs (nearest-
  profile matches, cycle-wide ISR range) across all 4 styles' runs, so the
  cross-style plots compare against literally the same ISR data each
  style's per-style plot used — not independently re-resolved per style.
- **Output location, confirmed (2026-10-06):** under
  `Runs/Tomography_Test/Data_Assimilation_Cycle/`, but in **dedicated
  subfolders**, not dumped flat into that directory — proposed
  `Data_Assimilation_Cycle/ISR_Comparison/<cycle label>/` for this work
  item's plots/tables, and (Section 4.1) `Data_Assimilation_Cycle/ISR_PCA_Basis/`
  for the new style's basis artifacts — matching the existing convention
  of per-run subfolders (`step11_both/`, `step15_n2000_10km/`, etc.).

## 5. Non-goals (for this pass)

- No 3-D (joint vertical+horizontal) ISR-PCA basis — Section 3's finding
  (single-site data carries no horizontal information).
- No attempt to use ISR data as a *direct assimilated observation* (an
  ISR-as-observation EnKF update, as opposed to ISR-as-validation /
  ISR-as-basis) — the user's request is specifically validation and
  parameterization, not a third observation type alongside RO/IGS. Could
  be a natural future extension (the `ISR_UKF`/`isr_aid_covariance.py`
  code under `Ionosphere_Tomography_Inverter`/`Austin_Demo_Code` already
  explored something adjacent to this, but those modules are explicitly
  out of the five-module survivor set per the observation-preparation
  refactor plan and are not reused here).
- No change to the observation-preparation RO/IGS pipeline, the EnKF
  analysis engine's core math, or the ANCHOR/`density_10ex` styles.
- No re-opening of the diagonal-boost amplitude recommendation for the
  three existing styles — Section 4.2 only extends the sweep to the new
  style.
- No use of ISR's native (irregular) altitude grid in place of the regular
  IRI2020 grid — profiles are resampled onto the regular grid for now
  (Section 4.1.1); the user flagged native-grid use as a possible future
  direction, not built this pass.
- No change to `extend_isr_edp_with_iri2020.py`'s topside (650-900 km)
  resolution (`EXT_ALT_STEP=10 km`, matching production) — the user noted
  a coarser topside resolution might be worth revisiting later; this plan
  only generalizes the script's station handling (Section 4.1.0), not its
  altitude-step choice.

## 6. Decisions (resolved 2026-10-06)

- **Scope of the new style**: 1-D vertical-only, modeled directly on
  `PCA_1D_10ex` (log10-space). No 3-D/hybrid variant. (Was Open Question 1.)
- **Section 4.1.0's extension rerun**: in scope for this plan;
  `TROISR2025_nonan.nc` is confirmed raw input. (Was Open Question 2.)
- **Station-agnostic design**: both new pieces of work must work for any
  ISR station's region, not just Tromsø — resolved via reading station
  identity from each ISR file's own attrs rather than hardcoding
  coordinates anywhere (Section 2). This also means Section 4.1.0 picks up
  a small generalization fix to `extend_isr_edp_with_iri2020.py` (read
  station lat/lon from the input file, Section 2).
- **Altitude grid**: resample ISR profiles onto the existing regular
  IRI2020 grid for now, not ISR's native (irregular) grid — a future
  direction, not built this pass (Section 5). (Was part of Open Question
  3's framing; the "how much resampling" question is resolved, but how
  *many profiles* to use for the basis fit is still open below.)
- **Section 4.3 has no blocking dependency on Section 4.1/4.2** and its
  core numeric comparison is straightforward given the existing
  `resolve_edp_query_point_and_profiles` machinery — but the
  visualization/summary tooling itself still needs to be built (it
  doesn't already exist). Sequencing updated accordingly in Section 7.

### Resolved (2026-10-06, second review round)

1. **How much of the ISR record to use for the basis fit**: all available
   profiles (all 8284, all 67 days) — no subsetting.
2. **Time-matching / presentation policy** for Section 4.3: not just a
   single-point question — the user wants *two* complementary views, both
   built: (a) nearest single ISR profile to each batch's midpoint
   (Plot type A, Section 4.3), and (b) the *range* of all ISR profiles
   across the entire cycle's full RO+IGS observation window, overlaid with
   every batch's analysis profile (Plot type B, Section 4.3) — see that
   section for the full description.
3. **New style naming**: `'PCA_1D_10ex_ISR'` confirmed.
4. **Output location**: `Data_Assimilation_Cycle/`, confirmed, but **in
   dedicated subfolders** (`ISR_Comparison/<cycle label>/`,
   `ISR_PCA_Basis/`) rather than flat in the parent directory — see
   Section 4.3's output-location bullet.

No open questions remain. Ready to proceed to implementation per Section 7's
sequencing, pending final go-ahead.

## 7. Suggested sequencing

1. Resolve the four still-open items in Section 6 (or accept the stated
   defaults).
2. Section 4.3: ISR-vs-analysis comparison/visualization tooling — no
   dependency on anything else in this plan, can start immediately against
   the existing real 2025-11-18 Tromsø RO+IGS cycle result.
3. Section 4.1.0: regenerate the extended ISR file against
   `TROISR2025_nonan.nc`, including the station-agnostic fix to
   `extend_isr_edp_with_iri2020.py`.
4. Section 4.1.1-4.1.3: `isr_pca_basis.py` + style registration + `CycleConfig`
   wiring, unit-tested on synthetic data.
5. Section 4.1.4-4.1.5: diagonal-boost wiring check + real-data smoke test
   (confirms the new style runs end-to-end before any comparison work).
6. Section 4.2: style comparison (accuracy, cost, basis-quality diagnostic,
   boosting re-sweep) on the real 2025-11-18 Tromsø window — can reuse
   Section 4.3's tooling (built in step 2) to additionally compare each
   style's analysis against ISR directly, not just RO/IGS O-A metrics.
