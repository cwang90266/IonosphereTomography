# Verification & Debugging Plan: IRI_Sample_Inputs, EDPSamples, Parameterization

Author: scoping pass by Claude, 2026-09-16
Status: **plan only — no fixes applied yet**

**Framing note (added after author feedback):** `IRI_Sample_Inputs` and
`EDPSamples` have had limited testing so far; `Parameterization` was just
written. All three should be read as a way of conveying the intended design,
not as near-final implementations. That changes the calculus for
`Parameterization` in particular: since nothing downstream yet depends on its
current call signatures, this is the cheapest moment to fix its structure,
not just its bugs — see §7 for structural recommendations alongside the
verification plan in §1–6.

## 1. Purpose and scope

This document scopes module-level verification and debugging for the three
building blocks of the ionosphere data-assimilation pipeline:

| Module | Path | Role |
|---|---|---|
| `IRI_Sample_Inputs` | `IRI_Sample_Inputs/IRI_Sample_inputs.py` | Downloads/parses IRI2020 driving indices (ap, F10.7, IG12, Rz12) and draws quantile/random samples of them around a target date-time. |
| `EDPSamples` | `EDPSamples/edp_samples.py` | `xarray.Dataset` subclass: builds altitude/geolocation/mesh grids, drives the IRI2020 Fortran executable to fill electron-density-profile (EDP) samples, and provides forward operators (ray integration, TEC) and I/O (NetCDF). |
| `Parameterization` | `Parameterization/Parameterization.py` | Maps EDPs to/from parameter vectors (`get_density`, `get_parameter`) and computes Jacobians (`get_Jacobian`) for use in (extended) Kalman filter updates. |

The three modules form a pipeline: `IRI_Sample_Inputs` → `EDPSamples` (via
`IRI2020_EDP`) → `Parameterization` (via `Parameterized_EDPSamples`). Each
module is tested in isolation first (mocking its upstream dependency), then a
thin end-to-end smoke test chains them together.

This plan only scopes the work. No code changes are made yet — the intent is
to agree on scope and priority before starting.

## 2. What I found while reading the code (informs prioritization)

Rather than propose generic "write unit tests" boilerplate, I read all three
files end-to-end first. That surfaced a number of concrete, reproducible
defects. These aren't the whole plan, but they should drive **prioritization**:
the parameterization module (used inside the Kalman filter) has the most
correctness-critical bugs, so it should be tested/fixed first even though it's
the smallest file.

### 2.1 `Parameterization.py` — highest priority (feeds the filter directly)

- `EDP_Parameterization.__init__`, `ANCHOR` branch: when `hyper_params` is
  supplied, the code sets `cls.bhyper_params = hyper_params` (typo) instead of
  `cls.hyper_params`. Any custom ANCHOR bounds are silently dropped; the
  object has no usable `hyper_params` and later calls raise `AttributeError`.
- `EDP_Parameterization.__init__`, `PCA_*` branches: `if not 'PCA' in
  hyper_params.keys:` is missing `()` — `dict.keys` is a method object, so
  `in` raises `TypeError` before the intended check ever runs.
- `Parameterized_EDPSamples.__init__` calls
  `Parameterization.get_parameter(edps, hyper_params=hyper_params)` in every
  branch, but `EDP_Parameterization.get_parameter(self, density, alt=None)`
  has no `hyper_params` keyword — this raises `TypeError` for **every**
  parameterization style, i.e. `Parameterized_EDPSamples` cannot currently be
  constructed at all.
- `EDP_Parameterization.get_parameter`, `ANCHOR` case calls
  `_fit_iri_params(density, alt, param_bounds=self.hyper_params)`, but
  `_fit_iri_params`'s keyword is `param_bound` (singular) — another
  `TypeError` once the previous bug is fixed.
- `_ne_profile_derivatives`: `params_shape = params_lin.shape` is a tuple;
  later `params_shape[0] = n_alt` attempts item assignment on a tuple →
  `TypeError`. This is on the `partial=False` (density-only) return path used
  by `get_density('ANCHOR', ...)`, so that call path is currently broken.
  Also `ndim = params_lin.ndim` is an int, but the guard clause references
  `ndim[0]` in its error message — a second crash if the dimension check ever
  fires.
- `get_PCA`: `nSize, n_sample = edps.shape()` — `.shape` is a tuple attribute,
  not callable → `TypeError`. Same `.shape()` mistake recurs in
  `Parameterized_EDPSamples.__init__` (`PCA_3D`/`PCA_3D_10ex` branches:
  `nD, nPCA = PCA.shape()`), and in `PCA2EDP_3D_map`
  (`PCA = PCA.shape([...])` should be `PCA.reshape([...])`).
- `get_PCA`: retains components where `singularval/singularval[-1] <=
  retaining_threshold`. Since `eigh` returns ascending eigenvalues,
  `singularval[-1]` is the *largest* singular value, so this keeps the
  **low-energy** components below the threshold rather than the dominant ones
  — inverted from the usual "retain the top `k` modes explaining X% of
  variance" intent. Also `eigenvect[:, idx]` with `idx` a tuple from
  `np.where` adds a spurious axis. Needs a decision on intended semantics
  before it can be tested meaningfully.
- `density_10ex`, `log10_param`, `Jacobian_density_10ex`, `EDP2PCA_1D`,
  `EDP2PCA_3D`: all mutate their input array in place via
  `array[idx] = floor_value` before returning. If callers reuse the original
  array (e.g. the raw EDP sample) afterward, its values are silently clamped.
  Also, `EDP2PCA_1D`/`EDP2PCA_3D`'s clamp uses `10*minlog10Density` (linear
  multiply) where `density_10ex`/`log10_param` correctly use
  `10**minlog10Density` (power) — an inconsistent floor between the two
  "10ex" code paths.
- `tuple(str, np.array)` as the return annotation on `get_Jacobian` looks like
  a call to the `tuple` builtin with two args (which would itself raise
  `TypeError`), but the module has `from __future__ import annotations`, so
  annotations are stored as unevaluated strings — this one is cosmetic, not a
  runtime bug. Worth fixing for clarity but not blocking.

**Net effect:** as written, `Parameterized_EDPSamples` cannot be instantiated
for *any* style, and `EDP_Parameterization.get_density/get_parameter` for
`'ANCHOR'` and PCA styles cannot run end-to-end. The `density_10ex` /
`log10_param` scalar path is closer to working but has the in-place-mutation
issue. This module needs debugging before it can be meaningfully unit tested;
see §5 for the proposed fix-then-test sequence.

### 2.2 `EDPSamples/edp_samples.py`

- **Coordinate-order inconsistency (systemic, high priority).** The
  `geolocation` property docstring and the `VAR_GEOLOCATION` NetCDF attrs say
  "latitude (column 0), longitude (column 1)". `subset_region` and
  `subset_union_triangles` explicitly comment `col 0 = lon, col 1 = lat`, and
  the geometry generators are inconsistent with each other:
  `genRectangularArea`, `genPolarArea`, `genGlobalArea`,
  `generate_occultation_mesh` all build vertices as `(lon, lat)`, while
  `genLineOfSight` builds `(lat, lon)`. The `"Point"` branch of `__init__`
  stores `[[Lon, Lat]]` (lon-first). This ambiguity can silently swap
  latitude and longitude depending on which `geo_type` was used, which would
  corrupt every downstream geometric computation (mesh containment, ray
  integration, plotting) without raising an error. This needs a single
  documented convention and a regression test that pins it per generator.
- `from_xarray`, `"Point"` case: reconstructs with
  `Lat=ds.attrs["Lon"]` — should be `ds.attrs["Lat"]`. A round-tripped
  `"Point"` dataset silently gets latitude replaced by longitude.
- `from_xarray`'s `match` has no case for `"Global"` prior to the duplicate
  `"Occultation"` case appearing twice (lines ~1507 and ~1545 — the second is
  unreachable); more importantly there is no `"LOS"`-producing case matching
  what `__init__`'s `geo_type` literal actually allows (`Literal["Point",
  "Rectangle", "Polar", "Occultation", "Global"]` — `"LOS"` isn't even a valid
  `geo_type`, so that whole branch in `from_xarray` is dead code, while
  `genLineOfSight` itself is a `@staticmethod` with no `geo_type` wired to
  it). `interp()` and `plot_geolocation()` similarly have no `"Global"` case
  in their `match` statements — calling either on a `"Global"`-type
  `EDPSamples` silently returns `None` / does nothing.
- `IRIPath = "~/Desktop/tomography_project/iri2020_new/src/iri2020/"` is
  hardcoded inside `get_IRI2020_EDP`, and `~` is not expanded (shell tilde
  expansion doesn't happen inside Python string formatting used to build the
  `ln -s` command, though it does happen to work here only because the string
  is passed to `subprocess.run(..., shell=True)`, which *does* expand `~` via
  the shell). This makes the driver non-portable across machines/users and
  impossible to unit test without either the real compiled executable at that
  exact path or a monkeypatched `driver`/path variable. Needs to become a
  parameter or environment variable before it can be tested in CI.
- `write_IRI2020_namelist` has dead defensive code:
  `if not hasattr(DateTime,'minute'): DateTime.minute = DateTime.minute`
  (assigns an attribute to itself, and `datetime` objects always have
  `.minute` anyway, so the branch never executes). Harmless but confusing;
  same pattern appears in `IRI_Sample_Inputs.__init__` in the other module.
- `read_IRI2020_binary_output` / `write_IRI2020_namelist` /
  `read_IRI2020_binary_output` round-trip is untested — this is the exact
  binary contract with the Fortran side (header ints, float32 layout,
  transpose order) and is the single point most likely to silently
  misalign if the Fortran driver's output format ever changes.
- `find_containing_triangles`, `get_observation_operator`, and
  `forward_model_mesh_tec` are all substantial, independently-implemented
  geometric/numerical routines with no visible tests. `get_observation_operator`
  (builds a sparse-ish linear `H` matrix) and `forward_model_mesh_tec` (does
  the equivalent ray integration directly) compute overlapping physics two
  different ways — they're a natural cross-check pair (`H @ flattened_edps`
  should match `forward_model_mesh_tec` on the same geometry/EDPs, up to
  interpolation-scheme differences worth quantifying).

### 2.3 `IRI_Sample_Inputs/IRI_Sample_inputs.py`

- `quantileSamples`: the 5-fold nested loop iterates
  `for ig12 in f107_range:` instead of `for ig12 in ig12_range:` — so every
  `ig12` value in the returned quantile sample table is actually drawn from
  the F10.7 quantile range, not the IG12 range. `randomSamples` does not have
  this bug (it correctly builds `ig12_idx` from `ig12_range`), so the two
  "equivalent" sampling methods currently produce structurally different
  columns for the same inputs — a good example of a regression a
  cross-method consistency test would catch immediately.
  Additionally, in `quantileSamples`, `ig12_range` (the correctly-built list
  from `self.ig_rz["ig"]`) is built but then never used — confirming this is
  a copy-paste bug (`ig12_range` shadowed by the loop reusing `f107_range`).
- Both `get_apf107` and `get_ig_rz` hit a network URL on first use and cache
  to a relative path (`apf107.dat`, `ig_rz.dat`) in the current working
  directory. This makes tests non-deterministic/network-dependent unless the
  cache file already exists in the CWD; a fixture with a small canned file
  content is needed to test the parsers in isolation from the network.
  `get_apf107`'s fixed-width column slicing (`line[0:3]`, `line[3:6]`, ...)
  and `get_ig_rz`'s comma-delimited block parsing are exactly the kind of
  format-dependent code that silently breaks if the upstream file format
  changes by one column — worth a small golden-file test.
  Note also `get_apf107`'s 2-digit-year heuristic (`yy < 30` → 2000s, else
  1900s) will mis-date years from 2030 onward — worth flagging even though
  it's not an immediate bug.
- `IRI_Sample_Inputs.__init__` does `datenum_f107.index(datenum_sim)` /
  `datenum_igrz.index(datenum_sim)` with no try/except — if the requested
  `DateTime_str` falls outside the downloaded `apf107.dat` / `ig_rz.dat`
  coverage window (e.g. a very recent or very old date, or before the cached
  file is refreshed), this raises an unguarded `ValueError`. Worth an
  explicit test for the "date out of range" case and a decision on desired
  behavior (clear error vs. clamping vs. re-download).
- `randomSamples`/`quantileSamples` have no seed parameter — `randomSamples`
  calls `np.random.default_rng()` fresh (unseeded) for every parameter, so
  results are non-reproducible across runs and even inconsistent *within* one
  call (four independent unseeded generators). Testing "does randomSamples
  produce values within the requested range" is fine without a seed, but
  reproducibility (e.g. for regression baselines) will require adding a
  `seed`/`rng` parameter.

## 3. General testing strategy

- **Framework:** `pytest`, with fixtures shared via a top-level `conftest.py`
  (proposed at `IonosphereTomography/conftest.py` or one per module folder if
  the modules stay independently importable).
- **Isolate from the network and the Fortran binary.** Both are external
  dependencies that must not gate ordinary unit-test runs:
  - `get_apf107`/`get_ig_rz`: test against small canned fixture files placed
    in a `tests/data/` folder, invoked via `tmp_path`/monkeypatched CWD, not
    the live URLs.
  - `get_IRI2020_EDP`: test `write_IRI2020_namelist` and
    `read_IRI2020_binary_output` independently with synthetic namelists/binary
    buffers (no subprocess call). Gate any test that actually shells out to
    the real `iri2020_namelist_driver` executable behind a
    `pytest.mark.skipif(not shutil.which(...))`-style guard (or an env var),
    so CI without the compiled Fortran binary still passes the rest.
- **Numerical correctness for the Parameterization module** relies heavily on
  **finite-difference checks** of analytic Jacobians (`_ne_profile_derivatives`'s
  `dNe_dP`, and the PCA "map" Jacobians) against `scipy.optimize.check_grad`-style
  central differences, at multiple points spanning each of the four EDP
  regions (topside, pure bottomside, intermediate blend, E-layer) so every
  branch of the piecewise function gets its own gradient check.
- **Round-trip tests** wherever a module claims an inverse pair:
  `get_parameter(get_density(x)) ≈ x` (for parameterizations where that's
  meant to hold, i.e. not lossy ones like PCA truncation, where instead
  reconstruction-error bounds are the right check), and NetCDF
  `saveNetCDF`/`fromNetCDF` round-trips preserving every field and attribute.
- **Property-based / invariant checks** over `hypothesis`-generated inputs
  where cheap (e.g. mesh generators: every triangle index is in range, no
  degenerate triangles, vertex count matches what the docstring promises;
  `interp_heights` weights sum to 1 inside the table).
- **Cross-implementation consistency checks** where the codebase already
  contains two independent routes to the same quantity (e.g.
  `get_observation_operator` vs `forward_model_mesh_tec`; `randomSamples` vs
  `quantileSamples` value ranges).
- **Regression/golden baselines** once known-good outputs exist (e.g. a
  vetted IRI2020 run for a fixed date/location), so future refactors don't
  silently change numerical behavior.

## 4. Module-specific test matrices

### 4.1 `IRI_Sample_Inputs`

| Area | Tests |
|---|---|
| `get_apf107` | Parses a small canned multi-line fixture correctly (field boundaries, 2-digit-year rollover at the `yy<30` boundary); uses local cache file when present and does not hit the network; raises/handles a malformed line predictably. |
| `get_ig_rz` | Parses fixture with known `Revision`/`Start_end_month`/`ig`/`rz`; raises `ValueError` on too-short input; handles blank-line variations. |
| `IRI_Sample_Inputs.__init__` | Correct `year/month/day/hour/minute/second` extraction for date-only vs full-datetime strings; correct `current_idx_f107`/`current_idx_igrz` for a known fixture; explicit test for out-of-range date (documents current unguarded-`ValueError` behavior, or the fixed behavior once addressed). |
| `save_to_file` / `fromPickle` | Round-trip preserves all attributes including cached `apf107`/`ig_rz` dicts. |
| `quantileSamples` | Cartesian-product size = product of range sizes incl. the trailing `None`; range clamping at data-array boundaries (`idx_start`/`idx_end`); **regression test pinning `ig12` to the IG12 fixture range, not F10.7's**, to catch the bug in §2.3; `attrs` correctly record the requested ranges. |
| `randomSamples` | All sampled values are drawn from the declared range (membership check); `nSample` controls output length; default (`None`) ranges only ever produce `None` for that column; cross-check that `randomSamples` and `quantileSamples` agree on the *set* of valid values per column once the `ig12` bug is fixed. |

### 4.2 `EDPSamples`

| Area | Tests |
|---|---|
| Coordinate convention | Written down explicitly (lat,lon or lon,lat) as the single source of truth; each generator (`genRectangularArea`, `genPolarArea`, `genGlobalArea`, `genLineOfSight`, `generate_occultation_mesh`) tested against that convention with a handful of known control points; `"Point"` construction checked against the same convention. |
| Mesh generators | `genRectangularArea`/`genPolarArea`/`genGlobalArea`: vertex count matches formula in the docstring; every triangle index in `[0, n_vertices)`; no degenerate (zero-area) triangles; polar mesh covers the cap with no gaps at the ring transitions; `genGlobalArea` mean nearest-neighbor spacing is close to `dSpace`. |
| `genLineOfSight` | Endpoints match inputs exactly; monotonic progression along the chord; `num_points<2` raises. |
| `find_containing_triangles` | Triangle centroids resolve to themselves; a point far outside all triangles (with no fallback) returns `-1`; degenerate/antimeridian-straddling triangles handled; returned barycentric weights sum to 1 and reproduce the query point when applied to the triangle's vertices. |
| `interp_heights` | Weights sum to 1 and reproduce a linear test function exactly for in-range queries; extrapolation weights outside `[h0,hN]` behave as documented (one weight <0 or >1). |
| `EDPSamples.__init__` | One test per `geo_type` covering required-argument validation (`ValueError` on missing args) and correct `data_vars`/`coords`/`attrs` shapes; `evaluate_iri=None` path produces correctly-shaped placeholder `edps`/`feature_edps` without calling the Fortran driver; `edps`/`feature_edps` shape assertions correctly reject mismatched arrays. |
| NetCDF round-trip | `saveNetCDF` → `fromNetCDF` → `from_xarray` reproduces the original dataset's `data_vars`, `coords`, and `attrs` for every `geo_type`, including the `"Point"` `Lat`/`Lon` bug fix from §2.2; explicit failing-then-fixed test for that bug. |
| `write_IRI2020_namelist` / `read_IRI2020_binary_output` | Namelist text contains every expected key for a small synthetic (2 altitude × 2 geo × 2 sample) input, with `NaN` sample values correctly replaced by `fill_value`; a hand-built synthetic binary buffer round-trips through `read_IRI2020_binary_output` into the expected `(nheight, npts, nSample)`-shaped arrays; a truncated/corrupt buffer raises the documented `ValueError` with the mismatch message. |
| `get_IRI2020_EDP` (integration, optional) | Skipped unless the compiled `iri2020_namelist_driver` is available locally/CI; when available, a small known (date, altitude, geolocation, sampling_parameters) case produces plausible EDP values (positive, F2-peak-shaped) and matches a previously vetted golden run. |
| `get_observation_operator` vs `forward_model_mesh_tec` | On a synthetic single-vertex (`n_geo==1`) or simple flat mesh with a known analytic EDP (e.g. constant density in an altitude band), both routines' TEC estimates agree with each other and with a hand-computed analytic integral within a documented tolerance; `H` row sums for rays fully outside the altitude range are exactly zero; TEC units (`/1e16`) sanity-checked against a simple constant-density slab. |
| `subset_region` / `subset_union_triangles` | Vertex/mesh compaction is index-consistent (`sub_mesh` indices all valid into `sub_geo`); antimeridian wraparound (`lon_min > lon_max`) selects the correct hemisphere; `clip_pts` margin behavior at the boundary; `ValueError` raised when fewer than 3 vertices survive. |
| Plotting methods | Smoke-tested only (figure is created without exception, correct number of artists) — not visually validated in CI; explicitly confirm the "no case for `Global`" gap in `plot_geolocation`/`interp` either raises clearly or is fixed to handle that `geo_type`. |

### 4.3 `Parameterization`

This module needs a short debugging pass (§2.1) before most of its tests can
even be written meaningfully — several call paths currently raise before
producing output. Proposed order: fix the wiring bugs first (signature
mismatches, typos, `.shape()` calls), *then* write the following tests, which
will immediately validate the fixes:

| Area | Tests |
|---|---|
| `density_10ex` / `log10_param` / `Jacobian_density_10ex` | Round-trip `density_10ex(log10_param(x)) == x` for `x` above the floor; values below `minlog10Density`/`10**minlog10Density` are correctly clamped; explicit test that the functions do **not** mutate their input array unless that's the intended contract (currently they do — decide and pin behavior); `Jacobian_density_10ex` matches a finite-difference derivative of `density_10ex` at several points. |
| `_ne_profile_derivatives` (ANCHOR forward model) | `partial=False` path returns density only, with correct shape, once the `params_shape` tuple-mutation bug is fixed; profile is continuous (no jump discontinuities) across the four region boundaries (`hmE`, `h_ST`, `hmF2`) for representative parameter sets; profile is non-negative everywhere; peak density/altitude in the F2 region roughly match the input `NmF2`/`hmF2` for well-separated layers. |
| `_ne_profile_derivatives` Jacobian (`partial=True`) | For each of the 4 regions independently, finite-difference check of `dNe_dP` against numerical differentiation of the density w.r.t. each of the 8 state components (`log10_NmF2, hmF2, H0, gamma, B0, B1, log10_NmE, hmE`), at several altitudes inside that region; the hand-derived implicit-function-theorem terms for `h_ST`'s dependence on `NmF2/B0/B1/NmE` (used in the intermediate-region blend) are the most error-prone lines in the file and deserve dedicated finite-difference cases with `h_ST` deliberately perturbed via each parameter. |
| `_find_hst_bisection` | Converges to the analytic root of `NmF2*exp(-x^B1)/cosh(x) - NmE = 0` for hand-picked `(NmF2,B1,NmE)` where the root is known/solvable in closed form or via an independent root-finder (`scipy.optimize.brentq`) as ground truth. |
| `extract_robust_f2_peak` | Recovers known `(NmF2, hmF2)` from noise-free synthetic profiles generated by `_ne_profile_derivatives` itself (self-consistency); degrades gracefully (documented fallback) on very sparse/noisy input; the module's own docstring claims "~0 km mean bias" — write the regression test the docstring implies but doesn't ship. |
| `_h0_seed_from_profile`, `_fit_topside_H0_gamma`, `_fit_bottomside_B0_B1` | Recover known `H0/gamma` and `B0/B1` from noise-free synthetic topside/bottomside profiles generated by `_ne_profile_derivatives` with known parameters (self-consistency, same pattern as the peak-finder test). |
| `_fit_iri_params` (full ANCHOR inversion) | End-to-end: generate a profile from a known 8-parameter vector via `_ne_profile_derivatives`, recover parameters via `_fit_iri_params`, regenerate the profile from recovered parameters, and check the two profiles agree within a log10-RMSE tolerance across a grid of representative "true" parameter sets (quiet/disturbed ionosphere, day/night hmF2 ranges, etc. — reuse the ranges in `EDP_Parameterization`'s default `hyper_params` bounds as the sampling space). |
| `PCA2EDP_1D`/`EDP2PCA_1D` and the `_3D` variants | Round-trip `EDP2PCA_1D(PCA2EDP_1D(c, PCA), PCA) ≈ c` for `c` in the PCA's column space; reconstruction error is bounded and decreases as more components are retained; dimension-mismatch assertions actually fire on mismatched shapes; **once `get_PCA`'s `.shape()` bug and retention-direction logic (§2.1) are resolved**, `get_PCA` on a synthetic low-rank `edps` matrix recovers the known rank/components. |
| `PCA2EDP_1D_map`/`_3D_map` (Jacobians) | Finite-difference check against `PCA2EDP_1D`/`_3D` for both `linear=True` and `linear=False` (log-space) branches. |
| `EDP_Parameterization` | One test per `style` covering constructor validation (missing `PCA` in `hyper_params` raises clearly, once the `.keys` bug is fixed; wrong `PCA.ndim` raises); `get_density`, `get_parameter`, `get_Jacobian` each produce correctly-shaped output for each style, once the wiring bugs in §2.1 are fixed. |
| `Parameterized_EDPSamples` | Constructs successfully for each style against a small synthetic `EDPSamples` (from module 2's test fixtures) once the constructor's `get_parameter(..., hyper_params=...)` call-signature bug is fixed; `param_vec`/`PCA` are correctly attached to the dataset's `data_vars`; NetCDF round-trip (`saveNetCDF`/`fromNetCDF`/`from_xarray`) preserves `Parameterization_style` and hyperparameter attrs and reconstructs an equivalent object. |
| `reconstruction_error` / `error_summary` (§8) | Once introduced, these become the implementation of the round-trip tests above rather than separate test-only code: `density_10ex`'s residual is ~0 off the floor; `'ANCHOR'`'s residual matches the log10-RMSE computed independently in the `_fit_iri_params` sweep; PCA styles' residual shrinks monotonically as `retaining_threshold`/component count increases. |

## 5. Proposed sequencing

1. **Fix the blocking wiring bugs in `Parameterization.py`** (§2.1: the
   `get_parameter`/`_fit_iri_params` keyword mismatches, the `bhyper_params`
   typo, the `.keys`/`.shape()` call-vs-attribute mistakes, the tuple
   mutation in `_ne_profile_derivatives`). These currently make entire code
   paths unreachable, so no amount of testing can validate them until fixed.
   This is a short, mechanical pass — I'd scope it separately from the
   Jacobian correctness work below. **Since `Parameterization` is brand new
   and nothing depends on it yet, I'd fold the §7.4 restructuring
   (strategy-object registry, tightened boundary between
   `EDP_Parameterization` and `Parameterized_EDPSamples`) into this same
   step** rather than patch the current call sites and re-restructure later —
   the fix and the restructuring touch the same lines.
2. **Module 1 (`IRI_Sample_Inputs`) unit tests**, including the fixture-based
   parser tests and the `quantileSamples` `ig12` regression test — cheapest
   module, no external dependencies once network calls are fixture-backed,
   and a good place to establish the `conftest.py`/fixture pattern.
3. **Module 2 (`EDPSamples`) geometry and I/O tests** that don't require the
   compiled Fortran binary: mesh generators, `find_containing_triangles`,
   `interp_heights`, dataset construction/validation, NetCDF round-trips
   (including nailing down and fixing the coordinate-order convention from
   §2.2 — this should happen before writing the bulk of the geometry tests,
   since the convention affects how every test's expected values are
   written).
4. **Module 3 (`Parameterization`) numerical correctness tests**: round-trips,
   Jacobian finite-difference checks region-by-region, and the
   `_fit_iri_params` inversion-accuracy sweep. This is the most
   math-intensive part of the plan and the highest-value target, since a
   silently wrong Jacobian would degrade EKF performance without an obvious
   symptom.
5. **Forward-operator tests in `EDPSamples`** that need either the real IRI2020
   executable or hand-built synthetic EDPs: `get_observation_operator` vs
   `forward_model_mesh_tec` cross-check, and (if the compiled driver is
   available) the `get_IRI2020_EDP` integration test.
6. **Cross-module smoke test**: `IRI_Sample_Inputs.randomSamples` →
   `EDPSamples(..., evaluate_iri=1, ...)` → `Parameterized_EDPSamples` on a
   tiny grid (e.g. 3 altitudes × 1 geolocation × 2 samples), checked for
   shape/attrs consistency only (not numerical accuracy) — a fast guard
   against interface drift between the three modules.

## 7. Structural and design recommendations

These are independent of the bug list in §2 — they're about shape, not
correctness. Since the surface area is still small and nothing downstream is
locked into today's interfaces (especially for `Parameterization`), doing
some of this now is cheaper than doing it after the EKF integration depends
on current signatures.

### 7.1 Cross-cutting

- **Package layout.** None of these three folders is an importable package
  (no `__init__.py`, unlike `Abel_Inverter`/`observation_preparation`/etc.),
  and `Parameterization.py` does `from edp_samples import EDPSamples`, which
  only resolves if `EDPSamples/` happens to be on `sys.path`. Turning
  `IRI_Sample_Inputs`, `EDPSamples`, and `Parameterization` into a real
  package (e.g. `ionotomo.iri_inputs`, `ionotomo.edp_samples`,
  `ionotomo.parameterization`) with proper relative imports would remove the
  sys.path fragility and make the three modules pip-installable/testable the
  same way as the rest of the repo.
- **`self` vs `cls`.** `EDPSamples.__init__`, `EDP_Parameterization.__init__`,
  and `Parameterized_EDPSamples.__init__` all name their instance parameter
  `cls`. It works (Python doesn't care about the name), but it reads as if
  the constructor is mutating the *class*, and it makes an eventual
  `@classmethod` factory (see below) confusing to add next to it. Renaming to
  `self` is a pure clarity win with zero risk.
- **Validation via bare `assert`.** Both `EDPSamples.__init__` and
  `from_xarray` validate structure with `assert cond, [msg]` — asserts
  disappear under `python -O`, and wrapping the message in a list literal
  means the traceback prints `AssertionError: ['message']` instead of a clean
  string. Prefer explicit `if not cond: raise ValueError(msg)`, which is also
  what most of the rest of the file already does.
- **Parallel `match`/`case` blocks that must be kept in sync by hand.** The
  `geo_type` switch is re-implemented independently in `__init__`,
  `from_xarray`, `interp`, and `plot_geolocation` — which is exactly how the
  "no `Global` case in `interp`/`plot_geolocation`" gaps happened (§2.2).
  Similarly, `Parameterization_Style` is re-switched independently in
  `EDP_Parameterization.__init__`, `.get_density`, `.get_parameter`, and
  `.get_Jacobian` (§7.4 below). A registry/strategy pattern — one dict
  mapping the enum value to a small object bundling everything that style
  needs — turns "did I forget a branch somewhere" into "does the registry
  have an entry," which is checkable in one place instead of four.
- **Pure functions mutating caller-owned arrays.** `density_10ex`,
  `log10_param`, `Jacobian_density_10ex`, `EDP2PCA_1D`, `EDP2PCA_3D` all clamp
  values via boolean-indexed assignment directly on the input array. Prefer
  `np.where`/`np.clip` to build and return a new array, so callers aren't
  surprised that a "get" call silently modified data they still hold a
  reference to.
- **No seam for external dependencies.** The live HTTP fetch in
  `get_apf107`/`get_ig_rz` and the hardcoded Fortran path/subprocess call in
  `get_IRI2020_EDP` are both inlined into the function that also does the
  parsing/science. The cursor-history notes for `edp_samples.py` describe an
  earlier design with an injectable `driver` callable
  (`framework/iri2020_driver.py`) that the current code doesn't seem to use
  anymore — worth restoring that seam (and adding an analogous one for the
  index-fetching functions) purely so tests can substitute a fixture without
  needing the network or the compiled binary.
- **One documented lat/lon convention, enforced by an accessor.** Rather than
  every method indexing `geo[:, 0]`/`geo[:, 1]` directly (and disagreeing
  about which is which — §2.2), add a tiny helper (e.g.
  `lon, lat = EDPSamples.split_lonlat(geo)`) that all internal code goes
  through. That doesn't fix the ambiguity by itself, but it means the
  convention only has to be correct in one place, and a future audit is a
  single grep instead of re-reading every method.

### 7.2 `IRI_Sample_Inputs`

- Separate "fetch the file" from "parse the file" (two small functions/one
  injectable source object per index file) so the parser can be tested
  without the network and the fetch logic can be tested/mocked without
  needing a working parser.
- Add an explicit `seed`/`rng` parameter to `randomSamples` instead of four
  independent unseeded `np.random.default_rng()` calls, for reproducible
  regression baselines.
- The "low/mid/high/`None`" list-per-column pattern (e.g. `f107_range`
  ending in `.append(None)`) is a homegrown convention that's easy to get
  subtly wrong positionally — it's arguably how the `ig12`/F10.7 mixup (§2.3)
  slipped in. A small named structure (e.g. a `SampleRange` dataclass with
  explicit `low`, `mid`, `high`, `include_unconstrained` fields) would make
  that class of copy-paste error more visible on review.
- The "expand a start/end month into a full month/year list" logic is
  duplicated almost verbatim between `show_iri_inputs` and
  `IRI_Sample_Inputs.__init__` — factor into one helper.

### 7.3 `EDPSamples`

- **Split the 2500-line class by concern.** It currently interleaves
  geometry generation (`genRectangularArea`, `genPolarArea`, ...), Fortran
  I/O (`write_IRI2020_namelist`, `read_IRI2020_binary_output`,
  `get_IRI2020_EDP`), forward operators (`get_observation_operator`,
  `forward_model_mesh_tec`), and plotting (`plot_geolocation`,
  `plot_mesh_globe`, `plot_edp_statistics`, `plot_edp_covariance`) inside one
  class. Most of this is already informally separable (many of the geometry
  helpers are `@staticmethod`s that don't touch `self` at all). Moving each
  concern into its own module (`geometry.py`, `iri_driver.py`,
  `forward_operators.py`, `plotting.py`) with `EDPSamples` composing/calling
  them would make each piece reviewable and testable in isolation, and would
  shrink the risk surface for any single change.
- **Replace the one large `__init__` (5 geo_types, ~15 optional kwargs) with
  small factory classmethods** — `EDPSamples.point(...)`,
  `.rectangle(...)`, `.polar(...)`, `.occultation(...)`, `.global_grid(...)`
  — each owning both its own construction *and* its own reconstruction logic
  (the part currently duplicated in `from_xarray`'s parallel `match`). This
  directly targets the class of bug in §2.2 where `__init__`, `from_xarray`,
  `interp`, and `plot_geolocation` have each independently forgotten a
  `geo_type` case — with per-geo_type factories there's exactly one place
  that knows how to build and rebuild a given grid type.
- **Prefer returning new objects over in-place mutation** consistently.
  `subset_union_triangles` already does this correctly (builds a fresh
  instance); it would be worth auditing the rest of the class (and
  `Parameterized_EDPSamples`, which mutates a passed-in `EDPSamples` instance
  as a side effect of wrapping it — see §7.4) to the same standard.

### 7.4 `Parameterization` — best time to settle this, since it's brand new

- **Strategy-object registry instead of three parallel `match` blocks.**
  `get_density`, `get_parameter`, and `get_Jacobian` each re-implement the
  same style dispatch. A dict from style name to a small
  dataclass/`NamedTuple` of three callables (`to_density`, `to_parameter`,
  `jacobian`, plus maybe a `needs_altitude: bool` flag) would mean adding or
  fixing a style touches one registry entry instead of three call sites that
  can silently drift apart — which is close to how the current
  `get_parameter`/`_fit_iri_params` keyword mismatch (§2.1) happened.
- **Tighten the boundary between `EDP_Parameterization` (pure array math) and
  `Parameterized_EDPSamples` (the xarray/dataset wiring).** Right now the
  dataset-wiring class re-implements per-style branching structurally
  identical to `EDP_Parameterization`'s own dispatch, and the two disagree on
  keyword names. If `Parameterized_EDPSamples` only ever called
  `EDP_Parameterization`'s three methods positionally
  (`get_parameter(density, alt)`), this whole class of mismatch is
  structurally impossible.
- **Document `needs_altitude` per style explicitly** (on the registry entry
  suggested above) rather than leaving it implicit in which `case` branches
  happen to use the `alt` argument.
- **Consider a standing finite-difference (or autodiff) cross-check for the
  analytic Jacobian.** `_ne_profile_derivatives`'s hand-derived
  implicit-function-theorem chain rule (for how `h_ST` depends on
  `NmF2`/`B0`/`B1`/`NmE`) is the single most algebra-heavy, most
  error-prone piece of code in all three modules. Keeping the analytic
  version for speed in production but having a finite-difference (or, if an
  autodiff library is an acceptable dependency, `jax`/`torch`-based) Jacobian
  as a permanent test oracle — not just a one-off test — would catch future
  regressions in that derivation cheaply.

## 8. New requirement: parameterization error diagnostics

`EDPSamples` already has ensemble-level inspection methods
(`plot_edp_statistics`, `plot_edp_covariance`) that visualize the *data*.
`Parameterization` should get the analogous thing for the *approximation
error the parameterization itself introduces* — PCA truncation error for the
`PCA_*` styles, and analytic-model fit error for `'ANCHOR'`. These turn out
to be the same underlying computation viewed two ways, which is worth
designing once rather than per-style.

### 8.1 Core primitive: style-agnostic round-trip error

Add one method to `EDP_Parameterization`:

```python
def reconstruction_error(self, density, alt=None, in_log10=False):
    """
    density -> get_parameter -> get_density -> density_hat
    Returns (residual, density_hat), residual = density - density_hat
    (or log10(density) - log10(density_hat) if in_log10=True).
    """
```

This one method *is* the diagnostic for every style, because "how well does
the parameterization represent this EDP" is always "encode then decode and
look at what's left over":

- `'density_10ex'`: residual should be ~0 except where the floor clamp is
  active — a good sanity check that the round-trip machinery itself is
  correct, and a natural first test case (§4.3 already scopes this
  round-trip; this method is exactly what that test should call).
- `'ANCHOR'`: residual **is** the fit error of the 8-parameter analytic model
  against the true (IRI-generated) profile — literally "what is the error in
  representing a given sample of EDP" as you described.
- `'PCA_1D'`/`'PCA_3D'` (+ `_10ex` variants): residual **is** the truncation
  error from discarding modes beyond `retaining_threshold`.

Because it's one method reused by three styles, it also becomes the single
place that needs updating if the parameterization contract changes, and it's
directly reusable by the round-trip tests in §4.3 rather than being
test-only code duplicated from the diagnostic.

### 8.2 Visualization — lives on `Parameterized_EDPSamples`, styled like `EDPSamples`'s plots

These need altitude/geo/sample context, so they belong on the dataset-wrapping
class (mirroring where `plot_edp_statistics`/`plot_edp_covariance` live on
`EDPSamples`, not as free functions), and should reuse the same visual
language already established there (percentile envelopes vs. altitude,
`fill_betweenx`, robust `nanpercentile`, altitude on the y-axis):

- **`plot_reconstruction_profile(sample_idx=0, geo_idx=0, ax=None)`** — a
  single profile's original vs. reconstructed density vs. altitude side by
  side with its residual, for one sample/geo point. For `'ANCHOR'`, mark the
  region boundaries (`hmE`, `h_ST`, `hmF2`) as horizontal reference lines,
  since the piecewise model's error is expected to concentrate near the
  region transitions — this is the plot that answers "where does the ANCHOR
  fit struggle." For PCA styles, annotate the number of retained components
  in the title.
- **`plot_reconstruction_error_statistics(ax=None)`** — the error analogue of
  `plot_edp_statistics`: percentile envelopes (1–99/5–95/16–84) of the
  residual aggregated over all samples and geo points, vs. altitude, plus a
  panel normalizing the residual by the local median |EDP| (a relative-error
  percentile envelope is more actionable than an absolute one, since EDP
  spans many orders of magnitude with altitude). Annotate overall RMSE.
- **`plot_pca_spectrum(ax=None)`** — a scree plot for `PCA_*` styles: singular
  value spectrum (log-y) plus cumulative explained-variance-fraction curve,
  with the `retaining_threshold` cutoff marked and the resulting number of
  retained components labeled. This is the "why" companion to
  `plot_reconstruction_profile`'s "what effect" — together they let you pick
  `retaining_threshold` by looking at both the spectrum and the resulting
  profile-space error, rather than the threshold alone. This can live on
  `EDP_Parameterization` directly (it only needs the PCA matrix + threshold,
  not the full dataset) with a thin pass-through on `Parameterized_EDPSamples`
  for convenience.
- **`error_summary() -> dict`** — the non-plot numeric companion: overall RMSE
  (linear and log10-space), an RMSE-vs-altitude profile, max-abs-error and
  where it occurs (altitude/sample/geo index); for `'ANCHOR'`, RMSE broken out
  per region (topside/bottomside/intermediate/E-layer, using the same region
  masks `_ne_profile_derivatives` already computes internally — worth a small
  refactor to expose `h_ST` per point, e.g. via `_find_hst_bisection` called
  directly from the diagnostic using the fitted parameter vector, rather than
  re-deriving it); for PCA styles, number of retained components and
  cumulative variance explained.

### 8.3 Note on a dependency between this and an open question

`plot_pca_spectrum`'s cutoff marker and `error_summary`'s "number of retained
components" both depend on which direction `get_PCA`'s `retaining_threshold`
comparison is supposed to go (§2.1, and the retaining-threshold question in
§9 below) — worth resolving that first, since building the scree plot is
actually a fast way to *see* whether the current `<=` comparison keeps the
dominant or the residual modes, which should make the intended direction
obvious once plotted against a real ensemble.

## 9. Open questions before starting

- **Is the compiled `iri2020_namelist_driver` executable available in this
  environment** (or reachable via a path we can point `IRIPath` at), so that
  tests in §4.2's "integration (optional)" row and §5 step 5 can actually run
  end-to-end? If not, those stay skipped/marked and we test everything else
  around the Fortran boundary with synthetic namelist/binary fixtures.
- **What is the intended geolocation column convention** — lat-first or
  lon-first? This needs an explicit decision (§2.2) since the current code is
  internally inconsistent; I'd rather confirm the intended convention with
  you than guess and pick the wrong one to standardize on.
- **What is the intended semantics of `get_PCA`'s `retaining_threshold`** —
  "keep components whose singular value is at least X% of the largest" (the
  usual meaning, which would mean the current `<=` comparison is inverted) or
  something else? Same question for whether `density_10ex`/`log10_param`/etc.
  mutating their input arrays in place is intentional (a memory-saving
  choice) or accidental.
- **Priority/time-boxing**: do you want the full matrix above, or should I
  start with just the Parameterization wiring fixes + Jacobian tests (§5
  steps 1 and 4), since that's the highest-risk code for the Kalman filter,
  and treat modules 1–2 as a follow-on pass?
- **How far should the §7 restructuring go, and for which modules?** For
  `Parameterization` I'd fold the strategy-registry redesign into the bug-fix
  pass regardless (§5 step 1), since it's the same lines either way. For
  `IRI_Sample_Inputs`/`EDPSamples`, which already have some tests against
  their current shape, there's a real tradeoff: restructuring now (package
  layout, splitting `EDPSamples` by concern, per-geo_type factories) reduces
  risk for everything built on top of them later, but touches more surface
  area than a pure bug-fix-and-test pass would. Options, roughly increasing
  in scope: (a) structural fixes only where they directly cause a bug I
  already found (e.g. the geo_type registry, to kill the missing-`Global`-case
  class of bug); (b) (a) plus the dependency-injection seams (Fortran driver,
  index fetch) so the test suite doesn't need network/binary access; (c) full
  §7 treatment including the package layout and splitting `EDPSamples` into
  separate modules. I'd lean toward (b) as the sweet spot, but this is your
  call to make given how much else is planned to build on these two modules.
