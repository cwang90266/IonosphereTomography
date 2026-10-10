# Plan: Refactoring RO/IGS Observation Preparation

Author: scoping pass by Claude, 2026-09-22
Status: **all five §10 implementation steps done as of 2026-09-22** (see
§10 for what landed and how each was verified, including against real
RO/IGS sample data). All seven §8 open questions were answered before
implementation began.

## 1. Purpose and scope

This document scopes a refactor of `observation_preparation/` — the two
routines that turn raw Radio Occultation (RO) and IGS ground-GNSS data into
the TEC/GNSS/receiver observation entries consumed by the rest of the
data-assimilation pipeline. It covers:

1. Consolidating `prepare_ro_observations.py` and `prepare_igs_observations.py`
   (plus scattered helper logic in `TEC_model/` and elsewhere) into one
   cohesive module.
2. Replacing the current per-ray/per-epoch CSV export with structured
   netCDF output.
3. Adding a region-of-interest (ROI) selection mode that requires the
   **entire line-of-sight (LOS) below a given altitude** to lie inside the
   ROI, not just the tangent point (RO) or a single pierce point (IGS).
4. Scoping diagnostic/verification tooling — geolocation maps, LOS/ROI
   penetration plots, and measured-vs-`EDPSamples`-predicted TEC comparisons.

No code changes are made yet. The intent is to agree on the target
structure and sequencing before starting.

## 2. Current state (what I read)

### 2.1 Package layout today

```
observation_preparation/
  __init__.py                 # re-exports both prepare_* entry points + roi_tools
  prepare_ro_observations.py  # RO: scan -> filter -> select -> parse -> clean -> (Abel) -> CSV/PNG
  prepare_igs_observations.py # IGS: run stations -> clean -> filter time/ROI -> (collapse) -> CSV/PNG
  roi_tools.py                # Fibonacci-sphere / geodesic-circle / square ROI geometry (shared)
  test_ro_preparation.py, test_igs_preparation.py,
  test_roi_consistency.py, test_igs_roi_voxels.py
```

Both preparation modules call out to code that stays elsewhere:

- `TEC_model/podTc_file_processing.py` — `parse_podTc2_nc_file`, `rayTangent`,
  `ECEFtolla`: low-level RO netCDF parsing and file-level geometry QC.
- `TEC_model/igs_tec_pipeline.py` — `process_igs_station`,
  `igs_obs_to_clean_entry`: RINEX parsing, dual-frequency combination,
  elevation cutoff, cycle-slip/arc splitting, carrier leveling, DCB
  correction.
- `Abel_Inverter/lei_abel_inverter.py` — `run_abel_inversion`, run on the
  full RO arc for diagnostic Ne profiles (RO only).
- `EDPSamples/edp_samples.py` — downstream consumer. `get_observation_operator(podTc2_data, ...)`
  reads only `podTc2_data['LEO']` and `['GNSS']` (ECEF km, shape `(3, n_rays)`)
  and builds the ray-integration `H` matrix against the model grid.
  `rayTangent`/`get_occultation_extrema` (staticmethods on `EDPSamples`)
  already compute tangent points and LOS/altitude-shell intersections —
  this is the natural building block for objective 3 (see §5).
- `Ensemble_Kalman_Engine/observation_operator.py` — wraps
  `EDPSamples.get_observation_operator`'s `H` plus a parameterization into
  `GenericObservationOperator`, whose `.forward()` predicts TEC from a
  state vector. This is the natural building block for the TEC-comparison
  diagnostic (see §7.3).

### 2.2 What each routine actually does today

**RO (`prepare_ro_observations`)**
1. `scan_ro_metadata`: lightweight netCDF-attribute-only scan (TEC-max
   tangent lat/lon, date/time, ids) over `*.0001_nc` files — avoids parsing
   full files just to filter.
2. `filter_ro_metadata`: keeps files whose **TEC-max tangent point** is
   within `radius_km` of `(center_lat, center_lon)` (haversine), and whose
   time is in `[start_time, end_time)`.
3. `select_ro_occultations`: optional count-based subsampling via
   `test_param_iono.select_arcs_by_count_bin` (an import from outside
   `observation_preparation` — see §4.2).
4. Parses each selected file (`parse_podTc2_nc_file`), applies second-stage
   QC (finite/positive TEC, `min_valid_rays`, uniform-stride decimation to
   `max_rays_per_occultation`), optionally runs Abel inversion on the full
   (undownsampled) arc.
5. Exports one flat CSV (`ro_<roi>_<timewindow>.csv`, one row per ray,
   metadata columns repeated per row), a companion Abel-profile CSV, and one
   three-panel PNG per occultation (tangent track on a map + ROI/voxels,
   TEC-vs-height, Abel Ne-vs-height).

**IGS (`prepare_igs_observations`)**
1. Runs `process_igs_station` per station/day, converts each raw arc via
   `igs_obs_to_clean_entry` (finite/positive TEC, `min_valid_epochs`,
   downsample to `max_rays_per_arc`).
2. `filter_igs_by_time`: keeps `entry['date']` (arc start) in
   `[start_time, end_time)`.
3. `filter_igs_epochs_by_roi`: **per-epoch** ROI filter — keeps individual
   IPP epochs within `radius_km` of the center (not just a representative
   point), masking every aligned per-epoch array (`tec`, `tangent_km`,
   `ipp_lat/lon`, `LEO`/`GNSS`, etc.) — already finer-grained than RO's
   point filter, but the *pierce point* is still a single fixed-height point
   per epoch, not the full LOS.
4. `epoch_mode="center"` (default) collapses each arc to its central epoch
   (current static-ionosphere experiment); `"all"` keeps every epoch.
5. Exports one flat CSV (`igs_<roi>_<timewindow>.csv`, one row per epoch)
   and one PNG (IPP map + ROI/voxels + TEC scatter).

### 2.3 The shared "clean entry" schema

Both routines converge on the same in-memory dict shape, which is exactly
what `EDPSamples.get_observation_operator` and the exporters consume:

| Key | Shape | Meaning |
|---|---|---|
| `tec` | `(n_rays,)` | Measured slant TEC (TECU) |
| `LEO` | `(3, n_rays)` | Receiver ECEF km (LEO for RO, ground station for IGS — name is RO-legacy) |
| `GNSS` | `(3, n_rays)` | Transmitter ECEF km |
| `tangent_km` / `ipp_lat`,`ipp_lon` | `(n_rays,)` | RO: tangent-point altitude. IGS: pierce-point lat/lon |
| `leo_id`/`rx_id`, `prn_id` | scalar | Receiver / transmitter identifiers |
| `date`, `label`, `tec_type`, `obs_source` | scalar | Provenance |
| RO only: `abel`, `occ_type`, `lat/lon_tecmax_tangent`, `snr_l1/l2` | — | Diagnostics |
| IGS only: `arc_time_sec`, `time_s`, `time_utc_h`, `elev_deg` | `(n_rays,)` | Per-epoch diagnostics |

This is exactly the user's "7 components" (TEC + GNSS-XYZ + REC-XYZ), plus
source-specific diagnostic fields. This confirms a unified schema is
realistic — RO and IGS already agree on the core 7 fields; they differ only
in per-ray diagnostic extras and in how the representative geolocation is
derived (tangent point vs. IPP).

### 2.4 Duplication and rough edges found

- `_haversine_km`, `_normalize_timestamp`, `_format_filename_number`,
  `_time_window_csv_tag`, `_bearing_and_distance_km` are copy-pasted
  verbatim between the two `prepare_*.py` files.
- The map-plotting scaffolding (`Orthographic`/`Robinson` projection setup,
  ROI-circle + Fibonacci-voxel overlay, legend) is duplicated between
  `_plot_ro_event` and the inline plotting block in `export_igs_outputs`.
- `select_ro_occultations` imports `test_param_iono.select_arcs_by_count_bin`
  from *outside* `observation_preparation` — a leftover coupling to the
  main/demo code the package's docstring says it's trying to avoid ("no
  KF/EKF/voxel dependency").
- ROI filtering is point-based in both routines: RO gates on one tangent
  point per occultation, IGS gates per-epoch on one IPP per epoch. Neither
  checks whether the rest of the ray (all altitudes below some limit, not
  just the tangent/pierce altitude) stays inside the ROI footprint —
  objective 3.
- Output is a flat, denormalized CSV: one row per ray/epoch with all
  per-occultation/per-arc metadata (filename, ids, date) repeated on every
  row, string-formatted dates, and (for RO) a separate, index-unaligned
  Abel-profile CSV joined only by filename. This is what objective 2 asks
  to replace.
- Diagnostics today are a single fixed PNG per run/occultation baked into
  the export function — there's no standalone, reusable verification tool,
  and nothing compares measured TEC against an `EDPSamples`-predicted
  value (objective 4).

## 3. Objectives (restated and expanded)

1. **One module.** Merge RO and IGS preparation into a single package with
   one shared schema, shared ROI/time-filtering primitives, shared
   plotting/export code, and thin source-specific adapters for what
   actually differs (podTc2 parsing vs. RINEX/IGS pipeline).
2. **netCDF output**, structured around the 7-component
   TEC/GNSS-ECEF/REC-ECEF (+ geodetic convenience + metadata) schema, so
   downstream code (`EDPSamples`, the Kalman engine) can load it directly
   instead of re-deriving `LEO`/`GNSS` arrays from a flat CSV.
3. **Full-LOS ROI containment**, selectable alongside (not necessarily
   replacing) the current point-based ROI filters: keep an
   occultation/arc only if the entire ray path below a configurable
   altitude limit (e.g. the LEO's own altitude, or the model top) stays
   within the ROI.
4. **Diagnostics**: geolocation maps, LOS/ROI-penetration plots, and
   measured-vs-predicted TEC comparisons, built on top of `EDPSamples`
   where possible rather than reimplementing geometry.

## 4. Proposed module layout

```
observation_preparation/
  __init__.py            # re-exports the public API below
  schema.py               # NEW — the unified observation-entry dataclass/schema,
                           #        shape/dtype contracts, netCDF variable names
  roi.py                  # roi_tools.py, renamed for symmetry with the other
                           # source-neutral modules; unchanged geometry helpers
  roi_selection.py        # NEW — point-based AND full-LOS ROI selection,
                           #        shared by both sources (see §5)
  time_filter.py           # NEW — _normalize_timestamp, date-window helpers
                           #        (dedup of logic copied in both prepare_*.py)
  ro_source.py             # was prepare_ro_observations.py, trimmed to what's
                           #        genuinely RO-specific: scan/parse/Abel
  igs_source.py             # was prepare_igs_observations.py, trimmed to what's
                           #        genuinely IGS-specific: station/day looping
  netcdf_io.py             # NEW — unified netCDF writer/reader for the schema
                           #        in schema.py (replaces export_ro_outputs /
                           #        export_igs_outputs' CSV halves)
  diagnostics.py           # NEW — geolocation maps, LOS-ROI penetration plots,
                           #        measured-vs-predicted TEC comparison (§7)
  prepare_observations.py  # NEW — thin unified entry point:
                           #        prepare_observations(source="RO"|"IGS", ...)
                           #        dispatches to ro_source/igs_source, applies
                           #        the shared ROI/time filters, writes netCDF
  test_*.py                 # updated to match
```

`select_ro_occultations`'s external dependency on
`test_param_iono.select_arcs_by_count_bin` should either be vendored into
`observation_preparation` (if the selection logic is genuinely
project-generic) or kept as an explicit, documented optional import — not
silently reached across the codebase. Worth a decision in §8.

**File layout confirmed multi-file** (§8b) — the split above stays, rather
than consolidating into one file the way `EDPSamples`/`Parameterization`
are written.

## 5. Full-LOS ROI containment (objective 3)

### 5.1 What exists to build on

`EDPSamples` already has the geometry primitives needed:

- `EDPSamples.rayTangent(LEO, GNSS, units='km')` → tangent point (ECEF),
  radius, geodetic altitude, vectorized over rays.
- `EDPSamples.get_occultation_extrema(LEO, GNSS, alt_limit, r_earth_km)` →
  three (lat, lon) points: the highest valid tangent point, and the
  entry/exit points where the *deepest* ray crosses the `alt_limit`
  altitude shell. This is already exactly "how far does this LOS's
  footprint reach below a given altitude" — currently used only to build a
  triangular mesh for one occultation (`generate_occultation_mesh`), but
  it's the right primitive to reuse rather than reimplement.
- `EDPSamples.get_observation_operator` itself samples each ray into
  `num_segments` points between `GNSS` and `LEO`, masks to
  `[altitude[0], altitude[-1]]`, and only those segments contribute to
  `H`. Any full-LOS ROI check should use the same sampling-and-masking
  approach so "is this ray inside the ROI" and "does this ray actually
  contribute to H" stay consistent by construction.

### 5.2 Proposed function — DECIDED (§8.1, §8.2)

Add to `roi_selection.py` (source-agnostic — works for RO `LEO`/`GNSS` or
IGS receiver/GNSS pairs alike, since both are ECEF `(3, n_rays)` pairs):

```python
def los_within_roi(
    LEO: np.ndarray,          # (3, n_rays) ECEF km — receiver
    GNSS: np.ndarray,         # (3, n_rays) ECEF km — transmitter
    center_lat: float,
    center_lon: float,
    radius_km: float,
    alt_limit_km: float,      # user-supplied constant; see note below
    num_segments: int = 200,
    fraction_required: float = 1.0,  # user-tunable, not just an escape hatch
) -> np.ndarray:               # (n_rays,) bool
    """True where at least `fraction_required` of each ray's segments below
    alt_limit_km fall within radius_km of (center_lat, center_lon)."""
```

Implementation sketch: sample each ray at `num_segments` points (mirroring
`get_observation_operator`'s own sampling), convert to geodetic, mask to
altitude `<= alt_limit_km`, compute great-circle distance of every masked
point from the ROI center, and require at least `fraction_required` of
them (not necessarily all) to be `<= radius_km`.

**`alt_limit_km` is a required, user-supplied constant for the first cut**
(§8.1) — the user pointed out receiver altitude actually varies across
LEOs in a batch, so there's no single "the receiver's altitude" to default
to automatically. A natural per-batch alternative (take the *lowest*
altitude among all LEOs actually selected) is noted here as a later
enhancement, not built now: it would need a first pass over the batch to
find that minimum before this filter can run, which complicates the
scan→filter→select ordering in §2.2 for no immediate benefit — start with
an explicit constant and revisit if a per-batch derived limit turns out to
matter in practice.

**`fraction_required` is a confirmed, first-class parameter** (§8.2), not
a deferred escape hatch — the user wants a tunable minimum fraction of the
sub-`alt_limit_km` LOS that must be inside the ROI, rather than an
all-or-nothing rule. Default `1.0` (strict) unless the user specifies
otherwise per run.

### 5.3 How it plugs into selection

Expose it as a mode, not a silent replacement, so existing behavior (and
existing tests / analyses that assume the tangent/IPP-only filter) doesn't
change unless asked for. `full_los` is now the intended default mode for
new preparation runs (§8.1); the point-based modes stay available for
reproducing/comparing against today's behavior:

```python
prepare_observations(..., roi_mode="full_los")          # NEW default, objective 3
prepare_observations(..., roi_mode="tangent_point")   # today's RO behavior
prepare_observations(..., roi_mode="pierce_point")     # today's IGS behavior
```

## 6. Unified schema and netCDF output (objective 2)

### 6.1 In-memory schema — key names DECIDED (§8.5), scope refined in §8a

Formalize the existing per-ray/per-epoch dict shape in `schema.py` as a
dataclass or `TypedDict` instead of an implicit convention, with the 7
core fields required and source-specific fields optional:

```python
@dataclass
class ObservationEntry:
    obs_type: str            # "RO" | "IGS"
    tec: np.ndarray          # (n_rays,)
    gnss_ecef_km: np.ndarray # (3, n_rays)  -- transmitter
    rec_ecef_km: np.ndarray  # (3, n_rays)  -- receiver (LEO or ground)
    # + geodetic convenience (lat/lon/alt for gnss and rec, derived not stored twice)
    # + provenance: date, label, rec_id, prn_id, tec_type
    # + source-specific extras as a free-form dict (abel, elev_deg, ipp_lat/lon, snr...)
```

`LEO`/`GNSS` are renamed to `rec_ecef_km`/`gnss_ecef_km` with no
backward-compatible alias — confirmed (§8.5). See §8a immediately below
for what this rename does and does not touch outside
`observation_preparation`.

### 6.2 netCDF layout — DECIDED (§8.3, §8.4): padded arrays, two files

Structure around two dimensions, `obs` (one occultation or arc) and `ray`
(one TEC/GNSS/REC sample within it), which maps directly onto what
`get_observation_operator` needs (`tec` `(n_rays,)`, `rec_ecef_km`/
`gnss_ecef_km` `(3, n_rays)` *per obs*):

```
dimensions:
    obs   = n_observations
    ray   = max_rays_per_obs        # padded; see ragged-array note below
    xyz   = 3

variables:
    obs_type(obs)            string  "RO" | "IGS"
    n_rays(obs)               int     valid ray count for this obs (<= ray)
    tec(obs, ray)             float32 TECU, NaN-padded
    gnss_ecef_km(obs, ray, xyz)  float64
    rec_ecef_km(obs, ray, xyz)   float64
    tangent_alt_km(obs, ray)  float32   RO: tangent height; IGS: NaN or pierce alt
    pierce_lat(obs, ray), pierce_lon(obs, ray)  float32  IGS IPP / RO tangent lat-lon
    elev_deg(obs, ray)         float32  IGS only, NaN for RO
    snr_l1(obs, ray), snr_l2(obs, ray)  float32  RO only, NaN for IGS
    date(obs)                  int64 (epoch seconds) or CF time
    rec_id(obs), prn_id(obs), label(obs)  string
    roi_center_lat, roi_center_lon, roi_radius_km, roi_mode,
    roi_alt_limit_km, roi_fraction_required  # global attrs
```

**Padded 2D `(obs, ray)` arrays**, confirmed (§8.3) — simple, works
naturally with `xarray`/`netCDF4`, a direct, low-risk translation of
today's CSV rows into a typed array. The CF ragged-array alternative is
dropped; revisit only if file size becomes a real problem in practice.

**RO and IGS write to two separate netCDF files**, confirmed (§8.4) —
`ro_<roi>_<timewindow>.nc` / `igs_<roi>_<timewindow>.nc`, mirroring
today's `ro_*.csv`/`igs_*.csv` split. No shared-file "groups" layer is
needed.

RO's Abel-inversion profiles (currently a second, index-unaligned CSV)
become a separate `abel_level` dimension (`Abel_Ne(obs, abel_level)`,
`Abel_alt_km(obs, abel_level)`, etc.) in the RO file, properly dimensioned
instead of joined only by filename string.

**CSV export is dropped entirely**, confirmed (§8.7) — no
`--also-write-csv` transition path. `export_ro_outputs`/`export_igs_outputs`
are replaced outright by `netcdf_io.write_observations(entries, path, roi=..., ...)`
and `netcdf_io.read_observations(path) -> list[ObservationEntry]`
(round-trips back into the same in-memory schema §6.1, so
`get_observation_operator` and the Kalman engine don't need to know
whether the entries came fresh from parsing or were reloaded from disk).

## 7. Diagnostic tooling (objective 4)

Proposed `diagnostics.py`, factored so each diagnostic is callable
standalone against either freshly-prepared entries or a reloaded netCDF
file (via `netcdf_io.read_observations`):

### 7.1 Geolocation / footprint map
Consolidates the map-drawing code currently duplicated in `_plot_ro_event`
and `export_igs_outputs` (Orthographic/Robinson base map, ROI circle +
Fibonacci voxels via `roi.py`) into one `plot_geolocation(entries, roi=...)`
used by both sources, replacing per-occultation PNGs with one combined
plot (with tangent tracks for RO, IPP scatter for IGS) plus an optional
per-obs detail view.

### 7.2 LOS/ROI penetration diagnostic
New — directly verifies objective 3's selection logic rather than just
trusting it: for each obs, plot the ray's altitude profile against
distance-from-ROI-center (using the same `alt_limit_km`/sampling as
`los_within_roi` in §5.2), shading the ROI radius, so it's visually
obvious how much of each LOS is inside vs. outside the ROI at each
altitude, and a histogram of "% of below-`alt_limit_km` LOS inside ROI"
across all obs to sanity-check the `fraction_required` threshold choice.

### 7.3 Measured vs. `EDPSamples`-predicted TEC
New — ties the pipeline together end-to-end: given a set of prepared
entries and an `EDPSamples` instance (built from IRI climatology or the
current ensemble mean), compute `H = EDPSamples.get_observation_operator(entry)`
per obs (or batched), predicted TEC `= H @ density.flatten()` (the same
calculation `GenericObservationOperator.forward` already does in
`Ensemble_Kalman_Engine`, reused rather than reimplemented), and plot
measured-vs-predicted scatter, residual-vs-tangent-altitude /
residual-vs-elevation, and summary statistics (bias, RMS) split by
`obs_type`. This is the most valuable sanity check for the whole
refactor, since it exercises scanning, filtering, ROI selection, netCDF
round-trip, and the observation operator in one pass.

Diagnostic outputs (maps, LOS-penetration plots, TEC-comparison figures)
are throwaway verification artifacts, not source or pipeline output —
per existing project convention, these should default to
`/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test`
(or a run-specific subfolder there), not into the repo or a scratch
directory, unless the user directs a diagnostic run's output elsewhere.

## 8. Decisions (resolved 2026-09-22)

1. **ROI containment default.** `full_los` (§5) is the new default
   selection mode. `alt_limit_km` is a required, user-supplied constant
   for now, not auto-derived from receiver altitude — receiver altitude
   varies across LEOs in a batch, so there's no single value to default
   to automatically without a first pass over the batch; deriving it from
   the batch's lowest LEO altitude is noted as a possible later
   enhancement, not built now (§5.2).
2. **Strictness.** `fraction_required` (§5.2) is a confirmed, tunable
   parameter (default `1.0`), not just a deferred escape hatch — a minimum
   fraction of the sub-`alt_limit_km` LOS must be inside the ROI, rather
   than an all-or-nothing rule.
3. **netCDF layout.** Padded `(obs, ray)` arrays, confirmed (§6.2). The CF
   ragged-array alternative is dropped.
4. **One file or two.** RO and IGS write to two separate netCDF files,
   confirmed (§6.2) — mirrors today's `ro_*.csv`/`igs_*.csv` split.
5. **`LEO`/`GNSS` key rename.** Renaming to `rec_ecef_km`/`gnss_ecef_km`
   is confirmed for the `observation_preparation` schema and netCDF
   variable names (§6.1). **New finding while scoping this, see §8a**:
   the `LEO`/`GNSS` key convention turns out to be used far more widely
   than just `EDPSamples.get_observation_operator` — this changes what
   "the rename" can mean without extra work, so §8a spells out the scope
   I'm assuming and flags one call site that will break if not also
   updated.
6. **`select_ro_occultations`'s external import.** Duplicate
   (vendor) `select_arcs_by_count_bin` into `observation_preparation`
   rather than importing it from `Austin_Demo_Code/test_param_iono.py`.
   The function is self-contained (a `list[dict]` in, a reproducible
   `zlib.crc32`-seeded subsample out, no further external dependencies),
   so vendoring is a straightforward copy.
7. **CSV export.** Dropped entirely (§6.2) — no transition-period
   `--also-write-csv` path.

### 8a. Scope of the `LEO`/`GNSS` rename — narrowed to the surviving modules

**Surviving module set, per user (2026-09-22):** once the new
data-assimilation system is complete, most subfolders in
`IonosphereTomography/` become obsolete. Only five are meant to persist:
`IRI_Sample_Inputs`, `EDPSamples`, `Parameterization`,
`Ensemble_Kalman_Engine`, `Observation_Preparation`. This resolves the
rename's scope directly: **propagate the `LEO`/`GNSS` → `rec_ecef_km`/
`gnss_ecef_km` rename only where it crosses into one of these five
folders; leave every other folder's internal convention alone.**

A repo-wide search for `['LEO']`/`['GNSS']` dict-key reads turned up the
convention in ~20 files. Sorted against the surviving set:

- **In scope** — `observation_preparation`'s own `ObservationEntry` schema
  and netCDF variable names (§6.1), and `EDPSamples.get_observation_operator`'s
  two dict-key reads (`podTc2_data['LEO']`/`['GNSS']`), because
  `Ensemble_Kalman_Engine.observation_operator.GenericObservationOperator`
  (both surviving modules) calls exactly this method with entries built by
  `observation_preparation`.
- **Out of scope, confirmed obsolete-path** — every
  `Ionosphere_Tomography_Inverter/Ionophy_Tomography_Inverter*.py`
  variant's own bespoke `get_observation_operator` method (already
  confirmed retired by [[general_enkf_module_plan]]), and essentially all
  of `Austin_Demo_Code/` (`demo.py`, `demo_group.py`, `tune_covariance.py`,
  `plotIonosphereTomography.py`, etc.) — none of these are in the
  surviving set, so their `LEO`/`GNSS` convention is left as-is
  permanently, not just for this refactor.
- **Not in the surviving set, but still a live dependency today** —
  `TEC_model/` (`podTc_file_processing.py`, `igs_tec_pipeline.py`) and
  `Abel_Inverter/lei_abel_inverter.py`. Neither is named in the survivor
  list, yet `observation_preparation` actively calls into both today for
  raw RO/IGS parsing and Abel inversion (§2.1, §9). Consistent with §9's
  existing non-goal (their internals are unchanged), this refactor treats
  them as external dependencies whose own `LEO`/`GNSS` convention is
  **not** renamed — `observation_preparation` already translates their
  raw output into its own clean-entry schema at the boundary (e.g.
  `_build_clean_ro_entry`), so the rename only has to happen on the
  `observation_preparation` side of that boundary, not inside `TEC_model`
  itself. If `TEC_model`'s parsing is later folded directly into
  `Observation_Preparation` (which would fit the original "one module
  containing all needed routines" framing in §1), its internals would
  become in-scope at that point — not assumed here.
- **Previously flagged as a required companion fix, now downgraded** —
  `TEC_model/igs_tec_pipeline.py`'s `stec_along_ray` (around line 2130)
  builds its own inline `{'LEO': ..., 'GNSS': ...}` dict and calls
  `edps.get_observation_operator(obs_dict, ...)` directly, independent of
  `observation_preparation`'s schema, so it would break once
  `get_observation_operator`'s key reads are renamed. A follow-up check
  found **`stec_along_ray` is never called anywhere in the codebase** (not
  even elsewhere in its own file) — it's already dead code today. Since
  `TEC_model` isn't in the surviving set anyway, this is now a
  non-blocking note rather than a fix required in the same edit: leave it
  broken (or fix trivially in passing, cheap either way) — it does not
  gate §10's sequencing.

### 8b. File layout and directory organization (resolved 2026-09-22)

Two further structural questions came up, both now resolved:

1. **Single file vs. multi-file for `observation_preparation` (§4).**
   `EDPSamples`/`Parameterization` are each one large `.py` file, which
   raised the question of whether `observation_preparation` should follow
   suit. **Decided: keep it split** across the files listed in §4.
   Splitting does not make importing harder for other code — `__init__.py`
   re-exports the public API regardless of internal layout, exactly as it
   already does today (`from observation_preparation import
   prepare_ro_observations` works even though that function lives in
   `prepare_ro_observations.py`, not `__init__.py`) — so the tradeoff is
   purely about development ergonomics, not import mechanics. And
   multi-file is already this package's own existing convention (today's
   `prepare_ro_observations.py`/`prepare_igs_observations.py`/`roi_tools.py`
   split), not a new one being introduced; `observation_preparation` is
   also gaining netCDF I/O and diagnostics/plotting on top of RO+IGS
   parsing, which would push a single-file version toward `edp_samples.py`
   territory (~125 KB) — worth avoiding.
2. **Move the 5 surviving modules (§8a) into a new subfolder now, to
   make the top-level directory cleaner.** **Decided: defer.** Every
   current import of `EDPSamples`, `Parameterization`,
   `Ensemble_Kalman_Engine`, `IRI_Sample_Inputs`, and
   `observation_preparation` elsewhere in the codebase (`Austin_Demo_Code`,
   test suites, etc.) uses their present top-level path; relocating them
   now would mean updating every one of those call sites for a purely
   cosmetic win, ahead of the new system actually being finished — which
   cuts against the user's own framing that the legacy folders become
   obsolete "once the new data assimilation system is completely
   developed," not before. Revisit the physical move (or archiving the
   legacy folders instead, which touches far more files but leaves the
   survivors' import paths alone) once that point is actually reached. No
   interim README/manifest index was requested either — nothing about
   directory layout changes as part of this refactor.

## 9. Non-goals

Per §8a, the surviving module set (`IRI_Sample_Inputs`, `EDPSamples`,
`Parameterization`, `Ensemble_Kalman_Engine`, `Observation_Preparation`)
bounds this refactor's touch points. Everything below stays outside that
boundary:

- `TEC_model/podTc_file_processing.py` and `TEC_model/igs_tec_pipeline.py`'s
  low-level parsing (RINEX handling, cycle-slip detection, DCB correction,
  geometry QC) and internal `LEO`/`GNSS` convention are unchanged — not in
  the surviving set, still a live dependency of `observation_preparation`
  for now (§8a). `igs_tec_pipeline.py`'s `stec_along_ray` would break under
  the rename but is confirmed dead code (never called) — noted, not fixed.
- `Abel_Inverter` logic is unchanged; only its output's *storage* moves
  from a second CSV to a netCDF dimension (§6.2). Also not in the
  surviving set, but still a live RO-path dependency, same footing as
  `TEC_model`.
- No changes to `EDPSamples`'s parameterization/observation-operator math
  itself, beyond the two dict-key reads in `get_observation_operator`
  (§8a).
- `Ionosphere_Tomography_Inverter`'s own bespoke `get_observation_operator`
  methods, `Austin_Demo_Code/`'s demo scripts, and `ConstellationSimulation/`
  keep the `LEO`/`GNSS` convention permanently — confirmed outside the
  surviving set (§8a), not just deferred for this refactor.
- No changes to `Ensemble_Kalman_Engine` beyond pointing it at
  `netcdf_io.read_observations` instead of directly calling
  `prepare_ro_observations`/`prepare_igs_observations`.

## 10. Suggested sequencing

1. **Done (2026-09-22).** `schema.py` (`ObservationEntry`, with the
   `rec_ecef_km`/`gnss_ecef_km` rename, §6.1/§8a — legacy-key translation
   lives in `ObservationEntry.from_dict`) + `netcdf_io.py`
   (`write_observations`/`read_observations`, padded `(obs, ray)` layout
   incl. the `abel_level` dimension) replacing CSV export outright.
   `EDPSamples.get_observation_operator`'s two dict-key reads renamed to
   match (the one place outside `observation_preparation` the rename had
   to land for `Ensemble_Kalman_Engine` to still run, §8a);
   `forward_model_mesh_tec`/`plot_mesh_globe` deliberately left on the
   legacy `LEO`/`GNSS` keys since their only callers
   (`TEC_model`/`Austin_Demo_Code`) are outside the rename's scope and
   would otherwise break. `prepare_ro_observations.py`/
   `prepare_igs_observations.py` updated to build entries under the new
   key names (translated right at the `TEC_model`/`igs_tec_pipeline`
   boundary) and to call `netcdf_io.write_observations` instead of the
   old CSV exporters (`_ro_csv_filename`/`_igs_csv_filename` renamed to
   `_ro_nc_filename`/`_igs_nc_filename`); the per-occultation/IPP-map PNG
   diagnostics are kept as-is (their consolidation into `diagnostics.py`
   is step 4, not this step). `TEC_model`'s `stec_along_ray` left
   unmodified as expected — confirmed dead code, and `TEC_model` isn't in
   the surviving module set anyway.

   Verified: `EDPSamples/test_edp_samples.py`'s `TestLineOfSightTEC` suite
   (5 tests) and the full `EDPSamples`/`Ensemble_Kalman_Engine` suites
   (154 tests total, jointly with `observation_preparation`'s new
   `test_netcdf_schema.py`) pass, including
   `Ensemble_Kalman_Engine/tests/test_end_to_end_real_data.py`'s real-data
   OSSE recovery check exercising the renamed
   `get_observation_operator` end-to-end. `observation_preparation/
   test_netcdf_schema.py` (new) round-trips synthetic RO+IGS entries
   through `write_observations`/`read_observations`, covering ragged ray
   counts, the Abel dimension, and `extra`-field pass-through. A scratch
   smoke test additionally exercised the real
   `_build_clean_ro_entry` → `export_ro_outputs` → netCDF path and
   `export_igs_outputs`, confirming both the renamed entries and the PNG
   diagnostics still work. `test_ro_preparation.py`/`test_igs_preparation.py`
   updated to reference the renamed keys, but remain unrunnable in this
   environment (hardcoded remote data paths, pre-existing and unrelated
   to this change) — not exercised end-to-end against real RO/IGS files
   this session.
2. **Done (2026-09-22).** Renamed `roi_tools.py`→`roi.py`,
   `prepare_ro_observations.py`→`ro_source.py`,
   `prepare_igs_observations.py`→`igs_source.py` (`git mv`, `__init__.py`
   and both ROI test scripts updated to match — nothing outside
   `observation_preparation` imported these internal module paths
   directly, so no other file needed touching). Deduped what was
   copy-pasted verbatim between the two source files: `haversine_km` +
   a new shared `build_roi_dict` helper into `roi_selection.py` (also
   §4/§5's designated home for step 3's `los_within_roi`);
   `normalize_timestamp`/`format_filename_number`/`time_window_tag` into
   `time_filter.py`. Dropped `_bearing_and_distance_km`, which turned out
   to be dead code duplicated in both files (defined, never called) —
   removing duplicated dead code is squarely within "dedup shared
   helpers," not scope creep. Vendored `select_arcs_by_count_bin` directly
   into `ro_source.py` (§8.6) — it's RO-only, so it lives with the one
   caller (`select_ro_occultations`) rather than in a shared module.
   **Deliberately did not** touch the map-plotting scaffolding duplicated
   between `_plot_ro_event` and `export_igs_outputs`'s inline block
   (base-map setup, ROI-circle + Fibonacci-voxel overlay) even though §2.4
   flagged it as duplication too — that consolidation is explicitly step
   4's job (§7.1's `plot_geolocation`, which *replaces* the per-run PNGs
   rather than just deduping their drawing code); doing a mechanical
   extraction now and a bigger rewrite in step 4 would have been wasted
   work.

   Verified: same 154-test suite (`EDPSamples` + `Ensemble_Kalman_Engine`
   + `observation_preparation`) still passes after the rename/dedup;
   `test_roi_consistency.py` (a script, not pytest) re-run directly;
   the step-1 scratch smoke test re-run against the renamed
   `ro_source`/`igs_source` modules; and a new smoke check exercising
   `select_ro_occultations`/`select_arcs_by_count_bin` directly for
   reproducibility (same `selection_key` → identical subsample) and
   nesting (bin=5 ⊂ bin=15 ⊂ all), matching the vendored function's own
   documented contract.
3. **Done (2026-09-22).** `roi_selection.py`'s `los_within_roi` (§5.2):
   samples each ray at `num_segments` points along the receiver↔transmitter
   chord (mirroring `get_observation_operator`'s own sampling, §5.1),
   masks to `alt_limit_km`, and requires ≥ `fraction_required` of the
   masked points within `radius_km`; a ray with no sample point at or
   below `alt_limit_km` at all fails closed (documented in the
   docstring), matching §5.2's "nothing to confirm containment of"
   reasoning. `build_roi_dict` extended with optional `alt_limit_km`/
   `fraction_required` (both were already reserved netCDF global attrs
   in §6.2).

   Wired into both sources as `roi_mode="full_los"`, the new default
   (§8.1), alongside the original modes (`ro_source.py`:
   `roi_mode="tangent_point"`; `igs_source.py`: `roi_mode="pierce_point"`).
   Both `prepare_ro_observations`/`prepare_igs_observations` now raise a
   clear `ValueError` if `roi_mode="full_los"` is used without
   `alt_limit_km` (§8.1's "no auto-default"). Granularity differs slightly
   by source, resolved by what each already had available: **RO** applies
   `los_within_roi` per-ray inside `_build_clean_ro_entry`, on the full
   pre-decimation `LEO`/`GNSS` geometry, ANDed into the existing
   finite-positive-TEC `valid` mask — chosen because IGS's own
   `filter_igs_epochs_by_roi` was already finer-grained than RO's
   tangent-point gate (§2.2), so extending RO to the same per-ray
   granularity is an improvement, not a new inconsistency, and it's the
   only granularity `_build_clean_ro_entry` has real per-ray geometry for.
   A structural consequence: `filter_ro_metadata`'s cheap scan-stage
   geo pre-filter (tangent point vs. `radius_km`) is **skipped** for
   `full_los` (new `apply_geo_filter` parameter) — the TEC-max tangent
   point being in the ROI is neither necessary nor sufficient for the
   real per-ray check, whose ray geometry only exists post-parse — so
   `full_los` parses every file in the time window regardless of
   location. This is a known, accepted cost, not optimized here; a
   cheaper geometric pre-filter is a possible future addition if it
   proves too slow in practice. **IGS** has no such constraint (every
   epoch is already fully parsed with real geometry before any ROI
   filtering runs today), so it gets a new sibling function,
   `filter_igs_epochs_by_full_los`, dispatched alongside the existing
   `filter_igs_epochs_by_roi` based on `roi_mode` — kept as a separate
   function rather than merged, since the two mask-computation strategies
   (single fixed-height point vs. full per-epoch ray sampling) are
   genuinely different, not the same logic wearing a different name.

   Verified: new `test_roi_selection.py` (6 tests) exercises
   `los_within_roi` in isolation with synthetic geometry -- a ray radial
   through the ROI center (fully inside), a ray toward the antipode
   (rejected), a ray that never reaches `alt_limit_km` (fails closed, not
   vacuously accepted), a symmetric center/antipode split ray where
   `fraction_required=0.9` rejects but `0.3` accepts (confirms it's a
   real tunable, not just on/off), and multi-ray vectorization. Full
   160-test suite (`EDPSamples` + `Ensemble_Kalman_Engine` +
   `observation_preparation`) still passes. A scratch smoke test exercised
   the real `_build_clean_ro_entry`/`filter_igs_epochs_by_full_los` wiring
   end-to-end with a synthetic occultation/arc mixing near-ROI and
   far-from-ROI rays, confirming `full_los` keeps only the near ones while
   `tangent_point`/`pierce_point` behavior is unchanged, plus the
   `ValueError` validation path; and a direct check that
   `export_ro_outputs`/`export_igs_outputs` correctly write
   `roi_alt_limit_km`/`roi_fraction_required` into the netCDF global
   attributes for `full_los` runs.

   `test_ro_preparation.py`/`test_igs_preparation.py` pinned to
   `roi_mode="tangent_point"`/`"pierce_point"` respectively (§10 step 5)
   rather than switched to the new default -- at the time, these were
   real-data scripts this environment couldn't run, so there was no way
   to verify a chosen `alt_limit_km`/`fraction_required` against real
   geometry. Real RO/IGS sample data became available immediately after
   (§10 step 5 below), resolving this.
4. **Done (2026-09-22).** `diagnostics.py`, all three pieces:

   **7.1 `plot_geolocation(entries, roi=None, output_path=None)`** --
   consolidates the map-drawing code duplicated between `_plot_ro_event`
   and `export_igs_outputs`'s inline block into one combined figure: RO
   tangent tracks colored by altitude, IGS pierce-point scatter, ROI
   center/voxels/radius overlay if `roi` is given. Also added
   `plot_obs_detail(entry, output_path=None)` for the "optional per-obs
   detail view" the plan called for -- the TEC-vs-tangent-height (RO) or
   TEC-vs-epoch (IGS) + Abel-Ne-vs-altitude panels the old per-occultation
   PNGs carried, now callable on demand per entry rather than
   auto-generated for every run.

   **Rewired `export_ro_outputs`/`export_igs_outputs`** to call
   `plot_geolocation` once instead of their own bespoke plotting --
   `_plot_ro_event` (RO's per-occultation 3-panel PNG) and IGS's inline
   map + "TEC vs. arc number" panel are deleted outright, not just
   deduped-in-place. This is a real, intended behavior change (one
   combined map instead of N per-occultation PNGs), confirmed against
   real data below. Cleaned up the now-unused `matplotlib`/`cartopy`/
   `ECEFtolla`/`roi.py`-geometry imports this left behind in both files.

   **7.2 `plot_los_penetration(entries, center_lat, center_lon, radius_km,
   alt_limit_km, ...)`** -- altitude-vs-distance-from-center scatter for
   every sampled ray point (shading the ROI radius, marking
   `alt_limit_km`), plus a histogram of `pooled_fraction_inside_roi`
   (new, separately exported and tested) -- one number per observation
   (pooling all of that observation's rays' sub-`alt_limit_km` points
   together), matching §7.2's "across all obs" framing, as opposed to
   `los_within_roi`'s per-ray threshold decision. Refactored
   `roi_selection.los_within_roi` to share a new `sample_ray_geodetic`
   helper with this diagnostic, rather than duplicating the ray-sampling
   math a second time.

   **7.3 `plot_tec_comparison(entries, edp_samples, ...)`** -- the
   measured-vs-`EDPSamples`-predicted TEC check: predicted TEC
   `= EDPSamples.get_observation_operator(entry.to_operator_dict()) @
   edp_samples.edps.mean(axis=-1).reshape(-1)` (H already carries the
   1e16 TECU conversion internally, confirmed by reading
   `get_observation_operator`'s own code -- no extra factor needed),
   using the ensemble-mean density per §7.3's "IRI climatology or the
   current ensemble mean" framing; returns `(figure_or_path, stats)` with
   per-`obs_type` bias/RMS.

   Verified: new `test_diagnostics.py` (8 tests) -- `plot_geolocation`/
   `plot_obs_detail`/`plot_los_penetration` produce valid figures for
   synthetic RO+IGS entries; `pooled_fraction_inside_roi` checked against
   known near-center vs. antipodal synthetic geometry and the
   never-reaches-`alt_limit_km` None case; `plot_tec_comparison` checked
   *quantitatively* against the same analytic uniform-density
   radial-ray closed form `EDPSamples/test_edp_samples.py` uses for its
   own H-matrix tests (bias/RMS ≈ 0 to within 0.05 TECU) -- this caught a
   copy-paste error in the test's own hardcoded expected value (a stale
   constant from a differently-configured altitude range) before it
   shipped, exactly the kind of mistake a purely "does it run" test would
   have missed. A new `observation_preparation/conftest.py` (matching the
   existing `EDPSamples/conftest.py`/`Parameterization/conftest.py`/
   `Ensemble_Kalman_Engine/tests/conftest.py` shim pattern) makes
   `edp_samples` importable for this suite.

   Then verified against the real RO/IGS data from step 5: re-ran both
   `test_ro_preparation.py`/`test_igs_preparation.py` end-to-end --
   `export_ro_outputs`/`export_igs_outputs` now produce exactly one
   `*_geolocation.png` + one `.nc` per run (down from 15 separate PNGs
   for RO, confirmed by directly inspecting the output directory before
   vs. after); both combined maps render correctly (RO tangent tracks /
   IGS pierce points clustered around Tromsø, ROI circle, voxel grid).
   `plot_los_penetration` against the real RO entries shows the expected
   "hook" shape (each ray dipping to a low tangent altitude near the ROI
   before sweeping out to its GNSS transmitter at ~20,000+ km altitude far
   away) and a sensible per-occultation fraction-inside-ROI histogram.
   `plot_tec_comparison` smoke-tested against a real, netCDF-loaded
   `EDPSamples` file (`TestCode/EDPSam_Point.nc`) plus the real RO
   entries -- the large residuals in that particular run are expected
   (that file's IRI climatology is for an unrelated location, ~1500 km
   from the ROI, not a bug) and just confirm the real-object integration
   path (dtypes/shapes from a netCDF-loaded `EDPSamples`, not a
   Python-constructed one) has no surprises the purely-synthetic unit
   test wouldn't catch.

   Full suite: 168/168 (160 + 8 new `test_diagnostics.py`).
5. **Done (2026-09-22), and expanded well beyond its original scope once
   real data arrived.** `test_roi_consistency.py`/`test_igs_roi_voxels.py`
   re-run directly (script-style, no changes needed beyond step 2's import
   fix) -- pass.

   The user provided real RO/IGS samples matching the two real-data
   scripts' existing settings exactly (Tromsø ROI, 2025-11-18 10:00-11:00):
   `RO_Data` (15 podTc2 files) and `RINEX_Cache` (TRO1/WUTH RINEX + a
   BRDC mixed-nav + a CAS DCB file) under
   `/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/`.
   Pointed both scripts' hardcoded paths there (were remote/nonexistent
   placeholders) and their output at
   `.../Runs/Tomography_Test/Claude_Test/{RO,IGS}_step5` per
   [[tomography-test-output-folder]], and actually ran the full pipeline
   end-to-end for the first time this session had real data to do so.

   **RO**: `test_ro_preparation.py` passes outright -- all 15 occultations
   accepted, all 15 Abel inversions succeed, and the produced `.nc`
   round-trips correctly through `read_observations`. `roi_mode="full_los"`
   also checked directly against this real geometry (not just synthetic,
   §10 step 3's caveat): at `alt_limit_km` = 700/800/900 km it keeps
   12/11/9 of the 15 occultations -- correctly *more* restrictive as the
   altitude ceiling rises (checking containment over a wider altitude
   range pulls in more of each ray's geographically-spread high-altitude
   end), confirming `los_within_roi`'s monotonic behavior holds on real
   occultation geometry, not just the hand-built synthetic cases in
   `test_roi_selection.py`.

   **IGS**: running `test_igs_preparation.py` against the real cache
   surfaced two genuine, pre-existing bugs that real-data testing was
   the first thing ever to exercise:
   - `prepare_igs_observations` called
     `process_igs_station(..., tlim=(...))`, but that function never had
     a `tlim` parameter (only the inner `IGSTECPipeline` class does, and
     `process_igs_station` never forwarded it) -- every call raised
     `TypeError`, silently caught by the per-station `except Exception`,
     so `prepare_igs_observations` had apparently never returned a single
     real arc before this. Fixed by dropping the dead `tlim` kwarg
     entirely: correctness is unaffected since `filter_igs_by_time`
     already re-applies `[start_time, end_time)` downstream on whatever
     `process_igs_station` returns -- this only costs a little wasted
     work (arcs outside the window get generated before being discarded),
     not correctness. `TEC_model` itself is untouched (§9's non-goal
     still holds; this was fixable entirely on the `observation_preparation`
     side of the boundary).
   - `collapse_igs_arc_to_central_epoch`'s `_IGS_ARC_PER_EPOCH_FIELDS`
     tuple didn't include `ipp_distance_km` (a diagnostic
     `filter_igs_epochs_by_roi`/`_by_full_los` attach, not
     `igs_obs_to_clean_entry`), so `epoch_mode="center"` collapsed every
     other per-epoch field to length 1 but left this one at its original,
     longer length -- an internally inconsistent entry that then broke
     `netcdf_io.write_observations`'s scalar/array classification
     (`_classify_extra`) with a cryptic `TypeError`. Fixed the field
     list (root cause), and separately hardened
     `write_observations`/`_classify_extra`: the old rule required
     `n_rays > 1` to ever classify a field as "ray" data, which
     misclassifies a genuine per-ray field as "scalar" whenever every
     entry in a batch happens to have exactly 1 ray (true for any
     `epoch_mode="center"` run) -- fixed to classify on shape match alone;
     and the scalar-write path now raises a clear, actionable `ValueError`
     naming the offending field and entry if a "scalar" field ever turns
     out not to reduce to one value, instead of a cryptic low-level
     `TypeError`, since `extra` is deliberately an open-ended per-source
     catch-all (`schema.py`) that `netcdf_io.py` can't fully anticipate.

   Also added `local_obs_by_station`/`local_nav`/`local_dcb` pass-through
   parameters to `prepare_igs_observations` (forwarding parameters
   `process_igs_station` already had) -- needed because no CDDIS/Earthdata
   `~/.netrc` is configured in this environment, so automatic fetching
   failed; a generically useful addition for anyone working from an
   already-fetched RINEX cache, not a sandbox-specific hack. (One
   incidental cleanup: an earlier failed attempt, before `local_nav`
   fetching was wired in, left corrupt placeholder files -- HTML error
   pages saved under `.gz` names -- in the user's `RINEX_Cache`; these
   were deleted since they were artifacts of this debugging, not part of
   the original sample data.)

   With those fixes: `test_igs_preparation.py` passes outright -- 23 arcs
   across both stations (`TRO1`, `WUTH`), and the produced `.nc` round-trips
   correctly. `roi_mode="full_los"` checked against the same real data
   (`min_valid_epochs=1`, so 36 arcs pass `pierce_point`): at
   `alt_limit_km` = 700/800/900 km, `full_los` keeps 31/30/29 -- the same
   monotonically-more-restrictive-as-altitude-rises pattern seen for RO,
   confirmed on real IGS geometry too, not just RO's.

   Full 160-test suite still green after both bug fixes.
