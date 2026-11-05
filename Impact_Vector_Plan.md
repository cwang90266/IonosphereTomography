# Impact Vector Computation Plan

Source note: `/Users/cwang/Documents/Consulting/PlanetIQ/Presentation/RO_Impact.pdf`
("Evaluation of Impact of Individual RO on EDP Changes"). Goal: for each RO
occultation that enters an assimilation batch, compute how much of the
analysis EDP change at an arbitrary query point's vertical column (ISR
sites being the first, not only, use case -- Decision 6) is attributable
to *that* RO, for both the linear (`raw`) and nonlinear (`ANCHOR`/`PCA_*`/
`density_10ex`) parameterization styles.

This plan grounds the note's math in what the actual driver code does --
in several places the real code is *not* the note's idealized formula, and
the plan is written around the real code, with the discrepancies called
out explicitly rather than smoothed over.

**All 6 open questions from the first draft were answered by the user on
2026-11-05 -- see Section 10 for the resolved decisions; the rest of this
document (Sections 1-9, 11) has been updated in place to match them,
rather than left as stale "leaning toward" language next to a separate
update block, since all 6 answers arrived in one pass immediately after
the plan was first written.**

## 1. The note's math, briefly

- Model EDP is a vector `v` in `R^N` (full grid). `f_EDP(lat, lon, alt)` is
  a length-`N` interpolation-weight vector; `EDP(lat,lon,alt) = f_EDP^T · v`.
- A linear filter: `Δv = K(u - H·v_forecast)`. `u` stacks all RO/IGS
  observations; `K` splits into column-blocks `K = [K_1,...,K_L]` matching
  `u`'s per-RO partition `u = [RO_1;...;RO_L]` (and `H` into matching
  row-blocks `H_i`). This column/row partition is always exact (no
  independence assumption): `K(u - Hv) = Σ_i K_i(RO_i - H_i v)`.
- Impact vector: `I_i(lat,lon,alt) = f_EDP(lat,lon,alt)^T · K_i(RO_i - H_i·v_forecast)`.
- Nonlinear/iterative case: `Δv^j = K^j(u - H^j·v_forecast^{j-1})` per
  Gauss-Newton iteration `j`; `I_i = f_EDP^T · Σ_j K_i^j(RO_i - H_i^j·v_forecast^{j-1})`.
  The note explicitly flags: *"current tomography inversion code may not
  keep the intermediate updates Δv^j"* -- new bookkeeping is required.

## 2. Reconciling the note's math with what `AnalysisEngine.analyze()` actually does

(`Ensemble_Kalman_Engine/analysis_engine.py` -- read directly, not
summarized, since this is the crux of the whole feature.)

**Finding 1 -- the per-iteration step is MAP/Gauss-Newton anchored at a
fixed `x_p`, not a literal `K^j(u - H^j v^{j-1})`:**

```
x_p, y_p = x_i, y_i                       # fixed anchor for the whole loop
x_i_current, y_i_current = x_i, y_i       # moving linearization point
for it in range(max_iterations):
    W_hat = obs_operator.linearized(x_i_current)      # Jacobian at current iterate
    K_i   = kalman_gain(S, W_hat @ S, R, ...)          # explicit (n_state, n_obs) matrix
    pseudo_residual = (y_obs - y_i_current) + W_hat @ (x_i_current - x_p)
    x_full_next = x_p + K_i @ pseudo_residual          # undamped GN/MAP point

    # backtracking: damp FROM x_i_current TOWARD x_full_next (not from x_p)
    x_candidate = x_i_current + alpha * (x_full_next - x_i_current)
    ... accept if residual improves, else shrink alpha ...

    # centering=="nearest_analog" (the DEFAULT for every nonlinear style):
    # may REPLACE x_candidate with a raw forecast ensemble member if it's
    # closer to the observations than the (possibly damped) GN point.
    x_i_current = x_candidate
```

The note's slide-6 formula (`P_{i+1} = P_i + G_i(u - H(P_i))`, iterated
literally) is *not* what the code does -- doing that literally loses the
prior's pull once the iterate wanders from the prior mean, which this
project already found empirically worse (see the code comment at
`analysis_engine.py:198-219`). The code instead anchors every iteration's
correction at the *original* `x_p`, matching the standard Gauss-Newton
MAP-with-Gaussian-prior scheme (Bell & Cathey 1993). This is still
**linear in the residual** at each iteration, so the goal (split the
residual by RO, propagate the split through) survives -- but the
*recursion* across iterations is different from the note's, and matters
for the decomposition (Finding 3 below).

**Finding 2 -- `K_i` is already an explicit `(n_state, n_obs)` matrix**
(`kalman_gain()`, `Ensemble_Kalman_Engine/kalman_core.py:62-116`): no need
to reconstruct it from ensemble-anomaly algebra. This simplifies the
implementation considerably: the per-RO split is a direct column-slice of
an already-materialized matrix, via the observation row-ranges
`CycleBatch.entry_ray_ranges` (`Assimilation_Cycle/observation_stream.py:105-127`)
already tracks which rows of `y_obs`/`H` belong to which RO/IGS entry.

**Finding 3 -- backtracking damping blends `x_i_current` and
`x_full_next`, not `x_p` and `x_full_next` -- so the per-RO contribution
of iteration `j` *decays* under later iterations' damping, it isn't just
individually scaled.** Let `D^j = x^j - x_p` (total accumulated change
from the anchor through iteration `j`). Substituting the actual damping
formula:

```
D^j = (1 - alpha^j)·D^{j-1} + alpha^j·(K^j @ r^j)
```

where `r^j` is that iteration's `pseudo_residual`. Expanding the
recursion, the FINAL state's deviation from `x_p` is a *decayed weighted
sum* of every iteration's own proposal, not a flat sum:

```
D^J = Σ_{k=1}^{J} w_k · (K^k @ r^k),     w_k = alpha^k · Π_{m=k+1}^{J} (1 - alpha^m)
```

Since `r^k` splits per RO exactly (`r^k = Σ_ℓ r^k_ℓ` via the row
partition) and `K^k @ r^k = Σ_ℓ K^k[:, idx_ℓ] @ r^k_ℓ` (exact column
partition), this still decomposes additively per RO -- **but the plan
must implement the `w_k` decay weights correctly, not assume a flat sum
across iterations.** This is a real correction to my own first-pass
derivation of this plan, caught only by reading the exact damping formula
rather than assuming the note's literal wording.

**Finding 4 -- `nearest_analog` centering (the current project-wide
default for every nonlinear style, `AnalysisConfig.centering="nearest_analog"`,
`analysis_engine.py:74`) breaks the decomposition twice over, and is NOT a
rare edge case:**

1. **At the very start of `analyze()`** (`analysis_engine.py:174-185`):
   when `centering=="nearest_analog"`, the anchor `x_p` itself is set to
   `X_f[:, i_star]` -- a raw forecast ensemble member closest to `y_obs`
   -- **not the forecast ensemble mean** `x_mean`. The note's `v_forecast`
   should mean the true ensemble mean (matching what
   `Assimilation_Cycle/package_run.py`'s `forecast_mean_density` already
   computes); the code's `x_p` is a *different* quantity when this
   centering is active. The gap `(x_p - x_mean)` is itself unattributable
   to any RO (it's a discrete nearest-neighbor choice, not a linear
   function of the residual), even before any iteration runs.
2. **Mid-loop** (`analysis_engine.py:287-296`): at any iteration, the
   (possibly-damped) GN candidate can be discarded and replaced by a raw
   ensemble member if that member is closer to `y_obs`. When this fires,
   the recursive `D^j` formula above does not apply for that step at
   all -- `x_i_current` is simply reset to a value with no linear
   dependence on that iteration's residual, and (because `x_p` never
   moves) the entire *prior* accumulated `D^{j-1}` becomes irrelevant to
   the final state too, discarded in favor of the raw member.

   When `centering=="mean"` (an existing, already-implemented alternative,
   `analysis_engine.py:180-183` / the `else` branch at 287-296) is used
   instead, *neither* of these resets ever happens: `x_p = x_mean` exactly,
   and every iteration's accepted point is always the (possibly damped)
   Gauss-Newton candidate. The whole trajectory is then exactly
   decomposable via Finding 3's weighted sum, with no gaps.

   **This is the single biggest design decision in this plan** --
   resolved by Decision 1 (Section 10): this plan builds *only* the
   `centering="mean"` diagnostic companion pass, explicitly accepting an
   **approximate** answer to "which RO was most impactful" (the user's
   own characterization) rather than attempting a partial-attribution
   accounting of the actual production `nearest_analog` trajectory. This
   simplifies the rest of the plan considerably: every subsequent section
   below assumes `centering="mean"`, so Finding 4's two failure modes
   never occur in the shipped feature, and the exactness identity (Section
   3) holds unconditionally rather than "only for whatever portion of the
   trajectory had no resets."

**Finding 5 -- `decode()` is purely nonlinear; a separate, per-style
Jacobian already exists.** `GenericObservationOperator.decode()`
(`Ensemble_Kalman_Engine/observation_operator.py:79-109`) is a forward
nonlinear call (`parameterization.get_density`), no linearization. The
Jacobian used for `W_hat` above comes from
`EDP_Parameterization.get_Jacobian()` (`Parameterization/Parameterization.py:1238`,
dispatching on each style's registered `jacobian` kind -- `'diagonal'`
for `raw`/`density_10ex`, a custom callable for `ANCHOR`, `'block_diagonal'`/
`'full'` for the PCA styles), composed with the *ray* observation operator
via `jacobian_utils.compose_observation_jacobian()`
(`Ensemble_Kalman_Engine/jacobian_utils.py`) to build `W_hat = H_grid @ J_decode`.
For the impact vector we need the *same* `J_decode`, but composed with
`f_EDP(query)` (a point/column interpolation operator) instead of
`H_grid` (a ray integration operator) -- structurally the same
composition, just a different left-hand operator. Needs verifying that
`compose_observation_jacobian()` is generic enough to accept an arbitrary
sparse matrix in place of `H_grid`, or needs a small generalization.

## 3. The decomposition this plan actually computes

For one batch, one style, one iteration `k` (`k=1` only for `raw`/linear,
`k=1..J` for the iterated nonlinear styles), and one query point/altitude:

```
a^k(query) = f_EDP(query)^T @ J_decode(x_i_current^k)      # (1, n_state), via Finding 5
g^k(query) = a^k(query) @ K^k                                # (1, n_obs_in_batch) -- cheap, one row-vector times the already-explicit K^k
I_ℓ^k(query) = w_k · g^k(query)[idx_ℓ] · r^k[idx_ℓ]           # scalar, idx_ℓ = entry_ray_ranges for RO ℓ
I_ℓ(query) = Σ_k I_ℓ^k(query)                                  # total impact of RO ℓ at this query point
```

`w_k` is the decay weight from Finding 3 (`w_k=1` for `raw`'s single
undamped step). `a^k(query)` is computed via a **row-restricted**
Jacobian evaluation/VJP -- `f_EDP(query)` is sparse (a handful of mesh/
altitude weights from `EDPSamples.interp()`), so this never requires
forming the full `(N, n_state)` decode Jacobian; only the few output rows
`f_EDP(query)` actually touches. For a whole ISR vertical column (many
altitudes, one lat/lon), batch all target altitudes into one small matrix
`F_EDP(site)` and compute `A^k(site) = F_EDP(site) @ J_decode(x_i_current^k)`
once per iteration, then `G^k(site) = A^k(site) @ K^k` once per iteration
(not once per RO) -- the per-RO cost is then just a cheap dot product
over that RO's own (small) ray count.

**Correction found during implementation (2026-11-05), replacing this
section's original "holds unconditionally, exactly" claim**: there are
two genuinely different properties, and only one of them is actually
exact for a nonlinear style.

1. **Partition exactness** (always exact, every style): splitting the
   observations across multiple ROs and summing their impacts reproduces
   *exactly* what a single undivided "RO" covering the same observations
   would report -- `Σ_ℓ I_ℓ(query) == I_combined(query)`. This is just
   the column/row partition identity from Section 1, and needs no
   approximation at all; confirmed by a dedicated test
   (`test_impact_vectors.py::TestImpactVectorExactness::test_partition_exactness_two_way_split_matches_single_combined_entry`,
   parametrized across `raw`, nonlinear-no-damping, and nonlinear-with-
   damping).
2. **Physical-change accuracy** (exact only for a *linear* decode):
   whether `Σ_ℓ I_ℓ(query)` matches the *actual* decoded change,
   `decode(x_final)[query] - decode(x_mean)[query]` (the existing
   `analysis_mean_density - forecast_mean_density` quantity from the diff
   plots added 2026-10-01). For `raw` (`to_density` is the identity),
   these two properties coincide, so the original "exact" claim was
   correct there. For a genuinely nonlinear decode (`density_10ex`/
   `ANCHOR`/PCA styles), each iteration's contribution is only ever a
   **first-order (Jacobian-at-that-iterate) approximation** of that
   iteration's true nonlinear change -- found empirically while writing
   the exactness test: forcing the same "exact" comparison for
   `density_10ex` gave a 27% relative error at a moderate perturbation
   scale, shrinking to 2% and then 0.07% as the scale shrunk 10x each
   time, the unmistakable signature of a linearization remainder, not a
   bug (confirmed by `test_nonlinear_style_approximation_error_shrinks_with_perturbation_scale`).
   This isn't a new weakness introduced by this plan -- it's inherent to
   using per-iteration local Jacobians at all for a nonlinear decode
   (exactly what the source note's own nonlinear-case formula already
   does), just not something the original Section 3 draft stated
   correctly.

Practical takeaway: trust the per-RO *ranking* (which RO mattered most,
and in which direction) for every style -- that only relies on property
1. Treat the summed impact's absolute magnitude, for a nonlinear style,
as a first-order estimate of the real decoded change, most accurate when
the batch's Gauss-Newton steps stayed small (few iterations, little or no
backtracking) -- not an exact accounting the way it is for `raw`.

Separately, unrelated to the above: this is the decomposition of a
`centering="mean"` *diagnostic* pass over the same forecast/observations/
`R` (Decision 1), not a reconstruction of whatever the production
`nearest_analog` run actually did -- so it answers "which RO would be
most impactful under plain Gauss-Newton," not "exactly how the production
analysis arose." That approximation was already explicitly accepted by
the user and is orthogonal to the linear-vs-nonlinear-decode one above.

## 4. Linear case (`raw` style)

Simplest case: `linearization=="linear"` is a *separate, non-looped*
branch (`analysis_engine.py:190-195`), always exactly one step,
`x_final = x_i + K_final @ (y_obs - y_i)`. No damping, no `nearest_analog`
mid-loop resets possible inside this branch -- **but the *initial*
centering check (Finding 4, point 1) still applies**: if
`centering=="nearest_analog"` at the top of `analyze()`, `x_i` here is
still a raw ensemble member, not `x_mean`, so the same initial-anchor gap
exists even for `raw`. `raw`'s `J_decode` is `'diagonal'` kind (trivial,
elementwise-scaled `f_EDP`), so this case is the cheapest to implement and
the natural first target.

Note `raw` is not currently one of `CycleConfig.default_styles()`
(currently 4 styles, all nonlinear/iterated) -- real-data verification of
the linear case needs a small dedicated run with `style="raw"` explicitly
requested.

## 5. Nonlinear case (`ANCHOR`/`PCA_*`/`density_10ex`)

Up to `max_iterations=8` (default) Gauss-Newton steps per batch, re-
linearized every iteration, anchored at `x_p` (Finding 1), with the decay-
weighted accumulation of Finding 3, and the `nearest_analog` reset
exposure of Finding 4. `J_decode`'s structural kind varies by style
(diagonal for `density_10ex`, custom for `ANCHOR`, block-diagonal/full for
the PCA styles) -- the row-restriction trick for `a^k(query)` needs
confirming against each kind's actual representation (likely easy for
diagonal/block-diagonal; for `'full'`, if the full dense Jacobian is
already materialized elsewhere for `PCA_3D_10ex`, row-slicing it is free;
if not, this needs its own small addition).

## 6. New instrumentation needed (a third flagged architecture exception)

This project's standing rule is "`Assimilation_Cycle` adds no logic to
the five survivor modules" with two existing, deliberate, documented
exceptions (`GeneralEnKFDriver`'s split, `EDPSamples.select_map_projection`).
This feature needs a third: **`AnalysisEngine.analyze()` must report its
own per-iteration internals to an optional caller-supplied hook**, since
`K^k`/`W_hat`/`x_p`/`alpha`/the centering decision are computed and
discarded inside that function today (`diag.K_mean`/`diag.x_mean_a` only
keep the *final* iteration, confirming the note's flagged gap).

Proposed: a new optional `on_iteration` parameter to `analyze()`
(default `None`, zero behavior change when omitted), called once per
accepted iteration (and once for the `"linear"` branch's single step)
with: iteration index, `x_i_current` (before), `K^k`, `W_hat^k`, `x_p`,
`r^k` (the pseudo-residual), `alpha` actually used, and whether this step
was a `nearest_analog` replacement. Threaded through
`GeneralEnKFDriver.assimilate_one_cycle` → `cycle_driver.run_cycle`'s
existing `on_batch` plumbing, mirroring exactly how `on_batch` itself is
already threaded today.

**Critically, the hook must let the caller consume and discard
immediately** -- the whole point is to never hold `K^k` (an
`(n_state, n_obs)` dense array, potentially ~100MB+ at full production
scale) for more than one iteration at a time. The caller (the new
`ImpactVectorAccumulator`, Section 7) projects through `f_EDP(query)`
inside the callback and keeps only a tiny running total
(`n_RO × n_sites × n_altitudes` scalars), not the raw matrices.

## 7. New `Assimilation_Cycle/impact_vectors.py`

- `QueryPoint(name, lat, lon, altitudes)`: the generic query-point type
  (Decision 6) -- an ISR site is just one way to build a list of these
  (`QueryPoint(isr.station.name, isr.station.lat, isr.station.lon, model_altitude)`,
  reusing `isr_comparison.IsrStation`/`load_isr_dataset`), but the API
  accepts arbitrary user-supplied `(name, lat, lon)` points too, not only
  ISR-file-derived ones. **`altitudes` defaults to `edp_samples.altitude`
  (the full model grid, 82 levels) -- Decision 3 -- not the ISR
  instrument's native (sparser) gates**, even when the query point comes
  from an ISR file; a separate, later step (Section 8's alignment ranking)
  is responsible for reconciling the two grids when an ISR comparison is
  actually wanted.
- `ImpactVectorAccumulator(edp_samples, query_points, entry_ray_ranges)`:
  precomputes each query point's sparse `F_EDP(point)` once (via
  `EDPSamples.interp()`, following the existing pattern in
  `output._interpolate_field_at_latlon`, extended to many altitudes in one
  call per the research finding in Section 2).
- `.on_iteration(...)` matching the new hook signature: computes
  `A^k(point) = F_EDP(point) @ J_decode(x_i_current)` and
  `G^k(point) = A^k(point) @ K^k` once per query point, then for each RO
  in `entry_ray_ranges`, accumulates `w_k · G^k(point)[:, idx_ℓ] @ r^k[idx_ℓ]`
  into a running `(n_altitudes,)` array keyed by `(entry.label, point.name)`.
  Since Decision 1 fixes `centering="mean"`, `w_k` is always the plain
  decay weight from Finding 3 with no resets to track -- no
  `nearest_analog`/attributable-fraction bookkeeping needed in the shipped
  feature (Finding 4's complications only mattered for the no-longer-
  pursued production-trajectory mode).
- `.finalize()`: returns `{(entry_label, point_name): impact_profile}`,
  one `(n_altitudes,)` array per RO per query point. Stays purely a
  function of the EDP-profile state (Decision 2) -- no attempt to reduce
  to scalar ionospheric parameters.

**Wiring is a separate diagnostic pass, not a hook on the production
run.** Because Decision 1 commits to `centering="mean"` specifically (not
whatever centering the production run used), computing impact vectors
means running a *second*, `centering="mean"`-forced `AnalysisEngine.analyze()`
call per batch per style, over the exact same forecast ensemble/
observations/`R` the production batch already used -- purely read-only,
its result discarded except for what the accumulator's `on_iteration`
extracts; it never feeds back into or alters the production analysis.
`package_run.py` (same place `IsrComparisonCollector`/`cross_style_data`
already hook in) gains a new opt-in `CycleConfig.compute_impact_vectors: bool = False`
and `CycleConfig.impact_vector_query_points: list[tuple[str, float, float]] | None = None`
(explicit points; when `cfg.isr_file_path` is also set, its station is
added to the list automatically). When enabled, for each batch/style,
`run_package` builds the accumulator and issues the extra `centering="mean"`
`analyze()` call with `on_iteration=accumulator.on_iteration`. This adds
real compute cost (one more full Gauss-Newton solve per batch per style,
the same cost class as the original solve, not a multiplier on top of
everything else) -- acceptable as an opt-in diagnostic, not something to
enable by default.

## 8. Output / visualization

- `plot_impact_vector_heatmap(point, impacts_by_ro, altitude)`: ROs
  (sorted by total `|impact|`, integrated over altitude -- Decision 2
  keeps the profile shape, so this is the natural default single-number
  sort key, with the full profile still visible in the heatmap itself) ×
  altitude heatmap, diverging colormap, reusing the same symmetric-
  `vmin`/`vmax`-around-zero convention just added to
  `EDPSamples.plot_horizontal_field` for the analysis-minus-forecast diff
  plots (2026-10-01) -- though this is a plain 2-D array plot
  (`pcolormesh`), not a map.
- `plot_impact_vector_ranking(point, impacts_by_ro)`: bar chart, total
  `|impact|` per RO -- directly answers the note's own motivating question
  ("which RO played the most critical role").
- Cross-style variant: same ranking, one bar-group per style, at a given
  query point -- matches this project's established per-style + cross-
  style convention (`cross_style_data`, `isr_comparison.py`'s own per-
  style/cross-style pairs).

**New: ranking by direction relative to ISR, not just magnitude
(Decision 5).** The PDF's own background slide asks not just "which RO
mattered most" but "whether these changes improve agreement to EDP
measured by ISR" -- this ranking answers that directly. For a query point
that has a matching ISR profile (nearest-in-time, reusing
`isr_comparison.nearest_isr_profile`), define the *correction direction*
`d = M - F` where `F` is the forecast mean profile on the model's 82-level
grid (already computed for the existing diff plots) and `M` is the ISR-
measured profile **interpolated onto that same 82-level grid** -- the
reverse direction from what `isr_comparison.py` does today (it interpolates
the *model* profile down onto the ISR instrument's native, sparser gates,
to avoid implying any extrapolation of real ISR data). Interpolating `M`
onto the model grid instead needs its own justification: it's fine
specifically *for this ranking*, since `d` is only ever used inside a dot
product against `I_ℓ` (both already native to the model grid) and is
masked to the altitude band actually covered by ISR's native gates
(`isr.altitude.min()..max()`) -- no value of `M` outside that band is ever
synthesized or displayed on its own. Mechanically this reuses
`isr_comparison._interp_log10` with its arguments swapped (that helper
already does log-space `np.interp` with `nan` outside range in either
direction; today's call site just happens to call it one way). Then for
each RO: `projection_ℓ = <I_ℓ, d_masked>` (a signed scalar) -- positive
means RO `ℓ`'s impact pushes the analysis *toward* the ISR-measured
profile over the masked band, negative means *away* from it. Rank by
`projection_ℓ` (most positive = most helpful toward ISR truth, most
negative = most harmful), reporting `||I_ℓ||` (plain magnitude) alongside
it so "large but unhelpful" and "small but unhelpful" stay distinguishable
rather than collapsed into one number.
`plot_impact_vector_isr_alignment(point, impacts_by_ro, forecast_profile, isr_density, isr_altitude, model_altitude)`:
a bar chart (signed, diverging color by sign) of `projection_ℓ`, this
ranking's dedicated visualization -- lives in `impact_vectors.py`
(Decision 5's first clause keeps the *computation* standalone; this
function takes an ISR profile as an explicit argument rather than
`impact_vectors.py` importing `isr_comparison.py` itself, so the
dependency only exists where it's actually used).

**ISR-aligned heatmap (confirmed with the user 2026-11-05)**: the plain
heatmap above colors by the RO's *raw* impact sign (density increased vs.
decreased); `plot_impact_vector_isr_aligned_heatmap(point, impacts_by_ro,
forecast_profile, isr_density, isr_altitude, model_altitude)` recolors
the *same* `(RO x altitude)` grid by alignment with ISR instead, so the
heatmap's color meaning matches the ranking bar chart's ("one consistent
color always means toward ISR, the other always means away from ISR, at
every altitude, for every RO") rather than requiring the viewer to
mentally cross-reference the impact heatmap against a separate ISR-vs-
forecast plot. Per-cell value: `I_ℓ(h) · sign(d_masked(h))` -- the chosen
convention keeps `I_ℓ`'s own units and magnitude (m⁻³) unchanged, only
flipping the sign locally wherever the ISR-correction direction `d`
itself is negative, so a large push in the wrong direction is visually as
prominent as a large push in the right one (the alternative considered --
coloring by the full product `I_ℓ(h)·d(h)`, whose altitude-sum exactly
reproduces the bar chart's `projection_ℓ` -- was not chosen, since its
units aren't a density and altitudes with a large ISR-forecast gap would
visually dominate even where a given RO's own impact is small). Cells
outside `d`'s masked (ISR-covered) altitude band are grayed out/excluded,
same masking rule as the ranking bar chart -- never silently extrapolated.

## 9. Testing strategy

Following this project's standing convention (synthetic unit tests first,
then real-data smoke verification):

**Tests 1-3 below were redesigned during implementation** once building
them surfaced the property-1-vs-2 distinction in Section 3's correction
-- comparing against the true decoded change (as originally specified)
only makes sense as an *exact* check for `raw`; for a nonlinear style it
needed to become a convergence check instead, with the real exactness
claim moved to the partition identity (Section 3, property 1), checked
against a *second*, differently-split run rather than against the
decoded ground truth. Implemented as
`Assimilation_Cycle/tests/test_impact_vectors.py::TestImpactVectorExactness`.

1. **Partition-exactness unit test, parametrized across `raw`, a
   nonlinear style with no damping, and a nonlinear style with real
   backtracking damping**: splitting the same synthetic observations
   across 2 fake RO entries and summing their impacts must exactly match
   a second run treating them as one combined entry (same seed, so
   bit-identical trajectories) -- true for every style, no approximation
   (Section 3, property 1).
2. **Decoded-change accuracy, `raw` specifically**: confirms property 1's
   always-exact total *also* matches the real
   `decode(x_final) - decode(x_mean)` change there, since `raw`'s decode
   is the identity (property 2 coincides with property 1 only for a
   linear decode).
3. **Nonlinear-style convergence test**: confirms the (expected,
   *not* exact) gap between property 1's total and the true decoded
   change shrinks monotonically as the problem is scaled toward more
   linear (smaller forecast spread / smaller true departure) -- catches a
   real bug (a flat or growing error) without over-claiming exactness
   property 2 never actually has for a nonlinear decode.
4. **`on_iteration` hook regression test** (in `Ensemble_Kalman_Engine`):
   confirm passing `on_iteration=None` (the default) changes nothing about
   `analyze()`'s existing behavior/outputs -- this is the new exception to
   the "no logic added to survivor modules" rule, so it needs its own
   explicit zero-regression proof, same as the other two exceptions got.
5. **ISR-alignment unit test** (covers both the ranking bar chart and the
   aligned heatmap): a hand-constructed toy case with known
   `forecast_profile`/`isr_density`/two ROs' impact profiles, engineered
   so the correct sign of `projection_ℓ` is known by inspection (one RO
   clearly pushing toward `M`, one clearly away) *and* so at least one RO
   has opposite-signed alignment at different altitudes (to confirm the
   heatmap's per-altitude recoloring is actually per-altitude, not just a
   single sign applied uniformly). Confirm `rank_by_isr_alignment`/
   `plot_impact_vector_isr_alignment` recovers the right sign and
   ordering; confirm `plot_impact_vector_isr_aligned_heatmap`'s per-cell
   values equal `I_ℓ(h)·sign(d_masked(h))` exactly; confirm both mask out
   altitudes beyond the ISR instrument's native coverage rather than
   silently extrapolating.
6. **Real-data verification**: reuse the cached real Tromsø 2025-11-18
   RO+IGS observations + the real ISR file (already used throughout the
   ISR integration work) to compute impact vectors for the real ROs at
   the real Tromsø site, for `ANCHOR` or a PCA style (nonlinear) and a
   dedicated `raw`-style run (linear, since `raw` isn't currently a
   default style). Sanity-check both rankings (plain magnitude, and the
   new ISR-alignment projection) qualitatively against the already-
   existing ISR nearest-point comparison plots.

## 10. Resolved decisions (confirmed by user, 2026-11-05)

1. **Centering: option (a), the `centering="mean"` diagnostic companion
   pass only** -- explicitly accepted as an *approximate* answer to which
   RO is most impactful (not a reconstruction of the exact production
   `nearest_analog` trajectory). No production-trajectory partial-
   attribution mode is being built. This is the simplification applied
   throughout Sections 2-9 above.
2. **Output stays as an EDP profile (impact vs. altitude) per RO** -- no
   reduction to scalar ionospheric parameters right now. **Flagged future
   extension, not designed here**: the user anticipates wanting the
   impact of each RO on parameters like `HmF2`/`NmF2`/`HmE`/`NmE`, which
   are nonlinear functionals of the EDP profile (`NmF2`/`NmE` are maxima
   over an altitude window, `HmF2`/`HmE` the argmax location) and
   meaningfully harder to evaluate than a linear profile-space impact --
   a classical analytic sensitivity (`∂NmF2/∂EDP`) is ill-defined for an
   argmax/max operator without smoothing, so this will likely need a
   finite-difference approach (perturb the forecast along each `I_ℓ`
   profile by a small step and re-evaluate the nonlinear parameter
   numerically) rather than a clean extension of this plan's linear-
   algebra decomposition. Not scoped further here; revisit when the user
   wants to pursue it.
3. **Altitude resolution: the model grid (82 levels), not the ISR
   instrument's native gates.** Applied in Section 7 (`QueryPoint`'s
   default `altitudes`).
4. **Scope: per-batch only**, confirmed -- no cross-batch accumulation
   for a future multi-batch cycle; out of scope for now.
5. **Module boundary: standalone `impact_vectors.py` for the core
   computation, confirmed.** Additionally, the ISR-comparison use case
   needs ranking by *direction* relative to ISR (pushing the analysis
   toward vs. away from the ISR-measured profile), not just magnitude --
   built as `rank_by_isr_alignment`/`plot_impact_vector_isr_alignment`
   (Section 8), taking an ISR profile as an explicit argument rather than
   making `impact_vectors.py` depend on `isr_comparison.py` wholesale.
6. **Generalize to arbitrary query points from the start** -- `QueryPoint`
   (Section 7) is not ISR-specific; an ISR station is one convenient way
   to populate one, via `CycleConfig.impact_vector_query_points` (explicit
   points) and/or `cfg.isr_file_path` (its station added automatically).

## 11. Suggested implementation order

**Progress (2026-11-05): steps 1-5 done.**

1. **Done -- no code change needed.** Confirmed by direct reading:
   `compose_observation_jacobian()` only ever uses its first argument's
   shape (`(n_rows, n_height*n_geo)`), never anything ray-geometry-
   specific -- already generic enough to accept `F_EDP(point)` in place
   of `H_grid`.
2. **Done.** `on_iteration` hook added to `AnalysisEngine.analyze()`
   (fires once for the `"linear"` branch, once per accepted Gauss-Newton
   iteration for `"iterated"`, with `(iteration_index, x_before, K, r)` --
   deliberately *not* `alpha`, since a step's decay weight isn't knowable
   until later steps' alphas are too; callers apply Finding 3's weights
   retroactively from `AnalysisDiagnostics.alpha_history`). Also added a
   small, purely-additive `GenericObservationOperator.get_jacobian_array(x)`
   (the un-composed Jacobian `linearized()` already builds internally,
   exposed so `impact_vectors.py` doesn't need to reach into
   `parameterization`/`_alt_arg` directly) and threaded `on_iteration`
   through `GeneralEnKFDriver.assimilate_one_cycle` as a pass-through
   (correcting this step's original text: `on_batch` does *not* reach
   into `analyze()` at all, so `on_iteration` needed its own new path,
   not a reuse of `on_batch`'s). 7 new tests in
   `Ensemble_Kalman_Engine/tests/test_analysis_engine.py::TestOnIterationHook`,
   including a from-scratch re-derivation of Finding 3's decay-weight
   formula checked directly against the engine's own accepted trajectory
   (both undamped and with real backtracking) -- all passed on first
   implementation, confirming Finding 3's math. Zero-regression test
   included (Section 9 test 4). Full `Ensemble_Kalman_Engine`/
   `Assimilation_Cycle`/`EDPSamples`/`Parameterization` suite (442 tests)
   still green.
3. **Done -- no code change needed.** `AnalysisConfig.default_for_style(style, centering="mean")`
   already accepts the override; no new plumbing required.
4. **Done.** `Assimilation_Cycle/impact_vectors.py`: `QueryPoint`,
   `query_point_from_isr_station`, `build_f_edp` (the `F_EDP(point)`
   construction, reusing `output._interpolate_field_at_latlon`'s mesh-
   lookup-is-altitude-independent shortcut, extended to arbitrary
   altitudes via `interp_heights` directly), and
   `ImpactVectorAccumulator` (`on_iteration`/`finalize`, deferring
   Finding 3's weighting to `finalize` as step 2 anticipated). `raw`'s
   exactness test passed immediately.
5. **Done, with a real correction to Section 3 along the way** (see that
   section's "Correction found during implementation" note): building
   the nonlinear-style exactness test surfaced that "exact" was only ever
   true for the *parameter-space partition* (property 1), not for the
   *decoded physical value* (property 2) once the decode itself is
   nonlinear -- found empirically (a 27% mismatch that shrank
   predictably as the test problem was scaled toward linear), not
   guessed. Section 3 and Section 9 (tests 1-3) have been corrected to
   match. 8 tests in `Assimilation_Cycle/tests/test_impact_vectors.py`,
   all passing.
6. **Done.** `plot_impact_vector_heatmap`, `plot_impact_vector_ranking`,
   `plot_impact_vector_ranking_cross_style` added to `impact_vectors.py`.
   Ranking metric (not an explicit open question, but a judgment call
   made while implementing): total `|impact|` summed over every
   requested altitude -- favors a broadly-distributed, noisy-looking
   impact profile over a sharply localized peak of similar area, visibly
   so in a synthetic smoke check; not revisited, since it's a reasonable
   default and easy to change later if real data makes it feel wrong. 5
   new tests (`TestVisualization` in `test_impact_vectors.py`), plus a
   visual smoke check (3 synthetic profiles with Gaussian-ish altitude
   envelopes) confirming the heatmap's symmetric diverging scale and
   ranking order are both sane and mutually consistent.
7. **Done.** `isr_correction_direction`/`rank_by_isr_alignment`/
   `plot_impact_vector_isr_alignment`/`plot_impact_vector_isr_aligned_heatmap`
   added to `impact_vectors.py`, reusing `isr_comparison._interp_log10`
   with swapped arguments (model -> ISR interpolation direction reversed
   to ISR -> model, justified since the result only ever feeds a masked
   dot product/display, never shown as a synthesized standalone ISR
   value). 9 new tests (`TestIsrAlignment`), including a fully hand-
   computable case (4 altitudes, `d` alternating sign, one RO exactly
   aligned with `d`, one exactly anti-aligned, one genuinely mixed-sign
   within its own row) that checks the aligned heatmap's per-cell values
   exactly, not just "ran without error" -- the real test of "recolors
   per-cell, not per-RO." Visual smoke check against a more realistic
   synthetic profile (Gaussian-ish F2-peak-shaped impacts) confirmed the
   ranking and heatmap agree and the ISR-coverage blanking (nothing
   synthesized below/above the instrument's native gates) renders
   correctly. Full suite: 492 passing.
8. **Done.** `CycleConfig` gained `compute_impact_vectors: bool = False`
   and `impact_vector_query_points: list[tuple[str, float, float]] | None = None`
   (`cycle_config.py`). `package_run.py`: builds the query-point list once
   per packaged run (explicit points plus, when `cfg.isr_file_path` is
   set, that station's point auto-added); per style, a new
   `_run_impact_vectors_for_batch` helper is invoked from the existing
   `_on_batch` callback (same wiring point as `IsrComparisonCollector`/
   `cross_style_data`) -- builds a fresh `ImpactVectorAccumulator` per
   batch, runs the extra `centering="mean"`-forced diagnostic
   `GeneralEnKFDriver(...).assimilate_one_cycle(..., on_iteration=accumulator.on_iteration)`
   call over `outcome.y_obs_used` (the actually-assimilated observation,
   correct under OSSE substitution too) and `batch.R`, discards that
   call's own ensemble result, and saves the plain heatmap/ranking plots
   for every configured query point plus the ISR-alignment ranking/
   aligned-heatmap pair whenever a query point's name matches the primary
   ISR station and a nearby ISR profile exists. A new
   `impact_vector_cross_style_data` dict (keyed by `(batch_index,
   point_name)`, mirroring `cross_style_data`'s own keying) feeds a
   cross-style ranking plot after the per-style loop. Zero regressions:
   full suite still 492 passing.

   **Real-data smoke test** (Section 9 test 6):
   `Driver_Scripts/Examples/impact_vector_wiring_smoke_test.py`, reusing
   the cached real Tromsø 2025-11-18 10:00-11:00 RO+IGS
   observations/EDPSamples (`step16_package_smoke`) and the real
   `TROISR2025_nonan.nc` ISR file, run for both `ANCHOR` (nonlinear) and
   `raw` (linear, confirming the plan's note that `raw` isn't a default
   style but is explicitly wanted for this test). Both produced all 4
   plots (heatmap, ranking, ISR-alignment ranking, ISR-aligned heatmap)
   for the real "EISCAT Tromso UHF ISR" station, plus the cross-style
   ranking. Visual sanity check against the real output: the RO entries
   (`podTc2_*`) show real, altitude-structured impact concentrated in the
   ~200-500km band (the F2-peak region), cleanly split by the
   ISR-alignment ranking into RO's pushing the analysis toward vs. away
   from the real ISR profile; the IGS entries (bare `G27.*`/`C28.*`/etc.
   labels) come back near-zero at this single vertical-column query
   point, which is physically expected -- a column-integrated TEC ray
   has far less leverage on one specific vertical profile than an RO
   whose tangent point passes near it. No qualitative red flags found.
9. **Done.** Added `Documentation/Assimilation_Cycle_README.md` Section
   3's "Per-RO/IGS impact vectors" field-reference table, Section 4's
   artifact list (`foo_{style}/impact_vectors/{point_name}/batchNNNN_*.png`
   + cross-style), and a new Section 11 (what it answers, cost, a minimal
   example, how to read the plots, and the "IGS entries read near-zero"
   gotcha). This plan document itself is the integration-plan-doc
   convention's artifact for this feature (this project's established
   pattern -- see `Assimilation_Cycle_Integration_Plan.md`/
   `ISR_Integration_Plan.md`'s own role -- is one plan doc per feature,
   not folding every feature into a single ever-growing doc), kept
   up to date in place through every step above rather than written
   freshly at the end. Memory updated to reflect the finished feature.

**Feature complete.** All 9 steps done, 492 tests passing, real-data
wiring smoke-tested end to end (Section 11 step 8's write-up above).
