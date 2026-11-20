# Spline Basis and Optimal Vertical Grid Placement: Implementation Plan

Status (2026-11-18): Phases 1–3 implemented in `Spline_Basis/`, together with parts
of Phases 4–5: multi-start, `residual_vs_n`, the grid file, and diagnostics.
112 tests pass. Not yet done: greedy initialization, the softmax objective,
and IRI generation from configuration. Pooled training (per-location sums
across all 34 hours) is done. After the user's update, `h0 = 60` and
`delta_min_km = 20`; the IRI members were regenerated from 60 km by
`extend_iri_profiles.py`. The minimum spacing can now vary with
altitude (`delta_min_km` as `[[h, δ], ...]`, enforced exactly in the
stretched coordinate `s = ∫ dh/δ`). The current grid file is
`grid_h60_dvar_linear_n14.json`: n = 14, δ = 10 km below 150 km rising to
20 km at 250 km, trained on IRI sampled every 5 km below 200 km. Development results are in
`Runs/Tomography_Test/Claude_Test/spline_knots/`.

## 0. Goal

1. Build a polynomial spline basis of degree `m` (cubic, `m = 3`, first; any
   `m >= 1` later) on a vertical grid `h0 = x_1 < x_2 < ... < x_n = h1`, where
   `n` counts both end points. The basis is

       P_k(t) = [x_k, ..., x_{k+m+1}] p_t,    p_t(x) = (x - t)_+^m

2. The boundary conditions are `Ne(h0) = 0` and `dNe/dh(h1) = 0`.
3. Given a collection of **IRI2020 climatological profiles**, find the
   placement of the `n` grid points that minimizes the residual of the best
   constrained spline fit to the collection.
4. Grid optimization is an **offline tool**. It is run occasionally and
   produces **one grid file used by all runs**. It is not part of the
   assimilation cycle. The cycle only loads the grid.
5. Later: register the basis as a parameterization style for the inversion.

## 1. Mathematical design

### 1.1 The basis is the B-spline basis

`P_k` is the Curry–Schoenberg B-spline, up to normalization:

- `M_k = (m+1) P_k` has unit integral.
- `N_k = (x_{k+m+1} - x_k) P_k` is the normalized B-spline. These form a
  partition of unity.
- Support is `[x_k, x_{k+m+1}]`, and the functions are `C^{m-1}` at simple
  knots.

Divided differences cancel catastrophically when knots are close together,
so they are used only as the reference implementation in tests (`mpmath` or
exact rationals, with confluent divided differences for repeated points).
Production evaluation uses the stable Cox–de Boor recursion, which gives the
same functions. It may wrap `scipy.interpolate.BSpline` internally. The basis
is stored as `N_k` by default.

### 1.2 Extra points and boundary conditions

**Extra points.** `h0` and `h1` are each repeated `m` more times (a clamped
knot vector):

    t = (h0 × (m+1), x_2, ..., x_{n-1}, h1 × (m+1))

This gives `K = n + m - 1` B-splines that span the full spline space on
`[h0, h1]`. The divided-difference definition remains valid with repeated
points: the confluent divided differences involve derivatives of `p_t`.

The clamped knot vector is the natural choice for these two boundary
conditions, because each condition then involves only the end coefficients:

- `y(h0) = c_1`, since only `N_1` is non-zero at `h0`.
- `y'(h1) = m (c_K - c_{K-1}) / (h1 - x_{n-1})`.

**Value at `h0`.** `Ne(h0) = 0` translates to `y(h0) = y_b`, where
`y_b = g⁻¹(0)` and `g` is the fit-space transform (§1.3):

| Fit space | `y_b` |
|---|---|
| linear | 0 |
| pow | 0 |
| log10 | `minlog10Density`, the floor already used by `density_10ex` (log10 cannot represent 0) |

The coefficient is fixed: `c_1 = y_b`.

**Derivative at `h1`.** Because `g` is strictly monotone, `dNe/dh = g'(y)·dy/dh`,
so `dNe/dh(h1) = 0` holds exactly when `dy/dh(h1) = 0`, in every fit space.
This condition is imposed by tying `c_K = c_{K-1}`, which merges `N_{K-1}` and
`N_K` into a single basis function `N_{K-1} + N_K`.

**Constrained representation.**

    y(h) = y_b N_1(h) + Σ_{k=1..K-2} a_k B̃_k(h)

Here `B̃_k = N_{k+1}` for `k < K-2`, and `B̃_{K-2} = N_{K-1} + N_K`.

The number of free coefficients is `K - 2 = n + m - 3`. **For cubic splines,
`n` grid points give exactly `n` free coefficients per profile.**

Both conditions are imposed through the basis itself rather than as
penalties. The constrained basis `B̃` (together with the offset `y_b N_1`)
therefore becomes the state basis for the later inversion.

Tests check `y(h0) = y_b` and `y'(h1) = 0` to machine precision for random
coefficients, for `m = 1, 2, 3, 5`. For `m = 1` the derivative condition
makes the last linear piece flat, which is allowed.

**Caveat.** IRI is not exactly 0 at `h0`, and its slope at `h1` is not
exactly 0. The diagnostics therefore report the boundary-induced error in the
two end intervals separately, so you can judge whether `h0` or `h1` should be
moved.

### 1.3 Fit space (configuration parameter)

The configuration parameter `fit_space` takes one of three values:

- `"linear"`
- `"log10"`, with `minlog10Density`
- `"pow"`, with `alpha`

The transforms `log10_param` / `pow_param` and their inverses
`density_10ex` / `density_pow` are reused from
[Parameterization.py](../Parameterization/Parameterization.py).

The optimizer only sees the transformed targets `Y = g⁻¹(Ne)` and the offset
`y_b`. It does not otherwise depend on the fit space.

The residual is defined in the fit space. The diagnostics also always report
errors in linear Ne and log10 Ne, so grids built in different fit spaces can
be compared directly.

### 1.3a Ratio-to-mean representation (user proposal, 2026-11-18; becomes the primary mode)

The spline represents the **ratio** to a reference mean profile, not Ne itself:

    Ne(h) = m̄(h) · r(h),    r(h) = Σ a_k B̃_k(h)    (fit_space = "linear")

In the tomography, the state becomes the ratio coefficients `a`, and the
mean `m̄` becomes part of the parameterization. `m̄` is stored with the grid,
or taken from the run's background (see "Choice of reference" below).

**Why this helps:**

- **The forward model stays linear.** The observation operator row for
  coefficient `k` is `∫ m̄(h(s)) B̃_k(h(s)) ds`, which can be precomputed per
  ray.
- **The natural prior is `r ≡ 1`,** so `a ≈ 1` and every coefficient has
  order-1 scale. This conditions the EnKF better than raw Ne.
- **The basis carries the mean shape.** In Ne, each basis function is
  `m̄·B̃_k`, so knots are spent where profiles *deviate* from the mean, not
  on reproducing the topside decay or the F2 peak shape.
- **Altitudes are on an equal footing.** In the example collection, the
  ratio spread is about 0.3–0.4 (1σ) at every F-region altitude from 200 to
  900 km, even though the mean Ne varies by more than 10× over that range.

**What it does *not* fix:**

1. **Relative weighting moves to low-density altitudes.** The ratio residual
   is a relative error, so altitudes where `m̄` is small count as much as the
   F2 peak.
   - In the example, the 150 km valley has a ratio σ of 1.28, against about
     0.35 elsewhere, and its 5th percentile is −1.49 (members with negative
     Ne).
   - That one level would dominate the objective and pull knots into the
     valley, where TEC sensitivity is negligible.
   - Mitigation: an optional altitude weight
     `w(h) = (m̄(h)/max m̄)^β`.
     - `β = 0` is the pure ratio.
     - `β = 1` is equivalent to an absolute-Ne residual.
     - The default is `β = 0`, with the diagnostics showing the per-altitude
       contribution to `R`.
2. **The spread between profiles remains.** Profiles with large ratios
   (day against night, solar maximum against minimum) still dominate a
   linear-ratio residual when the reference covers very different
   conditions.
   - This is reduced a lot if the reference is the mean of *comparable*
     profiles (see "Choice of reference").
   - Alternatively, use `fit_space = "log10"` on the ratio:
     `log10 Ne − log10 m̄`, which treats ×2 and ÷2 symmetrically.
   - `fit_space` stays a configuration parameter and applies to `r`:
     `linear` (the default for ratio mode), `log10`, or `pow`.
3. **Ne can still be negative** in linear-ratio mode. This is the same as
   for the existing linear styles.

**Choice of reference** (configuration `reference`):

| Option | `m̄` is the mean over… | Notes |
|---|---|---|
| `"collection"` | all profiles in the training collection (literal proposal) | One fixed mean stored in the grid file. Across seasons and local times the ratio spread becomes large (caveat 2). |
| `"ensemble"` | the members of the same EDPSamples set (same time and region) | Matches how the tomography will use it, with `m̄` = the run's background ensemble mean. The grid stays single and fixed; only `m̄` varies per run. |
| `"column"` (recommended) | the members at the same geo vertex | The same as `"ensemble"`, but per column, which matches the 3D background mean the EnKF already has. |

For the grid optimization, `"column"` or `"ensemble"` means that ratios are
formed group by group before all groups are pooled into one objective. The
grid is still single, as decided.

**Rules for computing the mean:**

- `m̄` is the arithmetic mean, floored at `10^minlog10Density`. The geometric
  mean is undefined for members with Ne ≤ 0.
- `m̄` is interpolated to ray points in log10 Ne.

**Boundary conditions under the ratio** (user decision 2026-11-18: keep the
conditions on Ne):

- **Lower bound: `Ne(h0) = 0` ⇒ `r(h0) = 0` ⇒ `c_1 = 0`.** This is exact,
  since `m̄(h0) > 0`. The same constraint as before, applied to `r`.
  - In the high-resolution example, `m̄(90 km)` is about 3% of the column
    maximum (5th–95th percentile: 2.4–4.4%), while `r(90 km) ≈ 1`.
  - The residual at the sample `h = h0` is therefore a constant: `r` is
    forced to 0 there whatever the grid. It is **excluded from `R`** because
    it carries no information about the grid.
  - The residual just above `h0` does depend on the grid: `r` must rise
    from 0 to about 1 within the first interval. The optimizer will tend to
    shrink that interval toward `δ_min`, effectively spending one grid point
    on the boundary condition.
  - The diagnostics report the cost of the first interval separately.
  - Option: a separate `delta_min_first_km`, to keep that interval from
    collapsing below the sample spacing.
- **Upper bound: `dNe/dh(h1) = 0` ⇒ `m̄' r + m̄ r' = 0` ⇒ `r'(h1) = r(h1)/H`,**
  where `H = −m̄/m̄'` is the column mean's topside scale height at `h1`.
  - With the clamped knots, `r'(h1) = m (c_K − c_{K-1})/Δ` and `r(h1) = c_K`,
    with `Δ = h1 − x_{n-1}`. So

        c_{K-1} = c_K · (1 − Δ/(m H)),

    still a linear tie. The two end basis functions merge into
    `N_K + (1 − Δ/(mH)) N_{K-1}`.
  - `H` is computed per column from the mean profile: a log-linear fit over
    the top 2–3 samples.
  - In the example `H` is 264–384 km (5th–95th percentile), so for
    `Δ ≈ 50 km` and `m = 3` the factor is about 0.95, a mild tie.
  - Guard: require `Δ < m H`. Otherwise the factor would be ≤ 0, and the
    code raises an error.
- **Consequence: the constrained basis `B̃` depends on the column** (through
  `H`). The fit therefore factors one QR per column, shared by that column's
  members: 161 small QRs per hour in the example, which is cheap.
  - The knot Jacobian also has an explicit column dependence through the
    tie factor `Δ = h1 − x_{n-1}`, and this is included in the VarPro
    gradient.
  - The tomography will need the same per-column `H`, taken from the
    background mean, so that `B̃` matches.
- Free coefficients per column: `K − 2 = n + m − 3`, which equals `n` for
  cubic splines (unchanged).

**Sample spacing.** The example profiles are sampled every 10 km. A basis
function whose support contains no samples makes the least-squares problem
singular (the Schoenberg–Whitney condition fails).

- `δ_min` must therefore be at least the sample spacing.
- `fit_profiles` checks the Schoenberg–Whitney condition, and a tiny
  optional smoothing penalty serves as a backstop.

### 1.4 Objective

Inputs:

- `Y`: transformed IRI profiles on a fine altitude grid `h_i`.
- `W`: weights.

The residual is

    R(x) = || (I - Π_{W B̃(x)}) W (Y - y_b N_1(x)) ||_F²

All profiles share the matrix `B̃(x)`, so a single QR factorization per
evaluation fits every profile.

Weight options (configurable):

- **Per-profile normalization by NmF2.** This is the default for
  `linear`/`pow`. It stops daytime profiles from dominating.
- **Optional altitude weights**, for example an E-region emphasis.
- **Uniform weights.** This is the default for `log10`.

## 2. Knot-placement optimization

### 2.1 Variables

- The interior points `x_2..x_{n-1}` are free. `h0` and `h1` are fixed.
- Ordering and the minimum spacing `δ_min` are guaranteed by the gap
  parameterization:

      gap_i = δ_min + (h1 - h0 - (n-1) δ_min) · softmax(θ)_i

- Optionally, some points can be pinned at fixed altitudes.
- **Validation of the spacing threshold** (`KnotOptimizationConfig.__post_init__`):
  - **Feasibility.** Raise an error if
    `(n-2)·delta_min_km + delta_min_first_km > h1 − h0`.
  - **Pinned points.** Raise an error if two pinned points (or a pinned
    point and `h0`/`h1`) are closer than the threshold.
  - **Sample spacing.** Warn if the threshold is smaller than the input
    profiles' sample spacing, because the Schoenberg–Whitney condition can
    then fail (§1.3a).
  - **Tight thresholds.** Warn if the threshold is within about 5% of the
    feasibility limit, because the grid is then nearly forced to be uniform.
- **Effect on the gap parameterization.** With a separate first-interval
  threshold, the per-gap floors become
  `δ_1 = delta_min_first_km` and `δ_i = delta_min_km` for `i ≥ 2`, and

      gap_i = δ_i + (h1 − h0 − Σδ) · softmax(θ)_i

  This keeps the threshold exact without penalties or projection.
- **Reported in the diagnostics and the grid file:** the threshold, and
  which gaps ended at or near it. Gaps sitting on the threshold mean the
  optimizer wanted finer spacing than allowed.

### 2.2 Variable projection with analytic knot gradients

The coefficients `a` are eliminated (Golub–Pereyra variable projection).

The knot derivative of the basis comes from the divided-difference
definition:

    ∂/∂x_j [..., x_j, ...] f = [..., x_j, x_j, ...] f

This is computed exactly by knot insertion. Because `B̃` consists of linear
combinations of `N_k`, its knot derivatives follow directly, and so do those
of the offset term `y_b N_1`.

The solver is `scipy.optimize.least_squares` (TRF) with the Kaufman Jacobian,
or L-BFGS on `R`.

### 2.3 Initialization (multi-start, seeded random number generator)

1. Uniform points.
2. Equidistribution of the asymptotically optimal density
   `ρ(h) ∝ ( mean_j |y_j^{(m+1)}(h)| ^ {2/(2m+3)} )`, with derivatives taken
   from a fine-knot smoothing spline.
3. Greedy removal from a dense grid (Lyche–Mørken style).
4. `n_random` perturbations of (2).

Each start is refined with §2.2, and the best result on the training set is
kept.

### 2.4 Generalization and choosing `n`

- The IRI collection is split into train and validation sets **by
  conditions**: hold out some months, solar-activity levels, and local times.
  Profiles are not split at random.
- `objective` is either `"mean"` or `"softmax"`. The soft maximum over
  per-profile residuals protects rare shapes such as night-time E/F valleys
  and high solar-activity topsides.
- `residual_vs_n(n_list)` produces training and validation curves, which you
  use to choose `n`.

## 3. Configuration and IRI2020 training collection

### 3.1 Configuration

The configuration is a dataclass `KnotOptimizationConfig`, saved and loaded
as JSON. The required fields have **no defaults**: the user must provide
them.

| Field | Required | Meaning |
|---|---|---|
| `h0_km`, `h1_km` | yes | Domain |
| `n_points` (or `n_points_list` for the residual-vs-n curve) | yes | Number of grid points, including `h0` and `h1` |
| `delta_min_km` | yes | Lower threshold for the spacing between adjacent grid points. A hard constraint, enforced by construction (§2.1) in every initializer and optimizer step |
| `delta_min_first_km` | no (= `delta_min_km`) | Optional separate threshold for the first interval `[h0, x_2]` (see §1.3a) |
| `degree` | no (3) | Spline degree `m` |
| `fit_space` | yes | `linear`, `log10`, or `pow` |
| `alpha` / `minlog10Density` | when relevant | Parameter of the fit-space transform |
| `fine_alt_step_km` | yes | Altitude resolution of the IRI training profiles |
| `iri_collection` | yes | Definition of the IRI sample set (§3.2) |
| `validation_split` | yes | Which months, solar levels, and local times are held out |
| `weights`, `objective`, `init`, `n_random`, `seed`, `pinned_points_km` | no | Optimizer options |
| `output_path` | yes | Where the grid file is written |

### 3.2 IRI2020 training collection

The collection is generated with the existing machinery:

- `EDPSamples(..., evaluate_iri=1)`
- `get_IRI2020_EDP`, which requires `IRI2020_PATH`

The configuration specifies:

- **Times:** dates or ranges spanning seasons and years, and UT hours (or
  local times).
- **Geolocations:** an EDPSamples `geo_type` and its parameters, for example
  Regional or Polar around the regions we run, or Global.
- **Sampling parameters:** f107/ap/ig/rz, either taken from each date or
  spread over a range, as `iri_spread_kwargs` does.

The collection is cached to NetCDF, so it can be regenerated or extended
without rerunning everything. The profiles in it are of shape
`(height, geo, sample)` and are flattened to `Y`. **ISR profiles are not
used.**

### 3.3 Output and use

- `optimize_vertical_grid.py` (in `Driver_Scripts/`) takes a configuration
  file and writes a **grid file** (NetCDF or JSON) containing:
  - `grid_km`, `degree`, `fit_space` and its parameters
  - the boundary-condition type
  - the full configuration, an IRI collection summary, and the training and
    validation residuals
  - the code version and date
- Normal runs only **load** this file through
  `SplineBasis.from_grid_file(path)`. Grid optimization never runs inside
  the assimilation cycle.

## 4. Code structure

```
Spline_Basis/
  spline_basis.py        # divided differences (reference), clamped knots, SplineBasis, constrained basis B̃
  spline_fit.py          # batched constrained weighted LSQ, fit-space transforms, residual metrics
  knot_optimization.py   # config dataclass, VarPro objective/Jacobian, θ↔grid, inits, optimize_knots, residual_vs_n
  iri_collection.py      # build/cache the IRI2020 training collection from config; train/validation split
  diagnostics.py         # basis plots, grid over profiles, residual vs n, per-altitude/per-condition errors
  test_spline_basis.py, test_spline_fit.py, test_knot_optimization.py, conftest.py
Driver_Scripts/optimize_vertical_grid.py
```

Main API:

- `SplineBasis(grid, degree=3, boundary="zero_value_low|zero_slope_high", fit_space=..., ...)`
  - `.design_matrix(h, deriv=0)` returns `B̃`.
  - `.offset(h)` returns `y_b N_1`.
  - `.to_profile(a, h)` returns Ne, with the inverse transform applied.
  - `.fit(h, Ne)` returns `a`.
  - `.knot_jacobian(h)`, `.integrals()`.
  - `.from_grid_file(path)`, `.save(path)`.
- `optimize_knots(config) -> KnotResult`
- `residual_vs_n(config) -> table`
- Diagnostics write to `Runs/Tomography_Test/Claude_Test/spline_knots/`.

## 5. Decisions (user, 2026-11-18)

1. `n` counts the grid points including `h0` and `h1`.
2. The boundary conditions are `Ne(h0) = 0` and `dNe/dh(h1) = 0`. They are
   imposed through the clamped extra points plus constraints on the
   coefficients (§1.2).
3. The fit space is a configuration parameter: `linear`, `log10`, or `pow`.
4. Only IRI2020 climatological profiles are used for optimization. ISR is
   not used.
5. `h0`, `h1`, `δ_min`, and `n` are user-provided configuration values.
6. A single grid is used for all runs. Optimization is an occasional offline
   step, not part of the cycle.

## 5a. Ratio mode decisions and example collection (2026-11-18)

**Decisions:**

1. Fit `r = Ne / m̄` (§1.3a). The tomography will estimate the ratio
   coefficients.
2. `reference = "column"`: `m̄` is the mean over the members at each geo
   vertex.
3. `dNe/dh(h1) = 0`, as a per-column linear tie (§1.3a).
4. `Ne(h0) = 0`, through `c_1 = 0`.

New configuration fields:

- `representation`: `"ratio"` (default) or `"density"`
- `reference`: `"column"` (default), `"ensemble"`, or `"collection"`
- `altitude_weight_beta` (default 0)
- `delta_min_first_km` (optional)
- `topside_H_fit_samples` (default 3)

**Example collection (development).**

The development file is
`…/tomography_output/MIL_aug2530_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp/2025-08-27_16001700/edp_samples.nc`
(Google Drive shared drive, 122 MB). It contains:

- The IRI2020 prior ensemble: `EDPs` of shape (82 heights × 161 geo ×
  2000 members), all values positive.
- Heights from **90 to 900 km in steps of 10 km**. So `h0 = 90` and
  `h1 = 900` for this data, and `δ_min ≥ 10 km`.
- 2025-08-27T16:30 UT, Regional with an 18° radius around Millstone Hill.
- Member spread: hour ±3 h, f107/ap ±30 days, ig/rz ±12 months.

Ratio statistics with the per-column mean:

- `r` has a median of about 1.
- 1σ is 0.18–0.32 between 90 and 400 km, rising to 0.43–0.46 above 600 km
  (5th–95th percentile range at the top: 0.45–1.9).
- No single level dominates, unlike the 150 km valley in the step12
  analysis set. The ratio objective is therefore well balanced, and
  `β = 0` is a reasonable default.

**Collections for the study** (one `edp_samples.nc` per hourly folder):

- **Development:** MIL 2025-08-27 16–17 UT (above).
- **Training:** all 34 MIL hourly folders, 2025-08-26 to 08-29, covering
  00–04 and 13–18 UT (day and night), from the `…3DL_3Dp_1Dp` run.
- **Validation (held-out conditions):** TRO May 18–21 and Oct 14–17 2025
  folders (`TRO_may1821_…`, `TRO_oct1417_…`). These test a different
  latitude and season.
- Pooling 34 × 161 × 2000 profiles is large: about 11M profiles,
  roughly 4 GB as float32.
  - The loader subsamples members per column (configurable, for example
    100–200) and can subsample geo vertices.
  - The per-column QR cost scales with the number of members, so this is
    the main cost control.
- The step12 analysis set (50 km sampling, contains negative values) is no
  longer used.

The production grid will still come from IRI2020 climatology (§5). This set
is IRI2020 prior output, so it qualifies; broader seasonal coverage can be
added later by generating more hours with `get_IRI2020_EDP`.

## 6. Phases

| Phase | Work | Done when |
|---|---|---|
| 1 | Divided-difference reference, clamped knots, `SplineBasis`, constrained basis `B̃`, integrals, derivatives | Recursion ≡ divided differences (including confluent ones); partition of unity; matches `scipy.BSpline`; boundary conditions hold exactly; reproduces any spline in the constrained space exactly; `m = 1, 2, 3, 5` |
| 2 | Ratio mode (per-column mean, per-column `H` tie), constrained per-column batched fit, transforms; residual metrics | MIL 08-27 16 UT dev file fit on a uniform grid; per-band error table in ratio, linear Ne and log10 Ne |
| 3 | Knot Jacobian, VarPro, θ-parameterization, single start | Gradient agrees with finite differences to 1e-6 relative; recovers planted grids on synthetic splines |
| 4 | Config dataclass, `iri_collection.py`, multi-start, softmax objective, `residual_vs_n` | Optimized grid beats uniform and density-only grids on the held-out conditions |
| 5 | Driver script, grid file, diagnostics; real IRI study | Residual-vs-n curves for each fit space; recommended `n`; grid file written |
| 6 (later) | `SPLINE` style in `EDP_Parameterization` loading the grid file; ray integrals; EnKF hookup | Separate plan |

## 7. Remaining open items (can be decided during Phase 4)

- **Altitude weight `β` in ratio mode.** The default is 0. Revisit if the
  topside, where `r` has 1σ ≈ 0.45, pulls too many grid points upward.
- **Optimizing the ratio in log10 or pow space** (the `fit_space` setting
  applied to `r`). This is the same configuration parameter as before; it
  is evaluated in the Phase 5 study.
- **Members per column** to subsample when pooling the 34 MIL hours.
