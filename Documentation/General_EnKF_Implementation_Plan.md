# Plan: A General-Purpose Ensemble Kalman Filter Module

Source: `Particle_Kalman_Filter.pptx` (Presentation folder), cross-referenced against
the current `IonosphereTomography` codebase.

## 1. Background: the scheme in the deck

The deck derives a Kalman filter analysis step that never forms the full
`N x N` prior covariance matrix, using an ensemble ("particles") to represent
it instead. Restated in one place for reference:

**Standard minimum-variance update.** State `x`, unbiased prior `x_p` with
`Var(x - x_p) = Q`, observation `y_bar = Hx + eps`, `E(eps)=0`,
`Var(eps)=R`, `eps` uncorrelated with `x - x_p`:

    x_a = x_p + K (y_bar - H x_p),      K = Q H^T (H Q H^T + R)^-1

**Ensemble representation of Q.** Given `n` samples `x_p,1 ... x_p,n` drawn
from the prior (e.g. IRI climatology):

    x_p ~ xbar_p = (1/n) sum_i x_p,i
    Q   ~ (1/(n-1)) sum_i (x_p,i - xbar_p)(x_p,i - xbar_p)^T = S S^T

where `S` is `N x n`, its `i`-th column being `(x_p,i - xbar_p) / sqrt(n-1)`.

**Analysis covariance, low-rank form.** The exact identity

    Q_a = Q - Q H^T (H Q H^T + R)^-1 H Q

is rewritten using only `N x n` / `M x n` matrices (never `N x N`):

    Q_a = S S^T - S Y^T (Y Y^T + R)^-1 Y S^T,      Y = H S

Note `Q_a` does not depend explicitly on the observation `y_bar` — expected,
since Kalman-filter optimality rests entirely on the linear-Gaussian
assumptions, not on the realized data.

**Propagating an actual ensemble (not just Q_a).** Since even `Q_a` shouldn't
be formed explicitly, the deck gives a **perturbed-observation** update that
produces a new analysis ensemble `S_a` with `Q_a ~ S_a S_a^T` directly. The
`i`-th column of `S_a` is

    x_p,i - x_p - K (H x_p,i - H x_p + eps_i),      eps_i ~ (0, R) iid

This is the standard stochastic/"perturbed-obs" EnKF analysis step (Burgers
et al. 1998 / Evensen). `S_a` becomes the forecast ensemble seed for the next
assimilation cycle.

## 2. What already exists in this repository

- **`Ionosphere_Tomography_Inverter/enkf_update.py`** already implements this
  math generically on `(n_state, n_members)` arrays: a stochastic
  perturbed-observation branch (matches Section 1's last step almost
  verbatim) and a deterministic EnSRF branch (a square-root variant of the
  same `Q_a` identity), plus Gaspari-Cohn localization and inflation. The
  core linear algebra does not know or care what the state variables mean.

- **However**, everything around that core is hardcoded to one specific,
  nonlinear state representation: the 8-parameter Chapman/ANCHOR profile
  (`NmF2, hmF2, H0, gamma, B0, B1, NmE, hmE`) defined in
  `ionospheric_state.py` (`N_STATE = 8`). The forecast ensemble generator,
  the physical-bounds clamp, the localization reshape (`n_state // n_grid`),
  and the forward/observation model (`observation_operator.py`, an analytic
  Chapman-profile TEC integrator) are all specific to that one
  parameterization. In effect, today's filter is "EnKF specialized to
  ANCHOR," not a general-purpose EnKF.

- **`Parameterization/Parameterization.py`** already provides most of the
  abstraction needed to fix this. `EDP_Parameterization` bundles a
  `style` with three operations — `get_density` (decode), `get_parameter`
  (encode), `get_Jacobian` — behind one interface, for styles: `density_10ex`
  (log10), `ANCHOR`, `PCA_1D`, `PCA_3D`, `PCA_1D_10ex`, `PCA_3D_10ex`.
  `Parameterized_EDPSamples` bundles a real `EDPSamples` ensemble with a
  chosen style and stores the encoded `param_vec`.

- **`EDPSamples.get_observation_operator()`**
  (`EDPSamples/edp_samples.py`) already builds the ray line-integral
  observation matrix `H_grid` (shape `n_rays x (n_height*n_geo)`) directly
  from ray geometry and the density grid. Because TEC is a *linear*
  functional of physical electron density regardless of how that density is
  parameterized, `H_grid` is reusable across every parameterization style —
  it is not specific to ANCHOR, PCA, or raw grid.

**Key consequence:** for any style, the forward model factors cleanly as

    Y_f[:, i] = H_grid @ flatten( parameterization.get_density(X_f[:, i]) )

evaluated per ensemble member. Only `get_density` (nonlinear, style-specific)
varies; `H_grid` (linear, physical, style-independent) does not. This is
exactly why an ensemble filter — propagate members through the true
nonlinear decode, then a fixed linear operator — is the right generalization,
and it removes the need for `get_Jacobian` / EKF-style linearization in the
assimilation step itself.

## 3. Goal

Build one general-purpose Ensemble Kalman Filter module where the state
representation (raw grid, log10, ANCHOR, PCA, or a future nonlinear style) is
a pluggable choice, not something the filter engine hardcodes. `enkf_update.py`
is the proof that the core math is correct (Section 1) and the reference the
new engine's linear algebra is built from — but, per the code-reuse policy
in Section 5.0, that proof is *replicated* into the new module, not imported.
The genuinely generic, already-shared infrastructure (`Parameterization.py`,
`EDPSamples`) is reused by import, as intended.

## 4. Proposed architecture

```
 EDPSamples (raw ensemble, IRI-drawn)
        |
        v  Parameterization.get_parameter()   [encode, any style]
 Parameterized_EDPSamples.param_vec  -------->  filter-space ensemble X_f
        |                                        (n_state, n_members)
        |                                             |
        |                                             v
        |                                 +--------------------------+
        |                                 |  Observation operator    |
        |                                 |  decode: get_density(.)  |
        |                                 |  predict: H_grid @ (.)   |
        |                                 +--------------------------+
        |                                             |
        |                                             v
        |                                   Y_f (n_obs, n_members)
        |                            [computed once, from the full
        |                             ensemble -> also gives Q ~ S S^T
        |                             and feeds nearest-analog search]
        v                                             v
        +------------->  Analysis engine (Section 6.4)  <-------------+
                                        |
              two independent axes, both configurable per style:
                linearization:  linear  |  iterated   (Remedy A, 6.2)
                centering:      mean    |  nearest_analog (Remedy B, 6.3)
                                        |
             [[ `iterated` mode only -- loop, capped at 5-10 passes: ]]
                1. pick center x_i  (ensemble mean; or nearest-analog
                   member to y_obs -- re-picked every pass if Remedy B
                   is on, held fixed if Remedy B is switched off, 6.4)
                2. re-linearize  W_hat_i = H_grid @ get_Jacobian(x_i)
                   (single-point evaluation, not the full ensemble;
                   skipped entirely in `linear` mode -- one pass, no
                   Jacobian, same math as Section 1)
                3. gain  G_i = alpha Q W_hat_i^T (W_hat_i Q W_hat_i^T+R)^-1
                   damped-Newton step  x_(i+1) = x_i + G_i(ybar - ...)
                4. repeat until  ||x_(i+1)-x_i||/||x_i|| < tol  or capped
                                        |
                                        v
           converged mean update, then the Section-1 perturbed-obs
                  formula builds the analysis ensemble  S_a
                                        |
                                        v
                       X_a (n_state, n_members)
                                        |
                                        v
                  optional analysis perturbation hook (5.3)
          (additive Gaussian dither and/or selective multiplicative
           covariance inflation; bounds clamp is one special,
           deterministic case of it)
                                        |
                                        v
            Parameterization.get_density()  ->  updated EDPSamples
             (decode for diagnostics / next-cycle forward propagation)
```

Everything below the first arrow is representation-agnostic: swapping
`style='ANCHOR'` for `style='PCA_3D'` or a new `style='raw'` should not
require touching the filter engine or the observation operator — only the
linearization/centering choice inside the analysis engine (typically
`linear`+`mean` for `raw`, `iterated`+`nearest_analog` for everything
nonlinear, per Section 6).

## 5. Work items

### 5.0 Module location and code-reuse policy

**Confirmed: the new module lives in its own subfolder,**
`IonosphereTomography/Ensemble_Kalman_Engine/`, as a standalone package —
not added inside `Ionosphere_Tomography_Inverter/`.

**Confirmed: the core EnKF analysis math is replicated, not imported.**
`enkf_update.py`'s gain computation, stochastic perturbed-obs update, and
EnSRF branch are rewritten from scratch inside `Ensemble_Kalman_Engine/`,
using the old file purely as a worked reference during development. The old
`Ionosphere_Tomography_Inverter/enkf_update.py` is left untouched and keeps
serving the existing ANCHOR-only pipeline; `Ensemble_Kalman_Engine` carries
no import dependency on it (or on `ionospheric_state.py` /
`observation_operator.py`). This is deliberate for maintainability: the two
filters can now evolve independently, and nothing in the new, general
module can be broken by a future change to the old, ANCHOR-specific one (or
vice versa).

This replication applies specifically to the *analysis engine* — the part
that is genuinely being rewritten (Section 5.5/6). It does **not** extend to
the infrastructure this whole plan is built around reusing:
- `Parameterization/Parameterization.py` (`EDP_Parameterization`,
  `Parameterized_EDPSamples`) — imported as-is.
- `EDPSamples/edp_samples.py` (`EDPSamples`, `get_observation_operator`,
  `forward_model_mesh_tec`) — imported as-is.

Those two are shared, general-purpose infrastructure, not "old EnKF code" —
duplicating them would defeat the point of Section 2's finding that they
already generalize across styles.

### 5.1 Add a `'raw'` (identity) parameterization style
Trivial addition to `Parameterization.py`: `to_density` / `to_parameter` are
the identity, `get_Jacobian` returns an identity diagonal. **Scope note
(confirmed):** `raw` and `density_10ex` are for small/toy examples and
correctness checks only — the realistic, production cases are `ANCHOR` and
`PCA`. This gives the direct-grid case the deck's own motivating example
describes as one pluggable style among others, and is a good first
correctness check (encode/decode is a no-op), without implying any need to
scale the raw-grid path to production-size meshes.

### 5.2 Generalize the ensemble/state container
Replace the ANCHOR-only `IonosphericState` (or add a sibling to it) that:
- Wraps a filter-space ensemble as a plain `(n_state, n_members)` array,
  regardless of which per-style shape `n_state` was flattened from:
  `(n_height, n_geo)` for `raw`/`density_10ex`, `(8, n_geo)` for `ANCHOR`,
  `(nPCA, n_geo)` for `PCA_1D`/`PCA_1D_10ex`, and `(nPCA,)` — no `n_geo`
  factor, since `PCA_3D` compresses across all horizontal grid points
  jointly — for `PCA_3D`/`PCA_3D_10ex`.
- Is constructed from a real `EDPSamples` draw via
  `Parameterized_EDPSamples` + `EDP_Parameterization.get_parameter`, so the
  initial ensemble is always real IRI-sampled data, not synthetic Gaussian
  draws (satisfies "leverage EDPSamples as the initial ensemble").
- Exposes `ensemble_mean`, flatten/unflatten, and the analysis-perturbation
  hook described in 5.3 (optional, and *not* needed in most cases per
  the confirmed answer below).

### 5.3 Analysis-state / analysis-covariance perturbation hook

**Confirmed design change:** the post-update bounds clamp is not, in
general, a separate mechanism — it is one (deterministic) special case of a
more general, optional **additive perturbation applied after the analysis**,
to the state and/or to the analysis covariance. Concretely:
- **State perturbation**: an additive term applied to each analysis member
  (or to the analysis mean). Bounds-clamping is the special case where that
  "perturbation" is `clip(x, lo, hi) - x` — deterministic, zero almost
  everywhere. A stochastic dither/jitter (to guard against ensemble
  collapse) is the same hook with a random additive term instead.
- **Covariance perturbation**: an additive (or multiplicative, as a special
  case) term applied to `Q_a` / the analysis ensemble spread — generalizes
  the existing multiplicative `inflation` parameter in `enkf_update.py`
  (currently applied *before* the update, to the forecast anomalies) into a
  configurable post-analysis step as well.

Design as a single pluggable hook, `perturb(X_a) -> X_a'`, defaulting to the
identity (confirmed: not needed in most cases), with bounds-clamping,
inflation, and dithering all implemented as instances of it rather than as
separate mechanisms.

**Confirmed: support both forms together, composable, not either/or:**
- **Additive Gaussian dither** — add zero-mean Gaussian noise to each
  analysis member, guarding against ensemble collapse.
- **Multiplicative covariance inflation applied selectively** — inflate the
  spread of only a chosen portion of the state vector's components (e.g.
  one ANCHOR parameter across all grid points, or one PCA mode), not
  uniformly across the whole state. Implement as a per-component weight
  vector/mask over the inflation factor (weight 1.0 = inflate normally,
  0.0 = leave untouched), rather than a single global scalar — generalizes
  the existing scalar `inflation` parameter in `enkf_update.py`.

### 5.4 Generic observation operator

**Confirmed: retire the bespoke ANCHOR analytic integrator entirely.**
`observation_operator.py`'s analytic Chapman-profile TEC integrator is not
kept as a fast path — every style, ANCHOR included, uses the same generic
route:
1. Decode every ensemble member with `EDP_Parameterization.get_density`.
2. Apply the precomputed, style-independent `H_grid`/integrator already
   implemented in `EDPSamples` (`get_observation_operator` /
   `forward_model_mesh_tec`) — the same EDP-to-TEC integrator for every
   style, no parallel implementation to maintain or keep in sync.

This simplifies the plan: there is no second numerical path to validate
against, and `observation_operator.py`'s bespoke integrator becomes dead
code to remove once the generic route is wired in.

### 5.5 Reimplement the analysis engine in `Ensemble_Kalman_Engine` (replicated, per 5.0)
The gain/update math itself already operates on generic `(n_state,
n_members)` arrays in `enkf_update.py` and needs no *algorithmic* change —
just a from-scratch rewrite in the new module (5.0), which naturally drops
the two ANCHOR-specific assumptions rather than requiring them to be
refactored out of the old file in place:
- Localization reshape (`n_params = n_state // n_grid` tiling) — meaningless
  for `PCA_3D`, where there is no "N parameters per grid point" structure;
  the new implementation takes a general localization matrix instead.
- `clamp_to_physical_bounds()` — not carried over at all; superseded by the
  general perturbation hook (5.3).

### 5.6 New orchestration driver
Lives in `Ensemble_Kalman_Engine/` alongside the analysis engine, playing
the same role `Ionophy_Tomography_Inverter_EnKF.py` plays for the old
pipeline but parameterized by `style`, running the same assimilation loop
regardless of representation.

### 5.7 Validation / testing
- **Closed-form check**: on a small toy problem (small `N`, known `H`),
  verify `Q_a ~ S_a S_a^T` (sample covariance of the generated analysis
  ensemble) matches the exact `Q_a = Q - Q H^T (H Q H^T + R)^-1 H Q` to
  within sampling error — a direct numerical check of Section 1's identity,
  independent of any ionosphere-specific code.
- **Per-style sanity cycle**: run one assimilation cycle for each of
  `raw`, `density_10ex`, `ANCHOR`, `PCA_3D` and confirm the decoded,
  updated EDPs are physically sane (no negative densities, no NaNs,
  reasonable reconstruction error).
- **OSSE-based recovery check** (see 5.8): the strongest available test —
  since the truth is known, this checks not just "sane output" but that
  the analysis actually moves *towards* the truth and that the reported
  analysis covariance is consistent with the actual error. Run this per
  style, and specifically for `ANCHOR`/`PCA` with `iterated` vs `linear`
  (Section 6), since that comparison is exactly what will show whether
  Remedy A/B are earning their cost.
- Save comparison plots to `Runs/Tomography_Test/Claude_Test`, per existing
  convention for this project's throwaway/verification artifacts.

### 5.8 OSSE (Observation System Simulation Experiment) support

An OSSE substitutes a known "truth" state for reality: synthetic
observations are generated by applying the *same* observation operator the
engine uses for real assimilation to a chosen truth state (plus noise drawn
consistent with `R`), so the filter can be scored against a known answer
instead of unverifiable real TEC data.

**Where it lives, and why:** the engine already requires an observation
operator (decode + `H_grid`, Section 4) and `R` as inputs to do real
assimilation — OSSE only needs that same interface, applied once to a truth
state instead of once per ensemble member. So OSSE support is a **small,
separate utility living alongside the engine in `Ensemble_Kalman_Engine/`**
(e.g. `osse.py`), not a hook inside the analysis engine's control flow, and
not something that waits on the full GNSS/LEO ray-geometry observation-
supply pipeline to be production-ready:

```
truth state x_true  +  observation operator  +  R
                    |
                    v
        y_obs = observation_operator(x_true) + noise(R)
                    |
                    v
        hand (y_obs, R) to the engine's normal analyze()/assimilate()
        entry point -- IDENTICAL code path to real data, engine has
        no "truth-known" special case
```

This split matters for correctness, not just convenience: if the engine had
an internal truth-aware code path, it would be tempting (even
unintentionally) to special-case OSSE behavior, which would invalidate the
test — the whole point of an OSSE is that the filter runs *blind* to
whether `y_obs` is real or synthetic, through the same code every time.

**What doesn't move:** the GNSS/LEO ray-geometry-to-`H_grid` construction
(satellite motion, ray tracing) stays exactly where it already lives
(`EDPSamples.get_observation_operator`, `observation_preparation/`) —
OSSE doesn't replace or duplicate that. It only needs *an* observation
operator instance, real-geometry-based once available, or a toy/synthetic
one for early unit tests of the engine itself, well before real data is in
the loop.

**Practical use:** `x_true` can be a single draw from the same
`EDPSamples`/climatology ensemble used to seed the forecast (held out, not
included as a member), or a hand-specified profile for a fully controlled
test. Pairs naturally with 5.7's validation plan — an OSSE with a known
`x_true` turns "does the output look physically sane" into "does the
analysis measurably converge towards a known answer, with a self-consistent
reported covariance."

## 6. Handling strong nonlinearity: successive linearization

Source: `Extended_ENsemble_Kalman_Filter.pptx` (same Presentation folder).
This deck directly addresses a real gap in Section 5: a single-shot ensemble
update is not reliable when the parameterization is strongly nonlinear and
the ensemble is deliberately wide (the normal case for a climatological
prior with ANCHOR or PCA parameters).

### 6.1 Why the plain EnKF gain can fail here

Restating the deck's own setup: TEC is *linear* in physical electron
density `E`, `y_TEC,j = W_j . E`, `W in R^(M x NK)` — this is exactly the
`H_grid` from `EDPSamples.get_observation_operator()`. Nonlinearity only
enters through the parameterization's decode step `E(P)`. For a nonlinear
observation `y = f(x) + eps_y`, the deck shows

    y - y_f = f(x) - f(xbar) + eps_y
            = grad_x f(xbar) . (x - xbar) + eps_y + O(||x - xbar||^2)

The ensemble-covariance gain `K = C_XY (C_YY + R)^-1` is implicitly a
first-order (linearized) correction — it is exact only when the
`O(||x-xbar||^2)` remainder is negligible. Since the ensemble is
*deliberately* spread out to cover the plausible range of the prior, this
remainder need not be small, so there is **no guarantee the one-shot
analysis even moves the state in the right direction**, let alone that it
is minimum-variance. This is the concrete failure mode behind the
suspicion that `enkf_update.py`, applied as-is to a highly nonlinear
parameterization, may not be trustworthy — the math is correct for the
linear case (Section 1) but the deck itself flags that it is not
sufficient for the nonlinear one.

### 6.2 Remedy A — Iterated EKF using existing Jacobians (confirmed default for nonlinear styles)

The deck's iteration (slide 6):

    P_{i+1} = P_i + G_i (ybar - W . E(P_i)),
    G_i = alpha * Q * W_hat_i^T (W_hat_i Q W_hat_i^T + R)^-1,
    W_hat_i = W . grad E(P_i)

iterated until `P_{i+1} - P_i` is sufficiently small, `alpha` acting as a
damped-Newton / trust-region step size.

Every piece of this already exists in the codebase:
- `W` = `EDPSamples.get_observation_operator()` — computed once, reused
  across iterations.
- `grad E(P_i)` = `EDP_Parameterization.get_Jacobian(P_i)` — already
  implemented for `density_10ex`, `ANCHOR`, and every `PCA_*` style.
- `Q` need **not** be recomputed or re-sampled each iteration: estimate it
  once, up front, from the initial ensemble as `Q ~ S S^T` (Section 1),
  exactly as the current code already does. Only the *observation*
  linearization `W_hat_i` changes between iterations, not the ensemble.

So the practical design is a hybrid: one ensemble draw from `EDPSamples`
gives `Q` (cheap, done once), and the analysis step becomes a short
Gauss-Newton-like loop (a handful of iterations, `alpha`-damped, converged
when the update stalls) that re-linearizes only the observation operator
at each step. Within `Ensemble_Kalman_Engine` (5.0), `linear` mode is the
one-shot single-pass case of this same loop (zero re-linearizations); the
two modes share one gain-matrix implementation rather than being separate
code paths.

Once `P_i` converges, generate the analysis ensemble `S_a` for the next
assimilation cycle using the deck's original perturbed-observation formula
(Section 1) with the *converged* linearized operator `W_hat` — i.e. the
iteration refines the mean update; the existing stochastic-update code
still produces the propagated ensemble spread around it.

**Confirmed: `iterated` is the default analysis mode for every nonlinear
parameterization** (`ANCHOR`, `density_10ex`, every `PCA_*` style) — not an
opt-in refinement. `linear` (today's one-shot update) remains available as
a cheap option, primarily meaningful for `raw` (where the forward map is
genuinely linear in the state, so iterating would converge in one step
anyway).

**Confirmed iteration hyperparameters:** `alpha` starts at 1.0 with
backtracking (halve on a diverging/non-decreasing residual); a cap of 5-10
Gauss-Newton iterations is reasonable given the parameterizations in use are
typically very smooth; convergence when `||P_{i+1} - P_i|| / ||P_i||` falls
below a small relative tolerance (e.g. 1e-3) or the cap is hit.

### 6.3 Remedy B — Ensemble nearest-analog recentering (general-purpose technique, not just a nonlinear fallback)

**Confirmed: this is worth building as a general improvement, not only a
Jacobian-free fallback for hard nonlinear cases.** The deck's slides 9-11
motivate it for nonlinearity, but the same idea is valuable even for a
*linear* forward model: rather than systematically centering the update on
the ensemble mean `xbar` (which may not be a representative/plausible state
at all, especially for a multimodal or skewed prior), center it on the
ensemble member `i*` whose forecast is closest to the actual observation,

    i* = argmin_i || Y - Y^i ||,      x_a = x_(i*) + K (Y - H x_(i*))

optionally re-deriving `C_XY`/`C_YY` from a local neighborhood of `i*`
rather than the full ensemble. The rationale generalizes beyond the
nonlinearity argument in Section 6.1: the nearest-analog member is, by
construction, already consistent with the observation to a good
approximation, so the correction the gain has to supply is smaller and less
reliant on the (possibly poor, for a nonlinear or non-Gaussian prior)
mean-centered linearization.

### 6.4 Design implication for the general module (Section 4)

The analysis step is two **orthogonal** choices, not a single mode switch:
1. **Linearization strategy**: `linear` (one-shot, Section 1) vs. `iterated`
   (Remedy A, Section 6.2) — `iterated` is the confirmed default for every
   nonlinear parameterization (`ANCHOR`, `density_10ex`, `PCA_*`); `linear`
   remains available and is the natural choice for `raw`.
2. **Centering strategy**: `mean` (today's default, centers on `xbar`/`Y_f`
   mean) vs. `nearest_analog` (Remedy B, Section 6.3) — confirmed worth
   building generally, applicable to *either* linearization strategy above
   (i.e. `iterated` can re-select/re-center on the nearest analog at each
   step too, not just as a one-shot alternative to `linear`).

`EDP_Parameterization.is_lossy`/`needs_altitude` already tell us per-style
metadata; a similar per-style hint (e.g. "has cheap Jacobian") can pick a
sensible default linearization strategy automatically, while centering
strategy defaults to `nearest_analog` across the board.

**Confirmed: when both are active, re-select the nearest-analog member at
every Gauss-Newton iteration** (the more principled option — the
linearization point moves each iteration, so the "closest forecast to the
observation" should be re-evaluated against it), accepting the extra
forward-search cost this implies. **Required escape hatch:** a config
option to disable Remedy B (fall back to `mean` centering) independent of
the `linear`/`iterated` choice, so it can be switched off later if the
combined cost proves too slow in practice — this should be a cheap toggle,
not a design that requires Remedy B to be present.

### 6.5 Note on localization (slide 12, separate topic)

The deck separately comments on the existing Gaspari-Cohn localization in
`enkf_update.py`: it scales down the update for state components far from
any occultation ray, which matters less when occultations cover the domain
near-uniformly. This isn't a correctness bug, just a design note — worth
revisiting if/when diagnostics show it's suppressing legitimate updates,
but out of scope for the nonlinearity fix above.

### 6.6 Note on state size `N` vs. ensemble size `n` for ANCHOR

**Confirmed clarification:** the 8-parameter ANCHOR state is per *vertical
profile* — the actual filter state size is `N = 8 * n_horizontal_grid`,
with `n_horizontal_grid` on the order of several hundred, so `N` is order
several-thousand. With `n ~ 2000` ensemble members, `N` is not far above
`n` for ANCHOR (unlike the deck's own motivating raw-grid case, where `N`
can be far larger than any affordable ensemble). Practical implication:
`S`/`Q` is still rank-deficient (`n < N`) and the low-rank trick (Section 1)
and localization both remain relevant, but the degree of rank-deficiency is
much milder for `ANCHOR`/`PCA` than for `raw`/`density_10ex` at production
mesh sizes. **Decision: continue with the existing localization approach as
implemented — no dedicated rank/tuning pass planned for this reason alone.**

## 7. Status: open questions resolved

All open questions from the previous round are now resolved (see the
"Confirmed" notes throughout Sections 5-6): parameterization scope (5.1),
the perturbation hook design (5.3), retirement of the ANCHOR fast path
(5.4), the N-vs-n rank-deficiency note (6.6), `iterated` as the default
nonlinear analysis mode with its hyperparameters (6.2), and combining it
with nearest-analog recentering, including the required off-switch for
Remedy B (6.4). No unresolved design questions remain — the plan (Sections
1-6) is ready for implementation to begin whenever you give the go-ahead.

## 8. Implementation status (2026-09-18)

First implementation pass complete and tested in `Ensemble_Kalman_Engine/`
(42 new tests, plus the 82 pre-existing `Parameterization/` tests still
passing after the 5.1 `'raw'`-style addition -- run with
`/opt/anaconda3/bin/python3 -m pytest Ensemble_Kalman_Engine/ Parameterization/ -q`).

**Built and tested:** 5.0 (module layout, no import dependency on the old
pipeline), 5.1 (`'raw'` style), 5.2 (`EnsembleState`), 5.3 (perturbation
hook: dither + selective inflation + bounds clamp, composable), 5.4
(`GenericObservationOperator`, validated against real `ANCHOR`/
`density_10ex`/`raw` Jacobians via finite differences), the analysis-engine
core math (5.5/Section 1, closed-form-checked), Section 6's `linear`/
`iterated` x `mean`/`nearest_analog` engine, 5.8 (OSSE). An end-to-end OSSE
recovery test now runs against real IRI-sampled data
(`TestCode/EDPSam_Point.nc`) through `GeneralEnKFDriver` (5.6) for all
three of `raw`/`density_10ex`/`ANCHOR`, with a synthesized (non-orbital)
vertical ray geometry standing in for real LEO/GNSS occultation paths.

**A correction found while testing (Section 6.2):** the deck's slide-6
iteration, `P_{i+1} = P_i + G_i(ybar - W.E(P_i))`, taken literally past the
first step, drops the prior's pull once `P_i` moves away from the prior
mean -- a toy-problem test caught it converging to a *worse* truth-recovery
error than the one-shot `linear` update, overfitting observation noise. The
implementation instead uses the standard MAP/Gauss-Newton-with-prior
recursion (anchors every correction to the original prior point `x_p`,
adding back a `W_hat_i (x_i - x_p)` term), which matches the deck exactly
on the first iteration and correctly balances prior vs. data on later ones.
See `analysis_engine.py`'s inline comment on this branch for the full
derivation.

**Not yet done / explicitly deferred:**
- Real LEO/GNSS occultation geometry (`podTc2_data` from
  `observation_preparation/`) has not been substituted for the synthesized
  vertical-path geometry used in the real-data test above.
- `PCA_1D`/`PCA_3D` styles are covered by the `jacobian_utils` unit tests
  (synthetic arrays matching their real shapes) but not yet by an
  integration test against a fitted `EDP_Parameterization` PCA basis or
  real data, unlike `raw`/`density_10ex`/`ANCHOR`.
- The old `Ionosphere_Tomography_Inverter/observation_operator.py` (the
  bespoke ANCHOR analytic integrator) has **not** been deleted. Section 5.4
  says it "becomes dead code to remove" once the generic route is wired in
  -- but the *old* pipeline (`enkf_update.py`,
  `Ionophy_Tomography_Inverter_EnKF.py`, etc.) still imports and uses it,
  and Section 5.0 says that pipeline is left untouched. Deleting it would
  break currently-working code outside this plan's scope, so it's left in
  place pending an explicit decision on the old pipeline's fate.
- The iteration hyperparameters (5-10 cap, `alpha` backtracking, 1e-3
  tolerance) and the perturbation hook's dither/selective-inflation
  defaults are implemented exactly as confirmed, but have not been tuned
  against real assimilation cycles beyond what the tests above exercise.
