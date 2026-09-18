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
a pluggable choice, not something the filter engine hardcodes — while
reusing the already-correct EnKF math in `enkf_update.py` and the already-
generic `Parameterization.py` / `EDPSamples` infrastructure, rather than
rewriting either.

## 4. Proposed architecture

```
 EDPSamples (raw ensemble, IRI-drawn)
        |
        v  Parameterization.get_parameter()   [encode, any style]
 Parameterized_EDPSamples.param_vec  -------->  filter-space ensemble X_f
        |                                        (n_state, n_members)
        |                                             |
        |                                             v
        |                                    ParametricObservationOperator
        |                                      decode: get_density(X_f[:,i])
        |                                      predict: H_grid @ flatten(.)
        |                                             |
        |                                             v
        |                                        Y_f (n_obs, n_members)
        |                                             |
        v                                             v
        +-------------------->  enkf_update()  <------+
                            (generic linear algebra,
                             already implemented)
                                    |
                                    v
                       X_a (n_state, n_members)
                                    |
                                    v
                     style-specific post-update hook
                (bounds clamp / re-normalization, optional)
                                    |
                                    v
            Parameterization.get_density()  ->  updated EDPSamples
             (decode for diagnostics / next-cycle forward propagation)
```

Everything below the first arrow is representation-agnostic: swapping
`style='ANCHOR'` for `style='PCA_3D'` or a new `style='raw'` should not
require touching the filter engine.

## 5. Work items

### 5.1 Add a `'raw'` (identity) parameterization style
Trivial addition to `Parameterization.py`: `to_density` / `to_parameter` are
the identity, `get_Jacobian` returns an identity diagonal. This gives the
direct-grid case the deck's own motivating example describes (state = raw
electron density mesh values) as just another pluggable style, and is a good
first correctness check (encode/decode is a no-op).

### 5.2 Generalize the ensemble/state container
Replace the ANCHOR-only `IonosphericState` (or add a sibling to it) that:
- Wraps a filter-space ensemble as a plain `(n_state, n_members)` array,
  regardless of whether `n_state` came from `(8, n_geo)` (ANCHOR),
  `(n_height, n_geo)` (raw/log10), or `(nPCA,)` (PCA_3D).
- Is constructed from a real `EDPSamples` draw via
  `Parameterized_EDPSamples` + `EDP_Parameterization.get_parameter`, so the
  initial ensemble is always real IRI-sampled data, not synthetic Gaussian
  draws (satisfies "leverage EDPSamples as the initial ensemble").
- Exposes `ensemble_mean`, flatten/unflatten, and a pluggable, optional
  post-update hook (bounds clamp for ANCHOR; identity for everything else).

### 5.3 Generic observation operator
New `ParametricObservationOperator` (or generalize the existing one) that:
1. Decodes every ensemble member with `EDP_Parameterization.get_density`.
2. Applies the precomputed, style-independent `H_grid` from
   `EDPSamples.get_observation_operator()`.

Keep the existing bespoke analytic Chapman/ANCHOR integrator
(`observation_operator.py`) available as an optional fast path for that one
style — but it must be validated to agree numerically with the generic
decode-then-`H_grid` route (see 5.5). This is the piece of real, new
numerical work in the plan; everything else in this section is mostly
wiring.

### 5.4 Decouple `enkf_update()` from ANCHOR-specific assumptions
The core gain/update math already operates on generic `(n_state, n_members)`
arrays and needs no changes. Two things currently assume the ANCHOR layout
and should move out to per-style hooks instead:
- Localization reshape (`n_params = n_state // n_grid` tiling) — meaningless
  for `PCA_3D`, where there is no "N parameters per grid point" structure.
- `clamp_to_physical_bounds()` — an ANCHOR-specific structural constraint
  (e.g. `hmF2 > hmE + 20`); should become an optional, style-supplied
  post-update hook rather than an implicit step.

### 5.5 New orchestration driver
A driver parallel to `Ionophy_Tomography_Inverter_EnKF.py`, parameterized by
`style`, running the same assimilation loop regardless of representation.

### 5.6 Validation / testing
- **Closed-form check**: on a small toy problem (small `N`, known `H`),
  verify `Q_a ~ S_a S_a^T` (sample covariance of the generated analysis
  ensemble) matches the exact `Q_a = Q - Q H^T (H Q H^T + R)^-1 H Q` to
  within sampling error — a direct numerical check of Section 1's identity,
  independent of any ionosphere-specific code.
- **Forward-model cross-check**: on real data, confirm the generic
  decode + `H_grid` observation operator agrees with the existing ANCHOR
  analytic integrator to numerical precision.
- **Per-style sanity cycle**: run one assimilation cycle for each of
  `raw`, `density_10ex`, `ANCHOR`, `PCA_3D` and confirm the decoded,
  updated EDPs are physically sane (no negative densities, no NaNs,
  reasonable reconstruction error).
- Save comparison plots to `Runs/Tomography_Test/Claude_Test`, per existing
  convention for this project's throwaway/verification artifacts.

## 6. Open questions before starting

- For non-ANCHOR, non-PCA styles (`raw`, `density_10ex`), the filter state
  is `(n_height * n_geo)`-dimensional per member — potentially large. Is the
  intended initial use case regional/small-mesh only, or should the plan
  budget time for a sparse/low-rank-friendly `H_grid` from the start?
- Should the post-update bounds/constraint hook be mandatory-but-often-a-
  no-op (uniform interface for every style), or genuinely optional
  (skipped entirely for styles that don't need it)?
- Do you want the ANCHOR analytic fast-path integrator kept long-term as a
  performance optimization, or is it acceptable to retire it once the
  generic decode+`H_grid` path is validated as equivalent?
- Ensemble size `n` vs. state size `N`: for `raw`/`density_10ex`, `n` will
  be far smaller than `N` (as in the deck's own motivation), so results may
  be more sensitive to sampling noise/localization than the current
  8-parameter ANCHOR case. Worth a dedicated tuning pass, or out of scope
  for this plan?
