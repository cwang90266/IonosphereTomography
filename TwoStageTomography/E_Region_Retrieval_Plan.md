# E-Region Electron Density Retrieval from RO TEC — Plan

Status: **draft for review**, 2026-11-01. Nothing in the package is
changed yet. The only new code is the read-only diagnostic
`TwoStageTomography/baseline_e_region_diagnostics.py` and the ionosonde
downloader `TwoStageTomography/fetch_tromso_dynasonde.py`.

**Decisions so far (2026-11-01):**

1. The primary E-region products are **NmE and hmE**, validated against
   **ionosonde** data. Full E-region profiles are a secondary product.
2. Tromsø ionosonde data: found on Madrigal (EISCAT Tromsø Dynasonde) for
   every day of the October 2025 runs (§5.7).
3. High-rate conPhs data is not available by default. The plan proceeds
   on 1 Hz podTc2 data; high-rate data is an optional later step, to be
   requested only if WP0 shows it is needed.
4. The E-region study uses two styles only: **ANCHOR** and
   **PCA_1D_10ex**. PCA_3D_10ex and PCA_1D_10ex_ISR are dropped from this
   study.
5. The method will be evaluated **over all seasons**. October 2025 is the
   development set only, and other periods will be tested later. Nothing
   season-specific may be tuned on October alone (§5.7).

Companion file: `Literature_Review.md` (what the community has done and
what we take from it).

---

## 1. Goal and the core difficulty

**Goal.** Get as much E-region (90–150 km) electron-density information as
possible out of the RO TEC profiles we already assimilate, without
degrading the F-region analysis that already works. The products to
deliver and validate are **NmE and hmE**, scored against ionosonde
foE/hmE.

**Core difficulty.** A limb ray with its tangent point at 100 km crosses
the F region twice, on both sides of the tangent point, over a path of
thousands of km. The E layer adds only a few percent to that ray's TEC.
The E-region signal is therefore small, and it is easily aliased into (or
out of) the F region.

**The opening.** The E-region signal is small in *absolute* TEC but it has
a distinctive *shape* in TEC versus tangent height. Below ~150 km the F
contribution changes smoothly (almost linearly) with tangent height, while
the E layer adds curvature over a 20–40 km band. Also, the TEC *changes*
along one occultation (from carrier phase) are far more precise than the
absolute TEC level (which carries DCB and leveling offsets). Most of the
plan comes from using this shape and this relative precision, instead of
fitting absolute TEC per ray with a 5 TECU error.

---

## 2. Baseline: what the October 2025 runs tell us

Source: `parallel_tomography_output_test` (30 hours, 2025-10-14 to -17,
Tromsø-centred 2000 km ROI, 4 styles × {no_boost, boosted}). I scored it
with `Run_Scoring/score_runs.py` and ran the new diagnostic
`baseline_e_region_diagnostics.py`. The `memory_tracking` runs (boost 0.2,
0.3 and 0.4) use the same set-up and only change the boost amplitude.

### 2.1 ISR skill (boosted; the two study styles)

| Style | Station | E rel. err. forecast → analysis | F rel. err. forecast → analysis | E gates moved the wrong way (median) |
| --- | --- | --- | --- | --- |
| PCA_1D_10ex | TRO | 169 % → 57 % | 59 % → 24 % | 13 % |
| PCA_1D_10ex | SVL | 30 % → 45 % (**worse**) | 51 % → 30 % | 60 % |
| ANCHOR | TRO | 87 % → 59 % | 58 % → 28 % | 25 % |
| ANCHOR | SVL | 22 % → 42 % (**worse**) | 53 % → 40 % | 50 % |

PCA_3D_10ex is almost identical to PCA_1D_10ex. ANCHOR starts from a
better E-region forecast (its fitted Chapman E layer smooths out IRI's
E-region shape) but ends at a similar analysis error.

- **F region:** improves consistently at both sites.
- **TRO E region:** the IRI prior is very poor (169 %; auroral E region).
  The analysis improves it mostly by pulling it back toward a plausible
  level, not by resolving E structure. hmE/NmE errors stay large: NmE
  65–95 %.
- **SVL E region:** the analysis makes it *worse* in most hours. The
  correction-direction metric shows the E-region change goes the wrong way
  in about 60 % of gates. This is the clearest sign that the E-region
  update is a by-product of F-region corrections carried through the
  prior's E–F cross-covariance, not a response to E-region information.
- ISR E peaks are rare (3–4 hours per site), so NmE/hmE statistics are
  thin.

### 2.2 Why the data currently cannot constrain the E region

From `baseline_e_region_diagnostics.py` (boosted, PCA_3D_10ex, all 30 h):

| Tangent alt. (km) | Rays | Median TEC | Median E-region part | Analysis residual RMS / mean |
| --- | --- | --- | --- | --- |
| 60–90 | 676 | 104 | 7.4 TECU | 3.2 / −1.7 |
| 90–110 | 478 | 113 | 8.5 TECU | 3.5 / −1.8 |
| 110–130 | 485 | 121 | 6.3 TECU | 2.9 / −0.6 |
| 130–150 | 510 | 122 | 1.4 TECU | 2.8 / −1.5 |
| 150–200 | 1415 | 139 | 0 | 2.8 / −1.7 |
| 200–400 | 7540 | 98–157 | 0 | 1.9–2.1 |
| 400–600 | 29530 | 35 | 0 | 1.9 / −0.1 |

Residual lag-1 autocorrelation within an occultation: **0.97** (boosted)
and 0.99 (no_boost).

Five findings follow:

1. **Signal ≈ noise as currently specified.** The whole E-region
   contribution to a low ray is 6–9 TECU. The run uses
   `OBS_SIGMA_TECU = 5.0` per ray. The analysis already fits E rays to
   ~3 TECU, inside σ. The filter has no reason to fit the E structure
   better, and any E-region error below ~5 TECU is invisible to it.
2. **The observation weight sits in the topside.** 71 % of rays have
   tangent points at 400–600 km; only 4.5 % are in 90–150 km. This comes
   from the uniform *time* stride (`max_rays_per_occultation = 200`) plus
   the slow tangent descent in the upper part of a profile.
3. **Decimation discards most E-region samples.** On 20 raw podTc2 files
   (1 Hz) there are 40–270 samples per occultation in 90–150 km. The
   stride (6–16) keeps only 4–20 of them, with a median spacing of about
   15 km (range 4–26 km). That is comparable to or coarser than the E-layer
   thickness. The stride also picks samples (aliasing) rather than
   averaging them, so it does not even reduce noise.
4. **The errors are not white.** A lag-1 residual autocorrelation of 0.97
   means the misfit is dominated by smooth, per-occultation errors (offset
   or leveling bias, slow drifts, representation error), not independent
   ray noise. A diagonal R with σ = 5 TECU is the wrong error model: it
   over-weights the redundant absolute level and under-weights the precise
   ray-to-ray shape that carries the E information.
5. **The grid and the prior are coarse in the E region.** The model grid
   is uniform 10 km from 90 to 900 km, so 90–150 km is six levels and the
   grid starts at 90 km while 676 rays have tangents below 90 km. The IRI
   ensemble varies the E region mostly together with the F region
   (driving indices), and the boost is uniform below 400 km. The PCA
   bases are fitted on the whole column, so E-region modes with small
   variance are the first to be cut by `retaining_threshold`.

Other things to note: the mean residual is negative (model > observed) on
rays below 200 km. The tomography is *not* bound by Abel's spherical
symmetry, since it models horizontal gradients, but the ISR comparison is
a point measurement while the E-region information in a ray is spread
over ~1000 km along the ray.

---

## 3. What the community does (summary; details in `Literature_Review.md`)

- **Plain Abel inversion is poor in the E region**, even in daytime
  (Kelley et al. 2009, vs Arecibo). Spherical symmetry errors propagate
  down from the F region and often give negative E-region densities (Yue
  et al. 2010; Shaikh et al. 2014).
- **Aided Abel**: horizontal gradients are supplied from outside, using
  VTEC maps (separability: Hernández-Pajares et al. 2000; Garcia-Fernandez
  et al. 2003; Aragón-Ángel et al.), a background model (IRI/NeQuick-aided:
  Wu et al. 2019), or a data-assimilation model (Yue et al. 2012). These
  give the largest documented E-region improvements. The principle is to
  **get the F region from elsewhere, then let the low rays speak for the
  E region**, which is our two-stage idea.
- **Removing the F contribution as a smooth background**: Wu (NASA GSFC;
  Remote Sensing 2022) fits a linear function to the excess phase above
  the E region and removes it before retrieving D/E-region Ne from
  high-rate RO. Wu et al. (2023) use optimal estimation on slant TEC and
  note the strong relation between hTEC and its vertical derivative.
  These retrievals use the *shape* of the low part of the profile.
- **High-rate data**: 50 Hz open-loop sampling gives tens of metres in
  tangent height in the lower part of the occultation and is what sporadic
  E work uses.
- **Sporadic E**: detected from L1 SNR fluctuations (Arras et al. 2008),
  TEC-perturbation gradients and S4 (Carmona et al. 2022 compare five
  methods). Es layers are thin and break any smooth E-layer prior. They
  need to be flagged. A 2025 J. Geodesy paper reconstructs Es layers in
  3-D with RO constraints.
- **Tomography and assimilation**: two-step tomography (Tikhonov then
  MART; NeQuick then MART) and multi-instrument fusion are common for the
  F region. EOF/IRI-ensemble covariances (Lin, Matsuo et al. 2015, 2017)
  are the closest analogue to our PCA styles. I found **no published
  F-then-E two-stage RO tomography with a refined E-region grid**. Our
  plan is new in that combination, but each piece has a precedent.
- **Validation**: ISR (Kelley 2009; Stolle et al. 2004 used EISCAT for
  CHAMP), ionosondes (foE), and the E-PROBED empirical RO E-region model
  (Salinas et al. 2024). E-PROBED agrees with ISR in the daytime but not at
  night, which tells us night-time E is where expectations should be
  lowest.

---

## 4. Strategy

The four ideas in the request map onto the plan as follows, along with
two more that the baseline shows to be needed:

| # | Idea | Plan section | Expected impact |
| --- | --- | --- | --- |
| A | Better observation-error model and observable (relative / differenced TEC, per-occultation bias) | §5.2 | **High.** This is what lets the E signal exceed the noise. |
| B | Full-resolution (not decimated) TEC in the E region (request item 2) | §5.1 | High once A is in place; small without it |
| C | Finer retrieval grid in the E region (request item 3) | §5.3 | Medium: reduces representation error, needs a matching prior |
| D | E-region degrees of freedom in the prior: separate E basis, E-specific boost (request item 4, first half) | §5.4 | High. Without it the update stays slaved to F. |
| E | Two-stage tomography: freeze F, solve E (request item 4, second half) | §6 | High; also makes B–D cheap |
| F | Large residuals in the lower TEC profile (request item 1): use them as a diagnostic, QC sporadic E | §5.5 | Protects against wrong-direction updates |

Ordering principle: **measure the information content first (OSSE + DFS),
then change the real-data system one lever at a time**, scored with the
existing `Run_Scoring` tools on the same 30 hours.

---

## 5. Work packages

### 5.0 WP0 — Information-content study (OSSE) — do first

The question to answer: *with a given sampling, error model and grid, how
many independent pieces of E-region information do the rays carry?*

- Use `Ensemble_Kalman_Engine/osse.py` with real ray geometry from a few
  representative hours. Truth = an ISR-based or perturbed-IRI field with a
  realistic E layer (NmE 0.5–2 × 10¹¹ m⁻³, hmE 100–120 km, and some cases
  with a thin Es layer).
- For each configuration, compute:
  - the **E-region degrees of freedom for signal**,
    DFS_E = trace over E levels of the averaging kernel
    A = K H, using the linear observation operator H and the ensemble
    covariance. `jacobian_utils.py` / `ObservationOperator.linearized`
    give H for the PCA styles;
  - the E-region posterior/prior spread ratio;
  - the analysis error against truth for E and F, and the share of the
    E-region change that is correct.
- Configurations: {current stride, full 1 Hz below 200 km, high-rate
  conPhs} × {diagonal σ = 5, bias + white noise model, differenced
  observable} × {10 km grid, 2.5 km E grid} × {current prior, E-specific
  prior (§5.4)} × {ANCHOR, PCA_1D_10ex}.
- Truth fields must not come from ANCHOR's own Chapman shape alone (that
  would favour ANCHOR). Use ISR/IRI profiles that include an E valley and
  thin layers.
- Also score the OSSE directly on the products: NmE and hmE error
  against truth, with the same peak finder as `score_runs.py`.
- **Deliverable:** a table of DFS_E, NmE error and hmE error per
  configuration. It tells us which levers matter before we spend
  real-data runs on them. Since NmE and hmE are the targets, DFS_E ≈ 2
  near the station is enough. If DFS_E stays below ~1 for every
  configuration, RO TEC alone can at most give one E-region number (NmE,
  or an E-layer content), and hmE would have to come from the prior.

### 5.1 WP1 — Observation sampling (request item 2)

Code: `observation_preparation/ro_source.py::_build_clean_ro_entry`.

1. **Altitude-adaptive sampling**: replace the single uniform stride with
   a tangent-altitude-dependent target spacing, for example:
   - below 160 km: every 1 Hz sample (about 0.2–3 km spacing);
   - 160–300 km: ~5 km;
   - above 300 km: ~10–20 km.

   Add this as a new option (for example `sampling="altitude_adaptive"`
   with `spacing_km` breakpoints) and keep `max_rays_per_occultation` as a
   cap.
2. **Bin-average instead of picking**: when decimating, average TEC (and
   use the mean geometry) over each bin. This lowers noise by √n instead
   of aliasing it. The ray operator then sees the bin's central ray. Check
   that the geometry error of the central ray stays negligible.
3. **High-rate data (optional, later)**: conPhs files are not produced by
   default, so the plan relies on 1 Hz podTc2, which already gives
   0.2–3 km spacing in the E band once decimation stops discarding it.
   Request conPhs only if WP0 shows that finer sampling raises DFS_E or
   hmE skill materially. `TEC_model/conPhs_file_processing.py` already
   parses those files into relative TEC, which suits the differenced
   observable in §5.2.
4. **Keep the ray counts balanced**: with all samples kept, the E-region
   rays rise from ~5 % to perhaps 30 % of the total. This is intended, but
   watch the F-region skill. If it drops, down-weight with the error model
   (§5.2), not by discarding samples.

Expected: little effect on its own with σ = 5 TECU (finding 1). The real
gain comes together with WP2.

### 5.2 WP2 — Observation error model and observable (enables everything else)

1. **Measure the noise.** Estimate the ray-to-ray (high-frequency) TEC
   noise per occultation from the residual after a smoothing spline at
   tangent heights above 600 km, where the true TEC is smooth. Also
   estimate it from the residual after a local quadratic fit in
   90–150 km. Compare with the S4/σφ fields in podTc2. My expectation is a
   few hundredths to tenths of a TECU, far below 5 TECU. This must be
   measured, not assumed.
2. **Error model R = bias + correlated + white.** Per occultation:
   - a per-occultation offset with σ_b ≈ several TECU (DCB/leveling);
   - a smooth correlated component (exponential in tangent height, length
     ~50–100 km) covering representation and slow drift errors;
   - a white component equal to the measured noise, plus a representation
     term that grows below 150 km for grid/basis error.

   `kalman_core` already takes a full R. Building it per occultation keeps
   it block-diagonal and cheap.
3. **Alternative: differenced observable** (simpler, more robust).
   Assimilate z = D y, where D takes differences between neighbouring
   samples along each occultation (first difference = the TEC "slope", the
   Abel-relevant quantity). The offset cancels exactly. H_z = D H is
   cheap. Noise becomes correlated (σ√2, lag-1 = −0.5) and must enter
   R_z = D R Dᵀ. A practical hybrid: keep a few absolute TEC values per
   occultation (anchoring the level, mainly for the F region) plus
   differenced values below ~200 km.
4. **Optional bias state**: add one offset parameter per occultation to
   the state vector (state augmentation). This is the cleanest option, but
   it needs a change in how `Parameterization` builds the state.
   Recommendation: start with option 3, and keep options 2 and 4 for
   comparison.

Expected: the E-region signal becomes several times larger than the
effective noise. This is the single most important change.

### 5.3 WP3 — Retrieval grid in the E region (request item 3)

Code: `EDPSamples` (the altitude array goes straight into the IRI
namelist `height_grid`, so non-uniform grids are possible).

- New altitude grid (to be confirmed by WP0):
  - 80–160 km at 2.5 km (33 levels; bottom moved from 90 to 80 km so rays
    with tangents below 90 km see a modelled region);
  - 160–300 km at 10 km;
  - 300–900 km at 20 km.

  This totals about 77 levels, roughly the current 82, so the cost is
  similar.
- Check the IRI2020 `height_grid` size limit and that the ray operator
  integrates correctly on a non-uniform grid (trapezoid weights and the
  interpolation scheme). The pixel-integration step along each ray must be
  finer than 2.5 km near the tangent point.
- Horizontal resolution is not refined: a low ray averages over ~1000 km
  along its path, so the E region cannot be horizontally resolved better
  than the F region from RO alone.
- Refit PCA bases on the new grid. Check (see WP4) that E-region modes
  survive the `retaining_threshold`.

### 5.4 WP4 — E-region degrees of freedom in the prior (request item 4, part 1)

The SVL wrong-direction result shows that today the E update is controlled
by the E–F cross-covariance of an IRI ensemble, not by the data. The fix
differs by style.

**ANCHOR** (state: log10 NmF2, hmF2, H0, γ, B0, B1, **log10 NmE, hmE**
at each grid point; the E layer is a Chapman layer with a fixed 15 km
scale height; the F2 bottomside joins it at the transition height h_ST):

- A1. **The products are state variables.** NmE and hmE come straight from
  the analysis, with no peak-finding. Their ensemble spread is a direct
  uncertainty estimate to report alongside them.
- A2. **E-specific spread**: inflate only the (log10 NmE, hmE) components.
  `kalman_core.apply_inflation` already takes a per-component inflation
  array, so no new code is needed. Set the prior spread from the
  ionosonde/ISR NmE and hmE variability relative to IRI, not from the
  IRI ensemble, which is too narrow.
- A3. **E–F decorrelation**: set the cross-covariance between the two E
  parameters and the six F parameters to zero or scale it by α. In
  parameter space this is a simple mask on the gain, or done by shuffling
  the E parameters between members.
- A4. **Coupling through h_ST**: NmE also sets where the bottomside joins
  the E layer, so a change in NmE moves the 120–200 km valley region.
  Check in WP0 whether this coupling aliases F bottomside error into NmE.
- A5. **Optional E scale height**: make the fixed 15 km a 9th parameter,
  either free or constrained by a tight prior. Test it only if hmE or NmE
  errors show a systematic layer-width bias.
- A6. **Grid matters most here**: on the 10 km grid a 15 km-scale Chapman
  layer is sampled at only ~3 points across its peak, so the ray integral
  of the E layer has discretization error. The 2.5 km E grid (§5.3)
  removes this directly.

**PCA_1D_10ex** (state: vertical PCA coefficients at each grid point,
one shared basis):

- P1. **Block parameterization**: fit separate PCA bases for an E block
  (80–160 km) and an F block (above ~150 km, with a 10–20 km overlap and
  a smooth blending weight). Concatenate the coefficients. Each block
  keeps its own `retaining_threshold`, so small-variance E modes are not
  cut by F-dominated variance.
- P2. **Altitude-dependent boost**: generalize `diagonal_boost.py`'s taper
  (now: full amplitude up to 400 km, tapering to 400–700 km) to an
  arbitrary amplitude profile, with a larger amplitude in 80–160 km than
  in the F region. Use a shorter vertical correlation length in the E
  region (about 5–10 km). The boost is applied before the basis is fitted,
  so it also adds E modes to the basis.
- P3. **E–F decorrelation**: build the E- and F-block ensembles from
  independently shuffled IRI members (α = 0), or partly shuffled
  (0 < α < 1).
- P4. **NmE/hmE extraction**: products come from the decoded profile with
  the `score_runs.py` peak finder (valley test), applied per member to get
  an ensemble spread.

**Both styles: E climatology.** The IRI auroral E region is poor at TRO
(forecast E error 87–169 %). Because evaluation covers all seasons, set
any data-derived prior spread (A2, P2) **by season and local time** from
multi-year ionosonde statistics, not from October 2025. E-PROBED is an
independent daytime check at lower latitudes.

### 5.5 WP5 — Lower-profile structure and sporadic E (request item 1)

The lower TEC profile has more structure and worse fits. Some of that is E
signal we want, and some is sporadic E or scintillation that no smooth
prior can fit and that will be aliased if it is assimilated as-is.

- **Flags**: podTc2 already provides `S4_L1`, `S4_L2`, `sigma_phi_L1/L2`
  and SNR. Compute per-occultation Es flags in the 85–130 km tangent band
  (Arras-type SNR std, TEC-perturbation gradient, S4). Store them in the
  observation entry next to the existing diagnostics.
- **Policy** (compare in WP7): (a) exclude flagged E-band rays; (b) keep
  them with an inflated representation error; (c) keep them and let a
  thin-layer basis element absorb them (later work).
- **Residual-shape diagnostic**: add a plot of residual versus tangent
  height per occultation, raw and second-differenced, before and after
  each change. A systematic curvature left in 90–150 km means E signal is
  still being missed.

### 5.6 WP6 — Two-stage tomography: see §6.

### 5.7 WP7 — Validation and scoring

**Primary reference: ionosonde NmE / hmE.** ISR profiles stay as a
secondary check.

*Data found (2026-11-01).* Madrigal (CEDAR) has the **EISCAT Tromsø
Dynasonde** (instrument 2950, kindat 4010, one file per day, 2-minute
records) for 13–18 Oct 2025, which covers all four run days. It is the
same product as the December files already in
`Data/Tomography_data/Ionosonde_Data/`, which
`validation_preparation/ionosonde_parameter_validation.py` already reads.
`TwoStageTomography/fetch_tromso_dynasonde.py` downloads it (Madrigal
asks for a user name, email and affiliation with each download). Svalbard
(2951), Kiruna (2952) and Sodankylä (2900) have no Madrigal data for
those days. GIRO/DIDBase may hold Sodankylä or Tromsø scalings as a
second source, but its old query endpoint now returns 404.

*Downloaded (2026-11-01):* `tromso_dynasonde_2025-10-14..17.nc` in
`/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/Ionosonde_Data/`
(720 two-minute records per day). Coverage of the 30 run hours:

- **foE is scaled in 29 of 30 hours**, against 3–4 hours with an ISR E
  peak per site. Daytime hours have 23–30 scaled records each; the
  night-time hours (00–03 UT) have 0–17.
- **`kmztre` is present in only 9 of 30 hours**, almost all at night. In
  daytime it is essentially absent.

*Finding: dynasonde hmE is not reliable as a reference.* The
true-height profile (`pf` against `gdalt`) is monotonic through the E
region in daytime: ionosonde inversion cannot see the E–F valley, so a
valley test rejects most daytime profiles. Taking hmE as the height
where the profile first reaches foE gives daytime values with a median
of 130 km and a 90th percentile of 167 km. That is well above the
105–115 km expected for a daytime E peak. Where `kmztre` also exists,
the two differ by a median of 33 km. Consequences:

- **NmE vs the ionosonde foE is the primary, robust validation.**
- **hmE**: validate against the **ISR** where it shows an E peak (rare
  but reliable). Report the dynasonde h(foE) and `kmztre` only as
  secondary, flagged proxies. Before relying on either, check them
  against the co-located TRO ISR on days with both. A second ionosonde
  with manually scaled h'E/hmE (GIRO) would help.
- This changes success criterion 2 below: the hmE target applies to
  ISR-matched cases.

*Other points to handle* (also seen in the 2025-12-15 file):

- `foe = 0` means "not scaled". The validator already masks values ≤ 0.
  On that polar-night day foE was present in only 27 % of records.
  October has more daylight and should do better, but expect night-time
  gaps. Score day and night separately.
- `kmztre` is a mean *real* height of the E trace, not a peak height. The
  README already flags it as a proxy. It was filled in about two thirds
  of the foE records.
- NmE = 1.24 × 10¹⁰ · foE² (foE in MHz), as the validator already does.
- Matching: use the ionosonde median over ±10 min of the analysis time.
  The model profile is taken at the station, as for the ISR.

Reuse `Run_Scoring` (E/F relative error, NmE/hmE, correction direction,
TEC fit by ray class) and add:

- an **ionosonde scorer** in `Run_Scoring` that reads the dynasonde files
  through the existing validator and reports NmE error, bias and counts
  for forecast and analysis, split by day/night. It also reports the hmE
  proxies (h(foE), `kmztre`) separately and flagged;
- an **overlap-conditioned** ISR score: report E skill separately for hours
  where low rays (tangent < 150 km) pass within some distance of the ISR
  site (the `HG_<CODE>` overlap already exists). Without overlap, E skill
  can only come from the prior;
- **Abel E-region comparison**: the per-occultation Abel profiles
  (`include_abel=True`) are a free baseline. The tomography should beat
  them in the E region;
- **E-region TEC share** check (`Eshare_a` vs `Eshare_f`) to catch analyses
  that push F residuals into the E region;
- a **longer evaluation set**: 30 hours with 3–4 ISR E peaks is too thin.
  Extend to more days, with daytime emphasis (strong E layer), and split
  results by day / night and quiet / disturbed.

**Success criteria** (proposed; to be agreed). Primary, against the
Tromsø ionosonde:

1. NmE: the analysis error is smaller than the forecast error in ≥ 70 %
   of daytime hours with a scaled foE, and the median daytime NmE error
   is < 25 %.
2. hmE against the **ISR** where it shows an E peak: the median absolute
   error is < 10 km, and the analysis is no worse than the forecast. The
   dynasonde hmE proxies are reported but not used as a pass/fail test
   (see the finding above).

Secondary:

3. Wrong-direction E gates against the ISR drop from ~60 % (SVL) to
   < 25 %.
4. F-region skill no worse than the current boosted run of the **same
   style** (F rel. error within +2 percentage points).
5. Criteria 1–2 hold in **every season** of the multi-season evaluation,
   not only on average. Day and night are reported separately in each
   season.

**Multi-season evaluation.** October 2025 is the development set. Choose
and freeze test periods covering the four seasons (one equinox, one
summer with midnight sun, one polar-night winter) before tuning anything
on them. **The first one will likely be near the summer solstice.**
Checked availability (2026-11-01):

- the Tromsø dynasonde is on Madrigal for **every day 7 Jun – 5 Jul
  2025**;
- the local TRO ISR file (`TROISR2025_nonan.nc`) has **no June or July
  data**, but has 867 profiles in May 2025 (midnight sun at Tromsø starts
  around 20 May). SVL ISR starts only in September 2025. Madrigal CEDAR
  lists no Tromsø or Svalbard ISR experiments for June 2025, though the
  EISCAT Madrigal server may still hold some;
- **choice:** a June solstice period gives NmE validation only (no ISR
  hmE). Late May 2025 gives NmE and ISR hmE in near-solstice conditions.
  Either needs PlanetIQ RO coverage in the ROI, which I have not checked
  yet. They need RO coverage in the ROI and Tromsø dynasonde data;
`fetch_tromso_dynasonde.py` takes any date range. Expectations differ by
season. A strong solar-controlled E layer in summer should be easiest. In
polar-night winter the E region is dominated by auroral precipitation,
foE is often unscaled, and success may be limited to NmE during active
periods. Report sample counts per season so thin seasons are visible.

---

## 6. Two-stage tomography design (WP6)

### 6.1 Formulation

Split the column at h_s ≈ 160 km with a blending band of about 150–180 km,
using weight w(h) that goes from 1 below the band to 0 above it.

**Stage 1 (F region).** Run the current full-column analysis with all
rays (with WP1/WP2 improvements). Optionally down-weight rays with
tangents below h_s so the stage-1 solution is not forced to explain E
structure. Keep the full analysis ensemble Xᴬ₁ (the
`ensemble_batch_NNNN.nc` files).

**Stage 2 (E region).** State = E-region parameters only (fine grid,
§5.3–5.4):

- **ANCHOR:** (log10 NmE, hmE[, H_E]) at each grid point, with the six F
  parameters of the paired stage-1 member held fixed. The ANCHOR profile
  function itself does the blending through h_ST, so w(h) below is not
  needed. Stage 2 is a 2–3 parameter per-column problem, which makes this
  the cheapest and simplest test of the two-stage idea.
- **PCA_1D_10ex:** the E-block coefficients (§5.4 P1), blended with the
  fixed F part as below.

For PCA_1D_10ex, the density field for member i is

    Ne_i(h, geo) = w(h) · Ne_E,i(h, geo) + (1 − w(h)) · Ne_F,i(h, geo)

where Ne_F,i is a stage-1 analysis member, held **fixed** in stage 2.

Observations are the rays with tangents below ~200 km (all samples; the
differenced observable from §5.2).

**The key point: "freeze" must not mean "assume perfect".** If the F region
is fixed at the stage-1 mean and treated as exact, every stage-1 F error
goes straight into the E region. That is the aliasing we are trying to
remove. Two equivalent ways to carry the F uncertainty:

- **Member pairing (recommended).** Pair each stage-2 E member with a
  stage-1 F member, as in the formula above. The ensemble covariance of
  H·Ne then automatically includes the F-region spread, and the gain
  discounts the part of each ray explained by F uncertainty. No new
  algebra is needed. It needs the same member count in both stages
  (2000), or resampling.
- **Explicit R inflation.** Use the stage-1 mean for F and set
  R₂ = R + H_F P_F Hᵀ_F, where H_F P_F Hᵀ_F is taken from the stage-1
  ensemble's ray-space covariance.

### 6.2 Why stage 2 can be cheap and fine

- The state is only the E block: about 33 levels × n_geo, or a few tens of
  PCA coefficients.
- The observations are only the low rays, about 10–30 % of all samples.
- The F contribution of each member is pre-computed once:
  y_F,i = H_F Ne_F,i.

This makes it practical to use the fine grid, the full-resolution
sampling, a dense R, and a larger ensemble or several boost settings in
stage 2. Stage 1 is unchanged and can keep its present cost.

### 6.3 Variants to compare (all in the OSSE first, then on real data)

| Variant | Description |
| --- | --- |
| J0 | Current joint analysis (baseline) |
| J1 | Joint analysis with WP1–WP5 (fine grid, block prior, new R). Tells whether two stages are needed at all. |
| T1 | Two-stage, F frozen at the stage-1 **mean**, no extra R (this is the "naive freeze"; it is expected to alias) |
| T2 | Two-stage with **member pairing** |
| T3 | T2 plus one more pass: re-solve F with the stage-2 E fixed (block Gauss–Seidel). Stop if E and F change by less than a tolerance. |

T3 also tests whether the F region benefits from a better E region on the
low rays. Those rays cross the F bottomside, so this is plausible.

### 6.4 Code placement

A new package, `TwoStageTomography/`, alongside `Assimilation_Cycle`:

- `e_block.py`: E-block grid, the block PCA basis, blending weight w(h),
  and the E-specific boost;
- `stage_two.py`: builds the paired ensembles, the F-contribution
  pre-computation, the low-ray selection and the stage-2 `AnalysisEngine`
  call;
- `observables.py`: differenced/relative TEC and the structured R (WP2).
  This may later move into `observation_preparation`;
- a driver modelled on `run_hourly_all4styles_boost_compare.py` that runs
  stage 1 exactly as now, then stage 2, and writes outputs in the format
  `Run_Scoring` already reads, plus the stage-2 files.

Reuse `Ensemble_Kalman_Engine` unchanged. Its `ObservationOperator` just
gets a decoder that maps the E-block state plus the fixed F member to the
full field.

---

## 7. Sequence and rough effort

| Step | Content | Depends on | Effort |
| --- | --- | --- | --- |
| 0 | ~~Download the Tromsø dynasonde files for 14–17 Oct 2025~~ (done 2026-11-01); add the ionosonde scorer and score the **existing** runs (the NmE baseline) | — | 2 days |
| 1 | WP0 OSSE + DFS study (incl. noise measurement from WP2.1) | — | 1–1.5 wk |
| 2 | WP2 error model / differenced observable + WP1 adaptive sampling | 1 | 1 wk |
| 3 | Real-data rerun of the 30 h with steps 2 only (joint, J1-lite) | 2 | 2–3 days compute |
| 4 | WP3 grid + WP4 block prior and E boost | 1 | 1.5 wk |
| 5 | WP6 two-stage T1/T2/T3 (OSSE, then real data) | 2, 4 | 1.5–2 wk |
| 6 | WP5 Es flags/policy; WP7 ionosonde + overlap scoring; longer data set | parallel from 2 | 1 wk |
| 7 | Decision report: which configuration, what E-region products are credible (full profile vs NmE/hmE vs E-layer content) | all | 2–3 days |

Decision points: after step 1 (is the E region observable enough to go
on, and which levers matter?) and after step 3 (do the observation-side
changes alone give the improvement?).

---

## 8. Risks and open questions

1. **Observability ceiling.** Even with ideal errors, the ~1000 km
   along-ray averaging limits horizontal resolution. The credible product
   may be E-region NmE/hmE or E-layer content on a coarse horizontal
   scale, not a detailed 3-D E region. WP0 will quantify this.
2. **Representativeness at high latitude.** The auroral E region at TRO is
   patchy on scales much smaller than a ray path. ISR point comparisons
   will always look noisy. Overlap-conditioned scoring and ionosondes
   help.
3. **High-rate data availability.** conPhs is not produced by default.
   It can be requested if WP0 justifies it. If so, check how far up the
   high-rate part reaches (the literature says often near 120 km).
4. **Few ionosonde E cases.** Night-time and polar-night foE is often
   unscaled, so the NmE/hmE sample may be small. Extend the evaluation
   period early (a sunlit month such as the equinox), and look for a
   second station via GIRO.
5. **Absolute TEC.** If the differenced observable is used everywhere, the
   absolute level must still be anchored by some absolute rays or by IGS
   VTEC. The hybrid in §5.2 handles this.
6. **Ensemble size versus finer state.** 2000 members against a few
   hundred E-block parameters is fine. Watch the conditioning of the dense
   R in the gain computation.
7. **Questions:**
   - *Answered 2026-11-01:* the products are NmE/hmE, validated against
     ionosonde; Tromsø ionosonde data was found on Madrigal; conPhs is
     not available by default but can be requested.
   - *Answered 2026-11-01:* styles = ANCHOR and PCA_1D_10ex; evaluation
     over all seasons, with periods beyond October added later.
   - *Partly answered 2026-11-01:* the first seasonal test period will
     likely be near the summer solstice. *Open:* June 2025 (dynasonde
     only) or late May 2025 (dynasonde plus TRO ISR); RO coverage for
     either still to be checked (§5.7).
