# Claude Code Implementation Prompts — Robust Testing for Parametric EKF

## Overview

Two objectives, split across two files:

- **Objective 1 (`test_param_iono.py`):** Simulation-based sweep — how many ROs are needed to retrieve NmF2+NmE (or the full profile) within 0.5 / 0.2 / 0.1 MHz for parametric EKF in RO-only, RO+IGS, and IGS-only modes.
- **Objective 2 (`demo_isr_da_comparison.py`):** Real-data validation — how well does parametric EKF assimilate TEC and improve the EDP at truth ISR sites, measured as % improvement and as fraction-within MHz thresholds.

**Parallelism guide:**
- ✅ **Prompt A + Prompt B can be run in parallel** (separate files, no shared code changes)
- ⛔ **Prompt C must run after Prompt A** (extends the sweep to add plotting)
- ⛔ **Prompt D must run after Prompt B** (extends the summary/reporting)

**Model recommendations:**
- Prompts A, B: `claude-opus-4-8` (complex restructuring + new functions)
- Prompts C, D: `claude-sonnet-5` (adding plots/summaries to completed foundation)

---

## Key background — plasma frequency conversion

```python
def ne_to_mhz(ne_m3: np.ndarray) -> np.ndarray:
    """Convert electron density [m⁻³] to plasma frequency [MHz].
    foF2 = 8.978e-6 * sqrt(NmF2)  → ~8.98 MHz at NmF2 = 1e12 m⁻³ (typical F2 peak)
    foE  = 8.978e-6 * sqrt(NmE)   → ~3.18 MHz at NmE = 1.25e11 m⁻³ (typical E peak)
    Profile: fp(alt) = 8.978e-6 * sqrt(Ne(alt))
    """
    return 8.978e-6 * np.sqrt(np.maximum(ne_m3, 0.0))
```

For E-layer peak extraction from a Ne profile: search for the local maximum in the altitude range 90–150 km.

---

## PROMPT A — `test_param_iono.py`: N_OCC sweep + multi-date framework

**Model:** `claude-opus-4-8`  
**File:** `/home/austinhunter/IonosphereTomography/test_param_iono.py`

### Context you need to read first

Read these specific sections of `test_param_iono.py` before making any changes:
- Lines 1–230 (all configuration constants)
- Lines 234–470 (`scan_and_select_files_per_window`, `_round_robin_by_constellation`)
- Lines 2451–2554 (`_process_time_window` — Steps 2–2b only)
- Lines 3408–3560 (`EKF_Param` function signature and opening logic)

### What to add

#### 1. Add helper functions after the imports block (after line ~95, before §0 config)

```python
# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def ne_to_mhz(ne_m3: np.ndarray) -> np.ndarray:
    """Electron density [m⁻³] → plasma frequency [MHz]."""
    return 8.978e-6 * np.sqrt(np.maximum(np.asarray(ne_m3, dtype=float), 0.0))


def extract_e_layer_peak(ne_arr: np.ndarray, alt_arr: np.ndarray,
                          e_alt_min: float = 90.0,
                          e_alt_max: float = 150.0) -> tuple[float, float]:
    """
    Return (NmE [m⁻³], hmE [km]) — the peak electron density in the E-layer
    altitude band [e_alt_min, e_alt_max] km. Returns (nan, nan) if the band
    contains no valid points.
    """
    mask = (alt_arr >= e_alt_min) & (alt_arr <= e_alt_max) & np.isfinite(ne_arr)
    if mask.sum() == 0:
        return np.nan, np.nan
    idx = int(np.argmax(ne_arr[mask]))
    band_ne  = ne_arr[mask]
    band_alt = alt_arr[mask]
    return float(band_ne[idx]), float(band_alt[idx])


def compute_retrieval_freq_metrics(
    truth_ne: np.ndarray,          # (n_alt,) truth profile at one grid point
    prior_ne: np.ndarray,          # (n_alt,)
    post_ne:  np.ndarray,          # (n_alt,)
    alt_grid: np.ndarray,          # (n_alt,) km
    mhz_thresholds: tuple = (0.5, 0.2, 0.1),
) -> dict:
    """
    Compute frequency-domain retrieval metrics comparing prior/posterior Ne
    profiles against truth at a single grid point.

    Returns a dict with:
        truth_foF2, prior_foF2, post_foF2          [MHz]
        truth_foE,  prior_foE,  post_foE           [MHz]
        prior_foF2_err, post_foF2_err              [MHz, signed: post - truth]
        prior_foE_err,  post_foE_err               [MHz]
        prior_fp_rmse,  post_fp_rmse               [MHz, profile RMSE in plasma freq]
        within_{X}mhz_foF2_prior/post              [bool, per threshold]
        within_{X}mhz_profile_prior/post           [bool]
    """
    # F2 peak
    truth_nmf2, _ = extract_robust_f2_peak(truth_ne, alt_grid)
    prior_nmf2, _ = extract_robust_f2_peak(prior_ne, alt_grid)
    post_nmf2,  _ = extract_robust_f2_peak(post_ne,  alt_grid)

    truth_foF2 = ne_to_mhz(truth_nmf2)
    prior_foF2 = ne_to_mhz(prior_nmf2)
    post_foF2  = ne_to_mhz(post_nmf2)

    # E layer peak
    truth_nme, _ = extract_e_layer_peak(truth_ne, alt_grid)
    prior_nme, _ = extract_e_layer_peak(prior_ne, alt_grid)
    post_nme,  _ = extract_e_layer_peak(post_ne,  alt_grid)

    truth_foE = ne_to_mhz(truth_nme)
    prior_foE = ne_to_mhz(prior_nme)
    post_foE  = ne_to_mhz(post_nme)

    # Full-profile plasma-frequency RMSE
    truth_fp = ne_to_mhz(truth_ne)
    prior_fp = ne_to_mhz(prior_ne)
    post_fp  = ne_to_mhz(post_ne)
    valid = np.isfinite(truth_fp) & np.isfinite(prior_fp) & np.isfinite(post_fp)
    prior_fp_rmse = float(np.sqrt(np.mean((prior_fp[valid] - truth_fp[valid])**2))) if valid.any() else np.nan
    post_fp_rmse  = float(np.sqrt(np.mean((post_fp[valid]  - truth_fp[valid])**2))) if valid.any() else np.nan

    out = dict(
        truth_foF2=float(truth_foF2), prior_foF2=float(prior_foF2), post_foF2=float(post_foF2),
        truth_foE=float(truth_foE),   prior_foE=float(prior_foE),   post_foE=float(post_foE),
        prior_foF2_err=float(prior_foF2 - truth_foF2),
        post_foF2_err =float(post_foF2  - truth_foF2),
        prior_foE_err =float(prior_foE  - truth_foE),
        post_foE_err  =float(post_foE   - truth_foE),
        prior_fp_rmse=prior_fp_rmse,
        post_fp_rmse =post_fp_rmse,
    )
    for thr in mhz_thresholds:
        thr_str = str(thr).replace(".", "")
        out[f"within_{thr_str}mhz_foF2_prior"] = bool(abs(out["prior_foF2_err"]) <= thr)
        out[f"within_{thr_str}mhz_foF2_post"]  = bool(abs(out["post_foF2_err"])  <= thr)
        out[f"within_{thr_str}mhz_foE_prior"]  = bool(abs(out["prior_foE_err"])  <= thr)
        out[f"within_{thr_str}mhz_foE_post"]   = bool(abs(out["post_foE_err"])   <= thr)
        out[f"within_{thr_str}mhz_profile_prior"] = bool(prior_fp_rmse <= thr)
        out[f"within_{thr_str}mhz_profile_post"]  = bool(post_fp_rmse  <= thr)
    return out
```

#### 2. Add multi-date configuration near the top of §0 (after line ~110)

Add this constant block after the `SAVE_DIR` / `IRI_CACHE_DIR` definitions:

```python
# ── §0b  Multi-date sweep configuration ─────────────────────────────────────
# Each entry is (YYYY, DOY, [hour_list]) defining which windows to test.
# Cover: winter solstice, spring equinox, summer solstice, autumn equinox;
# hours: 0, 6, 12, 18 UTC → captures diurnal variation.
SWEEP_DATES = [
    # (year, doy, description)
    (2025,  1,  "winter_solstice"),
    (2025, 80,  "spring_equinox"),
    (2025, 172, "summer_solstice"),
    (2025, 265, "nominal_day"),       # existing default
    (2025, 355, "autumn_equinox"),
]

# N_OCC sweep range
SWEEP_N_OCC_VALUES = list(range(10, 101, 10))   # [10, 20, 30, …, 100]

# Where to save sweep results
SWEEP_RESULTS_CSV = "./Data/occ_sweep_results.csv"
SWEEP_SAVE_DIR    = "./Figures/occ_sweep/"
```

#### 3. Add `run_occ_count_sweep()` function

Place this after `scan_and_select_files_per_window()` (after line ~470). This function:
- Takes a list of arc records (already scanned for one time window)
- For each N_OCC in `n_occ_values`, sub-samples exactly N_OCC arcs using round-robin constellation balancing
- Runs `EKF_Param` in each of the three modes (ro_only, ro_igs, igs_only)
- For ro_only: uses only the sub-sampled RO arcs
- For ro_igs: uses sub-sampled RO arcs + simulated IGS arcs (already built at window level)
- For igs_only: uses only simulated IGS arcs
- Returns a list of row dicts for CSV accumulation

The function signature:

```python
def run_occ_count_sweep(
    window: dict,               # standard window dict from scan_and_select_files_per_window
    model_state: "IonosphericState",    # already-built prior ensemble for this window
    truth_state: "IonosphericState",    # already-built truth state (1-deg IRI)
    grid_lats: np.ndarray,      # 5-deg model grid lats
    grid_lons: np.ndarray,      # 5-deg model grid lons
    alt_grid: np.ndarray,
    igs_arcs: list[dict],       # simulated IGS arc_truth_list entries (may be empty)
    n_occ_values: list[int] = SWEEP_N_OCC_VALUES,
    modes: tuple = FILTER_MODES,
    save_dir: str = SWEEP_SAVE_DIR,
) -> list[dict]:
```

Inside the function, for each n_occ:
- Sub-sample by calling `_round_robin_by_constellation(all_ro_arcs, n_occ)` (import from existing code)
- For each mode, build the appropriate `arc_truth_list`:
  - `ro_only`: just the sub-sampled RO arcs
  - `ro_igs`: sub-sampled RO arcs + `igs_arcs`
  - `igs_only`: just `igs_arcs` (n_occ is irrelevant here, but still include the row with n_occ for uniformity)
- Call `EKF_Param(arc_truth_list, model_state, grid_lats, grid_lons, alt_grid, jacobian_analytical=True)`
- Extract truth Ne at the ISR sites (ESR and TRO) using a cKDTree lookup into the 1-deg truth grid
- Call `compute_retrieval_freq_metrics(truth_ne, prior_ne, post_ne, alt_grid)` for each ISR site
- Append a row dict:
  ```python
  {
      "window_key": ..., "time_dt": ..., "doy": ..., "hour": ...,
      "mode": ..., "n_occ": ..., "site": ...,
      "converged": ..., "n_iterations": ...,
      "prior_rmse_tecu": ..., "post_rmse_tecu": ...,
      **freq_metrics,   # all keys from compute_retrieval_freq_metrics
  }
  ```

#### 4. Add `main_sweep()` and wire into `__main__`

Add at the bottom of the file:

```python
def main_sweep() -> None:
    """
    Entry point for the N_OCC sweep across multiple dates/seasons.
    Invoked with:  python test_param_iono.py --sweep
    Saves results to SWEEP_RESULTS_CSV.
    """
    import argparse
    ...
```

The sweep loop:
1. For each `(yyyy, doy, label)` in `SWEEP_DATES`:
   - Build `base_path = f".../{yyyy}.{doy:03d}/"`; skip if not present
   - Call `scan_and_select_files_per_window(base_path)` → windows
   - For each window: build IRI truth/model states (call existing `build_iri_state_grid_cached` + `build_truth_state` + `build_model_ensemble`), call `run_occ_count_sweep()`
2. Accumulate all row dicts, save to `SWEEP_RESULTS_CSV` as a pandas CSV

Modify `if __name__ == "__main__":` to check for `--sweep` in `sys.argv` and dispatch to `main_sweep()` vs `main()`.

#### 5. Important notes for this implementation

- `extract_robust_f2_peak` is already imported from `demo` — use it for F2
- The `_process_time_window` function already builds `truth_state`, `model_state`, and `grid_lats_5deg/grid_lons_5deg`. The sweep can reuse this infrastructure: call `_process_time_window` first to get the full result, then re-run only the EKF step with sub-sampled arc lists
- The `EKF_Param` function (line 3408) takes `arc_truth_list` as its first arg — these are dicts with keys: `rays`, `tec_truth`, `tp_lats`, `tp_lons`, `tang_km`, `conid`, `prn_id`
- `EKF_Param` returns `posterior_edp` shape `(n_alt, n_grid)` and `prior_edp` shape `(n_alt, n_grid)` — use these for the Ne-column lookup at ISR site lat/lon
- Do NOT modify `EKF_Param` itself — call it as-is

---

## PROMPT B — `demo_isr_da_comparison.py`: frequency-domain metrics + threshold reporting

**Model:** `claude-opus-4-8`  
**File:** `/home/austinhunter/IonosphereTomography/demo_isr_da_comparison.py`

### Context you need to read first

Read these specific sections before making changes:
- Lines 1–90 (imports and constants)
- Lines 1160–1315 (`compute_isr_metrics` — the full function)
- Lines 1353–1408 (`summarize_statistics`)
- Lines 1245–1268 (existing F2 peak extraction and RMSE logic inside `compute_isr_metrics`)

### What to add

#### 1. Add helpers near the top (after imports, before line 80 constants)

```python
# ── Plasma-frequency helpers ─────────────────────────────────────────────────

def ne_to_mhz(ne_m3) -> float | np.ndarray:
    """Electron density [m⁻³] → plasma frequency [MHz]. foF2 ≈ 8.98 MHz at 1e12 m⁻³."""
    return 8.978e-6 * np.sqrt(np.maximum(np.asarray(ne_m3, dtype=float), 0.0))


def extract_e_layer_peak(ne_arr: np.ndarray, alt_arr: np.ndarray,
                          e_alt_min: float = 90.0,
                          e_alt_max: float = 150.0) -> tuple[float, float]:
    """
    Return (NmE [m⁻³], hmE [km]) — the E-layer peak in the 90–150 km band.
    Returns (nan, nan) if no valid data in band.
    """
    mask = (alt_arr >= e_alt_min) & (alt_arr <= e_alt_max) & np.isfinite(ne_arr) & (ne_arr > 1e6)
    if mask.sum() == 0:
        return np.nan, np.nan
    band_ne  = ne_arr[mask]
    band_alt = alt_arr[mask]
    idx = int(np.argmax(band_ne))
    return float(band_ne[idx]), float(band_alt[idx])
```

#### 2. Extend `compute_isr_metrics()` to track frequency-domain metrics

The existing function (lines 1160–1314) already computes `prior_NmF2_err_pct`, `post_NmF2_err_pct`, `prior_hmF2_err_km`, `post_hmF2_err_km`. You need to add the following **after** the existing NmF2/hmF2 error computation but **before** `rows.append(...)`:

```python
# ── Frequency-domain metrics ─────────────────────────────────────────────────
# foF2 = critical frequency of the F2 layer as seen from the ground [MHz]
# foE  = blanketing frequency of the E layer [MHz]
# These are the HF propagation quantities that operators care about.

prior_foF2 = float(ne_to_mhz(pr_nm)) if np.isfinite(pr_nm) else np.nan
post_foF2  = float(ne_to_mhz(po_nm)) if np.isfinite(po_nm) else np.nan
isr_foF2   = float(ne_to_mhz(isr_nm)) if np.isfinite(isr_nm) else np.nan

prior_foF2_err = prior_foF2 - isr_foF2   # signed [MHz]
post_foF2_err  = post_foF2  - isr_foF2

# E-layer peak from ISR profile (interpolated filter profile at ISR altitudes)
isr_nme, isr_hme     = extract_e_layer_peak(isr_ne,         isr_alt)
prior_nme, prior_hme = extract_e_layer_peak(prior_at_isr,   isr_alt)
post_nme,  post_hme  = extract_e_layer_peak(post_at_isr,    isr_alt)

prior_foE = float(ne_to_mhz(prior_nme)) if np.isfinite(prior_nme) else np.nan
post_foE  = float(ne_to_mhz(post_nme))  if np.isfinite(post_nme)  else np.nan
isr_foE   = float(ne_to_mhz(isr_nme))   if np.isfinite(isr_nme)   else np.nan

prior_foE_err = prior_foE - isr_foE
post_foE_err  = post_foE  - isr_foE

# Full-profile RMSE in plasma frequency [MHz] — the "entire profile" metric
valid_fp = valid  # reuse existing gate (ne > 1e8, finite, below hmF2)
isr_fp_arr   = ne_to_mhz(isr_ne)
prior_fp_arr = ne_to_mhz(prior_at_isr)
post_fp_arr  = ne_to_mhz(post_at_isr)
prior_profile_fp_rmse = float(np.sqrt(np.mean((prior_fp_arr[valid_fp] - isr_fp_arr[valid_fp])**2))) \
                         if valid_fp.any() else np.nan
post_profile_fp_rmse  = float(np.sqrt(np.mean((post_fp_arr[valid_fp]  - isr_fp_arr[valid_fp])**2))) \
                         if valid_fp.any() else np.nan

# Threshold flags
MHZ_THRESHOLDS = [0.5, 0.2, 0.1]
threshold_fields: dict = {}
for thr in MHZ_THRESHOLDS:
    ts = str(thr).replace(".", "")
    threshold_fields[f"prior_foF2_within_{ts}mhz"] = bool(np.isfinite(prior_foF2_err) and abs(prior_foF2_err) <= thr)
    threshold_fields[f"post_foF2_within_{ts}mhz"]  = bool(np.isfinite(post_foF2_err)  and abs(post_foF2_err)  <= thr)
    threshold_fields[f"prior_foE_within_{ts}mhz"]  = bool(np.isfinite(prior_foE_err)  and abs(prior_foE_err)  <= thr)
    threshold_fields[f"post_foE_within_{ts}mhz"]   = bool(np.isfinite(post_foE_err)   and abs(post_foE_err)   <= thr)
    threshold_fields[f"prior_profile_within_{ts}mhz"] = bool(np.isfinite(prior_profile_fp_rmse) and prior_profile_fp_rmse <= thr)
    threshold_fields[f"post_profile_within_{ts}mhz"]  = bool(np.isfinite(post_profile_fp_rmse)  and post_profile_fp_rmse  <= thr)
```

Then add the new fields to the `rows.append({...})` dict:
```python
# (add after existing fields)
"isr_foF2":               isr_foF2,
"prior_foF2":             prior_foF2,
"post_foF2":              post_foF2,
"prior_foF2_err_mhz":     prior_foF2_err,
"post_foF2_err_mhz":      post_foF2_err,
"isr_foE":                isr_foE,
"prior_foE":              prior_foE,
"post_foE":               post_foE,
"prior_foE_err_mhz":      prior_foE_err,
"post_foE_err_mhz":       post_foE_err,
"prior_profile_fp_rmse_mhz": prior_profile_fp_rmse,
"post_profile_fp_rmse_mhz":  post_profile_fp_rmse,
**threshold_fields,
```

#### 3. Extend `summarize_statistics()` to print threshold tables

After the existing per-(obs_mode, filter_type) RMSE table at lines 1383–1403, add a second section:

```python
# ── Threshold-based performance tables ────────────────────────────────────
for metric_label, prior_col, post_col in [
    ("foF2 (critical freq)",    "prior_foF2_within_{}mhz", "post_foF2_within_{}mhz"),
    ("foE  (blanketing freq)",  "prior_foE_within_{}mhz",  "post_foE_within_{}mhz"),
    ("Profile fp RMSE",         "prior_profile_within_{}mhz", "post_profile_within_{}mhz"),
]:
    lines.append(f"\n  {metric_label} — fraction of cases within threshold:")
    lines.append(f"  {'obs_mode':<10} {'filter_type':<14} {'n':>4}  "
                 + "  ".join(f"{'prior|post @'+str(t)+'MHz':>20}" for t in [0.5, 0.2, 0.1]))
    for (obs_mode, filter_type), grp in combined.groupby(["obs_mode", "filter_type"]):
        n = len(grp)
        thr_parts = []
        for thr in [0.5, 0.2, 0.1]:
            ts = str(thr).replace(".", "")
            pc = prior_col.format(ts)
            po = post_col.format(ts)
            if pc in grp.columns and po in grp.columns:
                prior_frac = grp[pc].mean()
                post_frac  = grp[po].mean()
                thr_parts.append(f"{prior_frac:>8.1%}|{post_frac:<8.1%}")
            else:
                thr_parts.append("        N/A      ")
        lines.append(f"  {obs_mode:<10} {filter_type:<14} {n:>4}  " + "  ".join(thr_parts))
```

#### 4. Note on "best data" selection

The existing `main()` (line 1521) already filters: `edps = [e for e in edps if e.get("kindat") == "6400"]`. This is correct — kindat 6400 = fitted/post-processed ISR profiles, the highest quality. No change needed here.

Also, the existing `best_by_site` logic in `run_all_filters()` (lines 1088–1100) already selects the ISR scan closest to the window centre per site. This is the right approach for real data.

#### 5. Add n_occ tracking to existing metrics rows (minor addition)

In `compute_isr_metrics()`, the variable `n_ro_occultations` is already being captured from `day_info.get("n_ro_occultations", ...)` and written to the row. No change needed — it's already there.

---

## PROMPT C — `test_param_iono.py`: Sweep results plotting

**Model:** `claude-sonnet-5`  
**Prerequisite:** Must run **after Prompt A is complete**  
**File:** `/home/austinhunter/IonosphereTomography/test_param_iono.py`

### Task

Add `plot_occ_sweep_results(csv_path: str, save_dir: str)` function that reads `SWEEP_RESULTS_CSV` and generates the following figures:

#### Figure 1: N_OCC vs foF2 error (one panel per mode, one line per season)
- X-axis: number of occultations (10–100)
- Y-axis: median |foF2 error| in MHz across ISR sites
- Horizontal dashed lines at 0.5, 0.2, 0.1 MHz
- 3 panels: ro_only, ro_igs, igs_only
- Lines colored by season (winter/spring/summer/autumn)
- Save as `occ_sweep_foF2_vs_nocc.png`

#### Figure 2: N_OCC vs foE error (same layout as Figure 1)
- Save as `occ_sweep_foE_vs_nocc.png`

#### Figure 3: N_OCC vs profile fp RMSE (same layout)
- Save as `occ_sweep_profile_rmse_vs_nocc.png`

#### Figure 4: Threshold fraction vs N_OCC (3×3 subplot: mode × threshold)
- Each subplot: fraction of cases within threshold vs N_OCC
- 3 modes (rows), 3 thresholds 0.5/0.2/0.1 MHz (columns)
- Separate line for foF2 (solid), foE (dashed), profile (dotted)
- Save as `occ_sweep_threshold_fractions.png`

#### Figure 5: HF propagation perspective — blanketing frequency plot
- Shows foF2 and foE (prior vs. posterior) vs. truth at each ISR site
- Scatter plot: truth foF2/foE (x) vs. posterior foF2/foE (y), colored by N_OCC
- One panel per mode
- Diagonal line = perfect retrieval
- Save as `occ_sweep_hf_propagation.png`

Also wire `plot_occ_sweep_results` into `main_sweep()` so it auto-runs after the CSV is written.

---

## PROMPT D — `demo_isr_da_comparison.py`: Threshold summary plots

**Model:** `claude-sonnet-5`  
**Prerequisite:** Must run **after Prompt B is complete**  
**File:** `/home/austinhunter/IonosphereTomography/demo_isr_da_comparison.py`

### Task

Add `plot_isr_freq_metrics(metrics_csv: str, save_dir: str)` function that reads the accumulated `ISR_METRICS_CSV` and generates:

#### Figure 1: Improvement in foF2 error by obs_mode and filter_type
- Box plots of `post_foF2_err_mhz` (absolute) grouped by (obs_mode, filter_type)
- One panel for each ISR site (ESR, TRO)
- Horizontal lines at 0.5, 0.2, 0.1 MHz
- Also show prior distribution as a reference box in grey
- Save as `isr_foF2_improvement_boxplot.png`

#### Figure 2: foE improvement (same layout as Figure 1)
- Save as `isr_foE_improvement_boxplot.png`

#### Figure 3: Fraction within threshold — bar chart
- For each (obs_mode, filter_type) combination, show 6 bars:
  - foF2 prior/post within 0.5/0.2/0.1 MHz
  - Profile prior/post within 0.5/0.2/0.1 MHz
- 3 panels: one per threshold (0.5, 0.2, 0.1 MHz)
- Save as `isr_threshold_fractions.png`

#### Figure 4: foF2 scatter — truth vs. posterior by obs_mode
- Scatter: ISR foF2 (x) vs. posterior foF2 (y) for parametric_ekf only
- One panel per obs_mode (3 panels)
- Perfect retrieval diagonal
- Color by n_ro_occultations
- Save as `isr_foF2_scatter_by_mode.png`

#### Figure 5: HF propagation perspective
- Time-series (by t_centre) of ISR truth foF2 and foE, overlaid with prior/posterior from parametric EKF in each mode
- One row per ISR site (ESR, TRO)
- Save as `isr_hf_propagation_timeseries.png`

Wire `plot_isr_freq_metrics` into `summarize_statistics()` so it auto-runs if `SAVE_DIR` exists.

---

## Parallelism and execution order summary

```
              ┌─────────────────────────────────────────────────┐
              │                                                 │
     ┌────────▼───────┐                          ┌─────────────▼──────┐
     │   PROMPT A     │                          │    PROMPT B        │
     │ test_param     │  ◄── run in parallel ──► │ demo_isr_da_comp   │
     │ N_OCC sweep    │                          │ freq metrics       │
     │ + multi-date   │                          │ + thresholds       │
     └────────┬───────┘                          └─────────────┬──────┘
              │                                                │
     ┌────────▼───────┐                          ┌─────────────▼──────┐
     │   PROMPT C     │                          │    PROMPT D        │
     │ sweep result   │                          │ ISR freq metric    │
     │ plots          │                          │ plots              │
     └────────────────┘                          └────────────────────┘
```

All prompts should be given against the **main branch** files in `/home/austinhunter/IonosphereTomography/`, not the worktree copies under `.claude/worktrees/`.

---

## Testing each prompt after implementation

**After Prompt A:**
```bash
cd /home/austinhunter/IonosphereTomography
# Quick smoke test: single date, n_occ=[10,20] only
python test_param_iono.py --sweep --dates 265 --n-occ 10 20
# Should produce Data/occ_sweep_results.csv with ~6 rows (2 n_occ × 3 modes)
```

**After Prompt B:**
```bash
# Run with --status to check CSV columns
python demo_isr_da_comparison.py --status
# Run with existing cache to regenerate metrics with new columns
python demo_isr_da_comparison.py --force --days 1 --tier 1
```

**After Prompt C:**
```bash
# Plot from existing CSV (if it exists)
python test_param_iono.py --plot-sweep
```

**After Prompt D:**
```bash
# Auto-runs from summarize_statistics if CSV exists
python demo_isr_da_comparison.py --status
```
