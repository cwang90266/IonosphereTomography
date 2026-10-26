#!/usr/bin/env python3
"""
Diagnostic: two-ion topside weight_k as a function of slant path length.

Left panel  — weight_k vs slant distance for several elevation angles, with
              the individual O+ and H+ components overlaid for the vertical case.
Right panel — for each possible stopping threshold, what fraction of the total
              topside TEC (integrated to GPS altitude) is retained?  This is the
              direct tool for choosing where to set the early-exit guard in
              _process_single_ray.

Adjust the parameters at the top to match what you have in the inverter.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Parameters — keep in sync with _process_single_ray
# ---------------------------------------------------------------------------
H_O_m  = 150_000.0     # O+ scale height, m
H_H_m  = 1_000_000.0   # H+ scale height, m
alpha  = 0.05           # H+ fraction at grid top

GRID_TOP_KM = 600.0     # assumed top of the ionospheric grid
GNSS_ALT_KM = 20_200.0  # GPS MEO altitude

CURRENT_THRESHOLD = 1e-7   # what is in the file right now
PROPOSED_THRESHOLD = 1e-8  # the revised suggestion

# ---------------------------------------------------------------------------
# Geometry sweep
# ---------------------------------------------------------------------------
# Elevation angles chosen to span: near-vertical down to the cos_zenith floor
elev_deg   = [90, 60, 30, 15, 5.73]   # 5.73° → cos ζ ≈ 0.10
# cos(zenith_angle) = sin(elevation_angle) for a locally-flat geometry
cos_z      = [np.sin(np.radians(e)) for e in elev_deg]
line_labels = [f"elev = {e:.0f}°  (cos ζ = {c:.2f})" for e, c in zip(elev_deg, cos_z)]
colors      = plt.cm.plasma(np.linspace(0.15, 0.85, len(cos_z)))

max_v_m      = (GNSS_ALT_KM - GRID_TOP_KM) * 1000.0   # vertical distance to GPS in metres
max_slant_km = [(max_v_m / c) / 1000.0 for c in cos_z]  # slant distance per elevation

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Two-Ion Topside Model — weight$_k$ diagnostics\n"
    f"H$_{{O+}}$ = {H_O_m/1e3:.0f} km,  H$_{{H+}}$ = {H_H_m/1e3:.0f} km,  "
    f"α = {alpha},  grid top = {GRID_TOP_KM:.0f} km,  GPS alt = {GNSS_ALT_KM:.0f} km",
    fontsize=11,
)

# ---------------------------------------------------------------------------
# Panel 1: weight_k vs slant distance
# ---------------------------------------------------------------------------
slant_km = np.linspace(0, max(max_slant_km) * 1.02, 80_000)
slant_m  = slant_km * 1000.0

for cz, lab, col, ms in zip(cos_z, line_labels, colors, max_slant_km):
    h_m = slant_m * cz
    w   = (1 - alpha) * np.exp(-h_m / H_O_m) + alpha * np.exp(-h_m / H_H_m)
    ax1.semilogy(slant_km[slant_km <= ms], w[slant_km <= ms],
                 color=col, lw=1.8, label=lab)

# Individual O+ and H+ components for the vertical case
h_v   = slant_m * 1.0
ax1.semilogy(slant_km, (1 - alpha) * np.exp(-h_v / H_O_m),
             'k--', lw=1.1, alpha=0.45, label="O⁺ only  (vert.)")
ax1.semilogy(slant_km, alpha * np.exp(-h_v / H_H_m),
             'k:',  lw=1.1, alpha=0.45, label="H⁺ only  (vert.)")

# Threshold reference lines
ax1.axhline(CURRENT_THRESHOLD,  color='crimson', ls='--', lw=1.3, alpha=0.85,
            label=f"current threshold  ({CURRENT_THRESHOLD:.0e})")
ax1.axhline(PROPOSED_THRESHOLD, color='crimson', ls=':',  lw=1.3, alpha=0.85,
            label=f"proposed threshold ({PROPOSED_THRESHOLD:.0e})")

# GPS altitude annotation for the vertical ray
ax1.axvline(max_v_m / 1000.0, color='steelblue', ls=':', lw=1.1, alpha=0.6)
ax1.text(max_v_m / 1000.0 + 200, 3e-4, "GPS\n(vert.)",
         fontsize=8, color='steelblue', va='center')

ax1.set_xlabel("Slant path from grid top  (km)", fontsize=10)
ax1.set_ylabel("weight$_k$", fontsize=10)
ax1.set_title("Weight decay along slant path", fontsize=10)
ax1.set_xlim(0, max(max_slant_km) * 1.05)
ax1.set_ylim(1e-12, 1.5)
ax1.legend(fontsize=7.5, loc="upper right")
ax1.grid(True, which="both", alpha=0.25)
ax1.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# ---------------------------------------------------------------------------
# Panel 2: fraction of total topside TEC retained vs threshold value
# ---------------------------------------------------------------------------
thresholds = np.logspace(-12, -1, 600)

for cz, lab, col, ms_km in zip(cos_z, line_labels, colors, max_slant_km):
    s_fine  = np.linspace(0, ms_km * 1000.0, 200_000)   # metres
    h_fine  = s_fine * cz
    w_fine  = (1 - alpha) * np.exp(-h_fine / H_O_m) + alpha * np.exp(-h_fine / H_H_m)
    total   = np.trapezoid(w_fine, s_fine)

    fracs = np.empty(len(thresholds))
    for j, thr in enumerate(thresholds):
        below = np.searchsorted(-w_fine, -thr)   # first index where w < thr
        if below == 0:
            fracs[j] = 0.0
        elif below >= len(w_fine):
            fracs[j] = 100.0
        else:
            fracs[j] = np.trapezoid(w_fine[:below], s_fine[:below]) / total * 100.0

    ax2.semilogx(thresholds, fracs, color=col, lw=1.8, label=lab)

# Reference percentages
for pct, ls in [(99.9, ':'), (99.0, '--'), (95.0, '-.')]:
    ax2.axhline(pct, color='gray', ls=ls, lw=0.9, alpha=0.6)
    ax2.text(1.3e-12, pct + 0.4, f"{pct:.1f}%", fontsize=7.5, color='gray')

# Mark both thresholds
for thr, ls in [(CURRENT_THRESHOLD, '--'), (PROPOSED_THRESHOLD, ':')]:
    ax2.axvline(thr, color='crimson', ls=ls, lw=1.3, alpha=0.85)
    ax2.text(thr * 1.3, 30, f"{thr:.0e}", fontsize=8, color='crimson',
             rotation=90, va='bottom')

ax2.set_xlabel("Early-stopping threshold", fontsize=10)
ax2.set_ylabel("Topside TEC retained  (%)", fontsize=10)
ax2.set_title(
    "Fraction of total topside TEC captured before threshold\n"
    "(normalised to full slant integral to GPS altitude)",
    fontsize=10,
)
ax2.set_xlim(thresholds[0], thresholds[-1])
ax2.set_ylim(0, 102)
ax2.legend(fontsize=7.5, loc="lower left")
ax2.grid(True, which="both", alpha=0.25)
ax2.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

plt.tight_layout()
out_path = "topside_weight_diagnostic.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
