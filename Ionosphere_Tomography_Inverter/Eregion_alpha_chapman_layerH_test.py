# E-region alpha chapman layer H test
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# E-region alpha-Chapman sensitivity to scale height H_E
# Same formulation as Part 4 of _ne_profile_ensemble
# ============================================================

# ---- Fixed E-layer parameters ----
hmE = 110.0          # E-layer peak altitude [km]
NmE = 1.0e10         # E-layer peak electron density [m^-3]

# Scale heights to compare [km]
HE_values = np.arange(7.0, 21.0, 1.0)   # 7, 8, ..., 20 km

# Only examine E-region bottomside: h < hmE
alts_km = np.linspace(40.0, hmE, 500)


plt.figure(figsize=(15, 7))

# Continuous color mapping: blue (HE=7) -> red (HE=20)
cmap = plt.cm.jet
norm = plt.Normalize(vmin=HE_values.min(), vmax=HE_values.max())

for HE in HE_values:

    # --------------------------------------------------------
    # Part 4: E-layer bottomside alpha-Chapman
    #
    # ze   = (h - hmE) / H_E
    #
    # Ne_E = NmE * exp[0.5 * (1 - ze - exp(-ze))]
    # --------------------------------------------------------

    ze = np.clip((alts_km - hmE) / HE, -80, 80)

    exp_neg_ze = np.exp(-ze)

    Ne_E = NmE * np.exp(
        0.5 * (1.0 - ze - exp_neg_ze)
    )

    plt.plot(
        Ne_E,
        alts_km, 
        linewidth=2.5,
        label=f"$H_E$ = {HE:.0f} km",
        color=cmap(norm(HE))
    )


# ============================================================
# Plot formatting
# ============================================================

plt.xscale("log")

plt.xlabel(r"Ne $N_e$ [m$^{-3}$]", fontsize=20)
plt.ylabel("Altitude [km]", fontsize=20)
# xtick fontsize
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title(
    r"E-region $\alpha$-Chapman Sensitivity to Scale Height $H_E$",
    fontsize=25
)

plt.xlim(1e6, 2e11)
plt.ylim(40, 115)

plt.grid(True, which="both", alpha=0.3)

plt.legend(
    title=r"$H_E$",
    fontsize=15,
    ncol=2
)

plt.tight_layout()
plt.savefig(r"/home/pin/Desktop/tomography_project/Figures/Eregion_alpha_chapman_layerH_test.png", dpi=300)