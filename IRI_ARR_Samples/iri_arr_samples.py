# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import root_scalar

def calculate_h_star_star(h_array, HZ, h_ST, h_l):
    """
    Implements the IRI intermediate region parabolic height transformation (Equation 22a/b).
    """
    # Prevent division by zero if the curve naturally hits NmE exactly at HEF
    if np.isclose(h_ST, h_l):
        return h_array

    # Calculate the T parameter (Equation 22b)
    T = ((HZ - h_ST)**2) / (h_ST - h_l)
    
    # Calculate the term inside the square root
    inside_sqrt = T * ((T / 4.0) - (h_array - HZ))
    
    # Prevent floating-point negative square roots near boundaries
    inside_sqrt = np.maximum(inside_sqrt, 0.0)
    
    # To ensure eta = HZ at h = HZ, we dictate the sign based on T
    if T > 0:
        eta = HZ + (T / 2.0) - np.sqrt(inside_sqrt)
    else:
        eta = HZ + (T / 2.0) + np.sqrt(inside_sqrt)
        
    return eta

def get_ideal_f_density(h_val, hmF2, NmF2, B0, B1, hmF1=None, C1=None):
    """Calculates the pure F-region density at any height, including the C1 ledge."""
    if hmF1 is not None and C1 is not None and C1 > 0 and h_val < hmF1:
        dh_frac = (hmF1 - h_val) / hmF1
        h_star = hmF1 * (1.0 - (dh_frac)**(1.0 + C1)) # Eq. 20
        x = (hmF2 - h_star) / B0
    else:
        x = (hmF2 - h_val) / B0 # Eq. 13a
    return NmF2 * (np.exp(-(x**B1)) / np.cosh(x)) # Eq.  13


def calculate_iri_electron_density(altitudes, iri_params):
    """
    Fast, vectorized generation of the IRI-style electron density profile.
    """
    # Ensure altitudes is a NumPy array for vectorized operations
    h = np.asarray(altitudes, dtype=float)
    ne_profile = np.zeros_like(h)
    
    # Unpack parameters
    NmF2 = iri_params.get('NMF2')
    hmF2 = iri_params.get('HMF2')
    NmF1 = iri_params.get('NMF1')
    hmF1 = iri_params.get('HMF1')
    NmE  = iri_params.get('NME')
    hmE  = iri_params.get('HME')
    NmD  = iri_params.get('NMD')
    hmD  = iri_params.get('HMD')
    B0   = iri_params.get('B0')
    B1   = iri_params.get('B1')
    VNER = iri_params.get('VNER')
    HEF  = iri_params.get('HEF')
    C1 = iri_params.get('C1')

    # 1. Topside Region (above hmF2)
    m_top = h >= hmF2
    if np.any(m_top):
        H0 = iri_params.get('H0', 50.0)      
        gamma = iri_params.get('gamma', 0.15) 
        r = 100.0
        
        dh = h[m_top] - hmF2
        
        # Calculate restricted scale height array
        H_top = H0 * (1.0 + (r * gamma * dh) / (r * H0 + gamma * dh)) # Bilitza et al. Eq. 6a
        
        z = dh / H_top
        
        # Apply Epstein layer decay
        ne_profile[m_top] = 4.0 * NmF2 * np.exp(z) / ((1.0 + np.exp(z))**2) # Bilitza et al. Eq. 6

    ## ---------------------------------------------------------
    # 2 & 3. F-Region Bottomside & Parabolic Intermediate Region
    # ---------------------------------------------------------
    h_l = HEF if HEF is not None else (hmE if hmE is not None else 120.0) # Eq. 21b
    Nm_e_top = NmE if NmE is not None else 1e10 

    m_f = (h >= h_l) & (h < hmF2)
    if np.any(m_f):
        h_f = h[m_f]
        ne_f = np.zeros_like(h_f)

        # 1. Numerically find h_ST (Where the F-curve naturally meets NmE)
        def objective(h_guess):
            return get_ideal_f_density(h_guess, hmF2, NmF2, B0, B1, hmF1, C1) - Nm_e_top

        try:
            # Search for the height between 60km and hmF2 where density = Nm_e_top
            res = root_scalar(objective, bracket=[60.0, hmF2], method='brentq')
            h_ST = res.root
        except ValueError:
            h_ST = h_l # Fallback if curve never hits NmE

        # 2. Define HZ (The start of the Intermediate Region)
        # Standard approach: start the bend at the F1 peak or slightly above the target
        if hmF1 is not None:
            hF1 = hmF1
        else:
            hF1 = (hmF2 + h_l)/2  #Eq. 21a
            
        HZ = (h_ST + hF1)/2 #Eq. 21

        # 3. Apply the Parabolic Transformation (h**)
        m_int = h_f < HZ
        
        # Transform the heights inside the intermediate region
        h_transformed = np.copy(h_f)
        if np.any(m_int):
            h_transformed[m_int] = calculate_h_star_star(h_f[m_int], HZ, h_ST, h_l) # Eq 22

        # 4. Calculate Density using the final, transformed height array
        for i, h_val in enumerate(h_transformed):
            ne_f[i] = get_ideal_f_density(h_val, hmF2, NmF2, B0, B1, hmF1, C1)

        ne_profile[m_f] = ne_f
                

    # 4. E-Valley Region (between hmE and HEF)
    if hmE is not None and HEF is not None:
        m_valley = (h >= hmE) & (h < HEF)
        if np.any(m_valley):
            h_valley_bottom = hmE + ((HEF - hmE) / 2.0)
            a = (NmE - VNER) / ((hmE - h_valley_bottom)**2)
            ne_profile[m_valley] = VNER + a * (h[m_valley] - h_valley_bottom)**2

    # ---------------------------------------------------------
    # D-Region & E-Bottomside Shared Parameters
    # ---------------------------------------------------------
    F1 = 0.02
    F2 = -1.25e-3
    F3 = 8.79e-3

    if hmD is not None and hmE is not None:
        # Approximate HDX as the midpoint between the E-peak and D-inflection
        HDX = (hmE + hmD) / 2.0
    elif hmD is not None:
        HDX = hmD # Fallback if E-region is missing
    else:
        HDX = None

    # ---------------------------------------------------------
    # 5. E Bottomside Region (between HDX and hmE)
    # ---------------------------------------------------------
    if hmD is not None and hmE is not None and HDX is not None:
        m_eb = (h >= HDX) & (h < hmE)
        if np.any(m_eb):
            # 1. Evaluate the D-region polynomial exactly at HDX
            x_HDX = HDX - hmD
            NDX = NmD * np.exp(F1 * x_HDX + F2 * x_HDX**2 + F3 * x_HDX**3)
            
            # 2. Calculate the derivative (DN) of the polynomial at HDX
            DN = NDX * (F1 + 2.0 * F2 * x_HDX + 3.0 * F3 * x_HDX**2)
            
            # 3. Calculate continuity coefficients (K and D1)
            if NDX > 0 and NmE > 0 and NmE != NDX and hmE != HDX:
                K = - (DN * (hmE - HDX)) / (NDX * np.log(NDX / NmE))
                
                if K > 5.0:
                    K = 5.0
                    D1 = -np.log(NDX / NmE) / ((hmE - HDX)**K)
                else:
                    D1 = DN / (NDX * K * (hmE - HDX)**(K - 1))
            else:
                K = 2.0
                D1 = -np.log(NDX / NmE) / ((hmE - HDX)**K) if NmE > 0 else 0.0
                
            # 4. Apply the E-bottomside equation
            ne_profile[m_eb] = NmE * np.exp(-D1 * ((hmE - h[m_eb])**K))

    # ---------------------------------------------------------
    # 6. D Region (below HDX)
    # ---------------------------------------------------------
    if hmD is not None and HDX is not None:
        # The D-region sweeps all the way up to HDX to meet the E-bottomside
        m_d = h < HDX
        if np.any(m_d):
            x = (h[m_d] - hmD)
            # Eq. 30: 3rd-degree exponential polynomial
            ne_profile[m_d] = NmD * np.exp(F1 * x + F2 * x**2 + F3 * x**3)

    return ne_profile