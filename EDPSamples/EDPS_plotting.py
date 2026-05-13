import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_edp_statistics(eds):
    """
    Plots a 3-panel statistical overview of an EDPSamples dataset using robust
    non-Gaussian percentiles.
    """
    # Extract data from the dataset
    altitude = eds.coords['altitude'].values
    edps = eds.data_vars['EDPs'].values 
    
    # 1. NEW: Calculate percentiles instead of mean/std (axis=(1, 2) collapses geo and sample dims)
    # Using 1, 5, 16, 50 (Median), 84, 95, and 99 percentiles
    percentiles = np.nanpercentile(edps, [1, 5, 16, 50, 84, 95, 99], axis=(1, 2))
    p01, p05, p16, median_edp, p84, p95, p99 = percentiles
    
    # Avoid division by zero in subplot 2
    safe_median = np.where(median_edp == 0, np.nan, median_edp)

    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    fig.suptitle("EDP Ensemble Statistics (Non-Gaussian)", fontsize=16, fontweight='bold', y=0.95)

    # ---------------------------------------------------------
    # Subplot 1: Absolute Median and Percentile Ranges
    # ---------------------------------------------------------
    ax1 = axes[0]
    
    # Notice we don't need np.clip(..., 0, None) anymore! Percentiles of positive data cannot be negative.
    # Outer Band (1st to 99th percentile)
    ax1.fill_betweenx(altitude, p01, p99, color='lightblue', alpha=0.3, label='1st–99th %ile')
    # Middle Band (5th to 95th percentile)
    ax1.fill_betweenx(altitude, p05, p95, color='dodgerblue', alpha=0.5, label='5th–95th %ile')
    # Inner Band (16th to 84th percentile)
    ax1.fill_betweenx(altitude, p16, p84, color='blue', alpha=0.7, label='16th–84th %ile')
    
    # Plot Median instead of Mean
    ax1.plot(median_edp, altitude, color='black', lw=2, label='Median EDP')
    
    ax1.set_ylim([0, 700])
    ax1.set_xlabel("Electron Density (m⁻³)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Absolute Profile Spread")
    ax1.grid(True, alpha=0.4, linestyle=':')
    ax1.legend(loc='lower right')

    # ---------------------------------------------------------
    # Subplot 2: Normalized Spread (Percentage Deviation from Median)
    # ---------------------------------------------------------
    ax2 = axes[1]
    
    ax2.axvline(0, color='black', lw=2, label='Median (Baseline)')
    
    # NEW: Calculate normalized percentage bounds dynamically
    def norm_pct(p_array):
        return (p_array - safe_median) / safe_median * 100

    ax2.fill_betweenx(altitude, norm_pct(p01), norm_pct(p99), color='lightcoral', alpha=0.3, label='1st–99th %ile')
    ax2.fill_betweenx(altitude, norm_pct(p05), norm_pct(p95), color='indianred', alpha=0.5, label='5th–95th %ile')
    ax2.fill_betweenx(altitude, norm_pct(p16), norm_pct(p84), color='darkred', alpha=0.7, label='16th–84th %ile')

    ax2.set_ylim([0, 700])
    ax2.set_xlabel("Deviation from Median (%)")
    ax2.set_title("Normalized Profile Spread")
    ax2.grid(True, alpha=0.4, linestyle=':')
    
    # Restrict x-axis limits dynamically based on the 95th percentile to keep the plot readable
    max_plot_pct = np.nanmax(np.abs(norm_pct(p95))) 
    ax2.set_xlim(-max_plot_pct, max_plot_pct)
    ax2.legend(loc='lower right')

    # ---------------------------------------------------------
    # Subplot 3: 2D Probability Density Histogram
    # ---------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor('black') 
    
    n_profiles = edps.shape[1] * edps.shape[2]
    flat_alts = np.tile(altitude, n_profiles)
    flat_edps = edps.reshape(len(altitude), -1).flatten(order='F')
    
    valid_mask = ~np.isnan(flat_edps)
    
    h = ax3.hist2d(flat_edps[valid_mask], flat_alts[valid_mask], 
                   bins=[60, len(altitude)], cmap='viridis', norm=LogNorm())
    
    # NEW: Overlay the Median instead of Mean
    ax3.plot(median_edp, altitude, color='white', lw=1.5, linestyle='--', label='Median')

    ax3.set_ylim([0, 700])
    ax3.set_xlabel("Electron Density (m⁻³)")
    ax3.set_title("True Ensemble Probability Density")
    cbar = fig.colorbar(h[3], ax=ax3)
    cbar.set_label('Count (Density)')
    ax3.grid(True, alpha=0.3, color='white', linestyle=':')
    ax3.legend(loc='lower right')

    plt.tight_layout()
    plt.show()