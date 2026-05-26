"""
viz_mesh_profiles.py
--------------------
Two-panel visualization of a scalar function f(lat, lon, alt) defined on a
triangular lat/lon mesh with a common altitude grid.

Panel 1 — 2-D scalar field at the altitude of maximum horizontal variability,
           coloured by function value.  Optionally rendered in polar coordinates
           (latitude = radius, longitude = azimuth) for polar-region data.
Panel 2 — Vertical profiles at every mesh vertex, each curve coloured by
           the same colour that represents its position in Panel 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import matplotlib.cm as cm
#import matplotlib

#matplotlib.use('TkAgg')

# ── polar-projection helpers ───────────────────────────────────────────────────

def _polar_xy(lats, lons, pole):
    """
    Azimuthal equidistant projection centred on a pole.
      r     = co-latitude  (0 at pole, increases equatorward)
      theta = longitude in radians
    Returns (x, y) suitable for tripcolor on a regular Cartesian axes.
    """
    r     = (90.0 - lats) if pole == 'N' else (lats + 90.0)
    theta = np.deg2rad(lons)
    return r * np.cos(theta), r * np.sin(theta)


def _decorate_polar(ax, lats, lons, pole):
    """
    Overlay latitude rings and longitude spokes on the Cartesian axes used
    for the polar projection, then remove the regular axis frame and ticks.
    """
    theta_ring = np.linspace(0, 2 * np.pi, 721)

    lat_span = lats.max() - lats.min()
    dlat     = 5.0 if lat_span <= 20 else 10.0
    dlon     = 30.0

    # r of the outermost data point — governs spoke length and label radius
    r_outer = (90.0 - lats.min()) if pole == 'N' else (lats.max() + 90.0)
    label_r = r_outer * 1.10

    # Latitude rings
    lat_lo = np.floor(lats.min() / dlat) * dlat
    lat_hi = np.ceil (lats.max() / dlat) * dlat
    for lat in np.arange(lat_lo, lat_hi + dlat * 0.1, dlat):
        r = (90.0 - lat) if pole == 'N' else (lat + 90.0)
        if r < 0:
            continue
        ax.plot(r * np.cos(theta_ring), r * np.sin(theta_ring),
                color='gray', linewidth=0.5, alpha=0.5, zorder=1)
        # Label on the positive-x axis (lon = 0° direction)
        ax.text(r, 0, f' {lat:.0f}°', ha='left', va='center',
                fontsize=7, color='dimgray')

    # Longitude spokes
    lon_lo = np.floor(lons.min() / dlon) * dlon
    lon_hi = np.ceil (lons.max() / dlon) * dlon
    for lon in np.arange(lon_lo, lon_hi + dlon * 0.1, dlon):
        th = np.deg2rad(lon)
        ax.plot([0, r_outer * np.cos(th)], [0, r_outer * np.sin(th)],
                color='gray', linewidth=0.5, alpha=0.5, zorder=1)
        ax.text(label_r * np.cos(th), label_r * np.sin(th),
                f'{lon:.0f}°', ha='center', va='center',
                fontsize=7, color='dimgray')

    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)


# ── main function ──────────────────────────────────────────────────────────────

def visualize_scalar_field(vertices, triangles, altitudes, values,
                            cmap='viridis', figsize=(14, 6),
                            polar=False, pole='N',title_txt:str=None):
    """
    Parameters
    ----------
    vertices  : (N, 2) array-like  — [[lat, lon], ...] for each mesh vertex
    triangles : (M, 3) array-like  — vertex-index triples defining triangles
    altitudes : (K,)   array-like  — common altitude grid shared by all vertices
    values    : (N, K) array-like  — f(vertex_i, altitude_k)
    cmap      : matplotlib colormap name (default 'viridis')
    figsize   : figure size tuple
    polar     : if True, Panel 1 uses a polar projection (lat=radius,
                lon=azimuth) instead of a plain lat/lon Cartesian plot
    pole      : 'N' (default) — North Pole at centre (r = 90 − lat)
                'S'           — South Pole at centre (r = lat + 90)

    Returns
    -------
    fig : matplotlib Figure
    """
    vertices  = np.asarray(vertices,  dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    altitudes = np.asarray(altitudes, dtype=float)
    values    = np.asarray(values,    dtype=float)

    lats = vertices[:, 1]
    lons = vertices[:, 0]

    # ------------------------------------------------------------------
    # Altitude level with the greatest horizontal variability
    # ------------------------------------------------------------------
    horiz_std = np.std(values, axis=0)      # (K,) — std across vertices
    peak_idx  = int(np.argmax(horiz_std))
    peak_alt  = altitudes[peak_idx]
    peak_vals = values[:, peak_idx]         # (N,) — field at that level

    # Shared colour normalisation (same scale in both panels)
    vmin, vmax = peak_vals.min(), peak_vals.max()
    norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)
    try:
        colormap = plt.colormaps[cmap]      # matplotlib >= 3.5
    except AttributeError:
        colormap = cm.get_cmap(cmap)

    vertex_colors = colormap(norm(peak_vals))   # (N, 4) RGBA per vertex

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ------------------------------------------------------------------
    # Panel 1 — 2-D scalar field at the most-variable altitude
    # ------------------------------------------------------------------
    if polar:
        px, py  = _polar_xy(lats, lons, pole)
        triang  = mtri.Triangulation(px, py, triangles)
        tpc     = ax1.tripcolor(triang, peak_vals,
                                cmap=colormap, norm=norm, shading='gouraud')
        ax1.triplot(triang, color='k', linewidth=0.3, alpha=0.35)
        _decorate_polar(ax1, lats, lons, pole)
        pole_label = 'North' if pole == 'N' else 'South'
        if title_txt is None:
            ax1.set_title(
                f'Horizontal field at altitude = {peak_alt:.2f}\n'
                f'(max horizontal std = {horiz_std[peak_idx]:.3g})'
                f'  —  {pole_label} polar view'
            )
        else:
            ax1.set_title(
                f'{title_txt}: Horizontal field at altitude = {peak_alt:.2f}\n'
                f'(max horizontal std = {horiz_std[peak_idx]:.3g})'
                f'  —  {pole_label} polar view'
            )
            
    else:
        triang  = mtri.Triangulation(lons, lats, triangles)
        tpc     = ax1.tripcolor(triang, peak_vals,
                                cmap=colormap, norm=norm, shading='gouraud')
        ax1.triplot(triang, color='k', linewidth=0.3, alpha=0.35)
        ax1.set_xlabel('Longitude (°)')
        ax1.set_ylabel('Latitude (°)')
        if title_txt is None:
            ax1.set_title(
                f'Horizontal field  ·  altitude = {peak_alt:.2f}\n'
                f'(max horizontal std = {horiz_std[peak_idx]:.3g})'
            )
        else:
            ax1.set_title(
                f'{title_txt}: Horizontal field at altitude = {peak_alt:.2f}\n'
                f'(max horizontal std = {horiz_std[peak_idx]:.3g})'
            )

    plt.colorbar(tpc, ax=ax1, label='Function value')

    # ------------------------------------------------------------------
    # Panel 2 — vertical profiles, one per vertex.
    # Each curve carries the colour of its point in Panel 1.
    # ------------------------------------------------------------------
    for i in range(len(vertices)):
        ax2.plot(values[i], altitudes,
                 color=vertex_colors[i], linewidth=0.8, alpha=0.75)

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Function value at peak altitude')
    ax2.set_xlabel('Function value')
    ax2.set_ylabel('Altitude')
    if title_txt is None:
        ax2.set_title(
            'Vertical profiles\n'
            '(colour = value at peak altitude, matching left panel)'
        )
    else:
        ax2.set_title(
            f'{title_txt}: Vertical profiles\n'
            '(colour = value at peak altitude, matching left panel)'
        )

    plt.tight_layout()
    return fig


# ======================================================================
# Synthetic demos — replace with your real data
# ======================================================================
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    alts        = np.linspace(100.0, 1000.0, 60)
    alt_env     = np.exp(-((alts - 350.0) / 100.0) ** 2)   # peaks at 350 km

    # ---- mid-latitude demo (Cartesian) -----------------------------------
    n_mid       = 120
    lats_mid    = rng.uniform(20.0, 60.0, n_mid)
    lons_mid    = rng.uniform(-130.0, -60.0, n_mid)
    tri_mid     = mtri.Triangulation(lons_mid, lats_mid)

    horiz_mid   = np.exp(-((lats_mid - 40.0) ** 2 / 150.0
                            + (lons_mid + 95.0) ** 2 / 400.0))
    vals_mid    = np.outer(horiz_mid, alt_env)
    vals_mid   += 0.03 * rng.standard_normal(vals_mid.shape)

    fig1 = visualize_scalar_field(
        np.column_stack([lons_mid,lats_mid ]),
        tri_mid.triangles, alts, vals_mid,
        cmap='plasma', polar=False
    )
    fig1.suptitle('Mid-latitude demo — Cartesian view', y=1.01)

    # ---- North-polar demo ------------------------------------------------
    n_pol       = 120
    lats_pol    = rng.uniform(60.0, 88.0, n_pol)
    lons_pol    = rng.uniform(-180.0, 180.0, n_pol)

    # Triangulate in the projected plane to avoid distortion near the pole
    px_pol, py_pol = _polar_xy(lats_pol, lons_pol, pole='N')
    tri_pol        = mtri.Triangulation(px_pol, py_pol)

    horiz_pol   = np.exp(-((lats_pol - 75.0) ** 2 / 50.0
                            + (lons_pol - 45.0) ** 2 / 800.0))
    vals_pol    = np.outer(horiz_pol, alt_env)
    vals_pol   += 0.03 * rng.standard_normal(vals_pol.shape)

    fig2 = visualize_scalar_field(
        np.column_stack([lons_pol,lats_pol ]),
        tri_pol.triangles, alts, vals_pol,
        cmap='viridis', polar=True, pole='N'
    )
    fig2.suptitle('Polar demo — North Pole view', y=1.01)

    plt.show()
