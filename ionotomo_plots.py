"""
ionotomo_plots.py — Standalone geo-visualization helpers for ionospheric data.

All functions take plain NumPy arrays as input so they can be used independently
of any specific project data pipeline.

Requirements
------------
    numpy, matplotlib, cartopy, pyproj

    Install with:
        pip install numpy matplotlib cartopy pyproj

Quick-start
-----------
    from ionotomo_plots import (
        plot_global_ground_tracks,
        plot_regional_orthographic,
        ecef_to_lonlat,
        draw_raypath,
    )

    # 1 — Global scatter coloured by some scalar (e.g. group size, TEC, ...)
    plot_global_ground_tracks(
        lons, lats, c=values,
        cbar_label="Value", title="My occultations",
        save_path="globe.png",
    )

    # 2 — Regional orthographic globe with a triangular-mesh scalar overlay
    plot_regional_orthographic(
        center_lon=15.0, center_lat=52.0,
        vertices=verts,          # (N, 2) lon/lat array
        triangles=tris,          # (M, 3) vertex-index array
        scalar_data=delta_ne,    # (N,) values on vertices
        cbar_label="ΔNe [m⁻³]",
        save_path="region.png",
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ─────────────────────────────────────────────────────────────────────────────

# Module-level transformer (created once, reused on every call).
_ECEF_TRANSFORMER = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=geocent  +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=longlat  +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)


def ecef_to_lonlat(x_km, y_km, z_km):
    """
    Convert ECEF coordinates (kilometres) to geodetic longitude, latitude,
    and altitude.

    Parameters
    ----------
    x_km, y_km, z_km : float or array-like
        ECEF position components in **kilometres**.

    Returns
    -------
    lon_deg : same shape as input  — geodetic longitude  [°]
    lat_deg : same shape as input  — geodetic latitude   [°]
    alt_m   : same shape as input  — altitude above WGS84 ellipsoid [m]
    """
    lon_deg, lat_deg, alt_m = _ECEF_TRANSFORMER.transform(
        np.asarray(x_km) * 1e3,
        np.asarray(y_km) * 1e3,
        np.asarray(z_km) * 1e3,
    )
    return lon_deg, lat_deg, alt_m


# ─────────────────────────────────────────────────────────────────────────────
# Basemap helper
# ─────────────────────────────────────────────────────────────────────────────

def add_cartopy_basemap(
    ax,
    land_color="whitesmoke",
    ocean_color="aliceblue",
    coastline_scale="110m",
    border_scale="110m",
    gridlines=True,
    grid_alpha=0.4,
    grid_lw=0.3,
):
    """
    Add standard land / ocean / coastline features and optional grid lines to a
    Cartopy GeoAxes.

    Parameters
    ----------
    ax : cartopy GeoAxes
        Axes to decorate.
    land_color : str, optional
    ocean_color : str, optional
    coastline_scale : str
        Cartopy resolution string — "110m", "50m", or "10m".
    border_scale : str
        Cartopy resolution string for country borders.
    gridlines : bool
    grid_alpha, grid_lw : float
        Appearance of the gridline overlay.
    """
    ax.add_feature(cfeature.LAND,  facecolor=land_color,  zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0)
    ax.add_feature(
        cfeature.COASTLINE.with_scale(coastline_scale),
        linewidth=0.5, edgecolor="gray", zorder=1,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(border_scale),
        linewidth=0.3, edgecolor="lightgray", zorder=1,
    )
    if gridlines:
        ax.gridlines(lw=grid_lw, alpha=grid_alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Raypath drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_raypath(
    ax,
    leo_ecef_km,
    gnss_ecef_km,
    ray_index=0,
    color="steelblue",
    lw=1.5,
    ls="-",
    alt_max_km=800.0,
    n_pts=120,
    mark_tangent=True,
    tangent_ms=5,
    label=None,
    zorder=6,
):
    """
    Draw a single GNSS→LEO radio-occultation ray path on a Cartopy GeoAxes.

    The ray is parametrised as a straight line in ECEF space between the GNSS
    and LEO positions.  Only the ionospheric portion below ``alt_max_km`` is
    drawn.  The tangent (closest-approach) point is marked with a filled circle.

    Parameters
    ----------
    ax : cartopy GeoAxes
    leo_ecef_km : array, shape (3,) or (3, N)
        LEO satellite ECEF position(s) in kilometres.
        If shape is (3, N), ``ray_index`` selects the column to use.
    gnss_ecef_km : array, shape (3,) or (3, N)
        GNSS satellite ECEF position(s) in kilometres.
    ray_index : int
        Column index when arrays are (3, N).  Ignored for (3,) inputs.
    color : matplotlib colour
    lw : float — line width
    ls : str   — line style
    alt_max_km : float
        Only draw ray segments below this altitude [km].
    n_pts : int
        Number of sample points along the parametric line.
    mark_tangent : bool
        If True, plot a filled circle at the tangent point.
    tangent_ms : float — tangent-point marker size.
    label : str or None — legend label (applied to the tangent-point marker).
    zorder : int

    Returns
    -------
    plotted : bool — True if at least one segment was drawn.
    """
    leo_ecef_km  = np.asarray(leo_ecef_km,  dtype=float)
    gnss_ecef_km = np.asarray(gnss_ecef_km, dtype=float)

    # Extract the selected ray if inputs are (3, N)
    if leo_ecef_km.ndim == 2:
        leo_r  = leo_ecef_km[:,  ray_index]
        gnss_r = gnss_ecef_km[:, ray_index]
    else:
        leo_r  = leo_ecef_km
        gnss_r = gnss_ecef_km

    # Parametric interpolation from GNSS (t=0) to LEO (t=1)
    t_vals = np.linspace(0.0, 1.0, n_pts)
    pts    = gnss_r[:, None] + (leo_r[:, None] - gnss_r[:, None]) * t_vals
    lon, lat, alt_m = ecef_to_lonlat(pts[0], pts[1], pts[2])

    iono_mask = alt_m < alt_max_km * 1e3
    plotted   = False
    if np.any(iono_mask):
        ax.plot(
            lon[iono_mask], lat[iono_mask],
            transform=ccrs.Geodetic(),
            color=color, lw=lw, ls=ls, zorder=zorder,
        )
        plotted = True

    if mark_tangent:
        # Closest-approach point: minimise |gnss + t*(leo-gnss)|
        v   = leo_r - gnss_r
        t_s = np.clip(-np.dot(v, gnss_r) / np.dot(v, v), 0.0, 1.0)
        tp  = gnss_r + v * t_s
        tp_lon, tp_lat, _ = ecef_to_lonlat(tp[0], tp[1], tp[2])
        ax.plot(
            tp_lon, tp_lat,
            transform=ccrs.Geodetic(),
            marker="o", color=color, ms=tangent_ms,
            mec="black", mew=0.6,
            zorder=zorder + 1,
            label=label,
        )

    return plotted


# ─────────────────────────────────────────────────────────────────────────────
# Region-of-interest boundary
# ─────────────────────────────────────────────────────────────────────────────

def draw_roi_boundary(
    ax,
    roi_type,
    bounds,
    color="lime",
    lw=2.0,
    ls="-",
    alpha=0.9,
    label=None,
    n_pts=60,
    zorder=3,
):
    """
    Draw a geographic region-of-interest boundary on a Cartopy GeoAxes.

    Parameters
    ----------
    ax : cartopy GeoAxes
    roi_type : str
        ``"rect"``    — mid-latitude rectangle.
        ``"polar_n"`` — northern polar cap boundary (latitude circle).
        ``"polar_s"`` — southern polar cap boundary (latitude circle).
    bounds : tuple
        For ``"rect"``    : ``(lat_south, lat_north, lon_west, lon_east)`` in degrees.
        For ``"polar_n"`` : ``(lat_threshold,)``  — the boundary latitude (e.g. 65.0).
        For ``"polar_s"`` : ``(lat_threshold,)``  — the boundary latitude (e.g. -65.0).
    color, lw, ls, alpha, zorder : matplotlib style arguments.
    label : str or None — legend label (applied to the last segment drawn).
    n_pts : int — number of points used to draw each edge.
    """
    kw = dict(transform=ccrs.PlateCarree(), color=color,
              lw=lw, ls=ls, alpha=alpha, zorder=zorder)

    if roi_type == "rect":
        lat0, lat1, lon0, lon1 = bounds
        lats_lr = np.linspace(lat0, lat1, n_pts)
        lons_tb = np.linspace(lon0, lon1, n_pts)
        ax.plot(lons_tb,        [lat0] * n_pts, **kw)
        ax.plot(lons_tb,        [lat1] * n_pts, **kw)
        ax.plot([lon0] * n_pts, lats_lr,        **kw)
        ax.plot([lon1] * n_pts, lats_lr,        **kw,
                label=label or f"{lat0:.0f}–{lat1:.0f}°N  {lon0:.0f}–{lon1:.0f}°E")

    elif roi_type in ("polar_n", "polar_s"):
        (lat_threshold,) = bounds
        lons = np.linspace(-180, 180, 361)
        ax.plot(
            lons, [lat_threshold] * 361,
            **kw,
            label=label or f"Polar boundary ({lat_threshold:+.0f}°)",
        )
    else:
        raise ValueError(f"Unknown roi_type '{roi_type}'. "
                         "Use 'rect', 'polar_n', or 'polar_s'.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Global ground-track / tangent-point map
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_ground_tracks(
    lons,
    lats,
    c=None,
    cmap="plasma",
    vmin=None,
    vmax=None,
    cbar_label="",
    title="",
    show_grid=True,
    grid_dlat=20.0,
    grid_dlon=50.0,
    polar_threshold=65.0,
    grid_color="steelblue",
    grid_alpha=0.4,
    polar_line_color="darkorange",
    s=60,
    edgecolors="black",
    linewidths=0.4,
    land_color="lightgray",
    ocean_color="aliceblue",
    figsize=(16, 9),
    save_path=None,
    ax=None,
    dpi=150,
):
    """
    Robinson-projection world map with scatter points for occultation tangent
    (or any other geographic) positions.

    Parameters
    ----------
    lons, lats : array-like, shape (N,)
        Geodetic longitudes and latitudes of the points [degrees].
    c : array-like, shape (N,) or None
        Scalar values used to colour the scatter points.  If None, all points
        are drawn in a single default colour.
    cmap : str — matplotlib colormap name.
    vmin, vmax : float or None — colour scale limits.
    cbar_label : str — label for the colour bar.
    title : str — figure title.
    show_grid : bool
        If True, draw a lat/lon grid of mid-latitude region bins and polar-cap
        boundary lines.
    grid_dlat, grid_dlon : float
        Spacing of the mid-latitude grid lines [degrees].
    polar_threshold : float
        Latitude (absolute) at which polar-cap boundary lines are drawn.
    s : float — scatter point size.
    figsize : tuple
    save_path : str or None — if given, the figure is saved here.
    ax : cartopy GeoAxes or None
        Pass an existing axes to draw into; otherwise a new figure is created.
    dpi : int — resolution for saved figures.

    Returns
    -------
    fig, ax : matplotlib Figure and GeoAxes
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(
            1, 1, figsize=figsize,
            subplot_kw={"projection": ccrs.Robinson()},
        )
    else:
        fig = ax.get_figure()

    ax.set_global()
    add_cartopy_basemap(ax, land_color=land_color, ocean_color=ocean_color)

    # ── Mid-latitude grid lines
    if show_grid:
        for lat_line in np.arange(-90, 91, grid_dlat):
            if abs(lat_line) <= polar_threshold:
                ax.plot(
                    [-180, 180], [lat_line, lat_line],
                    transform=ccrs.PlateCarree(),
                    lw=0.6, ls="--", color=grid_color, alpha=grid_alpha, zorder=1,
                )
        for lon_line in np.arange(-180, 181, grid_dlon):
            ax.plot(
                [lon_line, lon_line],
                [-polar_threshold, polar_threshold],
                transform=ccrs.PlateCarree(),
                lw=0.6, ls="--", color=grid_color, alpha=grid_alpha, zorder=1,
            )
        for pole_lat in (polar_threshold, -polar_threshold):
            ax.plot(
                np.linspace(-180, 180, 360), [pole_lat] * 360,
                transform=ccrs.PlateCarree(),
                lw=1.2, ls="-", color=polar_line_color, alpha=0.7, zorder=2,
            )

    # ── Scatter
    scatter_kw = dict(
        transform=ccrs.Geodetic(),
        s=s,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=5,
    )
    if c is not None:
        c = np.asarray(c, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(c))
        if vmax is None:
            vmax = float(np.nanmax(c))
        norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
        sc   = ax.scatter(lons, lats, c=c, cmap=cmap, norm=norm, **scatter_kw)
        cbar = fig.colorbar(sc, ax=ax, orientation="vertical",
                            fraction=0.025, pad=0.04)
        cbar.set_label(cbar_label, fontsize=11)
    else:
        ax.scatter(lons, lats, **scatter_kw)

    if title:
        ax.set_title(title, fontsize=12)

    if save_path and own_figure:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Regional orthographic globe with mesh overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_regional_orthographic(
    center_lon,
    center_lat,
    vertices=None,
    triangles=None,
    scalar_data=None,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    symmetric_clim=True,
    cbar_label="",
    cbar_orientation="horizontal",
    show_mesh_edges=False,
    mesh_edge_color="gray",
    mesh_edge_lw=0.4,
    mesh_edge_alpha=0.5,
    center_marker=None,
    center_marker_color="yellow",
    center_marker_ms=14,
    raypaths=None,
    roi=None,
    land_color="whitesmoke",
    ocean_color="aliceblue",
    title="",
    figsize=(7, 7),
    save_path=None,
    ax=None,
    dpi=150,
):
    """
    Orthographic globe centred on a region of interest.

    Optionally overlays:
      • A triangular-mesh scalar field (e.g. ΔNe or absolute Ne).
      • Occultation ray paths.
      • A region-of-interest boundary.
      • A centre-vertex star marker.

    Parameters
    ----------
    center_lon, center_lat : float
        Centre of the orthographic projection [degrees].
    vertices : array, shape (N, 2), optional
        Mesh vertex positions — column 0 = longitude, column 1 = latitude.
    triangles : array, shape (M, 3) int, optional
        Vertex indices forming each triangle (same convention as
        ``matplotlib.tri.Triangulation``).
    scalar_data : array, shape (N,) or (N, 1), optional
        Scalar field to colour the mesh.  Requires ``vertices`` and
        ``triangles`` to also be provided.
    cmap : str
    vmin, vmax : float or None — colour scale limits.
    symmetric_clim : bool
        If True and both vmin/vmax are None, set limits to ±max(|data|) so a
        diverging colormap is centred at zero.
    cbar_label : str
    cbar_orientation : str — "horizontal" or "vertical".
    show_mesh_edges : bool
        If True, draw the triangle edges on top of the fill.
    center_marker : tuple (lon, lat) or None
        If given, a star (★) is drawn at this position.
    raypaths : list of dict or None
        Each dict must contain:
            ``"leo"``  — ECEF positions, shape (3,) or (3, N) [km]
            ``"gnss"`` — ECEF positions, shape (3,) or (3, N) [km]
        Optional keys (all have defaults):
            ``"ray_index"`` (int, default 0),
            ``"color"``     (matplotlib color, default "steelblue"),
            ``"lw"``        (float, default 1.5),
            ``"ls"``        (str, default "-"),
            ``"label"``     (str or None),
            ``"alt_max_km"`` (float, default 800).
    roi : dict or None
        Dictionary with keys:
            ``"type"``   — "rect", "polar_n", or "polar_s"
            ``"bounds"`` — as described in :func:`draw_roi_boundary`.
        Optional keys: ``"color"``, ``"lw"``, ``"label"``.
    title : str
    figsize : tuple
    save_path : str or None
    ax : cartopy GeoAxes or None
    dpi : int

    Returns
    -------
    fig, ax : matplotlib Figure and GeoAxes
    """
    own_figure = ax is None
    if own_figure:
        proj = ccrs.Orthographic(
            central_longitude=center_lon,
            central_latitude=center_lat,
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize,
                               subplot_kw={"projection": proj})
    else:
        fig = ax.get_figure()

    ax.set_global()
    add_cartopy_basemap(ax, land_color=land_color, ocean_color=ocean_color)

    # ── Triangular mesh scalar fill
    if vertices is not None and triangles is not None and scalar_data is not None:
        v      = np.asarray(vertices)
        t      = np.asarray(triangles, dtype=int)
        data   = np.asarray(scalar_data, dtype=float).ravel()

        if vmin is None and vmax is None and symmetric_clim:
            mx   = float(np.nanmax(np.abs(data)))
            vmin, vmax = -mx, mx
        elif vmin is None:
            vmin = float(np.nanmin(data))
        elif vmax is None:
            vmax = float(np.nanmax(data))

        if not np.isclose(vmin, vmax):
            tc = ax.tripcolor(
                v[:, 0], v[:, 1], t, data,
                transform=ccrs.Geodetic(),
                cmap=cmap, shading="flat",
                vmin=vmin, vmax=vmax,
                zorder=1,
            )
            cbar_kw = dict(
                ax=ax,
                orientation=cbar_orientation,
                shrink=0.75, pad=0.04, fraction=0.04,
            )
            cbar = fig.colorbar(tc, **cbar_kw)
            cbar.set_label(cbar_label, fontsize=9)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()

        if show_mesh_edges:
            ax.triplot(
                v[:, 0], v[:, 1], t,
                transform=ccrs.Geodetic(),
                color=mesh_edge_color, lw=mesh_edge_lw, alpha=mesh_edge_alpha,
                zorder=2,
            )

    # ── Centre-vertex marker
    if center_marker is not None:
        clon_m, clat_m = center_marker
        ax.plot(
            clon_m, clat_m,
            transform=ccrs.Geodetic(),
            marker="*", color=center_marker_color,
            ms=center_marker_ms, mec="black", mew=0.8,
            zorder=8, label="Centre vertex",
        )

    # ── Region-of-interest boundary
    if roi is not None:
        draw_roi_boundary(
            ax,
            roi_type=roi["type"],
            bounds=roi["bounds"],
            color=roi.get("color", "lime"),
            lw=roi.get("lw", 2.0),
            label=roi.get("label", None),
        )

    # ── Ray paths
    if raypaths is not None:
        for rp in raypaths:
            draw_raypath(
                ax,
                leo_ecef_km=rp["leo"],
                gnss_ecef_km=rp["gnss"],
                ray_index=rp.get("ray_index", 0),
                color=rp.get("color", "steelblue"),
                lw=rp.get("lw", 1.5),
                ls=rp.get("ls", "-"),
                label=rp.get("label", None),
                alt_max_km=rp.get("alt_max_km", 800.0),
            )

    if title:
        ax.set_title(title, fontsize=11)

    ax.legend(loc="lower left", fontsize=7, framealpha=0.75)

    if save_path and own_figure:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Mesh geometry diagnostic (triangular mesh + ray paths)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mesh_geometry(
    vertices,
    triangles,
    center_lon,
    center_lat,
    leo_ecef_km=None,
    gnss_ecef_km=None,
    tangent_lla=None,
    alt_limit_km=600.0,
    mesh_color="darkorange",
    mesh_lw=0.8,
    mesh_alpha=0.7,
    title=None,
    figsize=(6, 6),
    save_path=None,
    dpi=300,
):
    """
    Orthographic globe showing the triangular inversion mesh and (optionally)
    the occultation ray geometry.

    Parameters
    ----------
    vertices : array, shape (N, 2)
        Column 0 = longitude, column 1 = latitude [degrees].
    triangles : array, shape (M, 3)
        Triangle vertex index array.
    center_lon, center_lat : float
        Centre of the orthographic projection.
    leo_ecef_km : array, shape (3,) or (3, N), optional
        LEO satellite ECEF positions [km].
    gnss_ecef_km : array, shape (3,) or (3, N), optional
        GNSS satellite ECEF positions [km].
    tangent_lla : array, shape (3, N), optional
        Pre-computed tangent-point positions: rows are [lat_deg, lon_deg, alt_m].
        Only points below ``alt_limit_km`` are drawn.
    alt_limit_km : float
        Altitude threshold for displaying the tangent path [km].
    mesh_color, mesh_lw, mesh_alpha : mesh style.
    title : str or None
    figsize, save_path, dpi : figure output options.

    Returns
    -------
    fig, ax
    """
    proj = ccrs.Orthographic(
        central_longitude=center_lon, central_latitude=center_lat,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": proj})
    ax.set_global()
    add_cartopy_basemap(ax)

    v = np.asarray(vertices)
    t = np.asarray(triangles, dtype=int)
    ax.triplot(
        v[:, 0], v[:, 1], t,
        transform=ccrs.Geodetic(),
        color=mesh_color, lw=mesh_lw, alpha=mesh_alpha,
    )

    if leo_ecef_km is not None and gnss_ecef_km is not None:
        leo  = np.asarray(leo_ecef_km,  dtype=float)
        gnss = np.asarray(gnss_ecef_km, dtype=float)
        # Plot the first-column start positions
        col = (slice(None), 0) if leo.ndim == 2 else (slice(None),)
        leo_start  = leo[col]
        gnss_start = gnss[col]
        leo_lon,  leo_lat,  _ = ecef_to_lonlat(*leo_start)
        gnss_lon, gnss_lat, _ = ecef_to_lonlat(*gnss_start)
        ax.plot(leo_lon,  leo_lat,  transform=ccrs.Geodetic(),
                color="green", marker="^", ms=6, label="LEO start",  zorder=5)
        ax.plot(gnss_lon, gnss_lat, transform=ccrs.Geodetic(),
                color="red",   marker="s", ms=6, label="GNSS start", zorder=5)

    if tangent_lla is not None:
        tlla = np.asarray(tangent_lla, dtype=float)   # (3, N): lat, lon, alt_m
        mask = tlla[2, :] < alt_limit_km * 1e3
        if np.any(mask):
            ax.plot(
                tlla[1, mask], tlla[0, mask],
                transform=ccrs.Geodetic(),
                color="magenta", lw=2, label="Tangent path",
            )

    ax.set_title(
        title if title else f"Mesh geometry (below {alt_limit_km:.0f} km)",
        fontsize=11,
    )
    if leo_ecef_km is not None or tangent_lla is not None:
        ax.legend(loc="lower left", fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo — run `python ionotomo_plots.py` to generate example figures
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("example_figures", exist_ok=True)

    rng = np.random.default_rng(42)
    N   = 80

    # ── Demo 1: Global ground-track scatter ──────────────────────────────────
    lons_demo   = rng.uniform(-180,  180, N)
    lats_demo   = rng.uniform( -70,   70, N)
    values_demo = rng.integers(1, 6, N).astype(float)

    plot_global_ground_tracks(
        lons_demo, lats_demo,
        c=values_demo,
        cmap="plasma",
        cbar_label="Group size (occultations)",
        title="Demo — Occultation tangent points coloured by group size\n"
              "20°lat × 50°lon mid-lat grid  |  orange = polar caps (65°)",
        polar_threshold=65.0,
        save_path="example_figures/demo_global_ground_tracks.png",
    )
    print("Saved  example_figures/demo_global_ground_tracks.png")

    # ── Demo 2: Regional orthographic with mesh + ROI ────────────────────────
    # Simple rectangular mesh over Europe
    lats_v = np.linspace(40, 65, 6)
    lons_v = np.linspace(-10, 40, 7)
    lon_g, lat_g = np.meshgrid(lons_v, lats_v)
    verts = np.column_stack([lon_g.ravel(), lat_g.ravel()])

    from matplotlib.tri import Triangulation
    tri_obj = Triangulation(verts[:, 0], verts[:, 1])
    tris    = tri_obj.triangles

    scalar  = rng.standard_normal(len(verts)) * 5e10   # dummy ΔNe values

    plot_regional_orthographic(
        center_lon=15.0, center_lat=52.0,
        vertices=verts, triangles=tris, scalar_data=scalar,
        cmap="coolwarm",
        cbar_label="ΔNe [m⁻³]",
        cbar_orientation="horizontal",
        show_mesh_edges=True,
        center_marker=(15.0, 52.0),
        roi={"type": "rect", "bounds": (40, 65, -10, 40), "color": "lime"},
        title="Demo — Regional ΔNe overlay (Europe)",
        save_path="example_figures/demo_regional_orthographic.png",
    )
    print("Saved  example_figures/demo_regional_orthographic.png")

    # ── Demo 3: Mesh geometry ─────────────────────────────────────────────────
    plot_mesh_geometry(
        verts, tris,
        center_lon=15.0, center_lat=52.0,
        mesh_color="darkorange",
        title="Demo — Triangular mesh geometry",
        save_path="example_figures/demo_mesh_geometry.png",
    )
    print("Saved  example_figures/demo_mesh_geometry.png")
