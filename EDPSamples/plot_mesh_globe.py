#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:58:53 2026

@author: austinhunter
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

def ECEFtolla(ECEF):
    # ECEF can be shape (3,) for single point or (3, n) for multiple points
    if ECEF.ndim == 1:
        x, y, z = ECEF[0], ECEF[1], ECEF[2]
    else:
        x, y, z = ECEF[0, :], ECEF[1, :], ECEF[2, :]
    
    # Rest stays the same - pyproj handles arrays automatically
    transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True
        )
    lon_deg, lat_deg, alt = transformer.transform(1e3*x, 1e3*y, 1e3*z)
    
    return lat_deg, lon_deg, alt

def plot_globe_occultation_mesh(vertices, triangles, tecmax_lat, tecmax_lon, save_path, 
                                leo_data=None, gnss_data=None, tangent_lla=None, alt_limit=600.0):
    """
    Plots the triangular mesh and (optionally) the occultation ray paths 
    over an orthographic globe projection.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=tecmax_lon, 
                                               central_latitude=tecmax_lat))
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # --- Plot the Triangular Mesh ---
    # vertices[:, 0] is lon, vertices[:, 1] is lat
    mesh_lons = vertices[:, 0]
    mesh_lats = vertices[:, 1]
    print("attempting to plot occultation mesh region...")
    # triplot natively handles the (vertices, triangles) format we generated.
    # The Geodetic transform ensures the triangle edges wrap smoothly across the globe.
    print("Any NaNs in mesh?", np.isnan(vertices).any())
    ax.triplot(mesh_lons, mesh_lats, triangles, transform=ccrs.Geodetic(), 
               color='darkorange', linewidth=0.8, alpha=0.7)

    # --- Plot the LEO/GNSS Data (if provided) ---
    if leo_data is not None and gnss_data is not None:
        # NOTE: Assuming ECEFtolla is defined elsewhere in your script
        lat_leo, lon_leo, _ = ECEFtolla(leo_data[:, 0])
        lat_gnss, lon_gnss, _ = ECEFtolla(gnss_data[:, 0])
        
        ax.plot(lon_leo, lat_leo, transform=ccrs.Geodetic(), color='g', marker='^', markersize=6, label='LEO Start')
        ax.plot(lon_gnss, lat_gnss, transform=ccrs.Geodetic(), color='r', marker='s', markersize=6, label='GNSS Start')
        
    if tangent_lla is not None:
        mask = tangent_lla[2, :] < alt_limit * 1e3
        ax.plot(tangent_lla[1, mask], tangent_lla[0, mask], transform=ccrs.Geodetic(), color='m', linewidth=2, label='Tangent Path')
        
    plt.title(f"Occultation Mesh Geometry Below {alt_limit} km")
    
    # Only add legend if we actually plotted LEO/GNSS points
    if leo_data is not None:
        plt.legend(loc='lower left')
        
    # Save the figure
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(fig)
    # plt.close(fig)