# -*- coding: utf-8 -*-
"""
EDPSamples: EDP sample sets as specialized xarray.Dataset objects.

Stores altitude (1D), geolocation (lat/lon per point), sampling parameters
(one row per sample, multiple columns e.g. f107, f107_81, ig, ir), the main
EDPs field on (height, geo, sample), and optional Mesh (triangles as geo indices).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Any, MutableMapping
from dateutil.parser import parse

import matplotlib.pyplot as plt

import os
import logging
import subprocess

import numpy as np
import xarray as xr
import math
import pandas as pd

from scipy.spatial import cKDTree
from EDPSamples.generate_occultation_tri_mesh import generate_occultation_mesh

__all__ = ["EDPSamples"]

# WGS84 (geodetic <-> ECEF for satellite line-of-sight)
_WGS84_A = 6378137.0
_WGS84_E2 = 6.6943799901413165e-3


def _geodetic_to_ecef(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    alt_m: np.ndarray | float,
    ) -> np.ndarray:
    """Geodetic (deg, deg, m) to ECEF (m); broadcast. Last axis is x, y, z."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(alt_m, dtype=float)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + h) * sin_lat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def _ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ECEF (m) to geodetic latitude (deg), longitude (deg), height (m).
    xyz has shape (..., 3). Bowring-style solution, vectorized.
    """
    x = np.asarray(xyz[..., 0], dtype=float)
    y = np.asarray(xyz[..., 1], dtype=float)
    z = np.asarray(xyz[..., 2], dtype=float)
    b = _WGS84_A * np.sqrt(1.0 - _WGS84_E2)
    ep2 = (_WGS84_A * _WGS84_A - b * b) / (b * b)
    p = np.sqrt(x * x + y * y)
    theta = np.arctan2(z * _WGS84_A, p * b)
    st = np.sin(theta)
    ct = np.cos(theta)
    lat = np.arctan2(
        z + ep2 * b * st * st * st,
        p - _WGS84_E2 * _WGS84_A * ct * ct * ct,
    )
    lon = np.arctan2(y, x)
    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    h = p / np.cos(lat) - N
    pole = np.abs(p) < 1.0e-10
    if np.any(pole):
        h_pole = np.abs(z) - b
        lat = np.where(pole, np.sign(z) * (np.pi / 2.0), lat)
        h = np.where(pole, h_pole, h)
    return np.degrees(lat), np.degrees(lon), h

def get_IRI2020_EDP(DateTime: str,
                    altitude: np.ndarray,
                    geolocation: np.ndarray,
                    sampling_parameters: pd.DataFrame,
                    namelist_filename: str = "IRI2020_input_namelist.nml",
                    IRI_output_filename: str = "IRI_output_filename.dat") -> np.ndarray:
        
    flag = write_IRI2020_namelist(DateTime,
                    altitude,
                    geolocation,
                    sampling_parameters,
                    namelist_filename)
    assert flag == 0, "Fail to write input namelist for IRI2020"
    #
    # Execute IRI2020_namelist_driver
    #
    apf107 = Path("apf107.dat")
    ig_rz = Path("ig_rz.dat")

    assert apf107.is_file(), "The file apf107.dat must exist in the current folder"
    assert ig_rz.is_file(), "The file iG_rz.dat must exist in the current folder"

    iri_name = "iri2020_namelist_driver"
    if os.name == "nt":
        iri_name += ".exe"

    # %% run IRI with hard coded path to IRI2020 directory
    IRIPath = "/home/aistonhunter/IonosphereTomography/iri2020_new/src/iri2020/"
    exe = IRIPath+"iri2020_namelist_driver"
    IRIDataPath = IRIPath+"Data"
    data_path = IRIDataPath+"/mcsat*.dat"
    cmd = "ln -s " + data_path + " .;" 
    data_path = IRIDataPath+"/dgrf*.dat"
    cmd = cmd + "ln -s " + data_path + " .;" 
    data_path = IRIDataPath+"/*.asc"
    cmd = cmd+ "ln -s " + data_path + " .;" 
    cmd =cmd + str(exe) +" " + namelist_filename + " "+IRI_output_filename+";"
    data_path = "mcsat*.dat"
    cmd = cmd+"rm " + data_path + ";" 
    data_path = "dgrf*.dat"
    cmd = cmd+"rm " + data_path + ";" 
    data_path = "*.asc"
    cmd = cmd+"rm " + data_path + ";" 
    logging.info(" ".join(cmd))
    subprocess.run(cmd, shell = True)

    return read_IRI2020_binary_output(IRI_output_filename)
    
def write_IRI2020_namelist(DateTime: str,
                    altitude: np.ndarray,
                    geolocation: np.ndarray,
                    sampling_parameters: pd.DataFrame,
                    namelist_filename:str) -> int:
    # 
    # Write the namelist file to be read by IRI2020_namelis_driver
    #
    DateTime = parse(DateTime)
    #
    #    NAMELIST/EDPSamples/npts, nheight, nSample, height_grid,latitude,longitude, &
    #    idxF107D,idxap,idxIG12,idxRz12,idxfoF2,idxHmF2,idxB0,idxB1, idxHour, &
    #    year, month, day, hour, minute,second,phy_inputs,fill_value 
    #
    fill_value = -99999.
    npts = geolocation.shape[0]
    nheight = altitude.shape[0] 
    nSample = sampling_parameters.shape[0]
    with open(namelist_filename, 'w') as f:
        f.write('&EDPSamples\n')
        f.write(f'fill_value  = {fill_value}\n')    
        f.write(f'npts  = {npts}\n')    
        f.write(f'nheight  = {nheight}\n')    
        f.write(f'nSample  = {nSample}\n')
        f.write('idxHour = 1\n')
        f.write('idxF107D  = 2\n')
        f.write('idxap = 3\n')
        f.write('idxIG12  = 4\n')
        f.write('idxRz12 = 5\n')
        f.write('idxfoF2  = -1\n')
        f.write('idxHmF2 = -1\n')
        f.write('idxB0  = -1\n')
        f.write('idxB1  = -1\n')
        f.write(f'year  = {DateTime.year}\n')
        f.write(f'month  = {DateTime.month}\n')
        f.write(f'day  = {DateTime.day}\n')
        f.write(f'hour  = {DateTime.hour}\n')
        f.write(f'minute  = {DateTime.minute}\n')
        f.write(f'second  = {DateTime.second}\n')

        for idx in range(nheight):           
            f.write(f'height_grid({idx+1})  = {altitude[idx]}\n')
            
        for idx in range(npts):
            f.write(f'latitude({idx+1})  = {geolocation[idx,1]}\n')
            f.write(f'longitude({idx+1})  = {geolocation[idx,0]}\n')

        for idx in range(nSample):
            if np.isnan(sampling_parameters["hour"][idx]):
                f.write(f'phy_inputs(1,{idx+1})  = {fill_value}\n')
            else:
                f.write(f'phy_inputs(1,{idx+1})  = {sampling_parameters["hour"][idx]}\n')
                
            if np.isnan(sampling_parameters["f107"][idx]):
                f.write(f'phy_inputs(2,{idx+1})  = {fill_value}\n')
            else:
                f.write(f'phy_inputs(2,{idx+1})  = {sampling_parameters["f107"][idx]}\n')
                
            if np.isnan(sampling_parameters["ap"][idx]):
                f.write(f'phy_inputs(3,{idx+1})  = {fill_value}\n')
            else:                    
                f.write(f'phy_inputs(3,{idx+1})  = {sampling_parameters["ap"][idx]}\n')
                
            if np.isnan(sampling_parameters["ig12"][idx]):
                f.write(f'phy_inputs(4,{idx+1})  = {fill_value}\n')
            else:
                f.write(f'phy_inputs(4,{idx+1})  = {sampling_parameters["ig12"][idx]}\n')
                
            if np.isnan(sampling_parameters["rz12"][idx]):
                f.write(f'phy_inputs(5,{idx+1})  = {fill_value}\n')
            else:
                f.write(f'phy_inputs(5,{idx+1})  = {sampling_parameters["rz12"][idx]}\n')
        
        f.write('/ \n')

    return 0   
           

def read_IRI2020_binary_output(IRI_output_filename: str) -> np.ndarray:
    with open(IRI_output_filename, 'rb') as f:
        data =  f.read()
    
    npts=int.from_bytes(data[0:4],byteorder="little",signed=True) 
    nSample=int.from_bytes(data[4:8],byteorder="little",signed=True)
    nheight=int.from_bytes(data[8:12],byteorder="little",signed=True)
    print([nheight,npts,nSample])
    edps = np.ndarray([nheight,npts,nSample])
    cbyte = 12
    for idxSample in range(nSample):
        for idxpts in range(npts):
            for idxheight in range(nheight):
                edps[idxheight,idxpts,idxSample] = float(np.frombuffer(data[cbyte:cbyte+4],dtype=np.float32))
                cbyte += 4
                
    return edps

def plot_polar_mesh(vertices: np.ndarray, triangles: np.ndarray, pole: str = "north", ax=None):
    """
    Plots a triangular mesh in a polar projection centered on the north or south pole.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)
        Array of (longitude, latitude) in degrees.
    triangles : np.ndarray, shape (M, 3)
        Each row contains indices of the triangle corners.
    pole : str, "north" or "south"
        Which pole to center the projection on.
    ax : matplotlib axes, optional
        Existing axes to plot on.
    Returns
    -------
    ax : matplotlib axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    else:
        fig = ax.figure

    # Convert to radians for polar plotting
    lons = np.deg2rad(vertices[:, 0])
    lats = vertices[:, 1]

    # For north pole: r = 90 - lat, for south pole: r = 90 + lat
    if pole.lower() == "north":
        r = 90 - lats
        theta = lons
        ax.set_theta_zero_location('N')
        ax.set_ylim(0, 90-np.min(vertices[:,1]))
        ax.set_title("Polar Mesh (North Pole)", va='bottom')
    elif pole.lower() == "south":
        r = 90 + lats
        theta = lons
        ax.set_theta_zero_location('S')
        ax.set_ylim(0, 90-np.max(vertices[:,1]))
        ax.set_title("Polar Mesh (South Pole)", va='bottom')
    else:
        raise ValueError("pole must be 'north' or 'south'")

    # Plot each triangle
    for tri in triangles:
        pts = np.array([theta[tri], r[tri]])
        pts = np.column_stack((theta[tri], r[tri]))
        # Close the triangle
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color='k', lw=0.7, alpha=0.7)
        ax.fill(pts[:, 0], pts[:, 1], facecolor='C0', alpha=0.25, edgecolor=None)

    # Plot the vertices
    ax.scatter(theta, r, s=15, c='red', zorder=5, label='Vertices')

    # Optional: grid and labels for degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower right')
    if pole.lower() == "north":
        ax.set_yticklabels([f"{90-x}°" for x in ax.get_yticks()])
        ax.set_ylim(0, 90-np.min(vertices[:,1]))
    elif pole.lower() == "south":
        ax.set_yticklabels([f"{-90+x}°" for x in ax.get_yticks()])
        ax.set_ylim(0, 90+np.max(vertices[:,1]))
    else:
        raise ValueError("pole must be 'north' or 'south'")

    return ax

def plot_tri_mesh(vertices: np.ndarray, triangles: np.ndarray, ax=None):
    """
    Plot triangular mesh patches and their vertices on a map.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)
        Each row is (longitude, latitude) of a vertex.
    triangles : np.ndarray, shape (M, 3)
        Each row contains indices of the vertices composing one triangle.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Plot each triangle
    for tri in triangles:
        tri_pts = vertices[tri]
        # close the polygon by adding the first point to the end
        polygon = np.vstack([tri_pts, tri_pts[0]])
        ax.plot(polygon[:,0], polygon[:,1], 'k-', lw=0.8, alpha=0.7)
        ax.fill(polygon[:,0], polygon[:,1], facecolor='C0', alpha=0.25, edgecolor=None)

    # Plot the vertices
    ax.scatter(vertices[:,0], vertices[:,1], color='red', s=12, label='Vertices', zorder=5)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Triangular Mesh")
    ax.autoscale()
    ax.legend()
    ax.grid(True, alpha=0.4)
    return ax

def interp_heights(height_table, height_vector):
    """
    Compute interval indices and linear-interpolation weights for each
    element of `height_vector` against the monotonically increasing grid
    `height_table`.

    Parameters
    ----------
    height_table : (N,) ndarray
        Strictly increasing 1-D array of grid heights.
    height_vector : (M,) ndarray
        Arbitrary 1-D array of query heights.

    Returns
    -------
    idx : (M,) ndarray of int
        For each query height h, idx[k] = i such that
        height_table[i] <= h <= height_table[i+1].
        Values outside the table are clamped to the boundary intervals.
    w : (M, 2) ndarray of float
        Linear-interpolation weights. For a function f sampled on
        height_table, the interpolated value at height_vector[k] is
            f_interp = w[k,0] * f[idx[k]] + w[k,1] * f[idx[k]+1]
        Weights sum to 1 inside the table and are extrapolated linearly
        outside (so one weight may be < 0 or > 1).
    """
    height_table  = np.asarray(height_table,  dtype=float)
    height_vector = np.asarray(height_vector, dtype=float)

    n = height_table.size
    if n < 2:
        raise ValueError("height_table must have at least 2 entries.")

    # searchsorted gives, for each h, the insertion point in the table.
    # Subtract 1 to get the left edge of the containing interval, then
    # clamp so idx is always a valid left index in [0, n-2].
    idx = np.searchsorted(height_table, height_vector, side='right') - 1
    idx = np.clip(idx, 0, n - 2)

    h0 = height_table[idx]
    h1 = height_table[idx + 1]
    dh = h1 - h0                       # all > 0 because table is strictly increasing

    t = (height_vector - h0) / dh      # fractional position in interval
    w = np.empty((height_vector.size, 2), dtype=float)
    w[:, 0] = 1.0 - t                  # weight on f[idx]
    w[:, 1] = t                        # weight on f[idx+1]

    return idx, w

def find_containing_triangles(
    query_latlon,
    geolocation,
    mesh,
    degrees: bool = True,
    k: int = 8,
    max_k: int | None = None,
    eps: float = 1e-10,
    return_bary: bool = False,
):
    """Locate each query point in the spherical triangular mesh.

    Parameters
    ----------
    query_latlon : (M, 2) array_like
        Query points as [lat, lon].
    geolocation : (N, 2) array_like
        Mesh vertices as [lat, lon].
    mesh : (T, 3) array_like of int
        Vertex indices of the three corners of each triangle.
    degrees : bool, default True
        Whether the lat/lon inputs are in degrees.  If False, radians.
    k : int, default 8
        Initial number of nearest-centroid candidates to test per query.
    max_k : int, optional
        Maximum `k` to try if some queries remain unresolved.  Defaults to
        the number of triangles (i.e. brute force as a last resort).
    eps : float, default 1e-10
        Tolerance for the sign test (a point exactly on an edge has a
        scalar triple product equal to zero).
    return_bary : bool, default False
        If True, also return planar barycentric weights for each resolved
        query point.

    Returns
    -------
    triangle_idx : (M,) ndarray of int
        Index into `mesh` of the containing triangle, or -1 if no
        triangle contained the point.
    bary : (M, 3) ndarray, optional
        Planar barycentric weights w_A, w_B, w_C such that
        ``w_A * A + w_B * B + w_C * C`` is the projection of the query
        point onto the plane of triangle (A, B, C).  Only returned when
        ``return_bary=True``.
    """
    query_latlon = np.asarray(query_latlon, dtype=float).reshape(-1, 2)
    geolocation = np.asarray(geolocation, dtype=float).reshape(-1, 2)
    mesh = np.asarray(mesh, dtype=np.int64).reshape(-1, 3)

    n_tri = mesh.shape[0]
    if max_k is None:
        max_k = n_tri

    # 3D unit vectors
    V = _geodetic_to_ecef(geolocation[:, 0], geolocation[:, 1], degrees=degrees)  # (N, 3)
    Q = _geodetic_to_ecef(query_latlon[:, 0], query_latlon[:, 1], degrees=degrees)  # (M, 3)
    M = Q.shape[0]

    # Triangle corners
    A = V[mesh[:, 0]]
    B = V[mesh[:, 1]]
    C = V[mesh[:, 2]]

    # Per-triangle orientation reference (positive for CCW, negative for CW
    # as seen from outside the sphere).  Used to disambiguate the triangle
    # from its antipode in the containment test.
    orient = np.einsum("ij,ij->i", np.cross(A, B), C)  # (T,)

    # Triangle centroids, projected onto the unit sphere
    centroids = (A + B + C) / 3.0
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    tree = cKDTree(centroids)

    triangle_idx = np.full(M, -1, dtype=np.int64)

    current_k = min(k, n_tri)
    while True:
        unresolved = triangle_idx == -1
        if not np.any(unresolved):
            break

        # k nearest candidate triangles for the still-unresolved queries
        _, cand = tree.query(Q[unresolved], k=current_k)
        if current_k == 1:
            cand = cand[:, None]  # cKDTree drops the trailing axis when k=1

        # Test candidates column by column.  Most points resolve in the
        # first one or two columns, so this is cheap.
        unresolved_idx = np.where(unresolved)[0]
        for col in range(cand.shape[1]):
            still = triangle_idx[unresolved_idx] == -1
            if not np.any(still):
                break
            sub_idx = unresolved_idx[still]
            tris = cand[still, col]

            a = A[tris]
            b = B[tris]
            c = C[tris]
            p = Q[sub_idx]
            r = orient[tris]

            # Each scalar triple product, multiplied by the triangle's
            # orientation reference, must be non-negative for the point
            # to be on the "inside" side of every edge.
            s1 = np.einsum("ij,ij->i", np.cross(a, b), p) * r
            s2 = np.einsum("ij,ij->i", np.cross(b, c), p) * r
            s3 = np.einsum("ij,ij->i", np.cross(c, a), p) * r

            inside = (s1 >= -eps) & (s2 >= -eps) & (s3 >= -eps)
            triangle_idx[sub_idx[inside]] = tris[inside]

        # If anything is still unresolved, widen the search.
        if np.any(triangle_idx == -1) and current_k < max_k:
            current_k = min(current_k * 2, max_k)
        else:
            break

    if not return_bary:
        return triangle_idx

    # Planar barycentric weights for the resolved queries
    bary = np.zeros((M, 3))
    found = triangle_idx != -1
    if np.any(found):
        tris = triangle_idx[found]
        a = A[tris]
        b = B[tris]
        c = C[tris]
        p = Q[found]
        # Project p onto the triangle's plane and solve a 2x2 system in
        # barycentric form.  This is the standard planar barycentric
        # computation; it gives sensible interpolation weights for the
        # small spherical triangles found in geophysical meshes.
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denom = d00 * d11 - d01 * d01
        wB = (d11 * d20 - d01 * d21) / denom
        wC = (d00 * d21 - d01 * d20) / denom
        wA = 1.0 - wB - wC
        bary[found, 0] = wA
        bary[found, 1] = wB
        bary[found, 2] = wC

    return triangle_idx, bary


class EDPSamples(xr.Dataset):
    """
    Specialized ``xarray.Dataset`` for EDP samples.

    **Coordinates / variables**

    - ``altitude``: 1D along ``height``.
    - ``geolocation``: (geo, 2) latitude and longitude per surface/track point.
    - ``sampling_parameters``: (sample, param) with named parameters
      (e.g. f107, f107_81, ig, ir).
    - ``EDPs``: (height, geo, sample) main field.
    - ``Mesh``: optional (triangle, 3) int indices into ``geo`` for triangle vertices.

    Use :meth:`from_arrays` to construct. Geolocation grids can be built with
    :meth:`genRectangularArea`, :meth:`genPolarArea`, or :meth:`genLineOfSight`.
    Fill ``EDPs`` with :meth:`IRI2020_EDP`, which calls a Fortran driver hook
    (see ``framework/iri2020_driver.py``). Persist with :meth:`saveNetCDF` /
    :meth:`fromNetCDF`.
    """
    __slots__ = ()
    DIM_HEIGHT = "height"
    DIM_GEO = "geo_vertex"
    DIM_SAMPLE = "sample"
    DIM_TRIANGLE = "mesh_triangle"
    DIM_PARAM = "param"
    DIM_GEO_COMPONENT = "lat_lon"
    DIM_VERTEX = "vertex"

    COORD_ALTITUDE = "altitude"
    VAR_EDPS = "EDPs"
    VAR_GEOLOCATION = "geolocation"
    VAR_SAMPLING = "sampling_parameters"
    VAR_MESH = "Mesh"

    @staticmethod
    def genRectangularArea(
        minLon: float,
        maxLon: float,
        dLon: float,
        minLat: float,
        maxLat: float,
        dLat: float
    ) -> tuple[np.ndarray,np.ndarray]:
        """
        Evenly spaced latitude–longitude grid over
        ``[minLat, maxLat] × [minLon, maxLon]``.

        Parameters
        ----------
        minLon, maxLon, dLon, minLat, maxLat, dLat  : float
            Bounds in degrees (latitude north, longitude east).

        Returns
        -------
        vertice : Columns are longitude and latitude.
        triangles : Column are indices for vertice of the triangle patches.
        """
        # Make sure the number of grid points in Latitude and 
        # Longitude are odd integers.
        nLat=2*int((maxLat-minLat)/(2*dLat)) + 1
        nLon=2*int((maxLon-minLon)/(2*dLon)) + 1
        #   Create grid of points
        lat_vals = minLat+(maxLat-minLat)*np.arange(nLat )/(nLat-1)
        lon_vals = minLon+(maxLon-minLon)*np.arange(nLon)/(nLon-1)
        vertices = np.column_stack((lon_vals[0]*np.ones(len(lat_vals[::2])),lat_vals[::2]))
        triangles = []
        # The index of the previous column of the vertices.
        TopLeftColumn=0
        for i in range(0,nLon-1,2):
            TopCurrentColumn=TopLeftColumn+len(lat_vals[::2])
            vertices = np.vstack((vertices,np.column_stack((lon_vals[i+1]*np.ones(len(lat_vals[1::2])),lat_vals[1::2]))))
            TopRightColumn=TopCurrentColumn+len(lat_vals[1::2])
            vertices = np.vstack((vertices,np.column_stack((lon_vals[i+2]*np.ones(len(lat_vals[::2])),lat_vals[::2]))))
            for j in range(len(lat_vals[::2]) - 1):
                # Left triangle (v0, v1, v2)
                v0 = TopLeftColumn + j
                v1 = TopLeftColumn + j+ 1
                v2 = TopCurrentColumn + j
                triangles.append([v0, v1, v2])
                # Top triangle (v0, v1, v2)
                v0 = TopLeftColumn + j
                v1 = TopRightColumn + j
                v2 = TopCurrentColumn + j
                triangles.append([v0, v1, v2])
                # Right triangle (v1, v3, v2)
                v0 = TopRightColumn + j+ 1
                v1 = TopCurrentColumn + j
                v2 = TopRightColumn + j
                triangles.append([v0, v1, v2])
                v0 = TopRightColumn + j+ 1
                v1 = TopCurrentColumn + j
                v2 = TopLeftColumn + j+1
                triangles.append([v0, v1, v2])
                TopLeftColumn=TopRightColumn

        triangles = np.array(triangles, dtype=int)
        return vertices, triangles

    @staticmethod
    def genPolarArea(
        pole: Literal["north", "south"],
        minLat: float,
        dLat: float,
        ) -> tuple[np.ndarray,np.ndarray]:
        """
        Evenly spaced grid on a polar cap.

        **North:** latitudes from ``minLat`` to 90°N (``minLat < 90``).

        **South:** latitudes from -90°S to ``-minLat`` with **positive** ``minLat``
        (equatorward edge of the cap, e.g. 66.5° for Antarctic Circle).

        vertice : Columns are longitude and latitude.
        triangles : Column are indices for vertice of the triangle patches.
        """
        # Make sure the number of grid points in Latitude and 
        # Longitude are odd integers.
        if pole == "north":
            nLat=int((90-minLat)/dLat) + 1
            lat_vals = minLat+(90-minLat)*np.arange(nLat )/(nLat-1)        
        elif pole == "south":
            nLat=int((minLat+90)/dLat) + 1
            lat_vals = minLat-(minLat+90)*np.arange(nLat )/(nLat-1)        
        else:
            raise ValueError(f"Invalid pole: {pole}")

        dLon=60
        lon_vals=np.arange(0, 360, dLon)
        vertices = np.column_stack((np.array(0),lat_vals[-1]))
        vertices = np.vstack((vertices,
                np.column_stack((lon_vals,lat_vals[-2]*np.ones(len(lon_vals))))))
        triangles = []
        #print(vertices)
        # Triangles with poles as a vertex.
        for i in range(len(lon_vals)):
            v0 = 0
            v1 = i+1
            if i == len(lon_vals)-1:
                v2=1
            else:
                v2 = i+2
            triangles.append([v0, v1, v2])
            #print(triangles)

        StartPreviousRing=1
        dArc_Previous=0.0
        for j in range(len(lat_vals)-3,-1,-1):
            dArc_Current=np.cos(lat_vals[j]*math.pi/180.)*dLon*math.pi/180.
            if dArc_Current>1.1*dArc_Previous:
                # Longitude grid doubles with each decrease in latitude
                dArc_Previous=dArc_Current
                dLon=dLon/2
                StartCurrentRing=StartPreviousRing+len(lon_vals)
                lon_vals=np.arange(0, 360, dLon)
                vertices = np.vstack((vertices,
                    np.column_stack((lon_vals,lat_vals[j]*np.ones(len(lon_vals))))))
                #print(vertices)
                for i in range(0,len(lon_vals),2):
                    ip=int(i/2)
                    v0 = StartCurrentRing+i
                    v1 = StartPreviousRing+ip
                    v2 = StartCurrentRing+i+1
                    triangles.append([v0, v1, v2])
                    v0 = StartPreviousRing+ip
                    v1 = StartCurrentRing+i+1
                    if i==len(lon_vals)-2:
                        v2 = StartPreviousRing
                    else:
                        v2 = StartPreviousRing+ip+1
                    triangles.append([v0, v1, v2])
                    v0 = StartCurrentRing+i+1
                    if i == len(lon_vals)-2:
                        v1 = StartPreviousRing
                        v2 = StartCurrentRing
                    else:
                        v1 = StartPreviousRing+ip+1
                        v2 = StartCurrentRing+i+2
                    triangles.append([v0, v1, v2])
            else:
                # Longitude grid stays constant
                StartCurrentRing=StartPreviousRing+len(lon_vals)
                vertices = np.vstack((vertices,
                    np.column_stack((lon_vals,lat_vals[j]*np.ones(len(lon_vals))))))
                for i in range(0,len(lon_vals),2):
                    v0 = StartCurrentRing+i
                    v1 = StartPreviousRing+i
                    v2 = StartCurrentRing+i+1
                    triangles.append([v0, v1, v2])
                    v0 = StartPreviousRing+i
                    v1 = StartCurrentRing+i+1
                    v2 = StartPreviousRing+i+1
                    triangles.append([v0, v1, v2])
                    v0 = StartPreviousRing+i+1
                    v1 = StartCurrentRing+i+1
                    if i==len(lon_vals)-2:
                        v2 = StartPreviousRing
                    else:
                        v2 = StartPreviousRing+i+2
                    triangles.append([v0, v1, v2])
                    v0 = StartCurrentRing+i+1
                    if i == len(lon_vals)-2:
                        v1 = StartPreviousRing
                        v2 = StartCurrentRing
                    else:
                        v1 = StartPreviousRing+i+2
                        v2 = StartCurrentRing+i+2
                    triangles.append([v0, v1, v2])
            
            #print(triangles)
            StartPreviousRing=StartCurrentRing

        return vertices, triangles
    
    @staticmethod
    def genLineOfSight(
        lat1: float,
        lon1: float,
        alt1_m: float,
        lat2: float,
        lon2: float,
        alt2_m: float,
        num_points: int,
    ) -> tuple[np.ndarray,np.ndarray]:
        """
        Geolocations along the straight **ECEF** chord between two satellites
        (WGS84 geodetic inputs: degrees, degrees, metres above ellipsoid).

        Parameters
        ----------
        lat1, lon1, alt1_m
            First satellite.
        lat2, lon2, alt2_m
            Second satellite.
        num_points : int
            Samples along the segment, **including both endpoints** (>= 2).

        Returns
        -------
        ndarray, vertice (num_points, 2)
            Latitude and longitude in degrees. Height along the path is not stored.
        ndarray, segmenta (num_segment,2)
        """
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        r0 = np.asarray(_geodetic_to_ecef(lat1, lon1, alt1_m), dtype=float).reshape(3)
        r1 = np.asarray(_geodetic_to_ecef(lat2, lon2, alt2_m), dtype=float).reshape(3)
        t = np.linspace(0.0, 1.0, num_points, dtype=float)[:, np.newaxis]
        chord = (1.0 - t) * r0 + t * r1
        lat_deg, lon_deg, _h = _ecef_to_geodetic(chord)
        vertice = np.column_stack([np.asarray(lat_deg).ravel(), np.asarray(lon_deg).ravel()]
        ).astype(float)
        segment = np.column_stack((np.arange(num_points-1),np.arange(1,num_points)))
        return vertice, segment

    def __init__(self,
        DateTime: str,
        geo_type : Literal["Point","LOS","Rectangle","Polar","Occultation"],
        altitude: np.ndarray,
        sampling_parameters: pd.Dataframe,
        evaluate_iri: int = None,
        minLon: float =None,
        maxLon: float =None,
        dLon: float =None,
        minLat: float =None,
        maxLat: float =None,
        dLat: float =None,
        LOS_LEO: np.ndarray= None,
        LOS_GNSS: np.ndarray = None,
        LOS_nb_point: int = None,
        Lon: float = None,
        Lat: float = None,
        filename: str = None,
        pt1: tuple = None,
        pt2: tuple = None,
        pt3: tuple = None,
        alt_limit: float = 600.0,
        edps: np.ndarray = None,
        attrs = None):
        #) -> EDPSamples:
        """
        Build an ``EDPSamples`` dataset.

        Parameters
        ----------
        geo_type : Literal["Point","LOS","Rectangle","Polar"], type of 2D grid
        altitude : (n_height,) array
            1D altitude (or height coordinate) for each index along ``height``.
        sampling_parameters: pd.Dataframe,
        evaluate_iri: int flag. =1 if IRI2020 is to be evaluated.
        minLon: float =None, minimum longitude for rectangular lat, lon grid.
        maxLon: float =None, maximum longitude for rectangular lat, lon grid.
        dLon: float =None, logitudinal step for rectangular lat, lon grid.
        minLat: float =None, minimum latitude for rectangular lat, lon grid. Also
                for polar grid.
        maxLat: float =None, maximum latitude for rectangular lat, lon grid.
        dLat: float =None, Latitude step for rectangular and prolar grid.
        LOS_LEO: np.ndarray= None, vector of 3 float values for the ECEF or lla
                   coordinates of LEO end of LOS.
        LOS_GNSS: np.ndarray = None, vector of 3 float values for the ECEF or lla
                   coordinates of GNSS end of LOS.
        LOS_nb_point: int = None, number of intermediate points along the ground
                   track of LOS. These 3 parameters define the EDP samples along
                   LOS ground track.
        Lon: float = None, Longitude of single control point.
        Lat: float = None, Latitude of single control point.
        edps: numpy array.
        attrs : mapping, optional Dataset attributes.
        
        """
        sample_param_value=sampling_parameters.to_numpy()
        sample_Param_name=sampling_parameters.columns
        n_sample = sampling_parameters.shape[0]
        
        altitude = np.asarray(altitude)
        match geo_type:
            case "Point":
                if Lon == None or Lat == None:
                    raise ValueError("For geo_type point, Lon and Lat cannot be None.")
                geolocation =np.array([[Lon,Lat]])
                mesh = None
            case "LOS":
                if LOS_LEO == None or LOS_GNSS == None or LOS_nb_point == None:
                    raise ValueError("For geo_type LOS, LOS_start, LOS_end and LOS_nb_point cannot be None.")
                geolocation, mesh = self.genLineOfSight(LOS_LEO,LOS_GNSS,LOS_nb_point)
            case "Rectangle":
                if minLon == None or maxLon == None or dLon == None or minLat == None or maxLat == None or dLat == None:
                    raise ValueError("For geo_type Rectangle, minLon, maxLon, dLon,minLat,maxLat, and dLat cannot be None.")
                geolocation, mesh = self.genRectangularArea(minLon, maxLon, dLon,minLat,maxLat, dLat)
            case "Polar":
                if minLat == None or dLat == None :
                    raise ValueError("For geo_type minLat, and dLat cannot be None.")
                if minLat > 0:
                    pole="north"
                else:
                    pole="south"
                geolocation, mesh =self.genPolarArea(pole,minLat,dLat)
            case "Occultation":
                if filename is None and (pt1 is None or pt2 is None or pt3 is None):
                    raise ValueError("For geo_type Occultation, you must provide either a 'filename' or all three points ('pt1', 'pt2', 'pt3').")
                if dLat is None or dLon is None:
                    print("For geo_type Occultation, dLat and dLon assumed to be 5 deg.")
                    geolocation, mesh, pt1, pt2, pt3 = generate_occultation_mesh(
                        pt1=pt1, pt2=pt2, pt3=pt3, 
                        filename=filename, 
                        dLat=5, dLon=5, 
                        alt_limit=alt_limit
                    )
                else:
                # vertices = geolocation (lon, lat array), triangles = mesh (indices array)
                    geolocation, mesh, pt1, pt2, pt3 = generate_occultation_mesh(
                        pt1=pt1, pt2=pt2, pt3=pt3, 
                        filename=filename, 
                        dLat=dLat, dLon=dLon, 
                        alt_limit=alt_limit
                    )
            case _:
                raise ValueError(f"Invalid geo_type: {geo_type}")

            
        if altitude.ndim != 1:
            raise ValueError("altitude must be 1-D")
        n_height = altitude.shape[0]

        if geolocation.ndim != 2 or geolocation.shape[1] != 2:
            raise ValueError("geolocation must have shape (n_geo, 2) for lat, lon")
        n_geo = geolocation.shape[0]
        
        if edps.all() == None :
            if evaluate_iri == 1:
                edps=self.get_IRI2020_EDP(DateTime,altitude,geolocation,sampling_parameters)
                
            else:
                edps=np.ndarray((n_height,n_geo,n_sample))
        else:
            print(f"EDPS: {edps.shape[1]}, N_geo: {n_geo}")
            assert edps.ndim == 3, "EDP sample profile must be 3 dimensional"
            assert edps.shape[0] == n_height, "EDP samples'eading dimension must be height"
            assert edps.shape[1] == n_geo, "EDP sample second dimension must be vertices"
            assert edps.shape[2] == n_sample, "EDP sample last dimension must be param sample"
            

        coords: MutableMapping[str, Any] = {
            self.COORD_ALTITUDE: (self.DIM_HEIGHT, altitude),
            self.DIM_GEO_COMPONENT: (
                self.DIM_GEO_COMPONENT,
                np.array(["latitude", "longitude"], dtype=object)),
            self.DIM_PARAM: (
                self.DIM_PARAM, sample_Param_name    
            ),
            #self.DIM_PARAM: (self.DIM_PARAM, sampling_parameters),
        }
        #
        # Define variables
        #
        data_vars: MutableMapping[str, Any] = {
            self.VAR_GEOLOCATION: (
                (self.DIM_GEO, self.DIM_GEO_COMPONENT),
                geolocation,
                {
                    "long_name": "geolocation",
                    "description": "latitude (column 0), longitude (column 1)",
                },
            ),
            self.VAR_EDPS: (
                (self.DIM_HEIGHT, self.DIM_GEO, self.DIM_SAMPLE),
                edps,
                {"long_name": "EDPs",
                 "description": "electron density profiles samples"}
                ),
            self.VAR_SAMPLING: (
                (self.DIM_SAMPLE,self.DIM_PARAM),
                sample_param_value,
                {"long name": "sample_input_parameter",
                 "description": "sample inputs for IRI2020 model"}
            ),
        }
        if mesh is not None:
            mesh = np.asarray(mesh, dtype=np.int64)
            if mesh.ndim != 2 or mesh.shape[1] != 3:
                raise ValueError("mesh must have shape (n_triangle, 3)")
            if np.any(mesh < 0) or np.any(mesh >= n_geo):
                raise ValueError("mesh indices must be in [0, n_geo)")
            
            data_vars[self.VAR_MESH] = (
                (self.DIM_TRIANGLE, self.DIM_VERTEX),
                mesh,
                {
                    "long_name": "surface mesh",
                    "description": 
                        "triangles: three vertex indices into the geo dimension"
                },
            )
        if attrs == None:
            attrs={}
        attrs['DateTime'] = DateTime
        attrs['geo_type'] = geo_type
            
        match geo_type:
            case "Point":
               attrs["Lon"] = Lon
               attrs["Lat"] = Lat
            case "LOS":
                attrs["LOS_LEO"] = LOS_LEO
                attrs["LOS_GNSS"] = LOS_GNSS
                attrs["LOS_nb_point"]= LOS_nb_point
            case "Rectangle":
                attrs["minLon"] = minLon
                attrs["maxLon"] = maxLon
                attrs["dLon"] = dLon,
                attrs["minLat"] = minLat
                attrs["maxLat"] = maxLat
                attrs["dLat"] = dLat
            case "Polar":
                attrs["minLat"] = minLat
                attrs["dLat"] = dLat
                if minLat > 0:
                    attrs["pole"] = "north"
                else:
                    attrs["pole"] = "south"
                    
        super().__init__(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs)
        

    @classmethod
    def fromNetCDF(cls, path: str | Path, **kwargs: Any) -> EDPSamples:
        """
        Load an ``EDPSamples`` from NetCDF (e.g. written by :meth:`saveNetCDF`).

        The file is read fully into memory, then closed.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``.nc`` file.
        **kwargs
            Forwarded to :func:`xarray.open_dataset` (e.g. ``engine``, ``group``,
            ``decode_times``, ``mask_and_scale``).

        Returns
        -------
        EDPSamples
        """
        with xr.open_dataset(path, **kwargs) as ds:
            ds.load()
            EDPSam = EDPSamples.from_xarray(ds)   
            return EDPSam
        
    @classmethod
    def from_xarray(cls, ds) -> EDPSamples:
        assert "DateTime" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (0)."]
        assert "geo_type" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (1)."]

        assert cls.VAR_SAMPLING in ds.data_vars, ["Missing sampling_parameters in loaded dataset."]
        sampling_parameters_xr=ds.data_vars["sampling_parameters"]
        sampling_parameters=sampling_parameters_xr.to_dataframe().unstack()
        sampling_parameters=sampling_parameters.reset_index()
        sampling_parameters.columns=sampling_parameters.columns.droplevel(0)
        sampling_parameters=sampling_parameters.drop(sampling_parameters.columns[0],axis=1)

        assert cls.COORD_ALTITUDE in ds.coords, ["Missing altitude in loaded dataset."]
        altitude_xr = ds.coords["altitude"]
        altitude = altitude_xr.to_numpy()

        assert cls.VAR_EDPS in ds.data_vars, ["Missing EDPs in loaded dataset."]

        epds_xr = ds.data_vars[cls.VAR_EDPS]
        edps = epds_xr.to_numpy()
        
        match ds.attrs["geo_type"]:
            case "Point":
               assert "Lon" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (2)."]
               assert "Lat" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (3)."]
               return EDPSamples(ds.attrs["DateTime"],ds.attrs["geo_type"],
                                altitude, sampling_parameters, 
                                Lon=ds.attrs["Lon"],Lat=ds.attrs["Lon"],edps=edps,attrs=ds.attrs)
            case "LOS":
                assert "LOS_LEO" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (4)."]
                assert "LOS_GNSS" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (5)."]
                assert "LOS_nb_point" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (6)."]
                return EDPSamples(ds.attrs["DateTime"],ds.attrs["geo_type"],
                                altitude, sampling_parameters, 
                                LOS_LEO=ds.attrs["LOS_LEO"],LOS_GNSS=ds.attrs["LOS_GNSS"],
                                LOS_nb_point=ds.attrs["LOS_nb_point"],
                                edps=edps,attrs=ds.attrs)
            case "Rectangle":
                assert "minLon" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (7)."]
                assert "maxLon" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (8)."]
                assert "dLon" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (9)."]
                assert "minLat" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (10)."]
                assert "maxLat" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (11)."]
                assert "dLat" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (12)."]
                return EDPSamples(ds.attrs["DateTime"],ds.attrs["geo_type"],
                                altitude, sampling_parameters, 
                                minLon=ds.attrs["minLon"],maxLon=ds.attrs["maxLon"],
                                dLon=ds.attrs["dLon"],
                                minLat=ds.attrs["minLat"],maxLat=ds.attrs["maxLat"],
                                dLat=ds.attrs["dLat"],
                                edps=edps,attrs=ds.attrs)
            case "Polar":
                assert "minLat" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (13)."]
                assert "dLat" in ds.attrs, ["Loaded dataset does not have the structure of EDPSamples (14)."]
                return EDPSamples(ds.attrs["DateTime"],ds.attrs["geo_type"],
                                altitude, sampling_parameters, 
                                minLat=ds.attrs["minLat"],dLat=ds.attrs["dLat"],
                                edps=edps,attrs=ds.attrs)
        
        
    def saveNetCDF(self, path: str | Path, **kwargs: Any) -> None:
        """
        Write this dataset to a NetCDF file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path (typically ``.nc``).
        **kwargs
            Forwarded to :meth:`xarray.Dataset.to_netcdf` (e.g. ``mode``, ``format``,
            ``engine``, ``encoding``).
        """
        self.to_netcdf(path, **kwargs)

    def plot_geolocation(self):
        match self.attrs["geo_type"]:
            case "Point" | "LOS":
                fig, ax = plt.subplots(figsize=(8, 6))
                # Plot the vertices
                ax.scatter(self.geolocation[:,0], self.geolocation[:,1], color='red', s=12, label='Vertices', zorder=5)
                ax.set_xlabel("Longitude (deg)")
                ax.set_ylabel("Latitude (deg)")
                ax.autoscale()
                ax.legend()
                ax.grid(True, alpha=0.4)
                
            case "Rectangle":
                plot_tri_mesh(self.geolocation, self.mesh)                
            case "Polar":
                if self.attrs["minLat"] > 0:
                    plot_polar_mesh(self.geolocation, self.mesh, pole= "north")  
                else:
                    plot_polar_mesh(self.geolocation, self.mesh, pole= "south")  

    def interp(self, positions: np.array, 
               coordinate: Literal["lla","ecef"] = "lla") -> tuple[np.ndarray,  np.ndarray, np.ndarray,  np.ndarray]:
        if coordinate == "ecef":
            latitude,longitude, altitude = _ecef_to_geodetic(positions)
        else:
            latitude = positions[:,0]
            longitude = positions[:,1]
            altitude = positions[:,2]
            positions = _geodetic_to_ecef(latitude,longitude, altitude)
            
        idx_alt,weight_alt = interp_heights(self.altitude, altitude)
        
        match self.attrs["geo_type"]:
            case "Point": 
                return idx_alt, weight_alt
            case "LOS":
                # Need horizontal interpolation
                return idx_alt, weight_alt
            case "Rectangle" | "Polar":
                idx_mesh, weight_mesh = find_containing_triangles(
                    np.column_stack([latitude,longitude]),
                    self.geolocation,
                    self.mesh,
                    return_bary = True,
                    )
                return idx_alt, weight_alt, idx_mesh, weight_mesh
       
        
    @property
    def altitude(self) -> xr.DataArray:
        """1D altitude coordinate along ``height``."""
        return self.coords[self.COORD_ALTITUDE]

    @property
    def geolocation(self) -> xr.DataArray:
        """(geo, 2): latitude and longitude per point."""
        return self[self.VAR_GEOLOCATION]

    @property
    def sampling_parameters(self) -> xr.DataArray:
        """(sample, param); ``param`` labels columns (e.g. f107, ig)."""
        return self[self.VAR_SAMPLING]

    @property
    def edps(self) -> xr.DataArray:
        """Main field (height, geo, sample)."""
        return self[self.VAR_EDPS]

    @property
    def mesh(self) -> xr.DataArray | None:
        """(triangle, 3) vertex indices into ``geo``, or ``None`` if absent."""
        if self.VAR_MESH in self.data_vars:
            return self[self.VAR_MESH]
        return None

