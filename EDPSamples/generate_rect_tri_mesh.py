import numpy as np

def generate_rect_tri_mesh(minLat, maxLat, dLat, minLon, maxLon, dLon):
    """
    Generates a rectangular grid mesh of vertices (lat, lon) and corresponding
    triangle indices connecting the mesh as patches.

    Parameters
    ----------
    minLat, maxLat, dLat: float
        Latitude bounds and step size in degrees.
    minLon, maxLon, dLon: float
        Longitude bounds and step size in degrees.

    Returns
    -------
    vertices : np.ndarray, shape (N, 2)
        Array of vertex positions (lon, lat).
    triangles : np.ndarray, shape (M, 3)
        Array of triangle indices, each referencing 3 vertices.
    """
    # Make sure the number of grid points in Latitude and 
    # Longitude are odd integers.
    nLat=2*int((maxLat-minLat)/(2*dLat)) + 1
    nLon=2*int((maxLon-minLon)/(2*dLon)) + 1
    # Create grid of points
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