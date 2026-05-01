import numpy as np
from typing import Literal
import math

# Polar grid is based on spheric model of the Earth.

def generate_ploar_mesh(
    pole: Literal["north","south"],
    minLat: float, 
    dLat: float):
    """
    Generates a polar grid mesh of vertices (lat, lon) and corresponding
    triangle indices connecting the mesh as patches.

    Parameters
    ----------
        pole: Literal["north", "south"],
        minLat: float,
    minLat, dLat: float
        Latitude bounds and step size in degrees.

    Returns
    -------
    vertices : np.ndarray, shape (N, 2)
        Array of vertex positions (lon, lat).
    triangles : np.ndarray, shape (M, 3)
        Array of triangle indices, each referencing 3 vertices.
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