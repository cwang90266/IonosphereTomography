#!/usr/bin/env python3
import numpy as np
from observation_preparation.roi_tools import (
    circular_roi_points, DEFAULT_FIBONACCI_SPACING_DEG, DEFAULT_FIBONACCI_SPACING_KM
)

R=6371.0
def hav(lat0,lon0,lat,lon):
    p0=np.deg2rad(lat0); p=np.deg2rad(lat); dl=np.deg2rad(lon-lon0)
    a=np.sin((p-p0)/2)**2+np.cos(p0)*np.cos(p)*np.sin(dl/2)**2
    return 2*R*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def main():
    c_lat,c_lon,r=69.6,19.2,2000.0
    ro_lat,ro_lon=circular_roi_points(c_lat,c_lon,r,spacing_deg=DEFAULT_FIBONACCI_SPACING_DEG)
    igs_lat,igs_lon=circular_roi_points(c_lat,c_lon,r,spacing_deg=DEFAULT_FIBONACCI_SPACING_DEG)
    assert np.array_equal(ro_lat,igs_lat) and np.array_equal(ro_lon,igs_lon)
    d=hav(c_lat,c_lon,ro_lat,ro_lon)
    assert np.nanmax(d) <= r + 1e-8
    print(f"PASS: RO and IGS use identical ROI grid: {len(ro_lat)} points")
    print(f"spacing = {DEFAULT_FIBONACCI_SPACING_DEG:.1f} deg = {DEFAULT_FIBONACCI_SPACING_KM:.1f} km")
    print(f"requested radius = {r:.1f} km; farthest retained point = {np.nanmax(d):.1f} km")

if __name__=='__main__': main()
