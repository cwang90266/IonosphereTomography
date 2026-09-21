#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .roi_tools import circular_roi_points, geodesic_circle_latlon, DEFAULT_FIBONACCI_SPACING_DEG, DEFAULT_FIBONACCI_SPACING_KM

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--center-lat',type=float,default=69.6)
    ap.add_argument('--center-lon',type=float,default=19.2)
    ap.add_argument('--radius-km',type=float,default=2000.0)
    ap.add_argument('--output',type=Path,default=Path('igs_roi_voxels.png'))
    a=ap.parse_args()
    glat,glon=circular_roi_points(a.center_lat,a.center_lon,a.radius_km,spacing_deg=DEFAULT_FIBONACCI_SPACING_DEG)
    rlat,rlon=geodesic_circle_latlon(a.center_lat,a.center_lon,a.radius_km)
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(111,projection=ccrs.Orthographic(a.center_lon,a.center_lat))
    ax.set_global(); ax.add_feature(cfeature.LAND,facecolor='0.88'); ax.add_feature(cfeature.OCEAN,facecolor='white')
    ax.add_feature(cfeature.COASTLINE,linewidth=.65); ax.add_feature(cfeature.BORDERS,linewidth=.35,alpha=.55); ax.gridlines(linewidth=.35,alpha=.4)
    ax.scatter(glon,glat,s=30,c='k',transform=ccrs.PlateCarree(),zorder=3,label=f'IGS voxel columns ({DEFAULT_FIBONACCI_SPACING_DEG:.0f}° ≈ {DEFAULT_FIBONACCI_SPACING_KM:.0f} km)')
    ax.plot(rlon,rlat,color='green',lw=2,transform=ccrs.Geodetic(),zorder=2,label=f'ROI = {a.radius_km:.0f} km')
    ax.scatter([a.center_lon],[a.center_lat],marker='*',s=140,c='tab:blue',transform=ccrs.PlateCarree(),zorder=4,label='ROI / IGS reference center')
    ax.set_title('IGS common ROI and voxel layout')
    ax.legend(loc='lower left',fontsize=9)
    fig.tight_layout(); fig.savefig(a.output,dpi=170,bbox_inches='tight'); plt.close(fig)
    print(f'grid points: {len(glat)}')
    print(f'spacing: {DEFAULT_FIBONACCI_SPACING_DEG:.1f} deg = {DEFAULT_FIBONACCI_SPACING_KM:.1f} km')
    print(f'ROI radius: {a.radius_km:.1f} km')
    print(f'saved: {a.output}')
if __name__=='__main__': main()
