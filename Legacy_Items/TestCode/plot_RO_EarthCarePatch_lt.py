import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Specify the path to your csv file
def plot_RO_EathCarePatch_lt(csv_file_path):
    # Read the csv file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    patch_corner_names = [
        ('ec_frame_start_lon', 'ec_frame_start_lat'),
        ('ec_frame_start_lon', 'ec_frame_end_lat'),
        ('ec_frame_end_lon', 'ec_frame_end_lat'),
        ('ec_frame_end_lon', 'ec_frame_start_lat')
    ]

    # Create a 2D map plot
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Local Times vs Latitude of RO Data (Top) and EarthCare Frames (Bottom)")
    axs = fig.subplots(2, 1)
    # Find local time for RO data
    co_hour_vals=pd.to_datetime(df['ro_col_time']).dt.hour
    ltime_co=np.array(co_hour_vals+df['ro_col_lon']/15)
    idx=np.where(ltime_co<0)
    ltime_co[idx]=ltime_co[idx]+24
    idx=np.where(ltime_co>24)
    ltime_co[idx]=ltime_co[idx]-24
    # Find local time for earthCare frame
    fr_hour_vals=pd.to_datetime(df['ec_frame_start_time']).dt.hour
    fr_ltime_start=np.array(fr_hour_vals+df['ec_frame_start_lon']/15)
    idx=np.where(fr_ltime_start<0)
    fr_ltime_start[idx]=fr_ltime_start[idx]+24
    idx=np.where(fr_ltime_start>24)
    fr_ltime_start[idx]=fr_ltime_start[idx]-24

    fr_hour_vals=pd.to_datetime(df['ec_frame_start_time']).dt.hour
    fr_ltime_end=np.array(fr_hour_vals+df['ec_frame_end_lon']/15)
    idx=np.where(fr_ltime_end<0)
    fr_ltime_end[idx]=fr_ltime_end[idx]+24
    idx=np.where(fr_ltime_end>24)
    fr_ltime_end[idx]=fr_ltime_end[idx]-24

    # Normalize time for colormap
    lt_min=min(np.nanmin(ltime_co),np.nanmin(fr_ltime_start),np.nanmin(fr_ltime_end))
    lt_max=max(np.nanmax(ltime_co),np.nanmax(fr_ltime_start),np.nanmax(fr_ltime_end))
    norm = mpl.colors.Normalize(vmin=lt_min,vmax=lt_max)
    cmap = plt.get_cmap('viridis')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot RO observation points with color by time
    pn = axs[0]
    sc = pn.scatter(ltime_co, df['ro_col_lat'], c=ltime_co, cmap=cmap, norm=norm,
        label='RO Observations', s=50, marker='o', edgecolor='black')
    pn = axs[1]
    for idx, row in df.iterrows():
        if abs(fr_ltime_start[idx]-fr_ltime_end[idx])<12:
            lons = [fr_ltime_start[idx],fr_ltime_start[idx],fr_ltime_end[idx],fr_ltime_end[idx]]
            lats = [row[corner[1]] for corner in patch_corner_names]
            tc = fr_ltime_start[idx]
            patch_color = cmap(norm(tc))
            pn.fill(lons, lats, color=patch_color, alpha=0.4, label="EarthCare Patch" if idx == 0 else None)
        else:
            lons = [0,0,min(fr_ltime_end[idx],fr_ltime_start[idx]),
                        min(fr_ltime_end[idx],fr_ltime_start[idx])]
            lats = [row[corner[1]] for corner in patch_corner_names] 
            tc = 0.5*min(fr_ltime_end[idx],fr_ltime_start[idx])
            patch_color = cmap(norm(tc))
            pn.fill(lons, lats, color=patch_color, alpha=0.4, label="EarthCare Patch" if idx == 0 else None)
            lons = [24,24,max(fr_ltime_end[idx],fr_ltime_start[idx]),
                    max(fr_ltime_end[idx],fr_ltime_start[idx])]
            lats = [row[corner[1]] for corner in patch_corner_names]
            tc = 0.5*(24+max(fr_ltime_end[idx],fr_ltime_start[idx]))
            patch_color = cmap(norm(tc))
            pn.fill(lons, lats, color=patch_color, alpha=0.4, label="EarthCare Patch" if idx == 0 else None)

    plt.show()