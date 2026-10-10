import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib as mpl

# Specify the path to your csv file
def preview_RO_EathCarePatch(csv_file_path):
    # Read the csv file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # The code assumes the following column structure:
    #   - 'RO_lon', 'RO_lat' : coordinates of the RO observation
    #   - 'patch_lon1', 'patch_lat1' ... 'patch_lon4', 'patch_lat4': corners of the rectangular patch (assumed in order, e.g., lower left, lower right, upper right, upper left)

    # If your patch corners have different column names, adjust accordingly
    patch_corner_names = [
        ('ec_frame_start_lon', 'ec_frame_start_lat'),
        ('ec_frame_start_lon', 'ec_frame_end_lat'),
        ('ec_frame_end_lon', 'ec_frame_end_lat'),
        ('ec_frame_end_lon', 'ec_frame_start_lat')
    ]

    # Create a 2D map plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Add features to map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor='azure', alpha=0.3)


    # Convert the time column 'ro_col_time' to a sequence of numbers for colormap indexing
    #if pd.api.types.is_datetime64_any_dtype(df['ro_col_time']):
    #    time_vals = df['ro_col_time']
    #else:
        # Try convert if not already in datetime (if it's str)
        #time_vals = pd.to_datetime(df['ro_col_time'])
    co_hour_vals=pd.to_datetime(df['ro_col_time']).dt.hour
    ltime_co=np.array(co_hour_vals+df['ro_col_lon']/15)
    idx=np.where(ltime_co<0)
    ltime_co[idx]=ltime_co[idx]+24
    idx=np.where(ltime_co>24)
    ltime_co[idx]=ltime_co[idx]-24
    
    #if pd.api.types.is_datetime64_any_dtype(df['ec_frame_start_time']):
    #    fr_time_vals = df['ec_frame_start_time']
    #else:
        # Try convert if not already in datetime (if it's str)
        #fr_time_vals = pd.to_datetime(df['ec_frame_start_time'])
    fr_hour_vals=pd.to_datetime(df['ec_frame_start_time']).dt.hour
    fr_ltime=np.array(fr_hour_vals+(df['ec_frame_start_lon']+df['ec_frame_end_lon'])/30)
    idx=np.where(fr_ltime<0)
    fr_ltime[idx]=fr_ltime[idx]+24
    idx=np.where(fr_ltime>24)
    fr_ltime[idx]=fr_ltime[idx]-24

    # Normalize time for colormap
    lt_min=min(np.nanmin(ltime_co),np.nanmin(fr_ltime))
    lt_max=max(np.nanmax(ltime_co),np.nanmax(fr_ltime))
    norm = mpl.colors.Normalize(vmin=lt_min,vmax=lt_max)
    cmap = plt.get_cmap('viridis')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot RO observation points with color by time
    sc = ax.scatter(df['ro_col_lon'], df['ro_col_lat'], c=ltime_co, cmap=cmap, norm=norm,
        label='RO Observations', s=50, zorder=5, marker='o', edgecolor='black')

    # Plot rectangles for each patch colored by time
    for idx, row in df.iterrows():
        lons = [row[corner[0]] for corner in patch_corner_names]
        lats = [row[corner[1]] for corner in patch_corner_names]
        tc = fr_ltime[idx]
        patch_color = cmap(norm(tc))
        ax.fill(lons, lats, color=patch_color, alpha=0.4, transform=ccrs.PlateCarree(), zorder=1, label="EarthCare Patch" if idx == 0 else None)

    # Add colorbar, formatting the ticks as time
    cb = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=35)
    cb.set_label("RO Collocation Time")
    # Set colorbar ticks as datetime labels
    tick_locs = np.linspace(lt_min, lt_max, num=5)
    cb.set_ticks(tick_locs)
    cb.ax.set_yticklabels([mpl.dates.num2date(tick).strftime("%H:%M:%S") for tick in tick_locs])
    # Set labels and legend
    ax.set_title('RO Observation Locations and Collocated EarthCare Patches', fontsize=15)
    ax.legend(loc='upper right')


    plt.show()