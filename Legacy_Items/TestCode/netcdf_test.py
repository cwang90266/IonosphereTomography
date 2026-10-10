import xarray as xr

# Specify the path to your NetCDF file
netcdf_file_path = '/Users/cwang/Documents/Consulting/PlanetIQ/Data/SampleData/2025.314/podTc2_GN04.2025.314.23.45.0027.E30.00_0000.0001_nc'

# Read the NetCDF file into an xarray.Dataset
ds = xr.open_dataset(netcdf_file_path)


