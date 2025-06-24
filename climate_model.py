

# %% Download 5 days of 6-hourly temperature data from ERA5 (Europe)
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': '2m_temperature',
        'year': '2022',
        'month': '07',
        'day': ['01', '02', '03', '04', '05'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
        'area': [60, -10, 35, 30],  # [N, W, S, E] -> Europe-ish
    },
    'era5_europe_temp.nc'
)



# %% Inspect and plot the data using xarray and matplotlib

import xarray as xr
import matplotlib.pyplot as plt

# Load NetCDF file
ds = xr.open_dataset("era5_europe_temp.nc")
print(ds)
ds = ds.rename({'valid_time': 'time'})

# Extract temperature variable
temp = ds['t2m'] - 273.15  # Convert from Kelvin to Celsius

# Plot first timestep
temp.isel(time=0).plot(cmap='coolwarm')
plt.title("Temperature at t=0")
plt.show()

# %% Extract data as input and output 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load ERA5 temperature data and convert to Celsius
ds = xr.open_dataset("era5_europe_temp.nc")
temp = ds['t2m'] - 273.15  # Kelvin to Celsius

# Flip latitude if decreasing (ERA5 often is)
if np.all(np.diff(temp.latitude.values) < 0):
    temp = temp.sortby('latitude')

# Extract coordinates
lats = temp.latitude.values  # increasing order
lons = temp.longitude.values
times = temp.time.values     # this is the correct way to access time

# Get temperature array (time, lat, lon)
T = temp.values
nt, ny, nx = T.shape

# Create meshgrid using real lat/lon/time values
lon_grid, lat_grid, time_grid = np.meshgrid(
    lons, lats, times, indexing='xy'
)

# Flatten to create inputs: (lon, lat, time)
inputs = np.stack([
    lon_grid.ravel(),   # longitude
    lat_grid.ravel(),   # latitude
    time_grid.ravel()   # datetime64 values
], axis=1)

# Flatten temperature outputs
outputs = T.reshape(-1, 1)



# %%
import matplotlib.pyplot as plt

# Extract a time slice and reshape
time0 = times[0]
T_slice = outputs[inputs[:, 2] == time0].reshape((ny, nx))

plt.contourf(lons, lats, T_slice, levels=50, cmap='coolwarm')
plt.colorbar(label='Temperature (°C)')
plt.title("ERA5 Temperature Over Europe at t=0")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.gca().invert_yaxis()
plt.show()

