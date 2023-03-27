import pandas as pd
from pykrige.ok import OrdinaryKriging
import numpy as np

# Read the CSV file and select only rows 2 to 7899 (0-based index)
df = pd.read_csv('weather_data.csv', skiprows=range(1, 2), nrows=7898)

# Extract the latitude, longitude, and temperature columns
latitudes = df['lat'].values
longitudes = df['lon'].values
temperatures = df['2021-12-01 08:00:00'].values  # Temperature at time T1

# Perform Kriging interpolation on temperature data at time T1
ok = OrdinaryKriging(
    longitudes, latitudes, temperatures, variogram_model='linear',
    verbose=False, enable_plotting=False
)

# Create a grid for interpolation
grid_lon = np.linspace(min(longitudes), max(longitudes), 100)
grid_lat = np.linspace(min(latitudes), max(latitudes), 100)

# Get interpolated values on the grid
z, ss = ok.execute('grid', grid_lon, grid_lat)

# Time series interpolation
# Select columns with time series data (starting from column D)
time_series_df = df.iloc[:, 3:]

# Perform time series interpolation (e.g., linear interpolation)
interpolated_time_series_df = time_series_df.interpolate(method='linear')

# Save the interpolated time series data to a new CSV file
interpolated_time_series_df.to_csv('interpolated_time_series.csv', index=False)
