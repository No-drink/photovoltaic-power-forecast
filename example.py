import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

# Load and preprocess weather data
# Assume columns: latitude, longitude, time, weather, radiation, precipitation
weather_df = pd.read_csv('weather_data.csv')

# Perform time series interpolation on weather data
weather_df = weather_df.interpolate(method='linear')

# Perform Kriging interpolation on geographic data
# Assume uniform grid of coordinates for simplicity
lat_values = np.linspace(weather_df['latitude'].min(), weather_df['latitude'].max(), 119)
lon_values = np.linspace(weather_df['longitude'].min(), weather_df['longitude'].max(), 99)
grid_lat, grid_lon = np.meshgrid(lat_values, lon_values)

# Select features for interpolation (e.g., weather, radiation, precipitation)
for feature in ['weather', 'radiation', 'precipitation']:
    kriging_model = OrdinaryKriging(
        weather_df['longitude'], weather_df['latitude'], weather_df[feature],
        variogram_model='linear', verbose=False, enable_plotting=False
    )
    interpolated_feature, _ = kriging_model.execute('grid', grid_lon, grid_lat)
    # Flatten and concatenate interpolated feature to the dataframe
    weather_df[feature] = interpolated_feature.flatten()

# Load power generation data
# Assume columns: time, power
power_df = pd.read_csv('power_data.csv')

# Merge weather data with power generation data based on time
merged_df = pd.merge(weather_df, power_df, on='time')

# Split data into features (X) and target (y)
X = merged_df.drop(columns=['time', 'power'])
y = merged_df['power']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Apply Savitzky-Golay filter for data smoothing
smoothed_y_pred = savgol_filter(y_pred, window_length=5, polyorder=2)

# Evaluate model performance (e.g., mean squared error)
mse = np.mean((y_test - smoothed_y_pred) ** 2)
print(f'Mean Squared Error: {mse:.2f}')

# Save predictions to a CSV file
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': smoothed_y_pred})
predictions_df.to_csv('predictions.csv', index=False)
