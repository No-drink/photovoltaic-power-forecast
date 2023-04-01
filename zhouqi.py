# Part 1: Read and process data
import pandas as pd
import os

# Define folder path
folder = 'path/to/folder'

# Read all csv files in folder and concatenate into one dataframe
df = pd.concat([pd.read_csv(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.csv')])

# Drop duplicates based on time2 column, keeping rows with highest time1 value
df = df.sort_values('time1', ascending=False).drop_duplicates('time2')

# Part 2: Fit monthly periodic function
import numpy as np
from scipy.optimize import curve_fit

# Define function to fit
def monthly_periodic(x, a, b, c, d):
    return a + b*np.sin(2*np.pi*x/30) + c*np.cos(2*np.pi*x/30) + d*np.sin(4*np.pi*x/30)

# Group data by latitude and longitude
grouped = df.groupby(['latitude', 'longitude'])

# Fit function to each group and store coefficients
coefficients = {}
for group, data in grouped:
    x = data['time2'].values
    y = data['temperature'].values
    popt, _ = curve_fit(monthly_periodic, x, y)
    coefficients[group] = popt

# Part 3: Predict temperature based on time and location
def predict_temperature(time, latitude, longitude):
    if (latitude, longitude) not in coefficients:
        return None
    popt = coefficients[(latitude, longitude)]
    return monthly_periodic(time, *popt)

# Example usage
temperature = predict_temperature(15, 1.23, 4.56)  # Predict temperature at time 15 and location (1.23, 4.56)