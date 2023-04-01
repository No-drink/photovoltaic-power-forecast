import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.spatial.distance import cdist

def idw_interpolation(x, y, z, xi, yi, power=100):
    """Inverse Distance Weighting interpolation."""
    # Compute distance matrix
    dist_matrix = cdist(np.column_stack((x, y)), np.column_stack((xi, yi)), metric='euclidean')
    # Avoid division by zero
    dist_matrix[dist_matrix == 0] = 0.000001
    # Compute weights
    weights = 1 / (dist_matrix ** power)
    # Compute weighted average
    zi = np.sum(weights * z[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
    return zi

folder_path = "./data/21_12"
# loop through all excel files in the folder and append data to the dataframe
for file in os.listdir(folder_path):
    if file.endswith('.xlsx'):
        data = pd.read_excel(f"{folder_path}/{file}")
        # use pandas groupby function to group the data by the values in the second column (assuming it contains the time data)
        # then loop through each group and append the data to the dataframe
        for group_name, group_data in data.groupby(data.columns[1]):   
            # create an empty dataframe to store the data
            df = pd.DataFrame()  
            lon = (group_data.iloc[:, 5] + group_data.iloc[:, 6]) / 2
            lat = (group_data.iloc[:, 3] + group_data.iloc[:, 4]) / 2
            fdl = group_data.iloc[:, 9] / 100
            df = df.append(pd.DataFrame({'lat': lat, 'lon': lon, 'fdl': fdl}))
            # save the dataframe as a csv file with the filename as group_name + 'before'
            df.to_csv(f"{group_name}_before.csv")

            # create a 2D grid of points using numpy.meshgrid
            lon_grid = np.arange(118.125, 120.125, 0.025)
            lat_grid = np.arange(35.625, 37.375, 0.025)
            
            # create an instance of the Inverse Distance Weighting class
            xgrid, ygrid = np.meshgrid(lon_grid, lat_grid)
            print(df['lon'], df['lat'], df['fdl'])
            fdl_interp = idw_interpolation(df['lon'], df['lat'], df['fdl'], xgrid.flatten(), ygrid.flatten(), power=2)
            
            df_grid = pd.DataFrame(dict(lat=ygrid.flatten(),lon=xgrid.flatten()))
            df_grid["IDW"] = fdl_interp
            
            df_grid.to_csv(f"{group_name}_after.csv")
