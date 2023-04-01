import pandas as pd
from pykrige.ok import OrdinaryKriging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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
            lon = (group_data.iloc[:, 3] + group_data.iloc[:, 4]) / 2
            lat = (group_data.iloc[:, 5] + group_data.iloc[:, 6]) / 2
            fdl = group_data.iloc[:, 9] / 100
            df = df.append(pd.DataFrame({'lat': lat, 'lon': lon, 'fdl': fdl}))
            # save the dataframe as a csv file with the filename as group_name + 'before'
            df.to_csv(f"{group_name}_before.csv")

            # create a 2D grid of points using numpy.meshgrid
            lon_grid = np.arange(118.25, 120.125, 0.025)
            lat_grid = np.arange(36.125, 37.125, 0.025)
            
            # create an instance of the OrdinaryKriging class
            OK = OrdinaryKriging(df['lon'], df['lat'], df['fdl'], variogram_model='linear', verbose=False, enable_plotting=False)
            
            # interpolate the data using the grid of points
            fdl_interp, sigmas = OK.execute('grid', lon_grid, lat_grid)
            
            # reshape the interpolated data to match the shape of the grid
            # fdl_interp = fdl_interp.reshape(lon_grid.shape)
            
            xgrid, ygrid = np.meshgrid(lon_grid, lat_grid)
            df_grid = pd.DataFrame(dict(long=xgrid.flatten(),lat=ygrid.flatten()))
            df_grid["Krig_gaussian"] = fdl_interp.flatten()
            
            # # create a new dataframe to store the interpolated data
            # df_interp = pd.DataFrame({'lat': lat_grid.flatten(), 'lon': lon_grid.flatten(), 'fdl_interp': fdl_interp.flatten()})
            
            # merge the original dataframe with the interpolated dataframe based on the lat and lon columns
            # df = pd.merge(df, df_interp, on=['lat', 'lon'], how='outer')
            df_grid.to_csv(f"{group_name}_after.csv")