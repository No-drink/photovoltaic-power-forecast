# lat: 纬度
# lon: 经度
# TEM: 气温
# PRE_1h: 1小时降水量
# RHU: 相对湿度
# WD: 风向
# WS: 风速
# SR: 日照时数
# TCC: 总云量
import os
import re
import netCDF4
import pandas as pd

folder_path = "./data/given_data" # Change this to your folder path

# Analysis the data information
def datainform(file_name):
    dtm = re.findall(r"km(.+?).nc", file_name)[0]
    period, starttime, endtime = re.findall(r"(\d+)", dtm)
    starttime = pd.to_datetime(starttime)
    endtime = pd.to_datetime(endtime)
    return [starttime,endtime]

def read_all(folder_path, df):
    file_list = os.listdir(folder_path)
    for file in file_list:
        file2read = netCDF4.Dataset(f"{folder_path}/{file}", "r") # Open the file in read mode
        endtime = datainform(file)[1]
        datas_in = []
        for TEM in file2read.variables['TEM'][:].data:
            datas_in.extend(TEM)
        for PRE_1h in file2read.variables['PRE_1h'][:].data:
            datas_in.extend(PRE_1h)
        for RHU in file2read.variables['RHU'][:].data:
            datas_in.extend(RHU)
        for WD in file2read.variables['WD'][:].data:
            datas_in.extend(WD)
        for WS in file2read.variables['WS'][:].data:
            datas_in.extend(WS)
        for SR in file2read.variables['SR'][:].data:
            datas_in.extend(SR)
        for TCC in file2read.variables['TCC'][:].data:
            datas_in.extend(TCC)
        df.loc[:,endtime] = datas_in
    return True

# Generate the index
file_example = netCDF4.Dataset(f"./data/given_data/JTBZDB1D_yb_gdfs-meter_3d2.5km15min_202111301200_202112010800.nc", "r") # Open the file in read mode
lats = file_example.variables['lat'][:].data
lons = file_example.variables['lon'][:].data
categories = ['TEM','PRE_1h','RHU','WD','WS','SR','TCC']
index = pd.MultiIndex.from_product([categories, lats, lons], names=['category', 'lat', 'lon'])
weather_data = pd.DataFrame(index=index)

if (read_all(folder_path, weather_data)):
    weather_data.to_csv('weather_data.csv', index=True, header=True)