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

file_list = os.listdir(folder_path)
for file in file_list:
    if file.endswith(".nc"):
        file2read = netCDF4.Dataset(f"{folder_path}/{file}", "r") # Open the file in read mode
        starttime = datainform(file)[0]
        endtime = datainform(file)[1]
        start_of_year = pd.Timestamp(year=endtime.year, month=1, day=1, hour=0, minute=0, second=0)

        # 计算endtime与该年1月1日零点之间的时间差，并将其转换为小时数
        hours_since_start_of_year = (endtime - start_of_year) / pd.Timedelta(hours=1)
        duration = (endtime - starttime) / pd.Timedelta(hours=1)

        dtm = re.findall(r"km(.+?).nc", file)[0]
        period, starttime, endtime = re.findall(r"(\d+)", dtm)

        lats = file2read.variables['lat'][:].data
        lons = file2read.variables['lon'][:].data
        index = pd.MultiIndex.from_product([lats, lons], names=['lat', 'lon'])
        weather_data = pd.DataFrame(index=index)
        weather_data["TIME"] = hours_since_start_of_year
        weather_data["DUA"] = duration
        
        weather_data["TEM"] = [item for sublist in file2read.variables['TEM'][:].data for item in sublist]
        weather_data["PRE_1h"] = [item for sublist in file2read.variables['PRE_1h'][:].data for item in sublist]
        weather_data["RHU"] = [item for sublist in file2read.variables['RHU'][:].data for item in sublist]
        weather_data["WD"] = [item for sublist in file2read.variables['WD'][:].data for item in sublist]
        weather_data["WS"] = [item for sublist in file2read.variables['WS'][:].data for item in sublist]
        weather_data["SR"] = [item for sublist in file2read.variables['SR'][:].data for item in sublist]
        weather_data["TCC"] = [item for sublist in file2read.variables['TCC'][:].data for item in sublist]
        weather_data.to_csv(f'{folder_path}/{endtime}_{int(duration)}.csv', index=True, header=True)
    else:
        continue