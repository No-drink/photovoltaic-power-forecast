# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import griddata

# # 读取file1.csv和file2.csv
# file1 = pd.read_csv('20211203_after.csv')
# file2 = pd.read_csv('20211203_before.csv')

# # 从file1.csv和file2.csv中提取经度、纬度和值
# lat1 = file1['lat']
# lon1 = file1['lon']
# value1 = file1['IDW']

# lat2 = file2['lat']
# lon2 = file2['lon']
# value2 = file2['fdl']

# # 创建3D图像
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 将file1.csv的点进行插值以生成平滑曲面
# grid_x, grid_y = np.mgrid[min(lat1):max(lat1):100j, min(lon1):max(lon1):100j]
# grid_z = griddata((lat1, lon1), value1, (grid_x, grid_y), method='cubic')

# # 绘制平滑曲面
# surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.6)

# # 将file2.csv中的点标注在图像上
# scatter = ax.scatter(lat2, lon2, value2, c='red', marker='o')

# # 设置图像的坐标轴标签
# ax.set_xlabel('Latitude')
# ax.set_ylabel('Longitude')
# ax.set_zlabel('Power Generation')

# # 显示图像
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 读取file2.csv文件
# file2 = pd.read_csv('20211201_before.csv')

# # 从file2.csv中提取经度、纬度和值
# lat2 = file2['lat']
# lon2 = file2['lon']
# value2 = file2['fdl']

# # 创建3D图像
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 定义柱状图的宽度和深度
# width = 0.05
# depth = 0.05

# # 绘制柱状图
# for i in range(len(lat2)):
#     ax.bar3d(lat2[i], lon2[i], 0, width, depth, value2[i])

# # 设置图像的坐标轴标签
# ax.set_xlabel('Latitude')
# ax.set_ylabel('Longitude')
# ax.set_zlabel('Power Generation')

# # 显示图像
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取CSV文件
df = pd.read_csv('needtodraw.csv')

# 提取起始经度、终止经度、起始纬度、终止纬度和值
start_lon = df['QSJD']
end_lon = df['ZZJD']
start_lat = df['QSWD']
end_lat = df['ZZWD']
value = df['FDL']

# 计算每个柱子的宽度和深度
width = end_lon - start_lon
depth = end_lat - start_lat

# 创建3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制柱状图
ax.bar3d(start_lon, start_lat, 0, width, depth, value)

# 设置图像的坐标轴标签
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Power Generation')

# 显示图像
plt.show()
