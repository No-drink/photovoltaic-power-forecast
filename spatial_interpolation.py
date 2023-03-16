# 导入pykrige模块
import pykrige as pk
import numpy as np

# 假设你有一些2D数据，例如温度或高度
x = [1.0, 2.0, 3.0, 4.0] # x坐标
y = [1.0, 2.0, 3.0, 4.0] # y坐标
z = [10.0, 20.0, 30.0, 40.0] # 数据值

# 创建一个普通克里金对象
OK = pk.OrdinaryKriging(x,y,z)

# 指定一个网格来进行插值，例如5x5的网格
gridx = np.linspace(1.0,4.0,num=5)
gridy = np.linspace(1.0,4.0,num=5)

# 调用execute方法来进行插值，并得到插值结果和方差估计
z_ok,var_ok = OK.execute('grid',gridx,gridy)

# 创建一个通用克里金对象，指定趋势模型为线性（ax+by+c）
UK = pk.UniversalKriging(x,y,z,trend='linear')

# 调用execute方法来进行插值，并得到插值结果和方差估计
z_uk,var_uk = UK.execute('grid',gridx,gridy)