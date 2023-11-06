import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# 读取Excel文件
file_path = './data/demodata/JSGF001/附件2-场站出力.xlsx'  # 请修改为实际Excel文件的路径
df = pd.read_excel(file_path)

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
# 将时间列转化为距离2019年1月1日零时的小时数
start_time = pd.Timestamp('2019-01-01 00:00:00')
df['hours_since_start'] = (df.iloc[:, 0] - start_time) / pd.Timedelta(hours=1)

# 定义自变量和因变量
X = df['hours_since_start']
y = df.iloc[:, 1]

# 划分训练集和测试集（例如前80%作为训练集，后20%作为测试集）
split_index = int(len(y) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 建立ARIMA模型并训练
model = ARIMA(y_train, order=(2,1,2))  # 参数order设置为(2,1,2)，可根据实际情况调整
model_fit = model.fit()

# 进行预测
y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)

# 计算评价指标
mae = mean_absolute_error(y_test, y_pred)
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 过滤掉测试数据集中值为零的数据点
non_zero_indices = y_test != 0
y_test_filtered = y_test[non_zero_indices]
y_pred_filtered = y_pred[non_zero_indices]

# 计算 MAPE
mape = (abs((y_test_filtered - y_pred_filtered) / y_test_filtered)).mean() * 100

# 输出评价指标
print('Mean Absolute Error (MAE):', mae)
print('Mean Absolute Percentage Error (MAPE):', mape, '%')
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared (R2):', r2)

# 可视化预测结果
plt.figure(figsize=(10, 5))
plt.plot(X_test, y_test, label='Actual Power')
plt.plot(X_test, y_pred, label='Predicted Power', linestyle='--')
plt.xlabel('Hours since 2019-01-01 00:00:00')
plt.ylabel('Power (MW)')
plt.legend()
plt.title('ARIMA Prediction Result')
plt.show()
