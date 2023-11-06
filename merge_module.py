import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# 文件夹路径
csv_folder_path = './data/dataset'  # 请修改为实际的文件夹路径
eval_metrics_file = './eval_metrics_merge.txt'  # 评价指标保存的txt文件

# 读取CSV文件形成数据集
data = []
for csv_file_name in os.listdir(csv_folder_path):
    if csv_file_name.endswith('.csv'):
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)
        df = pd.read_csv(csv_file_path)
        data.append(df)

# 合并数据集
data = pd.concat(data, axis=0)

# 删除无关列
# data = data.drop(columns=['hours_from_2020_start1', 'hours_from_2020_start2'])

# 定义自变量和因变量
X = data.drop(columns=['power'])
y = data['power']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 假设X_train, y_train, X_test, y_test已经准备好
# 将训练集分成两部分
X_train1, X_train2 = np.split(X_train, 2)
y_train1, y_train2 = np.split(y_train, 2)

# 使用随机森林模型进行训练和预测
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train1, y_train1)
y_pred_rf = model_rf.predict(X_test)

# 数据标准化
scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train2)
X_test_scaled = scaler.transform(X_test)

# 使用人工神经网络模型进行训练和预测
model_ann = Sequential()
model_ann.add(Dense(128, input_dim=X_train2.shape[1], activation='relu'))
model_ann.add(Dense(64, activation='relu'))
model_ann.add(Dense(1, activation='linear'))
model_ann.compile(loss='mean_squared_error', optimizer='adam')
model_ann.fit(X_train2, y_train2, epochs=20, batch_size=32, verbose=0)
y_pred_ann = model_ann.predict(X_test_scaled).flatten()

# 将两个模型的预测结果进行加权平均
weights = [0.8, 0.2]  # 权重，可以根据实际情况调整
y_pred_ensemble = weights[0] * y_pred_rf + weights[1] * y_pred_ann

# 可视化预测结果
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Power')
plt.plot(y_pred_ensemble, label='Predicted Power (Ensemble)')
plt.xlabel('Index')
plt.ylabel('Power')
plt.legend()
plt.title('Model Ensemble Prediction Result')
plt.show()

# 计算评价指标
mae = mean_absolute_error(y_test, y_pred_ensemble)
# mape = (abs((test_data - y_pred) / test_data)).mean() * 100  # MAPE
mse = mean_squared_error(y_test, y_pred_ensemble)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred_ensemble)

# 过滤掉测试数据集中值为零的数据点
non_zero_indices = y_test != 0
y_test_filtered = y_test[non_zero_indices]
y_pred_filtered = y_pred_ensemble[non_zero_indices]

# 计算 MAPE
mape = (abs((y_test_filtered - y_pred_filtered) / y_test_filtered)).mean() * 100

# 将评价指标保存到txt文件中
with open(eval_metrics_file, 'w') as f:
    f.write(f'Mean Absolute Error (MAE): {mae}\n')
    f.write(f'Mean Absolute Percentage Error (MAPE): {mape}%\n')
    f.write(f'Mean Squared Error (MSE): {mse}\n')
    f.write(f'Root Mean Squared Error (RMSE): {rmse}\n')
    f.write(f'R-squared (R2): {r2}\n')

print('Merged module result and evaluation metrics saved successfully.')
