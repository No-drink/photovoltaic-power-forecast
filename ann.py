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
eval_metrics_file = './eval_metrics_ann.txt'  # 评价指标保存的txt文件

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

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建ANN模型
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# 预测
y_pred = model.predict(X_test).flatten()

# 可视化预测结果
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Power')
plt.plot(y_pred, label='Predicted Power')
plt.xlabel('Index')
plt.ylabel('Power')
plt.legend()
plt.title('ANN Prediction Result')
plt.savefig('ann_prediction_result.png')  # 保存图片
plt.show()

# 计算评价指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

# 过滤掉测试数据集中值为零的数据点
non_zero_indices = y_test != 0
y_test_filtered = y_test[non_zero_indices]
y_pred_filtered = y_pred[non_zero_indices]

# 计算 MAPE
mape = (abs((y_test_filtered - y_pred_filtered) / y_test_filtered)).mean() * 100

# 将评价指标保存到txt文件中
with open(eval_metrics_file, 'w') as f:
    f.write(f'Mean Absolute Error (MAE): {mae}\n')
    f.write(f'Mean Absolute Percentage Error (MAPE): {mape}\n')
    f.write(f'Mean Squared Error (MSE): {mse}\n')
    f.write(f'Root Mean Squared Error (RMSE): {rmse}\n')
    f.write(f'R-squared (R2): {r2}\n')

print('ANN prediction result and evaluation metrics saved successfully.')
