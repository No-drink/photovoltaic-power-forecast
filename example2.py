import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
from statsmodels.tsa.arima.model import ARIMA

# 创建模拟数据
X = np.random.rand(100, 5)  # 100个样本，5个特征
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.rand(100)  # 带噪声的线性关系
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. 线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 2. 决策树回归
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# 3. 随机森林回归
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# 4. 支持向量回归
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

# 5. 梯度提升回归
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# 6. 神经网络
nn = Sequential()
nn.add(Dense(units=32, activation='relu', input_dim=5))
nn.add(Dense(units=1))
nn.compile(optimizer='adam', loss='mean_squared_error')
nn.fit(X_train, y_train, epochs=10, batch_size=8)

# 7. ARIMA模型
# 假设y是时间序列数据
arima = ARIMA(y, order=(1, 0, 0))
arima_fit = arima.fit()

# 预测
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svr = svr.predict(X_test)
y_pred_gbr = gbr.predict(X_test)
y_pred_nn = nn.predict(X_test)
y_pred_arima = arima_fit.predict(start=len(y), end=len(y) + 9)  # 预测未来10个时间点

# 输出预测结果
print(y_pred_lr)
print(y_pred_dt)
print(y_pred_rf)
print(y_pred_svr)
print(y_pred_gbr)
print(y_pred_nn)
print(y_pred_arima)
