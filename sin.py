import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def fit_fourier_series(filenames, num_terms, predict_hours):
    # 读取多个csv文件并合并成一个大的DataFrame
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    # 提取时刻（小时数）和气温作为自变量和因变量
    hours = data['时刻'].values
    temperature = data['气温'].values
    
    # 基础角频率
    omega = 2 * np.pi / 24
    
    # 构建傅里叶级数特征
    fourier_features = np.column_stack([np.cos(omega * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.sin(omega * n * hours) for n in range(1, num_terms + 1)])
    
    # 使用线性回归模型进行拟合
    model = LinearRegression(fit_intercept=True)
    model.fit(fourier_features, temperature)
    
    # 输出傅里叶级数表达式
    fourier_expr = f"f(t) = {model.intercept_:.4f}"
    for i, coef in enumerate(model.coef_[:num_terms]):
        fourier_expr += f" + {coef:.4f} * cos({i + 1} * ωt)"
    for i, coef in enumerate(model.coef_[num_terms:]):
        fourier_expr += f" + {coef:.4f} * sin({i + 1} * ωt)"
    print(f"Fourier series expression: {fourier_expr}")
    
    # 使用模型进行预测
    predict_hours = np.array(predict_hours)
    predict_fourier_features = np.column_stack([np.cos(omega * n * predict_hours) for n in range(1, num_terms + 1)] +
                                               [np.sin(omega * n * predict_hours) for n in range(1, num_terms + 1)])
    predict_temperature = model.predict(predict_fourier_features)
    
    # 可视化结果
    plt.scatter(hours, temperature, label='Observed Data')
    plt.plot(predict_hours, predict_temperature, 'r-', label='Fitted Fourier Series')
    plt.xlabel('Hours')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
    
    return predict_temperature

# 示例：CSV文件名列表
filenames = ['file1.csv', 'file2.csv']

# 预测时刻（小时数）列表
predict_hours = np.linspace(0, 48, 100)  # 在0到48小时之间均匀取
