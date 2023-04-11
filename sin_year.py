import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 读取xlsx文件
def read_and_transform_xlsx(file_path):
    # 读取xlsx文件，假设时间在第一列，功率在第二列
    df = pd.read_excel(file_path, names=['时间', '功率'])

    # 定义一个函数将时间转换为距离该年度第一天零时零分的小时数
    def calculate_hours(time_str):
        # 解析时间字符串
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        # 计算该年度第一天零时零分的时间
        year_start = datetime(2019, 1, 1)
        # 计算时间差并转换为小时数
        hours_since_year_start = (dt - year_start).total_seconds() / 3600
        return hours_since_year_start

    # 将时间列转换为距离该年度第一天零时零分的小时数
    df['时间'] = df['时间'].apply(calculate_hours)

    return df

# 调用函数读取并转换数据
file_path = './data/demodata/JSGF001/附件2-场站出力.xlsx'
result_df = read_and_transform_xlsx(file_path)


def fit_fourier_series(data, num_terms, predict_hours):
    
    # 提取时刻（小时数）和气温作为自变量和因变量
    hours = data['时间'].values
    power = data['功率'].values
    
    # 基础角频率 (年周期)
    omega_y = 2 * np.pi / (365 * 24)
    omega_d = 2 * np.pi / 24
    
    # 构建傅里叶级数特征
    fourier_features = np.column_stack([np.cos(omega_y * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.sin(omega_y * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.cos(omega_d * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.sin(omega_d * n * hours) for n in range(1, num_terms + 1)]
                                       )
    
    # 使用线性回归模型进行拟合
    model = LinearRegression(fit_intercept=True)
    model.fit(fourier_features, power)
    
    # 输出傅里叶级数表达式
    fourier_expr = f"f(t) = {model.intercept_:.4f}"
    for i, coef in enumerate(model.coef_[:num_terms]):
        fourier_expr += f" + {coef:.4f} * cos({i + 1} * ωt)"
    for i, coef in enumerate(model.coef_[num_terms:]):
        fourier_expr += f" + {coef:.4f} * sin({i + 1} * ωt)"
    print(f"Fourier series expression: {fourier_expr}")
    
    # 使用模型进行预测
    predict_hours = np.array(predict_hours)
    predict_fourier_features = np.column_stack( [np.cos(omega_y * n * predict_hours) for n in range(1, num_terms + 1)] +
                                                [np.sin(omega_y * n * predict_hours) for n in range(1, num_terms + 1)] +
                                                [np.cos(omega_d * n * predict_hours) for n in range(1, num_terms + 1)] +
                                                [np.sin(omega_d * n * predict_hours) for n in range(1, num_terms + 1)] 
                                                )
    predict_power = model.predict(predict_fourier_features)
        

    # 可视化结果
    plt.scatter(hours, power, label='Observed Data')
    plt.plot(predict_hours, predict_power, 'r-', label='Fitted Fourier Series')
    plt.xlabel('Hours')
    plt.ylabel('power')
    plt.legend()
    plt.show()
    
    return predict_power


# 预测时刻（小时数）列表
predict_hours = np.linspace(0, 8760*2, 35040)  # 在0到48小时之间均匀取

predict_temp = fit_fourier_series(result_df, 45, predict_hours)

print(predict_temp)