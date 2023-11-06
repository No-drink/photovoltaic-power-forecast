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

# 傅里叶级数拟合
def fit_fourier_series(data, num_terms):
    # 提取时刻（小时数）和功率作为自变量和因变量
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
    
    return model, omega_y, omega_d

# 使用模型预测
def calculate_power(model, omega_y, omega_d, num_terms, predict_hours):
    predict_fourier_features = np.column_stack([np.cos(omega_y * n * predict_hours) for n in range(1, num_terms + 1)] +
                                               [np.sin(omega_y * n * predict_hours) for n in range(1, num_terms + 1)] +
                                               [np.cos(omega_d * n * predict_hours) for n in range(1, num_terms + 1)] +
                                               [np.sin(omega_d * n * predict_hours) for n in range(1, num_terms + 1)]
                                               )
    calculated_power = model.predict(predict_fourier_features)
    return calculated_power

# 指定xlsx文件路径
xlsx_file_path = './data/demodata/JSGF001/附件2-场站出力.xlsx'
# 指定CSV文件夹路径
csv_folder_path = './data/dataset'

# 调用函数读取并转换xlsx数据
result_df = read_and_transform_xlsx(xlsx_file_path)

# 拟合傅里叶级数，获取模型和角频率
num_terms = 45
model, omega_y, omega_d = fit_fourier_series(result_df, num_terms)

# 遍历CSV文件夹中的CSV文件
for csv_file_name in os.listdir(csv_folder_path):
    if csv_file_name.endswith('.csv'):
        # 获取CSV文件的完整路径
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)

        # 读取CSV文件
        df_csv = pd.read_csv(csv_file_path)

        # 获取CSV文件中的hours_from_2020_start2列作为需要预测的时间点（从2019年开始计算）
        predict_hours = df_csv['hours_from_2020_start2'].values

        # 使用拟合的傅里叶级数模型计算这些时刻的发电量
        predict_power = calculate_power(model, omega_y, omega_d, num_terms, predict_hours)

        # 将预测结果作为predic_power列附加到CSV文件中作为新的列
        df_csv['predic_power'] = predict_power

        # 将结果保存到同名的CSV文件中
        df_csv.to_csv(csv_file_path, index=False)

# 请根据实际情况修改xlsx文件路径和CSV文件夹路径
# 并确保已备份CSV文件，以防数据丢失
