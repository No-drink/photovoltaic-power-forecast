import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

def filter_files_with_smaller_x(folder_path):
    # 用于存储文件名前缀与X的映射关系
    prefix_x_map = {}
    # 结果列表，用于存储保留的文件名
    result_file_list = []
    # 读取文件夹中所有文件名称
    for file_name in os.listdir(folder_path):
        # 提取文件名中的前缀和X部分
        file_name = file_name[:-4]
        file_prefix, file_x = file_name.rsplit('_', 1)
        file_x = int(file_x)
        # 检查前缀是否已存在于映射关系中
        if file_prefix in prefix_x_map:
            # 如果存在，则比较X的大小
            if file_x < prefix_x_map[file_prefix]:
                # 从结果列表中移除X较大的文件名
                result_file_list.remove(f"{file_prefix}_{prefix_x_map[file_prefix]}")
                # 更新映射关系中的X值
                prefix_x_map[file_prefix] = file_x
                # 将X较小的文件名添加到结果列表中
                result_file_list.append(file_name)
        else:
            # 将前缀和X存入映射关系中
            prefix_x_map[file_prefix] = file_x
            # 将文件名添加到结果列表中
            result_file_list.append(file_name)
    return result_file_list

# 您可以将folder_path更改为您需要处理的文件夹路径
folder_path = "./data/given_data"
# 调用函数处理文件夹中的文件名，并获得结果列表
filtered_file_list = filter_files_with_smaller_x(folder_path)


def fit_fourier_series(filenames, num_terms, predict_hours):
    # 读取多个csv文件并合并成一个大的DataFrame
    dfs = []
    for filename in filenames:
        df = pd.read_csv(f"./data/given_data/{filename}.csv")
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    # 提取时刻（小时数）和气温作为自变量和因变量
    hours = data['TIME'].values
    temperature = data['TEM'].values
    
    # 基础角频率 (年周期)
    omega_y = 2 * np.pi / (365 * 24)
    omega_d = 2 * np.pi / 24
    
    # 构建傅里叶级数特征
    fourier_features = np.column_stack([np.cos(omega_y * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.sin(omega_y * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.cos(omega_d * n * hours) for n in range(1, num_terms + 1)] +
                                       [np.sin(omega_d * n * hours) for n in range(1, num_terms + 1)])
    
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
    predict_fourier_features = np.column_stack( [np.cos(omega_y * n * hours) for n in range(1, num_terms + 1)] +
                                                [np.sin(omega_y * n * hours) for n in range(1, num_terms + 1)] +
                                                [np.cos(omega_d * n * hours) for n in range(1, num_terms + 1)] +
                                                [np.sin(omega_d * n * hours) for n in range(1, num_terms + 1)] )
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

predict_temp = fit_fourier_series(filtered_file_list, 30, predict_hours)

print(predict_temp)