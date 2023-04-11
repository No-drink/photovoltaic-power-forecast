import pandas as pd
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
        year_start = datetime(dt.year, 1, 1)
        # 计算时间差并转换为小时数
        hours_since_year_start = (dt - year_start).total_seconds() / 3600
        return hours_since_year_start

    # 将时间列转换为距离该年度第一天零时零分的小时数
    df['时间'] = df['时间'].apply(calculate_hours)

    return df

# 调用函数读取并转换数据
file_path = './data/demodata/JSGF001/附件2-场站出力.xlsx'
result_df = read_and_transform_xlsx(file_path)

# 打印转换后的数据
print(result_df)