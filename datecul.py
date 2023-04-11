import pandas as pd
from datetime import datetime

def calculate_hours_difference_and_year_start(df):
    # 定义时间格式
    time_format = '%d.%m.%Y %H:%M:%S'
    # 定义2020年1月1日零时零分的时间
    year_2020_start = datetime.strptime('01-01-2019 00:00:00', '%d-%m-%Y %H:%M:%S')
    # 计算第一组时间与第二组时间之间相差的小时数
    df['hours_difference'] = (df.apply(lambda x: datetime.strptime(x['time2'], time_format) - datetime.strptime(x['time1'], time_format), axis=1)
                              .dt.total_seconds() / 3600)
    # 计算第一组时间距离2020年1月1日零时零分的小时数
    df['hours_from_2020_start1'] = df.apply(lambda x: (datetime.strptime(x['time1'], time_format) - year_2020_start).total_seconds() / 3600, axis=1)
    # 计算第二组时间距离2020年1月1日零时零分的小时数
    df['hours_from_2020_start2'] = df.apply(lambda x: (datetime.strptime(x['time2'], time_format) - year_2020_start).total_seconds() / 3600, axis=1)
    return df

# 读取CSV文件夹中的每一个文件
def process_csv_files(folder_path):
    import os
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # 读取CSV文件
            df = pd.read_csv(file_path)
            # 将时间列合并
            df['time1'] = df['dd.mm.yyyy'] + ' ' + df['hh:mi:ss']
            df['time2'] = df['dd.mm.yyyy.1'] + ' ' + df['hh:mi:ss.1']
            # 计算时间差
            df = calculate_hours_difference_and_year_start(df)
            # 删除临时列
            df.drop(['time1', 'time2'], axis=1, inplace=True)
            # 将结果保存到CSV文件
            df.to_csv(file_path, index=False)


# 调用函数处理文件夹中的所有CSV文件
folder_path = './data/demodata/气象预测数据/cepri_historic_2019010112_2020123112_JSGF001_JSGF001'  # 请将此处替换为实际的CSV文件夹路径
process_csv_files(folder_path)
