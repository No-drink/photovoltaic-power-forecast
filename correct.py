import os
import re

def convert_spaces_to_commas_in_csv_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 遍历所有文件
    for file in files:
        # 检查文件是否是 CSV 文件
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            
            # 读取文件内容
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # 处理每一行，将一个或多个空格替换为逗号，并保留换行符
                processed_lines = []
                for line in lines:
                    # 使用正则表达式将一个或多个空格替换为逗号，忽略行末的空格和换行符
                    processed_line = re.sub(r' +', ',', line.strip()) + '\n'
                    processed_lines.append(processed_line)
                
            # 将修改后的内容写回文件
            with open(file_path, 'w') as f:
                f.writelines(processed_lines)
                
    print('All CSV files in the folder have been successfully processed.')



# 指定文件夹路径
folder_path = './data/demodata/气象预测数据/cepri_historic_2019010112_2020123112_JSGF001_JSGF001'  # 请将此处替换为实际文件夹路径

# 调用函数处理文件夹中的 CSV 文件
convert_spaces_to_commas_in_csv_files(folder_path)
