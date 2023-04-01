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

# 输出结果列表
print(filtered_file_list)
