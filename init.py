import csv
from collections import defaultdict
import os
#test
# 定义需要读取和处理的列索引列表，索引从0开始
specified_columns = [4, 5, 7, 9, 10, 11, 12, 13, 16, 18]
# 定义不需要映射的列索引列表
at = [4,  16]


# 检查输入文件是否存在
input_file = 'adult_with_pii.csv'
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"The file {input_file} does not exist.")

# 打开CSV文件并读取数据
with open(input_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # 读取标题行，并且只保留指定的列
    header = next(csvreader)
    selected_header = [header[i] for i in specified_columns]

    # 初始化一个列表来存储所有行的数据
    data = []
    # 读取文件中的前1000行数据
    for _ in range(30000):
        try:
            row = next(csvreader)
            # 仅选择指定的列
            selected_row = [row[i] for i in specified_columns]
            data.append(selected_row)
        except StopIteration:
            # 如果数据不足1000行，提前终止循环
            break

# 对每列进行字符到数字的映射，除了at列表中的列
for col_index in range(len(selected_header)):
    # 如果当前列索引不在at列表中，则执行映射
    if specified_columns[col_index] not in at:
        char_to_number_map = defaultdict(lambda: len(char_to_number_map))
        for row in data:
            row[col_index] = char_to_number_map[row[col_index]]

# 输出文件的路径
output_file = 'output1234.csv'

# 将修改后的数据保存到新的CSV文件
try:
    with open(output_file, 'w', newline='') as output_csvfile:
        writer = csv.writer(output_csvfile)
        # 写入标题行
        writer.writerow(selected_header)
        # 写入修改后的数据
        writer.writerows(data)
    print(f"Data successfully written to {output_file}")
except IOError as e:
    print(f"An error occurred while writing to the file {output_file}: {e}")
