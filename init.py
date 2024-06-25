import csv
from collections import defaultdict
import os

# 定义需要读取和处理的列索引列表，索引从0开始
specified_columns = [4, 5, 7, 9, 10, 11, 12, 13, 16, 18]

# 定义不需要映射的列索引列表
at = []

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
    # 读取文件中的所有数据
    for row in csvreader:
        # 仅选择指定的列
        selected_row = [row[i] for i in specified_columns]
        data.append(selected_row)

# 对每列进行字符到数字的映射，并统计类别个数，除了at列表中的列
column_classes_count = [defaultdict(int) for _ in range(len(selected_header))]

for col_index in range(len(selected_header)):
    # 如果当前列索引不在at列表中，则执行映射
    if specified_columns[col_index] not in at:
        char_to_number_map = defaultdict(lambda: len(char_to_number_map))
        for row in data:
            row[col_index] = char_to_number_map[row[col_index]]
            column_classes_count[col_index][row[col_index]] += 1
    else:
        for row in data:
            column_classes_count[col_index][row[col_index]] += 1

# 创建一个数组来存储每列的类别总数
total_classes_array = [len(column_classes_count[col_index]) for col_index in range(len(selected_header))]

# 输出每列的类别总数数组
print(total_classes_array)

# 创建输出目录
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# 将数据分块存储到多个CSV文件中，每个文件包含累积的行数据
chunk_size = 1000
num_chunks = (len(data) + chunk_size - 1) // chunk_size  # 计算总块数

for chunk_index in range(num_chunks):
    end_index = (chunk_index + 1) * chunk_size
    chunk_data = data[:end_index]
    output_file = f'{output_dir}/output{chunk_index}.csv'

    try:
        with open(output_file, 'w', newline='') as output_csvfile:
            writer = csv.writer(output_csvfile)
            # 写入标题行
            writer.writerow(selected_header)
            # 写入当前块的数据
            writer.writerows(chunk_data)
        print(f"Data successfully written to {output_file}")
    except IOError as e:
        print(f"An error occurred while writing to the file {output_file}: {e}")
