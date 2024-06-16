import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决图中负号显示问题

# 导入自定义函数
from function import *
from function1 import *
from fig import *

# 读取数据
old_data = np.genfromtxt('output123.csv', delimiter=',', skip_header=1, dtype=int)
header = np.genfromtxt('output123.csv', delimiter=',', dtype=str, max_rows=1)

# 定义get_pij函数
def get_pij(r, w, h, e):
    return r * w * ( h) * e

# 层次分析法比较矩阵
comparison_matrix1 = np.array([
    [1,   1/3, 1/4, 1/4, 1/5, 1/6, 1/4, 1/7, 1/4, 1/8],
    [3,    1,  1/2, 1/2, 1/3, 1/4, 1/3, 1/6, 1/2, 1/7],
    [4,    2,    1,  3/2, 1/2, 1/2, 1/3, 1/5, 3/2, 1/6],
    [4,    2,  2/3,   1,  1/2, 1/3, 1/4, 1/5, 1/2, 1/6],
    [5,    3,    2,    2,   1,  1/2, 1/3, 1/4, 1/2, 1/5],
    [6,    4,    2,  3/2,   2,   1,  1/2, 1/3, 3/2, 1/4],
    [4,    3,  3/2,    2,   3,   2,    1,  1/3,  1/2, 1/3],
    [7,    6,    5,    5,   4,   3,    3,    1,  1/2, 1/3],
    [4,    2,  2/3,  3/2,  2,  2/3,  2,    2,    1,  1/4],
    [8,    7,    6,    6,   5,   4,    3,    3,    4,    1]
])



comparison_matrix1 = np.array([
    [1,   1/3, 1/4, 1/4, 1/5, 1/6, 1/4, 1/7, 1/6, 1/8],
    [3,    1,  3/2, 1/2, 1/3, 1/4, 1/3, 1/6, 1/4, 1/7],
    [4,  2/3,    1,  3/2, 1/2, 1/2, 1/3, 1/5, 2/3, 1/6],
    [4,    2,  2/3,   1,  1/2, 1/3, 1/4, 1/5, 1/3, 1/6],
    [5,    3,    2,    2,   1,  1/2, 1/3, 1/4, 1/2, 1/5],
    [6,    4,    2,  3/2,   2,   1,  1/2, 1/3, 1/3, 1/4],
    [4,    3,  3/2,    2,   3,   2,    1,  1/3,  1/2, 1/3],
    [7,    6,    5,    5,   4,   3,    3,    1,  1/2, 1/3],
    [6,    4,  3/2,    3,   2,  3/2,  2,  2/3,    1,  1/4],
    [8,    7,    6,    6,   5,   4,    3,    3,    4,    1]
])



print(header)
# 计算AHP权重
ahp = generate_ahp_weights(comparison_matrix1)
# 计算信息熵
entropy = calculate_entropy(old_data)
# 计算区分度和独立度并结合起来得到最终权重
w = calculate_weights(old_data)
# 初始化P矩阵，形状与old_data相同
P = np.zeros_like(old_data, dtype=float)
print(ahp)
print(entropy)
print(w)
# 计算P矩阵
for i, row in enumerate(old_data):
    for j, x in enumerate(row):
        r = calculate_privacy_coefficient(old_data, [x], [j])
        P[i, j] = get_pij(r, w[j], ahp[j], entropy[j])
P_row_sums = P.sum(axis=1, keepdims=True)
P_normalized = P / P_row_sums
Q=P_normalized.sum(axis=0)
Q=Q/Q.sum()
print(Q)