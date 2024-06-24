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
old_data = np.genfromtxt('output1234.csv', delimiter=',', skip_header=1, dtype=int)
header = np.genfromtxt('output1234.csv', delimiter=',', dtype=str, max_rows=1)
max_class=[66, 7, 16, 7, 15, 6, 5, 2, 56, 2]
# 定义get_pij函数
def get_pij(r, w, h, e,p=1):
    return r * w * ( h) * e*p

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
p1=get_p1(old_data, 0)

for i, row in enumerate(old_data):
    for j, x in enumerate(row):
        r = p1[i][j]
        r = -np.log(r)
        P[i, j] = get_pij(r, w[j], ahp[j], entropy[j])
P_row_sums = P.sum(axis=1, keepdims=True)
P_normalized = P / P_row_sums
Q=P_normalized.sum(axis=0)
print(Q)
print(sum(Q))
average_privacy_loss=0
Q=[1/i  for i in Q]
w_laplace=Q/sum(Q)*len(Q)
#w_laplace=np.ones(len(Q))
for q in range(old_data.shape[1]):
    a=calculate_average_privacy_loss(old_data, q,[0],max_class)
    average_privacy_loss+=a
print( average_privacy_loss/old_data.shape[1])
print('------')
#-------------

beta0 = 0.5
beta=np.full(len(max_class),beta0)*w_laplace

# 给数据添加拉普拉斯噪声
noisy_data = laplace_mech(old_data, beta)
#-------------

p2=get_p111(noisy_data,beta,max_class)
entropy1,p_i=get_p_entropy(old_data,noisy_data,beta,max_class)

average_privacy_loss=0
for q in range(noisy_data.shape[1]):
    a=calculate_average_privacy_loss(noisy_data, q,beta,max_class)
    average_privacy_loss+=a
print( average_privacy_loss/noisy_data.shape[1])

for i, row in enumerate(noisy_data):
    for j, x in enumerate(row):
        r = p2[i][j]
        r = -np.log(r)
        P[i, j] = get_pij(r, w[j], ahp[j], entropy1[j],p=p_i[i][j])
P_normalized = P / P_row_sums
Q=P_normalized.sum(axis=0)
print(Q)
print(sum(Q))

'''
awsfg


print('bb', (P.sum(axis=1, keepdims=True)[0]))
P_normalized = P / P_row_sums
Q=P_normalized.sum(axis=0)
print(Q.sum()) 
print(p1[ :,0][:10])
print(pp1[ :,0][:10])
print(get_p1(old_data,0)[:10,0])
print( (get_p111(noisy_data,beta,max_class) )[:10,0])
print(entropy)
print(get_p_entropy(old_data,noisy_data,beta,max_class))

print("原始数据:")
print(old_data)
print("添加噪声后的数据:")
print(noisy_data)
privacy_loss=[ calculate_average_privacy_loss(old_data,j) for j in  range( (old_data.shape[1]))]
print(privacy_loss )
print(Q)
 
print(get_entropy(get_p1(old_data,0)))
print(get_entropy(get_p1(noisy_data,beta) ))
print(get_entropy(get_p11(noisy_data,beta,max_class) ))
print(get_p1(old_data,0)[:10,0])
print(get_p1(old_data,0)[:,0].sum())
print(get_p1(old_data,beta)[:10,0])
print(get_p1(old_data,beta)[:,0].sum())
print(get_p11(noisy_data,beta,max_class)[:10,0])
print(get_p11(noisy_data,beta,max_class)[:,0].sum())
'''