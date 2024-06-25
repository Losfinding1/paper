import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False         # 解决图中负号显示问题

# 导入自定义函数
from function import *
from function1 import *
from fig import *

# 读取数据

header = np.genfromtxt('output1234.csv', delimiter=',', dtype=str, max_rows=1)

# 定义每列的最大类别数量
max_class = [66, 7, 16, 7, 15, 6, 5, 2, 56, 2]

# 定义get_pij函数
def get_pij(r, w, h, e, p=1):
    return r * w * h * e * p
def process_data(data, max_class, weights, ahp_weights, entropy_values, get_p_func):
    # 初始化P矩阵，形状与old_data相同
    P = np.zeros_like(data, dtype=float)

    # 计算概率
    p = get_p_func(data, 0)

    # 计算P矩阵
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            r = p[i][j]
            r = -np.log(r)
            P[i, j] = get_pij(r, weights[j], ahp_weights[j], entropy_values[j])

    # 归一化P矩阵
    P_row_sums = P.sum(axis=1, keepdims=True)
    P_normalized = P / P_row_sums

    # 计算Q值
    Q = P_normalized.sum(axis=0)


    # 计算平均隐私损失
    average_privacy_loss = 0
    for q in range(data.shape[1]):
        a = calculate_average_privacy_loss(data, q, [0], max_class)
        average_privacy_loss += a



    return Q, average_privacy_loss / data.shape[1], P_row_sums
def process_noisy_data(noisy_data, old_data, max_class, weights, ahp_weights, P_row_sums, beta):
    # 计算噪声数据的概率p2
    p2 = get_p111(noisy_data, beta, max_class)

    # 计算噪声数据的信息熵和p_i
    entropy1, p_i = get_p_entropy(old_data, noisy_data, beta, max_class)

    # 计算噪声数据的平均隐私损失
    average_privacy_loss_noisy = 0
    for q in range(noisy_data.shape[1]):
        a = calculate_average_privacy_loss(noisy_data, q, beta, max_class)
        average_privacy_loss_noisy += a


    # 重新计算P矩阵
    P = np.zeros_like(noisy_data, dtype=float)
    for i, row in enumerate(noisy_data):
        for j, x in enumerate(row):
            r = p2[i][j]
            r = -np.log(r)
            P[i, j] = get_pij(r, weights[j], ahp_weights[j], entropy1[j], p=p_i[i][j])

    # 归一化P矩阵
    P_normalized = P / P_row_sums
    Q = P_normalized.sum(axis=0)


    return Q, average_privacy_loss_noisy / noisy_data.shape[1]

def add_laplace_noise(data,  Q, mode, beta0=0.5,):
    if(mode==0):
        w_laplace = np.full(len(max_class),1)
    if(mode==1):
        entropy = calculate_entropy(old_data)
        Q_inv = np.array([ 1/i for i in entropy])
        w_laplace = Q_inv / sum(Q_inv) * len(Q_inv)
    if(mode==2):
        Q_inv = np.array([1 / i for i in max_class])
        w_laplace = Q_inv / sum(Q_inv) * len(Q_inv)
    if(mode==3):
        Q_inv = [1 / i for i in Q]
        w_laplace = Q_inv / sum(Q_inv) * len(Q_inv)

    beta = np.full(len(max_class), beta0) * w_laplace
    noisy_data = laplace_mech(data, beta)
    return noisy_data, beta
# 层次分析法比较矩阵
comparison_matrix1 = np.array([
    [1, 1/3, 1/4, 1/4, 1/5, 1/6, 1/4, 1/7, 1/4, 1/8],
    [3, 1, 1/2, 1/2, 1/3, 1/4, 1/3, 1/6, 1/2, 1/7],
    [4, 2, 1, 3/2, 1/2, 1/2, 1/3, 1/5, 3/2, 1/6],
    [4, 2, 2/3, 1, 1/2, 1/3, 1/4, 1/5, 1/2, 1/6],
    [5, 3, 2, 2, 1, 1/2, 1/3, 1/4, 1/2, 1/5],
    [6, 4, 2, 3/2, 2, 1, 1/2, 1/3, 3/2, 1/4],
    [4, 3, 3/2, 2, 3, 2, 1, 1/3, 1/2, 1/3],
    [7, 6, 5, 5, 4, 3, 3, 1, 1/2, 1/3],
    [4, 2, 2/3, 3/2, 2, 2/3, 2, 2, 1, 1/4],
    [8, 7, 6, 6, 5, 4, 3, 3, 4, 1]
])
comparison_matrix2 = np.array([
    [1, 0.316, 0.25, 0.25, 0.2, 0.167, 0.25, 0.143*1.1, 0.25, 0.125*1.1],
    [2.86, 1, 0.5, 0.5, 0.4, 0.286, 0.4, 0.286*1.1, 0.5, 0.143*1.1],
    [4, 2, 1, 1.5, 0.5, 0.5, 0.333, 0.4*1.1, 1.5, 0.167*1.1],
    [4, 2.4, 0.667, 1, 0.5, 0.333, 0.333, 0.4*1.1, 0.5, 0.167*1.1],
    [5, 3, 2, 2, 1, 0.5, 0.333, 0.333*1.1, 0.5, 0.2*1.1],
    [6, 4, 2, 1.5, 2, 1, 0.5, 0.4*1.1, 1.5, 0.333*1.1],
    [4, 3, 1.5, 2, 3, 2, 1, 0.4*1.1, 0.5, 0.333*1.1],
    [7, 6, 5, 5, 4, 3, 3, 1, 0.667, 0.4],
    [4, 2, 0.667, 1.5, 2, 0.667, 2, 2, 1, 0.333],
    [8, 7, 6, 6, 5, 4, 3, 3, 4, 1]
])



yshl=[]
yssl=[]
for i in range(1,2):
    filepath='outputs/output'+str(i)+'.csv'
    old_data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=int)
    #print("数据头部:", header)

    # 计算AHP权重
    ahp = generate_ahp_weights(comparison_matrix2)
    #print("AHP权重:", ahp)

    # 计算信息熵
    entropy = calculate_entropy(old_data)
   # print("信息熵:", entropy)

    # 计算区分度和独立度并结合起来得到最终权重
    w = calculate_weights(old_data)
    #print("权重:", w)

    Q, avg_privacy_loss, P_row_sums = process_data(old_data, max_class, w, ahp, entropy, get_p1)
    # print("平均隐私损失:", avg_privacy_loss)
    # print(sum(Q))
    # print('------')
    hl=[]
    sl=[]
    hl.append(sum(Q))
    sl.append(avg_privacy_loss)
    for ss in range(4):
        noisy_data, beta = add_laplace_noise(old_data, Q,mode=ss, beta0=0.5)
        Q_noisy, avg_privacy_loss_noisy = process_noisy_data(noisy_data, old_data, max_class, w, ahp,
                                                              P_row_sums, beta)
        hl.append(sum(Q_noisy))
        sl.append(avg_privacy_loss_noisy)
    # print("平均隐私损失:", avg_privacy_loss_noisy)
    # print(sum(Q_noisy))



    yshl.append(hl)
    yssl.append(sl)
    print('---------------------------------------------------------------------------------')
print(yshl)
print(yssl)






