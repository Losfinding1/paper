from collections import Counter
import  numpy as np


def calculate_combined_weights(w, p_final):
    """
    计算结合客观权重和主观权重的最终权重。

    参数：
    w -- numpy 数组，表示客观权重
    p_final -- numpy 数组，表示主观权重

    返回值：
    w_final -- numpy 数组，表示结合后的最终权重
    """

    # 计算 α 和 β
    def Co(w, p_final):
        return np.sqrt(np.sum((w - p_final) ** 2) / (2 * len(w)))

    Co_value = Co(w, p_final)
    alpha = Co_value / (1 + Co_value)
    beta = 1 - alpha

    # 计算最终权重 w_final
    w_final = alpha * w + beta * p_final
    return w_final
def calculate_privacy_coefficient(data, values, indices):
    """
    计算数据矩阵中符合指定条件的行数，并基于此计算隐私系数 r。

    参数：
    data -- numpy 数组，表示数据矩阵
    values -- 数值数组，表示需要匹配的值
    indices -- 索引数组，表示需要匹配的列

    返回值：
    r -- 浮点数，表示隐私系数
    """
    if len(values) != len(indices):
        raise ValueError("Values and indices must have the same length.")

    # 找到所有符合条件的行
    mask = np.all(data[:, indices] == values, axis=1)
    count = np.sum(mask)

    # 计算隐私系数
    r = 1 / count if count > 0 else 0
    return r
import numpy as np
def calculate_entropy(data):
    def entropy(prob_dist):
        epsilon = 1e-10  # 防止对数零值的微小正数
        prob_dist = np.clip(prob_dist, epsilon, None)  # 避免零概率
        return -np.sum(prob_dist * np.log(prob_dist))

    num_features = data.shape[1]
    entropies = np.zeros(num_features)

    for j in range(num_features):
        values, counts = np.unique(data[:, j], return_counts=True)  # 计算每个值的出现次数
        prob_dist = counts / counts.sum()  # 归一化为概率分布
        unique_prob_dist=prob_dist
        entropies[j] = entropy(unique_prob_dist)  # 计算信息熵

    return entropies

def calculate_variation_coefficient(data):
    """
    计算区分度（Variation Coefficient）。

    参数：
    data -- numpy 数组，表示数据矩阵

    返回值：
    vc -- numpy 数组，表示每个指标的区分度
    """
    mean = np.mean(data, axis=0)  # 计算均值
    std = np.std(data, axis=0, ddof=0)  # 计算标准差
    vc = std / mean  # 计算区分度
    return vc

def calculate_independence(data):
    """
    计算独立度（Independence）。

    参数：
    data -- numpy 数组，表示数据矩阵

    返回值：
    independence -- numpy 数组，表示每个指标的独立度
    """
    corr_matrix = np.corrcoef(data, rowvar=False)  # 计算相关矩阵
    pinv_corr_matrix = np.linalg.pinv(corr_matrix)  # 计算伪逆相关矩阵
    independence = np.sum(pinv_corr_matrix, axis=0)  # 每列求和得到独立度
    return independence

def calculate_weights(data):
    """
    计算最终的指标权重 w_j。

    参数：
    data -- numpy 数组，表示数据矩阵

    返回值：
    weights -- numpy 数组，表示每个指标的权重
    """
    vc = calculate_variation_coefficient(data)  # 计算区分度
    independence = calculate_independence(data)  # 计算独立度



    weights = vc * independence  # 结合区分度和独立度
    weights /= np.sum(weights)  # 归一化权重
    return weights


