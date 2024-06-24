import numpy as np


def expected(probabilities):
    expectations = []

    for prob in probabilities:

        n = len(prob)
        expectation = n
        prob_same = 1
        for i in range(n):
            prob_same *= (1 - prob[i])

            if prob_same < 0.95:
                break
            expectation -= 1
        expectations.append(expectation)

    return expectations


def calculate_probability_matrix(a):
    n, m = a.shape
    b = np.zeros((n, n))
    bb = np.zeros((n, n))
    for i in range(n):
        # 计算第 i 行向量与所有行向量的乘积
        row_products = np.dot(a[i], a.T)
        # 将结果按从大到小排序
        sorted_indices = np.argsort(row_products)[::1]
        sorted_products = row_products[sorted_indices]
        # 填入矩阵 b 的第 i 行
        b[i] = sorted_products
        bb[i] = sorted_indices
    return b, bb


def get_p_entropy(old_data, old_data1, beta, max_class):
    def entropy(prob_dist):
        epsilon = 1e-10  # 防止对数零值的微小正数
        prob_dist = np.clip(prob_dist, epsilon, None)  # 避免零概率
        return -np.sum(prob_dist * np.log(prob_dist))

    if beta == 0:
        return get_p(old_data1)
    noise_mapping_prob = np.zeros_like(old_data1)
    A = []
    for idy, (i, max_c) in enumerate(zip(zip(*old_data1), max_class)):
        probs = []
        for idx, km in enumerate(i):
            prob = []
            prob_i = 0
            for k in range(max_c):
                pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                if (k == old_data[idx][idy]): prob_i = pdf
                prob.append(pdf)
            prob = np.array(prob)
            probs.append(prob / sum(prob))
            noise_mapping_prob[idx][idy] = prob_i / sum(prob)
        probs = np.array(probs)  # 转换为NumPy数组
        pr2, indice = calculate_probability_matrix(probs)
        value = 0
        index = 0
        jg = []
        while value < len(i):
            current_class = np.array(expected([pr2[index]]))[0]
            jg.append(current_class)
            class_entities = indice[index][-current_class:]

            classes = []
            classes.append(class_entities)
            # 从剩余实体中移除已经分配到类的实体
            remaining_indices = np.setdiff1d(np.arange(len(pr2)), np.concatenate(classes))

            probs = probs[remaining_indices]
            pr2, indice = calculate_probability_matrix(probs)
            index = 0

            value += len(class_entities)
        jg = np.array(jg) / sum(jg)
        A.append(entropy(jg))

    return A, noise_mapping_prob


# 示例使用：
old_data1 = [
    [1, 2, 3],
    [2, 3, 3],
    [3, 1, 2],
    [3, 1, 2],
    [2, 1, 1]
] # 输入数据
old_data2 = [
    [1, 2, 3],
    [2, 3, 3],
    [3, 1, 2],
    [3, 1, 2],
    [2, 1, 1]
] # 输入数据
for i in range(len(old_data1)):
    for j in  range(len(old_data1[i])):
        old_data1[i][j]-=1
for i in range(len(old_data1)):
    for j in  range(len(old_data1[i])):
        old_data2[i][j]=old_data1[i][j]+0.1
old_data2[1][1]-=0.5
beta = 0.3  # 示例 beta 值
max_class = [3, 3, 3]  # 示例 max_class 值

classes = get_p_entropy(old_data1,old_data2, beta, max_class)
print(classes)
