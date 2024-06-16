from collections import Counter
import  numpy as np
def frobenius_norm(matrix):
    return np.sqrt(np.sum(matrix**2))
def n(m):
    b=frobenius_norm(m)
    k=frobenius_norm(np.ones(m.shape))
    #print(k)
    return  b/k
def get_utility(m1,m2,beta):
    m=np.abs(m1-m2)

    m=get_pp(m,beta)

    l=n(m)
    return  l
def calculate_ahp_weights(matrix):
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 提取最大特征值对应的特征向量
    max_index = np.argmax(eigenvalues)
    principal_eigenvector = eigenvectors[:, max_index].real
    # 归一化得到权重
    weights = principal_eigenvector / np.sum(principal_eigenvector)
    return weights
def consistency_ratio(matrix, weights):
    # 计算一致性指标CI
    n = matrix.shape[0]
    lamda_max = np.sum(np.dot(matrix, weights) / weights) / n
    CI = (lamda_max - n) / (n - 1)

    # 随机一致性指标RI（来源：Saaty的RI值表）
    RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

    # 计算一致性比率CR
    CR = CI / RI[n - 1] if n - 1 < len(RI) else CI
    return CR, CI, lamda_max
def generate_ahp_weights(matrix):
    # 计算权重
    weights = calculate_ahp_weights(matrix)
    # 计算CR值
    CR, CI, lamda_max = consistency_ratio(matrix, weights)
    if CR < 0.1:
        return weights
    else:
        raise ValueError("一致性比率CR过高，需要重新输入判断矩阵。")
def laplace_range(a, beta):
    # 计算累积分布函数两端的概率
    lower_prob = (1 - a) / 2
    upper_prob = 1 - lower_prob

    # 计算拉普拉斯分布的逆CDF
    lower_bound = -beta * np.log(2 * lower_prob)
    upper_bound = beta * np.log(2 * upper_prob)

    return lower_bound, upper_bound
def filter_matrix1(data, j,  a, beta):

    lower_bound, upper_bound = laplace_range(a, beta)
    frequency_dict = {}
    i=data[:,j]

    for km in i:
        x = km - int(km)
        pdf_left = (1 / (2 * beta)) * np.exp(-np.abs(x) / beta)
        pdf_right = (1 / (2 * beta)) * np.exp(-np.abs(1 - x) / beta)
        left = (pdf_left) / (pdf_left + pdf_right)
        right = 1 - left

        if (int(km) in frequency_dict):
            frequency_dict[int(km)] += left/len(i)
        else:
            frequency_dict[int(km)] = left/len(i)
        if (int(km) + 1 in frequency_dict):
            frequency_dict[int(km) + 1] += right/len(i)
        else:
            frequency_dict[int(km) + 1] = right/len(i)
        S=[]
        for key in frequency_dict.keys():
            s=[]
            low= key - lower_bound
            up= key + upper_bound
            # 检查矩阵A的第j列中哪些行的值在(lower_bound, upper_bound)之间
            for row in data:
                if low <= row[j] <= up:
                    s.append(row)

            S.append(s)
            qw=list( frequency_dict.values())

    return S,qw
def filter_matrix(data, i):
    # 筛选满足条件的行
    #filtered_rows = [row for row in data if row[j] ==p]
    #return np.array(filtered_rows)
    unique_values, counts = np.unique(data[:, i], return_counts=True)
    counts1=[i/data.shape[0] for i  in counts]
    grouped_matrices = [data[data[:, i] == value] for value in unique_values]
    return grouped_matrices,counts1
def noisyCount(sensitivety, epsilon):
    beta =   epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    return n_value
def laplace_mech(data, sensitivety, epsilon):
    data = data.astype(float)  # 将数据转换为浮点数类型
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] += noisyCount(sensitivety, epsilon)
            #data[i][j] = max(0, min(data[i][j], 1))  # 应用阈值
    return data
def get_p(old_data):
    A = []
    for i in zip(*old_data):
        b = []
        value_counts = Counter(i)
        # 打印结果
        frequency_dict = {value: count / len(i) for value, count in value_counts.items()}
        #b=list(frequency_dict.values())
        b = [frequency_dict[j] for j in i]
        b = np.array(b)  # 转换为NumPy数组
        A.append(b)

    return  np.array(A).T


def get_pp(old_data1,beta):
    A=[]

    for i in old_data1:
        b=[]
        for x in i:
            pdf_left = (1 / (2 * beta)) * np.exp(-np.abs(x) / beta)
            pdf_right = (1 / (2 * beta)) * np.exp(-np.abs(1 - x) / beta)

            left = (pdf_left) / (pdf_left + pdf_right)
            b.append(left)
        A.append(b)

    return np.array(A)
def get_p1(old_data1,beta):
    if(beta==0): return get_p(old_data1)
    A = []

    for i in zip(*old_data1):
      frequency_dict = {}
      for  km in i:
        x=km-int(km)
        pdf_left= (1 / (2 * beta)) * np.exp(-np.abs(x) / beta)
        pdf_right = (1 / (2 * beta)) * np.exp(-np.abs(1-x) / beta)
        left=(pdf_left)/(pdf_left+pdf_right)
        right=1-left
        if(int(km) in frequency_dict):
         frequency_dict[int(km)]+=left
        else:  frequency_dict[int(km)]=left
        if (int(km)+1 in frequency_dict):
            frequency_dict[int(km)+1] += right
        else:
            frequency_dict[int(km)+1] = right
      b = []
      for km in i:
          x = km - int(km)
          pdf_left = (1 / (2 * beta)) * np.exp(-np.abs(x) / beta)
          pdf_right = (1 / (2 * beta)) * np.exp(-np.abs(1 - x) / beta)
          left = (pdf_left) / (pdf_left + pdf_right)
          right = 1 - left
          b.append((left*frequency_dict[int(km)]+right*frequency_dict[int(km+1)])/len(i))

      b = np.array(b)  # 转换为NumPy数组

      A.append(b)

    return np.array(A).T
def get_entropy(old_data1,w):
    epsilon = 1e-10  # 一个小的正数，防止对数函数中的零值
    old_data=get_p(old_data1)
    B = []

    for b in zip(*old_data):
        # 使用 np.clip 避免 b 中的零值
        b = np.clip(b, epsilon, None)
        entropy = -(b * np.log(b)).sum()  # 计算熵
        B.append(entropy)
    B=B*w

    return np.array(B).sum()
def get_entropy1(old_data1,w):
 epsilon = 1e-10  # 一个小的正数，防止对数函数中的零值
 sum0=0

 for m,old_data2 in enumerate(old_data1[0]):

    old_data=get_p(old_data2)
    B = []

    for b in zip(*old_data):
        # 使用 np.clip 避免 b 中的零值
        b = np.clip(b, epsilon, None)
        entropy = -(b * np.log(b)).sum()  # 计算熵
        B.append(entropy)
    B=B*w
    sum0=sum0+np.array(B).sum()*old_data1[1][m]
 return  sum0
def get_entropy2(old_data1,beta,w):
 epsilon = 1e-10  # 一个小的正数，防止对数函数中的零值
 sum0=0

 old_data=get_p1(old_data1 ,beta)
 B = []

 for b in zip(*old_data):
     # 使用 np.clip 避免 b 中的零值
     b = np.clip(b, epsilon, None)
     entropy = -(b * np.log(b)).sum()  # 计算熵
     B.append(entropy)
 B = B * w

 return np.array(B).sum()
def get_entropy3(old_data1,j,a,beta,w):
    epsilon = 1e-10  # 一个小的正数，防止对数函数中的零值
    sum0 = 0
    for m, old_data2 in enumerate(old_data1[0]):

        old_data = get_p1(old_data2,beta)
        B = []
        for b in zip(*old_data):
            # 使用 np.clip 避免 b 中的零值
            b = np.clip(b, epsilon, None)
            entropy = -(b * np.log(b)).sum()  # 计算熵
            B.append(entropy)
        B = B * w
        sum0 = sum0 + np.array(B).sum() * old_data1[1][m]
    return sum0
def get_quank(old_data):
    # 差异系数
        data = get_p(old_data)
        mean = np.mean(data, axis=0)  # 计算平均值
        std = np.std(data, axis=0, ddof=0)  # 计算标准差
        cv = std / mean
        w = cv / cv.sum()
        return w
def get_quanz(old_data):
    # 差异系数
    num_cols = old_data.shape[1]
    # 创建一个随机向量
    random_vector = np.random.choice([0, 0.1, 0.2, 0.3], size=num_cols)
    return random_vector
def get_quan(old_data):
    w=get_quanz(old_data)*get_quanz(old_data)
    w=w/w.sum()
    return  w