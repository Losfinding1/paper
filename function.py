from collections import Counter
import  numpy as np
def frobenius_norm(matrix):
    return np.sqrt(np.sum(matrix**2))
def nr(m):
    b=frobenius_norm(m)
    k=frobenius_norm(np.ones(m.shape))
    #print(k)
    return  b/k
def get_utility(m1,m2,beta):
    m=np.abs(m1-m2)

    m=get_pp(m,beta)

    l=nr(m)
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
def noisyCount(beta):

    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    return n_value
def laplace_mech(data, beta):
    data = data.astype(float)  # 将数据转换为浮点数类型
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] += noisyCount(beta[j])
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


def get_p(old_data):
    A = []
    for i in zip(*old_data):
        b = []
        value_counts = Counter(i)
        frequency_dict = {value: count / len(i) for value, count in value_counts.items()}
        b = [frequency_dict[j] for j in i]
        b = np.array(b)  # 转换为NumPy数组
        A.append(b)
    return np.array(A).T

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
def expected(probabilities):
    expectations=[]
    for prob in probabilities:
        n = len(prob)
        #print('n',n)
        expectation = n
        prob_same=1
        for i in range(n):
            prob_same *=(1- prob[i])
            if(prob_same<0.95): break
            expectation-=1
        expectations.append(expectation)
    return expectations

def calculate_probability_matrix(a):
    n, m = a.shape
    b = np.zeros((n, n))
    bb= np.zeros((n, n))
    for i in range(n):
        # 计算第 i 行向量与所有行向量的乘积
        row_products = np.dot(a[i], a.T)
        # 将结果按从大到小排序
        sorted_indices = np.argsort(row_products)[::1]
        sorted_products = row_products[sorted_indices]
        # 填入矩阵 b 的第 i 行
        b[i] = sorted_products
        bb[i]=sorted_indices
    return b,bb
def get_p11(old_data1, beta, max_class):
    if beta == 0:
        return get_p(old_data1)

    A = []

    for i, max_c in zip(zip(*old_data1), max_class):
        frequency_dict = {k: 0 for k in range(max_c )}
        for km in i:
            x = km - int(km)
            for k in range(max_c ):
                pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                frequency_dict[k] += pdf

        # Normalize the frequencies
        total_count = sum(frequency_dict.values())
        for k in frequency_dict:
            frequency_dict[k] = frequency_dict[k]/total_count*len(i)

        b = []
        for km in i:
            probs = []
            for k in range(max_c):
                pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                probs.append(pdf * frequency_dict[k]) 
            b.append(sum(probs) / len(i))

        b = np.array(b)  # 转换为NumPy数组
        A.append(b)

    return np.array(A).T
def get_p111(old_data1, beta0, max_class):
    beta0 = np.array(beta0)
    if beta0.any() == 0:
        return get_p(old_data1)

    A = []
    for idy, (i, max_c) in enumerate(zip(zip(*old_data1), max_class)):
        probs= []
        beta=beta0[idy]
        for km in i:
            prob = []
            for k in range(max_c):
                pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                prob.append(pdf)
            prob = np.array(prob)
            probs.append(prob / sum(prob))
        probs = np.array(probs)  # 转换为NumPy数组
        pr2,_=calculate_probability_matrix(probs)
        jg=np.array(expected(pr2))

        b=jg/len(i)
        A.append(b)

    return np.array(A).T
def get_p_entropy(old_data,old_data1, beta0, max_class):
    def entropy(prob_dist):
        epsilon = 1e-10  # 防止对数零值的微小正数
        prob_dist = np.clip(prob_dist, epsilon, None)  # 避免零概率
        return -np.sum(prob_dist * np.log(prob_dist))

    beta0 = np.array(beta0)
    if beta0.any() == 0:
        return get_p(old_data1)
    noise_mapping_prob = np.zeros_like(old_data1)
    A = []
    for idy, (i, max_c) in enumerate(zip(zip(*old_data1), max_class)):
        probs = []
        beta = beta0[idy]
        for idx,km in enumerate(i):
            prob = []
            prob_i=0
            for k in range(max_c):
                pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                if(k==old_data[idx][idy]): prob_i=k
                prob.append(pdf)
            prob=np.array(prob)
            index=int(old_data[idx][idy])
            vector = np.abs(np.arange(max_c) -index )
            noise_mapping_prob[idx][idy]=1-(prob@vector)/max(vector)
            probs.append(prob/sum(prob))
        probs = np.array(probs)  # 转换为NumPy数组


        pr2, indice = calculate_probability_matrix(probs)
        value = 0
        index = 0
        jg = []
        while value < len(i):
            current_class = np.array(expected([pr2[index]]))[0]
            jg.append(current_class)
            class_entities = indice[index][ -current_class:]

            classes = []
            classes.append(class_entities)
            # 从剩余实体中移除已经分配到类的实体
            remaining_indices = np.setdiff1d(np.arange(len(pr2)), np.concatenate(classes))

            probs=probs[remaining_indices]
            pr2, indice = calculate_probability_matrix(probs )
            index = 0

            value += len(class_entities)
        jg=np.array(jg)/sum(jg)

        A.append(entropy(jg))


    return A,noise_mapping_prob
def get_entropy(old_data ):
    epsilon = 1e-10  # 一个小的正数，防止对数函数中的零值

    B = []

    for b in zip(*old_data):
        # 使用 np.clip 避免 b 中的零值
        b = np.clip(b, epsilon, None)
        entropy = -(b * np.log(b)).sum()  # 计算熵
        B.append(entropy)

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

def calculate_entropy(probabilities):
    """Calculate the entropy of a given probability distribution."""
    # Avoid log(0) by using np.where to handle zero probabilities
    probabilities = np.array(probabilities)
    non_zero_probs = probabilities[probabilities > 0]
    return -np.sum(non_zero_probs * np.log2(non_zero_probs))


def calculate_average_privacy_loss(data, j,beta0,max_class):
    """
    Calculate the average privacy loss after revealing a single attribute.

    Parameters:
    - data: 2D numpy array where each row is an entity and each column is an attribute.
    - j: Index of the attribute to be exposed.

    Returns:
    - Average privacy loss.
    """
    n = data.shape[0]
    mode=0
    beta0=np.array(beta0)
    initial_probs = np.ones(n) / n  # Assuming equal probability for each entity initially
    attribute_values = data[:, j]

    # Initial entropy (S0)
    S0 = calculate_entropy(initial_probs)

    # Get unique attribute values and their corresponding indices
    unique_values = np.arange(max_class[j])
    partitions = [np.where(attribute_values == value)[0] for value in unique_values]
    if beta0.any() == 0:
     if(mode==0):
         partition_entropies = []
         partition_probs = []

         for partition in partitions:
             partition_prob = np.sum([initial_probs[i] for i in partition])
             partition_probs.append(partition_prob)

             if partition_prob > 0:

                 partition_entropy =  (len(partition))
             else:
                 partition_entropy = 0

             partition_entropies.append(partition_entropy)

         average_privacy_loss = np.sum([
             partition_probs[i] * (partition_entropies[i])
             for i in range(len(partitions))
         ])*S0



     else:

        # Calculate partition entropies and probabilities
        partition_entropies = []
        partition_probs = []

        for partition in partitions:
            partition_prob = np.sum([initial_probs[i] for i in partition])
            partition_probs.append(partition_prob)

            if partition_prob > 0:
                conditional_probs = [initial_probs[i] / partition_prob for i in partition]

                partition_entropy = calculate_entropy(conditional_probs)
            else:
                partition_entropy = 0

            partition_entropies.append(partition_entropy)


        # Calculate the average privacy loss
        average_privacy_loss = np.sum([
            partition_probs[i] * (S0 - partition_entropies[i])
            for i in range(len(partitions))
        ])
    elif(mode==0):
        beta=beta0[j]
        probs = []
        for km in attribute_values:
            prob = []
            for k in unique_values:
                pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                prob.append(pdf)
            prob = np.array(prob)
            probs.append(prob / sum(prob))
        probs = np.array(probs)  # 转换为NumPy数组
        pp = np.sum(probs, axis=0)
        pp = pp / sum(pp)
        average_privacy_loss = 0

        for id, col in enumerate(zip(*probs)):
            col = sorted(col )
            col=np.array(col)
            '''
            pro=col@np.arange(1,len(col)+1)

            average_privacy_loss+=pp[id]*pro
         '''
            jg = np.array(expected([col]))[0]
            pro = np.ones(jg) / jg # Assuming
            average_privacy_loss += pp[id] * (jg)

        average_privacy_loss=average_privacy_loss*S0


    else:
            beta=beta0[j]
            probs = []
            for km in attribute_values:
                prob = []
                for k in unique_values:
                    pdf = (1 / (2 * beta)) * np.exp(-np.abs(k - km) / beta)
                    prob.append(pdf)
                prob = np.array(prob)
                probs.append(prob / sum(prob))
            probs = np.array(probs)  # 转换为NumPy数组
            pp = np.sum(probs, axis=0)
            pp = pp / sum(pp)
            average_privacy_loss = 0
            for id, col in enumerate(zip(*probs)):
                col = col / sum(col)
                average_privacy_loss += pp[id] * (S0-calculate_entropy(col))
    return average_privacy_loss/S0

