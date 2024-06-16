import numpy as np

def exponential_mechanism(choices, scores, sensitivity, epsilon):
    exp_scores = [np.exp((epsilon * score) / (2 * sensitivity)) for score in scores]

    total_exp_scores = sum(exp_scores)
    probabilities = [exp_score / total_exp_scores for exp_score in exp_scores]
    print(probabilities)
    return np.random.choice(choices, p=probabilities)

# 假设有10个类，小明属于第5类
num_classes = 10
original_class = 5

# 设置打分函数，原始类得分最高，其他类得分较低
scores = [0] * num_classes
scores[original_class - 1] = 1  # 第5类得分最高

# 设置敏感度和隐私预算
sensitivity = 1
epsilon = 100

# 使用指数机制添加噪声
classes = list(range(1, num_classes + 1))
noisy_class = exponential_mechanism(classes, scores, sensitivity, epsilon)

print("Original class:", original_class)
print("Noisy class:", noisy_class)
