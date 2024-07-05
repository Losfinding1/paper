import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

# 生成数据分布散点图 以及输出数据
plt.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体：解决plot不能显示中文问题
plt.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size'] = 12
header1=['Age','Workclass' ,'Education' ,'Marital\nStatus', 'Occupation',
 'Relation\nship' ,'Race' ,'Sex' ,'Hours\nper week' ,'Target']
label=['正常加噪','类别加噪','隐私加噪']
def figure_fenbutu(filepath,mode=-1,bet=-1):
    index = []
    expert = []

    header = np.genfromtxt(filepath, delimiter=',', dtype=str, max_rows=1)
    old_data = pd.read_csv(filepath, skiprows=1, header=None)
    old_data.columns = header  # 将第一行作为列名
    mean_vals =old_data.median(axis=0)
    # 归一化平均值

    min_vals = old_data.min(axis=0)
    max_vals = old_data.max(axis=0)
    rows, cols = old_data.shape
    norm_mean_vals =  (mean_vals - min_vals) / (max_vals - min_vals)
    old_data = (old_data - min_vals) / (max_vals - min_vals)

    for i in range(rows):
        expert.append('expert' + str(i + 1))
    for k in range(cols):
        index.append('index' + str(k + 1))

    # 生成横坐标、纵坐标和颜色值
    x = np.tile(np.arange(cols), rows)  # 横坐标为列数，重复 rows 次
    y = old_data.values.flatten()  # 纵坐标为矩阵元素的值（展开为一维数组）

    # 使用高斯核密度估计计算每个点的密度
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # 设置归一化器，将密度值映射到[0, 1]范围内


    plt.figure(figsize=(16, 9))
    # 绘制散点图，颜色根据密度调整，点的大小设为20
    plt.scatter(x, y, c=z, cmap=plt.cm.Spectral_r, s=20)

    # 添加颜色条
    plt.colorbar(label='密度')
    plt.xticks(range(len(header)), header1,fontsize=12)
    plt.xlabel('指标')
    plt.ylabel('类')
    ti='数据类分布图'
    if(mode>=0): ti='噪声尺度'+str(bet)+'下'+label[mode]+ti
    else: ti='原始'+ti
    plt.title(ti)
    title_text = plt.gca().get_title()


    plt.savefig('picture1/' + title_text + "0.png", dpi=300)
    #plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x, y, c=z, cmap=plt.cm.Spectral, s=20, norm=norm)
    # plt.xlabel('Index number')
    # plt.ylabel('degree')
    # plt.title('Experts Scoring Distribution Map of System')
    #
    # plt.colorbar(label='Density')
    #
    # plt.show()
filepath = 'outputs/output' + str(0) + '.csv'
bet=[0.1,0.2,0.3,0.4,0.5]
figure_fenbutu(filepath)
for i in range(4):
    for ss in range(3):
        filepath1 = 'data/noisydata' + str(ss) + '_' + str(bet[i]) + '.csv'
        figure_fenbutu(filepath1,mode=ss,bet=bet[i])
