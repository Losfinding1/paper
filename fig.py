import  numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

def figure0(indicators, y1,y2,y3):

    x = range(len(indicators))
    bar_width = 0.25
    x_2 = [i + bar_width for i in x]
    x_3 = [i + bar_width for i in x_2]
    # 绘制四组柱状图
    plt.bar(x, y1, width=bar_width, label='原始熵')
    plt.bar(x_2, y2, width=bar_width, label='背景知识攻击后熵')
    plt.bar(x_3, y3, width=bar_width,label='加噪保护后受到攻击的熵')
    # 设置x轴标签和标题
    plt.xlabel('时间')
    plt.ylabel('熵值')
    plt.title('不同时间下各系统熵值变化')

    # 设置x轴刻度和标签
    plt.xticks([i + 1.5 * bar_width for i in x], indicators)

    # 添加图例
    plt.legend()

    plt.show()
def figure(x,y,name):
    ff = [np.mean(k, axis=0) for k in y]
    y1=[i[0] for i in y]
    y2 = [i[4] for i in y]
    y3=[i[9] for i in y]

    plt.plot(x, ff,label="10节点平均")
    plt.plot(x, y1, label="时刻1")
    plt.plot(x, y2, label="时刻5")
    plt.plot(x, y3,label="时刻10")
    plt.title(name+"随隐私保护程度的变化")
    plt.xlabel("隐私保护程度（噪声水平）")
    plt.ylabel(name)
    plt.legend()
    plt.show()

def figure1(x1, y, name):
        x=range(len(x1))

        plt.plot(x, y[0], label="噪声水平"+str(x1[0]))
        plt.plot(x, y[4], label="噪声水平"+str(x1[4]))
        plt.plot(x, y[9], label="噪声水平"+str(x1[9]))
        plt.title('不同噪声水平下熵随时刻变化')
        plt.xlabel("时刻")
        plt.ylabel(name)
        plt.legend()
        plt.show()


def figure2(indicators, y1,y2,y3,y4):

    x = range(len(indicators))
    bar_width = 0.2
    x_2 = [i + bar_width for i in x]
    x_3 = [i + bar_width for i in x_2]
    x_4 = [i + bar_width for i in x_3]

    # 绘制四组柱状图
    plt.bar(x, y1, width=bar_width, label='ahp赋权')
    plt.bar(x_2, y2, width=bar_width, label='时刻1动态权重')
    plt.bar(x_3, y3, width=bar_width,label='时刻5动态权重')
    plt.bar(x_4, y4, width=bar_width, label='时刻10动态权重')

    # 设置x轴标签和标题
    plt.xlabel('指标')
    plt.ylabel('数值')
    plt.title('每个指标有四个柱子的柱状图')

    # 设置x轴刻度和标签
    plt.xticks([i + 1.5 * bar_width for i in x], indicators)

    # 添加图例
    plt.legend()
    plt.title("权重分布图")
    plt.xlabel("指标")
    plt.ylabel("权重")
    plt.legend()
    plt.show()