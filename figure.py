import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pylab import mpl



'''
[[999.9999999999993, 846.5055789670716, 724.8327986858405, 783.0011604495818, 880.5027551170303], [999.9999999999993, 453.1317901998822, 437.0398950009458, 512.242603283388, 843.0317568452442], [999.9999999999993, 319.6954901393189, 228.49435533907155, 363.2271885449567, 742.4030056161998], [999.9999999999993, 229.71256624563176, 139.43258675762704, 245.96554813718774, 680.411003402058]]
[[358.1002000000001, 379.00095448293405, 381.21923685177205, 369.46011735093396, 454.2945134665015], [358.1002000000001, 581.7550844871702, 495.1392702114258, 419.4161320456666, 545.7652485777663], [358.1002000000001, 639.8940363208452, 576.7337395274392, 472.07444309746717, 582.6990159556022], [358.1002000000001, 685.7579058057432, 689.760332412295, 554.3801307204083, 612.595220026265]]
[[358.1002     320.8264224  276.32020636 289.28770063 400.00757074]
 [358.1002     263.61172289 216.39561466 214.84281134 460.09743633]
 [358.1002     204.57123758 131.78040402 171.47027275 432.59750082]
 [358.1002     157.52720837  96.17506739 136.35841273 416.81652834]]

[[999.9999999999993, 451.37690276214323, 432.7773219156546, 508.1111646493535, 841.7944693822831], [2000.0, 846.4152584257685, 767.4122307924091, 952.3150254400834, 1655.3783501653159], [2999.9999999999973, 1213.053976128911, 1151.0865113571563, 1423.8966638406584, 2457.5154754100236], [4000.0000000000045, 1614.1109015080403, 1481.503924530704, 1911.2904526050368, 3233.3046434335424]]
[[358.1002000000001, 581.5626206943496, 495.94396232777734, 420.4322096399989, 543.9473906379051], [718.3332500000001, 1186.224739476657, 1019.951186380642, 857.6174384243684, 1108.771145325946], [1071.7616333333335, 1782.9396913594733, 1537.5976021211397, 1286.449983630086, 1669.8398540792332], [1430.1618750000005, 2393.673918410598, 2061.8725802199233, 1723.6664743338256, 2234.6565208051056]]
[[ 358.1002      262.50393449  214.63329984  213.6262997   457.89190507]
 [ 718.33325     502.01935971  391.36150762  408.36098635  917.71787463]
 [1071.76163333  720.93402727  589.96928657  610.5906133  1367.88576095]
 [1430.161875    965.91379159  763.66807987  823.60681897 1806.3313263 ]]
 
 '''
# 输出比较实验的各图
TNR = {'fontname': 'Times New Roman'}
mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894", "#86BCB6",
          "#E15759", "#E19D9A"]

hl = [[999.9999999999993, 375.1524967427027, 322.9473186263375, 439.91134693813393, 809.2606074179779],
      [999.9999999999993, 172.4825678288234, 93.84185284150496, 194.5989596843499, 618.5133738646838],
      [999.9999999999993, 105.5708242898979, 50.16838152151831, 118.19069969767241, 471.9983404765742],
      [999.9999999999993, 63.26768206063693, 30.533950952662273, 33.53924518259269, 382.23789048758204]]
xl = [[0.23711040346982962, 0.20822837449763285, 0.19881368417791886, 0.20022906846913552, 0.21635824653866553],
      [0.23711040346982962, 0.15939977131970465, 0.15349819633977005, 0.16732164381505457, 0.19425664561998052],
      [0.23711040346982962, 0.1280973313767197, 0.12272040820567483, 0.14242664387391452, 0.17667863625667116],
      [0.23711040346982962, 0.10776820690688582, 0.1007736871154444, 0.12404106883463492, 0.16233770361685032]]
hl1=[[999.9999999999993, 170.00860020625242, 99.14280504653728, 191.74305737845225, 599.0318270020848], [2000.0, 306.37154782630284, 171.86612724084608, 343.11305783759065, 1161.7049410601614], [2999.9999999999973, 432.87286926402226, 232.45524922889544, 518.2395945385817, 1623.3780402585135], [4000.0000000000045, 554.8309245013022, 292.71985276097286, 674.991906127216, 2120.6085115870465]]
xl1=[[0.23711040346982962, 0.15927944202052952, 0.15312438031101624, 0.16736777476666082, 0.19446824234770088], [0.214477766682788, 0.1442918227495029, 0.13920403634504877, 0.15213147177124547, 0.1758429043696826], [0.20444629277725745, 0.1384248526833891, 0.13284211137091143, 0.1450906969666505, 0.1676108095284565], [0.19730644217917678, 0.13362957160216873, 0.12862411805893242, 0.140382878861077, 0.16188009140952256]]
# hl1=[[999.9999999999993, 378.00549833109807, 341.81676107286467, 439.64002692361566, 814.5485934199312], [2000.0, 688.7351686791379, 594.0923005872694, 832.2321729766404, 1534.6588918597176], [2999.9999999999973, 992.6144983408004, 833.0622241138681, 1105.4651805339618, 2262.4914507957133], [4000.0000000000045, 1286.9002191427192, 975.8397972425894, 1542.4204707373306, 2996.7299506347563]]
# xl1=[[0.23711040346982962, 0.20898029506796517, 0.19859014245503842, 0.19997893072423484, 0.2166432758364329], [0.214477766682788, 0.18921133081756952, 0.1803417542915865, 0.18130672268799236, 0.19599066390217781], [0.20444629277725745, 0.18075731035432713, 0.17211054774052043, 0.17340658475566814, 0.1870054493365237], [0.19730644217917678, 0.17418109912386884, 0.1664551316948601, 0.16726494131540554, 0.18071052477971727]]
bet = [0.1, 0.2, 0.3, 0.4, 0.5]
def figure4():
    bet = [0.25, 0.5, 0.75, 1.0]
    plt.figure()

    y1 = np.array([hl[0][0], hl[1][0], hl[2][0], hl[3][0]]).round(4)
    y3 = np.array([hl[0][1], hl[1][1], hl[2][1], hl[3][1]]).round(4)
    y2 = np.array([hl[0][-1], hl[1][-1], hl[2][-1], hl[3][-1]]).round(4)

    y4 = (np.array([xl[0][0], xl[1][0], xl[2][0], xl[3][0]]) * 100).round(2)
    y6 = (np.array([xl[0][1], xl[1][1], xl[2][1], xl[3][1]]) * 100).round(2)
    y5 = (np.array([xl[0][-1], xl[1][-1], xl[2][-1], xl[3][-1]]) * 100).round(2)

    bar_width = 0.25
    x2 = np.array([i - bar_width for i in range(len(bet))])
    x3 = np.array([i + bar_width for i in range(len(bet))])

    fig, ax1 = plt.subplots(figsize=(16, 9))

    ax1.bar(x2, y1, width=bar_width, label='原始数据', color='white', edgecolor='black')
    ax1.bar(range(len(bet)), y2, width=bar_width, label='隐私加噪', hatch='...', color='white', edgecolor=colors[2])
    ax1.bar(x3, y3, width=bar_width, label='正常加噪', hatch='', color=colors[9], edgecolor=colors[8])

    for i in range(len(bet)):
        ax1.text(i - bar_width, y1[i], str(y1[i]), ha='center', fontsize=12, **TNR)
        ax1.text(i, y2[i], str(y2[i]), ha='center', fontsize=12, **TNR)
        ax1.text(i + bar_width, y3[i], str(y3[i]), ha='center', fontsize=12, **TNR)

    ax1.set_xlabel('噪声水平', fontsize=16)
    ax1.set_ylabel('隐私含量', fontsize=16)
    ax1.set_xticks(range(len(bet)))
    ax1.set_xticklabels(bet, fontsize=16)
    ax1.set_ylim(0, max(y1.max(), y2.max(), y3.max()) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(x2, y4, color='gray', marker='o', linestyle='dotted', linewidth=2, label='原始数据')
    ax2.plot(range(len(bet)), y5, color=colors[2], marker='s', linestyle='--', linewidth=2, label='隐私加噪')
    ax2.plot(x3, y6, color=colors[8], marker='^', linestyle='-', linewidth=2, label='正常加噪')

    ax2.set_ylabel('隐私泄露程度 (%)', fontsize=16)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=6, loc='upper left', fontsize=16)

    for i in range(len(bet)):
        ax2.text(i - bar_width, y4[i], str(y4[i]), ha='center', fontsize=12, **TNR)
        ax2.text(i, y5[i], str(y5[i]), ha='center', fontsize=12, **TNR)
        ax2.text(i + bar_width, y6[i], str(y6[i]), ha='center', fontsize=12, **TNR)

    plt.title('不同噪声水平下模型隐私含量和隐私泄露程度对比', fontsize=20)
    title_text = plt.gca().get_title()
    plt.savefig('picture/' + title_text + "0.png", dpi=300)
    plt.show()
def figure5():
    bet = [1000, 2000, 3000  ,4000]
    plt.figure()

    y1 = np.array([hl1[0][0], hl1[1][0], hl1[2][0], hl1[3][0]]).round(4)
    y3 = np.array([hl1[0][1], hl1[1][1], hl1[2][1], hl1[3][1]]).round(4)
    y2 = np.array([hl1[0][-1], hl1[1][-1], hl1[2][-1], hl1[3][-1]]).round(4)

    y4 = (np.array([xl1[0][0], xl1[1][0], xl1[2][0], xl1[3][0]]) * 100).round(2)
    y6 = (np.array([xl1[0][1], xl1[1][1], xl1[2][1], xl1[3][1]]) * 100).round(2)
    y5 = (np.array([xl1[0][-1], xl1[1][-1], xl1[2][-1], xl1[3][-1]]) * 100).round(2)

    bar_width = 0.25
    x2 = np.array([i - bar_width for i in range(len(bet))])
    x3 = np.array([i + bar_width for i in range(len(bet))])

    fig, ax1 = plt.subplots(figsize=(16, 9))

    ax1.bar(x2, y1, width=bar_width, label='原始数据', color='white', edgecolor='black')
    ax1.bar(range(len(bet)), y2, width=bar_width, label='隐私加噪', hatch='...', color='white', edgecolor=colors[2])
    ax1.bar(x3, y3, width=bar_width, label='正常加噪', hatch='', color=colors[9], edgecolor=colors[8])

    for i in range(len(bet)):
        ax1.text(i - bar_width, y1[i], str(y1[i]), ha='center', fontsize=12, **TNR)
        ax1.text(i, y2[i], str(y2[i]), ha='center', fontsize=12, **TNR)
        ax1.text(i + bar_width, y3[i], str(y3[i]), ha='center', fontsize=12, **TNR)

    ax1.set_xlabel('数据集大小', fontsize=16)
    ax1.set_ylabel('隐私含量', fontsize=16)
    ax1.set_xticks(range(len(bet)))
    ax1.set_xticklabels(bet, fontsize=16)
    ax1.set_ylim(0, max(y1.max(), y2.max(), y3.max()) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(x2, y4, color='gray', marker='o', linestyle='dotted', linewidth=2, label='原始数据')
    ax2.plot(range(len(bet)), y5, color=colors[2], marker='s', linestyle='--', linewidth=2, label='隐私加噪')
    ax2.plot(x3, y6, color=colors[8], marker='^', linestyle='-', linewidth=2, label='正常加噪')

    ax2.set_ylabel('隐私泄露程度 (%)', fontsize=16)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=6, loc='upper left', fontsize=16)

    for i in range(len(bet)):
        ax2.text(i - bar_width, y4[i], str(y4[i]), ha='center', fontsize=12, **TNR)
        ax2.text(i, y5[i], str(y5[i]), ha='center', fontsize=12, **TNR)
        ax2.text(i + bar_width, y6[i], str(y6[i]), ha='center', fontsize=12, **TNR)

    plt.title('不同数据集下模型隐私含量和隐私泄露程度对比', fontsize=20)
    title_text = plt.gca().get_title()
    plt.savefig('picture/'+title_text + "0.png", dpi=300)
    plt.show()

def figure1(data,name,i):
    if (i == 1):
        data = np.round(data, 3)
    else:
        data = np.round(data, ).astype(int)

    fig, ax1 = plt.subplots(figsize=(16, 9))

    data = data.T  # 转置数据以匹配 bar 图的数据列

    bar_width = 0.15
    x1 = np.array([i - bar_width for i in range(len(bet))])
    x2 = np.array([i for i in range(len(bet))])
    x3 = np.array([i + bar_width for i in range(len(bet))])
    x4 = np.array([i + 2*bar_width for i in range(len(bet))])



    ax1.bar(x1, data[0], width=bar_width, label='原始数据', color='white', edgecolor='black')
    ax1.bar(x2, data[1], width=bar_width, label='正常加噪', hatch='', color=colors[9], edgecolor=colors[8])
    ax1.bar(x3, data[2], width=bar_width, label='类别加噪', hatch='/', color='white', edgecolor=colors[7])
    ax1.bar(x4, data[3], width=bar_width, label='隐私加噪', hatch='...', color='white', edgecolor=colors[2])

    for i in range(len(bet)):
        ax1.text(x1[i], data[0][i], str(data[0][i]), ha='center', fontsize=12, **TNR)
        ax1.text(x2[i], data[1][i], str(data[1][i]), ha='center', fontsize=12, **TNR)
        ax1.text(x3[i], data[2][i], str(data[2][i]), ha='center', fontsize=12, **TNR)
        ax1.text(x4[i], data[3][i], str(data[3][i]), ha='center', fontsize=12, **TNR)

    ax1.set_xlabel('噪声水平', fontsize=16)
    ax1.set_ylabel(name, fontsize=16)
    ax1.set_xticks(range(len(bet)))
    ax1.set_xticklabels(bet, fontsize=16)
    ax1.legend()
    ax1.set_title('不同噪声水平下模型'+name+'对比', fontsize=20)

    title_text = ax1.get_title()
    plt.savefig('picture/' + title_text + "0.png", dpi=300)
    plt.show()
def figure2(data,name,i):

    if(i==1) :  data=np.round(data,3)
    else: data=np.round(data,).astype(int)



    fig, ax1 = plt.subplots(figsize=(16, 9))

    data = data.T  # 转置数据以匹配 bar 图的数据列

    bar_width = 0.2
    x1 = np.array([i - bar_width for i in range(len(bet))])
    x2 = np.array([i for i in range(len(bet))])
    x3 = np.array([i + bar_width for i in range(len(bet))])
    x4 = np.array([i + 2*bar_width for i in range(len(bet))])

    bet1 = [1000,5000,10000,15000,20000]
    ax1.bar(x1, data[0], width=bar_width, label='原始数据', color='white', edgecolor='black')
    ax1.bar(x2, data[1], width=bar_width, label='正常加噪', hatch='', color=colors[9], edgecolor=colors[8])
    ax1.bar(x3, data[2], width=bar_width, label='类别加噪', hatch='/', color='white', edgecolor=colors[7])
    ax1.bar(x4, data[3], width=bar_width, label='隐私加噪', hatch='...', color='white', edgecolor=colors[2])

    for i in range(len(bet)):
        ax1.text(x1[i], data[0][i], str(data[0][i]), ha='center', fontsize=12, **TNR)
        ax1.text(x2[i], data[1][i], str(data[1][i]), ha='center', fontsize=12, **TNR)
        ax1.text(x3[i], data[2][i], str(data[2][i]), ha='center', fontsize=12, **TNR)
        ax1.text(x4[i], data[3][i], str(data[3][i]), ha='center', fontsize=12, **TNR)

    ax1.set_xlabel('数据集大小', fontsize=16)
    ax1.set_ylabel(name, fontsize=16)
    ax1.set_xticks(range(len(bet)))
    ax1.set_xticklabels(bet1, fontsize=16)
    ax1.legend()
    ax1.set_title('噪声水平0.2时不同数据集大小下模型'+name+'对比', fontsize=20)

    title_text = ax1.get_title()
    plt.savefig('picture/' + title_text + "0.png", dpi=300)
    plt.show()
# figure4()
# figure5()
names=['隐私含量', '隐私保护程度' ,'安全隐私含量']
A=np.load('a.npy')
B=np.load('b.npy')
for i in range(len(A)):
    figure1(A[i],names[i],i)
for i in range(len(B)):
    figure2(B[i],names[i],i)