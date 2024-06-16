import  numpy as np
from collections import Counter
old_data = np.genfromtxt('output123.csv', delimiter=',', skip_header=1, dtype=int)
header = np.genfromtxt('output123.csv', delimiter=',', dtype=str, max_rows=1)

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           #
from function import *
from fig import *
from function1 import  *
comparison_matrix = np.array([
    [1,   2,   4,   1,   2,   1,   3,   1,   2,   1],
    [1/2, 1,   3,   1/2, 2,   1/2, 2,   1,   3,   1/2],
    [1/4, 1/3, 1,   1/4, 1/3, 1/3, 1/2, 1/3, 4,   1/3],
    [1,   2,   4,   1,   2,   1,   3,   1,   2,   1],
    [1/2, 1/2, 3,   1/2, 1,   1/2, 2,   1,   3,   1/2],
    [1,   2,   3,   1,   2,   1,   3,   1,   2,   1],
    [1/3, 1/2, 2,   1/3, 1/2, 1/3, 1,   1/2, 2,   1/3],
    [1,   1,   3,   1,   1,   1,   2,   1,   3,   1],
    [1/2, 1/3, 1/4, 1/2, 1/3, 1/2, 1/2, 1/3, 1,   1/2],
    [1,   2,   3,   1,   2,   1,   3,   1,   2,   1]
])
ahp=generate_ahp_weights(comparison_matrix)
print(ahp)


e1=[]
e2=[]
e=[]


utility=[]
f=1
epl= [ i for i in np.arange(0, 0.5, 0.05)]
B=[]
C=[]
wb=[]
wc=[]
w=[]

for r in range(1,11):
    B .append(old_data[:100*r:2])
    wb.append(get_quan(B[-1]))
    # 提取偶数行（索引1, 3, 5, ...）
    C .append(old_data[1:100*r:2])
    wc.append(get_quan(C[-1]))
    w.append([(wb[-1][i]+wc[-1][i])/2 for i in range(len(wb[-1]))])

for m in epl:
 entropy1=[]

 utility1 = []
 for r in range(10):
    entropy1.append(get_entropy2(laplace_mech(B[r],f,m),m,wb[r])+get_entropy2(laplace_mech(C[r],f,m),m,wc[r]))

    utility1.append(get_utility(B[r],laplace_mech(B[r],f,m),m)/2+get_utility(C[r],laplace_mech(C[r],f,m),m)/2)
 entropy.append(entropy1)
 utility.append(utility1)
e=entropy[0]

for r in range(10):
  e1.append(get_entropy1(filter_matrix(B[r], 2),
  wb[r])+get_entropy1(filter_matrix( C[r] , 2 ),wc[r]))
  e2.append(get_entropy3(filter_matrix1(laplace_mech(B[r],f,epl[5]),2,0.95,1/epl[5]),2,0.95,1/epl[5],
  wb[r])+get_entropy3(filter_matrix1(laplace_mech(C[r],f,epl[5]),2,0.95,1/epl[5]),2,0.95,1/epl[5],wc[r]))

figure0(range(len(e1)),e,e1,e2)
figure(epl,entropy,"熵")
figure(epl,utility,"数据效用")
figure1(epl,entropy,"熵")

figure2(header,ahp,w[0],w[4],w[9])






'''
    b1=[]
    B1=
    B2 = filter_matrix(laplace_mech(B,f,epl[5]), 2, B[0][2])
    B3=filter_matrix1(laplace_mech(B,f,epl[5]),2,B[0][2],0.95,0.05)
    C1 = laplace_mech(C, f, epl[-1])
    C2 = filter_matrix(C, 2, C[0][2])
    C3 = filter_matrix1(C1, 2, C[0][2], 0.95, 0.05)
    eb = get_entropy(B, wb)
    eb1 = get_entropy(B1, wb)
    eb2 = get_entropy(B2, wb)
    eb3 = get_entropy(B3, wb)
    ec = get_entropy(C, wc)
    ec1 = get_entropy( C1, wc)
    ec2 = get_entropy(C2, wc)
    ec3 = get_entropy(C3, wc)
    for m in epl:
     b1.append(get_entropy(laplace_mech(B,f,m), wb)+get_entropy(laplace_mech(C,f,m), wc))
    I0 = get_entropy(B, wb) - get_entropy(B2, wb)
    I1 = get_entropy(B1, wb) - get_entropy(B3, wb)
    I0 = I0+get_entropy(C,wc) - get_entropy(C2,wc)
    I1 = I1+get_entropy(C1,wc) - get_entropy(C3,wc)
    I0=I0/(get_entropy(B,wb)+get_entropy(C,wc))
    I1= I1/ (get_entropy(B1,wb) + get_entropy(C1,wc))
    e1.append(get_entropy(B,wb))
    e2.append( get_entropy(B1,wb))
    entropy1.append(b1)
    #e.append(get_entropy(B3,wb))

    #print(B1.shape,B2.shape,B3.shape)

'''
