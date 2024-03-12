import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.stats import binom

# 試行回数nと確率p
def cal_p(n,N,k):
    return 1-math.comb(N-n,k)/math.comb(N,k)

def cal_E(n,N,k):
    s=0
    for i in range(k):
        s+=(i+1)*math.comb(n,i+1)*math.comb(N-n,k-(i+1))/(math.comb(N,k)-math.comb(N-n,k))
    return s
N=1000
M=200
k=5

p=[0]

for i in range(N):
    p.append(cal_p(i+1,N,k))

flcert=[]
pro_1=[]
pro_2=[]

idx=[]
# for i in range(N+1):
#     heatmap.append([])
# heatmap.append()
x=range(M+1)


for i in range(N+1):
    idx.append(i)

    f = binom(M, p[i])
    pp=f.pmf(x)
    sum=0
    for j in range(M/2+1):
        sum+=pp[j]
    flcert.append(sum)

    if i<N/2:
        pro_1.append(1)
        pro_2.append(1)
    elif i<N-cal_E(i,N,k):
        pro_1.append(1)
        pro_2.append(0)
    else:
        pro_1.append(0)
        pro_2.append(0)

line_w=3
g1=plt.plot(idx,flcert,color="red", linewidth=line_w)
g2=plt.plot(idx,pro_1,color="blue", linewidth=line_w)
# ax.fill_between(A, Acc, facecolor='red', alpha=0.5)

plt.xlabel('Number of Malicious Clients')
plt.ylabel('Success Percentage')
plt.ylim(0,1.05)
plt.xlim(0,100)
plt.xticks([0,100,200,300,400,500,600,700,800,900,1000])
# plt.grid(True)
plt.legend((g1[0], g2[0]), ("FLCert", "Proposed Method"),loc='lower right')
txt="Area_1.png"
plt.savefig(txt)
plt.clf()
plt.close()

g1=plt.plot(idx,flcert,color="red", linewidth=line_w)
g3=plt.plot(idx,pro_2,color="blue", linewidth=line_w)
# ax.fill_between(A, Acc, facecolor='red', alpha=0.5)
# print(pro_2)
plt.xlabel('Number of Malicious Clients')
plt.ylabel('Success Percentage')
plt.ylim(0,1.05)
plt.xlim(0,100)
plt.xticks([0,100,200,300,400,500,600,700,800,900,1000])
# plt.grid(True)
plt.legend((g1[0], g3[0]), ("FLCert", "Proposed Method"),loc='lower right')
txt="Area_2.png"
plt.savefig(txt)
plt.clf()
plt.close()

print("figure generated")