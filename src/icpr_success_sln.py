import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.stats import binom

def cal_p(n,N,k):
    return 1-math.comb(N-n,k)/math.comb(N,k)

def cal_p_Sn(p_ami,p_mal,M,m_mal):
    # print("S_n",1-(1-p_ami)**(M-m_mal)*(1-p_mal)**(m_mal)-p_ami**(M-m_mal)*p_mal**m_mal)
    return 1-(1-p_ami)**(M-m_mal)*(1-p_mal)**(m_mal)-p_ami**(M-m_mal)*p_mal**m_mal

def cal_E_E_mal(M,m_mal,p_ami,p_mal):
    return (p_ami-p_mal)/cal_p_Sn(p_ami,p_mal,M,m_mal)

N=1000
M=200
k=5
p_ami=0.95
p_mal=0.1
p=[0]

for i in range(N):
    p.append(cal_p(i+1,N,k))

flcert=[]
pro_1=[]

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
    for j in range(M//2+1):
        sum+=pp[j]
    flcert.append(sum)

    sum_p=0
    for j in range(M+1):
        # print(i,j)
        if (N-i)==0:
            sum_p+=0
        elif j==0:
            sum_p+=pp[j]
        elif j==M:
            sum_p+=0
        elif i/(N-i)<cal_E_E_mal(M,j,p_ami,p_mal):
            sum_p+=pp[j]
    if i==0:
        pro_1.append(1)
    else:
        pro_1.append(sum_p)

m_mal=0

p_sln=1-(1-p_ami)**M-p_ami**M
p_sln_tar=1-(1-p_ami)**(M-m_mal)*(1-p_mal)**(m_mal)-p_ami**(M-m_mal)*p_mal**m_mal

for i in range(M):
    m_mal=i
    p_sln=1-(1-p_ami)**M-p_ami**M
    p_sln_tar=1-(1-p_ami)**(M-m_mal)*(1-p_mal)**(m_mal)-p_ami**(M-m_mal)*p_mal**m_mal
    print("E_untarget-E_target",((1-p_ami)-(1-p_ami)**(M-m_mal)*(1-p_mal)**(m_mal))/p_sln_tar-((1-p_ami)-(1-p_ami)**M)/p_sln)
    print(p_sln_tar/p_sln)



print()

# line_w=3
# g1=plt.plot(idx,flcert,color="red", linewidth=line_w)
# g2=plt.plot(idx,pro_1,color="blue", linewidth=line_w)
# # ax.fill_between(A, Acc, facecolor='red', alpha=0.5)

# plt.title("Target Attack\np_ami="+str(p_ami)+"   p_mal="+str(p_mal))
# plt.xlabel('Number of Malicious Clients')
# plt.ylabel('Success Rate')
# plt.ylim(0,1.05)
# plt.xlim(0,100)
# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000])
# # plt.grid(True)
# plt.legend((g1[0], g2[0]), ("FLCert", "Proposed Method"),loc='lower right')
# txt="Area_icpr.png"
# plt.savefig(txt)
# plt.clf()
# plt.close()

# print("figure generated")