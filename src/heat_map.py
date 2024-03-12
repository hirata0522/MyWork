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

N=1000
M=200
k=5

n_mal=[0]
p=[0]

for i in range(N):
    n_mal.append(i+1)
    p.append(cal_p(i+1,N,k))

heatmap=[]
# for i in range(N+1):
#     heatmap.append([])
# heatmap.append()
x=[]
for i in range(M+1):
    x.append(M-i)


for i in range(N+1):
    f = binom(M, p[i])
    heatmap.append(f.pmf(x))


heat_=np.array(heatmap).T
df = pd.DataFrame(data=heat_,index=x)
# plt.figure()
# sns.heatmap([row[::1] for row in heat_],
#     cmap="bwr",
#     index=x)
plt.figure()
# sns.heatmap(df, cmap="CMRmap"
#             ,vmin=-0.4)
# sns.heatmap(df, cmap="CMRmap"
#             ,vmax=0.25
#             ,vmin=0)
sns.heatmap(df, cmap="CMRmap"
            ,xticklabels=100
            ,yticklabels=20
            ,vmax=0.25
            ,vmin=0
            ,cbar=True
            )
#gnuplot,gnuplot2,hot,CMRmap
# sns.heatmap(df
#             ,vmax=0.25
#             ,vmin=0)
# sns.heatmap(df, cmap="afmhot",
#     vmin=-0.5)
plt.ylabel("Number of attacked models")
plt.xlabel("Number of malicious clients")
plt.savefig('heatmap.png')
# plt.close('all')