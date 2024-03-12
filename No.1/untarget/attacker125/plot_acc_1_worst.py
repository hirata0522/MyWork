import numpy as np
import matplotlib.pyplot as plt
import pickle

N=1000
M=200
k=5
#どの攻撃者数までやるか
j=200
#メモリの幅
h=25

idx=1

group=[150,125,100,75,50]
client=[57,91,130,179,243]

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179


# pklファイルから配列を読み込みます
with open('Acc_untarget_'+str(group[idx])+'_1_worst.pkl', 'rb') as f:
    Acc = pickle.load(f)

plt.clf()
fig, ax = plt.subplots()
A=[]
B=[]
for i in range(len(Acc)):
    A.append((200-i)/2)
    B.append(Acc[0])
g1=plt.plot(A,Acc,color="red")
g2=plt.plot(A,B,color="blue",linestyle="dashed")
# ax.fill_between(A, Acc, facecolor='red', alpha=0.5)

plt.title("Untarget Attack\nMalicious Clients : "+str(client[idx]))
plt.xlabel('Model Size (%)')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.xlim(0,100)
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
# plt.grid(True)
plt.legend((g1[0], g2[0]), ("Proposed method", "FLCert"))
txt="Acc_untarget_"+str(group[idx])+"_1_worst.png"
plt.savefig(txt)
plt.clf()
plt.close()

print("figure generated")