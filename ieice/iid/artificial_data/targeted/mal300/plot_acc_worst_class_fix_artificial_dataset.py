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

idx=2

group=[100,200,300,400,500]
client=[100,200,300,400,500]

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179


# pklファイルから配列を読み込みます
with open('Acc_artificial_target_'+str(group[idx])+'_worst_class_fix.pkl', 'rb') as f:
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

plt.title("Target Attack (Artificial Dataset)\nMalicious Clients : "+str(client[idx]))
plt.xlabel('Model Size (%)')
plt.ylabel('Accuracy')
plt.ylim(0,1.1)
plt.xlim(0,100)
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
# plt.grid(True)
plt.legend((g1[0], g2[0]), ("Proposed method", "FLCert"))
txt="Acc_artificial_target_"+str(group[idx])+"_worst_class_fix.png"
plt.savefig(txt)
txt="Acc_artificial_target_"+str(group[idx])+"_worst_class_fix.eps"
plt.savefig(txt)
plt.clf()
plt.close()

print("figure generated")