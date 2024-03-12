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

idx=0

group=[100,200,300,400,500]
client=[100,200,300,400,500]

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179
idx=0
Acc_0=[]

# pklファイルから配列を読み込みます
with open('Acc_untarget_'+str(group[idx])+'_worst_class.pkl', 'rb') as f:
    Acc_0 = pickle.load(f)

plt.clf()
fig, ax = plt.subplots()
A=[]
B=[]

for i in range(len(Acc_0)):
    A.append((200-i)/2)
    B.append(Acc_0[0])
g1=plt.plot(A,Acc_0,color="red")
g2=plt.plot(A,B,color="red",linestyle="dashed")
# ax.fill_between(A, Acc, facecolor='red', alpha=0.5)

idx=1
Acc_1=[]
B_1=[]
# pklファイルから配列を読み込みます
with open('Acc_untarget_'+str(group[idx])+'_worst_class.pkl', 'rb') as f:
    Acc_1 = pickle.load(f)

for i in range(len(Acc_1)):
    B_1.append(Acc_1[0])
g3=plt.plot(A,Acc_1,color="blue")
g4=plt.plot(A,B_1,color="blue",linestyle="dashed")

idx=2
Acc_2=[]
B_2=[]
# pklファイルから配列を読み込みます
with open('Acc_untarget_'+str(group[idx])+'_worst_class.pkl', 'rb') as f:
    Acc_2 = pickle.load(f)

for i in range(len(Acc_2)):
    B_2.append(Acc_2[0])
g5=plt.plot(A,Acc_2,color="green")
g6=plt.plot(A,B_2,color="green",linestyle="dashed")

idx=3
Acc_3=[]
B_3=[]
# pklファイルから配列を読み込みます
with open('Acc_untarget_'+str(group[idx])+'_worst_class.pkl', 'rb') as f:
    Acc_3 = pickle.load(f)

for i in range(len(Acc_3)):
    B_3.append(Acc_3[0])
g7=plt.plot(A,Acc_3,color="purple")
g8=plt.plot(A,B_3,color="purple",linestyle="dashed")

idx=4
Acc_4=[]
B_4=[]
# pklファイルから配列を読み込みます
with open('Acc_untarget_'+str(group[idx])+'_worst_class.pkl', 'rb') as f:
    Acc_4 = pickle.load(f)

for i in range(len(Acc_4)):
    B_4.append(Acc_4[0])
g9=plt.plot(A,Acc_4,color="orange")
g10=plt.plot(A,B_4,color="orange",linestyle="dashed")

g11=plt.plot([100],[100],color="black",linestyle="dashed")
g12=plt.plot([100],[100],color="black")

# plt.title("Untarget Attack")
plt.xlabel('Model Size (%)')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.xlim(0,100)
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
# plt.grid(True)
plt.legend((g1[0],g11[0], g3[0],g12[0],g5[0],g7[0],g9[0]), (str(group[0]),"FLCert",str(group[1]),"Proposed Method",str(group[2]),str(group[3]), str(group[4])),
           loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)
txt="Acc_untarget_class.eps"

plt.tight_layout()
plt.savefig(txt)
plt.clf()
plt.close()
plt.tight_layout()

print("figure generated")