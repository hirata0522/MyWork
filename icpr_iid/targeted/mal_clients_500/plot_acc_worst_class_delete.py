import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_matrix_from_file(file_name='matrix.txt'):
    """
    Load a matrix from a text file.

    Parameters:
    - file_name: Name of the input text file.

    Returns:
    - NumPy array containing the loaded matrix.
    """
    return np.loadtxt(file_name, dtype=int)

def create_if_group_poisoned(X,num):
    row = len(X)
    col = len(X[0])
    #print("行数")
    #print(row)
    #print("列数")
    #print(col)
    if num > col:
        print("攻撃者数は行列の列数よりも少なく入力してください。")
        return None
    a = [[1] if i < num else [0] for i in range(col)]
    
    #print(np.dot(X, a))

    y = [0 if np.dot(X[i], a).item() == 0 else 1 for i in range(row)]
    

    return y


N=1000
M=200
k=5
#どの攻撃者数までやるか
j=200
#メモリの幅
h=25

idx=2

group=[100,300,500,700,900]
client=[100,300,500,700,900]

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179


# pklファイルから配列を読み込みます
with open('delete_target_'+str(group[idx])+'_worst_class.pkl', 'rb') as f:
    delete_models = pickle.load(f)

X=load_matrix_from_file(file_name='#matrix.txt')

Y=create_if_group_poisoned(X,group[idx])

mal=[]
ami=[]

x_t=[]

mal_num=0
ami_num=0

for i in range(len(delete_models)):
    x_t.append((200-i)/2)
    if Y[delete_models[i]]==0:
        ami_num+=1
    else:
        mal_num+=1
    ami.append(ami_num)
    mal.append(mal_num)

plt.clf()
fig, ax = plt.subplots()


g1=plt.plot(x_t,mal,color="red")
g2=plt.plot(x_t,ami,color="blue")
# ax.fill_between(A, Acc, facecolor='red', alpha=0.5)

plt.title("Target Attack\nMalicious Clients : "+str(client[idx]))
plt.xlabel('Model Size (%)')
plt.ylabel('Number of models')
# plt.ylim(0,1)
plt.xlim(0,100)
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
# plt.grid(True)
plt.legend((g1[0], g2[0]), ("attacked models", "non-attacked models"))
txt="delete_target_"+str(group[idx])+"_worst_class.png"
plt.savefig(txt)
plt.clf()
plt.close()

print("figure generated")