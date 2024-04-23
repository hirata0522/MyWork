import numpy as np
import matplotlib.pyplot as plt
import math


#ファイルから行列を読み込む関数
def load_matrix_from_file(file_name='matrix.txt'):
    """
    Load a matrix from a text file.

    Parameters:
    - file_name: Name of the input text file.

    Returns:
    - NumPy array containing the loaded matrix.
    """
    return np.loadtxt(file_name, dtype=int)

#行列の各行について1である要素のインデックスを格納した配列を返す関数
def get_indices_of_ones(matrix):
    """
    Get the indices of ones in each row of the matrix.

    Parameters:
    - matrix: The input matrix.

    Returns:
    - List of NumPy arrays containing the indices of ones for each row.
    """
    return [np.where(row == 1)[0] for row in matrix]

#行列表示用の関数
def print_matrix(matrix):
    for row in matrix:
        print(row)

#テスト行列Xにnum人の攻撃者が含まれる際
#汚染されたモデルの真値が格納されたベクトルを返す関数
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

def check_elements(arr,num_old,num):
    for element in arr:
        if num_old-1 < element <= num-1:
            return True
    return False

#ファイルから配列を読み込む関数
def load_arrays_from_files_float(file_prefix='#group_', file_extension='txt'):
    """
    Load multiple NumPy arrays from files.

    Parameters:
    - file_prefix: Prefix for the input file names (default is 'array_').
    - file_extension: File extension for the input files (default is 'txt').

    Returns:
    - List of NumPy arrays loaded from files.
    """
    arrays = []
    i = 1
    while True:
        filename = f"{file_prefix}{i}.{file_extension}"
        try:
            arr = np.loadtxt(filename, dtype=float)
            arrays.append(arr)
            i += 1
        except FileNotFoundError:
            break  # ファイルが見つからない場合は終了
    return arrays
 
# num = np.array([191,182,167,160,151,145,141,136,130,130,109,109,110,108,96,84,78,88,80,72])
# group = np.array([5, 10, 15, 20, 25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])


N=1000
M=200
k=5
#どの攻撃者数までやるか
j=200
#メモリの幅
h=25

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179

num=57

final_loss=load_arrays_from_files_float(file_prefix="#loss",file_extension="txt")[0]
# print(final_loss)
X=load_matrix_from_file(file_name="#matrix.txt")
Y=create_if_group_poisoned(X,num)

plt.clf()
for i in range(M):
    if Y[i]==1:
        # print(i,final_loss[i])
        plt.scatter(num,final_loss[i],c="red",marker="x")
    else:
        plt.scatter(num,final_loss[i],c="blue")
plt.title("✕:contaminated / O:normal")
plt.xlabel('number of attacker')
plt.ylabel('loss')
# plt.ylim(0.00000125,0.000002)
txt="#loss_attacker"+str(num)+".png"
plt.savefig(txt)
plt.clf()
plt.close()

print("figure generated")