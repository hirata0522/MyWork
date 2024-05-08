import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

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

# モデルの構築
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(30, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(50, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def evaluate_target_attack(x_test,y_test,model,class1):
    x_test_main=[]
    y_test_main=[]

    x_test_target=[]

    for i in range(len(x_test)):
        if y_test[i]!=class1:
            x_test_main.append(x_test[i])
            y_test_main.append(y_test[i])
        else:
            x_test_target.append(x_test[i])

    x_test_main=np.array(x_test_main)
    y_test_main=np.array(y_test_main)

    x_test_target=np.array(x_test_target)

    pred=model.predict(x_test_main)
    out=np.argmax(pred, axis=1)
    ans_main=0
    for i in range(len(x_test_main)):
        if out[i]==y_test_main[i]:
            ans_main+=1
    main_acc=ans_main/len(x_test_main)

    pred=model.predict(x_test_target)
    out_t=np.argmax(pred, axis=1)
    ans_target=0
    for i in range(len(x_test_target)):
        if out_t[i]==class1:
            ans_target+=1
    target_acc=ans_target/len(x_test_target)

    return [main_acc,target_acc]

def count_smaller_elements(arr,num):
    tmp=[]
    for i in range(arr.shape[0]):
        count=0
        for j in range(arr.shape[1]):
            if arr[i][j] == 1 and j<num:
                count += 1
        tmp.append(count)

    return tmp

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

M=200
num=200

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

X=load_matrix_from_file("#matrix.txt")
num_mal=count_smaller_elements(X,num)
Y=create_if_group_poisoned(X,num)

print(num_mal)


print(sum(t>0 for t in Y))
print(sum(x>0 for x in num_mal))

acc=[[],[],[],[],[],[]]
for i in range(M):
    txt="#group_"+str(i+1)+".h5"
    model1=tf.keras.models.load_model(txt)
    model=create_model()
    model.set_weights(model1.get_weights())
    [main_acc,target_acc] = evaluate_target_attack(x_test,y_test,model,0)
    acc[num_mal[i]].append(target_acc)

fig, ax=plt.subplots()

bp = ax.boxplot(acc) 
ax.set_xticklabels(['0%\nmodels:'+str(len(acc[0])), '20%\n'+str(len(acc[1])), '40%\n'+str(len(acc[2])), '60%\n'+str(len(acc[3])), '80%\n'+str(len(acc[4])), '100%\n'+str(len(acc[5]))])
plt.title("Target Attack\n malicious :"+str(num))
plt.ylabel("Target accuracy")
plt.xlabel("Percentage of malicious clients in group")

plt.savefig("Acc_mal"+str(num)+".png", bbox_inches='tight')
# plt.show()
