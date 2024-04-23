import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tensorflow import keras
#from tensorflow.keras import layers
from keras import layers
import time
import random
import statistics
import pickle

time_strt=time.time()

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

# 並列処理(ローカルモデルの訓練)を行うための関数
def train_local_model(i, group_x, group_y, clients, global_model, batch_size, epochs,num,poisoned_group_x,poisoned_group_y,N):
    if i>=num:
        #ローカルモデルをグローバルモデルで初期化
        local_model = global_model
        #クライアントiが属するグループのデータを用いてモデルの学習を行う
        local_model.fit(group_x[clients[i]], group_y[clients[i]], batch_size=batch_size, epochs=epochs, verbose=0)
        return local_model.get_weights()
    else:
        # print(i,"poisoned")
        #ローカルモデルをグローバルモデルで初期化
        local_model = global_model
        global_weights=global_model.get_weights()
        #クライアントiが属するグループのデータを用いてモデルの学習を行う
        local_model.fit(poisoned_group_x[clients[i]], poisoned_group_y[clients[i]], batch_size=batch_size, epochs=epochs, verbose=0,shuffle=False)

        #グローバルモデルとバックドアモデルの差を計算
        original_weights=local_model.get_weights()
        sub_weights = [gl *(-1) for gl in global_weights]
        diff_weights=[np.add(original_weights[j], sub_weights[j]) for j in range(len(sub_weights))]
        #差をスケールアップ(ここではクライアント数5倍)
        tmp_weights=[diff *(N) for diff in diff_weights]
        #スケールアップしたモデルの重みの差に当初のグローバルモデルの重みを加え、最終的な返り値とする
        poisoned_weights = [np.add(tmp_weights[j], global_weights[j]) for j in range(len(global_weights))]
        return poisoned_weights

#配列をファイル出力によって保存する関数
def save_arrays_to_files(arrays, file_prefix='#group_', file_extension='txt'):
    """
    Save multiple NumPy arrays to files.

    Parameters:
    - arrays: List of NumPy arrays to be saved.
    - file_prefix: Prefix for the output file names (default is 'array_').
    - file_extension: File extension for the output files (default is 'txt').
    """
    for i, arr in enumerate(arrays):
        filename = f"{file_prefix}{i + 1}.{file_extension}"
        np.savetxt(filename, arr, fmt='%d')  # fmt='%d'は整数をテキストとして保存するための指定

#ファイルから配列を読み込む関数
def load_arrays_from_files(file_prefix='#group_', file_extension='txt'):
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
            arr = np.loadtxt(filename, dtype=int)
            arrays.append(arr)
            i += 1
        except FileNotFoundError:
            break  # ファイルが見つからない場合は終了
    return arrays

#iid度を制御するパラメータL(=1~10)を入力としてMNISTを10のグループに分割する
#そのインデックスが格納された配列をファイル出力する
#ついでにMNISTのデータも返す
def create_datasets(L,class1,class2):
    #mnisyのダウンロード、必要な処理を行う
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    #各グループに含まれるデータのインデックスを格納する配列

    group_target=[]

    group_x=[[],[],[],[],[],[],[],[],[],[]]
    group_y=[[],[],[],[],[],[],[],[],[],[]]
    
    #乱数を用いて各グループにデータを振り分ける
    cn=0
    for i in range(len(x_train)):
        if y_train[i]==class1:
            group_target.append(x_train[i])
            # print(i)
        
        rnd=random.randint(0, 9)
        if rnd<L:
            cn+=1
            group_x[y_train[i]].append(x_train[i])
            group_y[y_train[i]].append(y_train[i])

        else:
            numbers = [I for I in range(10) if I != y_train[i]]
            num=random.choice(numbers)
            group_x[num].append(x_train[i])
            group_y[num].append(y_train[i])


    # print(cn)    

    for i in range(len(group_x)):
        group_x[i]=np.array(group_x[i])
    group_target=np.array(group_target)

    print(len(group_target))


    #インデックスが格納された配列のファイル出力
    with open('#MNIST_dataset_x.pkl', 'wb') as f:
        pickle.dump(group_x, f)
    with open('#MNIST_dataset_y.pkl', 'wb') as f:
        pickle.dump(group_y, f)


    p_group_x=[[],[],[],[],[],[],[],[],[],[]]
    p_group_y=[[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(group_x)):
        for j in range(len(group_x[i])):
            if group_y[i][j]!=class1:
                p_group_x[i].append(group_x[i][j])
                p_group_y[i].append(group_y[i][j])
    poisoned_group_x=[[],[],[],[],[],[],[],[],[],[]]
    poisoned_group_y=[[],[],[],[],[],[],[],[],[],[]]
    
    num2=0
    len_cl1=len(group_target)
    for i in range(10):
        num1=0
        for j in range(len(group_x[i])):
            if j%32<22:
                poisoned_group_x[i].append(p_group_x[i][num1%len(p_group_x[i])])
                poisoned_group_y[i].append(p_group_y[i][num1%len(p_group_x[i])])
                num1+=1
            else:
                poisoned_group_x[i].append(group_target[num2%len_cl1])
                poisoned_group_y[i].append(class2)
                num2+=1

    for i in range(len(group_x)):
        poisoned_group_x[i]=np.array(poisoned_group_x[i])
        poisoned_group_y[i]=np.array(poisoned_group_y[i])


    with open('#MNIST_poisoned_dataset_x.pkl', 'wb') as f:
        pickle.dump(poisoned_group_x, f)
    with open('#MNIST_poisoned_dataset_y.pkl', 'wb') as f:
        pickle.dump(poisoned_group_y, f)    
    # save_arrays_to_files(index_group,file_prefix='#MNIST_',file_extension='txt')
    # save_arrays_to_files([index_target],file_prefix='#targetclass_',file_extension='txt')

    # #MNISTデータは後々使うので返り値として渡す
    # return [x_train, y_train,x_test,y_test]

def create_poisoned_dataset():

    with open('#MNIST_poisoned_dataset_x.pkl', 'rb') as ff:
        poisoned_group_x = pickle.load(ff)
    with open('#MNIST_poisoned_dataset_y.pkl', 'rb') as ff:
        poisoned_group_y = pickle.load(ff)  

    for i in range(len(poisoned_group_x)):
        poisoned_group_x[i]=np.array(poisoned_group_x[i])
        poisoned_group_y[i]=np.array(poisoned_group_y[i])  

    # cnt=[0,0,0,0,0,0,0,0,0,0]
    # print(len(poisoned_group_x[1]))
    # for i in range(len(poisoned_group_x[1])):
    #     cnt[poisoned_group_y[1][i]]+=1
    # print(cnt)   
    for j in range(len(poisoned_group_y)):
        cnt=[0,0,0,0,0,0,0,0,0,0]
        # print(len(group_y[j]))
        for i in range(len(poisoned_group_y[j])):
            cnt[poisoned_group_y[j][i]]+=1
        print("poisoned dataset ",j+1,":",cnt)   
    
    return [poisoned_group_x,poisoned_group_y]

#インデックスが格納された配列を読み込み、訓練データが格納された配列を返す関数
def load_data():
    # loaded_arrays = load_arrays_from_files(file_prefix=file_prefix, file_extension=file_extension)
    # # print(len(loaded_arrays))
    
    # #グループ0~9までを初期化
    # group_x=[[],[],[],[],[],[],[],[],[],[]]
    # group_y=[[],[],[],[],[],[],[],[],[],[]]

    # for i in range(len(loaded_arrays)):
    #     for j in range(len(loaded_arrays[i])):
    #         # group_x[i].append(x_train[loaded_arrays[i][j]])
    #         # group_y[i].append(y_train[loaded_arrays[i][j]])
    #         group_x[i].append(x_train[j])
    #         group_y[i].append(y_train[j])
    
    # for i in range(len(group_x)):
    #     group_x[i]=np.array(group_x[i])
    #     group_y[i]=np.array(group_y[i])
    
    # cnt=[0,0,0,0,0,0,0,0,0,0]
    # print(len(group_y[1]))
    # for i in range(len(group_y[1])):
    #     cnt[y_train[group_y[1][i]]]+=1
    # print(cnt,"load")    
    with open('#MNIST_dataset_x.pkl', 'rb') as f:
        group_x = pickle.load(f)
    with open('#MNIST_dataset_y.pkl', 'rb') as f:
        group_y = pickle.load(f)    
    
    for i in range(len(group_x)):
        group_x[i]=np.array(group_x[i])
        group_y[i]=np.array(group_y[i])
    
    for j in range(len(group_y)):
        cnt=[0,0,0,0,0,0,0,0,0,0]
        # print(len(group_y[j]))
        for i in range(len(group_y[j])):
            cnt[group_y[j][i]]+=1
        print("dataset ",j+1,":",cnt)   


    return [group_x,group_y]

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
    

# データの準備
#iidの度合いを表すパラメータL(=1~9)の設定
L=5
class1=0
class2=6
create_datasets(L,class1,class2)
[group_x,group_y]=load_data()
[poisoned_group_x,poisoned_group_y]=create_poisoned_dataset()
#print(poisoned_group_y)

# フェデレーテッドラーニングの設定
global_model = create_model()
batch_size = 32
num_clients = 5
#攻撃者数
num=1
N=num_clients

# データの準備
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 各クライアントが割り当てられるグループをランダムに決定
clients = []

for i in range(N):
    rnd=random.randint(0, 9)
    clients.append(rnd)

#クライアントの所属するグループ情報をファイル出力し保存
# save_arrays_to_files([clients],file_prefix="#clients_",file_extension="txt")
#読み込んで変数に代入
#後ろの[0]は渡される型が配列in配列になっているため
clients=load_arrays_from_files(file_prefix="#clients_",file_extension="txt")[0]

# 重みを格納するリストを初期化
averaged_weights = []
global_iter=200
epochs = 5

global_model=create_model()

for I in range(global_iter):
    print(I+1)
    #平均の重みを格納する配列を初期化
    averaged_weights = []

    #各クライアントでモデルの学習・更新情報の集計
    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = [executor.submit(train_local_model, i, group_x, group_y, clients, global_model, batch_size, epochs,num,poisoned_group_x,poisoned_group_y,N) for i in range(num_clients)]
        local_weights_list = [future.result() for future in futures]
    
        #モデルの重みの和をとる
        for local_weights in local_weights_list:
            if not averaged_weights:
                averaged_weights = local_weights
            else:
                averaged_weights = [np.add(averaged_weights[j], local_weights[j]) for j in range(len(averaged_weights))]

    # グローバルモデルの更新
    averaged_weights = [aw / num_clients for aw in averaged_weights]
    global_model.set_weights(averaged_weights)

# テストデータで評価
[main_acc,target_acc] = evaluate_target_attack(x_test,y_test,global_model,class1)
print('\nTest main accuracy:', main_acc)
print('\nTest target accuracy:', target_acc)

f = open("#FedAvg_target.txt",'a')

f.write('Parameter\n\n')

f.write("algorithm:FedAvg\n")
f.write("dataset:MNIST\n\n")

f.write("number of clients:"+str(num_clients)+"\n")
f.write("learning rate:0.001\n")
f.write("batch size:"+str(batch_size)+"\n")
f.write("global iteration:"+str(global_iter)+"\n")
f.write("local iteration:"+str(epochs)+"\n\n")


f.write("model:CNN\n\n")

f.write("layer:\n")
f.write("Input:28 x 28 x 1\n")
f.write("Convolution + ReLU : 3 x 3 x 30\n")
f.write("Max Pooling : 2 x 2\n")
f.write("Convolution + ReLU : 3 x 3 x 50\n")
f.write("Max Pooling : 2 x 2\n")
f.write("Fully Connected + ReLU : 100\n")
f.write("Softmax : 10\n\n")

f.write("number of malicious clients:"+str(num)+"\n\n")

f.write("target class:"+str(class1)+"\n\n")

f.write("test acc:"+str(main_acc)+"\n")
f.write("target acc:"+str(target_acc)+"\n")

time_end=time.time()
time_use=time_end-time_strt

f.write("time:"+str(time_use)+"\n")
f.write("-----------------------\n")

f.close()


# モデルをファイルに出力
global_model.save('fl_target.h5')