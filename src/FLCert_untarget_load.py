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
import math


time_strt=time.time()

#各行k個の1を含むテスト行列(row×col)をランダムに作成する関数
def generate_test_matrix(row, col, k):
    
    if math.comb(col,k)<row:
        print("可能な組み合わせが行数を下回っています")
        return None
    
    if k > col:
        print("クライアント数は行数よりも少ない数で入力してください。")
        return None
    
    if row <= 0 or col <= 0 or k <= 0:
        print("行数、列数、クライアント数は正の値である必要があります。")
        return None
    
    unique_rows = set()
    matrix = []
    
    while len(unique_rows) < row:
        row_list = [0] * col
        ones_indices = random.sample(range(col), k)
        
        for idx in ones_indices:
            row_list[idx] = 1
        
        row_tuple = tuple(row_list)

        if row_tuple not in unique_rows:
            unique_rows.add(row_tuple)
            matrix.append(row_list)
    
    return matrix

#行列をファイル出力する関数
def save_matrix_to_file(matrix, file_name='matrix.txt'):
    """
    Save a matrix to a text file.

    Parameters:
    - matrix: The matrix to be saved.
    - file_name: Name of the output text file.
    """
    np.savetxt(file_name, matrix, fmt='%d')

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
def train_local_model(i, group_x, group_y, clients, global_model, batch_size, epochs,k,scale,global_model_old,TF):
    if TF[i]:
        #ローカルモデルをグローバルモデルで初期化
        local_model = global_model
        #クライアントiが属するグループのデータを用いてモデルの学習を行う
        local_model.fit(group_x[clients[i]], group_y[clients[i]], batch_size=batch_size, epochs=epochs, verbose=0)
        return local_model.get_weights()
    else:
        # print(i)
        #ローカルモデルをグローバルモデルで初期化
        old_weight = global_model_old.get_weights()
        global_weights=global_model.get_weights()

        #グローバルモデルとバックドアモデルの差を計算
        sub_weights = [gl *(-1) for gl in global_weights]
        diff_weights=[np.add(old_weight[j], sub_weights[j]) for j in range(len(sub_weights))]
        #差をスケールアップ(ここではクライアント数5倍)
        tmp_weights=[diff *(-1*k*scale) for diff in diff_weights]
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
        filename = f"{file_prefix}{i +1}.{file_extension}"
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
def create_datasets(L):
    #mnisyのダウンロード、必要な処理を行う
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    #各グループに含まれるデータのインデックスを格納する配列
    index_group=[[],[],[],[],[],[],[],[],[],[]]
    
    #乱数を用いて各グループにデータを振り分ける
    for i in range(len(x_train)):
        
        rnd=random.randint(0, 9)
        if rnd<L:
            index_group[y_train[i]].append(i)
        else:
            numbers = [I for I in range(10) if I != y_train[i]]
            num=random.choice(numbers)
            index_group[num].append(i)


    for i in range(len(index_group)):
        index_group[i]=np.array(index_group[i])
    # print(index_target)

    #インデックスが格納された配列のファイル出力
    # save_arrays_to_files(index_group,file_prefix='#MNIST_',file_extension='txt')
    # save_arrays_to_files([index_target],file_prefix='#targetclass_',file_extension='txt')

    #MNISTデータは後々使うので返り値として渡す
    return [x_train, y_train,x_test,y_test]

#インデックスが格納された配列を読み込み、訓練データが格納された配列を返す関数
def load_data(file_prefix,file_extension,x_train, y_train):
    loaded_arrays = load_arrays_from_files(file_prefix=file_prefix, file_extension=file_extension)
    # print(len(loaded_arrays))
    
    #グループ0~9までを初期化
    group_x=[[],[],[],[],[],[],[],[],[],[]]
    group_y=[[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(loaded_arrays)):
        for j in range(len(loaded_arrays[i])):
            group_x[i].append(x_train[j])
            group_y[i].append(y_train[j])
    
    for i in range(len(group_x)):
        group_x[i]=np.array(group_x[i])
        group_y[i]=np.array(group_y[i])
    
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

#FLの学習を行う関数
#FLの学習を行う関数
def FL(group_x, group_y, clients, batch_size, epochs,global_iter,k,TF,scale):
    # フェデレーテッドラーニングの設定

    #グローバルモデルの初期化
   # 重みを格納するリストを初期化
    averaged_weights = []
    global_model=create_model()
    global_model_old=create_model()

    for I in range(global_iter):
        print(I+1)
        #平均の重みを格納する配列を初期化
        averaged_weights = []

        #各クライアントでモデルの学習・更新情報の集計
        with ThreadPoolExecutor(max_workers=k) as executor:
            futures = [executor.submit(train_local_model, i, group_x, group_y, clients, global_model, batch_size, epochs,k,scale,global_model_old,TF) for i in range(k)]
            #i, group_x, group_y, clients, global_model, batch_size, epochs,k,scale,global_model_old,TF
            local_weights_list = [future.result() for future in futures]
        
            #モデルの重みの和をとる
            for local_weights in local_weights_list:
                if not averaged_weights:
                    averaged_weights = local_weights
                else:
                    averaged_weights = [np.add(averaged_weights[j], local_weights[j]) for j in range(len(averaged_weights))]

        # グローバルモデルの更新
        averaged_weights = [aw / k for aw in averaged_weights]
        global_model_old=global_model
        global_model.set_weights(averaged_weights)

    #グローバルモデルの重みを返す
    return global_model.get_weights()

#FLCertを実行する関数
def FLCert(M,N,k,batch_size, epochs,global_iter,num,scale,strt):
    # データの準備
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # データの準備
    #iidの度合いを表すパラメータL(=1~9)の設定
    L=1

    # [x_train, y_train,x_test,y_test]=create_datasets(L)
    [group_x,group_y]=load_data('#MNIST_','txt',x_train,y_train)
    # # 各クライアントが割り当てられるグループをランダムに決定
    # clients = []

    # clients=[random.randint(0, 9) for j in range(N)]

    # #クライアントの所属するグループ情報をファイル出力し保存
    # save_arrays_to_files([clients],file_prefix="#clients_",file_extension="txt")
    #読み込んで変数に代入
    #後ろの[0]は渡される型が配列in配列になっているため
    clients=load_arrays_from_files(file_prefix="#clients_",file_extension="txt")[0]
    
    # #グループ分け
    # X=generate_test_matrix(M,N,k)

    # #行列のファイル出力
    # save_matrix_to_file(X, file_name='#matrix.txt')
    X=load_matrix_from_file(file_name='#matrix.txt')
    # print(X)

    group_clients=get_indices_of_ones(X)
    print(group_clients)
    
    # f.write('Matrix\n')
    # f.write('X=\n')
    # for i in range(M):
    #     # f.write("Group "+str(i+1)+" :")
    #     x_idx=[str(j)+"\t" for j, x in enumerate(X[i][:]) if x == 1]
    #     #f.writelines(x_idx)
    #     if i<M-1:
    #         f.write("\n")
    # f.write('\n')

    # f.write('---------------\n')
    # f.close()

    #グローバルグモデルを格納する辞書を初期化
    # models={}
    tmp_clients=[]
    TF_clients=[]
    for i in range(M):
        tmp_clients.append([clients[j] for j in group_clients[i]])
        tmp=[]
        for j in range(len(group_clients[i])):
            if group_clients[i][j]<num:
                tmp.append(False)
            else:
                tmp.append(True)
        TF_clients.append(tmp)
    #print(tmp_clients)
    print(TF_clients)
    
    for i in range(strt-1,M):
        print("グループ%s :学習中" %str(i+1))
        #グループ分けからデータを抽出し、それを用いてFLを実行
        # tmp_clients=[clients[j] for j in group_clients[i]]
        # print(tmp_clients)
        model=create_model()
        weight=FL(group_x, group_y, tmp_clients[i], batch_size, epochs,global_iter,k,TF_clients[i],scale)
        model.set_weights(weight)
        #作成したモデルを保存
        txt="#group_"+str(i+1)+".h5"
        model.save(txt)
    
#多数決を行う関数
def majority_vote(x_test,y_test,M):
    #各モデルの推論結果を保存する配列outputの初期化
    output=[]
    #print(len(x_test))
    #print(len(models))
    append_output=output.append
    for i in range(M):
        #モデルの読み込み
        txt="#group_"+str(i+1)+".h5"
        model=tf.keras.models.load_model(txt)
        #モデルを用いて推論の実行
        pred=model.predict(x_test)

        #予測値を整数値で取得
        out=np.argmax(pred, axis=1)
        #結果の保存
        append_output(out)
    #正答数を記録する変数の初期化
    correct_num=0

    for i in range(len(x_test)):
        #i番目の画像に対する各モデルの推論結果を取得する配列の初期化
        ans=[]
        append_ans=ans.append
        for j in range(M):
            append_ans(output[j][i])
        #最頻値（多数決の結果を取得する）
        num=statistics.mode(ans)
        
        #多数決が正しかった場合は正答数を加算
        if y_test[i]==num:
            correct_num+=1
    
    acc=correct_num/len(x_test)
    return acc

#パラメータ設定
M=2
N=100
k=5
#正常なグループ数150
num=15

#strtは学習を開始するグループの番号
strt=80

batch_size = 32
global_iter=5
epochs=1

scale=100
# M=2
# N=10
# k=2

# batch_size = 32
# num_clients=N
# global_iter=1
# epochs=1

# テストデータで評価

FLCert(M,N,k,batch_size, epochs,global_iter,num,scale,strt)

# データの準備
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)



acc = majority_vote(x_test,y_test,M)

print('\nTest accuracy:', acc)


f = open("#FLCert_untarget_parallel.txt",'a')

f.write('Parameter\n\n')

f.write('Number of Groups\n')
f.write('M=')
f.writelines(str(M))
f.write('\n')
f.write('\n')

f.write('Number of Clients\n')
f.write('N=')
f.writelines(str(N))
f.write('\n')
f.write('\n')

f.write('Clients for each Groups\n')
f.write('k=')
f.writelines(str(k))
f.write('\n')
f.write('\n')

# f.write('Number of Attackers\n')
# f.write('num=')
# f.writelines(str(num))
# f.write('\n')
# f.write('\n')

# f.write('Datas for each Clients\n')
# f.write('D=')
# f.writelines(str(60000/N))
# f.write('\n')
# f.write('\n')
# f.write('\n')

f.write("algorithm:FedAvg\n")
f.write("dataset:MNIST\n\n")

f.write("number of clients:"+str(N)+"\n")
f.write("number of groups:"+str(M)+"\n")
f.write("clients for each group:"+str(k)+"\n\n")

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

for i in range(M):
    txt="#group_"+str(i+1)+".h5"
    model1=tf.keras.models.load_model(txt)
    model=create_model()
    model.set_weights(model1.get_weights())
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    f.write('\nGroup '+str(i+1)+' /Test accuracy:'+str(test_acc))

f.write("\n\ntest acc:"+str(acc)+"\n")

time_end=time.time()
time_use=time_end-time_strt

f.write("time:"+str(time_use)+"\n\n")
f.write("-----------------------\n")

f.close()