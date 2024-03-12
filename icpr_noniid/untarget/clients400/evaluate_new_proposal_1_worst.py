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
import pickle

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
def save_matrix_to_file_float(matrix, file_name='matrix.txt'):
    """
    Save a matrix to a text file.

    Parameters:
    - matrix: The matrix to be saved.
    - file_name: Name of the output text file.
    """
    np.savetxt(file_name, matrix, fmt='%.100f')

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
def train_local_model(i, group_x, group_y, clients, global_model, batch_size, epochs,poisoned_group_x,poisoned_group_y,k,TF):
    if TF[i]:
        #ローカルモデルをグローバルモデルで初期化
        local_model = global_model
        #クライアントiが属するグループのデータを用いてモデルの学習を行う
        local_model.fit(group_x[clients[i]], group_y[clients[i]], batch_size=batch_size, epochs=epochs, verbose=0)
        return local_model.get_weights()
    else:
        # print(i)
        #ローカルモデルをグローバルモデルで初期化
        local_model = global_model
        global_weights=global_model.get_weights()
        # print(poisoned_group_y[clients[i]])
        #クライアントiが属するグループのデータを用いてモデルの学習を行う
        local_model.fit(poisoned_group_x[clients[i]], poisoned_group_y[clients[i]], batch_size=batch_size, epochs=epochs, verbose=0,shuffle=False)

        #グローバルモデルとバックドアモデルの差を計算
        original_weights=local_model.get_weights()
        sub_weights = [gl *(-1) for gl in global_weights]
        diff_weights=[np.add(original_weights[j], sub_weights[j]) for j in range(len(sub_weights))]
        #差をスケールアップ(ここではクライアント数5倍)
        tmp_weights=[diff *(k) for diff in diff_weights]
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

#配列をファイル出力によって保存する関数
def save_arrays_to_files_float(arrays, file_prefix='#group_', file_extension='txt'):
    """
    Save multiple NumPy arrays to files.

    Parameters:
    - arrays: List of NumPy arrays to be saved.
    - file_prefix: Prefix for the output file names (default is 'array_').
    - file_extension: File extension for the output files (default is 'txt').
    """
    for i, arr in enumerate(arrays):
        filename = f"{file_prefix}{i +1}.{file_extension}"
        np.savetxt(filename, arr, fmt='%.100f')

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

#iid度を制御するパラメータL(=1~10)を入力としてMNISTを10のグループに分割する
#そのインデックスが格納された配列をファイル出力する
#ついでにMNISTのデータも返す
def create_datasets(L,class1):
    #mnisyのダウンロード、必要な処理を行う
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    #各グループに含まれるデータのインデックスを格納する配列
    index_group=[[],[],[],[],[],[],[],[],[],[]]
    index_target=[]
    
    #乱数を用いて各グループにデータを振り分ける
    for i in range(len(x_train)):
        if y_train[i]==class1:
            index_target.append(i)
        
        rnd=random.randint(0, 9)
        if rnd<L:
            index_group[y_train[i]].append(i)
        else:
            numbers = [I for I in range(10) if I != y_train[i]]
            num=random.choice(numbers)
            index_group[num].append(i)


    for i in range(len(index_group)):
        index_group[i]=np.array(index_group[i])
    index_target=np.array(index_target)
    # print(index_target)

    #インデックスが格納された配列のファイル出力
    save_arrays_to_files(index_group,file_prefix='#MNIST_',file_extension='txt')
    save_arrays_to_files([index_target],file_prefix='#targetclass_',file_extension='txt')

    #MNISTデータは後々使うので返り値として渡す
    return [x_train, y_train,x_test,y_test]

def create_poisoned_dataset(x_train, y_train,class1,class2):
    [group_x,group_y]=load_data("#MNIST_","txt",x_train, y_train)
    tmp=load_arrays_from_files(file_prefix='#targetclass_',file_extension='txt')[0]
    group_target=[]
    for i in range(len(tmp)):
        group_target.append(x_train[tmp[i]])

    p_group_x=[[],[],[],[],[],[],[],[],[],[]]
    p_group_y=[[],[],[],[],[],[],[],[],[],[]]

    #groupx,yのデータから標的クラスとなるデータのみを取り除く
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
                poisoned_group_x[i].append(p_group_x[i][num1])
                poisoned_group_y[i].append(p_group_y[i][num1])
                num1+=1
            else:
                poisoned_group_x[i].append(group_target[num2%len_cl1])
                poisoned_group_y[i].append(class2)
                num2+=1

    for i in range(len(group_x)):
        poisoned_group_x[i]=np.array(poisoned_group_x[i])
        poisoned_group_y[i]=np.array(poisoned_group_y[i])
    
    return [poisoned_group_x,poisoned_group_y]

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
def FL(group_x, group_y, clients, batch_size, epochs,global_iter,poisoned_group_x,poisoned_group_y,k,TF):
    # フェデレーテッドラーニングの設定

    #グローバルモデルの初期化
   # 重みを格納するリストを初期化
    averaged_weights = []
    global_model=create_model()

    for I in range(global_iter):
        # print(I+1)
        #平均の重みを格納する配列を初期化
        averaged_weights = []

        #各クライアントでモデルの学習・更新情報の集計
        with ThreadPoolExecutor(max_workers=k) as executor:
            futures = [executor.submit(train_local_model, i, group_x, group_y, clients, global_model, batch_size, epochs,poisoned_group_x,poisoned_group_y,k,TF) for i in range(k)]
            local_weights_list = [future.result() for future in futures]
        
            #モデルの重みの和をとる
            for local_weights in local_weights_list:
                if not averaged_weights:
                    averaged_weights = local_weights
                else:
                    averaged_weights = [np.add(averaged_weights[j], local_weights[j]) for j in range(len(averaged_weights))]

        # グローバルモデルの更新
        averaged_weights = [aw / k for aw in averaged_weights]
        global_model.set_weights(averaged_weights)

    #グローバルモデルの重みを返す
    return global_model.get_weights()

#閾値の計算
def calculate_threshold(nums):
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    if n < 2:
        raise ValueError("List should contain at least 2 elements.")
    
    max_diff = sorted_nums[1] - sorted_nums[0]
    max_diff_median = (sorted_nums[0] + sorted_nums[1]) / 2

    for i in range(1, n - 1):
        diff = sorted_nums[i + 1] - sorted_nums[i]
        if diff > max_diff:
            max_diff = diff
            max_diff_median = (sorted_nums[i] + sorted_nums[i + 1]) / 2

    return max_diff_median
    
def pred_mnist(int):
    NNUM=[400,300,500,700,900]
    GGROUP=[400,300,500,700,900]
    M=200
    gr=GGROUP[int]
    # データの準備
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    [group_x,group_y]=load_data('#MNIST_','txt',x_train,y_train)
    
    clients=load_arrays_from_files(file_prefix="#clients_",file_extension="txt")[0]
    
    X=load_matrix_from_file(file_name='#matrix.txt')
    Y=create_if_group_poisoned(X,NNUM[int])
    print(Y)
    # print(X)

    #Y_MNISTはi番目の要素はi番目のデータセットに対するY_GROUP
    #Y_GROUPはj番目の要素がj番目のデータに対する各モデルの推論結果
    Y_MNIST=[]
    for i in range(10):
        print(i+1,"推論実行中")
        Y_GROUP=[]
        for j in range(len(group_x[i])):
            Y_GROUP.append([])

        for j in range(M):
            txt="#group_"+str(j+1)+".h5"
            model=tf.keras.models.load_model(txt)
            #モデルを用いて推論の実行
            pred=model.predict(group_x[i],verbose=0)
            

            #予測値を整数値で取得
            out=np.argmax(pred, axis=1)

            # if j+1==4 or j+1==6:
            #     print(out)

            #結果の保存
            for k in range(len(group_x[i])):
                Y_GROUP[k].append(out[k])
        # print(Y_GROUP)

        Y_MNIST.append(Y_GROUP)
    
    # Y_MNISTを保存します
    with open('Y_MNIST_untarget_'+str(gr)+'.pkl', 'wb') as f:
        pickle.dump(Y_MNIST, f)

#FLCertを実行する関数
def eval_1(N,M,int):
    NNUM=[400,300,500,700,900]
    GGROUP=[400,300,500,700,900]
    num=NNUM[int]
    gr=GGROUP[int]
    # データの準備
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    [group_x,group_y]=load_data('#MNIST_','txt',x_train,y_train)
    
    clients=load_arrays_from_files(file_prefix="#clients_",file_extension="txt")[0]
    
    X=load_matrix_from_file(file_name='#matrix.txt')
    Y=create_if_group_poisoned(X,num)
    # print(Y)

    with open('Y_MNIST_untarget_'+str(gr)+'.pkl', 'rb') as f:
        Y_MNIST = pickle.load(f)
    
    models=[]
    for i in range(M):
        models.append(i)
    #modelsには削除されていないモデルのインデックスを保存 

    delete_models=[]


    
    f = open("#Proposal_untarget_"+str(gr)+"_1_worst.txt",'a')

    #ここ以下をwhileで記述し、
    while(len(models)>0):
        # print(models)
        #各ローカルデータセットの損失を格納する配列
        loss=[]

        cs=[True,True,True,True,True,True,True,True,True,True]
        
        for i in range(10):
            loss_group=np.zeros(len(models))
            num_cs=0
            for j in range(len(group_x[i])):
                #各数字が何回出てきたかカウント
                num_pred=[0,0,0,0,0,0,0,0,0,0]
                for k in range(len(models)):
                    num_pred[Y_MNIST[i][j][models[k]]]+=1
                
                if num_pred.count(0)!=9:
                    num_cs+=1
                    # print(num_pred)

                
                #各入力に対する損失の計算(損失を格納する配列に加算)
                for k in range(len(models)):
                    
                    if Y_MNIST[i][j][models[k]]==group_y[i][j]:
                        loss_group[k]+=0
                    else:
                        # loss_group[k]+=1/num_pred[Y_MNIST[i][j][models[k]]]
                        loss_group[k]+=1

            #要素数で割り平均の計算
            for j in range(len(models)):
                if num_cs!=0:
                    loss_group[j]=loss_group[j]/num_cs
            
            if num_cs==0:
                cs[i]=False
            # print(num_cs)

            #各ローカルデータセットの損失を格納する配列に格納
            loss.append(loss_group)
        

        final_loss=np.zeros(len(models))

        cs_clients=0

        for i in range(N):
            tmp=np.zeros(len(models))
            if i<num:
                # num0=0
                # for j in range(len(models)):
                #     if X[models[j]][i]==0:
                #         num0+=1
                for j in range(len(models)):
                    flag=True
                    # model j が攻撃されているのか調べる
                    # 攻撃されていたらflagをfalseに
                    for ii in range(num):
                        if X[models[j]][ii]==1:
                            flag=False
                            # tmp[j]=1/num0*(10-1)
                    # 攻撃されていないモデルのスコアを1に
                    if flag:
                        tmp[j]=1

                        # ここで(クラス数-1)掛けるのが最適


                # print("attacker",tmp)
            else:
                tmp=loss[clients[i]]
                # print("honest client",tmp)
            if cs[clients[i]]:
                
                for j in range(len(models)):
                    final_loss[j]=tmp[j]+final_loss[j]
                    cs_clients+=1
                # print("client",i,"tmp_loss",tmp,"final_loss",final_loss)
                    
        
        for i in range(len(models)):
            if cs_clients!=0:
                final_loss[i]=final_loss[i]/N

        # save_arrays_to_files_float([final_loss],"#loss","txt")
        # print("final loss:",final_loss)
        
        max_val=np.max(final_loss)
        max_idx=np.argmax(final_loss)
        # print(max_val,final_loss[max_idx])
        # if max_val==final_loss[max_idx]:
        #     print("Not Match",max_val,final_loss[max_idx])

        if Y[models[max_idx]]==0:
            print(models[max_idx]+1)
            # print(max_val)
            f.write(str(models[max_idx]+1))
            f.write("\n")

        else:
            print(models[max_idx]+1,"Attacked")
            # print(max_val)
            f.write(str(models[max_idx]+1))
            f.write("   Attacked\n")

        delete_models.append(models[max_idx])
        models.remove(models[max_idx])
    f.write("-----------------------------")
    f.close()
    # Y_MNISTを保存します
    with open('delete_untarget_'+str(gr)+'_1_worst.pkl', 'wb') as f:
        pickle.dump(delete_models, f)


def show_res(idx):
    with open('Y_MNIST.pkl', 'rb') as f:
        Y_MNIST = pickle.load(f)
    # print(Y_MNIST)

    for I in range(10):
        for J in range(len(Y_MNIST[I])):
            num=Y_MNIST[I][J][0]
            flag=False
            for K in range(len(idx)):
                print(idx[K],Y_MNIST[I][J][K])
                if num!=Y_MNIST[I][J][K]:
                    flag=True
            if flag:
                for K in range(len(idx)):
                    print(idx[K],Y_MNIST[I][J][K])

def pred_test(int):
    NNUM=[100,300,500,700,900]
    GGROUP=[400,300,500,700,900]
    gr=GGROUP[int]
    M=200
    # データの準備
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    Y_GROUP=[]
    for j in range(len(x_test)):
        Y_GROUP.append([])

    for j in range(M):
        txt="#group_"+str(j+1)+".h5"
        model=tf.keras.models.load_model(txt)
        #モデルを用いて推論の実行
        pred=model.predict(x_test,verbose=0)
        

        #予測値を整数値で取得
        out=np.argmax(pred, axis=1)

        # if j+1==4 or j+1==6:
        #     print(out)

        #結果の保存
        for k in range(len(x_test)):
            Y_GROUP[k].append(out[k])
    # print(Y_GROUP)
    
    # Y_MNISTを保存します
    with open('Test_untarget_'+str(gr)+'_1_worst.pkl', 'wb') as f:
        pickle.dump(Y_GROUP, f)
    with open('Test_answer.pkl', 'wb') as f:
        pickle.dump(y_test, f)

def acc_u(idx):
    NNUM=[100,300,500,700,900]
    GGROUP=[400,300,500,700,900]
    gr=GGROUP[idx]

    with open('Test_untarget_'+str(gr)+'_1_worst.pkl', 'rb') as f:
        Pred = pickle.load(f) 
    with open('Test_answer.pkl', 'rb') as f:
        Ans = pickle.load(f) 
    with open('delete_untarget_'+str(gr)+'_1_worst.pkl', 'rb') as f:
        delete_models = pickle.load(f) 
    
    MODELS=[]
    for I in range(200):
        MODELS.append(I)

    f =open('FLCert_untarget_'+str(gr)+'_1_worst.txt','a')
    nn=0
    # print(delete_models)
    res=[]

    while len(MODELS)>0:
        correct_num=0
        for i in range(len(Ans)):
            #i番目の画像に対する各モデルの推論結果を取得する配列の初期化
            ans=[]
            append_ans=ans.append
            for j in range(len(MODELS)):
                append_ans(Pred[i][MODELS[j]])
            #最頻値（多数決の結果を取得する）
            num=statistics.mode(ans)
            
            #多数決が正しかった場合は正答数を加算
            if Ans[i]==num:
                correct_num+=1
        acc=correct_num/len(Ans)
        # print(len(Ans))
        f.write("Number of Models:"+str(len(MODELS))+"\t Acc:"+str(acc)+"\n")
        res.append(acc)
        # print(delete_models[nn])
        MODELS.remove(delete_models[nn])
        nn+=1
    
    f.close()
    # Y_MNISTを保存します
    with open('Acc_untarget_'+str(gr)+'_1_worst.pkl', 'wb') as FFF:
        pickle.dump(res, FFF)
    
def acc_t(idx):
    NNUM=[100,300,500,700,900]
    GGROUP=[400,300,500,700,900]
    gr=GGROUP[idx]
    class1=0
    res=[]

    with open('Test_untarget_'+str(gr)+'_1_worst.pkl', 'rb') as f:
        Pred = pickle.load(f) 
    with open('Test_answer.pkl', 'rb') as f:
        Ans = pickle.load(f) 
    with open('delete_untarget_'+str(gr)+'_1_worst.pkl', 'rb') as f:
        delete_models = pickle.load(f) 
    
    MODELS=[]
    for I in range(200):
        MODELS.append(I)

    f =open('FLCert_untarget_'+str(gr)+'_1_worst.txt','a')
    nn=0
    # print(delete_models)

    while len(MODELS)>0:
        correct_num=0
        class_num=0
        for i in range(len(Ans)):
            #i番目の画像に対する各モデルの推論結果を取得する配列の初期化
            ans=[]
            append_ans=ans.append
            if Ans[i]==class1:
                for j in range(len(MODELS)):
                    append_ans(Pred[i][MODELS[j]])
                #最頻値（多数決の結果を取得する）
                num=statistics.mode(ans)
                class_num+=1
                #多数決が正しかった場合は正答数を加算
                if Ans[i]==num:
                    correct_num+=1
        acc=correct_num/class_num
        # print(class_num)
        f.write("Number of Models:"+str(len(MODELS))+"\t Target Acc:"+str(acc)+"\n")
        res.append(acc)
        MODELS.remove(delete_models[nn])
        nn+=1
    
    f.close() 
    # Y_MNISTを保存します
    with open('Acc_untarget_'+str(gr)+'_1_worst.pkl', 'wb') as FFF:
        pickle.dump(res, FFF) 


#パラメータ設定
M=200
N=1000
k=5

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179

idx=0
# pred_mnist(idx)
eval_1(N,M,idx)
# show_res([196,197,198])
pred_test(idx)
acc_u(idx)
