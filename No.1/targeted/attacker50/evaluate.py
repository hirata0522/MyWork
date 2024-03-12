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
#テスト行列Xとモデルの正常・異常を格納した行列Yを入力してCOMPを計算する関数
def calculate_COMP(X,Y):    
    
    row = len(X)
    col = len(X[0])
    
    ans=[1]*col

    for i in range(row):
        if Y[i]==0:
            for j in range(col):
                if X[i][j]==1:
                    ans[j]=0
    
    return ans

#テスト行列Xとモデルの正常・異常を格納した行列Yを入力してSCOMPを計算する関数
def calculate_DD(X,Y):
    row = len(X)
    col = len(X[0])
    
    ans=calculate_COMP(X,Y)
    # print("COMP")
    # print(ans)
    
    DD=[]
    
    for i in range(row):
        if Y[i]==1:
            cnt=0
            num=-1
            for j in range(col):
                if X[i][j]==1 and ans[j]==1:
                    cnt+=1
                    num=j

            if cnt==1:
                DD.append(num)
    # print("DD")
    # print(DD)

    ans=[0]*col

    for i in range(len(DD)):
        ans[DD[i]]=1
    
    return ans

#テスト行列Xとモデルの正常・異常を格納した行列Yを入力してSCOMPを計算する関数
def calculate_SCOMP(X,Y):    
    
    row = len(X)
    col = len(X[0])
    
    ans=calculate_COMP(X,Y)
    # print("COMP")
    # print(ans)
    
    DD=[]
    
    for i in range(row):
        if Y[i]==1:
            cnt=0
            num=-1
            for j in range(col):
                if X[i][j]==1 and ans[j]==1:
                    cnt+=1
                    num=j

            if cnt==1:
                DD.append(num)
    #print("DD")
    #print(DD)


    ans_new=[-1]*col

    for i in range(len(DD)):
        ans_new[DD[i]]=1
    #print("ans_new")
    #print(ans_new)

    for i in range(col):
        if ans[i]==0:
            ans_new[i]=0
    # print("ans_new1")
    # print(ans_new)
    
    ANS_NEW=np.copy(ans_new)
    # print("ans_new")
    # print(ans_new)
    # print("ANS_NEW")
    # print(ANS_NEW)

    for index, value in enumerate(ANS_NEW):
        if value == -1:
         ANS_NEW[index] = 0
    
    # print("ans_new")
    # print(ans_new)
    # print("ANS_NEW")
    # print(ANS_NEW)    

    if -1 not in ans_new:
        #print("no")
        return ans_new

    #print("SCOMP")
    X=np.array(X)
    ANS_NEW=np.array(ANS_NEW)
    Y=np.array(Y)
    #print(ans_new)
    #print(ans_new[1])

    # print("ans_new")
    # print(ans_new)
    # print("ANS_NEW")
    # print(ANS_NEW) 
    if np.array_equal(X@ANS_NEW,Y):
            #print("YES")
            return ANS_NEW

    while(True):
        
        max=0
        max_idx=-1
        
        for i in range(col):
            #print("ans_new[i]")
            #print(ans_new[i])
            if ans_new[i]==-1:
                #print("ans_new[i]=-1")
                NUM=0
                for j in range(row):
                    if X[j][i]==1 and Y[j]==1:
                        NUM+=1
                #print("NUM")
                #print(NUM)
                if NUM>max:
                    max_idx=i
                    max=NUM
                # print(NUM)
            #print("max_index")

        if max_idx==-1:
            break

        # print("max_index")
        # print(max_idx)

        ans_new[max_idx]=1
        ANS_NEW[max_idx]=1
        # print("ans_new")
        # print(ans_new)
        # print("ANS_NEW")
        # print(ANS_NEW)  
        


        test=X@ANS_NEW
        

        for index, value in enumerate(test):
          if value > 0:
                test[index] = 1
        
        #print("test")
        #print(test)
        
        if np.array_equal(test,Y):
            #print("yes")
            break


    
    return ANS_NEW

#FP/FNを計算する関数
def calculate_FP_FN(N,num,comp):
    FP=0
    FN=0


    for i in range(num):
        FN+=1-comp[i]
    
    FN=FN/num

    for i in range(N-num):
        FP+=comp[num+i]

    FP=FP/(N-num)

    return[FP,FN]

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
    
#FLCertを実行する関数
def eval(N,M,num):
    # データの準備
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    [group_x,group_y]=load_data('#MNIST_','txt',x_train,y_train)
    
    clients=load_arrays_from_files(file_prefix="#clients_",file_extension="txt")[0]
    
    # #グループ分け
    # X=generate_test_matrix(M,N,k)

    # #行列のファイル出力
    # save_matrix_to_file(X, file_name='#matrix.txt')
    X=load_matrix_from_file(file_name='#matrix.txt')
    Y=create_if_group_poisoned(X,num)
    # print(X)

    group_clients=get_indices_of_ones(X)
    # print(group_clients)
    
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
    Y_MNIST=[]
    for i in range(10):
        Y_GROUP=[]
        for j in range(len(group_x[i])):
            Y_GROUP.append([])

        for j in range(M):
            txt="#group_"+str(j+1)+".h5"
            model=tf.keras.models.load_model(txt)
            #モデルを用いて推論の実行
            pred=model.predict(group_x[i])
            

            #予測値を整数値で取得
            out=np.argmax(pred, axis=1)

            if j+1==4 or j+1==6:
                print(out)

            #結果の保存
            for k in range(len(group_x[i])):
                Y_GROUP[k].append(out[k])
        # print(Y_GROUP)

        Y_MNIST.append(Y_GROUP)

    #各ローカルデータセットの損失を格納する配列
    loss=[]

    cs=[True,True,True,True,True,True,True,True,True,True]
    
    for i in range(10):
        loss_group=np.zeros(M)
        num_cs=0
        for j in range(len(group_x[i])):
            #各数字が何回出てきたかカウント
            num_pred=[0,0,0,0,0,0,0,0,0,0]
            for k in range(10):
                num_pred[k]+=Y_MNIST[i][j].count(k)
            
            if num_pred.count(0)!=9:
                num_cs+=1
            
            #各入力に対する損失の計算(損失を格納する配列に加算)
            for k in range(M):
                
                if Y_MNIST[i][j][k]==group_y[i][j]:
                    loss_group[k]+=0
                else:
                    loss_group[k]+=1/num_pred[Y_MNIST[i][j][k]]
                    # print(num_pred)
                    # print(group_y[i][j])
                    # print("MNIST",i,"Group",k+1,"Pred",Y_MNIST[i][j][k])
                    # print("No",loss_group[k])
        #要素数で割り平均の計算
        for j in range(M):
            if num_cs!=0:
                # print(len(group_x[i]))
                loss_group[j]=loss_group[j]/num_cs
        
        if num_cs==0:
            cs[i]=False
        print(num_cs)

        
        #各ローカルデータセットの損失を格納する配列に格納
        loss.append(loss_group)
    
    print("loss:",loss)
    save_arrays_to_files_float(loss,file_prefix="#MNIST_loss_",file_extension=".txt")

    final_loss=np.zeros(M)

    cs_clients=0

    for i in range(N):
        tmp=np.zeros(M)

        if i<num:
            num0=0
            for j in range(M):
                if X[j][i]==0:
                    num0+=1
            for j in range(M):
                if X[j][i]==0:
                    tmp[j]=9/num0
                    # ここで(クラス数-1)掛けるのが最適

            # print("attacker",tmp)
        else:
            tmp=loss[clients[i]]
            # print("honest client",tmp)
        if cs[clients[i]]:
            
            for j in range(M):
                final_loss[j]=tmp[j]+final_loss[j]
                cs_clients+=1
            # print("client",i,"tmp_loss",tmp,"final_loss",final_loss)
                
    
    for i in range(M):
        if cs_clients!=0:
            final_loss[i]=final_loss[i]/cs_clients

    save_arrays_to_files_float([final_loss],"#loss","txt")
    print("final loss:",final_loss)

    threshold=calculate_threshold(final_loss)
    print("threshold",threshold)

    pred_group=np.zeros(M)

    for i in range(M):
        if final_loss[i]>threshold:
            pred_group[i]=1

    save_arrays_to_files([pred_group],"#group_pred","txt")
    
    FP=0
    FN=0
    TP=0
    TN=0

    for i in range(M):
        if Y[i]==0:
            if pred_group[i]==0:
                TN+=1
            else:
                FP+=1
        else:
            if pred_group[i]==0:
                FN+=1
            else:
                TP+=1
    
    pred_clients=calculate_SCOMP(X,pred_group)
    [FPP_clients,FNP_clients]=calculate_FP_FN(N,num,pred_clients)

    f = open("#FLCert_target_evaluate.txt",'a')
    f.write('Parameter\n\n')

    f.write('Number of Groups\n')
    f.write('M=')
    f.write(str(M))
    f.write('\n')
    f.write('\n')

    f.write('Number of Clients\n')
    f.write('N=')
    f.write(str(N))
    f.write('\n')
    f.write('\n')

    k=5

    f.write('Clients for each Groups\n')
    f.write('k=')
    f.write(str(k))
    f.write('\n')
    f.write('\n')

    f.write('Threshold\n')
    f.write(str(threshold))
    f.write('\n')
    f.write('\n')

    f.write('\n\n Group Prediction\n')
    f.write("\nTN=")
    f.write(str(TN))
    f.write("\nFP=")
    f.write(str(FP))
    f.write("\nFN=")
    f.write(str(FN))
    f.write("\nTP=")
    f.write(str(TP))

    f.write('\n')
    f.write("\nFP percentage=")
    f.write(str(FP/(TN+FP)))
    f.write("\nFN percentage=")
    f.write(str(FN/(TP+FN)))

    f.write('\nClients Prediction\n')
    f.write("\nTN=")
    f.write(str((1-FPP_clients)*(N-num)))
    f.write("\nFP=")
    f.write(str(FPP_clients*(N-num)))
    f.write("\nFN=")
    f.write(str(FNP_clients*num))
    f.write("\nTP=")
    f.write(str((1-FNP_clients)*num))

    f.write('\n')
    f.write("\nFP percentage=")
    f.write(str(FPP_clients))
    f.write("\nFN percentage=")
    f.write(str(FNP_clients))

    f.write("\n-----------------------\n")

    f.close()



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
M=200
N=1000
k=5

# not attacked model: 150/ attacker: 57
# not attacked model: 125/ attacker: 91
# not attacked model: 100/ attacker: 130
# not attacked model:  75/ attacker: 179

num=130

eval(N,M,num)

# # データの準備
# mnist = keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# acc = majority_vote(x_test,y_test,M)

# print('\nTest accuracy:', acc)


# f = open("#FLCert_target_parallel.txt",'a')

# f.write('Parameter\n\n')

# f.write('Number of Groups\n')
# f.write('M=')
# f.writelines(str(M))
# f.write('\n')
# f.write('\n')

# f.write('Number of Clients\n')
# f.write('N=')
# f.writelines(str(N))
# f.write('\n')
# f.write('\n')

# f.write('Clients for each Groups\n')
# f.write('k=')
# f.writelines(str(k))
# f.write('\n')
# f.write('\n')

# f.write("algorithm:FedAvg\n")
# f.write("dataset:MNIST\n\n")

# f.write("number of clients:"+str(N)+"\n")
# f.write("number of groups:"+str(M)+"\n")
# f.write("clients for each group:"+str(k)+"\n\n")

# f.write("learning rate:0.001\n")
# f.write("batch size:"+str(batch_size)+"\n")
# f.write("global iteration:"+str(global_iter)+"\n")
# f.write("local iteration:"+str(epochs)+"\n\n")

# f.write("model:CNN\n\n")

# f.write("layer:\n")
# f.write("Input:28 x 28 x 1\n")
# f.write("Convolution + ReLU : 3 x 3 x 30\n")
# f.write("Max Pooling : 2 x 2\n")
# f.write("Convolution + ReLU : 3 x 3 x 50\n")
# f.write("Max Pooling : 2 x 2\n")
# f.write("Fully Connected + ReLU : 100\n")
# f.write("Softmax : 10\n\n")

# for i in range(M):
#     txt="#group_"+str(i+1)+".h5"
#     model1=tf.keras.models.load_model(txt)
#     model=create_model()
#     model.set_weights(model1.get_weights())
#     [main_acc,target_acc] = evaluate_target_attack(x_test,y_test,model,0)
#     f.write('\nGroup '+str(i+1)+' /Main accuracy:'+str(main_acc)+' /Target accuracy:'+str(target_acc))

# f.write("\n\ntest acc:"+str(acc)+"\n")

# time_end=time.time()
# time_use=time_end-time_strt

# f.write("time:"+str(time_use)+"\n\n")
# f.write("-----------------------\n")

# f.close()