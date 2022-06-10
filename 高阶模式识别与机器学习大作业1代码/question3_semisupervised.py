import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense,Conv2D,Flatten,BatchNormalization,Activation,DepthwiseConv2D,AveragePooling2D,SeparableConv2D
from tensorflow.keras.constraints import max_norm

io_Sx = r'C:/Users/chesily/Desktop/附件1-P300脑机接口数据/S4/S4_train_data.xlsx'
io_Sx_index = r'C:/Users/chesily/Desktop/附件1-P300脑机接口数据/S4/S4_train_event.xlsx'

char_Sx = {'P1':["char01(B)", 1, 8], 'P2':["char02(D)", 1, 10],
           'P3':["char03(G)", 2, 7], 'P4':["char04(L)", 2, 12],
           'P5':["char05(O)", 3, 9], 'P6':["char06(Q)", 3, 11],
           'P7':["char07(S)", 4, 7], 'P8':["char08(V)", 4, 10],
           'P9':["char09(Z)", 5, 8], 'P10':["char10(4)", 5, 12],
           'P11':["char11(7)", 6, 9], 'P12':["char12(9)", 6, 11]}

def preprocess(io_Sx, io_Sx_index):

    x_p = []  #1样本数据
    y_p = []  #1样本标签
    x_n = []  #0样本数据
    y_n = []  #0样本标签
    for chari in list(char_Sx.keys()):
        #加载数据集
        data_Sx_chari = pd.read_excel(io_Sx, sheet_name = char_Sx[chari][0], header=None)
        data_Sx_chari_index = pd.read_excel(io_Sx_index, sheet_name = char_Sx[chari][0], header=None)

        #数据标签清洗
        index = []
        for i in range(data_Sx_chari_index.shape[0]):
            if data_Sx_chari_index.iloc[i,0] < 100:
                index_i = data_Sx_chari_index.iloc[i,:].tolist()
                index.append(index_i)
        index_df = pd.DataFrame(index, index=None)
        # print(index_df)

        #数据标签制作
        index_df['2'] = 0
        for i in range(index_df.shape[0]):
            if index_df.iloc[i,0] == char_Sx[chari][1] or index_df.iloc[i,0] == char_Sx[chari][2]:
                index_df.iloc[i,2] = 1
            else:
                index_df.iloc[i,2] = 0

        #滤波去噪
        filter_num, filter_den =  signal.butter(5,[0.004,0.08],'bandpass')
        for i in range(20):
            data_Sx_chari.iloc[:,i] = signal.filtfilt(filter_num, filter_den, data_Sx_chari.iloc[:,i])
        
        #特征提取
        train_data = []
        index_list = index_df.iloc[:,1].tolist()
        for index in index_list:
            for i in range(index+37, index+165):
                data = data_Sx_chari.loc[i]
                train_data.append(data)
        train_data = pd.DataFrame(train_data)
        # print(train_data)
        train_data = train_data.drop(labels = [4, 8, 18], axis = 1)

        #信号降采样
        train_data_downsampling = []
        for i in range(train_data.shape[0]):
            if i % 4 == 0:
                data = train_data.iloc[i,:]
                train_data_downsampling.append(data)
        train_data_downsampling = pd.DataFrame(train_data_downsampling)

        #批量归一化
        train_data_normalize= train_data_downsampling.apply(lambda x: (x - np.mean(x)) / np.std(x))

        #数据拉直及合并
        for i in range(index_df.shape[0]):
            if index_df.iloc[i,2] == 1:
                x_p.append(np.array(train_data_normalize.iloc[i*32:(i+1)*32,:]).reshape(-1))
                y_p.append(np.array([1]))
            if index_df.iloc[i,2] == 0:
                x_n.append(np.array(train_data_normalize.iloc[i*32:(i+1)*32,:]).reshape(-1))
                y_n.append(np.array([0]))

    #均衡样本标签，使1标签样本数和0标签样本数相同
    x_p = np.repeat(x_p,5,axis=0)
    y_p = np.repeat(y_p,5,axis=0)

    #将数据乱序
    np.random.shuffle(x_p)
    np.random.shuffle(x_n)

    #将数据按照4:4:2的比例划分为有标签训练集、无标签训练集和测试集
    x_train = np.vstack([x_p[0:int(np.array(x_p).shape[0]*0.4)],x_n[0:int(np.array(x_n).shape[0]*0.4)]])
    y_train = np.vstack([y_p[0:int(np.array(y_p).shape[0]*0.4)],y_n[0:int(np.array(y_n).shape[0]*0.4)]])
    x_mask = np.vstack([x_p[int(np.array(x_p).shape[0]*0.4):int(np.array(x_p).shape[0]*0.8)],x_n[int(np.array(x_n).shape[0]*0.4):int(np.array(x_n).shape[0]*0.8)]])
    x_val = np.vstack([x_p[int(np.array(x_p).shape[0]*0.8):],x_n[int(np.array(x_n).shape[0]*0.8):]])
    y_val = np.vstack([y_p[int(np.array(y_p).shape[0]*0.8):],y_n[int(np.array(y_n).shape[0]*0.8):]])

    np.random.seed(123)
    np.random.shuffle(x_train)
    np.random.seed(123)
    np.random.shuffle(x_mask)
    np.random.seed(123)
    np.random.shuffle(x_val)
    np.random.seed(123)
    np.random.shuffle(y_train)
    np.random.seed(123)
    np.random.shuffle(y_val)

    return x_train, y_train, x_mask, x_val, y_val

def main():
    x_train, y_train, x_mask, x_val, y_val = preprocess(io_Sx, io_Sx_index)
    
    x_train =x_train.reshape(x_train.shape[0],32,17,1)
    x_mask =x_mask.reshape(x_mask.shape[0],32,17,1)
    x_val =x_val.reshape(x_val.shape[0],32,17,1)
    #CNN模型
    model_cnn = tf.keras.Sequential([Conv2D(filters = 8, kernel_size = (15,1), padding = 'same', use_bias = False),
                                BatchNormalization(),
                                DepthwiseConv2D(kernel_size = (1,17), use_bias = False, depth_multiplier = 2,  depthwise_constraint = max_norm(1.)),
                                BatchNormalization(),
                                Activation('elu'),
                                AveragePooling2D((4, 1)),
                                Dropout(0.25),
                                SeparableConv2D(filters = 16, kernel_size = (16,1), use_bias = False, padding = 'same'),
                                BatchNormalization(),
                                Activation('elu'),
                                AveragePooling2D((8, 1)),
                                Dropout(0.25),
                                Flatten(),
                                Dense(2, activation = 'softmax', kernel_constraint = max_norm(0.25))])

    model_cnn.compile(optimizer = tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                  metrics = ['sparse_categorical_accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'cnn.ckpt', save_weights_only = True, save_best_only = True)
    history = model_cnn.fit(x_train, y_train, batch_size = 20, epochs = 50, validation_data = (x_val,  y_val), validation_freq = 1, callbacks = [cp_callback])

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    mask_x_colle = []
    mask_y_colle = []
    temp = []

    while(len(x_mask) > 0):
        for x in x_mask:
            y_ = model_cnn.predict(x.reshape(1, x.shape[0], x.shape[1], 1))
            prob = np.max(y_)
            if prob > 0.8:
                mask_x_colle.append(x)
                mask_y_colle.append(np.argmax(y_, axis = 1))
            else:
                temp.append(x)
        if len(mask_x_colle) == 0:
            break
        x_train = np.vstack([x_train, mask_x_colle])
        y_train = np.vstack([y_train, mask_y_colle])
        print(mask_y_colle)
        print('x_train.shape',x_train.shape,'y_train.shape',y_train.shape)
        
        history = model_cnn.fit(x_train, y_train, batch_size = 20, epochs = 10, validation_data = (x_val, y_val), validation_freq = 1, callbacks=[cp_callback])
        acc.extend(history.history['sparse_categorical_accuracy'])
        val_acc.extend(history.history['val_sparse_categorical_accuracy'])
        mask_x_colle = []
        mask_y_colle = []
        x_mask = temp
        temp = []

    plt.subplot(2,1,1)
    plt.title('train_acc_and_test_acc')
    plt.plot(acc,label='train')
    # plt.xlabel('epoches')
    plt.ylabel('acc')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(val_acc,label='test')
    plt.xlabel('epoches')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
