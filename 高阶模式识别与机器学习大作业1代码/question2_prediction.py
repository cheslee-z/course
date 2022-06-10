import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense,Conv2D,Flatten,BatchNormalization,Activation,DepthwiseConv2D,AveragePooling2D,SeparableConv2D
from tensorflow.keras.constraints import max_norm

Sx_test_data_path = r'C:/Users/chesily/Desktop/附件1-P300脑机接口数据/S4/S4_test_data.xlsx'
Sx_test_event_path = r'C:/Users/chesily/Desktop/附件1-P300脑机接口数据/S4/S4_test_event.xlsx'

char_Sx_test = {'P13':["char13"], 'P14':["char14"],
                'P15':["char15"], 'P16':["char16"],
                'P17':["char17"], 'P18':["char18"],
                'P19':["char19"], 'P20':["char20"],
                'P21':["char21"], 'P22':["char22"]}

rc_dic = [['A', 'B', 'C', 'D', 'E', 'F'],
          ['G', 'H', 'I', 'J', 'K', 'L'],
          ['M', 'N', 'O', 'P', 'Q', 'R'],
          ['S', 'T', 'U', 'V', 'W', 'X'],
          ['Y', 'Z', '1', '2', '3', '4'],
          ['5', '6', '7', '8', '9', '0']]

def test_preprocess(Sx_test_data_path, Sx_test_event_path):

    x_test = []  #测试集数据
    y_test = []  #测试集标签
    for chari in list(char_Sx_test.keys()):
        #加载数据集
        Sx_chari_data = pd.read_excel(Sx_test_data_path, sheet_name = char_Sx_test[chari][0], header=None)
        Sx_chari_index = pd.read_excel(Sx_test_event_path, sheet_name = char_Sx_test[chari][0], header=None)

        #数据标签清洗
        index = []
        for i in range(Sx_chari_index.shape[0]):
            if Sx_chari_index.iloc[i,0] < 100:
                index_i =Sx_chari_index.iloc[i,:].tolist()
                index.append(index_i)
        index_df = pd.DataFrame(index, index=None)
        y_test.append(index_df.iloc[:,0].tolist())

        #滤波去噪
        filter_num, filter_den =  signal.butter(5,[0.004,0.08],'bandpass')
        for i in range(20):
            Sx_chari_data.iloc[:,i] = signal.filtfilt(filter_num, filter_den, Sx_chari_data.iloc[:,i])
        
        #特征提取
        test_data = []
        index_list = index_df.iloc[:,1].tolist()
        for index in index_list:
            for i in range(index+37, index+165):
                data = Sx_chari_data.loc[i]
                test_data.append(data)
        test_data = pd.DataFrame(test_data)
        test_data = test_data.drop(labels = [4, 8, 18], axis = 1)

        #信号降采样
        test_data_downsampling = []
        for i in range(test_data.shape[0]):
            if i % 4 == 0:
                data = test_data.iloc[i,:]
                test_data_downsampling.append(data)
        test_data_downsampling = pd.DataFrame(test_data_downsampling)

        #批量归一化
        test_data_normalize = test_data_downsampling.apply(lambda x: (x - np.mean(x)) / np.std(x))

        #数据拉直及合并
        for i in range(index_df.shape[0]):
            x_test.append(np.array(test_data_normalize.iloc[i*32:(i+1)*32,:]).reshape(-1))

    y_test = np.array(y_test).reshape(-1)

    return x_test, y_test

def main():
   
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

    model_cnn.compile(optimizer = tf.keras.optimizers.Adam(0.001),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                  metrics = ['sparse_categorical_accuracy'])

    print('---------------- load model------------')

    model_cnn.load_weights('cnn.ckpt')

    x_test, y_test = test_preprocess(Sx_test_data_path, Sx_test_event_path)
    x_test = np.array(x_test).reshape(np.array(x_test).shape[0],32,17,1)
    y_pred = model_cnn.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred_label = []
    for i in range(y_pred.shape[0]):
        y_pred_label.append(1 if y_pred.iloc[i,0]<y_pred.iloc[i,1] else 0)

    result_label = np.ones((5,10))
    result_label *= 10
    result_label = result_label.astype(str)
    for i in range(10):
        result = [0,0,0,0,0,0,0,0,0,0,0,0]
        for j in range(5):
            for k in range(12):
                if y_pred_label[i*60+j*12+k] == 1:
                    result[y_test[i*60+j*12+k]-1] += 1
            row = np.argmax(result[:6])
            col = np.argmax(result[6:])
            result_label[j][i] = rc_dic[row][col]
    print(result_label)

if __name__ == "__main__":
    main()