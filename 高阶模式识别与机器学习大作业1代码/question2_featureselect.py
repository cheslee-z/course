import pandas as pd
import numpy as np
from scipy import signal
from sklearn.ensemble import ExtraTreesClassifier

io_Sx = r'C:/Users/chesily/Desktop/附件1-P300脑机接口数据/S5/S5_train_data.xlsx'
io_Sx_index = r'C:/Users/chesily/Desktop/附件1-P300脑机接口数据/S5/S5_train_event.xlsx'

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

        #信号降采样
        train_data_downsampling = []
        for i in range(train_data.shape[0]):
            if i % 4 == 0:
                data = train_data.iloc[i,:]
                train_data_downsampling.append(data)
        train_data_downsampling = pd.DataFrame(train_data_downsampling)

        #批量归一化
        train_data_normalize = train_data_downsampling.apply(lambda x: (x - np.mean(x)) / np.std(x))

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

    #将数据按照4:1的比例划分为训练集和测试集
    x_train = np.vstack([x_p[0:int(np.array(x_p).shape[0])],x_n[0:int(np.array(x_n).shape[0])]])
    y_train = np.vstack([y_p[0:int(np.array(y_p).shape[0])],y_n[0:int(np.array(y_n).shape[0])]])

    np.random.seed(123)
    np.random.shuffle(x_train)
    np.random.seed(123)
    np.random.shuffle(y_train)

    return x_train, y_train


def main():
    data, label = preprocess(io_Sx, io_Sx_index)  

    model_tree = ExtraTreesClassifier()
    
    #模型训练及准确率计算
    model_tree = model_tree.fit(data, label)
    score = np.array(model_tree.feature_importances_).reshape(32,20)
    score_20 = np.zeros((20,1))
    for i in range(score.shape[1]):
        for j in range(score.shape[0]):
            score_20[i] += score[j][i]
    
    score_20 = np.around(score_20*100, decimals=2)
    print(score_20)
if __name__ == "__main__":
    main()