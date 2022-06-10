from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt


def load_dataset(dataset_path, n_train_data):
    """加载数据集，对数据进行预处理，并划分训练集和验证集
    :param dataset_path: 数据集文件路径
    :param n_train_data: 训练集的数据量
    :return: 划分好的训练集和验证集
    """
    dataset = []
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(dataset_path, 'r') as file:
        # 读取CSV文件，以逗号为分隔符
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            # 将字符串类型的特征值转换为浮点型
            row[0:4] = list(map(float, row[0:4]))
            # 将标签替换为整型
            row[4] = label_dict[row[4]]
            # 将处理好的数据加入数据集中
            dataset.append(row)

    # 对数据进行归一化处理
    dataset = np.array(dataset)
    mms = MinMaxScaler()
    for i in range(dataset.shape[1] - 1):
        dataset[:, i] = mms.fit_transform(dataset[:, i].reshape(-1, 1)).flatten()

    # 将类标转为整型
    dataset = dataset.tolist()
    for row in dataset:
        row[4] = int(row[4])
    # 打乱数据集
    random.shuffle(dataset)

    # 划分训练集和验证集
    train_data = dataset[0:n_train_data]
    val_data = dataset[n_train_data:]

    return train_data, val_data


def fun_z(weights, inputs):
    """计算感知器的判别函数：z = weight * inputs
    :param weights: 权重矩阵
    :param inputs: 一个样本数据
    :return: 感知器的判别函数
    """
    z = 0
    for i in range(len(weights)):
        z += weights[i] * inputs[i]
    return z


def update_parameters(perceptron, row, l_rate):
    """更新感知器的参数（权重矩阵）
    :param perceptron: 感知器
    :param row: 一个样本数据
    :param l_rate: 学习率
    :return:
    """
    for i in range(len(perceptron)):
        inputs = row[:-1]
        outputs = row[-1]
        if i != outputs:
            if fun_z(perceptron[outputs],inputs) <= fun_z(perceptron[i],inputs):
                for j in range(len(inputs)):
                    perceptron[outputs][j] += l_rate * inputs[j]
                    perceptron[i][j] -= l_rate * inputs[j]



def initialize_perceptron(n_inputs,n_outputs):
    """初始化感知器（初始化权重矩阵）
    :param n_inputs: 特征列数
    :param n_outputs: 分类的总类别数
    :return: 初始化后的感知器
    """
    perceptron = [[random.random() for i in range(n_inputs)] for i in range(n_outputs)]
    return perceptron


def train(train_data, l_rate, epochs, val_data):
    """训练感知器（迭代n_epoch个回合）
    :param train_data: 训练集
    :param l_rate: 学习率
    :param epochs: 迭代的回合数
    :param val_data: 验证集
    :return: 训练好的感知器
    """
    # 获取特征列数
    n_inputs = len(train_data[0]) - 1
    # 获取分类的总类别数
    n_outputs = len(set([row[-1] for row in train_data]))
    # 初始化感知器
    perceptron = initialize_perceptron(n_inputs, n_outputs)

    acc = []
    for epoch in range(epochs):  # 训练epochs个回合
        for row in train_data:
            # 更新参数
            update_parameters(perceptron, row, l_rate)
        # 保存当前epoch模型在验证集上的准确率
        acc.append(validation(perceptron, val_data))
    # 绘制出训练过程中模型在验证集上的准确率变化
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(acc)
    plt.show()

    return perceptron


def validation(perceptron, val_data):
    """测试模型在验证集上的效果
    :param perceptron: 感知器
    :param val_data: 验证集
    :return: 模型在验证集上的准确率
    """
    # 获取预测类标
    predicted_label = []
    for row in val_data:
        prediction = predict(perceptron, row)
        predicted_label.append(prediction)
    # 获取真实类标
    actual_label = [row[-1] for row in val_data]
    # 计算准确率
    accuracy = accuracy_calculation(actual_label, predicted_label)
    return accuracy


def accuracy_calculation(actual_label, predicted_label):
    """计算准确率
    :param actual_label: 真实类标
    :param predicted_label: 模型预测的类标
    :return: 准确率（百分制）
    """
    correct_count = 0
    for i in range(len(actual_label)):
        if actual_label[i] == predicted_label[i]:
            correct_count += 1
    return correct_count / float(len(actual_label)) * 100.0


def predict(perceptron, row):
    """使用模型对当前输入的数据进行预测
    :param perception: 感知器
    :param row: 一个数据样本
    :return: 预测结果
    """
    outputs = []
    for i in range(len(perceptron)):
        inputs = row[:-1]
        outputs.append(fun_z(perceptron[i],inputs))
    return outputs.index(max(outputs))


if __name__ == "__main__":
    file_path = './iris.csv'

    # 参数设置
    l_rate = 0.1  # 学习率
    epochs = 300  # 迭代训练的次数
    n_train_data = 130  # 训练集的大小（总共150条数据，训练集130条，验证集20条）

    # 加载数据并划分训练集和验证集
    train_data, val_data = load_dataset(file_path, n_train_data)
    # 训练模型
    perceptron = train(train_data, l_rate, epochs, val_data)
