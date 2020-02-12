import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import operator
import torchvision.transforms
# import torch.transforms


# MNIST dataset
train_dataset = dsets.MNIST(root='../ml/pymnist',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=torchvision.transforms.ToTensor(),  # 转换成tensor变量
                            download=False)  # 不从网络上下载图片
test_dataset = dsets.MNIST(root='../ml/pymnist',  # 选择数据的根目录
                           train=False,  # 选择测试集
                           transform=torchvision.transforms.ToTensor(),  # 转换成tensor变量
                           download=False)  # 不从网络上下载图片


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # input_size代表输入的神经元个数； hidden_size代表隐藏层神经元的个数；output_size代表输出层神经元的个数；
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)          # 784 * 50 ,
        self.params['b1'] = np.zeros(hidden_size)                                               # 50
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)         # 50 * 10 ,
        self.params['b2'] = np.zeros(output_size)                                               # 10

    def loss(self, x, y):
        p = self.predict(x)
        return cross_entropy_error(p, y)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = _relu(a1)
        a2 = np.dot(z1, W2) + b2
        p = _softmax(a2)
        return p

    def accuracy(self, x, t):
        p = self.predict(x)
        y = np.argmax(t, axis=1)
        p = np.argmax(p, axis=1)
        accuracy = np.sum(p == y) / float(x.shape[0])
        return accuracy

    # x:输入数据, y:监督数据
    def numerical_gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


def _relu(in_data):
    return np.maximum(0, in_data)


def _softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)


# 计算基于小批次的损失函数的损失值
def cross_entropy_error(p, y):  # y代表真实值，p代表预测值
    delta = 1e-7
    batch_size = p.shape[0]
    return -np.sum(y * np.log(p + delta)) / batch_size


# 数值微分
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
        it.iternext()
    return grad


if __name__ == '__main__':
    x_train = train_dataset.train_data.numpy().reshape(-1, 28 * 28)
    y_train_tmp = train_dataset.train_labels.reshape(train_dataset.train_labels.shape[0], 1)
    y_train = torch.zeros(y_train_tmp.shape[0], 10).scatter_(1, y_train_tmp, 1).numpy()
    x_test = test_dataset.test_data.numpy().reshape(-1, 28 * 28)
    y_test_tmp = test_dataset.test_labels.reshape(test_dataset.test_labels.shape[0], 1)
    y_test = torch.zeros(y_test_tmp.shape[0], 10).scatter_(1, y_test_tmp, 1).numpy()

    # 超参数
    iters_num = 1000  # 适当设定循环的次数
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.001
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        grad = network.numerical_gradient(x_batch, y_batch)
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        # 记录学习过程
        loss = network.loss(x_batch, y_batch)
        if i % 5 == 0:
            print(loss)
            print(network.accuracy(x_test, y_test))
