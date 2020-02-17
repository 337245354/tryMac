import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import operator
import torchvision.transforms
from collections import OrderedDict


class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = np.maximum(0, x)
        out = self.x
        return out

    def backward(self, dout):
        dx = dout
        dx[self.x <= 0] = 0
        return dx


class _sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.p = None  # Softmax的输出
        self.y = None  # 监督数据代表真值，one-hot vector

    def forward(self, x, y):
        self.y = y
        self.p = softmax(x)
        self.loss = cross_entropy_error(self.p, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.p - self.y) / batch_size
        return dx


def softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)


def cross_entropy_error(p, y):
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


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x:输入数据, y:监督数据
    def loss(self, x, y):
        p = self.predict(x)
        return self.lastLayer.forward(p, y)

    def accuracy(self, x, y):
        p = self.predict(x)
        p = np.argmax(y, axis=1)
        if y.ndim != 1: y = np.argmax(y, axis=1)
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

    def gradient(self, x, y):
        # forward
        self.loss(x, y)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads


if __name__ == '__main__':
    train_dataset = dsets.MNIST(root='../ml/pymnist',  # 选择数据的根目录
                                train=True,  # 选择训练集
                                transform=torchvision.transforms.ToTensor(),  # 转换成tensor变量
                                download=False)  # 不从网络上下载图片
    test_dataset = dsets.MNIST(root='../ml/pymnist',  # 选择数据的根目录
                               train=False,  # 选择测试集
                               transform=torchvision.transforms.ToTensor(),  # 转换成tensor变量
                               download=False)  # 不从网络上下载图片

    x_train = train_dataset.train_data.numpy().reshape(-1, 28 * 28)
    y_train_tmp = train_dataset.train_labels.reshape(train_dataset.train_labels.shape[0], 1)
    y_train = torch.zeros(y_train_tmp.shape[0], 10).scatter_(1, y_train_tmp, 1).numpy()
    x_test = test_dataset.test_data.numpy().reshape(-1, 28 * 28)
    y_test_tmp = test_dataset.test_labels.reshape(test_dataset.test_labels.shape[0], 1)
    y_test = torch.zeros(y_test_tmp.shape[0], 10).scatter_(1, y_test_tmp, 1).numpy()

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:100]
    y_batch = y_train[:100]
    grad_numerical = network.numerical_gradient(x_batch, y_batch)  # 基于数值微分
    grad_backprop = network.gradient(x_batch, y_batch)   # 基于误差反向传播
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))
