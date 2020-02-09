import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import operator
import torchvision.transforms
batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root='../ml/pymnist',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=torchvision.transforms.ToTensor(),  # 转换成tensor变量
                            download=False)  # 不从网络上下载图片
test_dataset = dsets.MNIST(root='../ml/pymnist',  # 选择数据的根目录
                           train=False,  # 选择测试集
                           transform=torchvision.transforms.ToTensor(),  # 转换成tensor变量
                           download=False)  # 不从网络上下载图片


def init_network():
    network = {}
    weight_scale = 1e-3
    network['W1'] = np.random.randn(784, 50) * weight_scale
    network['b1'] = np.ones(50)
    network['W2'] = np.random.randn(50, 100) * weight_scale
    network['b2'] = np.ones(100)
    network['W3'] = np.random.randn(100, 10) * weight_scale
    network['b3'] = np.ones(10)
    return network

def _relu(x):
    return np.maximum(0, x)

def _softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = x.dot(w1)
    z1 = _relu(a1)
    a2 = z1.dot(w2)
    z2 = _relu(a2)
    a3 = z2.dot(w3)
    y = _softmax(a3)
    return y
    # 找到在每列中评分值最大的索引（即预测的分类）

if __name__ == '__main__':
    x = test_dataset.test_data.numpy().reshape(-1, 28 * 28)
    labels = test_dataset.test_labels
    finallabels = labels.reshape(labels.shape[0], 1)
    bestloss = float('inf')
    network = init_network()
    x_batch = x
    y_batch = forward(network, x_batch)
    Yte_predict = np.argmax(y_batch , axis=1)
    one_hot_labels = torch.zeros(x.shape[0], 10).scatter_(1, finallabels, 1)
    true_labels = np.argmax(one_hot_labels.numpy(), axis=1)
    # 计算准确率
    print(np.mean(Yte_predict == true_labels))
