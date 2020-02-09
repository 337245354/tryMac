import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import operator
import torchvision.transforms
# import torch.transforms

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
    network={}
    weight_scale = 1e-3
    network['W1']=np.random.randn(784,50) * weight_scale
    network['b1']=np.ones(50)
    network['W2']=np.random.randn(50,100) * weight_scale
    network['b2']=np.ones(100)
    network['W3']=np.random.randn(100,10) * weight_scale
    network['b3']=np.ones(10)
    return network

def forward(network,x):
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = x.dot(w1) + b1
    z1 = _relu(a1)
    a2 = z1.dot(w2) + b2
    z2 = _relu(a2)
    a3 = z2.dot(w3) + b3
    y = a3
    return y

def _relu(x):
    return np.maximum(0,x)


if __name__ == '__main__':
    print("start")
    network = init_network()
    accuracy_cnt = 0
    x = test_dataset.test_data.numpy().reshape(-1, 28 * 28)
    labels = test_dataset.test_labels.numpy() #tensor转numpy
    for i in range(len(x)):
        y = forward(network, x[i])
        p = np.argmax(y) #获取概率最高的元素的索引
        if p == labels[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x) * 100) + "%")
