import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root = './ml/pymnist', #选择数据的根目录
                            train = True, #选择训练集
                            transform = None, #不考虑使用任何数据预处理
                            download = True) #从网络上下载图片
test_dataset = dsets.MNIST(root = './ml/pymnist', #选择数据的根目录
                           train = False, #选择测试集
                           transform = None, #不考虑使用任何数据预处理
                           download = True) #从网络上下载图片
# #加载数据
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True) #将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = True)
# print("train_data:", train_dataset.train_data.size())
# print("train_labels:", train_dataset.train_labels.size())
# print("test_data:", test_dataset.test_data.size())
# print("test_labels:", test_dataset.test_labels.size())

digit = train_loader.dataset.train_data[0] #取第一个图片的数据
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
# print(train_loader.dataset.train_labels[0]) #输出对应的标签


