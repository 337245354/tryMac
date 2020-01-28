import matplotlib
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import operator
import knn

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

digit = train_loader.dataset.train_data[0] #取第一个图片的数据
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()
# print(train_loader.dataset.train_labels[0]) #输出对应的标签

def getXmean(X_train):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    #将图片从二维展开为一维
    mean_image = np.mean(X_train, axis=0)
    #求出训练集中所有图片每个像素位置上的平均值
    return mean_image

def centralized(X_test,mean_image):
    X_test = np.reshape(X_test, (X_test.shape[0], -1)) #将图片从二维展开为一维
    X_test = X_test.astype(np.float)
    X_test -= mean_image #减去均值图像，实现零均值化
    return X_test

if __name__ == '__main__':
    #
    # X_train = train_loader.dataset.train_data.numpy() #需要转为numpy矩阵
    # X_train = X_train.reshape(X_train.shape[0],28*28)#需要reshape之后才能放入knn分类器,需要训练的图矩阵
    # y_train = train_loader.dataset.train_labels.numpy()#训练的label
    # X_test = test_loader.dataset.test_data[:1000].numpy()
    # X_test = X_test.reshape(X_test.shape[0],28*28)
    # y_test = test_loader.dataset.test_labels[:1000].numpy()
    # num_test = y_test.shape[0]
    # y_test_pred = knn.kNN_classify(5, 'M', X_train, y_train, X_test)
    # num_correct = np.sum(y_test_pred == y_test)
    # accuracy = float(num_correct) / num_test
    # print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

    # 使用预处理来完成knn
    X_train = train_loader.dataset.train_data.numpy()
    mean_image = getXmean(X_train)
    X_train = centralized(X_train, mean_image)
    y_train = train_loader.dataset.train_labels.numpy()
    X_test = test_loader.dataset.test_data[:1000].numpy()
    X_test = centralized(X_test, mean_image)
    y_test = test_loader.dataset.test_labels[:1000].numpy()
    num_test = y_test.shape[0]
    y_test_pred = knn.kNN_classify(5, 'M', X_train, y_train, X_test)            #E--96%    / M--0.087%
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

