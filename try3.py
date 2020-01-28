import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import knn
import numpy as np

batch_size = 100
#Cifar10 dataset
train_dataset = dsets.CIFAR10(root = './ml/pycifar', #选择数据的根目录
                               train = True, #选择训练集
                                download = True) #从网络上下载图片
test_dataset = dsets.CIFAR10(root = './ml/pycifar', #选择数据的根目录
                            train = False, #选择测试集
                            download = True) #从网络上下载图片
#加载数据
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True) #将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
digit = train_loader.dataset.data[0]

# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()
# print(classes[train_loader.dataset.targets[0]]) #打印出是

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
    X_train = train_loader.dataset.data
    mean_image = getXmean(X_train)
    X_train = centralized(X_train,mean_image)
    y_train = train_loader.dataset.targets
    X_test = test_loader.dataset.data[:100]
    X_test = centralized(X_test,mean_image)
    y_test = test_loader.dataset.targets[:100]
    num_test = len(y_test)
    y_test_pred = knn.kNN_classify(6, 'M', X_train, y_train, X_test)#这里并没有使用封装好的类
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#     M ---16/100
