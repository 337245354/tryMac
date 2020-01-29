import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import knn
import numpy as np
import try3_4

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
    X_train = X_train.reshape(X_train.shape[0], -1)
    mean_image = getXmean(X_train)
    X_train = centralized(X_train, mean_image)
    y_train = train_loader.dataset.targets
    y_train = np.array(y_train)
    X_test = test_loader.dataset.data
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = centralized(X_test, mean_image)
    y_test = test_loader.dataset.targets
    y_test = np.array(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
#     M(k=6) ---36/100     E(k=6) ---35/100

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20] #k的值一般选择1~20以内
    num_training=X_train.shape[0]
    X_train_folds = []
    y_train_folds = []
    indices = np.array_split(np.arange(num_training), indices_or_sections=num_folds) #把下标分成5个部分
    for i in indices:
        X_train_folds.append(X_train[i])
    y_train_folds.append(y_train[i])
    k_to_accuracies = {}
    for k in k_choices:
        #进行交叉验证
        acc = []
        for i in range(num_folds):
            x = X_train_folds[0:i] + X_train_folds[i+1:] #训练集不包括验证集
            x = np.concatenate(x, axis=0) #使用concatenate将4个训练集拼在一起
            y = y_train_folds[0:i] + y_train_folds[i+1:]
            y = np.concatenate(y) #对label进行同样的操作
            test_x = X_train_folds[i] #单独拿出验证集
            test_y = y_train_folds[i]
            classifier = try3_4.Knn() #定义model
            classifier.fit(x, y) #读入训练集
            #dist = classifier.compute_distances_no_loops(test_x)
            #计算距离矩阵
            y_pred = classifier.predict(k,'M',test_x) #预测结果
            accuracy = np.mean(y_pred == test_y) #计算准确率
            acc.append(accuracy)
    k_to_accuracies[k] = acc #计算交叉验证的平均准确率
    #输出准确度
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

