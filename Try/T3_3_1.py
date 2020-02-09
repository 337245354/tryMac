import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import operator


class selfA:
    def __init__(self, X_train, y_train):
        self.Xtr = X_train
        self.ytr = y_train
#

def fit(self,X_train,y_train): #我们统一下命名规范，X_train代表的是训练数据集，而y_train代表的是对应训练集数据的标签
    self.Xtr = X_train
    self.ytr = y_train


def predict(self,k, dis, X_test):
    assert dis == 'E' or dis == 'M', 'dis must E or M'
    num_test = X_test.shape[0] #测试样本的数量
    labellist = []
    #使用欧拉公式作为距离度量
    if (dis == 'E'):
        for i in range(num_test):
            distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[self.ytr[i]] = classCount.get(self.ytr[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)


# #加载数据
batch_size = 100
# train_dataset = dsets.MNIST(root = '../ml/pymnist', #选择数据的根目录
#                             train = True, #选择训练集
#                             transform = None, #不考虑使用任何数据预处理
#                             download = True) #从网络上下载图片
# test_dataset = dsets.MNIST(root = '../ml/pymnist', #选择数据的根目录
#                            train = False, #选择测试集
#                            transform = None, #不考虑使用任何数据预处理
#                            download = True) #从网络上下载图片
train_loader = torch.utils.data.DataLoader(dataset = dsets.MNIST(train = True),
                                            batch_size = batch_size,
                                            shuffle = True) #将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = dsets.MNIST(train = False),
                                        batch_size = batch_size,
                                        shuffle = True)


if __name__ == '__main__':
    fit(selfA, train_loader, test_loader)
    predict(selfA, 1, 'E', test_loader)

    #
    # X_train = train_loader.dataset.train_data.numpy() #需要转为numpy矩阵
    # X_train = X_train.reshape(X_train.shape[0],28*28)#需要reshape之后才能放入knn分类器
    # y_train = train_loader.dataset.train_labels.numpy()
    # X_test = test_loader.dataset.test_data[:1000].numpy()
    # X_test = X_test.reshape(X_test.shape[0],28*28)
    # y_test = test_loader.dataset.test_labels[:1000].numpy()
    # num_test = y_test.shape[0]
    # y_test_pred = knn.kNN_classify(5, 'M', X_train, y_train, X_test)
    # num_correct = np.sum(y_test_pred == y_test)
    # accuracy = float(num_correct) / num_test
    # print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))