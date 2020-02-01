import numpy as np
import matplotlib.pyplot as plt
import operator
##给出训练数据以及对应的类别
def createDataSet():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5],[1.1,1.0],[0.5,1.5]])
    labels = np.array(['A','A','B','B','A','B'])
    return group,labels

if __name__=='__main__':
    group,labels = createDataSet()
    plt.scatter(group[labels=='A',0],group[labels=='A',1],color = 'r', marker='*')
    #对于类别为A的数据集我们使用红色六角形表示
    plt.scatter(group[labels=='B',0],group[labels=='B',1],color = 'g', marker='+')
    #对于类别为B的数据集我们使用绿色十字形表示
    # plt.show()

def kNN_classify(k,dis,X_train,x_train,Y_test):
    assert dis == 'E' or dis == 'M', 'dis must E or M，E代表欧式距离，M代表曼哈顿距离'
    num_test = Y_test.shape[0] #测试样本的数量
    labellist = []
    ''' 使用欧拉公式作为距离度量'''
    if (dis == 'E'):
        for i in range(num_test):
        #实现欧式距离公式
            distances = np.sqrt(np.sum(((X_train - np.tile(Y_test[i], (X_train.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances) #距离由小到大进行排序，并返回index值
            topK = nearest_k[:k] #选取前k个距离⌥
            classCount = {}
            for i in topK: #统计每个类别的个数
                classCount[x_train[i]] = classCount.get(x_train[i],0) + 1
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
    ''' 使用曼哈顿公式作为距离度量'''
    if (dis == 'M'):
        for i in range(num_test):
            distances = np.sum(np.abs(X_train - np.tile(Y_test[i], (X_train.shape[0], 1))),axis=1)
            nearest_k = np.argsort(distances) #距离由小到大进行排序，并返回index值
            topK = nearest_k[:k] #选取前k个距离
            classCount = {}
            for i in topK: #统计每个类别的个数
                classCount[x_train[i]] = classCount.get(x_train[i],0) + 1
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
#使用曼哈顿公式作为距离度量
#读者自行补充完成
if __name__ == '__main__':
    group, labels = createDataSet()
    y_test_pred = kNN_classify(1, 'M', group, labels, np.array([[1.0,2.1],[0.4,2.0],[0.4,1.4]]))
    print(y_test_pred) #打印输出['A' 'A' 'B']