import numpy as np
from numpy import linalg

class MLinearRegression:
    def __init__(self):
        self.coef_ = None #代表的是权重
        self.interception_ = None #代表的是截距
        self._theta = None #代表的是权重+截距

''' 规范下代码，X_train代表的是矩阵X大写，y_train代表的是向量y小写'''
def fit(self,X_train, y_train):
    assert X_train.shape[0] == y_train.shape[0], \
    "训练集的矩阵行数与标签的行数保持一致"
    ones = np.ones((X_train.shape[0], 1))
    X_b = np.hstack((ones, X_train)) #将X矩阵转为X_b矩阵，其中第一列为1，其余不变
    self._theta = linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    self.interception_ = self._theta[0]
    self.coef_ = self._theta[1:]
    return self

def predict(self,X_predict):
    ones = np.ones((X_predict.shape[0], 1))
    X_b = np.hstack((ones, X_predict)) #将X矩阵转为X_b矩阵，其中第一列为1，其余不变
    return X_b.dot(self._theta) #得到的即为预测值

def mean_squared_error(self, y_true, y_predict):
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

def score(self,X_test,y_test): #使用r square
    y_predict = self.predict(X_test)
    return 1 - (self.mean_squared_error(y_test,y_predict) / np.var(y_test))