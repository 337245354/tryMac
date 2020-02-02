import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter

class LogisticRegression:
    def __init__(self):
        """初始化Logistic regression模型"""
        self.coef_ = None  # 维度
        self.intercept_ = None  # 截距
        self._theta = None
        # sigmoid函数，私有化函数

    def _sigmoid(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        return y

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], '训练数据集的长度需要与标签长度保持一致'

        # 计算损失函数
        def J(theta, X_b, y):
            p_predcit = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(p_predcit) + (1 - y) * np.log(1 - p_predcit)) / len(y)
            except:
                return float('inf')

        # 求sigmoid梯度的导数
        def dJ(theta, X_b, y):
            x = self._sigmoid(X_b.dot(theta))
            return X_b.T.dot(x - y) / len(X_b)

        # 模拟梯度下降
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                i_iter += 1
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
            return theta
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])  # 列向量
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]  # 维度
        return self


    def predict_proba(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        proba = self.predict_proba(X_predict)
        return np.array(proba > 0.5, dtype='int')

    def score(self, X_test, y_test):
        y_predict = myLogstic.predict(X_test)
        re = (y_test == y_predict)
        re1 = Counter(re)
        a = re1[True] / (re1[True] + re1[False])
        return a



if __name__ == '__main__':


    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    X = X[y != 2]
    y = y[y != 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    myLogstic = LogisticRegression()
    myLogstic.fit(X_train, y_train)
    y_predict = myLogstic.predict(X_test)
    print("参数:", myLogstic._theta)
    print("测试数据准确度:", myLogstic.score(X_test, y_test))
    print("训练数据准确度:", myLogstic.score(X_train, y_train))
