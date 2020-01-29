import numpy as np
class SimpleLinearRegressionSelf:
    def __init__(self):
        """初始化Simple linear regression模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "一元线性回归模型仅处理向量，而不能处理矩阵"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        denominator = 0.0
        numerator = 0.0
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)  # 按照a的公式得到分子
            denominator += (x_i - x_mean) ** 2  # 按照a的公式得到分母
        self.a_ = numerator / denominator  # 得到a
        self.b_ = y_mean - self.a_ * x_mean  # 得到b
        return self

    def predict(self, x_test_group):
        return np.array([self._predict(x_test) for x_test in x_test_group])

    # 对于输入向量集合中的每一个向量都进行一次预测，预测的具体实现被封装在_predict函数中。
    def _predict(self, x_test):
        return self.a_ * x_test + self.b_  # 求取每一个输入的x_test以得到预

    def mean_squared_error(self, y_true, y_predict):
        return np.sum((y_true - y_predict) ** 2) / len(y_true)

    def r_square(self, y_true, y_predict):
        return 1 - (self.mean_squared_error(y_true, y_predict) / np.var(y_true))


if __name__ == '__main__':
    x = np.array([1, 2, 4, 6, 8])
    y = np.array([2, 5, 7, 8, 9])
    lr = SimpleLinearRegressionSelf()  # 封装模型的类名
    lr.fit(x, y)  # 训练模型得到a和b
    print(lr.predict([7,8]))  # 输出结果与之前一致，都是8.74
    print(lr.r_square([8, 9], lr.predict([6, 8])))  # 得到的得分是0.09443783462224864


