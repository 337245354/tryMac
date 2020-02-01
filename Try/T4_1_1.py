import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    x = np.array([1,2,4,6,8]) #铺设管子的长度
    y = np.array([2,5,7,8,9]) #费用
    x_mean = np.mean(x) #求出x向量的均值
    y_mean = np.mean(y) #求出y向量的均值
    print(x_mean)
    print(y_mean)

    denominator = 0.0  # 分母
    numerator = 0.0  # 分子
    for x_i, y_i in zip(x, y):  # 将x，y向量合并起来形成元组(1,2),(2,5)
        numerator += (x_i - x_mean) * (y_i - y_mean)  # 按照a的公式得到分子
        denominator += (x_i - x_mean) ** 2  # 按照a的公式得到分母
    a = numerator / denominator  # 得到a
    b = y_mean - a * x_mean  # 得到b

    y_predict = a * x + b  # 求得预测值y_predict
    plt.scatter(x, y, color='b')  # 画出所有训练集的数据
    plt.plot(x, y_predict, color='r')  # 画出拟合直线，颜色为红色
    plt.xlabel('管子的长度', fontproperties='simHei', fontsize=15)  # 设置x轴的标题
    plt.ylabel('收费', fontproperties='simHei', fontsize=15)  # 设置y轴的标题
    plt.show()

    x_test = 7
    y_predict_value = a * x_test + b
    print(y_predict_value)