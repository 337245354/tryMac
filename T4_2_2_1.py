import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    plot_x = np.linspace(-1, 6, 141) #从-1到6选取141个点
    plot_y = (plot_x - 2.5) ** 2 - 1#二次方程的损失函数
    plt.scatter(plot_x[5], plot_y[5], color='r') #设置起始点，颜色为红色
    plt.plot(plot_x, plot_y)
    #设置坐标轴名称
    plt.xlabel('theta', fontproperties='simHei', fontsize=15)
    plt.ylabel('损失函数', fontproperties='simHei', fontsize=15)
    plt.show()