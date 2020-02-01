import numpy as np
import matplotlib.pyplot as plt

def J(theta): #损失函数
    return (theta-2.5)**2 -1
def dJ(theta): #损失函数的导数
    return 2 * (theta - 2.5)

if __name__ == '__main__':
    plot_x = np.linspace(-1, 6, 141) #从-1到6选取141个点
    theta = 0.0 #初始点
    theta_history = [theta]
    eta = 0.1 #步长
    epsilon = 1e-8 #精度问题或者eta的设置无法使得导数为0
    while True:
        gradient = dJ(theta) #求导数
        last_theta = theta #先记录下上一个theta的值
        theta = theta - eta * gradient #得到一个新的theta
        theta_history.append(theta)
        if(abs(J(theta) - J(last_theta)) < epsilon):
            break #当两个theta值非常接近的时候，终止循环
    plt.plot(plot_x,J(plot_x),color='r')
    plt.plot(np.array(theta_history),J(np.array(theta_history)),color='b',marker='x')
    plt.show() #一开始的时候导数比较大，因为斜率比较陡，后面慢慢平缓了
    print(len(theta_history)) #一共走了46步