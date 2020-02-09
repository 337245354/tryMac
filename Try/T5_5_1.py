import numpy as np

def mean_squared_error(p,y):
    return np.sum((p-y)**2)/y.shape[0]

def cross_entropy_error(p,y):
    delta = 1e-7
    batch_size = p.shape[0]
    return -np.sum(y * np.log(p + delta)) / batch_size

y = np.array([0,1,0,0]) #y是真实标签cross_entropy_error
p = np.array([0.3,0.2,0.1,0.4]) #通过Softmax得到的概率值


print(mean_squared_error(p,y))
print(cross_entropy_error(p,y))



