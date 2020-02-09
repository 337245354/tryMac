import numpy as np

#x为输入的向量
def _softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    return exp_x / np.sum(exp_x)


def _relu(x):
    return np.maximum(0,x)

scores = np.array([-1, 2, -3]) # example with 3 classes and each having large scores
p = _softmax(scores)
print(p)