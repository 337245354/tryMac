import numpy as np

class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = np.maximum(0, x)
        out = self.x
        return out

    def backward(self, dout):
        dx = dout
        dx[self.x <= 0] = 0
        return dx


class _sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.p = None  # Softmax的输出
        self.y = None  # 监督数据代表真值，one-hot vector

    def forward(self, x, y):
        self.y = y
        self.p = softmax(x)
        self.loss = cross_entropy_error(self.p, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.p - self.y) / batch_size
        return dx


def softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)


def cross_entropy_error(p,y):
    delta = 1e-7
    batch_size = p.shape[0]
    return -np.sum(y * np.log(p + delta)) / batch_size