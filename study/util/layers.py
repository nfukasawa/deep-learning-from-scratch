# coding: utf-8
import numpy as np
import util.activation_funcs as act
import util.loss_funcs as loss

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x: np.array) -> np.array:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout: np.array) -> np.array:
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x: np.array) -> np.array:
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout: np.array) -> np.array:
        dx = dout * (1.0 - self.out)
        return dx

"""
class Affine:
    def __init__(self, W:np.array, b:np.array):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.array) -> np.array:
        self.x = x
        out = np.dot(x, self.W) + self.b
    
    def backward(self, dout: np.array) -> np.array:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
"""

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.array, t: np.array) -> np.array:
        self.t = t
        self.y = act.softmax(x)
        self.loss = loss.cross_entropy_error(self.y, self.t)
    
    def backward(self, dout=1) -> np.array:
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx