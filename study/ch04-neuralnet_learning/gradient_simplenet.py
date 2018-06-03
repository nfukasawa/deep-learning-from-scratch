# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import common.activation_funcs as act
import common.loss_funcs as loss
import common.gradient_funcs as grad

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    
    def predict(self, x:np.array) -> np.array:
        return np.dot(x, self.W)

    def loss(self, x:np.array, t:np.array) -> np.array:
        z = self.predict(x)
        y = act.softmax(z)
        return loss.cross_entropy_error(y, t)
        
if __name__ == '__main__':
    net = simpleNet()
    print(net.W)

    x = np.array([0.6,0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0,0,1])
    print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)
    
    dW = grad.numerical_gradient(f, net.W)
    print(dW)


