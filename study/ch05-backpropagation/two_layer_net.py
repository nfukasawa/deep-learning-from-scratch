# coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
import util.layers as layer
import util.gradient_funcs as grad
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size:int, hidden_size:int, output_size:int, weight_init_std:float=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # gen layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = layer.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = layer.Relu()
        self.layers['Affine2'] = layer.Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = layer.SoftmaxWithLoss()

    def predict(self, x:np.array) -> np.array:
        for l in self.layers.values():
            x = l.forward(x)
        return x

    def loss(self, x:np.array, t:np.array) -> np.array:
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x:np.array, t:np.array) -> np.array:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x:np.array, t:np.array) -> np.array:
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = grad.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = grad.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = grad.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = grad.numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x:np.array, t:np.array) -> np.array:
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for l in layers:
            dout = l.backward(dout)
        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        

