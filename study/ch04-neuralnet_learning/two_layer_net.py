# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import common.activation_funcs as act
import common.loss_funcs as loss
import common.gradient_funcs as grad

class TwoLayerNet:
    def __init__(self, input_size:int, hidden_size:int, output_size:int, weight_init_std:float=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x:np.array) -> np.array:
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = act.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = act.softmax(a2)
        return y

    def loss(self, x:np.array, t:np.array) -> np.array:
        y = self.predict(x)
        return loss.cross_entropy_error(y,t)

    def accuracy(self, x:np.array, t:np.array) -> np.array:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        acurracy = np.sum(y == t) / float(x.shape[0])
        return acurracy

    def numerical_gradient(self, x:np.array, t:np.array) -> np.array:
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = grad.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = grad.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = grad.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = grad.numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    ################################
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = act.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = act.softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = act.sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
