# coding: utf-8
import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    x = np.arange(-5.0, 5.0, 0.1)
    
    y1 = step_function(x)
    plt.plot(x, y1, label='Step function')

    y2 = sigmoid(x)
    plt.plot(x, y2, label='Sigmoid', linestyle = "--")

    y3 = relu(x)
    plt.plot(x, y3, label='ReLU', linestyle = ":")

    plt.ylim(-0.1, 5.1)
    plt.legend()
    plt.show()