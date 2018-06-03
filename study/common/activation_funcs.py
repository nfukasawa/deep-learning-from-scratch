# coding: utf-8
import numpy as np

def step_function(x: np.array) -> np.array:
    return np.array(x > 0, dtype=np.int)

def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x: np.array) -> np.array:
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x: np.array) -> np.array:
    return np.maximum(x, 0)

def identity_function(x: np.array) -> np.array:
    return x

def softmax(x: np.array) -> np.array:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))
    
"""
def softmax(a: np.array) -> np.array:
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)
"""



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    x = np.arange(-5.0, 5.0, 0.1)
    
    y1 = step_function(x)
    plt.plot(x, y1, label='Step function')

    y2 = sigmoid(x)
    plt.plot(x, y2, label='Sigmoid')

    y3 = relu(x)
    plt.plot(x, y3, label='ReLU')

    y4 = identity_function(x)
    plt.plot(x, y4, label='Indetity function')

    y5 = softmax(x)
    plt.plot(x, y5, label='Softmax')

    plt.ylim(-0.1, 5.1)
    plt.legend()
    plt.show()