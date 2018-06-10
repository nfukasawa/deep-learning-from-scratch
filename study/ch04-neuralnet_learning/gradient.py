# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import types
import util.gradient_funcs as grad

def gradient_descent(f: types.FunctionType, init_x: np.array, lr:np.float=0.01, step_num:np.int=100) -> np.array:
    x = init_x
    for i in range(step_num):
        g = grad.numerical_gradient(f, x)
        x -= lr * g
    return x

if __name__ == '__main__':
    def function_2(x):
        return x[0]**2 + x[1]**2
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr=0.1))
