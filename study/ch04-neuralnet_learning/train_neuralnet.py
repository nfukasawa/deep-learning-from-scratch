import sys, os
print(os.pardir)
import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# hyper params
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

nw = TwoLayerNet(input_size=784, hidden_size=50,output_size=10)

for i in range(iters_num):
    print(str(i) + "/" + str(iters_num))
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #grad = nw.numerical_gradient(x_batch, t_batch)
    grad = nw.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        nw.params[key] -= learning_rate * grad[key]
    
    loss = nw.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = nw.accuracy(x_train, t_train)
        test_acc = nw.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

#x = np.arange(len(train_loss_list))
#plt.plot(x, train_loss_list, label='train loss')
#plt.ylabel("loss")

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.ylim(0, 1.0)
plt.ylabel("accuracy")

plt.xlabel("iteration")
plt.legend(loc='lower right')
plt.show()