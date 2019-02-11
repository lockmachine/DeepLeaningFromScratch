#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.optimizer import *
from common.multi_layer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# データ量を削減
mask_train = np.random.choice(x_train.shape[0], 300)
x_train = x_train[mask_train]
t_train = t_train[mask_train]

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]   # 300
batch_size = 100

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(10000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 学習フェーズ
    grads = network.gradient(x_train, t_train)
    optimizer.update(network.params, grads)
    
    if i % iter_per_epoch == 0:
        loss_train = network.loss(x_train, t_train)
        loss_test = network.loss(x_test, t_test)
        train_loss_list.append(loss_train)
        test_loss_list.append(loss_test)
        
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_train)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        epoch_cnt += 1
        print("---{}/{}---".format(epoch_cnt, max_epochs))
        print("loss_train : " + str(loss_train))
        print("loss_test  : " + str(loss_test))
        print("acc_train  : " + str(train_acc))
        print("acc_test   : " + str(test_acc))
        if epoch_cnt >= max_epochs:
            break

plt.plot(train_loss_list, label="train_loss_list")
plt.plot(test_loss_list, label="test_loss_list")
plt.plot(train_acc_list, label="train_acc_list")
plt.plot(test_acc_list, label="test_acc_list")
plt.legend()
plt.show()
