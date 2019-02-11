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

(x_train, t_train), (x_test, t_test) = load_mnist()

iter_num = 2000
mask_train = np.random.choice(x_train.shape[0], 1000)
mask_test = np.random.choice(x_test.shape[0], 1000)
x_train = x_train[mask_train]
t_train = t_train[mask_train]
x_test = x_test[mask_test]
t_test = t_test[mask_test]
train_size = x_train.shape[0]
batch_size = 128


optimizer = SGD(lr=0.01)

network_st = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100], output_size=10)
network_bn = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100], output_size=10, use_batchnorm=True)

loss_list = {"network_st":[], "network_bn":[], "network_st_test":[], "network_bn_test":[]}

for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    x_test_batch = x_test[batch_mask]
    t_test_batch = t_test[batch_mask]
    
    # 学習フェーズ
    # BatchNormalizationなし
    for _network in (network_st, network_bn):
        grads = _network.gradient(x_batch, t_batch)
        optimizer.update(_network.params, grads)
        loss = _network.loss(x_batch, t_batch)
        if _network == network_st:
            loss_list["network_st"].append(loss)
            loss_list["network_st_test"].append(_network.loss(x_test_batch, t_test_batch))
        else:
            loss_list["network_bn"].append(loss)
            loss_list["network_bn_test"].append(_network.loss(x_test_batch, t_test_batch))
    
    if i % 100 == 0:
        print("--- iter_num = " + str(i) + "---")
        print("network_st loss : " + str(loss_list["network_st"][-1]))
        print("network_bn loss : " + str(loss_list["network_bn"][-1]))
    
plt.plot(np.arange(iter_num), loss_list["network_st"])
plt.plot(np.arange(iter_num), loss_list["network_bn"])
plt.plot(np.arange(iter_num), loss_list["network_st_test"])
plt.plot(np.arange(iter_num), loss_list["network_bn_test"])
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.show()
