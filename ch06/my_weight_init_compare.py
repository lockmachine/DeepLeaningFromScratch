#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)
train_size = x_train.shape[0]
iter_num = 2000
batch_size = 128
loss = {}
input_size = 784
hidden_size_list = [100, 100, 100, 100]
output_size = 10


# 1:実験の設定==========
weight_type = {"std":0.01, "Xavier":"sigmoid", "He":"relu"}
optimizer = SGD(lr=0.01)

# ネットワークの定義
network = {}
for key, weight in weight_type.items():
    network[key] = MultiLayerNet(input_size, hidden_size_list, output_size, weight_init_std=weight_type[key])
    
    loss[key] = []

# 2:訓練の開始==========
for i in range(iter_num):
    # 60000サンプルから128サンプルをランダムに選ぶ
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 
    for key in weight_type.keys():
        # 勾配を求めて重みを更新する
        grads = network[key].gradient(x_batch, t_batch)
        optimizer.update(network[key].params, grads)
        
        loss[key].append(network[key].loss(x_batch, t_batch))
        
    if i % 100 == 0:
        print("--iter_num = {}--".format(i))
        for key in weight_type.keys():
            print("{} loss : {}".format(key, loss[key][-1]))


# 3.グラフの描画==========
for key in weight_type.keys(): 
    plt.plot(np.arange(iter_num), smooth_curve(loss[key]), label=key)
    
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.show()
