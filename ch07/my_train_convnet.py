#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from SimpleConvNet import SimpleConvNet
from common.trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import Adam
import time

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# パラメータの設定
batch_size = 100
train_num = 100000
epoch_max = 20
sample_num = 5000
mask = np.random.choice(x_train.shape[0], sample_num)
x_train = x_train[mask]
t_train = t_train[mask]
mask_test = np.random.choice(x_test.shape[0], sample_num)
x_test = x_test[mask_test]
t_test = t_test[mask_test]
train_per_epoch = sample_num / batch_size   # 5000 / 100 = 50train/1epoch

input_dim = x_train[0].shape  # (1, 28, 28)
conv_param = {}
conv_param["filter_num"] = 30
conv_param["filter_size"] = 5
conv_param["pad"] = 0
conv_param["stride"] = 1
hidden_size = 100
output_size = 10
weight_init_std = 0.01
learning_rate = 0.001
train_loss_list = []
train_acc_list = []
test_acc_list = []
optimizer = Adam()

# ネットワークの生成
network = SimpleConvNet(input_dim=input_dim,
                        conv_param=conv_param,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        weight_init_std=weight_init_std)

"""
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=epoch_max, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
"""
epoch_num = 0
start = time.time()
for i in range(train_num):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    batch_mask_test = np.random.choice(x_test.shape[0], batch_size)
    x_test_batch = x_test[batch_mask_test]
    t_test_batch = t_test[batch_mask_test]
    
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    optimizer.update(network.params, grad)
    #for key in grad.keys():
    #    network.params[key] -= learning_rate * grad[key]
    
    # 損失関数計算
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    train_acc = network.accuracy(x_batch, t_batch)
    test_acc = network.accuracy(x_test_batch, t_test_batch)
    
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print(".", end="", flush=True)
    
    
    if i % train_per_epoch == 0:
        # 1 epoch 終了
        epoch_num += 1
        end = time.time()
        print("\n---{}/{} epochs (ProcTime:{})---".format(epoch_num, epoch_max, end-start))
        print("loss({}), train_acc({}), test_acc({})".format(loss, train_acc, test_acc))
        # 所定のepoch回数トレーニングしたら終了
        if epoch_num >= epoch_max:
            break
            

# 1110秒かかった
plt.plot(train_loss_list, label="train_loss")
plt.plot(train_acc_list, label="train_acc")
plt.plot(test_acc_list, label="test_acc")
plt.xlabel("train_num")
plt.ylabel("loss")
plt.legend()
plt.show()
