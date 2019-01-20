#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
#from two_layer_net import TwoLayerNet
from two_layer_net_at_ch05 import TwoLayerNet
import time
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []
train_acc_list = [] # 学習データにおける認識精度リスト
test_acc_list = [] # テストデータにおける認識精度リスト

# ミニバッチ学習の実装

# ハイパーパラメータ
iters_num = 1800#10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 1 エポック当たりの繰り返し回数 (600)
iter_per_epoch = max(train_size / batch_size, 1)    # (60000 / 100)

network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=10)

start = time.time()

for i in range(iters_num):
    # ミニバッチの取得
    # 1~train_size のなかで、batch_size 分だけのランダムなインデックスをマスクとする
    batch_mask = np.random.choice(train_size, batch_size)
    
    batch_x_train = x_train[batch_mask, :]  # 100x784
    batch_t_train = t_train[batch_mask, :]  # 100x10
    
    # 勾配の計算
    #grads = network.numerical_gradient(batch_x_train, batch_t_train)    # 数値微分版だと1回のループに90秒程度かかった
    grads = network.gradient(batch_x_train, batch_t_train)  # 高速版 1会のループに0.008秒程度
    
    # パラメータの更新
    network.params["W1"] -= (learning_rate * grads["W1"])
    network.params["W2"] -= (learning_rate * grads["W2"])
    network.params["b1"] -= (learning_rate * grads["b1"])
    network.params["b2"] -= (learning_rate * grads["b2"])
    
    # 学習経過の記録
    loss = network.loss(batch_x_train, batch_t_train)
    train_loss_list.append(loss)

    # 1 エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    if i % 100 == 0:
        print(".", end="", flush=True)  # 進捗確認

print("\nFinished!")

end = time.time()

plt.plot(train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

plt.plot(train_acc_list)
plt.plot(test_acc_list)
plt.xlabel("iteration")
plt.ylabel("acc")
plt.show()


print("ProcTime[{}]".format(end-start))
