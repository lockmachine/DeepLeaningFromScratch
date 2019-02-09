#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def relu(x):
    return np.maximum(0, x)
    
x = np.random.randn(1000, 100)  # 1000個のデータ
node_num = 100                  # 隠れ層のノード(ニューロン)の数
hidden_layer_size = 5           # 隠れ層が5層
activations = {}                # アクティベーションの結果を格納

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    #w = np.random.randn(node_num, node_num) * 1 # 標準偏差が1
    #w = np.random.randn(node_num, node_num) * 0.01 # 標準偏差が0.01
    #w = np.random.randn(node_num, node_num) / np.sqrt(node_num) # 標準偏差が1/sqrt(n)(Xavierの初期化)
    w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num) # 標準偏差がsqrt(2/n)(Heの初期化)
    
    z = np.dot(x, w)
    #a = sigmoid(z)
    #a = np.tanh(z)
    a = relu(z) # Heの初期化が適している
    activations[i] = a
    

# アクティベーションのヒストグラムを描画する
for key, acti in activations.items():
    plt.subplot(1, len(activations), key+1)
    plt.title("{}-layer".format(key+1))
    plt.hist(acti.flatten(), bins=30, range=(0, 1))
plt.show()
    
