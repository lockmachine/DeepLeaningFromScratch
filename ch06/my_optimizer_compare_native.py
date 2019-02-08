#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2/20.0 + y**2
    
def df(x, y):
    return x/10.0, 2.0*y
    
init_pos = (-7.0, 2.0)  # 開始位置の設定

# パラメーターの初期値の設定
params = {}
params["x"], params["y"] = init_pos[0], init_pos[1]

# 勾配の初期値の設定
grads = {}
grads["x"], grads["y"] = (0, 0)

# 順序付き辞書として、各Optimizerのインスタンスを生成
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

fig = plt.figure()
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    
    # パラメーターの初期化
    params["x"], params["y"] = init_pos[0], init_pos[1]
    
    # 30 回分の学習
    for i in range(30):
        # 学習中のパラメータを保存
        x_history.append(params["x"])
        y_history.append(params["y"])
        
        grads["x"], grads["y"] = df(params["x"], params["y"])
        optimizer.update(params, grads)
        
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-5, 5, 0.1)

    # グリッド作成
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    
    """*2次元プロット-------------------------------------------
    plt.subplot(2, 2, idx)  # プロットの図の構成を 2x2 とし、idx番目に表示する
    idx += 1
    
    plt.plot(x_history, y_history, "o-", color="red")
    plt.contour(X, Y, Z)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.plot(0, 0, "+")
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    -------------------------------------------*"""
    
    
    # グラフを複数同時に表示したいとき
    ax = fig.add_subplot(2, 2, idx, projection="3d")
    idx += 1
    
    # グラフをひとつづつ表示したいとき（ループ前の "fig = plt.figure()" は消す）
    """
    # figure メソッドで2次元の図を生成する
    fig = plt.figure()
    # 3次元版に変換
    ax = Axes3D(fig)
    """
    
    ax.plot_wireframe(X, Y, Z)
    
    a = x_history
    b = y_history
    c = f(np.array(x_history), np.array(y_history))
    
    ax.plot(a, b, c, "r-+")
    
plt.show()
