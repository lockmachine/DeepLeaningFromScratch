#!/usr/bin/env python3
# coding: utf-8
import numpy as np

class myRelu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)    # x が 0 以下の場合にTrue
        out = x.copy()          # 引数を直接操作しない
        out[self.mask] = 0      # x が 0 以下となるインデックスだけ値を 0 にする
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0     # forward で 0 だった場合は backward も 0 とする
        return dout

class mySigmoid:
    def __init__(self):
        self.out = None
        pass
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
        
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
        
class myAffine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None    # テンソル対応用
        
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None
        
    def forward(self, x):
        # テンソル対応
        self.original_x_shape = self.shape  # 元々の入力データの形状を記憶
        x = x.reshape(x.shape[0], -1)       # 1 次元目の要素数で固定した行列形状にする
        
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape) # 入力データの形状を元に戻す（テンソル対応）
        return dx
    
def main():
    pass
    


if __name__ == "__main__":
    main()
