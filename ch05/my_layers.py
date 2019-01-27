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
    
def main():
    pass
    


if __name__ == "__main__":
    main()
