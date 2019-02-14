#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.util import im2col, col2im


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        # フィルターのサイズ取得
        FN, C, FH, FW = self.W.shape
        
        # 入力データのサイズ取得
        N, C, H, W = x.shape
        
        # 出力サイズ(FN, OH, OW)
        OH = int((H + 2*self.pad - FH) / self.stride) + 1
        OW = int((W + 2*self.pad - FW) / self.stride) + 1
        
        col = im2col(x, FH, FW, self.stride, self.pad)  # (OH*OW)x(C*FW*FH)
        col_W = self.W.reshape(FN, -1).T    # フィルターの展開 FNx(FW*FH*C)
        out = np.dot(col, col_W) + self.b
        
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)
        
        return out
