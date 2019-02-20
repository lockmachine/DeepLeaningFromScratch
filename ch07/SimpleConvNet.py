#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from my_Convolution import myConvolution
from common.layers import *
#from common.functions import

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                    conv_param={"fileter_num":30, "filter_size":5, "pad":0, "stride":1},
                    hidden_size=100, output_size=10, weight_init_std=0.01):
        # フィルター情報のコピー
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        
        # 入力画像サイズは 1channel x 28height x 28width
        input_size = input_dim[1]
        
        # 畳み込み層の出力サイズ
        conv_output_size = int((input_size + 2*filter_pad - filter_size) / filter_stride + 1)
        
        # プーリング層の出力サイズ
        pool_output_size = int((filter_num * (conv_output_size/2) * (conv_output_size/2)))
