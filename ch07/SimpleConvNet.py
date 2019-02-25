#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import pickle
from my_Convolution import myConvolution, myPooling
from common.layers import *
from collections import OrderedDict
#from common.functions import

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                    conv_param={"filter_num":30, "filter_size":5, "pad":0, "stride":1},
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
        
        # パラメータの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)
        
        # 各層の生成
        self.layers = OrderedDict()
        self.layers["Conv1"] = myConvolution(self.params["W1"],
        # 第1層
                                             self.params["b1"],
                                             stride=filter_stride,
                                             pad=filter_pad)
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = myPooling(pool_h=2, pool_w=2, stride=2)
        
        # 第2層
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        
        # 第3層
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        
        # 出力層
        self.lastlayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
        
    def gradient(self, x, t):
        # 順伝播
        self.loss(x, t)
        
        # 逆伝播
        # 最後の層の逆伝播から実行
        dout = 1
        dout = self.lastlayer.backward(dout)
        
        # 残りの層の逆伝播を後ろの層から実行
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # 算出した勾配を記憶
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db
        
        return grads
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        acc_list = (y == t)
        acc = np.sum(acc_list) / acc_list.shape[0]
        
        #print(y)
        #print(t)
        #exit()
        return acc
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, value in self.params.items():
            params[key] = value
        with open(file_name, "wb") as f:
            pickle.dump(params, f)
            
    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, value in params.items():
            self.params[key] = value
        
        # 各層のパラメーターに戻す
        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i+1)]
            self.layers[key].b = self.params["b" + str(i+1)]
            
