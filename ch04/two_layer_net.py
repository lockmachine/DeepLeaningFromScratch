#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from pprint import pprint

class TwoLayerNet:

	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		# 重みの初期化
		self.params = {}
		self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)

		# バイアスの初期化
		self.params["b1"] = np.zeros(hidden_size)
		self.params["b2"] = np.zeros(output_size)

	def predict(self, x):
		W1, W2 = self.params["W1"], self.params["W2"]
		b1, b2 = self.params["b1"], self.params["b2"]

		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y

	# y:入力データ, t:教師データ としたときの損失関数
	def loss(self, x, t):
		y = self.predict(x)
		e = cross_entropy_error(y, t)
#		pprint(y)
#		pprint(e)
		return e

	# x:入力データ, t:教師データ としたときの予測精度計算
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)

		accuracy = np.sum(y == t) / y.shape[0]
		return accuracy

	# x:入力データ, t:教師データとしたときの損失関数の勾配計算
	# common.gradient の numerical_gradientとは異なることに注意
	def numerical_gradient(self, x, t):
		# 損失関数を定義
#		def loss_W(W):
#			return self.loss(x, t)
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		# 各重みとバイアスの勾配を求める
#		pprint("W1")
		grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
#		pprint("W2")
		grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
		grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
		grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
#		pprint(grads)
		return grads

def main():
	input_size = 3
	hidden_size = 4
	output_size = 2
	network = TwoLayerNet(input_size, hidden_size, output_size)

	x = np.array([1,2,3])
	t = np.array([0,0,1])
	print("numerical_gradient:{}".format(network.numerical_gradient(x, t)))
	print("loss:{}".format(network.loss(x, t)))

if __name__ == "__main__":
	main()
