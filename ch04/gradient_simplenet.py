#!/usr/bin/env python3
# coding: utf-8
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plot
#from softmax import softmax
from common.functions import softmax
from mean_squared_error import cross_entropy_error
from numerical_gradient import numerical_gradient
#from common.functions import softmax, cross_entropy_error
#from common.gradient import numerical_gradient

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3)	# ガウス分布で初期化

	def predict(self, x):
		return np.dot(x, self.W)

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)
		return loss

"""
def f(W):
	return net.loss(x, t)
"""
f = lambda W: net.loss(x, t)

if __name__ == "__main__":
	net = simpleNet()
	print("net.W = " + str(net.W))

	x = np.array([0.6, 0.9])
	p = net.predict(x)
	print("net.predict(x) = " + str(p))

	t = np.array([0, 0, 1])	# 正解ラベル
	l = net.loss(x, t)
	print("net.loss(x, t) = " + str(l))

	dW = numerical_gradient(f, net.W)
	print(dW)
