#!/usr/bin/env python3
# coding: utf-8
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from numerical_gradient import numerical_gradient

def gradient_descent(f, init_x, lr = 0.01, step_num=100):
	x = init_x
	grad_data = np.zeros(0)

	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x = x - lr * grad
		# 途中の勾配を保存
		if i == 0:
			x_history = x.reshape(1, -1)
			grad_data = grad.reshape(1, -1)
		elif i % 10 == 0:
			x_history = np.append(x_history, x.reshape(1, -1), axis=0)
			grad_data = np.append(grad_data, grad.reshape(1, -1), axis=0)
	return x, grad_data, x_history

def function_2(x):
	if x.ndim == 1:
		return np.sum(x**2)
	else:
		return np.sum(x**2, axis=1)

if __name__ == "__main__":
	# f(x0, x1) = x0^2 + x1^2 の最小値を勾配法で求めよ
	init_x = np.array([-3.0, 4.0])

#	x = np.arange(-4.0, 4.0, 0.1)	# (80, )
#	y = np.arange(-4.0, 4.0, 0.1)	# (80, )
#	x_con = np.vstack((x, y))		# (2, 80)
#	Z = function_2(x_con)
#	X, Y = np.meshgrid(x, y)		# メッシュグリッドの生成
#	plt.contour(X, Y, Z)
#	plt.gca().set_aspect('equal')
#	plt.show()

	# 学習率が大きすぎる例
	x_large, grad_x_large, x_large_history = gradient_descent(function_2, init_x, lr = 10.0, step_num=100)
	print("学習率が大きすぎる場合:" + str(x_large))
	print(function_2(x_large))
#	plt.plot(grad_x_large[:, 0], grad_x_large[:, 1], "o")
	plt.plot(x_large_history[:, 0], x_large_history[:, 1], "o")

	# 学習率が小さすぎる例
	x_small, grad_x_small, x_small_history = gradient_descent(function_2, init_x, lr = 1e-10, step_num=100)
	print("学習率が小さすぎる場合:" + str(x_small))
	print(function_2(x_small))
#	plt.plot(grad_x_small[:, 0], grad_x_small[:, 1], "x")
	plt.plot(x_small_history[:, 0], x_small_history[:, 1], "x")

	# デフォルトの学習率
	x_default, grad_x_default, x_default_history = gradient_descent(function_2, init_x, lr = 0.01, step_num=100)
	print("デフォルトの学習率:" + str(x_default))
	print(function_2(x_default))
#	plt.plot(grad_x_default[:, 0], grad_x_default[:, 1], ".")
	plt.plot(x_default_history[:, 0], x_default_history[:, 1], ".")
	plt.xlim(-5, 5)
	plt.ylim(-5, 5)
	plt.show()
