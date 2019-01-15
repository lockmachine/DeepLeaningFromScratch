#!/usr/bin/env python3
# coding: utf-8
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plot

def function_2(x):
	if x.ndim == 1:
		return np.sum(x**2)
	else:
		return np.sum(x**2, axis=1)

def numerical_gradient(f, x):
	h = 1e-4	# 0.0001
	grad = np.zeros_like(x)	# xと同じ形状の配列を生成

	it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
#	for idx in range(x.size):
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		# f(x+h) の計算
		x[idx] = float(tmp_val) + h
		fxh1 = f(x)

		# f(x-h) の計算
		x[idx] = float(tmp_val) - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val
		it.iternext()

	return grad

if __name__ == "__main__":
	a = numerical_gradient(function_2, np.array([3.0, 4.0]))
	print("numerical_gradient(function_2, np.array([3.0, 4.0]) = " + str(a))
	b = numerical_gradient(function_2, np.array([0.0, 2.0]))
	print("numerical_gradient(function_2, np.array([0.0, 2.0]) = " + str(b))
	c = numerical_gradient(function_2, np.array([3.0, 0.0]))
	print("numerical_gradient(function_2, np.array([3.0, 0.0]) = " + str(c))
