import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from sigmoid import sigmoid
from softmax import softmax
import pickle
from pprint import pprint

def get_data():
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(normalize=True, flatten=True, one_hot_label=False)

	return x_test, t_test

def init_network():
	with open("sample_weight.pkl", "rb") as f:
		network = pickle.load(f)

	return network

def predict(network, x):
	W1, W2, W3 = network["W1"], network["W2"], network["W3"]
	b1, b2, b3 = network["b1"], network["b2"], network["b3"]

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	z3 = softmax(a3)

	return z3

if __name__ == "__main__":
	x, t = get_data()		# x = (10000, 784) t = (10000, )
	network = init_network()

	y = predict(network, x)	# (10000, )

	accuracy_cnt = 0

	for i in range(len(x)):
		index = np.argmax(y[i])	# 各画像で最も確率の高い要素のインデックスを取得する
		if index == t[i]:
			accuracy_cnt += 1

	pprint("Accuracy:" + str(float(accuracy_cnt) / len(x)))
