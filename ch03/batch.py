import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from neuralnet_mnist import get_data, init_network, predict
import pprint as pprint

if __name__ == "__main__":
	x, t = get_data()
	network = init_network()
	W1, W2, W3 = network["W1"], network["W2"], network["W3"]

	print("x.shape = " ,x.shape)		# (10000, 784)
	print("x[0].shape = ", x[0].shape)	# (784, )
	print("W1.shape = ", W1.shape)		# (784, 50)
	print("W3.shape = ", W3.shape)		# (50, 100)
	print("W2.shape = ", W2.shape)		# (100, 10)

	batch_size = 100	# バッチの数
	accuracy_cnt = 0

	for i in range(0, len(x), batch_size):	# 0-10000 まで 100 単位で実行する
		x_batch = x[i:i+batch_size,:]
		y_batch = predict(network, x_batch)	# (100, 10)

		index = np.argmax(y_batch, axis=1)	# (100, )

		accuracy_cnt += np.sum(index == t[i:i+batch_size])

	print("Accuracy:{}".format(accuracy_cnt / len(x)))
