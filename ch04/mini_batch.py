import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =\
	load_mnist(normalize=True, flatten=True, one_hot_label=True)

print(x_train.shape)	# (60000, 784)
print(t_train.shape)	# (60000, 10)

train_size = x_train.shape[0]	# 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)	# 0から60000未満の中からランダムに10個の数字を選び出す
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print("batch_mask = " + str(batch_mask))
print("x_batch.shape = " + str(x_batch.shape) + "\nx_batch = \n" + str(x_batch))
print("t_batch.shape = " + str(t_batch.shape) + "\nt_batch = \n" + str(t_batch))
