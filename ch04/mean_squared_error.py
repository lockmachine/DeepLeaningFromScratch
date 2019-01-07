import numpy as np

def mean_squared_error(y, t):
	e = 0.5 * np.sum((y-t)**2)
	return e

def cross_entropy_error(y, t):
	if y.ndim == 1:	# y の次元が1次元なら
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	e = -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
	return e

if __name__ == "__main__":
	t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
	y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
	print("mean_squared_error : " + str(mean_squared_error(y, t)))
	print("cross_entropy_error: " + str(cross_entropy_error(y, t)))

	y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
	print("mean_squared_error : " + str(mean_squared_error(y, t)))
	print("cross_entropy_error: " + str(cross_entropy_error(y, t)))
