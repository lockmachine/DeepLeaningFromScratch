import numpy as np

def mean_squared_error(y, t):
	e = 0.5 * np.sum((y-t)**2)
	return e

def cross_entropy_error(y, t):
	delta = 1e-7
	e = -np.sum(t * np.log(y+delta))
	return e

if __name__ == "__main__":
	t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
	y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
	print("mean_squared_error : " + str(mean_squared_error(y, t)))
	print("cross_entropy_error: " + str(cross_entropy_error(y, t)))

	y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
	print("mean_squared_error : " + str(mean_squared_error(y, t)))
	print("cross_entropy_error: " + str(cross_entropy_error(y, t)))
