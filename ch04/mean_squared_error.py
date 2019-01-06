import numpy as np

def mean_squared_error(y, t):
	e = 0.5 * np.sum((y-t)**2)
	return e
	
