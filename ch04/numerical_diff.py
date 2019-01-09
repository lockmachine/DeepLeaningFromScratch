import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x - h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x1, x2):
	return x1**2 + x2**2

if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)   # 0 から 20 まで 0.1 刻みの配列
    y = function_1(x)

#    plt.xlabel("x")
#    plt.ylabel("f(x)")
#    plt.plot(x, y)

    print("numerical_diff(function_1, 5) = " + str(numerical_diff(function_1, 5)))
    print("numerical_diff(function_1,10) = " + str(numerical_diff(function_1, 10)))

    tan1 = numerical_diff(function_1, 5) * x + function_1(5) - 5 * numerical_diff(function_1, 5)
    tan2 = numerical_diff(function_1,10) * x + function_1(10) - 10 * numerical_diff(function_1, 10)
#    plt.plot(x, tan1)
#    plt.plot(x, tan2)
#    plt.show()

    x2d_1 = np.arange(-20.0, 20.0, 0.1)
    x2d_2 = np.arange(-20.0, 20.0, 0.1)

    X1, X2 = np.meshgrid(x2d_1, x2d_2)

    Y = function_2(X1, X2)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X1, X2, Y)

    plt.show()
