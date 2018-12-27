print("\n# 3.2.2 ステップ関数の実装")

import numpy as np

# for文を使ったステップ関数の実装
# 返り値が リスト型 になる
def step_function(x):
    y = []
    print(x)
    for i in x:
        if i > 0:
            y.append(1)
        else:
            y.append(0)
    return y

# numpyメソッドを使ったステップ関数の実装
# 返り値が ndarray になる
def step_function_short(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1, 1, 2])

y = step_function(x)
print(y)
print(type(y))  # リスト型

y_short = step_function_short(x)
print(y_short)
print(type(y_short))    # ndarray型

print("\n#3.2.3 ステップ関数のグラフ")
import matplotlib.pyplot as plt

def step_function_very_short(x):
    return np.array(x > 0, np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function_very_short(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

print("\n#3.2.4 シグモイド関数の実装")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

print("\n#3.2.7 ReLU 関数")
def relu(x):
    return np.maximum(0, x)

x = np.arange(-10.0, 10.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()
