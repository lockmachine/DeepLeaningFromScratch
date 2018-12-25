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
