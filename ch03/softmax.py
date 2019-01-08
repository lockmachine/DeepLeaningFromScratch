import numpy as np
import matplotlib.pyplot as plt
import pprint

# 3.5.1 恒等関数とソフトマックス関数
# 実装上の注意
'''
オーバーフローに注意する
exp(10) は 20,000 を超えて、exp(100) は 0 が 40 個以上も並ぶ大きな値になる
exp(1000) の結果は無限大を表す inf が返ってくる
'''
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # オーバーフロー対策->データの最大値を減算することで、データの中心化を行う
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])

    exp_a = np.exp(a)   # 指数関数
    print(exp_a)

    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)

    y = exp_a / sum_exp_a
    print(y)

    a = np.array([1010, 1000, 990])
    print(np.exp(a) / np.sum(np.exp(a)))    # ソフトマックスの計算（正しく計算されない）

    print(softmax(a))
