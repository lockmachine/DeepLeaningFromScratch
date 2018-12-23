print("\n# 2.3.1 簡単な実装")

def AND(x1, x2):
	w1, w2, theta = 0.5, 0.5, 0.7
	tmp = x1*w1 + x2*w2
	if tmp <= theta:
		return 0
	elif tmp > theta:
		return 1

print(AND(0, 0))	# 0 を出力
print(AND(0, 1))	# 0 を出力
print(AND(1, 0))	# 0 を出力
print(AND(1, 1))	# 1 を出力

print("\n# 2.3.2 重みとバイアスの導入")
import numpy as np
x = np.array([0, 1])		# 入力
w = np.array([0.5, 0.5])	# 重み
b = -0.7					# バイアス

print(w*x)
print(np.sum(w*x))
print(np.sum(w*x) + b)
