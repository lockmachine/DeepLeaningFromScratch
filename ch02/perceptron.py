print("\n# 2.3.1 簡単な実装")

'''# 2.3.3 のためにコメントアウト
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
'''

print("\n# 2.3.2 重みとバイアスの導入")
import numpy as np
x = np.array([0, 1])		# 入力
w = np.array([0.5, 0.5])	# 重み
b = -0.7					# バイアス

print(w*x)					# 入力x重み
print(np.sum(w*x))			# 入力x重みの総和
print(np.sum(w*x) + b)		# 入力x重みの総和にバイアスをかける

print("\n# 2.3.3 重みとバイアスによる実装")

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(x*w) + b
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1


def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5])
	b = 0.7
	tmp = np.sum(x*w) + b
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1


def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.3
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1

print("\n# 2.5.2 XORゲートの実装")
def XOR(x1, x2):
	y_nand = NAND(x1, x2)	# 第1層の出力
	y_or = OR(x1, x2)		# 第2層の出力
	y = AND(y_nand, y_or)	# 出力層の出力
	return y

print(XOR(0, 0))	# 0 を
print(XOR(0, 1))	# 1 を
print(XOR(1, 0))	# 1 を
print(XOR(1, 1))	# 0 を
