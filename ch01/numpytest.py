import numpy as np

x = np.array([1.0, 2.0, 3.0])
print("x:")
print(x)

tp = type(x)
#print("tpye(x) = " + tp)   #これはできない
print(type(x))
print(type("string") is str)    #type()の返り値とstrを比較することで、そのオブジェクトがstr型であるかを判定する

y = np.array([2.0, 4.0, 6.0])
print("y:")
print(y)

print("x + y:")
print(x + y)    # 要素ごとの足し算

print("x - y:")
print(x - y)    # 要素ごとの引き算

print("x * y:")
print(x * y)    # 要素ごとの掛け算

print("x / y:")
print(x / y)    # 要素ごとの割り算

print("x / 2.0:")
print(x / 2.0)	# 各要素を2で割る

# NumpyのN次元配列
print("NumpyのN次元配列")
A = np.array([[1, 2],[3, 4]])
print(A)

print(A.shape)
print(A.dtype)

B = np.array([[3, 0],[0, 6]])
print(B)
print(A + B)
print(A * B)

print(A)
print(A * 10)
