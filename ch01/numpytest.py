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


print("\n# 1.5.4 NumPyのN次元配列")
A = np.array([[1, 2], [3, 4]])
print(A)

print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A + B)

print(A * B)

print(A * 10)

print("\n# 1.5.5 ブロードキャスト")
B = np.array([10, 20])
print(A * B)	# [[10, 40], [30, 80]]

print("\n# 1.5.6 要素へのアクセス")
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])
print(X[:,1])

print("for文")
for row in X:
	print(row)

X = X.flatten()
print(X)	# X.shapeは（6, )
print(X[np.array([0, 2, 4])])	#インデックスが０，２，４番目の要素を取得

print(X > 15)
print(X[X >15])
