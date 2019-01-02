import numpy as np
import matplotlib.pyplot as plt

print("\n#3.3.1 多次元配列")
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1,2], [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])


print("\n#3.3.2 行列の積")
print("2x2行列と2x2行列の積")
A = np.array([[1,2], [3,4]])
print(A)
B = np.array([[5,6], [7,8]])
print(B)

# 行列の計算
print(np.dot(A, B))

print("2x3行列と3x2行列の積")
A = np.array([[1,2,3], [4,5,6]])
print(A.shape)
B = np.array([[1,2], [3,4], [5,6]])
print(B.shape)
print(np.dot(A, B))

print("2x3行列と2x2行列の積は計算できない")
C = np.array([[1,2], [3,4]])
print(C.shape)
# print(np.dot(A, C))   # これは計算できない

print("3x2行列と1x2ベクトルの計算")
A = np.array([[1,2], [3,4], [5,6]])
print(A.shape)
B = np.array([7,8])
print(B.shape)
print(np.dot(A, B))

print("\n#3.3.3 ニューラルネットワークの行列の積")
X = np.array([1, 2])
print(X)
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
print("X*Wの行列積")
Y = np.dot(X, W)
print(Y)
