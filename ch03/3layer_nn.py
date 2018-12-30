import numpy as np
import sigmoid as sig
import identity_function as identity

print("\n#3.4 3層ニューラルネットワークの実装")
print("\n#3.4.2 各層における信号伝達の実装")

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print("X = ")
print(X)
print("W1 = ")
print(W1)
print("B1 = ")
print(B1)

print("A1 = X * W1 + B1")
A1 = np.dot(X, W1) + B1
print(A1)

print("Aに活性化関数としてシグモイド関数を適用する")
Z1 = sig.sigmoid(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print("A2 = Z1 * W2 + B2")
A2 = np.dot(Z1, W2) + B2
Z2 = sig.sigmoid(A2)
print("Z2")
print(Z2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print("A3 = Z2 * W3 + B3")
A3 = np.dot(Z2, W3) + B3
Z3 = identity.identity_function(A3)
print("Z3")
print(Z3)
