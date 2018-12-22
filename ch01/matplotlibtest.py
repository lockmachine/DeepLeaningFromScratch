import numpy as np
import matplotlib.pyplot as plt

print("\n # 1.6.1 単純なグラフの描画")
# データの作成
x = np.arange(0, 6, 0.1)	# 0 から 5.9 まで 0.1 刻みで生成
y = np.sin(x)
print(x)

plt.plot(x, y)
plt.show()

print("\n # 1.6.2 pyplotの機能")
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos")
plt.xlabel("x")	# x 軸のラべル
plt.ylabel("y")	# y 軸のラベル
plt.title("sin & cos")	# タイトル


plt.legend()
plt.show()

print("\n # 1.6.3 画像の表示")
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("../dataset/lena.png")	# 画像の読み込み（適切なパスを設定する！）
plt.imshow(img)

plt.show()
# 写経終了
