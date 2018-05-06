"""
シグモイド関数を用いることで、バイアスを自動的に決定することができる。
ちなみに、シグモイド関数は微分に大きく関係している。
"""

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1) # -5.0から5.0まで0.1刻みでNumpy配列を生成
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y軸の範囲を固定
plt.show()
