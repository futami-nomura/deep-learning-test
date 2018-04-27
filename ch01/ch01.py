import scipy as sp


# 元データを確認する
data = sp.genfromtxt("web_traffic.tsv",delimiter="\t")
print (data.shape)


# データを次元ごとに分割する
x = data[:,0]
y = data[:,1]

# 不要なデータがあった場合に排除する
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i' %w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()

"""
- ノイズが含まれたデータに対して、その背後にある本当のモデルを見つける
- そのモデルを用いて、予測を立てる
"""

# ユークリッド平方距離によって実数値との乖離を求める関数
def error(f,x,y):
    return sp.sum((f(x)-y)**2)


"""
np.polyfit(x, y, n) : n 次式で 2 変数の回帰分析
np.polyval(p, t) : p で表される多項式に t を代入し値を計算
"""
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

plt.scatter(x,y)
plt.title("web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i' %w for w in range(10)])
plt.autoscale(tight=True)
f1 = sp.poly1d(fp1)
print(error(f1,x,y)) # どれくらい外れているかを算出している
fx = sp.linspace(0,x[-1], 1000) # プロット用に"x値"を生成
plt.plot(fx,f1(fx),linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")
f2p = sp.polyfit(x, y, 2)
print(f2p)
f2 = sp.poly1d(f2p)
plt.plot(fx,f2(fx),linewidth=4)
plt.legend(["d=%i" % f2.order], loc="upper left")
plt.grid()
plt.show()


"""
変化点を元に計算する
inflection = 3.5 * 7 * 24 # 変化点（急に変化する点の時間を計算
xa = x[:inflection] # 変化前のデータポイント
ya = y[:inflection]
xb = x[inflection:] # 変化後のデータポイント
yb = y[inflection:]

fa = sp.poly1d(sp.ployfit(xa,ya,1))
fb = sp.poly1d(sp.polyfit(xb,yb,1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("Error inflection=%f" % (fa_error + fb_error))
"""
