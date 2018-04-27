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
