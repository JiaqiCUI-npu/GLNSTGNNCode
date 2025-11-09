import random

import numpy as np
from pandas import Series
from scipy.io import loadmat
import sys
sys.path.append("../examples")
from SGLN import LassoNetRegressor
# from lassonet import LassoNetRegressor

mnist = loadmat('../examples/data/Mnist/mnist-original.mat')

X = mnist['data'].T               #这个一定要转置一下，因为这里面的行列是反的！！！！！
X = X / 255.
y = mnist['label'].T.flatten()     #将数据展开
y= Series(y.astype(np.uint8))              #将格式变为uint8

data = []
target = []
j = 0
for i in range(10):
    filter = y == i
    data.append(X[filter])

#每个类中随机挑选100个数据
sample_num = 100
X_train = []
y_train = []


for i in range(10):
    X_train += random.sample(list(data[i]), sample_num)

flag = random.sample(list(range(0, 6903)), 1)
print(flag)
X_train = np.array(X_train).T
y_train = np.array(data[0][flag[0]])


model = LassoNetRegressor(
    hidden_dims=(100,),
    verbose=2,
    patience=(100, 5)
)


Groups = []
for i in range(10):
    group = np.arange(start=i*100, stop = (i+1)*100, step=1)
    Groups.append(group)

path = model.path(X_train, y_train, Groups=Groups)
# path = model.path(X_train, y_train)

importances = model.feature_importances_.numpy()


# 绘制测试数据的散点图
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt


x=[]
y=[]


df = importances / importances.max()
for i in range(len(df)):
    if df[i] != 0:
        yi = [df[i], -0.15]
        y.append(yi)
        xi = [i,i]
        x.append(xi)



for i in range(len(x)):
    plt.scatter(x[i], y[i], color='black', s=5)

    plt.plot(x[i], y[i])


# 设置x轴刻度
x_major_locator = MultipleLocator(100)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

plt.ylim(0.8, 1)

plt.title("The coef_ of the picture'0'")

plt.savefig("figures/SGLN-fig.png")