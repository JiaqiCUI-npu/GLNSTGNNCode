from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(".")
from SGLN import LassoNetRegressor

# X, y = fetch_openml(name="mnist_784", return_X_y=True)  # 得到的X是DataFrame类型
# filter = y == "3"
# X = X[filter].values / 255

mnist = loadmat('data/Mnist/mnist-original.mat')

X = mnist['data'].T               #这个一定要转置一下，因为这里面的行列是反的！！！！！
# X = X / 255.
y = mnist['label'].T.flatten()     #将数据展开
y= Series(y.astype(np.uint8))              #将格式变为uint8

filter = y == 3
X = X[filter] / 255.

# 分组
matrix = np.arange(784).reshape((28, 28))
n = 7    # 矩阵每行分成几份
Groups = []
m1 = np.split(matrix, n, axis=0)
for m in m1:
    group = np.split(m, n, axis=1)
    Groups = Groups + group

for i in range(len(Groups)):
    Groups[i] = Groups[i].flatten()

model = LassoNetRegressor(M=30, n_iters=(3000, 500), path_multiplier=1.05, verbose=2)
path = model.path(X, X, Groups=Groups)

img = model.feature_importances_.reshape(28, 28)

plt.title("Feature importance to reconstruct 3")
plt.imshow(img)
plt.colorbar()
plt.savefig("mnist-reconstruction-importance.png")


n_selected = []
score = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    X_pred = model.predict(X)
    n_selected.append(save.selected.sum())
    score.append(mean_squared_error(X_pred, X))
    lambda_.append(save.lambda_)

to_plot = [160, 220, 300]

for i, save in zip(n_selected, path):
    if not to_plot:
        break
    if i > to_plot[-1]:
        continue
    to_plot.pop()
    plt.clf()
    plt.title(f"Linear model with {i} features")
    weight = save.state_dict["skip.weight"]
    img = (weight[1] - weight[0]).reshape(28, 28)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(f"mnist-reconstruction-{i}.png")

plt.clf()

fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, score, ".-")
plt.xlabel("number of selected features")
plt.ylabel("MSE")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, score, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("MSE")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("mnist-reconstruction-training.png")


plt.subplot(221)
plt.imshow(X[150].reshape(28, 28))
plt.subplot(222)
plt.imshow(model.predict(X[150]).reshape(28, 28))
plt.subplot(223)
plt.imshow(X[250].reshape(28, 28))
plt.subplot(224)
plt.imshow(model.predict(X[250]).reshape(28, 28))
