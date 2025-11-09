import random
from pandas import Series
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(".")
from lassonet import LassoNetRegressor
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt
from matplotlib import rcParams
from read_data import read_mnist

#加载数据
path = 'data/Mnist/mnist-original.mat'
X, y = read_mnist(path)
data = []
for i in range(10):
    filter = y == i
    data.append(X[filter])

#构造需要的数据集
sample_num = 100
X_train = []
y_train = []
y_lable = []
every_class_num = 10
for i in range(0, 10):
     X_train += list(data[i])[0 : sample_num]
     y_train += list(data[i])[sample_num: sample_num + every_class_num]
     class_lable = every_class_num * [i]
     y_lable += class_lable
X_train = np.array(X_train).T

#训练
print("########## Training Start ##########")

classify_weight = []
for k, y in enumerate(y_train):
    model = LassoNetRegressor(
        hidden_dims=(100,),
        path_multiplier=1.05,   # 控制Lambda_1变化趋势
        verbose=0,
        patience=(100, 5),
        lambda_start=0.5
    )
    path = model.path(X_train, y)
    weight = path[-2].state_dict["skip.weight"]
    importances = np.linalg.norm(weight, ord=2, axis=0)
    this_weight = []
    for i in range(0, 10):
        this_weight.append(sum(importances[i*100: (i+1)*100]) / sum(importances))
    print(str(k) + "   finished")
    print(this_weight)
    classify_weight.append(this_weight)
print("########## Training End ##########")

#输出准确率
pred = np.argmax(np.array(classify_weight), axis=1)
print("The accuracyscore: ", accuracy_score(y_lable, pred))
# print(y_lable)
# print(pred)

#平均置信度
sum = 0
num = 0
for i in range(len(pred)):
    if(pred[i] == y_lable[i]):
        sum += classify_weight[i][pred[i]]
        num += 1
print("The average confidence: ", sum * 1.0 / num)

#输出热图
# data = np.array(classify_weight)
# config = {
#     "font.family": 'Times New Roman',
#     "font.size": 12
# }
# rcParams.update(config)
# plt.figure(dpi=100)
# ax = sns.heatmap(data,annot=False)
# fig = ax.get_figure()
# fig.savefig("exp2_classify/figures/heatmap_lassonet.pdf", bbox_inches='tight')