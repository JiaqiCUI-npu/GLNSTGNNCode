from pandas import Series
import numpy as np
from read_data import read_mnist
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt
from matplotlib import rcParams
from groupyr import SGL

#加载数据
path = 'data/Mnist/mnist-original.mat'
X, y = read_mnist(path)
data = []
for i in range(10):
    filter = y == i
    data.append(X[filter])

#构造数据集
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
Groups = []
for i in range(10):
    group = np.arange(start=i*sample_num, stop = (i+1)*sample_num, step=1)
    Groups.append(group)
classify_weight = []
for k, y in enumerate(y_train):
    model = SGL(groups=Groups, l1_ratio=0, alpha=0.02).fit(X_train, y)
    importances = abs(model.coef_)
    this_weight = []
    for i in range(0, 10):
        this_weight.append(sum(importances[i*sample_num: (i+1)*sample_num]) / sum(importances))
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
# fig.savefig("exp2_classify/figures/heatmap_grouplasso.pdf", bbox_inches='tight')