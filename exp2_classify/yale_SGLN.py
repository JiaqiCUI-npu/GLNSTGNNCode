import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from read_data import read_yale
import sys
sys.path.append(".")
from SGLN import LassoNetRegressor

# 加载数据
path = 'data/Yale_32x32.mat'
X, y = read_yale(path)  # X(165,1024), y(165,),总共15个人，每人11个表情
category = 15

data = []

for i in range(category):
    filter = y == (i+1)
    data.append(X[filter])

#构造数据集
sample_num = 8
X_train = []
y_train = []
y_lable = []
every_class_num = 3

for i in range(category):
     X_train += list(data[i])[0 : sample_num]
     y_train += list(data[i])[sample_num: ]
     class_lable = every_class_num * [i]
     y_lable += class_lable
X_train = np.array(X_train).T


print("################# Start ###############")
Groups = []
for i in range(category):
    group = np.arange(start=i*sample_num, stop = (i+1)*sample_num, step=1)
    Groups.append(group)

model = LassoNetRegressor(
    hidden_dims=(100,),
    #n_iters=(3000, 500),   # 控制epochs
    path_multiplier=1.05,   # 控制Lambda_1变化趋势
    lambda_start=0.2,
    alpha=0.5,
    verbose=False,   # 决定要不要输出log信息
)
classify_weight = []
for k, y in enumerate(y_train):
    path = model.path(X_train, y, Groups = Groups)
    weight = path[-2].state_dict["skip.weight"]
    importances = np.linalg.norm(weight, ord=2, axis=0)
    print(str(k) + " finished")
    this_weight = []
    for i in range(0, category):
        this_weight.append(sum(importances[i*sample_num: (i+1)*sample_num]) / sum(importances))
    print(this_weight)
    classify_weight.append(this_weight)
print("################# End #################")

#输出准确率
pred = np.argmax(np.array(classify_weight), axis=1)
print("The accuracy score: ", accuracy_score(y_lable, pred))
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
# fig.savefig("exp2_classify/figures/heatmap_Lasso.pdf", bbox_inches='tight')