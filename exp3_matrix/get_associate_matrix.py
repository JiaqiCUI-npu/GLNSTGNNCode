# 导包
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas import Series
from sklearn.preprocessing import LabelEncoder
import torch
import sys
sys.path.append(".")
import SGLN
from SGLN import LassoNetRegressor
from scipy.io import loadmat
from scipy.io import savemat


#加载数据
data = loadmat('data/side_information.mat')

X_train = data['X'].T
y_train = data['X']
sample_num = 5
matrix = []
Groups = [[] for _ in range(10)]
#训练
print("########## Training Start ##########")
for i in range(3):
    for j in range(10):
        Groups[j].append(10 * i + j)

classify_weight = []
for k, y in enumerate(y_train):
    model = LassoNetRegressor(
        hidden_dims=(100,),
        #n_iters=(3000, 500),   # 控制epochs
        path_multiplier=1.05,   # 控制Lambda_1变化趋势
        lambda_start=0.2,
        alpha=0.5,
        verbose=False,   # 决定要不要输出log信息
    )
    path = model.path(X_train, y, Groups = Groups)
    weight = path[-2].state_dict["skip.weight"]
    importances = np.linalg.norm(weight, ord=2, axis=0)
    total = sum(importances)
    new_coff = [round(num / total, 3) for num in importances ]
    print(str(k) + " finished")
    print(new_coff)
    matrix.append(new_coff)
print("########## Training End ##########")

matrix = np.array(matrix)


