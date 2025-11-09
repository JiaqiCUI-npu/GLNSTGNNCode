from scipy.io import loadmat, savemat
import cv2
import numpy as np
from pandas import Series
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib import rcParams


def data_augmentation(old_data):
    new_data = old_data.reshape(32, 32)
    new_data = cv2.flip(new_data, 0)  # 水平翻转
    new_data = cv2.GaussianBlur(new_data, (3, 3), 0)  # 高斯模糊
    # new_data = np.array(new_data * 0.9, np.uint8)  # 变暗20%
    new_data = new_data.reshape(1024, )
    return new_data

def read_data(path):
    yale = loadmat(path)
    X = yale['fea']
    X = X / 255.
    y = yale['gnd'].T.flatten()  # 将数据展开
    y = Series(y.astype(np.uint8))
    for i in range(len(X)):
        if i % 11 == 2:
            y[i] = 1
        elif i % 11 == 7:
            y[i] = 0
        else:
            y[i] = 2
    filter = y.isin([0, 1])
    data = X[filter]
    labels = LabelEncoder().fit_transform(y[filter])
    return data, labels

def see_data(img):
    config = {
        "font.family": 'Times New Roman',
        "font.size": 12
    }
    rcParams.update(config)
    plt.title(f"The image after augmentation")

    img = img.reshape(32, 32).T
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":

    #读取数据
    data, labels = read_data('../data/Yale_32x32.mat')
    new_data = []
    new_labels = []
    # 遍历所有图片
    for i in range(data.shape[0]):
        img = data_augmentation(data[i])
        new_data.append(img)
        new_labels.append(labels[i])

    # 可视化
    # see_data(data[4])
    # see_data(new_data[4])
    # 新mat文件
    new_mat = {'data':np.concatenate((data, new_data), axis=0), 'labels':np.concatenate((labels, new_labels), axis=0)}

    savemat('data_aug.mat', new_mat)

