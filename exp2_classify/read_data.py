from pandas import Series
import numpy as np
from scipy.io import loadmat

def read_mnist(path):
    mnist = loadmat(path)
    X = mnist['data'].T               #这个一定要转置一下，因为这里面的行列是反的！！！！！
    X = X / 255.
    y = mnist['label'].T.flatten()     #将数据展开
    y= Series(y.astype(np.uint8))              #将格式变为uint8
    return X, y

def read_yale(path):
    # 加载数据
    yale = loadmat(path)
    #载入YALE32mat文件的方法，得到的x是一个字典，可以print一下他的shape看一下里面的属性
    X = yale['fea']
    X = X / 255.
    print(X.shape)
    y = yale['gnd'].T.flatten()     #将数据展开
    y= Series(y.astype(np.uint8))
    return X, y

def read_coil(path):
    coil = loadmat(path)
    X = coil['fea']
    print(X.shape)
    y = coil['gnd'].T.flatten()     #将数据展开
    y= Series(y.astype(np.uint8))
    return X, y
