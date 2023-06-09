import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from h5py import Dataset, Group, File
import matplotlib.pyplot as plt


path = '../spca_dat/spca_dat/sample_151510.h5'  # 默认文件
def load_data(path = path):
    content = {}
    keys = []
    with File(path, 'r') as f:
        for key in f.keys():
            keys.append(key)
            content[key] = f[key][()]
            # print(f[key], key, f[key].name)
            # print("key:", key)
            # print(f[key][()])

    X = content[keys[0]]
    Y = content[keys[1]]
    pos = content[keys[2]]
    return X, Y, pos

def sklearn_PCA(inputs, k):
    pca = PCA(n_components=k)  # 定义所需要分析主成分的个数n
    pca.fit(inputs)  # 对基础数据集进行相关的计算，求取相应的主成分
    # print(pca.components_)  # 输出相应的n个主成分的单位向量方向
    return pca.transform(inputs)  # 进行数据的降维

def sklearn_GMM(inputs, k):
    gmm = GMM(n_components=k).fit(inputs)
    labels = gmm.predict(inputs)
    return labels

def sklearn_kmeans(inputs, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # 创建一个K-均值聚类对象
    kmeans.fit(inputs)  # 拟合算法
    cluster_assignment = kmeans.predict(inputs)  # 获取聚类分配
    return cluster_assignment


def display(pos, Y, show=True):
    keyset = list(set(Y))
    nodes = {}
    # print(keyset)
    for idx in range(len(keyset)):
        if keyset[idx] not in nodes.keys():
            nodes[keyset[idx]] = []
    for idx in range(len(pos)):
        nodes[Y[idx]].append(pos[idx])
    for idx in range(len(nodes.keys())):
        key = list(nodes.keys())[idx]
        currentnodes = np.array(nodes[key]).T
        plt.scatter(currentnodes[0], currentnodes[1], s=5,color=plt.cm.Set3(idx))
    if show:
        plt.show()


