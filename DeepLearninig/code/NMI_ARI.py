import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from myDeepCluster import myDeepCluster, Normalization
from data_process import display, load_data


ifplot = False  # 若要绘图，则设为True
if __name__ == "__main__":
    dir = "..//spca_dat//spca_dat"
    h5files = os.listdir(dir)
    samples = [item for item in h5files if item[:6] == 'sample'] # 全部的sample文件
    ids     = [item[7:13] for item in samples]  # 全部的文件标号,str格式
    # samples, ids = samples[:2],ids[:2]

    NMIs, ARIs, idxs = [], [], []
    plainNMI, plainARI = [], []
    for idx in range(len(samples)):
        path = samples[idx]
        id = ids[idx]
        print(path, id)
        X, Y, pos = load_data(dir + "//"+ path)
        X, Y, pos = np.array(X, dtype=np.float32), np.array(Y), np.array(pos, dtype=np.float32).T
        Normalization(X)
        model = myDeepCluster(X.shape[-1], 256, 64)
        model.show = False # 若要显示训练进度，则设置为True
        model.AEtrain(X)
        y_pred, nmi, ari = model.Clustering(X, Y, len(set(Y)))
        NMIs.append(nmi)
        ARIs.append(ari)
        idxs.append(idx)

        if ifplot:
            save_path = "figures//"+ id + ".png"

            plt.subplot(121)
            display(pos, y_pred, show=False)
            plt.xlabel("My Deep Cluster")
            plt.subplot(122)
            display(pos, Y, show=False)
            plt.xlabel("Real Label")

            plt.savefig(save_path)
            plt.show()

        pca = PCA(n_components=64)
        pca.fit(X)
        z = pca.transform(X)
        kmeans = KMeans(n_clusters=len(set(Y)), n_init=20)
        plain_y_pred = kmeans.fit_predict(X)
        plainNMI.append(np.round(metrics.normalized_mutual_info_score(Y, plain_y_pred), 5))
        plainARI.append(np.round(metrics.adjusted_rand_score(Y, plain_y_pred), 5))

        print("%.5f, %.5f\n%.5f, %.5f"%(NMIs[-1], ARIs[-1],plainNMI[-1], plainARI[-1]))

    plt.subplot(121)
    plt.plot(idxs, NMIs, color='r',label='MyDeepCluster')
    plt.plot(idxs, plainNMI, color='b', label='PCA+Kmeans')
    plt.xlabel('NMI')

    plt.subplot(122)
    plt.plot(idxs, ARIs, color='r', label='MyDeepCluster')
    plt.plot(idxs, plainARI, color='b', label='PCA+Kmeans')
    plt.xlabel("ARI")

    plt.legend()
    plt.show()
