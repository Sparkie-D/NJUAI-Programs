import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from data_process import display
from VAEbased import cluster_acc, sklearn_PCA
import h5py
from sklearn import metrics


data_mat = h5py.File('..//spca_dat//spca_dat//sample_151510.h5', 'r')

adata = np.array(data_mat['X'], dtype=np.float32)
adata = sklearn_PCA(adata, 30)
Y = np.array(data_mat['Y'])
pos = np.array(data_mat['pos']).T

adata = sc.AnnData(adata)

sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
sc.tl.louvain(adata, resolution=.8)
Y_raw = adata.obs['louvain'].values.tolist()
Y_pred = [eval(item) for item in Y_raw]


# print(cluster_acc(Y, Y_pred))

plt.subplot(121)
display(pos, Y_pred, False)
plt.subplot(122)
display(pos, Y, True)

nmi = np.round(metrics.normalized_mutual_info_score(Y, Y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(Y, Y_pred), 5)
print("nmi=%.8f, ari=%.8f"%(nmi,ari))

# sc.pl.pca(adata, color='louvain')