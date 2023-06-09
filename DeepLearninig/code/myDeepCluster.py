import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
from sklearn import metrics
from sklearn.cluster import KMeans
from data_process import load_data, display
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class myDeepCluster(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(myDeepCluster, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.ReLU()
        )
        self.recon_loss = nn.MSELoss()
        # self.recon_loss = nn.CrossEntropyLoss()
        self.show=True
        self.to(device)

    def Q(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.centers) ** 2, dim=2))
        q = q ** ((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def P(self, q):
        p = q ** 2 / q.sum(0)
        res = (p.t() / p.sum(1)).t()
        return res

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

    def AEtrain(self, x, batch_size=256, lr=0.001, epochs=200, save=False):
        print("Training AE")
        x = torch.tensor(x, dtype=torch.float32).to(device)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            # tot_loss = 0
            start = 0
            end = start + batch_size
            while end < x.shape[0]:
                # print(batch_idx, xbatch)
                xbatch = x[start:end]
                zbatch, x_new = self.forward(xbatch)
                loss = self.recon_loss(xbatch, x_new)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # tot_loss += loss
                if end >= x.shape[0] -1:
                    break

                start = end
                end = start + batch_size if start + batch_size < x.shape[0] else x.shape[0] - 1
            if self.show:
                print("epoch %d, loss = %.6f"%( epoch, loss))

            if epoch % 100 == 0 and save:
                torch.save(self, f'model//myDeepCluster_{epoch}.pth')

    def Clustering(self, x, Y, n_clusters, batch_size=256, epochs=100, lr=0.01, show=False):
        print("fit clustering")
        self.centers = nn.Parameter(torch.Tensor(n_clusters, self.latent_dim).to(device))

        # 初始化聚类
        inputs = torch.tensor(x, dtype=torch.float32).to(device)
        kmeans = KMeans(n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder(inputs).cpu().detach().numpy())
        self.centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        final_y_pred = self.y_pred
        final_nmi, final_ari, min_loss, tot_closs, tot_rloss = 0, 0, 0, 0, 0
        for epoch in range(epochs):
            latent_x, _ = self.forward(inputs)
            q = self.Q(latent_x)
            p = self.P(q)
            # print(p.shape, q.shape)
            self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            nmi = metrics.normalized_mutual_info_score(Y, self.y_pred)
            ari = metrics.adjusted_rand_score(Y, self.y_pred)
            if nmi > final_nmi:
                final_y_pred = self.y_pred
                final_nmi = nmi
                final_ari = ari
            if self.show:
                print("epoch %3d, nmi=%.4f, ari=%.4f, closs=%.4f, rloss=%.4f" % (epoch, nmi, ari, tot_closs, tot_rloss))

            start = 0
            end = start + batch_size
            tot_closs, tot_rloss = 0, 0
            while True:
                batch_input = torch.autograd.Variable(inputs[start:end]).to(device)
                batch_p = torch.autograd.Variable(p[start:end]).to(device)
                batch_z, batch_output = self.forward(batch_input)
                batch_q = self.Q(batch_z)
                # batch_p = self.P(batch_q)
                closs = self.cluster_loss(batch_p, batch_q)
                rloss = self.recon_loss(batch_input, batch_output)

                loss = closs + rloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_closs += closs
                tot_rloss += rloss

                if end >= x.shape[0] - 1:
                    break

                start = end
                end = start + batch_size if start + batch_size < x.shape[0] else x.shape[0] - 1

        return final_y_pred, final_nmi, final_ari

def Normalization(X):
    for i in range(len(X[0])):
        line = X[:, i]
        minval = min(line)
        maxval = max(line)
        minval = minval if maxval > minval else minval - 1
        # print(line, maxval, minval)
        for j in range(len(line)):
            line[j] = (line[j] - minval) / (maxval - minval)


if __name__ == "__main__":
    path = '..//spca_dat//spca_dat//sample_151510.h5'
    X, Y, pos = load_data(path)
    X, Y, pos = np.array(X, dtype=np.float32), np.array(Y), np.array(pos, dtype=np.float32).T
    # X = torch.tensor(X).to(device)
    Normalization(X)
    model = myDeepCluster(X.shape[-1], 256, 64)
    model.show = False  # 取消注释显示训练过程
    # print(str(model))
    model.AEtrain(X)
    # model = torch.load("model//myDeepCluster_200.pth")
    y_pred, nmi, ari = model.Clustering(X, Y, len(set(Y)), show=True)

    print("Result nmi = %.4f, ari = %.4f"%(nmi, ari))

    plt.subplot(121)
    display(pos, y_pred, show=False)
    plt.xlabel("my DeepCluster")

    plt.subplot(122)
    display(pos, Y, show=False)
    plt.xlabel("Real Label")

    plt.show()









