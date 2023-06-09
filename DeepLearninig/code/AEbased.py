import torch
import MLP
from sklearn.decomposition import PCA
import data_process
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from math import log, sqrt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, x_dim),
            # nn.Softmax(dim=0)
        )

    def forward(self, z):
        return self.model(z)

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        tmp = self.encoder(x)
        return self.decoder(tmp)

def PreTrainAE(model, inputs, batch_size, loss_fn, optimizer, EPOCH):
    start = 0
    end = start + batch_size
    for epoch in range(EPOCH):
        while end < inputs.shape[0]:
            Input = inputs[start:end]
            output = model(Input)
            loss = loss_fn(output, Input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if end >= inputs.shape[0] - 1:
                break
            else:
                start = end
                end = start + batch_size if start + batch_size < inputs.shape[0] else inputs.shape[0]-1

        if epoch % 200 == 0:
            print("epoch", epoch, " current loss :", loss.item())
            torch.save(model, f'model//AEbasedmodel_{epoch}.pth')
            # print(f'model//AEbasedmodel_{epoch}.pth saved.')

def sklearn_kmeans(inputs, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)  # 创建一个K-均值聚类对象
    kmeans.fit(inputs)  # 拟合算法
    cluster_assignment = kmeans.predict(inputs)  # 获取聚类分配
    return cluster_assignment

def Normalization(X):
    for i in range(len(X[0])):
        line = X[:, i]
        minval = min(line)
        maxval = max(line)
        minval = minval if maxval > minval else minval - 1
        # print(line, maxval, minval)
        for j in range(len(line)):
            line[j] = (line[j] - minval) / (maxval - minval)


def sklearn_PCA(inputs, k):
    pca = PCA(n_components=10)  # 定义所需要分析主成分的个数n
    pca.fit(inputs)  # 对基础数据集进行相关的计算，求取相应的主成分
    # print(pca.components_)  # 输出相应的n个主成分的单位向量方向
    return pca.transform(inputs)  # 进行数据的降维

def preprocess_data():
    X, Y, pos = data_process.load_data()
    X, Y, pos = np.array(X, dtype=np.float32), np.array(Y), np.array(pos, dtype=np.float32).T
    inputs = np.concatenate((X, pos), axis=1) # 将位置信息和基因表达整合在一起输入AE
    inputs = torch.tensor(inputs)
    inputs = inputs.to(torch.float32).to(device) # 不需要时间戳
    return inputs, Y, pos

def OneHotEncoder(labels):
    lset = list(set(labels))
    res = [[0 for i in range(len(lset))] for j in range(len(labels))]
    for i in range(len(labels)):
        for j in range(len(lset)):
            if labels[i] == lset[j]:
                res[i][j] = 1
    return res

def distance(x, x_):
    # x, x_ = x.cpu(), x_.cpu()
    return sqrt(sum((x-x_)**2))

def WriteDistanceRect(inputs):
    inputs = inputs.cpu().numpy()
    cnt = 0
    print('Counting Distance Rect...')
    with open('distanceRect.txt', 'w') as f:
        for i in range(len(inputs)):
            for j in range(len(inputs)):
                tmp = distance(inputs[i], inputs[j])
                f.write('%f\n'%tmp)
                cnt += 1
                if cnt % 10000 == 0:
                    print('%.3f%'%cnt/len(inputs)**2*100)

def ReadDistanceRect(path, lineSize):
    res = []
    with open(path, 'r') as f:
        line = []
        for i in range(lineSize):
            line.append(f.readline())
        res.append(map(float, line))
    return np.array(res)

def countPredDis(inputs, centers):
    res = []
    for item in inputs:
        pred_dis = [distance(item, jtem) for jtem in centers]
        sums = sum(pred_dis)
        pred_dis = [item/sums for item in pred_dis]
        res.append(pred_dis)
    return torch.tensor(res, requires_grad=True).to(device)

# def countRealDis(targets):
#     targetSet = list(set(targets))
#     counts = [float(list(targets).count(item)) for item in targetSet]
#     res = [[item / len(targets) for item in counts] for i in range(targets.shape[0])]
#     return torch.tensor(res, requires_grad=True).to(device)

def countRealDis(Y):
    return torch.tensor(OneHotEncoder(Y), dtype=torch.float32, requires_grad=True).to(device)

def Q(latent_item, centers):
    # 自由度为1的student分布作为Q
    # res = torch.tensor([latent_item for item in centers], dtype=torch.float32, requires_grad=True)
    # for idx in range(len(res)):
    #     res[idx] = (1 + sum((res[idx] - centers[idx]) ** 2)) ** (-1)
    # sum_ups = sum(res)
    # for idx in range(len(res)):
    #     res[idx] = res[idx] / sum_ups
    # return res
    q = 1.0 / (1.0 + torch.sum((centers.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
    q = q ** ((self.alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, dim=1)).t()

# def P

class KLloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.KL = torch.nn.KLDivLoss(reduction='sum')

    # def KL_loss(self, pred_dis, real_dis):
    #     res = 0
    #     for i in range(len(pred_dis)):
    #         for j in range(len(pred_dis[0])):
    #             res += -pred_dis[i][j] * log(pred_dis[i][j] / real_dis[j], 2)
    #     return res

    def KL_loss(self, pred_dis, real_dis):
        res = torch.tensor(0, requires_grad=True, dtype=torch.float32).to(device)
        for pred in pred_dis:
            # print(pred, real_dis)
            res += self.KL(pred_dis.log(), real_dis)
        return res

    # def KL_loss(self, pred_dis, real_dis):
    #     return self.KL(pred_dis.log(), real_dis+1e-6) / pred_dis.shape[0]

    def forward(self, pred_dis, real_dis):
        return self.KL_loss(pred_dis, real_dis)

def clustering(model, inputs, Y, cluster_loss, recon_loss, optimizer, max_epoch = 10000):
    real_dis = countRealDis(Y)
    for epoch in range(max_epoch):
        latent_X = model.encoder(inputs)
        # latent_X = inputs
        kmeans = KMeans(n_clusters=len(set(Y)), n_init=10)  # 创建一个K-均值聚类对象
        tmp = latent_X.cpu().detach().numpy()
        kmeans.fit(tmp)  # 拟合算法
        y_pred = kmeans.predict(tmp)
        centers = kmeans.cluster_centers_ # 获取聚类中心
        # print(centers)
        pred_dis = countPredDis(tmp, centers)

        loss = cluster_loss(pred_dis, real_dis) + recon_loss(model(inputs), inputs)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, "loss = ",loss.item())
            torch.save(model, f'model//SCbasedmodel_{epoch}.pth')

    return y_pred

if __name__ == '__main__':
    # 数据处理阶段
    inputs, Y, pos = preprocess_data() # targets = Y
    x_dim  = inputs.shape[-1]
    latent_size = 30 # 最终降维到的维度
    hidden_size = int((x_dim + latent_size) * 2 / 3)
    # 预训练AE阶段
    lr = 0.001
    batch_size = 256
    max_epoch = 10000
    PreTrain = False
    encoder = Encoder(x_dim, hidden_size, latent_size)
    decoder = Decoder(x_dim, hidden_size, latent_size)
    model = AutoEncoder(encoder, decoder).to(device)
    AEoptimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    PreTrain = True
    if PreTrain:
        print("PreTraining AE")
        PreTrainAE(model, inputs, batch_size, loss_fn, AEoptimizer, max_epoch)

    # 迭代聚类阶段
    lr = 0.01
    max_epoch = 300
    SCoptimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cluster_loss = KLloss()
    # cluster_loss = nn.CrossEntropyLoss().to(device)
    recon_loss = nn.MSELoss().to(device)
    print("Training to predict Y")
    model = torch.load("model//AEbasedmodel_4600.pth")
    # y_pred = clustering(model, inputs, Y, cluster_loss, recon_loss, SCoptimizer, max_epoch)



    plt.subplot(121)
    # y_pred = sklearn_kmeans(model(inputs).cpu().detach().numpy(), len(set(Y)))
    y_pred = sklearn_kmeans(inputs.cpu().detach().numpy(), len(set(Y)))
    data_process.display(pos, y_pred, show=False)
    plt.xlabel("AE + kmeans")

    plt.subplot(122)
    data_process.display(pos, Y, show=False)
    plt.xlabel("Real Label")

    nmi = metrics.normalized_mutual_info_score(Y, y_pred)
    ari = metrics.adjusted_rand_score(Y, y_pred)

    print(nmi, ari)

    # plt.legend()
    plt.show()
    # WriteDistanceRect(inputs) # *****不要轻易打开注释
    # ReadDistanceRect('distanceRect.txt', inputs.shape[0])
    # with open('distanceRect.txt', 'r') as f:
    #     res = f.readlines()
    # print(res)

