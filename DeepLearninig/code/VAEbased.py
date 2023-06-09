import torch
from MLP import one_hot_encode
from data_process import sklearn_PCA, sklearn_kmeans, sklearn_GMM
import data_process
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from sympy import DiracDelta
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.sigma = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, x):
        return self.mu(x), self.sigma(x)

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

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu,sigma = self.encoder(x)
        sample = rand_sampling(mu, sigma)
        output = self.decoder(sample)
        return output, mu, sigma

def rand_sampling(mu, sigma):
    part = torch.exp(sigma / 2)
    sample = torch.randn_like(part)
    return (mu + sample * part).to(device)

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def recon_loss(self, x, x_):
        los = nn.MSELoss() # 最终输入的值为各个样本的loss之和
        # los = nn.MSELoss() # 效果贼差
        # print('recon loss = ',los(x,x_) )
        return los(x, x_)

    def kl_div(self, mu, sigma):
        res = -0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp()) # 这里解释sigma的值为方差取对数，方便计算
        # print("kl div=",res)
        return res #

    def forward(self, x, x_, **otherinputs):
        reco = self.recon_loss(x, x_)
        kldv = self.kl_div(otherinputs['mu'], otherinputs['sigma']) # 把mu和sigma作为参数传进来
        return reco + kldv # 最终的输出为二者之和

def factorial(a):
    res = 1
    while a > 0:
        res *= a
        a -= 1
    return res

def NB(x, r, p):
    t1 = factorial(x+r-1) / (factorial(r-1)*factorial(x))
    t2 = p**x
    t3 = (1-p) **r
    return t1*t2*t3

def ZINB(x, r, p, pi):
    return pi*DiracDelta(x) + (1-pi)*NB(x, r, p)

def train(model, inputs, targets, batch_size, loss_fn, optimizer, EPOCH):
    start = 0
    end = start + batch_size
    for epoch in range(EPOCH):
        while end < inputs.shape[0]:
            Input = inputs[start:end]
            output, mu, sigma = model(Input)
            # print(output.shape)
            VAEloss = loss_fn(output, Input, mu=mu, sigma=sigma)
            # y_pred = sklearn_kmeans(model.encoder(Input)[0].cpu().detach().numpy(), len(set(targets)))
            # clusterloss = cluster_acc(targets[start:end], y_pred)
            loss = VAEloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if end >= inputs.shape[0] - 1:
                break
            else:
                start = end
                end = start + batch_size if start + batch_size < inputs.shape[0] else inputs.shape[0]-1

        if epoch % 200 == 0:
            print("current loss :", loss.item())
            torch.save(model, f'model//VAEbasedmodel_{epoch}.pth')
            print(f'model//VAEbasedmodel_{epoch}.pth saved.')

def cluster_acc(y_true, y_pred):
    Yset = set(y_true)
    no = 0
    Ydict = {}
    for item in Yset:
        Ydict[item] = no
        no += 1
    y_true_val = [Ydict[item] for item in y_true]
    same = 0
    for idx in range(len(y_true_val)):
        if y_true_val[idx] == y_pred[idx]:
            same += 1
    res = same/len(y_true)
    # print(cluster_acc, res)
    return res

def Normalization(X):
    for i in range(len(X[0])):
        line = X[:, i]
        minval = min(line)
        maxval = max(line)
        minval = minval if maxval > minval else minval - 1
        # print(line, maxval, minval)
        for j in range(len(line)):
            line[j] = (line[j] - minval) / (maxval - minval)

def VAEbased():
    lr = 0.0001
    batch_size = 100
    TrainMode = False
    X, Y, pos = data_process.load_data()
    X, Y, pos = np.array(X, dtype=np.float32), np.array(Y), np.array(pos, dtype=np.float32).T
    # Normalization(X)  # 归一化
    # X = PCA.pca(X, 600)
    inputs = np.concatenate((X, pos), axis=1) # 将位置信息和基因表达整合在一起输入AE
    # inputs = X # 输入AE的仅有基因信息
    targets = one_hot_encode(Y)
    inputs, targets = torch.tensor(inputs), torch.tensor(targets)
    inputs = inputs.to(torch.float32).to(device) # 不需要时间戳
    targets = targets.to(torch.float32).to(device)
    x_dim  =inputs.shape[-1]
    hidden_size = int((inputs.shape[-1] + targets.shape[-1]) * 2 / 3)
    latent_size = 100 # 最终降维到的维度
    encoder = Encoder(x_dim, hidden_size, latent_size)
    decoder = Decoder(x_dim, hidden_size, latent_size)
    autoEncoder = VariationalAutoEncoder(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=lr)

    # loss_fn = ZINBLoss()
    loss_fn = Loss()
    loss_fn.to(device)
    TrainMode = True
    if TrainMode:
        train(autoEncoder, inputs, Y, batch_size, loss_fn, optimizer, EPOCH=10000)
    else:
        autoEncoder = torch.load("model//VAEbasedmodel_7600.pth")
    mu, sigma = autoEncoder.encoder(inputs)
    # newInputs = rand_sampling(mu, sigma).cpu()
    newInputs = mu.cpu().detach().numpy()
    # PCA.pca(newInputs, 10)


    # 绘图部分
    plt.subplot(121)
    # newInputs = sklearn_PCA(newInputs, 30)
    cluster_assignment = sklearn_kmeans(newInputs, len(set(Y)))
    data_process.display(pos, cluster_assignment, show=False)
    plt.xlabel("VAE")

    plt.subplot(122)
    #按照标签进行绘图
    data_process.display(pos, Y, show=False)
    plt.xlabel('real labels')
    plt.show()


if __name__ == '__main__':
    # x = [[1, 2, 3], [2, 2, 2], [1, 2, 1]]
    # x = np.array(x,dtype=np.float32)
    # # print(x[:, 1])
    # Normalization(x)
    # print(x)
    VAEbased()
