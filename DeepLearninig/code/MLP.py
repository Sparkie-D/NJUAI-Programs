import torch
import data_process
# import PCA
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
from data_process import sklearn_PCA
from myDeepCluster import Normalization
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class myMLP(nn.Module):
    def __init__(self, input_dim, hidden_num, output_dim):
        super(myMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, output_dim),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

def MLPtrain(model, loss_fn, optimizer,trainSet, testSet, epoch, batch_size):
    '''
    @param inputs: 输入数据
    @param epoch:总共训练轮数
    '''
    # 训练阶段
    inputs, targets = trainSet
    print("input shape:", inputs.shape)
    for i in range(epoch+1):
        # 开始训练
        start = 0
        end = start + batch_size
        # print(start, end, bias)
        while end < inputs.shape[0]:
            # 获取输出
            outputs = model(inputs[start:end]) # 每次一个batch的数据
            loss = loss_fn(outputs, targets[start:end])

            # 优化过程
            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 反向传播
            optimizer.step()        # 权重更新

            if end >= inputs.shape[0] - 1:
                break
            else:
                start = end
                end = start + batch_size if start + batch_size < inputs.shape[0] else inputs.shape[0]-1

            # print('loss', loss)

        if i % 10 == 0 :
            # 观察训练效果
            print('epoch ', i)
            with torch.no_grad():
                print('train loss:', loss.item())
                testInput, testTag = testSet
                # test_loss = loss_fn(model(testInput), testTag).item()
                # print("test loss:", test_loss)
                # test(model, trainSet, testSet, save_path, device)
                torch.save(model, f'model//MLPmodel_{i}.pth')

    print("MLP train finished.")

def one_hot_encode(Y):
    Yset = list(set(Y))
    Ydict = {}
    idx = 0
    for item in Yset:
        Ydict[item] = [0 for i in range(len(Yset))]
        Ydict[item][idx] = 1
        idx += 1
    res = []
    for y in Y:
        res.append(Ydict[y])
    return np.array(res)

def decode(one_hotY, Y):
    Yset = list(set(Y))
    one_hotYlist = one_hotY.tolist()
    res = []
    for lis in one_hotYlist:
        pos = lis.index(max(lis))
        # print(pos)
        res.append(Yset[pos])
    return res

if __name__ == '__main__':
    train = False
    # 超参数
    lr = 0.001
    epoch = 1000
    batch_size = 256
    # 数据预处理
    X, Y, pos = data_process.load_data()
    X, Y, pos = np.array(X, dtype=np.float32), np.array(Y), np.array(pos, dtype=np.float32).T
    # X = PCA.pca(X, 300)
    # print(X)
    X = sklearn_PCA(X, 300)
    # inputs = np.concatenate((X, pos), axis=1)
    inputs = X #不要位置信息
    Normalization(inputs)
    targets = one_hot_encode(Y)
    inputs, targets = torch.tensor(inputs), torch.tensor(targets)
    inputs = inputs.to(torch.float32).to(device) # 不需要时间戳
    targets = targets.to(torch.float32).to(device)

    # 训练集和测试集
    # trainLenth = int(inputs.shape[0] * 9/10)
    trainLenth = inputs.shape[0] # 不设置测试集
    trainInput = inputs[:trainLenth]
    trainTag =  targets[:trainLenth]
    testInput = inputs[trainLenth:]
    testTag = targets[trainLenth:]

    # 模型预设
    input_size = inputs.shape[1]
    output_size = targets.shape[1]
    # hidden_size = int(2 * (input_size + output_size) / 3)
    hidden_size = input_size + output_size + 10
    print(input_size, hidden_size, output_size)
    model = myMLP(input_size, hidden_size, output_size)
    model.to(device)

    # 选择损失函数
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss(reduction='sum')
    loss_fn.to(device)

    # 选择优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    train = True  # 注释此行则仅输出
    if train:
        MLPtrain(model, loss_fn, optimizer, (trainInput, trainTag), (testInput, testTag), epoch, batch_size)
        train = False

    # 效果展示
    print(str(model))
    if not train:
        model = torch.load("model//MLPmodel_1000.pth")
    res_one_hot_Y = model(inputs)
    y_pred = decode(res_one_hot_Y, Y)
    plt.subplot(121)
    data_process.display(pos, y_pred, show=False)
    plt.xlabel("MLP")
    plt.subplot(122)
    data_process.display(pos, Y, show=True)
    plt.xlabel("real label")
    nmi = metrics.normalized_mutual_info_score(Y, y_pred)
    ari = metrics.adjusted_rand_score(Y, y_pred)
    print(nmi, ari)