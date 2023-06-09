import random
import torchvision
import matplotlib.pyplot as plt


def load_mnist(preprocess=True):
    train_set = torchvision.datasets.MNIST(
        root='MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_set = torchvision.datasets.MNIST(
        root='MNIST',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_x, train_y = train_set.data.numpy(), train_set.targets.numpy()
    test_x, test_y = test_set.data.numpy(), test_set.targets.numpy()    # 转成numpy数据

    if preprocess:  # 对数据进行预处理：将特征变化到一个维度上、对输出数据做归一化、打乱训练集
        train_x, test_x = train_x.reshape(train_x.shape[0], -1), test_x.reshape(test_x.shape[0], -1)    # 合并后两个维度
        train_x, test_x = train_x / 255.0, test_x / 255.0               # 像素值小于255，做归一化
        indexes = list(range(train_x.shape[0]))
        random.shuffle(indexes)
        train_x, train_y = train_x[indexes], train_y[indexes]           # 打乱数据集
    return train_x, train_y, test_x, test_y

def visual_dataset(data, label):
    label = label.tolist()
    plt.bar(list(range(10)), [label.count(i) for i in range(10)])
    plt.title('Samples of each class')
    plt.show()

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_mnist(preprocess=False)
    visual_dataset(train_x, train_y)