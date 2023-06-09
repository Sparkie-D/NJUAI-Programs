import matplotlib.pyplot as plt
from utils import load_mnist
from network import MLP

# def analysis():
#     # 分析各种参数对模型预测的影响
#     x_train, y_train, x_test, y_test = load_mnist()
#
#     # 创建MLP模型
#     # hidden_sizes = [20, 50, 100, 200, 500, 1000, 2000]
#     init_methods = ['normal', 'uniform', 'xavier']
#     for method in init_methods:
#         model = MLP(input_size=784,             # 输入神经元数
#                     hidden_size=100,    # 隐藏层神经元数
#                     output_size=10,             # 输出神经元数
#                     learning_rate=1e-3,         # 学习率
#                     epochs=20,                  # 最大迭代轮数
#                     activate='ReLU',            # 激活函数
#                     batch_size=20,              # 批次大小
#                     init_method=method)       # 权重初始化方法，可选normal、uniform、xavier
#
#         # 训练模型
#         epochs, train_accs, test_accs = model.train(x_train, y_train, x_test, y_test)
#         # 绘图
#         plt.subplot(121)        # 在第一个子图中绘制迭代轮数-训练集准确率图
#         plt.plot(epochs, train_accs, label=method)
#         plt.title('Accuracy on Train Set')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         # plt.ylim((70, 100))
#         plt.legend()
#
#         plt.subplot(122)        # 在第二个子图中绘制迭代轮数-测试集准确率图
#         plt.plot(epochs, test_accs, label=method)
#         plt.title('Accuracy on Test Set')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         # plt.ylim((70, 100))
#         plt.legend()
#     plt.show()


def analysis():
    # 分析各种参数对模型预测的影响
    x_train, y_train, x_test, y_test = load_mnist()

    # 创建MLP模型
    # hidden_sizes = [20, 50, 100, 200, 500, 1000, 2000]
    hidden_sizes = [3000]
    for hidden_size in hidden_sizes:
        model = MLP(input_size=784,             # 输入神经元数
                    hidden_size=hidden_size,    # 隐藏层神经元数
                    output_size=10,             # 输出神经元数
                    learning_rate=1e-2,         # 学习率
                    epochs=200,                  # 最大迭代轮数
                    activate='ReLU',            # 激活函数
                    batch_size=20,              # 批次大小
                    init_method='xavier')       # 权重初始化方法，可选normal、uniform、xavier

        # 训练模型
        epochs, train_accs, test_accs = model.train(x_train, y_train, x_test, y_test)
        # 绘图
        plt.subplot(121)        # 在第一个子图中绘制迭代轮数-训练集准确率图
        plt.plot(epochs, train_accs, label=f'hidden={hidden_size}')
        plt.title('Accuracy on Train Set')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        # plt.ylim((70, 100))
        plt.legend()

        plt.subplot(122)        # 在第二个子图中绘制迭代轮数-测试集准确率图
        plt.plot(epochs, test_accs, label=f'hidden={hidden_size}')
        plt.title('Accuracy on Test Set')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        # plt.ylim((70, 100))
        plt.legend()
    plt.show()


if __name__ == '__main__':
    # analysis()      # 分析因素对精度的影响
    # 载入MNIST数据集
    x_train, y_train, x_test, y_test = load_mnist()

    # 创建MLP模型
    activations = ['ReLU', 'Sigmoid']           # 所有激活函数
    lines = {'ReLU': 'c*-', 'Sigmoid': 'm.-'}   # 用于绘制不同形式的曲线
    for activate in activations:
        print(f"Training MLP using activation [{activate}]")
        lr = 1e-2 if activate == 'ReLU' else 1e-3
        model = MLP(input_size=784,             # 输入神经元数
                    hidden_size=100,            # 隐藏层神经元数
                    output_size=10,             # 输出神经元数
                    learning_rate=lr,           # 学习率
                    epochs=20,                  # 最大迭代轮数
                    activate=activate,          # 激活函数
                    batch_size=20,              # 批次大小
                    init_method='xavier')       # 权重初始化方法，可选normal、uniform、xavier

        # 训练模型
        epochs, train_accs, test_accs = model.train(x_train, y_train, x_test, y_test)
        # 绘图
        plt.subplot(121)        # 在第一个子图中绘制迭代轮数-训练集准确率图
        plt.plot(epochs, train_accs, lines[activate], label=activate)
        plt.title('Accuracy on Train Set')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim((70, 100))
        plt.legend()

        plt.subplot(122)        # 在第二个子图中绘制迭代轮数-测试集准确率图
        plt.plot(epochs, test_accs, lines[activate], label=activate)
        plt.title('Accuracy on Test Set')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim((70, 100))
        plt.legend()
    plt.show()
