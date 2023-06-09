import numpy as np
from activation import ReLU, Sigmoid


class MLP:
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=1e-2,
                 epochs=20,
                 activate='ReLU',
                 batch_size=20,
                 init_method='uniform'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.zeros((input_size, hidden_size))                   # 输入层到隐藏层的权重
        self.b1 = np.zeros(hidden_size)                                 # 隐藏层的偏置
        self.W2 = np.zeros((hidden_size, output_size))                  # 隐藏层到输出层的权重
        self.b2 = np.zeros(output_size)                                 # 输出层的偏置
        self.hidden = None                                              # 隐藏层的输出
        self.output = None                                              # 模型输出
        self.lr = learning_rate                                         # 学习率
        self.MAX_EPOCHS = epochs                                        # 训练轮数
        self.activate = ReLU() if activate == 'ReLU' else Sigmoid()     # 激活函数
        self.batch_size = batch_size                                    # 分批投入数据
        self._init_parameters(init_method)                              # 权重初始化

    def _init_parameters(self, method='uniform'):
        if method == 'normal':
            # 正态分布初始化
            self.W1 = np.random.randn(self.input_size, self.hidden_size)
            self.W2 = np.random.randn(self.hidden_size, self.output_size)
        elif method == 'uniform':
            # 均匀分布初始化
            self.W1 = np.random.uniform(low=-0.1, high=0.1, size=(self.input_size, self.hidden_size))
            self.W2 = np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_size, self.output_size))
        elif method == 'xavier':
            # xavier初始化
            std = np.sqrt(2.0 / (self.input_size + self.hidden_size))
            self.W1 = np.random.normal(loc=0.0, scale=std, size=(self.input_size, self.hidden_size))
            std = np.sqrt(2.0 / (self.hidden_size + self.output_size))
            self.W2 = np.random.normal(loc=0.0, scale=std, size=(self.hidden_size, self.output_size))

    def forward(self, x):
        # 前向传播过程
        self.hidden = self.activate(np.dot(x, self.W1) + self.b1)   # o1 = activate(w1x+b1)
        self.output = np.dot(self.hidden, self.W2) + self.b2        # o2 = w2o1+b2
        return self.output

    def train(self, x_train, y_train, x_test, y_test):
        num_classes = self.W2.shape[1]                  # 类别总数
        y_train_encoded = np.eye(num_classes)[y_train]  # 独热编码

        epochs, train_accs, test_accs = [], [], []      # 记录输出轮和准确率，用于绘图
        for epoch in range(self.MAX_EPOCHS):
            start = 0                                   # 定义该批次的起点
            epoch_loss = 0                              # 统计批次损失和
            while start < x_train.shape[0]:
                # 取出该批次数据
                batch_x, batch_y = x_train[start:start + self.batch_size], \
                    y_train_encoded[start:start + self.batch_size]
                # 前向传播
                output = self.forward(batch_x)
                # 计算损失函数
                loss = np.mean((output - batch_y) ** 2)

                # 反向传播
                # grad_output = 2 * (output - batch_y) / x_train.shape[0]
                grad_output = output - batch_y  # 取消系数
                grad_W2 = np.dot(self.hidden.T, grad_output)
                grad_b2 = np.sum(grad_output, axis=0)
                grad_hidden = np.dot(grad_output, self.W2.T) * self.activate.differential(self.hidden)
                grad_W1 = np.dot(batch_x.T, grad_hidden)
                grad_b1 = np.sum(grad_hidden, axis=0)

                # 参数更新
                self.W2 -= self.lr * grad_W2
                self.b2 -= self.lr * grad_b2
                self.W1 -= self.lr * grad_W1
                self.b1 -= self.lr * grad_b1

                start += self.batch_size                # 更新起点
                epoch_loss += loss * self.batch_size    # 累加轮次损失和

            if epoch % 1 == 0:
                # 输出观察收敛情况
                train_acc = self.evaluate(x_train, y_train) * 100
                test_acc = self.evaluate(x_test, y_test) * 100
                print(f"Epoch {epoch + 1}: Loss={epoch_loss / x_train.shape[0]:.4f}, "
                      f"train accuracy={train_acc:.2f}% "
                      f"test accuracy={test_acc:.2f}% ")
                epochs.append(epoch)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
        return epochs, train_accs, test_accs

    def predict(self, x):
        # 将10维输出转化为1维标签
        return np.argmax(self.forward(x), axis=1)

    def evaluate(self, x_test, y_test):
        # 评估模型由输入数据得到的输出标签和真实标签的准确率
        predictions = self.predict(x_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy


