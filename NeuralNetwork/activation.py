import numpy as np


class Sigmoid:
    # def __call__(self, x):
    #     # 出现指数溢出错误，因为-x可能很大
    #     return 1/(1+np.exp(-x))
    def __call__(self, x):
        # 分别处理正值和负值，保证exp(.) < 1
        indices_pos = np.nonzero(x >= 0)
        indices_neg = np.nonzero(x < 0)

        y = np.zeros_like(x)
        y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))                      # 对正值计算1 / (1+exp(-x)
        y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))  # 对负值计算exp(x) / (1+exp(x))

        return y

    def differential(self, x):
        # Sigmoid的导数为sgimoid'(x) = sigmoid(x) * (1-sigmoid(x))
        return self(x) * (1-self(x))


class ReLU:
    def __call__(self, x):
        return np.maximum(x, 0)

    def differential(self, x):
        # ReLU的导数：大于0的为1，小于等于0的为0
        pos_x = x + np.abs(x)       # 获得非负矩阵，其中正值翻倍，负值变0
        pos_x[pos_x > 0] = 1        # 正值部分全为1
        return pos_x


if __name__ == '__main__':
    sigmoid = Sigmoid()
    x = np.array([[1,-2,3], [1,1,-1]])
    print(sigmoid(x))
    print(sigmoid.differential(x))
    relu = ReLU()
    print(relu(x))
    print(relu.differential(x))