import pandas as pd
import numpy as np

def data_load(path):
    data = pd.read_csv(path, header=None)
    # print(data.shape)
    data = data.drop_duplicates() # 去除重复行
    # data.loc[:, (data == 0).all(axis=0)]
    # data = data.loc[:, ~(data == 0).all(axis=0)] # 删掉全0列
    # print(data)
    data = np.array(data)
    # np.set_printoptions(suppress=True)
    # print(data)
    return data[:, :-1], data[:, -1]

def update_omega(x, y):
    # 能计算出结果就返回结果，否则返回全0
    # print("in update omega:", x.shape)
    res = np.array([0 for i in range(x.shape[1])])
    if x.shape[1] > 0:
        x = x.T
        xxt = np.dot(x, x.T)
        # print(x, xxt)
        if np.linalg.matrix_rank(xxt) == xxt.shape[0]:
            xxtr = np.linalg.inv(xxt)
            xy = np.dot(x, y)
            res = np.dot(xxtr, xy)
    return res

def tag(num):
    if num == 0:
        return '0'
    else:
        return '1'

def display(solution):
    # print(len(solution))
    tmp = [tag(i) for i in solution]
    res = ''.join(tmp)
    # print(res)

def sol2attr(x, solution):
    # 把输入的01串转化为属性串
    # display(solution)
    # print(len(solution))
    res = [i for i in range(len(solution)) if solution[i] == 1]
    # print(x, solution)
    # print("here", res)
    return x[:, res]

def list2str(sol):
    # 把输入的01列表转化为字符串
    return ''.join(list(map(str, sol)))

def str2list(solstr):
    arr = list(solstr)
    # print(arr)
    return list(map(int, arr))


def RankCount(Population):
    Rank.clear()
    P = Population.copy()
    k = 0
    while not len(P) == 0:
        Q = []
        for item in P:
            # item = P[idx1]
            # print(len(P))
            isBounded = False
            # if item[0].count(1) > 0:
            #     print("counting rank for", item[1], item[2])
            for other in P:
                if item is not other and (
                        (item[1] >= other[1] and item[2] > other[2]) or item[1] > other[1] and item[2] >= other[2]):
                    isBounded = True
                    break
            if not isBounded:
                # print(item.count(1), 'added into Rank ', k)
                Q.append(item)
        for item in Q:
            P.remove(item)
            if k in Rank.keys():
                Rank[k].append(item)
            else:
                Rank[k] = [item]
        k += 1
    # display_Rank()


def sortbyRank(Population):
    res = []
    # print(Rank.keys())
    for key in Rank.keys():
        cur = Rank[key]
        for item in Population:
            if item in cur:
                res.append(item)
    for idx in range(len(Population)):
        Population[idx] = res[idx]


if __name__ == '__main__':
    # x, y = data_load("Data_ALE/mcs_ds_edited_iter_shuffled.csv")
    x, y = data_load("BlogFeedback/blogData_train.csv")
    print(x.shape)
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    solution = [1, 0, 1]
    print(sol2attr(x, solution))
    print(list2str(solution))
    solution = list2str(solution)
    print(str2list(solution))