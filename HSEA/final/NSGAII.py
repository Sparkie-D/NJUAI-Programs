import random
import utils
import numpy as np
import matplotlib.pyplot as plt
# 对应于要解决的问题
import sparseRegression # 稀疏回归任务，注释掉最大覆盖一行的引用
# import maxCoverage  # 最大覆盖任务，注释掉稀疏回归一行的引用

# 超参数
PopulationSize = 20
MAX_EPOCH = 100
k =  20# 父代选择时的子种群大小
lamda =  2# 每一轮中得到的最终父代个数，binary selection得到最终两个解
# 稀疏回归的K值需要在sparseRegression文件中更改

Rank = {}
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
                if item is not other and ((item[1] >= other[1] and item[2] > other[2]) or item[1] > other[1] and item[2] >= other[2]):
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

def display_Rank():
    print("displaying Rank: ")
    for key in Rank.keys():
        print(key, " : ", end=' ')
        for item in Rank[key]:
            print(item[0].count(1),"(",item[1], item[2], ")", end=' ')
        print()
    print('end of Rank.')

def display_Population(Population):
    print("Population: ", end=' ')
    for item in Population:
        print(item[0].count(1), end=' ')
    print()

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

def BinaryTournamentSelection(Population, k, lamda):
    # 调用前确保已经维护了Population的rank字典
    res = []
    for i in range(lamda):
        # 选lambda个
        chosen = random.sample(Population, k)
        bestIdx = 0
        bestRank = len(Population) # rank越小越好，不可能大于这个值
        for idx in range(len(chosen)):
            item = chosen[idx]
            currentRank = len(Population)
            for key in Rank.keys():
                if item in Rank[key]:
                    currentRank = key
                    break
            if currentRank > bestRank:
                bestIdx = idx
                bestRank = currentRank
        res.append(chosen[bestIdx])
    return res

def Mutation(children):
    prob = 1/len(children[0]) # 翻转概率1/n
    for i in range(len(children)):
        child = children[i]
        for idx in range(len(child)):
            randval = random.random()
            # print(randval)
            if randval < prob:
                # print("flip")
                child[idx] = (child[idx] + 1) % 2


def Recombination(parents):
    p1 = parents[0][0]
    p2 = parents[1][0]
    breakPoint = random.randint(0, len(p1))
    c1 = p1[:breakPoint] + p2[breakPoint:]
    c2 = p2[:breakPoint] + p1[breakPoint:]
    return [c1, c2]

def NSGAIIfunction(problem, max_epoch):
    epochs = []
    fitness = []
    Population = problem.initPopulation()
    epochs.append(0)
    fitness.append(Population[0][1])
    if len(Population) < PopulationSize:
        remain = PopulationSize - len(Population)
        for i in range(remain):
            Population.append(Population[-1])
    for epoch in range(max_epoch):
        RankCount(Population)
        parents = BinaryTournamentSelection(Population, k, lamda) # 含有两个解
        # 生成子代解
        children = Recombination(parents)
        Mutation(children)
        # print('children:', children[0].count(1))
        # fitness eval
        for idx in range(len(children)):
            sol = children[idx]
            ob1 = problem.obj1(sol)
            ob2 = problem.obj2(sol)
            children[idx] = (sol, ob1, ob2)
            # print(children[idx][1], children[idx][2])
        # survivor selection
        for item in children:
            Population.append(item)
        RankCount(Population)
        sortbyRank(Population)
        # display_Population(Population)
        Population = Population[:PopulationSize]
        # print(Population[0][1], Population[0][2])
        # print(Population)
        if epoch  % 10 == 0:
            print([(item[1], item[2]) for item in Rank[0]])
            epochs.append(epoch)
            legals = [item[1] for item in Rank[0] if item[0].count(1) <= problem.K]
            if len(legals) > 0:
                fitness.append(min(legals))
            else:
                fitness.append(random.choice(Rank[0])[1])
            # print(len(Population))

    plt.plot(epochs, fitness, label='NSGAII')
    # plt.show()



if __name__ == '__main__':
    problem = sparseRegression
    # problem = maxCoverage
    NSGAIIfunction(problem, MAX_EPOCH)
    plt.show()