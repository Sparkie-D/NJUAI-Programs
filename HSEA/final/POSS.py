import random
import utils
from NSGAII import RankCount, sortbyRank
import numpy as np
import matplotlib.pyplot as plt
# 对应于要解决的问题
import sparseRegression # 稀疏回归任务
import maxCoverage   # 最大覆盖任务

# 超参数
# PopulationUpperBound = 40
# MAX_EPOCH = 1000
# 稀疏回归的K值需要在sparseRegression文件中更改


def flip(solution):
    child = solution.copy()
    # print("child")
    prob = 1/len(child)
    # print(prob)
    for idx in range(len(child)):
        randval = random.random()
        # print(randval)
        if randval < prob:
            # print("flip")
            child[idx] = (child[idx] + 1) % 2
    # utils.display(child)
    # print(child.count(1))
    return child

Rank = {}

def POSSfunction(problem=sparseRegression, MAX_EPOCH = 1000):
    t = 0
    epoch = []
    bestfitness = []
    Population = problem.initPopulation() # 初始化的种群
    while(t < MAX_EPOCH):
        s = random.choice(Population)
        child = flip(s[0])
        # print(child == s)
        bad = []
        ob1 = problem.obj1(child)
        ob2 = problem.obj2(child)
        # print(child.count(1), ob1, ob2)
        notGood = False
        for item in Population:
            if item[1] >= ob1 and item[2] >= ob2:
                bad.append(item)
            elif (item[1] < ob1 and item[2] <= ob2) or (item[1] <= ob1 and item[2] < ob2):
                notGood = True
        if not notGood:
            # print("Population append child score = ", ob1, ob2)
            Population.append((child, ob1, ob2))
            if len(bad) > 0:
                # print("Population remove ", len(bad), ' itmes')
                for item in bad:
                    Population.remove(item)
        t += 1
        # 限制种群大小
        # if len(Population) > PopulationUpperBound:
        #     RankCount(Population)
        #     sortbyRank(Population)
        #     Population = Population[:PopulationUpperBound]
    # 综合，输出
    fitnesses = []
    for item in Population:
        tmp = item[0].count(1) - problem.K
        if tmp <= 0:
            fitnesses.append((item[1], tmp))
    if len(fitnesses) <= 0:
        # 没有合格项，随意返回
        minval = min([item[1] for item in Population])
    else:
        sorted(fitnesses, key=lambda t: t[0]) # 在所有合格项中挑选最优
        minval = fitnesses[0][0]
    # bestfitness.append(minval)
    return minval


    
if __name__ == '__main__':
    problem = sparseRegression
    # problem = maxCoverage
    epochs = []
    fitnesses = []
    for epoch in range(0, 2000, 100):
        epochs.append(epoch)
        bestfitness = POSSfunction(problem, epoch)
        fitnesses.append(bestfitness)
        print(bestfitness)
    plt.plot(epochs, fitnesses)
    plt.show()



