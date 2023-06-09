import random
import matplotlib.pyplot as plt
import sparseRegression
import maxCoverage

PopulationSize = 10
k = 5
lamda = 3

def fitnessCount(problem, sol):
    # 代替problem的objective 1
    return problem.obj1(sol)

def isLeagal(problem, sol):
    # 代替problem的objective 2
    if sol.count(1) - problem.K <= 0:
        return True
    return False

def TournamentSelection(problem, Population, k, lamda):
    res = []
    while len(res) < lamda:
        # 选lambda个
        chosen = random.sample(Population, k)
        fitnesses = [fitnessCount(problem, item[0]) for item in chosen]
        bestIdx = fitnesses.index(min(fitnesses))
        res.append(chosen[bestIdx])
    return res

def Recombination(parents):
    p1 = parents[0][0]
    p2 = parents[1][0]
    breakPoint = random.randint(0, len(p1))
    c1 = p1[:breakPoint] + p2[breakPoint:]
    c2 = p2[:breakPoint] + p1[breakPoint:]
    return [c1, c2]


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

def SortByFitness(Population):
    fitnesses = [item[1] for item in Population]
    for i in range(len(Population)):
        for j in range(i+1, len(Population)):
            if fitnesses[i] > fitnesses[j]:
                tmp = Population[i]
                Population[i] = Population[j]
                Population[j] = tmp

def NEWALGfunction(problem, max_epoch):
    epochs = []
    fitness = []
    Population = [(item[0], item[1]) for item in problem.initPopulation()]
    if len(Population) < PopulationSize:
        remain = PopulationSize - len(Population)
        for i in range(remain):
            Population.append(Population[-1])
    for epoch in range(max_epoch):
        parentSet = TournamentSelection(problem, Population, k, lamda)
        # 生成子代解，需要至少4个孩子
        children = []
        while len(children) < 4:
            parents = random.sample(parentSet, 2) # 每次从父代候选集中抽取2个作为此时的父代解
            tmpChildren = Recombination(parents)
            Mutation(tmpChildren)
            for item in tmpChildren:
                if isLeagal(problem, item):
                    children.append(item)
            # print(len(children))
        # print('children:', children[0].count(1))
        # fitness eval
        for idx in range(len(children)):
            sol = children[idx]
            ob1 = fitnessCount(problem, sol)
            children[idx] = (sol, ob1)

        # survivor selection
        for item in children:
            Population.append(item)
        SortByFitness(Population)
        Population = Population[:PopulationSize]
        # print(len(Population))

        if epoch % 10 == 0:
            epochs.append(epoch)
            fitnesses = [item[1] for item in Population]
            idx = fitnesses.index(min(fitnesses))
            fitness.append(min(fitnesses))
            print(epoch, min(fitnesses), Population[idx][0].count(1))
            # print(len(Population))

    plt.plot(epochs, fitness, label='My Algorithm')
    # plt.show()


if __name__ == '__main__':
    problem = sparseRegression
    # problem = maxCoverage
    max_epoch = 300
    NEWALGfunction(problem, max_epoch)
    plt.show()