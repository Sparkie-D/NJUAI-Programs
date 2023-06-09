import random
import matplotlib.pyplot as plt
import sparseRegression # 稀疏回归任务
import maxCoverage  # 最大覆盖任务

PopulationSize = 10
MAX_EPOCH = 100

def ProblemWrapper(problem, idx):
    def goal(sol):
        obj1 = problem.obj1
        obj2 = problem.obj2
        lambda1 = (idx)/(PopulationSize-1)
        lambda2 = 1 - lambda1
        return lambda1 * obj1(sol) + lambda2 *obj2(sol)
    return goal

def Mutation(children):
    # print(children)
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
    p1 = parents[0]
    p2 = parents[1]
    breakPoint = random.randint(0, len(p1))
    c1 = p1[:breakPoint] + p2[breakPoint:]
    c2 = p2[:breakPoint] + p1[breakPoint:]
    return [c1, c2]


def MOEADfunction(problem, max_epoch):
    max_epoch = int(max_epoch / PopulationSize)
    x = [0]
    y = []
    subproblems = [ProblemWrapper(problem, i) for i in range(PopulationSize)] #子问题优化目标，元素为函数
    Population = [item[0] for item in problem.initPopulation()]
    y.append(problem.initPopulation()[0][1])
    if len(Population) < PopulationSize:
        remain = PopulationSize - len(Population)
        for i in range(remain):
            Population.append(Population[-1])
    Fitnesses = [subproblems[i](Population[i]) for i in range(PopulationSize)]
    for epoch in range(max_epoch):
        # if epoch % 10 == 0:
        print(epoch, [item.count(1) for item in Population], Fitnesses)
        for idx in range(len(Population)):
            left = (idx + len(Population) - 1) % len(Population)
            right = (idx + len(Population) + 1) % len(Population) # 左右邻居的下标
            children1 = Recombination([Population[left], Population[idx]])
            children2 = Recombination([Population[idx], Population[right]])
            children = children1 + children2
            Mutation(children)
            # print([item.count(1) for item in children])
            # 从子代4个解中筛选出最好的
            bestchild = children[0]
            bestfitness = Fitnesses[idx] # 此时为父代解的fitness
            for child in children:
                childfitness = [subproblems[idx](child) for child in children]
                bestfitness = min(childfitness)
                bestchild = children[childfitness.index(bestfitness)] # 得分最好的
            for j in [left, idx, right]:
                newfitness = subproblems[j](bestchild) # 当前子代解针对邻居问题的fitness
                # print(newfitness, Fitnesses[j])
                if Fitnesses[j] > newfitness and bestchild.count(1) <= problem.K:
                    # print("replace ", j, Fitnesses[j], "->", newfitness)
                    Fitnesses[j] = newfitness
                    Population[j] = bestchild
        # FitnessGoal = [subproblems[-1](item) for item in Population if item.count(1) - problem.K <= 0]
        FitnessGoal = [subproblems[-1](item) for item in Population]
        x.append(epoch)
        if len(FitnessGoal) > 0:
            y.append(min(FitnessGoal))
        else:
            y.append(y[-1])

    plt.plot([item* PopulationSize for item in x], y, label="MOEA/D")
    # plt.show()


if __name__ == '__main__':
    # problem = sparseRegression
    problem = maxCoverage
    MOEADfunction(problem, MAX_EPOCH)
    plt.show()