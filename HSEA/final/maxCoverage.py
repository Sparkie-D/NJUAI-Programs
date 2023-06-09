import random
maxNodeSize = 2000  # 最多顶点数
VerticeSetNum = 100  # 点集的数目
K = 20             # 最多选择的集合数
zeroProbability  = 0.95 # 每个点集中0的比例

# VerticeSets = [[random.randint(0, 2) for i in range(maxNodeSize)] for j in range(VerticeSetNum)]

# 创建点集的集合
VerticeSets = []
WeightSet = [random.random() for i in range(maxNodeSize)]
for i in range(VerticeSetNum):
    currentVerticeSet = []
    for j in range(maxNodeSize):
        if random.random() < zeroProbability:
            currentVerticeSet.append(0)
        else:
            currentVerticeSet.append(1)
    VerticeSets.append(currentVerticeSet)

def obj1(solution):
    # 无权图的obj1
    res = [0 for i in range(maxNodeSize)]
    for idx in range(len(solution)):
        if solution[idx] == 1:
            for jdx in range(len(VerticeSets[idx])):
                res[jdx] = 1 if res[jdx] + VerticeSets[idx][jdx] >= 1 else 0
    return maxNodeSize - res.count(1)

# def obj1(solution):
#     # 有权图的obj1
#     res = [0 for i in range(maxNodeSize)]
#     for idx in range(len(solution)):
#         if solution[idx] == 1:
#             for jdx in range(len(VerticeSets[idx])):
#                 res[jdx] = 1 if res[jdx] + VerticeSets[idx][jdx] >= 1 else 0
#     resval = 0
#     for idx in range(len(res)):
#         resval += res[idx] * WeightSet[idx]
#     return -resval

def obj2(solution):
    # return value of objective 2
    return abs(solution.count(1) - K)

def initPopulation():
    s = [0 for i in range(VerticeSetNum)]
    return [(s, obj1(s), obj2(s))]