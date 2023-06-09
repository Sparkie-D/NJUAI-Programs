import utils
import numpy as np
path = "BlogFeedback/blogData_train.csv"
ALLset, Y = utils.data_load(path)
K = 60

def obj1(solution):
    # return value of objective 1
    xnew = utils.sol2attr(ALLset, solution)
    omega = utils.update_omega(xnew, Y)
    # print(xnew.shape, omega.shape)
    Xw = np.dot(xnew, omega)
    return sum((Y-Xw)**2)/Y.shape[0]

def obj2(solution):
    # return value of objective 2
    return abs(solution.count(1) - K)

def initPopulation():
    s = [0 for i in range(ALLset.shape[1])]
    return [(s, obj1(s), obj2(s))]
