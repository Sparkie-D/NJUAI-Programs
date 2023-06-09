import datetime

import matplotlib.pyplot as plt
from MOEAD import MOEADfunction
from POSS import POSSfunction
from NSGAII import NSGAIIfunction
from NEWALG import NEWALGfunction
import sparseRegression
import maxCoverage

if __name__ == '__main__':
    # plt.subplot(131)
    problem = sparseRegression
    # problem = maxCoverage
    max_epoch = 100

    epochs = []
    fitnesses = []

    print("Running POSS function...")
    for epoch in range(1, max_epoch, 10):
        epochs.append(epoch)
        bestfitness = POSSfunction(problem, epoch)
        if len(fitnesses) > 0:
            if bestfitness < fitnesses[-1]:
                fitnesses.append(bestfitness)
            else:
                fitnesses.append(fitnesses[-1])
        else:
            initPop = problem.initPopulation()
            # print(initPop)
            fitnesses.append(initPop[0][1])
        if epoch % 50 == 1:
            print(epoch, bestfitness)
    plt.plot(epochs, fitnesses, label='POSS')
    # plt.show()



    print("Running MOEA/D function...")
    MOEADfunction(problem, max_epoch)
    # plt.show()

    print("Running NSGAIIfunction...")
    NSGAIIfunction(problem, max_epoch)
    # plt.show()

    print("Running improved function...")
    NEWALGfunction(problem, max_epoch)
    # plt.show()

    plt.legend() # 显示图例
    plt.show()