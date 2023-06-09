# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import random

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

import math
def myHeuristic(state, problem=None):
    """
        you may need code other Heuristic function to replace  NullHeuristic
        """
    "*** YOUR CODE HERE ***"

    def _euclideanDistance(xy1, xy2):
        # print(xy1, xy2)
        return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

    distance = util.manhattanDistance
    # distance = _euclideanDistance
    if type(state[0]) != type((1,1)):
        # print('state is a position?', type(state[0]) == type((1,1)))
        # PositionSearchProblem
        return distance(problem.goal, state)
    else:
        # foodSearchProblem
        print("Using the wrong heuristic ! Please check again!")
        return 0


# def aStarSearch(problem, heuristic=nullHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first.
#
#         Your search algorithm needs to return a list of actions that reaches the
#         goal. Make sure to implement a graph search algorithm.
#
#         To get started, you might want to try some of these simple commands to
#         understand the search problem that is being passed in:
#
#         print("Start:", problem.getStartState())
#         print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
#         print("Start's successors:", problem.getSuccessors(problem.getStartState()))
#         """
#     "*** YOUR CODE HERE ***"
#     # util.raiseNotDefined()
#     # compute path
#     state = problem.getStartState()
#     open_set  = []
#     close_set = []
#     timer = 0
#     if not type(state[0]) == type(1) and len(problem.heuristicInfo.keys()) == 0:
#         problem.heuristicInfo['state'] = [] # for food
#     while(not problem.isGoalState(state)):
#         explored = [child for father, child, _ in close_set] # record explored nodes
#         successors = problem.getSuccessors(state) # (pos, direct, cost)
#         for pos, direct, cost in successors:
#             if(pos not in explored):
#                 open_set.append((heuristic(pos, problem), state, pos, direct)) # (f, father, child, dir)
#         print(state, open_set)
#         # best = min(open_set) # choose min f to explore
#         scores = [open_tuple[0] for open_tuple in open_set]
#         best_score = min(scores)
#         choices = [choice for choice in open_set if choice[0] == best_score]
#         best = random.choice(choices)
#         open_set.remove(best)
#         close_set.append([best[1], best[2], best[3]])  # explore current state
#         state = best[2] # child
#         if not type(state[0]) == type(1):
#             problem.heuristicInfo['state'].append(state[0]) # for food
#         timer += 1
#         if (timer == 10) :
#             break
#     # transform position into action
#     path = []
#     # print('start state', problem.getStartState())
#     while(state != problem.getStartState()):
#         for father, child, direction in close_set:
#             if state == child:
#                 # print(father, ' -> ', child, ', current = ', state)
#                 path.append((state, direction))
#                 state = father
#                 if(state == problem.getStartState()):
#                     break
#     path.reverse()
#     actions = []
#     for pos, direct in path:
#         actions.append(direct)
#     return actions

def aStarSearch(problem, heuristic=nullHeuristic):
    def getMinHeuristicItem(listOfTuple):
        '''input: a list of tuple like :(priority, item)'''
        values = [pair[0] for pair in listOfTuple]
        # print(values)
        best_value = min(values)
        choices = [pair for pair in listOfTuple if pair[0] == best_value]
        return random.choice(choices)

    open_set, close_set = [], []
    # open:  [(f, state, [actions])]
    # close: [state]
    state = problem.getStartState()
    open_set.append((0, state, [])) # record initial state
    timer = 0
    while True:
        current = getMinHeuristicItem(open_set)
        open_set.remove(current)
        if problem.isGoalState(current[1]):
            break
        _, state, actions = current
        close_set.append(state)
        successors = problem.getSuccessors(state)
        for successor in successors:
            suc_state, action, cost = successor
            if suc_state not in close_set:
                # create new nodes
                suc_actions = actions.copy()
                suc_actions.append(action)
                h = heuristic(suc_state, problem)
                if str(heuristic)[:-20] != str(nullHeuristic)[:-20]:
                    # print(str(heuristic)[:-20],str(nullHeuristic)[:-20], str(heuristic)[:-20] == str(nullHeuristic)[:-20])
                    # print('not null')
                    g = problem.getCostOfActions(suc_actions)
                else:
                    g = 0
                open_set.append((h + g, suc_state, suc_actions))
        # print(state, open_set)
        # timer += 1 # for debug
        # if timer == 100000:
        #     break
    return current[2]


# Abbreviations
astar = aStarSearch