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

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    stack = Stack()
    stack.push(problem.getStartState())

    parents = {}
    visited = set()
    result = []

    while not stack.isEmpty():
        current_state = stack.pop()

        if current_state in visited:
            continue

        if problem.isGoalState(current_state):
            # Backtrack to get the list of actions
            while current_state != problem.getStartState():
                next_state, action = parents[current_state]
                current_state = next_state
                result.append(action)
            return result[::-1]

        visited.add(current_state)

        for next_state, action, _ in problem.getSuccessors(current_state):
            if next_state not in visited:
                parents[next_state] = (current_state, action)
                stack.push(next_state)

    # If no solution is found
    return []
    # util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queue = Queue()
    start_state = problem.getStartState()
    queue.push(start_state)
    visited = {start_state}
    parents = {}

    while not queue.isEmpty():
        current_state = queue.pop()

        if problem.isGoalState(current_state):
            # Backtrack to get the list of actions
            path = []
            while current_state != start_state:
                next_state, action, _ = parents[current_state]
                path.append(action)
                current_state = next_state
            return path[::-1]

        for successor, action, _ in problem.getSuccessors(current_state):
            if successor in visited:
                continue
            visited.add(successor)
            queue.push(successor)
            parents[successor] = (current_state, action, _)

    # If no solution is found
    return []
    # util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    fringe = PriorityQueue()
    distances = {problem.getStartState(): 0}
    parents = {}

    fringe.push(problem.getStartState(), 0)

    while not fringe.isEmpty():
        current_state = fringe.pop()

        if problem.isGoalState(current_state):
            # Backtrack to get the list of actions
            path = []
            while current_state != problem.getStartState():
                next_state, action = parents[current_state]
                current_state = next_state
                path.append(action)
            return path[::-1]

        for next_state, action, cost in problem.getSuccessors(current_state):
            new_cost = distances[current_state] + cost

            if next_state not in distances or new_cost < distances[next_state]:
                distances[next_state] = new_cost
                fringe.update(next_state, new_cost)
                parents[next_state] = (current_state, action)

    # If no solution is found
    return []
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from collections import defaultdict
    fringe = util.PriorityQueue()
    distance = defaultdict(lambda: float('inf'))
    parents = {}

    start_state = problem.getStartState()
    fringe.push(start_state, heuristic(start_state, problem))
    distance[start_state] = 0

    while not fringe.isEmpty():
        current_state = fringe.pop()

        if problem.isGoalState(current_state):
            path = []
            while current_state != start_state:
                next_state, side = parents[current_state]
                path.append(side)
                current_state = next_state
            return path[::-1]

        for next_state, side, cost in problem.getSuccessors(current_state):
            new_distance = distance[current_state] + cost

            if new_distance < distance[next_state]:
                distance[next_state] = new_distance
                fringe.push(next_state, new_distance + heuristic(next_state, problem))
                parents[next_state] = (current_state, side)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
