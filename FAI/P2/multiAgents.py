# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Evaluate ghost distances
        ghost_distances = [manhattanDistance(newPos, ghost.configuration.pos)
                        for ghost in newGhostStates
                        if ghost.scaredTimer == 0]

        # Calculate the minimum distance to a non-scared ghost
        min_ghost_dist = min(ghost_distances, default=float('inf'))

        # If Pacman is in immediate danger, return negative infinity
        if min_ghost_dist == 0:
            return float('-inf')

        # Evaluate remaining food
        num_food = successorGameState.getNumFood()

        # If there is no remaining food, return positive infinity
        if num_food == 0:
            return float('inf')

        current_food = currentGameState.getFood()

        # If Pacman is on food, set minimum food distance to 0
        min_food_dist = 0 if current_food[newPos[0]][newPos[1]] else float('inf')

        # Calculate the minimum distance to remaining food
        food_distances = [
            manhattanDistance(newPos, (x, y))
            for x, y in current_food.asList()
        ]

        # If there is remaining food, calculate the minimum distance
        min_food_dist = min(food_distances, default=0)

        # Calculate danmage and points
        danmage = 1 / (min_ghost_dist - 0.00001)
        points = 1 / (min_food_dist + 0.00001)

        # Combine danger and profit to get the final score
        score = points - danmage

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions()
        successors = (gameState.generateSuccessor(0, action) for action in legalActions)
        scores = [self.minmax(successor, 1, self.depth) for successor in successors]
        i = scores.index(max(scores))
        return legalActions[i]

    def minmax(self, state, agentIndex: int, depth: int):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # pacman play
            selectBestScore = max
            nextAgent = 1
            nextDepth = depth
        else:  # ghost play
            selectBestScore = min
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = (depth - 1) if nextAgent == 0 else depth

        legalActions = state.getLegalActions(agentIndex)
        successors = (state.generateSuccessor(agentIndex, action) for action in legalActions)
        scores = [self.minmax(successor, nextAgent, nextDepth)
                  for successor in successors]
        return selectBestScore(scores)
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxi(gameState, depth, alpha, beta):
            """
            Max function for Pacman.
            """
            if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                return self.evaluationFunction(gameState)

            value = float('-inf')

            for action in gameState.getLegalActions(0):
                value = max(value, mini(gameState.generateSuccessor(0, action), depth + 1, 1, alpha, beta))

                if value > beta:
                    return value

                alpha = max(alpha, value)

            return value

        def mini(gameState, depth, agentIndex, alpha, beta):
            """
            Min function for Ghosts.
            """
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == gameState.getNumAgents() - 1:
                    value = min(value, maxi(successor, depth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                else:
                    value = min(value, mini(successor, depth, agentIndex + 1, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)

            return value


        # Alpha-Beta Pruning
        alpha = float('-inf')
        beta = float('inf')
        currentScore = float('-inf')
        returnAction = ''
        actions = gameState.getLegalActions(0)
        
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = mini(nextState, 0, 1, alpha, beta)

            if score > currentScore:
                returnAction = action
                currentScore = score
   
            if score > beta:
                return returnAction
            
            alpha = max(alpha, score)
        
        return returnAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxi(gameState, depth):
            currentDepth = depth + 1
            
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            
            value = float('-inf')
            actions = gameState.getLegalActions(0)
            
            for action in actions:
                successor= gameState.generateSuccessor(0, action)
                value = max (value, expectedLevel(successor, currentDepth, 1))
            
            return value
        
        
        def expectedLevel(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            actions = gameState.getLegalActions(agentIndex)
            totalValue = 0
            totalActions = len(actions)
            
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex, action)
                
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxi(successor, depth)
                else:
                    expectedvalue = expectedLevel(successor, depth, agentIndex + 1)
                
                totalValue = totalValue + expectedvalue
            
            if totalActions == 0:
                return  0
            
            return float(totalValue) / float(totalActions)
        
        actions = gameState.getLegalActions(0)
        currentScore = float('-inf')
        returnAction = ''
        
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = expectedLevel(nextState, 0, 1)
            
            if score > currentScore:
                returnAction = action
                currentScore = score
        
        return returnAction
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Evaluates the current game state to generate an optimal score for Pacman.

    - This function calculates a score considering Pacman's position,
    remaining food, ghost states, and distances. 
    - Generates score based on current game score, reciprocal distances to food, eaten food count, power pellets, and
    remaining scared time of ghosts. 
    - (should ideally) help prioritize food, avoid non-vulnerable ghosts, and target vulnerable ghosts. 
    """
    "*** YOUR CODE HERE ***"
    
    pacman_position = currentGameState.getPacmanPosition()
    food_positions = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()

    food_distances = [1.0 / manhattanDistance(pacman_position, food) for food in food_positions]
    ghost_distances = [manhattanDistance(pacman_position, ghost.getPosition()) for ghost in ghost_states]

    total_power_pellets = len(currentGameState.getCapsules())
    eaten_food = len(food_positions)
    total_scared_time = sum(ghost.scaredTimer for ghost in ghost_states)
    total_ghost_distances = sum(ghost_distances)

    score = currentGameState.getScore() + sum(food_distances) + eaten_food

    if total_scared_time > 0:
        score += total_scared_time + (-1 * total_power_pellets) + (-1 * total_ghost_distances)
    else:
        score += total_ghost_distances + total_power_pellets

    return score

# Abbreviation
better = betterEvaluationFunction
