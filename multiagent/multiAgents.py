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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        score = 0
        i = 0
        pending_food = newFood.asList()
        ghost_positions = successorGameState.getGhostPositions()

        food_distance = float('-inf')

        if len(pending_food):
            for food in pending_food:
                local_distance = util.manhattanDistance(newPos, food)
                if food_distance < local_distance:
                    food_distance = local_distance
        else:
            food_distance = 0

        score = successorGameState.getScore() - food_distance

        ghost_distance = float('-inf')

        if len(ghost_positions):
            i = 0
            for ghostpos in ghost_positions:
                local_distance = util.manhattanDistance(newPos, ghostpos)
                if ghost_distance < local_distance:
                    ghost_distance = local_distance
                    min_ghost_index = i
                i+=1

            if ghost_distance <= 1 and newScaredTimes[min_ghost_index] == 0:
                return float('-inf')

            if ghost_distance <= 1 and newScaredTimes[min_ghost_index] > 0:
                return float('inf')

        else:
            ghost_distance = 0

        if newScaredTimes[min_ghost_index] > 0:
            score -= ghost_distance
        else:
            score += ghost_distance
        
        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agent):
            if state.isWin() or state.isLose():
                return state.getScore()
            if agent == 0:
                pacman_actions = state.getLegalActions(0)
                best_action = Directions.STOP
                max_score = float("-inf")

                for action in pacman_actions:
                    score = minimax(state.generateSuccessor(0, action), depth, 1)
                    if max_score < score:
                        max_score = score
                        best_action = action

                if depth == 0:
                    return best_action
                return max_score
            else:
                next_agent = agent + 1
                if next_agent == state.getNumAgents():
                    next_agent = 0
                
                min_score = float("+inf")
                agent_actions = state.getLegalActions(agent)

                for action in agent_actions:
                    if next_agent == 0:
                        if depth == self.depth-1:
                            score = self.evaluationFunction(state.generateSuccessor(agent, action))
                        else:
                            score = minimax(state.generateSuccessor(agent, action), depth+1, 0)
                    else:
                        score = minimax(state.generateSuccessor(agent, action), depth, next_agent)

                    min_score = min(min_score, score)

                return min_score

        return minimax(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(state, depth, agent, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            if agent == 0:
                pacman_actions = state.getLegalActions(0)
                best_action = Directions.STOP
                max_score = float("-inf")

                for action in pacman_actions:
                    score = alphabeta(state.generateSuccessor(0, action), depth, 1, alpha, beta)
                    if max_score < score:
                        max_score = score
                        best_action = action

                    alpha = max(max_score, alpha)

                    if alpha > beta:
                        break

                if depth == 0:
                    return best_action
                return max_score
            else:
                next_agent = agent + 1
                if next_agent == state.getNumAgents():
                    next_agent = 0
                
                min_score = float("+inf")
                agent_actions = state.getLegalActions(agent)

                for action in agent_actions:
                    if next_agent == 0:
                        if depth == self.depth-1:
                            score = self.evaluationFunction(state.generateSuccessor(agent, action))
                        else:
                            score = alphabeta(state.generateSuccessor(agent, action), depth+1, 0, alpha, beta)
                    else:
                        score = alphabeta(state.generateSuccessor(agent, action), depth, next_agent, alpha, beta)

                    min_score = min(min_score, score)

                    beta = min(min_score, beta)

                    if alpha > beta:
                        break

                return min_score

        return alphabeta(gameState, 0, 0, float("-inf"), float("+inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agent):
            if state.isWin() or state.isLose():
                return state.getScore()
            if agent == 0:
                pacman_actions = state.getLegalActions(0)
                best_action = Directions.STOP
                max_score = float("-inf")

                for action in pacman_actions:
                    score = expectimax(state.generateSuccessor(0, action), depth, 1)
                    if max_score < score:
                        max_score = score
                        best_action = action

                if depth == 0:
                    return best_action
                return max_score
            else:
                next_agent = agent + 1
                if next_agent == state.getNumAgents():
                    next_agent = 0

                agent_actions = state.getLegalActions(agent)

                score = 0

                for action in agent_actions:
                    if next_agent == 0:
                        if depth == self.depth-1:
                            score += self.evaluationFunction(state.generateSuccessor(agent, action))
                        else:
                            score += expectimax(state.generateSuccessor(agent, action), depth+1, 0)
                    else:
                        score += expectimax(state.generateSuccessor(agent, action), depth, next_agent)

                return score/len(agent_actions)

        return expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return float('-inf')
    if currentGameState.isWin():
        return float('inf')
    
    score = currentGameState.getScore()
    foodList = currentGameState.getFood().asList()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Score based on remaining food
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodList]
    if len(foodDistances) > 0:
        minFoodDistance = min(foodDistances)
        score += 1.0 / minFoodDistance

    # Score based on nearest ghost
    ghostDistances = [manhattanDistance(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates]
    minGhostDistance = min(ghostDistances)
    if minGhostDistance > 0:
        if minGhostDistance <= 1:
            # Reward if scared ghost
            if scaredTimes[ghostDistances.index(minGhostDistance)] > 0:
                return float('inf')
            else:
                score -= 1.0 / minGhostDistance
        else:
            # Penalize if not
            maxScaredTime = max(scaredTimes)
            if maxScaredTime > 0:
                score += 1.0 / minGhostDistance
            else:
                score -= 1.0 / minGhostDistance

    return score

# Abbreviation
better = betterEvaluationFunction