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
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore(), None
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state), None

      value_action = []
      for action in state.getLegalActions(agentIndex):
          next_agent = nextAgent(state, agentIndex)
          succ = state.generateSuccessor(agentIndex, action)
          if next_agent != 0: # next agent is ghost
            value_action.append((value(succ, next_agent, depth)[0], action))
          else: # next agent is pacman
            value_action.append((value(succ, next_agent, depth - 1)[0], action))

      if agentIndex == 0: # pacman
        return max(value_action)
      else: # ghost
        return min(value_action)

    return value(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore()
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state)

      value_action = []
      for action in state.getLegalActions(agentIndex):
          next_agent = nextAgent(state, agentIndex)
          succ = state.generateSuccessor(agentIndex, action)
          if next_agent != 0: # next agent is ghost
            value_action.append(value(succ, next_agent, depth))
          else: # next agent is pacman
            value_action.append(value(succ, next_agent, depth - 1))

      if agentIndex == 0:
        return max(value_action)
      else:
        return min(value_action)

    next_agent = nextAgent(gameState, self.index)
    succ = gameState.generateSuccessor(self.index, action)
    if next_agent != 0: # next agent is ghost
      return value(succ, next_agent, self.depth)
    else: # next agent is pacman
      return value(succ, next_agent, self.depth - 1)
  
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore(), None
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state), None

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if agentIndex == 0: # pacman
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append((value(succ, next_agent, new_depth)[0], action))
        return max(value_action)
      else : # ghost
        sum = 0
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          sum = sum + value(succ, next_agent, new_depth)[0]
        return sum/len(state.getLegalActions(agentIndex)), None

    return value(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore()
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state)

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if agentIndex == 0: # pacman
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append(value(succ, next_agent, new_depth))
        return max(value_action)
      else : # ghost
        sum = 0
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          sum += value(succ, next_agent, new_depth)
        return sum / len(state.getLegalActions(agentIndex))

    next_agent = nextAgent(gameState, self.index)
    succ = gameState.generateSuccessor(self.index, action)
    if next_agent != 0: # next agent is ghost
      return value(succ, next_agent, self.depth)
    else: # next agent is pacman
      return value(succ, next_agent, self.depth - 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore(), None
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state), None

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if agentIndex == 0: # pacman
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append((value(succ, next_agent, new_depth)[0], action))
        return max(value_action)
      else : # ghost
        expect = 0
        for action in state.getLegalActions(agentIndex):
          if(action == Directions.STOP):
            prob = 0.5 + 0.5 * 1/(len(state.getLegalActions(agentIndex)))
          else:
            prob = 0.5 * 1/(len(state.getLegalActions(agentIndex)))
          succ = state.generateSuccessor(agentIndex, action)
          expect += value(succ, next_agent, new_depth)[0] * prob
        return expect, None

    return value(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore()
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state)

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if agentIndex == 0: # pacman
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append(value(succ, next_agent, new_depth))
        return max(value_action)
      else : # ghost
        expect = 0
        for action in state.getLegalActions(agentIndex):
          if(action == Directions.STOP):
            prob = 0.5 + 0.5 * 1/(len(state.getLegalActions(agentIndex)))
          else:
            prob = 0.5 * 1/(len(state.getLegalActions(agentIndex)))
          succ = state.generateSuccessor(agentIndex, action)
          expect += value(succ, next_agent, new_depth) * prob
        return expect

    next_agent = nextAgent(gameState, self.index)
    succ = gameState.generateSuccessor(self.index, action)
    if next_agent != 0: # next agent is ghost
      return value(succ, next_agent, self.depth)
    else: # next agent is pacman
      return value(succ, next_agent, self.depth - 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore(), None
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state), None

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if (agentIndex == 0) or ((agentIndex % 2) == 1) : # pacman or odd ghost
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append((value(succ, next_agent, new_depth)[0], action))
        if agentIndex == 0:
          return max(value_action)
        else:
          return min(value_action)
      else : # even ghost
        sum = 0
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          sum = sum + value(succ, next_agent, new_depth)[0]
        return sum/len(state.getLegalActions(agentIndex)), None

    return value(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore()
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state)

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if (agentIndex == 0) or ((agentIndex % 2) == 1) : # pacman or odd ghost
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append(value(succ, next_agent, new_depth))
        if agentIndex == 0:
          return max(value_action)
        else:
          return min(value_action)
      else : # even ghost
        sum = 0
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          sum = sum + value(succ, next_agent, new_depth)
        return sum/len(state.getLegalActions(agentIndex))

    next_agent = nextAgent(gameState, self.index)
    succ = gameState.generateSuccessor(self.index, action)
    if next_agent != 0: # next agent is ghost
      return value(succ, next_agent, self.depth)
    else: # next agent is pacman
      return value(succ, next_agent, self.depth - 1)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if agentIndex == state.getNumAgents() - 1:
        return 0
      else:
        return agentIndex + 1
      
    def value(state, agentIndex, depth, alpha, beta):
      
      # isEnd(s)
      # Terminal states can be found by one of the following: pacman won, pacman lost or there are no legal moves. 
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore(), None
      
      # depty == 0
      if depth == 0:
        return self.evaluationFunction(state), None

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # next agent is ghost
        new_depth = depth
      else: # next agent is pacman
        new_depth = depth - 1

      if (agentIndex == 0): # pacman
        max_value = float("-inf"), None
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append((value(succ, next_agent, new_depth, alpha, beta)[0], action))
          max_value = max(value_action)
          alpha = max(alpha, max_value[0])
          if(beta <= alpha):
            break
        return max_value
      elif ((agentIndex % 2) == 1):# odd ghost
        min_value = float("inf"), None
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append((value(succ, next_agent, new_depth, alpha, beta)[0], action))
          min_value = min(value_action)
          beta = min(beta, min_value[0])
          if(beta <= alpha):
            break
        return min_value
      else : # even ghost
        sum = 0
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          sum = sum + value(succ, next_agent, new_depth, alpha, beta)[0]
        expect = sum/len(state.getLegalActions(agentIndex)), None
        beta = min(beta, expect[0])
        return expect

    return value(gameState, self.index, self.depth, float("-inf"), float("inf"))[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def nextAgent(state, agentIndex):
      if(agentIndex == state.getNumAgents() - 1):
        return 0
      else:
        return agentIndex + 1
    
    def value(state, agentIndex, depth, alpha, beta):

      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) is None: 
        return state.getScore()
      
      if depth == 0:
        return self.evaluationFunction(state)

      next_agent = nextAgent(state, agentIndex)
      if next_agent != 0: # ghost
        new_depth = depth
      else:
        new_depth = depth - 1

      if (agentIndex == 0):
        max_value = float("-inf")
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append(value(succ, next_agent, new_depth, alpha, beta))
          max_value = max(value_action)
          alpha = max(max_value, alpha)
          if (beta <= alpha):
            break
        return max_value
      elif (agentIndex % 2) == 1:
        min_value = float("inf")
        value_action = []
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          value_action.append(value(succ, next_agent, new_depth, alpha, beta))
          min_value = min(value_action)
          beta = min(beta, min_value)
          if (beta <= alpha):
            break
        return min_value
      else:
        sum = 0
        for action in state.getLegalActions(agentIndex):
          succ = state.generateSuccessor(agentIndex, action)
          sum += value(succ, next_agent, new_depth, alpha, beta)
        expect = sum / len(state.getLegalActions(agentIndex))
        beta = min(beta, expect)
        return expect

    next_agent = nextAgent(gameState, self.index)
    succ = gameState.generateSuccessor(self.index, action)
    if next_agent != 0: # next agent is ghost
      return value(succ, next_agent, self.depth, float("-inf"), float("inf"))
    else: # next agent is pacman
      return value(succ, next_agent, self.depth - 1, float("-inf"), float("inf"))

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  features = []
  weights = []
  
  # initial score
  features.append(currentGameState.getScore())
  weights.append(1)

  # pacman position
  pac_Pos = currentGameState.getPacmanPosition() 

  # food distance feature
  food_Radius = 20
  foodGrid = currentGameState.getFood()
  for x in range( max(0, pac_Pos[0] - food_Radius), min(foodGrid.width, pac_Pos[0] + food_Radius)):
      for y in range( max(0, pac_Pos[1] - food_Radius), min(foodGrid.height, pac_Pos[1] + food_Radius)):
          if ( foodGrid[x][y] ):
              food_feature = 1 / (2 ** manhattanDistance((x,y), pac_Pos))
              features.append(food_feature)
              weights.append(10)

  # ghost distance feature
  ghostStates = currentGameState.getGhostStates()
  scared_ghost = 0
  for ghost in ghostStates:
      ghost_pos = ghost.getPosition()
      ghost_to_pac_dist = manhattanDistance(ghost_pos, pac_Pos)
      if ghost_to_pac_dist == 0:
          ghost_feature = 1
          ghost_weight = -1000
      else:
          ghost_feature = 1 / ghost_to_pac_dist
          if ghost.scaredTimer > 0:
              scared_ghost += 1
              ghost_weight = 110
          else:
              ghost_weight = -30
      features.append(ghost_feature)
      weights.append(ghost_weight)

  # capsule distance feature
  if scared_ghost == 0: # if there are not any scared ghost
    for capsule_pos in currentGameState.getCapsules():
        capsule_feature = 1 / manhattanDistance(capsule_pos, pac_Pos)
        features.append(capsule_feature)
        weights.append(15)

  total_score = 0
  for i in range(len(features)):
    total_score += features[i] * weights[i]
  return total_score

  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
