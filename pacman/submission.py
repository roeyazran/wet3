import random, util
from game import Agent

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
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
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
  if gameState.isWin() or gameState.isLose():
    return gameState.getScore()
  PacmanLocation = gameState.getPacmanState().configuration.pos
  GhostStatesArr =  gameState.getGhostStates()
  GhostsDistances = [util.manhattanDistance(PacmanLocation, Ghost.configuration.pos) for Ghost in GhostStatesArr]
  risk = 1/(min(GhostsDistances)) if min(GhostsDistances) != 0 else 1
  food_items = 0
  CurrentFood =gameState.getFood()
  x,y = util.nearestPoint(PacmanLocation)
  FoodSquares =0
  for i in list(range(-3,3)):
    for j in list(range(-3, 3)):
      if  isLeagalPos(gameState,x+i,y+j):
        FoodSquares += 1
        if CurrentFood[x+i][y+j]:
          food_items += 1
  food_density = 0
  CapsuleActive = 4 if gameState.getGhostState(GhostsDistances.index(min(GhostsDistances))+1).scaredTimer >= 0.5*min(GhostsDistances) else -1

  minFoodDist= findMinFoodDistance(gameState)
  punishWalls = punishNearWalls(gameState)
  return 0.5*gameState.getScore()+0.5*abs(gameState.getScore())*(0.6*risk*CapsuleActive+0.05*food_density+0.3*(1/(minFoodDist[0]))-0.05*punishWalls)


def findMinFoodDistance(gameState):
  CurrentFood =gameState.getFood()
  PacmanLocation =  util.nearestPoint(gameState.getPacmanState().configuration.pos)
  if gameState.data._foodEaten == (PacmanLocation[0],PacmanLocation[1]):
    return (1,(PacmanLocation[0],PacmanLocation[1]))
  for i in list(range(1, max(gameState.data.layout.width, gameState.data.layout.height))):
    topBaseRow = PacmanLocation[1] + i
    leftBaseCol = PacmanLocation[0] - i
    bottomBaseRow = PacmanLocation[1] - i
    rightBaseCol = PacmanLocation[0] + i
    for j in list(range(0, i+1)):
      # Sacnning in squares arount pacman
      # top row
      if isLeagalPos(gameState, PacmanLocation[0] + j, topBaseRow) and (CurrentFood[PacmanLocation[0] + j][topBaseRow] or (PacmanLocation[0] + j, topBaseRow) in gameState.getCapsules()):
        return (i,(PacmanLocation[0] + j, topBaseRow))

      if isLeagalPos(gameState, PacmanLocation[0] - j, topBaseRow) and (CurrentFood[PacmanLocation[0] - j][topBaseRow] or (PacmanLocation[0] - j, topBaseRow) in gameState.getCapsules()):
        return (i,(PacmanLocation[0] - j, topBaseRow))

      # bottomRow
      if isLeagalPos(gameState, PacmanLocation[0] + j, bottomBaseRow) and (CurrentFood[PacmanLocation[0] + j][
        bottomBaseRow] or (PacmanLocation[0] + j, bottomBaseRow) in gameState.getCapsules()):
        return (i,(PacmanLocation[0] + j, bottomBaseRow))

      if isLeagalPos(gameState, PacmanLocation[0] - j, bottomBaseRow) and (CurrentFood[PacmanLocation[0] - j][
        bottomBaseRow] or (PacmanLocation[0] - j, bottomBaseRow) in gameState.getCapsules()):
        return (i,(PacmanLocation[0] - j, bottomBaseRow))

      # left col
      if isLeagalPos(gameState, leftBaseCol, PacmanLocation[1] + j) and (CurrentFood[leftBaseCol][PacmanLocation[1] + j]
        or (leftBaseCol, PacmanLocation[1] + j) in gameState.getCapsules()):
        return (i,(leftBaseCol, PacmanLocation[1] + j))

      if isLeagalPos(gameState, leftBaseCol, PacmanLocation[1] - j) and (CurrentFood[leftBaseCol][PacmanLocation[1] - j]
        or (leftBaseCol, PacmanLocation[1] - j) in gameState.getCapsules()):
        return (i,(leftBaseCol, PacmanLocation[1] - j))

      # right col
      if isLeagalPos(gameState, rightBaseCol, PacmanLocation[1] + j) and (CurrentFood[rightBaseCol][
        PacmanLocation[1] + j] or (rightBaseCol, PacmanLocation[1] + j) in gameState.getCapsules()):
        return (i,(rightBaseCol, PacmanLocation[1] + j))

      if isLeagalPos(gameState, rightBaseCol, PacmanLocation[1] - j) and (CurrentFood[rightBaseCol][PacmanLocation[1] - j] or (rightBaseCol, PacmanLocation[1] - j) in gameState.getCapsules()):
        return (i,(rightBaseCol, PacmanLocation[1] - j))

def punishNearWalls(gameState):
  CurrentFood =gameState.getFood()
  CurrentWalls = gameState.getWalls()
  PacmanLocation =  util.nearestPoint(gameState.getPacmanState().configuration.pos)
  punish = 0
  if gameState.data._foodEaten == (PacmanLocation[0],PacmanLocation[1]):
    return punish
  punish+=1  #punish for no food at point
  (right,left,top,bottom) = (0,0,0,0)
  if isLeagalPos(gameState,PacmanLocation[0]+1 ,PacmanLocation[1]) and CurrentWalls[PacmanLocation[0]+1][PacmanLocation[1]]:
    punish+=1 #punish for each wall in current location
    right = 1
  if isLeagalPos(gameState,PacmanLocation[0]-1 ,PacmanLocation[1]) and CurrentWalls[PacmanLocation[0]-1][PacmanLocation[1]]:
    punish+=1 #punish for each wall in current location
    left=1
  if isLeagalPos(gameState,PacmanLocation[0] ,PacmanLocation[1]+1) and CurrentWalls[PacmanLocation[0]][PacmanLocation[1]+1]:
    punish+=1 #punish for each wall in current location
    top=1
  if isLeagalPos(gameState,PacmanLocation[0] ,PacmanLocation[1]-1) and CurrentWalls[PacmanLocation[0]][PacmanLocation[1]-1]:
    punish+=1 #punish for each wall in current location
    bottom=1

  punish = ((top and left) + (top and right) +  (bottom and left) + (bottom and right))

  return punish/8;  #maximom is 1

def isLeagalPos(gameState,x,y):
  if x >= 0 and x < gameState.data.layout.width and y >= 0 and y < gameState.data.layout.height:
    return True
  return False


#     ********* MultiAgent Search Agents- sections c,d,e,f*********


class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of you
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

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
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
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
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """



    return self.Minimax(gameState,self.depth)[0]

    # END_YOUR_CODE

  def Minimax(self,gameState, Depth):
    if gameState.isLose() or gameState.isWin() or Depth ==0: return (None,self.evaluationFunction(gameState))
    currentAgentIndex = 0 if  gameState.data._agentMoved == None else (gameState.data._agentMoved + 1) % gameState.getNumAgents()
    print(currentAgentIndex )
    if currentAgentIndex == 0:
    # now its pacman turn
      MaxCandidateflag = 1
      MaxSuccessorValue=0
      BestMove = None
      for action in gameState.getLegalActions(currentAgentIndex):
        NewSucc=gameState.generateSuccessor(currentAgentIndex, action)
        (NewBestMove, NewSuccVal)= self.Minimax(NewSucc, Depth - 1)
        if MaxCandidateflag == 1:
          MaxCandidateflag = 0
          MaxSuccessorValue = NewSuccVal
          BestMove = action
        elif MaxSuccessorValue < NewSuccVal:
          MaxSuccessorValue = NewSuccVal
          BestMove = action
      return (BestMove,MaxSuccessorValue)

    MinCandidateflag = 1

    MinSuccesorValue =0
    worstMove = None
    for action in gameState.getLegalActions(currentAgentIndex):
      NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
      (NewWorstMove, NewSuccVal) = self.Minimax(NewSucc, Depth)
      if MinCandidateflag == 1:
        MinCandidateflag = 0
        MinSuccesorValue = NewSuccVal
        WoMove = action
      elif MinSuccesorValue > NewSuccVal:
        MinSuccesorValue = NewSuccVal
        worstMove = action
    return (worstMove,MinSuccesorValue)


          ######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



