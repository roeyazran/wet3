import random, util, numpy
from game import Agent,Actions

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
  # print ("Dfds")
  if gameState.isWin() or gameState.isLose():
    return gameState.getScore()
  PacmanLocation = gameState.getPacmanState().configuration.pos
  GhostStatesArr = gameState.getGhostStates()
  GhostsDistances = [util.manhattanDistance(PacmanLocation, Ghost.configuration.pos) for Ghost in GhostStatesArr]
  risk = 1/(min(GhostsDistances)) if min(GhostsDistances) != 0 else 1
  CapsuleActive = 4 if gameState.getGhostState(GhostsDistances.index(min(GhostsDistances)) + 1).scaredTimer >= 0.5 * min(GhostsDistances) else -1
  # do we need to use -eaten above, for states which show the ghost was eaten

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

  minFoodDist= findMinFoodDistance(gameState)
  punishWalls = punishNearWalls(gameState)

  # (inhartionx,inhartiony) = util.nearestPoint((PacmanLocation[0]+Pacdx, PacmanLocation[1]+Pacdy))
  WalkToWallPunish = PunishWallRewardFood(gameState)
  # print(PacmanLocation," Walk to wall Punish: ", WalkToWallPunish)
  return 0.5*gameState.getScore()+0.5*abs(gameState.getScore())*(0.7*risk*CapsuleActive+0.*food_density+(1/(minFoodDist[0]))+0.1*WalkToWallPunish - 0.2*punishWalls)

def PunishWallRewardFood(gameState):
  RawData = CloestFoodOrWallInDir(gameState)
  if RawData ==0 : return 1
  return 1/RawData


# 0 for food in palce negative value if sees wall in front before food,  positive if sees food
def CloestFoodOrWallInDir(gameState):
  i=0
  PacmanLocation= gameState.getPacmanState().configuration.pos
  if PacmanLocation == gameState.data._foodEaten:
    return 0
  (Pacdx, Pacdy)= Actions.directionToVector(gameState.getPacmanState().getDirection())
  (initx, inity)= gameState.getPacmanPosition()
  (initx, inity) = util.nearestPoint((PacmanLocation[0] + Pacdx, PacmanLocation[1] + Pacdy))
  while isLeagalPos(gameState,initx,inity):
    i+=1
    if gameState.getFood()[initx][inity]:
      return i
    if gameState.hasWall(initx,inity):
      return -i
    (initx, inity) = util.nearestPoint((initx + Pacdx, inity + Pacdy))
  return i

# def ClosestWallInDirestion(gameState):
#   i=0
#   #irelevant factor when there is food in the state
#   (initx, inity)= gameState.getPacmanPosition()
#   (inhartionx,inhartiony) = util.nearestPoint((PacmanLocation[0]+Pacdx, PacmanLocation[1]+Pacdy))
#   while isLeagalPos(gameState,initx,inity)
#     i+=1;
#     #there is food on the way
#     if gameState.getFood()[initx][inity]:
#       return 0
#   return i;

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
  if not isLeagalPos(gameState,PacmanLocation[0]+1 ,PacmanLocation[1]) or CurrentWalls[PacmanLocation[0]+1][PacmanLocation[1]]:
    punish+=1 #punish for each wall in current location
    right = 1
  if not isLeagalPos(gameState,PacmanLocation[0]-1 ,PacmanLocation[1]) and CurrentWalls[PacmanLocation[0]-1][PacmanLocation[1]]:
    punish+=1 #punish for each wall in current location
    left=1
  if not isLeagalPos(gameState,PacmanLocation[0] ,PacmanLocation[1]+1) and CurrentWalls[PacmanLocation[0]][PacmanLocation[1]+1]:
    punish+=1 #punish for each wall in current location
    top=1
  if not isLeagalPos(gameState,PacmanLocation[0] ,PacmanLocation[1]-1) and CurrentWalls[PacmanLocation[0]][PacmanLocation[1]-1]:
    punish+=1 #punish for each wall in current location
    bottom=1

  punish = ((top and left) + (top and right) +  (bottom and left) + (bottom and right))

  return punish/4  #maximom is 1

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

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
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

    bestMove= None
    MaxSuccValue = - numpy.inf
    for action in gameState.getLegalActions(0):
      NewSucc = gameState.generateSuccessor(0, action)
      NewSuccVal = self.Minimax(NewSucc, self.depth- 1)
      if NewSuccVal > MaxSuccValue:
        MaxSuccValue = NewSuccVal
        bestMove = action
    return bestMove

    # END_YOUR_CODE

  def Minimax(self,gameState, Depth):
    if gameState.isLose() or gameState.isWin() or Depth ==0: return (self.evaluationFunction(gameState))
    currentAgentIndex = 0 if  gameState.data._agentMoved == None else (gameState.data._agentMoved + 1) % gameState.getNumAgents()
    if currentAgentIndex == 0:
    # now its pacman turn
      MaxCandidateflag = 1
      MaxSuccessorValue=0
      for action in gameState.getLegalActions(currentAgentIndex):
        NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
        NewSuccVal = self.Minimax(NewSucc, Depth - 1)
        if MaxCandidateflag == 1:
          MaxCandidateflag = 0
          MaxSuccessorValue = NewSuccVal
          # BestMove = action
        elif MaxSuccessorValue < NewSuccVal:
          MaxSuccessorValue = NewSuccVal
          # BestMove = action
      return MaxSuccessorValue

    MinCandidateflag = 1
    MinSuccesorValue =0
    for action in gameState.getLegalActions(currentAgentIndex):
      NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
      NewSuccVal = self.Minimax(NewSucc, Depth)
      if MinCandidateflag == 1:
        MinCandidateflag = 0
        MinSuccesorValue = NewSuccVal
      elif MinSuccesorValue > NewSuccVal:
        MinSuccesorValue = NewSuccVal
    return MinSuccesorValue


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
    (bestMove,BestScore) = (None, 0)
    alpha = -numpy.inf
    MaxSuccValue = - numpy.inf
    for action in gameState.getLegalActions(0):
      NewSucc = gameState.generateSuccessor(0, action)
      NewSuccVal = self.AlphaBeta(NewSucc, alpha ,numpy.inf , self.depth- 1)
      if NewSuccVal > MaxSuccValue:
        MaxSuccValue = NewSuccVal
        alpha = MaxSuccValue
        bestMove = action
    return bestMove
    # END_YOUR_CODE

  def AlphaBeta(self, gameState,alpha,beta,Depth):
    if gameState.isLose() or gameState.isWin() or Depth ==0: return (self.evaluationFunction(gameState))
    currentAgentIndex = 0 if  gameState.data._agentMoved == None else (gameState.data._agentMoved + 1) % gameState.getNumAgents()
    if currentAgentIndex == 0:
    # now its pacman turn - Max vertex
      MaxCandidateflag = 1
      MaxSuccessorValue=0
      for action in gameState.getLegalActions(currentAgentIndex):
        NewSucc=gameState.generateSuccessor(currentAgentIndex, action)
        NewSuccVal = self.AlphaBeta(NewSucc, alpha,beta,Depth - 1)
        alpha = max(alpha, NewSuccVal)
        if MaxCandidateflag == 1:
          MaxCandidateflag = 0
          MaxSuccessorValue = NewSuccVal
        elif MaxSuccessorValue < NewSuccVal:
          MaxSuccessorValue = NewSuccVal
        if MaxSuccessorValue >= beta:
            return numpy.inf
      return MaxSuccessorValue

    MinCandidateflag = 1
    MinSuccesorValue =0
    for action in gameState.getLegalActions(currentAgentIndex):
      NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
      NewSuccVal = self.AlphaBeta(NewSucc, alpha, beta, Depth)
      beta = min(beta, NewSuccVal)
      if MinCandidateflag == 1:
        MinCandidateflag = 0
        MinSuccesorValue = NewSuccVal
      elif MinSuccesorValue > NewSuccVal:
        MinSuccesorValue = NewSuccVal
      if MinCandidateflag <= alpha:
        return - numpy.inf
    return MinSuccesorValue



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

    bestAction = None
    maxStatVal = - numpy.inf
    # print("~~~~~~START~~~~~~")
    for action in gameState.getLegalActions():
        # print(action)
        nextPacManState = gameState.generateSuccessor(0, action)
        stateVal = self.Expectimax(nextPacManState, self.depth-1)
        if max(stateVal, maxStatVal) == stateVal:
            maxStatVal = stateVal
            bestAction = action
    # print(bestAction)
    return bestAction

  def Expectimax(self, gameState, Depth):
    if gameState.isLose() or gameState.isWin() or Depth ==0: return (self.evaluationFunction(gameState))
    currentAgentIndex = 0 if  gameState.data._agentMoved == None else (gameState.data._agentMoved + 1) % gameState.getNumAgents()
    if currentAgentIndex == 0:
    # now its pacman turn
    #   print("PACMAN TURN")
      maxStatVal = - numpy.inf
      for action in gameState.getLegalActions(currentAgentIndex):
        NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
        maxStatVal = max(maxStatVal, self.Expectimax(NewSucc, Depth - 1))
      # print ("pacmac max chosen: ",maxStatVal )
      return maxStatVal

    else:
      dist = util.Counter()
      for a in gameState.getLegalActions(currentAgentIndex): dist[a] = 1.0
      dist.normalize()
      # print("ghost:", currentAgentIndex, "| have:", len(dist)," moves")
      EStatVal = 0
      for action in gameState.getLegalActions(currentAgentIndex):
        # print("ghost:", currentAgentIndex, "act:", action)
        NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
        EStatVal += dist[action]* self.Expectimax(NewSucc, Depth)
      # print("ghost:", currentAgentIndex, "| retrun:", EStatVal)
      return EStatVal


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
    bestAction = None
    maxStatVal = - numpy.inf
    # print("~~~~~~START~~~~~~")
    for action in gameState.getLegalActions():
        # print(action)
        nextPacManState = gameState.generateSuccessor(0, action)
        stateVal = self.DirectionalExpectimax(nextPacManState, self.depth-1)
        if max(stateVal, maxStatVal) == stateVal:
            maxStatVal = stateVal
            bestAction = action
    # print(bestAction)
    return bestAction

  def DirectionalExpectimax(self, gameState, Depth):
    if gameState.isLose() or gameState.isWin() or Depth == 0: return (self.evaluationFunction(gameState))
    currentAgentIndex = 0 if gameState.data._agentMoved == None else (gameState.data._agentMoved + 1) % gameState.getNumAgents()
    if currentAgentIndex == 0:
      # now its pacman turn
      # print("PACMAN TURN")
      maxStatVal = - numpy.inf
      for action in gameState.getLegalActions(currentAgentIndex):
        NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
        maxStatVal = max(maxStatVal, self.DirectionalExpectimax(NewSucc, Depth - 1))
      # print("pacmac max chosen: ", maxStatVal)
      return maxStatVal

    else:
      ghostState = gameState.getGhostState(currentAgentIndex)
      legalActions = gameState.getLegalActions(currentAgentIndex)
      pos = gameState.getGhostPosition(currentAgentIndex)
      isScared = ghostState.scaredTimer > 0

      speed = 1
      if isScared: speed = 0.5

      actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
      newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
      pacmanPosition = gameState.getPacmanPosition()

      # Select best actions given the state
      distancesToPacman = [util.manhattanDistance(pos, pacmanPosition) for pos in newPositions]
      if isScared:
        bestScore = max(distancesToPacman)
        bestProb = 0.8
      else:
        bestScore = min(distancesToPacman)
        bestProb = 0.8
      bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

      # Construct distribution
      dist = util.Counter()
      for a in bestActions: dist[a] = bestProb / len(bestActions)
      for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
      dist.normalize()

      # print("ghost:", currentAgentIndex, "| have:", len(dist), " moves")
      EStatVal = 0
      for action in gameState.getLegalActions(currentAgentIndex):
        # print("ghost:", currentAgentIndex, "act:", action)
        NewSucc = gameState.generateSuccessor(currentAgentIndex, action)
        EStatVal += dist[action] * self.DirectionalExpectimax(NewSucc, Depth)
      # print("ghost:", currentAgentIndex, "| retrun:", EStatVal)
      return EStatVal


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



