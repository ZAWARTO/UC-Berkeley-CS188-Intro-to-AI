# UC Berkeley CS188

## Introduction

This repository contains my personal implementations of the course's assignments on artificial intelligence algorithms in Pacman UC Berkeley CS188.

The original code provided in the course was in Python 2, but I have taken the time to port it to Python 3. Additionally, I have simplified the programming syntax in the exercises to make it easier to understand the algorithms I have developed for the solutions.

I hope that this repository will serve as a useful resource for anyone interested in learning more about AI algorithms and their application in Pacman games.

## Project 1: Search
### Q1: Depth First Search
Code
```python
start_state = problem.getStartState()
next_state = []

visited = []

state_stack = util.Stack()
state_stack.push((start_state, []))

while not state_stack.isEmpty() and not problem.isGoalState(next_state):
    actual_state, path = state_stack.pop()
    adjacents = problem.getSuccessors(actual_state)

    if actual_state not in visited:
        visited.append(actual_state)
        for adjacent in adjacents:
            state = adjacent[0]
            if state not in visited:
                next_state = adjacent[0]
                action = adjacent[1]
                state_stack.push((state, path + [action]))

path.append(action)
return path 
```
Solution

The algorithm starts by initializing a stack `state_stack` with a tuple containing the start state and an empty list to represent the path. Then enters a loop that continues until either the stack is empty or a goal state is found.

In each iteration of the loop, the algorithm pops a state from the stack and checks if it has already been visited. If it has not been visited, the state is added to the visited list. The algorithm then generates a list of successors for the current state using the `getSuccessors()` method of the problem.

For each successor, if it has not been visited, the algorithm creates a new tuple with the successor state and the action that led to it, and pushes it onto the stack. The action is added to the path list to represent the current path from the start state.

Finally, if the loop ends because a goal state has been found, the algorithm returns the path to that state, which is constructed by appending the last action to the path list.

### Q2: Breadth First Search
Code
```python
start_state = problem.getStartState()

visited = []

state_queue = util.Queue()
state_queue.push((start_state, []))

while not state_queue.isEmpty():
    actual_state, path = state_queue.pop()

    if problem.isGoalState(actual_state):
        return path

    if actual_state not in visited:
        adjacents = problem.getSuccessors(actual_state)

        for adjacent in adjacents:
            state = adjacent[0]
            if state not in visited:
                action = adjacent[1]
                state_queue.push((state, path + [action]))
    visited.append(actual_state)
return path
```
Solution

The algorithm starts by initializing a queue `state_queue` with a tuple containing the start state and an empty list to represent the path. Then enters a loop that continues until the queue is empty.

In each iteration of the loop, the algorithm dequeues a state from the queue and checks if it is a goal state. If it is, the algorithm returns the path to that state.

If the state is not a goal state, the algorithm checks if it has already been visited. If it has not been visited, the state is added to the visited list. The algorithm then generates a list of successors for the current state using the `getSuccessors()` method of the problem.

For each successor, if it has not been visited, the algorithm creates a new tuple with the successor state and the action that led to it, and pushes it onto the queue. The action is added to the path list to represent the current path from the start state.

Finally, if the loop ends because the queue is empty and no goal state has been found, the algorithm returns the last path that was explored.

### Q3: Uniform Cost Search
Code
```python
start_state = problem.getStartState()

visited = []

state_priority_queue = util.PriorityQueue()
state_priority_queue.push((start_state, []), 0)

while not state_priority_queue.isEmpty():
    actual_state, path = state_priority_queue.pop()

    if problem.isGoalState(actual_state):
        return path

    if actual_state not in visited:
        adjacents = problem.getSuccessors(actual_state)

        for adjacent in adjacents:
            state = adjacent[0]
            if state not in visited:
                action = adjacent[1]
                state_priority_queue.push((state, path + [action]), problem.getCostOfActions(path + [action]))
    visited.append(actual_state)
return path
```
Solution

The algorithm starts by initializing a priority queue `state_priority_queue` with a tuple containing the start state, an empty list to represent the path, and a priority of 0. The priority represents the cost of the path to the current state. The algorithm then enters a loop that continues until the priority queue is empty.

In each iteration of the loop, the algorithm dequeues a state from the priority queue and checks if it is a goal state. If it is, the algorithm returns the path to that state.

If the state is not a goal state, the algorithm checks if it has already been visited. If it has not been visited, the state is added to the visited list. The algorithm then generates a list of successors for the current state using the `getSuccessors()` method of the problem.

For each successor, if it has not been visited, the algorithm creates a new tuple with the successor state and the action that led to it, and pushes it onto the priority queue with a priority equal to the cost of the path to the new state. The action is added to the path list to represent the current path from the start state.

Finally, if the loop ends because the priority queue is empty and no goal state has been found, the algorithm returns the last path that was explored.

### Q4: A* Search
Code
```python
start_state = problem.getStartState()

visited = []

state_priority_queue = util.PriorityQueue()
state_priority_queue.push((start_state, []), nullHeuristic(start_state, problem))

while not state_priority_queue.isEmpty():
    actual_state, path = state_priority_queue.pop()

    if problem.isGoalState(actual_state):
        return path

    if actual_state not in visited:
        adjacents = problem.getSuccessors(actual_state)

        for adjacent in adjacents:
            state = adjacent[0]
            if state not in visited:
                action = adjacent[1]
                cost = problem.getCostOfActions(path + [action]) + heuristic(state, problem)
                state_priority_queue.push((state, path + [action]), cost)
    visited.append(actual_state)
return path
```
Solution

The algorithm starts by initializing a priority queue `state_priority_queue` with a tuple containing the start state, an empty list to represent the path, and a priority equal to the sum of the cost of the path to the current state and the heuristic estimate of the cost to the goal state. The heuristic estimate is calculated using the `nullHeuristic()` function, which returns 0.

The algorithm then enters a loop that continues until the priority queue is empty. In each iteration of the loop, the algorithm dequeues a state from the priority queue and checks if it is a goal state. If it is, the algorithm returns the path to that state.

If the state is not a goal state, the algorithm checks if it has already been visited. If it has not been visited, the state is added to the visited list. The algorithm then generates a list of successors for the current state using the `getSuccessors()` method of the problem.

For each successor, if it has not been visited, the algorithm calculates the cost of the path to the new state by adding the cost of the path to the current state and the heuristic estimate of the cost to the goal state. The successor state, the action that led to it, and the cost are then added to the priority queue as a tuple. The action is added to the path list to represent the current path from the start state.

Finally, if the loop ends because the priority queue is empty and no goal state has been found, the algorithm returns the last path that was explored.

### Q5: Corners Problem: Representation
Code
```python
def __init__(self, startingGameState):
    ... #Irrelevant code
    self.costFn = lambda x: 1
    
def getStartState(self):
    """
    Returns the start state (in your state space, not the full Pacman state
    space)
    """
    "*** YOUR CODE HERE ***"
    return ((self.startingPosition, []))

def isGoalState(self, state):
    """
    Returns whether this search state is a goal state of the problem.
    """
    "*** YOUR CODE HERE ***"
    if len(state[1]) == 4:
        return True
    return False

def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.

     As noted in search.py:
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost'
        is the incremental cost of expanding to that successor
    """

    successors = []
    x,y = state[0]

    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        # Add a successor state to the successor list if the action is legal
        # Here's a code snippet for figuring out whether a new position hits a wall:
        #   x,y = currentPosition
        #   dx, dy = Actions.directionToVector(action)
        #   nextx, nexty = int(x + dx), int(y + dy)
        #   hitsWall = self.walls[nextx][nexty]

        "*** YOUR CODE HERE ***"

        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)

        if not self.walls[nextx][nexty]:
            nextState = ((nextx, nexty), state[1])
            cost = self.costFn(nextState)
            corners = list(state[1])

            if nextState[0] in self.corners and nextState[0] not in corners:
                corners.append(nextState[0])

            successors.append((((nextx, nexty), corners), action, cost))

    self._expanded += 1 # DO NOT CHANGE
    return successors
```

Solution

The `__init__` function sets up the class attributes, and `self.costFn = lambda x: 1` sets the cost function to a constant function with cost 1 for all actions.

`getStartState(self)`: In this implementation, the start state is represented as a tuple of two values. The first value is the starting position of Pacman in the maze, represented as a (x,y) coordinate tuple. The second value is an empty list, which will be used to track the corners Pacman has visited.

The method simply returns a tuple containing the starting position and an empty list as the list of corners visited.

`isGoalState(self, state)`: This function checks if the length of the second element of the input state tuple is equal to 4. If it is, then the method returns True, indicating that the given state is a goal state. Otherwise, the method returns False. This means that the goal of the problem is to visit all the four corners in the given Pacman world.

`getSuccessors(self, state)`: The function takes a state and returns a list of successor states, along with the actions required to get there and their associated costs.

The function loops through four actions (North, South, East, West), and for each action, it checks if the resulting position is a legal move (i.e., it does not hit a wall). If the move is legal, the function computes the next state by updating the current position and adding it to the list of corners visited so far, if it's a corner. It also computes the cost of transitioning to the next state using the cost function.

The function then appends a tuple of (nextState, action, cost) to the list of successors. Finally, the function increments the count of the number of expanded states, and returns the list of successors.

### Q6: Corners Problem: Heuristic
Code

```python
corners = problem.corners # These are the corner coordinates
walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

"*** YOUR CODE HERE ***"
pos = state[0]
visited = state[1]
heuristic = 0

pending_corners = []
for i in range(4):
    if corners[i] not in visited:
        pending_corners.append(corners[i])

while(len(pending_corners)!=0):
    #Distance to minimize, infinite
    global_distance = float('inf')
    #Select nearest corner
    for corner in pending_corners:
        local_distance = util.manhattanDistance(pos, corner)
        if global_distance > local_distance:
            global_distance = local_distance
            nearest_corner = corner
    #Update heuristic
    heuristic = heuristic + global_distance
    pos = nearest_corner
    #Remove objective corner from pending corners
    pending_corners.remove(nearest_corner) 
return heuristic
```

Solution

The function receives a state that consists of the current position of Pacman and a list of visited corners. The corners and walls of the maze are obtained from the problem object.

The heuristic function computes an estimate of the minimum cost to reach a goal state, which in this case is visiting all corners. The function starts by initializing the heuristic cost to 0 and creating a list of pending corners that have not been visited yet.

The function then enters a loop that continues until all corners have been visited. In each iteration, the function selects the nearest corner from the pending corners list using the Manhattan distance between the current position of Pacman and each corner.

The function updates the heuristic cost by adding the distance to the nearest corner, sets Pacman's current position to the nearest corner, and removes the visited corner from the pending corners list. The process repeats until all corners have been visited, and the final heuristic cost is returned.

The function does not use walls in its computation, only the distance to the corners. Solved with 901 nodes expanded.

### Q7: Eating All The Dots: Heuristic
Code

```python
pending_food = foodGrid.asList()

#Distance to minimize, infinite
global_distance = float('inf')
#Distance to maximize, 0
global_distance2 = 0

for food in pending_food:
    local_distance = util.manhattanDistance(position, food)
    global_distance2 = 0
    if global_distance > local_distance:
        global_distance = local_distance

    for food2 in pending_food:
        local_distance = util.manhattanDistance(food, food2)
        if global_distance2 < local_distance:
            global_distance2 = local_distance

#Update heuristic
if(len(pending_food)):
    heuristic = global_distance + global_distance2
else:
    heuristic = global_distance2

return heuristic
```

Solution

The heuristic in the provided code takes into account both the distance to the nearest goal and the distance between different pending foods on the board. By only considering the distance to the nearest fruit, the total cost of the remaining path can be underestimated, leading the agent to make suboptimal decisions.

Calculating the distance to the nearest fruit is useful for an agent that only wants to eat that particular fruit and doesn't care about the rest of the board. However, in the case of Pac-Man, the objective is to eat all the food on the board, not just the nearest one. Calculating the distance to the nearest fruit does not provide information on how the other foods are positioned relative to each other, and thus an optimal decision cannot be made.

By considering the distance between all pending foods on the board, the agent can gain a better understanding of the total cost of the remaining path and thus make more informed and optimal decisions. Therefore, the heuristic in the provided code is better than calculating only the distance to the nearest fruit.

First, the code creates a list of all the remaining food dots on the grid using `foodGrid.asList()`. Then, two variables are initialized: `global_distance` and `global_distance2`. `global_distance` is set to a very large number to ensure that the first food dot's distance is smaller, and `global_distance2` is initialized to 0 to ensure that the first food dot is farther away than the maximum distance between any two dots.

Then, the code enters a loop that iterates through each food dot and calculates the Manhattan distance between the current position of Pacman and each dot using `util.manhattanDistance(position, food)`. The if statement inside the loop checks if the current dot is closer than the previous closest one `global_distance`. If it is, `global_distance` is updated to this new distance.

Afterwards, the code enters another loop nested inside the first one. This loop calculates the maximum distance between any two food dots by comparing the Manhattan distance again `util.manhattanDistance(food, food2)`. If the distance between two pellets is larger than the current maximum distance `global_distance2`, it is updated.

Finally, the heuristic value is calculated based on whether there are any remaining food dots. If there are, heuristic is set to the sum of `global_distance` and `global_distance2`. Otherwise, heuristic is set to `global_distance2`.

Path found with total cost of 60 in 10.9 seconds
Search nodes expanded: 7798

### Q8: Suboptimal Search
Code

```python
def findPathToClosestDot(self, gameState):
...
return search.breadthFirstSearch(problem)
```

In class AnyFoodSearchProblem(PositionSearchProblem)
```python
def isGoalState(self, state):
    """
    The state is Pacman's position. Fill this in with a goal test that will
    complete the problem definition.
    """
    x,y = state

    "*** YOUR CODE HERE ***"
    return self.food[x][y]
```
Solution

The suboptimal search uses the BFS algorithm to eat the closest dot. The goal state is reached when Pacman's position contains a dot.

## Project 2: Multi-Agent Search
### Q1: Reflex Agent
Code

```python
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
```

Solution

First, the function calculates the distance to the closest food dot using Manhattan distance. If there are no food dots remaining, the food distance is set to 0. The score is then updated to be the current score of the game minus the food distance.

After that, the function calculates the distance to the closest ghost using Manhattan distance. If there are no ghosts remaining, the ghost distance is set to 0. If a ghost is 1 unit away and not scared, the function returns negative infinity, indicating that this state is not desirable because of the posibility of an inminent game over. If a ghost is 1 unit away and scared, the function returns positive infinity, indicating that this state is very desirable.

Finally, the function updates the score based on whether the closest ghost is scared or not. If it is scared, the score is decreased by the ghost distance, otherwise it is increased by the ghost distance.

10/10 wins, average score of 821.

### Q2: Minimax
Code

```python
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
```

Solution

The function first checks if the game has been won or lost. If so, it returns the score of the game.

If the agent is Pacman (represented by agent = 0), the function finds the legal actions available to Pacman in the current state of the game. For each action, the function recursively calls itself on the successor state generated by that action and the depth incremented by 1. The function then compares the scores of each successor state and returns the maximum score. If the depth is 0, meaning we are at the root of the game tree, the function returns the best action found instead of the score.

If the agent is a ghost (represented by agent > 0), the function first determines the index of the next agent. If the current agent is the last agent, the next agent is set to 0. The function then finds the legal actions available to the current ghost agent and recursively calls itself on the successor states generated by each action and the next agent index. The function then returns the minimum score found among the successor states.

The function returns the result of calling minimax on the initial state of the game with depth 0 and agent 0, representing Pacman. This determines the optimal move for Pacman in the current state of the game.

### Q3: Alpha-Beta Pruning
Code
```python
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
```
Solution

The function first checks if the game has been won or lost. If so, it returns the score of the state. If the current agent is Pacman (agent=0), it obtains the legal actions of Pacman and initializes the best action as 'STOP' and the maximum score as negative infinity.

For each legal action, the function generates the successor state and recursively calls the alpha-beta function with the next agent (i.e., the next ghost). The returned score is compared with the maximum score found so far, and if it is greater, it updates the maximum score and the best action.

The function also updates the alpha value to be the maximum of the current alpha value and the maximum score found so far. If the alpha value is greater than or equal to the beta value, the function breaks out of the loop and returns the maximum score (or best action if depth is 0).

If the current agent is a ghost, the function initializes the minimum score as positive infinity and obtains the legal actions of the ghost. For each legal action, the function generates the successor state and recursively calls the alpha-beta function with the next agent. The returned score is compared with the minimum score found so far, and if it is less, it updates the minimum score.

The function also updates the beta value to be the minimum of the current beta value and the minimum score found so far. If the alpha value is greater than or equal to the beta value, the function breaks out of the loop and returns the minimum score.

Finally, the function returns the maximum score (or best action if depth is 0) if Pacman is the current agent or returns the minimum score if a ghost is the current agent. The function call at the end passes the initial game state, a depth of 0, the Pacman agent (agent=0), and initial alpha and beta values of negative and positive infinity, respectively.

### Q4: Expectimax
Code

```python
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
```

Solution

This function takes in the current state of the game, the current depth of the search, and the current agent to act. If the state is a terminal state (either a win or a loss), it returns the score of the state. 

If the current agent is Pacman (agent 0), it chooses the action with the highest expected score by recursively calling expectimax on each possible action and taking the maximum score. 

If the current agent is a ghost (agent 1 or higher), it calculates the expected score for each action by recursively calling expectimax on each possible successor state and taking the average score. The score/len(agent_actions) expression returns the average score over all possible actions.

At the end, the function returns the best action for Pacman at the current state, or the expected score for a ghost at the current state.

### Q5: Evaluation Function
Code

```python
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
```

Solution

The first two if statements check if the game is already won or lost, in which case the function returns the appropriate score of inf or -inf.

The function then initializes a variable score with the current game score, and gets the list of remaining food pellets, Pacman's position, the positions of the ghosts, and their scared times.

The function then computes a score based on the remaining food. It calculates the Manhattan distance between Pacman's position and the positions of all remaining food pellets, and takes the reciprocal of the minimum distance. This score is added to the overall score of the state.

The function then computes a score based on the nearest ghost. It calculates the Manhattan distance between Pacman's position and the positions of all ghosts, and takes the reciprocal of the minimum distance. If the minimum distance is less than or equal to 1, the function checks if the ghost is scared. If it is scared, the function returns inf (since Pacman can eat the ghost). Otherwise, the function subtracts the reciprocal of the minimum distance from the overall score. If the minimum distance is greater than 1, the function checks if any ghosts are scared. If so, it adds the reciprocal of the minimum distance to the score. Otherwise, it subtracts the reciprocal of the minimum distance from the score.

Perfect score.

## Project 5: Classification
### Q1: Perceptron
Code
```python

def train( self, trainingData, trainingLabels, validationData, validationLabels):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    for iteration in range(self.max_iterations):
        print ("Starting iteration ", iteration, "...")
        for i in range(len(trainingData)):
            f = trainingData[i]
            y = trainingLabels[i]
            score = util.Counter()

            for label in self.legalLabels:
                score[label] = self.weights[label] * f

            y_prim = score.argMax()

            if y_prim != y:
                self.weights[y] += f
                self.weights[y_prim] -= f

```
Solution

The function begins by initializing `self.features` to be a list of the keys of the features of the first data point in the training data. Then, the function enters a loop that will iterate for `self.max_iterations`, which is a hyperparameter that determines the maximum number of iterations the algorithm will run.

Inside the loop, the algorithm loops through each data point in the training data, represented by the counter f and the corresponding label y. It computes a score for each label using the dot product of the weight vector and the feature vector, and stores these scores in a `util.Counter()` object called score. The label with the highest score is then assigned to y_prim using the `argMax()` function of score.

If the predicted label `y_prim` is different from the true label y, the algorithm updates the weight vector as follows: it adds the feature vector f to the weight vector corresponding to the true label y, and subtracts f from the weight vector corresponding to the predicted label `y_prim`.

### Q2: MIRA
Code
```python
    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        newWeights = self.weights.copy()
        bestWeights = newWeights
        bestAccuracy = 0.0

        for c in Cgrid:
            self.initializeWeightsToZero()

            self.weights = newWeights.copy()
            for iteration in range(self.max_iterations):
                print ("Starting iteration ", iteration, "...")
                for i in range(len(trainingData)):
                    f = trainingData[i]
                    y = trainingLabels[i]
                    score = util.Counter()
                    
                    for label in self.legalLabels:
                        score[label] = self.weights[label] * f
                        
                    y_prim = score.argMax()

                    if y_prim != y:
                        tau = min(c, ((self.weights[y_prim] - self.weights[y]) * f + 1.0) / (2*sum(v**2 for v in f.values())))

                        for key in trainingData[i]:
                            self.weights[y][key] += f[key] * tau
                        for key in trainingData[i]:
                            self.weights[y_prim][key] -= f[key] * tau

            guesses = self.classify(validationData)

            count = 0
            for i in range(len(guesses)):
                if guesses[i] == validationLabels[i]:
                    count += 1

            accuracy = float(count)/float(len(guesses))
            
            if accuracy > bestAccuracy:
                bestWeights = self.weights.copy()
                bestAccuracy = accuracy

        self.weights = bestWeights  
```
Solution

The MIRA algorithm takes a hyperparameter C which controls the amount of relaxation that is allowed during weight updates. The higher the value of C, the less the weights can be updated, and vice versa.

The function `trainAndTune` takes as input the training and validation data and labels, as well as a grid of values for C. It initializes a new weight vector and keeps track of the best weights and accuracy seen so far. It then loops over the values of C in the grid and trains a new weight vector for each C value. The weight vector is updated iteratively for each training example using the MIRA update rule, which computes a weight update based on the difference between the predicted and correct labels and the size of the feature vector. After each training iteration, the weight vector is tested on the validation set and the accuracy is computed. If the accuracy is higher than the previous best accuracy, the weights and accuracy are updated.

Finally, the function sets the weight vector to the best weights found during training and returns.
