"""
### Author: Jacob Parmer, Auburn University
###
### Last Updated: September 9, 2020
"""

import numpy as np
from copy import deepcopy
from pudb import set_trace

# constants for special squares
BLACK_HOLE_SQUARE = 21
BLANK_SQUARE = 0

class State: 

    """
    Constructor. If "grid" variable is provided, sets self.grid = provided variable.
    If not, sets it to an empty 5x5 numpy array.

    """
    def __init__(self, grid = np.array([])):

        if (grid.size != 0):
            self.grid = grid

            # Determines the position of the BLANK_SQUARE in the grid
            i = 1
            j = 1
            for line in grid:
                for item in line:
                    if (item == 0): 
                        self.emptyCol = i
                        self.emptyRow = j
                        break
                    else: 
                        i = i + 1

                i = 1
                j = j + 1
            
        else:
            # init state provided by homework document
            self.grid = np.array([[2, 3, 7, 4, 5],
                         [1, BLACK_HOLE_SQUARE, 11, BLACK_HOLE_SQUARE, 8],
                         [6, 10, BLANK_SQUARE, 12, 15],
                         [9, BLACK_HOLE_SQUARE, 14, BLACK_HOLE_SQUARE, 20],
                         [13, 16, 17, 18, 19]])
            
            self.emptyCol = 3
            self.emptyRow = 3

    """
    Allows for post-initialization setting of state array.

    """
    def setStateGrid(self, grid):
        self.grid = grid
        return

    """
    Grabs the grid of this state. 

    """
    def getStateGrid(self):
        return self.grid

    """
    Determines the column and row values of a given integer item provided that it is in the
    state grid. If it is not in the state grid, then return -1, -1

    """
    def findPos(self, item):
        column = 1
        row = 1
        for line in self.grid:
            for num in line:
                if (num == item):
                    return row, column
                column = column + 1
            column = 1
            row = row + 1

        return -1, -1

class Search:

    def __init__(self, currState, goalState):
        self.currState = currState
        self.goalState = goalState

    """
    Allows for post-initialization setting of current state.

    """
    def setCurrentState(self, currState):
        self.currState = currState
        return


    """
    Uses Manhattan Distance heuristic to determine cost from given state to goal state

    """
    def distanceFromGoal(self, state):
        
        distance = 0
        
        for line in state.grid:
            for item in line:
                if (item == BLACK_HOLE_SQUARE or item == BLANK_SQUARE):
                    continue
                currRow, currColumn = state.findPos(item)
                goalRow, goalColumn = self.goalState.findPos(item)        
                distance = distance + (abs(goalRow - currRow) + abs(goalColumn - currColumn))
            
        return distance


    """
    Successor function for search. Determines what possible states we could be in given the
    current state.

    """
    def getNextStates(self):
        
        nextStates = []
        MAX_POSSIBLE_DIRECTIONS = 4
        
        # searches and switches boxes left; down; right; up;
        for direction in range(MAX_POSSIBLE_DIRECTIONS):
            if direction == 0:
                tempPos = (self.currState.emptyCol - 1, self.currState.emptyRow)
                if (self.isValidSwap(tempPos)):
                    nextStates.append(self.swap(tempPos))

            elif direction == 1:
                tempPos = (self.currState.emptyCol, self.currState.emptyRow + 1)
                if (self.isValidSwap(tempPos)):
                    nextStates.append(self.swap(tempPos))

            elif direction == 2:
                tempPos = (self.currState.emptyCol + 1, self.currState.emptyRow)
                if (self.isValidSwap(tempPos)):
                    nextStates.append(self.swap(tempPos))

            elif direction == 3:
                tempPos = (self.currState.emptyCol, self.currState.emptyRow - 1) 
                if (self.isValidSwap(tempPos)):
                    nextStates.append(self.swap(tempPos))


        return nextStates

    """
    Checks that the following conditions are met for the row and column value of potential
    location swap candidate:
        1.) Row value is not greater than the number of rows in the array, or less than 1
        2.) Column value is not greater than the number of columns in the array, or less than 1
        3.) Specified row/column pair is not a BLACK_HOLE_SQUARE, making it unmovable
    
        INPUT: swappingPos (tuple) - Column value as entry 0, Row value as entry 1
    """
    def isValidSwap(self, swappingPos):
        
        success = True

        if (swappingPos[1] > self.currState.grid.shape[1] or swappingPos[1] < 1):
            success = False

        elif (swappingPos[0] > self.currState.grid.shape[0] or swappingPos[0] < 1):
            success = False

        elif (self.currState.grid[swappingPos[1] - 1][swappingPos[0] - 1] == BLACK_HOLE_SQUARE):
            success = False

        return success

    """
    Takes current state and returns new state with swapped grid values between currState's 
    emptyPos and the given swappingPos. DOES NOT CHECK if this is valid. isValidSwap should
    be called before this to determine swap validity.

    """
    def swap(self, swappingPos):
        
        newState = deepcopy(self.currState) 

        # Holds item value at swapping position
        tempItem = newState.grid[swappingPos[1] - 1][swappingPos[0] - 1]

        newState.grid[newState.emptyRow - 1][newState.emptyCol - 1] = tempItem
        newState.grid[swappingPos[1] - 1][swappingPos[0] - 1] = 0

        newState.emptyCol = swappingPos[0]
        newState.emptyRow = swappingPos[1]

        return newState


    """
    Detects if the current state is the goal state.
    If distance to goal state is 0, returns True.
    If distance to goal state is not 0, returns False.

    """
    def goalReached(self):
        return (self.distanceFromGoal(self.currState) == 0)

def main():

    #set_trace()

    initState = State()
    goalGrid = np.array([[1, 2, 3, 4, 5],
                        [6, BLACK_HOLE_SQUARE, 7, BLACK_HOLE_SQUARE, 8],
                        [9, 10, BLANK_SQUARE, 11, 12],
                        [13, BLACK_HOLE_SQUARE, 14, BLACK_HOLE_SQUARE, 15],
                        [16, 17, 18, 19, 20]])
    goalState = State(goalGrid)

    stateCount = 0
    search = Search(initState, goalState)
    while not (search.goalReached()):
        successorStates = search.getNextStates()
        successorDistances = []
        for state in successorStates:
            successorDistances.append(search.distanceFromGoal(state))

        lowestDistance = np.inf 
        nextState = None
        for i, distance in enumerate(successorDistances):
            if (distance < lowestDistance):
                lowestDistance = distance
                nextState = successorStates[i]

        search.setCurrentState(nextState)

        print(search.currState.grid)
        stateCount = stateCount + 1

    print(stateCount)
    
if __name__ == "__main__":
    main()


        

