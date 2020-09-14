"""
### Author: Jacob Parmer, Auburn University
###
### Last Updated: September 14, 2020
"""

import numpy as np
import random as rnd
import math
import time
from copy import deepcopy
from pudb import set_trace

# consts defining queen spaces and empty spaces on chess board
EMPTY = 0
QUEEN = 1

class ChessBoard:

    """
    Creates a chessboard of size nxn, initializes all queens to the first row of the chessboard.
    Chessboard is represented as a 2d numpy array, empty spaces on the board are represented by
    the value "0", and spaces with queens are represented by the value "1"

    """
    def __init__(self, n = 25):

        if (n < 5):
            print("Size of chess board cannot be less than 5.")
            exit()

        self.n = n
        self.board = np.zeros((n,n), dtype=int) # inits board as empty
        queens = np.ones((n,), dtype=int)
        np.put(self.board, range(n), queens) # adds queens to the first row

    """
    Takes in a new numpy array as a chess board, and sets the object's board value accordingly. 
    This fails if the new board is not nxn.

    """
    def setBoard(self, newBoard):
        
        if (newBoard.shape()[0] != self.n):
            print(f"ERROR: New board was not set to be of size {self.n}. Exiting.")
            exit()

        self.board = newBoard
        return

    """
    Resets the board to a random state in which there is still 1 queen per column. 

    """
    def randomResetBoard(self):

        self.board = np.zeros((self.n, self.n), dtype=int)
        for x in range(self.n):
            randRow = rnd.randrange(self.n)
            np.put(self.board, ((randRow * self.n) + x), QUEEN)


class SimAnnealing:

    """
    Returns all possible board movements given the current state of the board and the
    limitation that queens may only move in their respective columns.

    """
    def getNeighbors(cb):
        neighbors = []

        for i, column in enumerate(cb.board.transpose()):
            for x in range(cb.n - 1):
                neighbor = deepcopy(cb)
                newCol = np.roll(column, x+1).transpose()

                neighbor.board[:,i] = newCol
                neighbors.append(neighbor)
                
        return neighbors
   

    """
    Determines the score heuristic for Simulated Annealing algorithm.
    Does this by detecting number of queens in board that are interfering with one another.

    """
    def getScore(cb):
        
        score = 0

        for i, row in enumerate(cb.board):
            for j, space in enumerate(row):

                if (cb.board[i][j] == QUEEN):

                    diag = np.diagonal(cb.board, offset=(j - i))        
                    idiag = np.fliplr(cb.board).diagonal(offset=(cb.n - j - 1) - i)

                    # checks for more than 1 queens in the row, in the diagonal, or in the
                    # inverse diagonal.
                    if (np.count_nonzero(row) > 1):
                        score = score + 1
                    elif (np.count_nonzero(diag) > 1):
                        score = score + 1
                    elif (np.count_nonzero(idiag) > 1):
                        score = score + 1

        return score

    def findNextState(cb, neighbors, temp):
        
        if (temp == 0): # if T = 0, return current board
            return cb

        nextBoard = ChessBoard(cb.n)
        index = rnd.randrange(len(neighbors)) # selects random neighbor from list
        nextBoard = neighbors[index]

        Edelta = SimAnnealing.getScore(cb) - SimAnnealing.getScore(nextBoard)

        if (Edelta > 0): # if the chosen board is better than the current one, return it
            return nextBoard 
        else: # if not, maybe return it
            Edelta = Edelta * 10 # Scaling Edelta so that probability works out to a good number
            probability = math.exp((Edelta / temp))
            val = rnd.random()
            if (val < probability):
                return nextBoard
            else:
                return cb

def main():

    #set_trace()

    n = 25
    temp = 12.0
    current_tries = 0
    
    cb = ChessBoard()
    finished = False

    start_time = time.time()

    while (temp >= 0):
        neighbors = SimAnnealing.getNeighbors(cb)
        cb = SimAnnealing.findNextState(cb, neighbors, temp)
        temp = temp - 0.001

    print(cb.board)
    print(SimAnnealing.getScore(cb))

    print(f"total time taken for Simulated Annealing: {time.time() - start_time} \n") 
if __name__ == "__main__":
    main()

