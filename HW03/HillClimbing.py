"""
### Author: Jacob Parmer, Auburn University
###
### Last Updated: September 12, 2020
"""

import numpy as np
import random as rnd
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
        self.updateScore() 

    """
    Determines the score heuristic for hill climbing algorithm. Does this by detecting number of
    queens in board that are interfering with one another.

    """
    def updateScore(self):
        
        score = 0

        for i, row in enumerate(self.board):
            for j, space in enumerate(row):

                if (self.board[i][j] == QUEEN):

                    diag = np.diagonal(self.board, offset=(j - i))        
                    idiag = np.fliplr(self.board).diagonal(offset=(self.n - j - 1) - i)

                    # checks for more than 1 queens in the row, in the diagonal, or in the
                    # inverse diagonal.
                    if (np.count_nonzero(row) > 1):
                        score = score + 1
                    elif (np.count_nonzero(diag) > 1):
                        score = score + 1
                    elif (np.count_nonzero(idiag) > 1):
                        score = score + 1

        self.score = score
        return

    def setBoard(self, newBoard):
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

        self.updateScore()

class nQueens:

    """
    Returns all possible board movements given the current state of the board and the
    limitation that queens may only move in their respective columns.

    """
    def getNeighbors(chessBoard, n):
        neighbors = []

        for i, column in enumerate(chessBoard.board.transpose()):
            for x in range(n - 1):
                neighbor = deepcopy(chessBoard)
                newCol = np.roll(column, x+1).transpose()

                neighbor.board[:,i] = newCol
                neighbor.updateScore()
                neighbors.append(neighbor)
                
        return neighbors

def main():

    #set_trace()

    n = 25
    MAX_TRIES = 100 
    current_tries = 0
    
    # Hill climbing without random restarts

    board = ChessBoard(n)

    while (board.score != 0 and current_tries < MAX_TRIES):
        neighbors = nQueens.getNeighbors(board, n)
        for neighbor in neighbors:
            if (neighbor.score <= board.score):
                board = neighbor

        current_tries = current_tries + 1
                
    print(board.board)
    print(board.score)

    # Hill climbing with random restarts

    board2 = ChessBoard(n)
    current_tries = 0

    while (board2.score != 0):
        if (current_tries > MAX_TRIES):
            board2.randomResetBoard()
            current_tries = 0

        neighbors2 = nQueens.getNeighbors(board2, n)
        for neighbor2 in neighbors2:
            if (neighbor2.score <= board2.score):
                board2 = neighbor2

        current_tries = current_tries + 1

    print(board2.board)
    print(board2.score)

if __name__ == "__main__":
    main()

