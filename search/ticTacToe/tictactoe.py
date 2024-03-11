"""
Tic Tac Toe Player
"""

import math
import copy


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count_x = 0
    count_o = 0
    
    # if number of x's on board are less or equal than number of o's then return true
    # remember - the first turn is of x
    for i in range(len(board)):
        for j in range(len(board)):
            if j == X:
                count_x+1
            elif j == O:
                count_o+=1
    
    if count_x <= count_o:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = []
    
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == None:
                possible_actions.append(i, j)

    return possible_actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    
    player = player(board)
    
    new_board = copy.deepcopy(board)
    
    new_board[action[0]][action[1]] = player

    return new_board
    
    

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    for i in range(3):
        # if the elements are the same vertically
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            return board[i][0]
        # if elements are same horizontally
        elif board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
            return board[0][i]
    
    # if elements are same diagonally
    if board[0][0] == board[1][1] == board[2][2] and board[2][2] is not None:
        return board[2][2]
    elif board[0][2] == board[1][1] == board[2][0] and board[2][0] is not None:
        return board[2][0]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    
    empty_spaces = False
    
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == None:
                empty_spaces = True
                break
        if empty_spaces == True:
            break
        
    if winner(board):
        return False
    elif empty_spaces:
        return True
        


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    winner = winner(board)
    
    if winner == X:
        return 1
    elif winner == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    if terminal(board):
        return None
    else:
        print("IN MINImAx")