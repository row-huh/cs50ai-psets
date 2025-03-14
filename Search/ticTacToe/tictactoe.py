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
            if board[i][j] == X:
                count_x += 1
            elif board[i][j] == O:
                count_o += 1
    
    if count_x <= count_o:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    
    # add the indices of all empty spacees to the set of possible actions
    for i in range(3):
        for j in range(3):
            if board[i][j] == None:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    
    # if any element in the tuple of action is negative - raise error
    if (action[0] < 0 or action[1] < 0):
        raise IndexError("Negative Indexes are not allowed")
    
    current_player = player(board)
    new_board = copy.deepcopy(board)
    
    # if the index denoted by action tuple is empty --> add the player's symbol at that location
    if new_board[action[0]][action[1]] == None:
        new_board[action[0]][action[1]] = current_player
    else:
        # Otherwise, raise invalid move exception
        raise Exception(
            f"Invalid Action - {(action[0],action[1])} already has an element {new_board[action[0]][action[1]]}")

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
    
    # if there is any empty space in the board, empty_spaces variable is True, otherwise False
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == None:
                empty_spaces = True
                break
        if empty_spaces == True:
            break
        
    if winner(board):
        return True
    elif empty_spaces:
        return False
    else:
        return True
        

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    winning_player = winner(board)
    
    if winning_player == X:
        return 1
    elif winning_player == O:
        return -1
    else:
        return 0


best_action = tuple()


def minimax(board):
    """
    Returns the optimal action (tuple of two indexes) for the current player on the board.
    """
    alpha = float('-inf')
    beta = float('inf')
    
    if player(board) == X:
        # Return the action associated with the maximum utility
        return max_value(board, alpha, beta)[1]  
    else:
        # Return the action associated with the minimum utility
        return min_value(board, alpha, beta)[1]  


def max_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None  # Return utility value and no action
    v = float('-inf')
    best_action = None
    for action in actions(board):
        new_v, _ = min_value(result(board, action), alpha, beta)
        if new_v > v:
            v = new_v
            best_action = action
        alpha = max(alpha, v)
        
        if beta <= alpha:
            break
    return v, best_action


def min_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None  # Return utility value and no action
    v = float('inf')
    best_action = None
    for action in actions(board):
        new_v, _ = max_value(result(board, action), alpha, beta)
        if new_v < v:
            v = new_v
            best_action = action
        beta = min(beta, v)
        if beta <= alpha:
            break    
    return v, best_action