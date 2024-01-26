import random
import numpy as np

"""
This script provides various functions that facilitate the gameplay and analysis of tic-tac-toe.

The 'check_winner' function determines the status of the game: win for 'X' (-1), win for 'O' (1), draw (2), or game still ongoing (0).

'summarize_game_history' collates the win/loss/draw statistics from a set of played games stored in game_history.

'make_move' updates the game board with a valid move made by a player. The function checks whether the required cell is empty (0), if it is, the function updates the board with the player's move and returns True; if not, it simply returns False.

'get_human_move' is a function for human players, which prompts them for a cell to select for the next move. It ensures the input is valid (existing cell number and not already occupied).

'switch_player' is a simple utility function which switches between players after every turn - if the current player is 'X', it changes it to 'O' and vice versa.

'epsilon_greedy_move' implements the epsilon-greedy strategy for choosing the next move for an RL agent. It involves a trade-off between exploitation (using what the model has learnt to take the best move) and exploration (trying out random moves).

The exploration occurs with a probability of epsilon and results in a random selection of a valid cell. The exploitation move, on the other hand, involves prediction of the reward for each cell based on the current board state, and selection of the cell with the highest predicted reward.

'check_potential_win' is a helper function which checks if any of the players can potentially win in the next move i.e., they have two cells in a row, column or diagonal, and can win the game if they occupy the third one.
"""

# Function to check if there is a winner or draw
def check_winner(board):
    for i in range(3):
        # Check rows
        if abs(sum(board[i*3:(i+1)*3])) == 3:
            return board[i*3]
        # Check columns
        if abs(sum(board[i::3])) == 3:
            return board[i]
    # Check diagonals
    if abs(board[0] + board[4] + board[8]) == 3:
        return board[0]
    if abs(board[2] + board[4] + board[6]) == 3:
        return board[2]
    # Check for draw
    if 0 not in board:
        return 2  # Draw
    return 0  # Game ongoing

# Function to summarize the outcomes of games in the game history
def summarize_game_history(game_history):
    wins_for_X = 0
    wins_for_O = 0
    draws = 0

    for board_state, move in game_history:
        winner = check_winner(board_state)
        if winner == 1: 
            wins_for_X += 1
        elif winner == -1:
            wins_for_O += 1
        elif winner == 2:
            draws += 1

    return wins_for_X, wins_for_O, draws

# Function to update the board state with the player's move
def make_move(board, move, player):
    if board[move] == 0:
        board[move] = player
        return True
    return False

# Function for a human player
def get_human_move(board):
    valid_moves = [i for i in range(9) if board[i] == 0]
    move = None
    while move not in valid_moves:
        try:
            move = int(input("Enter your move (0-8): "))
            if move not in valid_moves:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return move

# Function to switch players between moves
def switch_player(player):
    return -player

# Function to select the next move using epsilon-greedy strategy
def epsilon_greedy_move_default(model, board, epsilon, show_text):
    if random.random() < epsilon:
        # Exploration: Choose a random move
        valid_moves = [i for i in range(9) if board[i] == 0]
        if show_text:
            print("\r\033[KAI is exploring: Chose a random move.", end='')
        return random.choice(valid_moves)
    else:
        # Exploitation: Choose the best move based on model prediction
        board_state = np.array([board])
        predictions = model.predict(board_state, verbose=0)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -1e7
        if show_text:
            print("\r\033[KAI is exploiting: Chose the best predicted move.", end='')
        return np.argmax(predictions)

def epsilon_greedy_move_value(model, board, player, epsilon, show_text):
    if random.random() < epsilon:
        # Exploration: Choose a random move
        valid_moves = [i for i in range(9) if board[i] == 0]
        if show_text:
            print("\r\033[KAI is exploring: Chose a random move.", end='')
        return random.choice(valid_moves)
    else:
        # Exploitation: Choose the best move based on model prediction
        best_move = None
        best_value = -float('inf')
        for i in range(9):
            if board[i] == 0:
                new_board = board.copy()
                new_board[i] = player
                board_state = np.array([new_board])
                predicted_value = model.predict(board_state, verbose=0)[0]
                if predicted_value > best_value:
                    best_value = predicted_value
                    best_move = i
        if show_text:
            print("\r\033[KAI is exploiting: Chose the best predicted move.", end='')
        return best_move if best_move is not None else random.choice([i for i in range(9) if board[i] == 0])


def check_potential_win(board, player):
    for i in range(3):
        # Check rows and columns for potential win
        if sum(board[i*3:(i+1)*3]) == 2 * player or sum(board[i::3]) == 2 * player:
            return True
        # Check diagonals for potential win
        if i == 0 and (board[0] + board[4] + board[8] == 2 * player or 
                       board[2] + board[4] + board[6] == 2 * player):
            return True
    return False