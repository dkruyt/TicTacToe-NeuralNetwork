import random
import numpy as np

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
def epsilon_greedy_move(model, board, epsilon):
    if random.random() < epsilon:
        # Exploration: Choose a random move
        valid_moves = [i for i in range(9) if board[i] == 0]
        #print("AI is exploring: Chose a random move.")
        return random.choice(valid_moves)
    else:
        # Exploitation: Choose the best move based on model prediction
        board_state = np.array([board])
        predictions = model.predict(board_state, verbose=0)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -1e7
        #print("AI is exploiting: Chose the best predicted move.")
        return np.argmax(predictions)

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