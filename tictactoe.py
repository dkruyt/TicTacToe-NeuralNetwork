import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from colorama import Fore, Back, Style

import os
import time
import platform

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

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

def make_move(board, move, player):
    if board[move] == 0:
        board[move] = player
        return True
    return False

def switch_player(player):
    return -player

# Board printing function
def print_board(board):
    symbols = {1: Fore.RED + 'X', -1: Fore.GREEN + 'O', 0: Style.RESET_ALL +' '}
    for i in range(3):
        print('\033[39m|'.join(symbols[board[i*3 + j]] for j in range(3)))
        if i < 2:
            print(Style.RESET_ALL + '-----')
    print(Style.RESET_ALL)
    print()

def update_model(model, game_history, winner):
    for board_state, move in game_history:
        target = np.zeros(9)
        if winner == 1:  # If 'X' won
            target[move] = 1.0  # Reinforce the move if it was made by 'X'
        elif winner == -1:  # If 'O' won
            target[move] = -1.0  # Penalize the move if it was made by 'X'

        model.fit(np.array([board_state]), np.array([target]), verbose=0, batch_size=32)


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

def simulate_game_and_train(model):
    board = [0]*9
    player = starting_player  # Set starting player based on the parameter
    global game_history
    #game_history = []

    while True:
        valid_move_made = False
        while not valid_move_made:
            #clear_screen()
            #print_board(board)
            #time.sleep(0.1)  # Pauses the program
            #print("Player", 'O' if player == -1 else 'X', "'s turn")

            # Get model's move (using random move for untrained model)
            valid_moves = [i for i in range(9) if board[i] == 0]
            if not valid_moves:
                #clear_screen()
                print_board(board)
                print(f"Winner: DRAW")
                return 2  # Draw, no more valid moves
            move = random.choice(valid_moves)

            # Make the move
            valid_move_made = make_move(board, move, player)

        # Record the move for training
        game_history.append((board.copy(), move))

        # Check for game end
        winner = check_winner(board)
        if winner != 0:
            #clear_screen()
            print_board(board)
            print(f"Winner: {'X' if player == 1 else 'O' if player == -1 else 'Draw'}")
            update_model(model, game_history, winner)
            return winner  # 1 or -1 for a player win, 2 for draw
        player = switch_player(player)

# Neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(9,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='softmax')  # Output layer with one node for each board position
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Initialize counters
wins_for_X = 0
wins_for_O = 0
draws = 0

# Load training data if exists
if os.path.exists('game_history.pkl'):
    with open('game_history.pkl', 'rb') as f:
        game_history = pickle.load(f)
    
    wins_for_X, wins_for_O, draws = summarize_game_history(game_history)
    print(f"Loaded training data history")
    print(f"Summary: Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}")
else:
    print("Initializing new training data")
    game_history = []

    winner = []

# Train the model over multiple games
starting_player = 1  # Start with 'X' in the first game
n_games = 250
for game_number in range(1, n_games + 1):
    winner = simulate_game_and_train(model)

    # Update counters
    if winner == 1:
        wins_for_X += 1
    elif winner == -1:
        wins_for_O += 1
    elif winner == 2:
        draws += 1

    # Print game number and statistics
    print(f"Starting Player: {Fore.RED + 'X' if starting_player == 1 else Fore.GREEN + 'O'}" + Style.RESET_ALL)
    print(f"Game {game_number}: Winner - {Fore.RED + 'X' if winner == 1 else Fore.GREEN + 'O' if winner == -1 else 'Draw'}" + Style.RESET_ALL)
    print(f"Total: {game_number}, Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}\n")
    #time.sleep(2)  # Pauses the program


    # Switch starting player for the next game
    starting_player = -starting_player

# Save the game history using pickle
with open('game_history.pkl', 'wb') as f:
    pickle.dump(game_history, f)