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

def epsilon_greedy_move(model, board, epsilon):
    if random.random() < epsilon:
        valid_moves = [i for i in range(9) if board[i] == 0]
        return random.choice(valid_moves)
    else:
        board_state = np.array([board])
        predictions = model.predict(board_state)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -1e7
        return np.argmax(predictions)

def update_model(model, game_history, winner):
    for board_state, move in game_history:
        target = np.zeros(9)
        if winner == 1:
            target[move] = 1.0
        elif winner == -1:
            target[move] = -1.0
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

def simulate_game_and_train(model, epsilon):
    board = [0]*9
    player = starting_player
    global game_history
    global wins_for_X
    global wins_for_O
    global draws

    while True:

        clear_screen()
        print("Player", 'O' if player == -1 else 'X', "'s turn")
        print()
        print_board(board)
        time.sleep(0.5)  # Pauses the program
    
        # Use epsilon-greedy strategy for move selection
        move = epsilon_greedy_move(model, board, epsilon)

        # Make the move
        valid_move_made = make_move(board, move, player)
        if not valid_move_made:
            continue  # Skip the rest if the move was invalid

        # Record the move for training
        game_history.append((board.copy(), move))

        # Check for game end
        winner = check_winner(board)
        if winner != 0:
            clear_screen()
            print("Player", 'O' if player == -1 else 'X', "'s turn")
            print()
            print_board(board)
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
            
            update_model(model, game_history, winner)
            return winner
        player = switch_player(player)

# Neural network model with linear output layer activation
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(9,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='linear')  # Linear activation for output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

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
n_games = 5

# Initialize counters
wins_for_X = 0
wins_for_O = 0
draws = 0

# Main training loop
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
epsilon = epsilon_start

for game_number in range(1, n_games + 1):
    winner = simulate_game_and_train(model, epsilon)

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Switch starting player for the next game
    starting_player = -starting_player

# Save the game history using pickle
with open('game_history.pkl', 'wb') as f:
    pickle.dump(game_history, f)