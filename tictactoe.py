import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import time
import platform
import argparse


# Set up the argument parser
parser = argparse.ArgumentParser(description='Run Tic-Tac-Toe game with optional visuals.')
parser.add_argument('--show-visuals', action='store_true', help='Enable game visuals')
parser.add_argument('--show-text', action='store_true', help='Enable game text')
parser.add_argument('--delay', action='store_true', help='add delay')
parser.add_argument('--human-player', type=str, choices=['X', 'O', 'None'], default='None', help='Play as a human player with X or O, or None for AI vs AI')
parser.add_argument('--games', type=int, default=10, help='Number of games to play')

args = parser.parse_args()

# Ensure interactive mode is on for live updating of plots
plt.ion()

figi, axi = plt.subplots()
figo, axo = plt.subplots()

# Define a function to visualize the input layer of the neural network
def visualize_input_layer(input_layer, game_number, wins_for_X, wins_for_O, draws):
    clear_output(wait=True)
    axi.clear()  # Clear the axes to remove old content

    # Reshape input layer to 3x3 grid to match Tic-Tac-Toe board
    input_grid = np.array(input_layer).reshape((3, 3))

    # Use a simple color map: empty = white, X = red, O = green
    color_map = {0: 'white', 1: 'red', -1: 'green'}
    for (i, j), value in np.ndenumerate(input_grid):
        color = color_map[value]
        rect = plt.Rectangle([j, 2 - i], 1, 1, color=color)  # Reverse the order of the row index
        axi.add_patch(rect)

        # Adding cell numbers as text annotations inside each cell
        cell_number = i * 3 + j
        axi.text(j + 0.5, 2.5 - i, str(cell_number), ha='center', va='center', color='blue', fontsize=12)

    # Add title and axis labels
    axi.set_title("Neural Network Input Layer")
    axi.set_xlabel("Column in Tic-Tac-Toe Board")
    axi.set_ylabel("Row in Tic-Tac-Toe Board")

    # Set aspect ratio to equal to make the plot square
    axi.set_aspect('equal', adjustable='box')
    axi.set_xlim(0, 3)
    axi.set_ylim(0, 3)

    # Center the tick labels
    axi.set_xticks(np.arange(0.5, 3, 1))
    axi.set_yticks(np.arange(0.5, 3, 1))
    axi.set_xticklabels(['0', '1', '2'])
    axi.set_yticklabels(['0', '1', '2'][::-1])

    # Additional Game Info
    info_text = f"Total: {game_number}, Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}"
    axi.text(0.5, -0.1, info_text, ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    # Render the plot
    plt.draw()
    plt.pause(0.01)  # Adjust the pause time as needed

# Define function to visualize activations in the output layer
def visualize_output_layer(output_layer_activation):
    clear_output(wait=True)
    axo.clear()  # Clear the axes to remove old content

    output_grid = output_layer_activation.reshape((3, 3))
    axo.imshow(output_grid, cmap='hot', interpolation='nearest')

    # Annotations
    axo.set_title("Neural Network Output Layer Activation")
    axo.set_xlabel("Column in Tic-Tac-Toe Board")
    axo.set_ylabel("Row in Tic-Tac-Toe Board")

    axo.set_aspect('equal', adjustable='box')
    axo.set_xticks(np.arange(0, 3, 1))
    axo.set_yticks(np.arange(0, 3, 1))

    # Adding value annotations on each cell
    for (i, j), value in np.ndenumerate(output_grid):
        axo.text(j, i, f'{value:.2f}', ha='center', va='center', color='gray')

    # Render the plot
    plt.draw()
    plt.pause(0.1)  # Adjust the pause time as needed

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

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

# Function to print the current state of the board
def print_board(board):
    symbols = {1: Fore.RED + 'X', -1: Fore.GREEN + 'O', 0: Style.RESET_ALL +' '}
    for i in range(3):
        print('\033[39m|'.join(symbols[board[i*3 + j]] for j in range(3)))
        if i < 2:
            print(Style.RESET_ALL + '-----')
    print(Style.RESET_ALL)
    print()

# Function to select the next move using epsilon-greedy strategy
def epsilon_greedy_move(model, board, epsilon):
    if random.random() < epsilon:
        valid_moves = [i for i in range(9) if board[i] == 0]
        return random.choice(valid_moves)
    else:
        board_state = np.array([board])
        predictions = model.predict(board_state, verbose=0)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -1e7
        return np.argmax(predictions)

# Function to update the neural network model with new game data
def update_model(model, batch_game_history):
    X_train = []  # Training data inputs
    y_train = []  # Training data outputs (targets)

    for game_history in batch_game_history:
        for board_state, move in game_history:
            target = np.zeros(9)
            winner = check_winner(board_state)
            # Set rewards/punishments based on game outcome
            if winner == 1:
                target[move] = 1  # smaller reward for win
            elif winner == -1:
                target[move] = -1  # penalty for loss
            elif winner == 2:
                target[move] = 0.5  # small reward for draw
            X_train.append(board_state)
            y_train.append(target)

    model.fit(np.array(X_train), np.array(y_train), verbose=0, batch_size=32)

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

def simulate_game_and_train(model, epsilon):
    board = [0]*9
    player = starting_player
    global game_history
    global wins_for_X, wins_for_O, draws

    current_game_history = []  # List to store moves of the current game

    while True:
        if args.show_text:
            clear_screen()
            print(f"Total: {game_number}, Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}\n")
            print(f"Starting Player: {Fore.RED + 'X' if starting_player == 1 else Fore.GREEN + 'O'}" + Style.RESET_ALL)
            print("Player", 'O' if player == -1 else 'X', "'s turn")
            print()
            print_board(board)

        board_state = np.array([board])
        predictions = model.predict(board_state, verbose=0)[0]
        if args.show_visuals:
            visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
            visualize_output_layer(predictions)

        # Determine the move based on player type
        if (args.human_player == 'X' and player == 1) or (args.human_player == 'O' and player == -1):
            move = get_human_move(board)
        else:
            # Use epsilon-greedy strategy for move selection
            move = epsilon_greedy_move(model, board, epsilon)

        #board_state = np.array([board])
        #predictions = model.predict(board_state, verbose=0)[0]
        #if args.show_visuals:
        #    visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
        #    visualize_output_layer(predictions)

        if args.delay:
            time.sleep(1)  # Pauses the program

        # Make the move
        valid_move_made = make_move(board, move, player)
        if not valid_move_made:
            continue  # Skip the rest if the move was invalid

        # Record the move for training
        current_game_history.append((board.copy(), move))
        
        # Check for game end
        winner = check_winner(board)
        if winner != 0:

            # Update counters
            if winner == 1:
                wins_for_X += 1
            elif winner == -1:
                wins_for_O += 1
            elif winner == 2:
                draws += 1

            if args.show_text:
                clear_screen()
                print(f"Total: {game_number}, Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}\n")
                print(f"Starting Player: {Fore.RED + 'X' if starting_player == 1 else Fore.GREEN + 'O'}" + Style.RESET_ALL)
                print("Player", 'O' if player == -1 else 'X', "'s turn")
                print()
                print_board(board)
            if args.show_visuals:
                visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
                visualize_output_layer(predictions)

            # Print winner
            print(f"Game {game_number}: Winner - {Fore.RED + 'X' if winner == 1 else Fore.GREEN + 'O' if winner == -1 else 'Draw'}" + Style.RESET_ALL)
            
            return current_game_history  # Return the history of this game
 
        player = switch_player(player)

# Neural network model with linear output layer activation
if os.path.exists('tic_tac_toe_model.keras'):
    model = tf.keras.models.load_model('tic_tac_toe_model.keras')
    print("Model loaded successfully.")
    model.summary()  # Print the summary of the model

else:
    # Define and compile the model as before if it doesn't exist
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(9,)),
        layers.Dropout(0.1),  # Dropout layer
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),  # Another dropout layer
        layers.Dense(9, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("New model created.")
    model.summary()  # Print the summary of the new model

# Just a sleep show you can read the model summary
time.sleep(5)  # Pauses the program

# Train the model over multiple games
starting_player = 1  # Start with 'X' in the first game
#n_games = 1000
n_games = args.games

# Initialize counters
wins_for_X = 0
wins_for_O = 0
draws = 0

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.99
epsilon = epsilon_start

batch_size = 10  # Define the number of games after which model will be updated
batch_game_history = []

for game_number in range(1, n_games + 1):
    current_game_history = simulate_game_and_train(model, epsilon)
    batch_game_history.append(current_game_history)  # Append the history of the current game

    # Check if it's time to update the model
    if game_number % batch_size == 0 or game_number == n_games:
        print(f"update model")
        update_model(model, batch_game_history)
        batch_game_history = []  # Reset for the next batch

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Switch starting player for the next game
    starting_player = -starting_player

    if args.delay:
        time.sleep(5)  # Pauses the program
    
    if (args.human_player == 'X') or (args.human_player == 'O'):
        time.sleep(5)  # Pauses the program
   
model.save('tic_tac_toe_model.keras')  # Saves the model in Keras format
