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
parser.add_argument('--show-visuals', action='store_true', 
                    help='Enable game visuals (default: False)')
parser.add_argument('--show-text', action='store_true', 
                    help='Enable game text (default: False)')
parser.add_argument('--delay', action='store_true', 
                    help='Add delay (default: False)')
parser.add_argument('--human-player', type=str, choices=['X', 'O', 'None'], default='None', 
                    help='Play as a human player with X or O, or None for AI vs AI (default: None)')
parser.add_argument('--games', type=int, default=10, 
                    help='Number of games to play (default: 10)')
parser.add_argument('--model-name', type=str, default='tic_tac_toe_model.keras', 
                    help='Filename for saving/loading the model (default: tic_tac_toe_model.keras)')
parser.add_argument('--dense-units', type=int, default=32, 
                    help='Number of units in the Dense layers (default: 32)')
parser.add_argument('--dropout-rate', type=float, default=0.1, 
                    help='Dropout rate for the Dropout layers (default: 0.1)')
parser.add_argument('--epsilon-start', type=float, default=1.0, 
                    help='Starting value of epsilon for epsilon-greedy strategy (default: 1.0)')
parser.add_argument('--epsilon-end', type=float, default=0.1, 
                    help='Ending value of epsilon for epsilon-greedy strategy (default: 0.1)')
parser.add_argument('--epsilon-decay', type=float, default=0.99, 
                    help='Decay rate of epsilon after each game (default: 0.99)')

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
    info_text = f"Round: {game_number}, Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}"
    axi.text(0.5, -0.1, info_text, ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    # Render the plot
    plt.draw()
    plt.pause(0.01)  # Adjust the pause time as needed

# Define function to visualize activations in the output layer
def visualize_output_layer(output_layer_activation, board, colormap='autumn'):
    clear_output(wait=True)
    axo.clear()  # Clear the axes to remove old content

    output_grid = output_layer_activation.reshape((3, 3))
    heatmap = axo.imshow(output_grid, cmap=colormap, interpolation='nearest')

    # Annotations
    axo.set_title("Neural Network Output Layer Activation")
    axo.set_xlabel("Column in Tic-Tac-Toe Board")
    axo.set_ylabel("Row in Tic-Tac-Toe Board")

    axo.set_aspect('equal', adjustable='box')
    axo.set_xticks(np.arange(0, 3, 1))
    axo.set_yticks(np.arange(0, 3, 1))

    # Adding value annotations on each cell
    for (i, j), value in np.ndenumerate(output_grid):
        if board[i * 3 + j] != 0:  # Check if the move is taken
            text_color = 'lightgray'  # Gray out the text for taken moves
        else:
            text_color = 'blue'  # Use a different color for available moves
        axo.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)

    # Render the plot
    plt.draw()
    plt.pause(0.1)  # Adjust the pause time as needed

# Global variable to keep track of figures
weight_figures = {}

def visualize_model_weights_and_biases(model):
    global weight_figures

    for i, layer in enumerate(model.layers):
        weights_biases = layer.get_weights()
        if len(weights_biases) > 0:
            weights, biases = weights_biases

            # Visualize Weights
            if i not in weight_figures:
                weight_figures[i] = plt.figure(figsize=(12, 4))

            plt.figure(weight_figures[i].number)
            plt.clf()  # Clear the current figure

            plt.subplot(1, 2, 1)
            plt.imshow(weights, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"Weights of Layer {i+1}: {layer.name}")
            plt.xlabel('Neurons in the following layer')
            plt.ylabel('Neurons in the current layer')

            # Visualize Biases
            plt.subplot(1, 2, 2)
            plt.plot(biases)
            plt.title(f"Biases of Layer {i+1}: {layer.name}")
            plt.xlabel('Neurons')
            plt.ylabel('Bias Value')

            plt.draw()
            plt.pause(0.001)  # Pause to update the figure

# Global variables for the figure and axes
global nn_fig, nn_ax

def visualize_detailed_network(model, input_data, output_data):
    global nn_fig, nn_ax

    # Determine layer sizes
    layer_sizes = [input_data.shape[1]] + \
                  [layer.units for layer in model.layers if hasattr(layer, 'units')] + \
                  [output_data.shape[1]]
    
    # Create or clear the figure and axes
    if 'nn_fig' not in globals():
        nn_fig, nn_ax = plt.subplots(figsize=(12, 8))
    else:
        nn_ax.clear()

    n_layers = len(layer_sizes)
    v_spacing = (1.0 / float(max(layer_sizes))) * 0.8
    h_spacing = 0.8 / float(n_layers - 1)

    # Input-Arrows
    input_arrows_x = np.linspace(0, 0.1, input_data.shape[1])
    input_arrows_y = np.linspace(0, 1, input_data.shape[1], endpoint=False) + v_spacing / 2.
    for i, y in zip(input_data[0], input_arrows_y):
        nn_ax.arrow(0, y, 0.1, 0, head_width=0.02, head_length=0.02, fc='green', ec='green')
        nn_ax.text(-0.05, y, f'{i:.2f}', ha='right', va='center', fontsize=10)

    # Neurons and Connections
    for n, layer_size in enumerate(layer_sizes):
        layer_x = n * h_spacing
        layer_y = np.linspace(0, 1, layer_size, endpoint=False) + v_spacing / 2.
        for i, neuron_y in enumerate(layer_y):
            circle = plt.Circle((layer_x, neuron_y), v_spacing/4., color='w', ec='k', zorder=4)
            nn_ax.add_artist(circle)

            if n > 0:  # Not input layer
                for prev_neuron_y in np.linspace(0, 1, layer_sizes[n - 1], endpoint=False) + v_spacing / 2.:
                    line = plt.Line2D([layer_x - h_spacing, layer_x], [prev_neuron_y, neuron_y], c='gray')
                    nn_ax.add_artist(line)

    # Output-Values
    if output_data is not None:
        output_arrows_x = np.linspace(1 - 0.1, 1, output_data.shape[1])
        output_arrows_y = np.linspace(0, 1, output_data.shape[1], endpoint=False) + v_spacing / 2.
        for i, y in zip(output_data[0], output_arrows_y):
            nn_ax.arrow(1 - 0.1, y, 0.1, 0, head_width=0.02, head_length=0.02, fc='red', ec='red')
            nn_ax.text(1.05, y, f'{i:.2f}', ha='left', va='center', fontsize=10)

    nn_ax.axis('off')
    plt.show()
    plt.pause(0.001)  # Pause to update the figure

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

# Function to assign improved rewards based on game outcome
def assign_rewards(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5

    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward
        game_history[i] = (board_state, target)
        current_reward *= decay_factor

# Update the model training function
def update_model(model, batch_game_history):
    X_train = []
    y_train = []

    for game_history in batch_game_history:
        assign_rewards(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        for board_state, target in game_history:
            X_train.append(board_state)
            y_train.append(target)

    model.fit(np.array(X_train), np.array(y_train), verbose=1, batch_size=32)

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
        predictions = model.predict(board_state, verbose=0)
        if args.show_visuals:
            visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
            visualize_output_layer(predictions, board)

            visualize_detailed_network(model, board_state , predictions)

        # Determine the move based on player type
        if (args.human_player == 'X' and player == 1) or (args.human_player == 'O' and player == -1):
            move = get_human_move(board)
        else:
            if (args.human_player == 'X') or (args.human_player == 'O'):
                visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
                time.sleep(3)  # Pauses the program
             # Use epsilon-greedy strategy for move selection  
            move = epsilon_greedy_move(model, board, epsilon)

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
                visualize_output_layer(predictions, board)

            # Print winner
            print(f"Game {game_number}: Winner - {Fore.RED + 'X' if winner == 1 else Fore.GREEN + 'O' if winner == -1 else 'Draw'}" + Style.RESET_ALL)
            
            return current_game_history  # Return the history of this game
 
        player = switch_player(player)

# Neural network model with linear output layer activation
if os.path.exists(args.model_name):
    model = tf.keras.models.load_model(args.model_name)
    print("Model loaded successfully.")
    model.summary()  # Print the summary of the model

else:
    # Define and compile the model as before if it doesn't exist
    model = keras.Sequential([
        layers.Dense(args.dense_units, activation='relu', input_shape=(9,)),
        layers.Dropout(args.dropout_rate),
        layers.Dense(args.dense_units, activation='relu'),
        layers.Dropout(args.dropout_rate),
        layers.Dense(9, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("New model created.")
    model.summary()  # Print the summary of the new model

initial_weights = model.get_weights()

# After model is created or loaded
visualize_model_weights_and_biases(model)

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

epsilon_start = args.epsilon_start
epsilon_end = args.epsilon_end
epsilon_decay = args.epsilon_decay
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
        visualize_model_weights_and_biases(model)
        batch_game_history = []  # Reset for the next batch

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Switch starting player for the next game
    starting_player = -starting_player

    if args.delay:
        time.sleep(5)  # Pauses the program
    
    if (args.human_player == 'X') or (args.human_player == 'O'):
        time.sleep(3)  # Pauses the program

# Get new weights after training
new_weights = model.get_weights()

# Compare the initial and new weights
for initial, new in zip(initial_weights, new_weights):
    if not np.array_equal(initial, new):
        print("Weights have changed after training.")
        break
else:
    print("Weights have not changed after training.")

# Optionally, you can quantify the change in weights using a metric like mean absolute difference
weight_changes = [np.mean(np.abs(w_new - w_initial)) for w_initial, w_new in zip(initial_weights, new_weights)]
print("Mean absolute changes in weights per layer:", weight_changes)

model.save(args.model_name)  # Saves the model in Keras format
