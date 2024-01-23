import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
                    help='Number of Neurons in the Dense layers (default: 32)')
parser.add_argument('--dropout-rate', type=float, default=0.2, 
                    help='Dropout rate for the Dropout layers (default: 0.2)')
parser.add_argument('--epsilon-start', type=float, default=1.0, 
                    help='Starting value of epsilon for epsilon-greedy strategy (default: 1.0)')
parser.add_argument('--epsilon-end', type=float, default=0.1, 
                    help='Ending value of epsilon for epsilon-greedy strategy (default: 0.1)')
parser.add_argument('--epsilon-decay', type=float, default=0.99, 
                    help='Decay rate of epsilon after each game (default: 0.99)')
parser.add_argument('--model-type', type=str, choices=['MLP', 'CNN', 'RNN'], default='MLP', 
                    help='Type of model to use (MLP, CNN, RNN) (default: MLP)')

args = parser.parse_args()

def print_tensorflow_info():
    print(f"TensorFlow Version: {tf.__version__}")
    devices = tf.config.list_physical_devices()
    if not devices:
        print("No physical devices found.")
    else:
        print("Found the following physical devices:")
        for idx, device in enumerate(devices):
            print(f"  Device {idx + 1}:")
            print(f"    Type: {device.device_type}")
            print(f"    Name: {device.name}")
            if device.device_type == 'GPU':
                details = tf.config.experimental.get_device_details(device)
                print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
                print(f"    Memory: {device.memory_limit_bytes / (1024**3):.2f} GB")

print_tensorflow_info()

# Conditional execution based on --show-visuals argument
if (args.show_visuals) or (args.human_player):
    # Ensure interactive mode is on for live updating of plots
    plt.ion()
    figi, axi = plt.subplots()
    if (args.show_visuals):
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
            if type(layer) != layers.SimpleRNN:
                weights, biases = weights_biases

                if len(weights.shape) == 4:  # Convolutional layer
                    n_filters = weights.shape[3]
                    # Check if figure exists
                    if i not in weight_figures:
                        weight_figures[i], axes = plt.subplots(1, n_filters, figsize=(n_filters * 2, 2))
                    else:
                        axes = weight_figures[i].axes  # Get existing axes
                    for j in range(n_filters):
                        filter_weights = weights[:, :, :, j]
                        filter_weights = np.squeeze(filter_weights)

                        ax = axes[j] if n_filters > 1 else axes
                        ax.clear()  # Clear existing plot
                        ax.imshow(filter_weights, aspect='auto', cmap='viridis')
                        ax.set_title(f'Filter {j+1}')
                        ax.axis('off')

                else:  # Other layers (like Dense)
                    if i not in weight_figures:
                        weight_figures[i] = plt.figure(figsize=(12, 4))

                    plt.figure(weight_figures[i].number)
                    plt.clf()  

                    plt.subplot(1, 2, 1)
                    plt.imshow(weights, aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Weights of Layer {i+1}: {layer.name}")
                    plt.xlabel('Neurons in the following layer')
                    plt.ylabel('Neurons in the current layer')

                    plt.subplot(1, 2, 2)
                    plt.plot(biases)
                    plt.title(f"Biases of Layer {i+1}: {layer.name}")
                    plt.xlabel('Neurons')
                    plt.ylabel('Bias Value')

            elif type(layer) == layers.SimpleRNN:
                weights = weights_biases[0][:, :layer.units]
                recurrent_weights = weights_biases[0][:, layer.units:]
                biases = weights_biases[1]

                if i not in weight_figures:
                    weight_figures[i] = plt.figure(figsize=(18, 6))

                plt.figure(weight_figures[i].number)
                plt.clf()

                plt.subplot(1, 3, 1)
                plt.imshow(weights, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f"Input Weights of Layer {i+1}: {layer.name}")
                plt.xlabel('Units')
                plt.ylabel('Input Features')

                plt.subplot(1, 3, 2)
                plt.imshow(recurrent_weights, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f"Recurrent Weights of Layer {i+1}: {layer.name}")
                plt.xlabel('Units')
                plt.ylabel('Units')

                plt.subplot(1, 3, 3)
                plt.plot(biases)
                plt.title(f"Biases of Layer {i+1}: {layer.name}")
                plt.xlabel('Units')
                plt.ylabel('Bias Value')

            plt.draw()
            plt.pause(0.001)  # Pause to update the figure


# Global variables for the figure and axes
global nn_fig, nn_ax

def visualize_detailed_network(model, input_data, output_data):
    global nn_fig, nn_ax

    # Initialize layer_sizes with the size of the input data
    layer_sizes = [np.prod(input_data.shape[1:])]  

    for layer in model.layers:
        if hasattr(layer, 'units'):  # For Dense layers
            layer_sizes.append(layer.units)
        elif isinstance(layer, keras.layers.Conv2D):  # For Conv2D layers
            # Add the number of filters as the size of the layer
            layer_sizes.append(layer.filters)
        elif isinstance(layer, keras.layers.Flatten) or isinstance(layer, keras.layers.Reshape):
            # For Flatten/Reshape layers, compute the size based on output shape
            layer_sizes.append(np.prod(layer.output_shape[1:]))
        else:
            # For other layer types (like Dropout), ignore them in size calculation
            continue  # Skip adding to layer_sizes

    # Add the size of the output data
    layer_sizes.append(np.prod(output_data.shape[1:]))
    
    # Create or clear the figure and axes
    if 'nn_fig' not in globals():
        nn_fig, nn_ax = plt.subplots(figsize=(12, 8))
    else:
        nn_ax.clear()

    n_layers = len(layer_sizes)
    v_spacing = (1.0 / float(max(layer_sizes))) * 0.8
    h_spacing = 0.8 / float(n_layers - 1)

    # Define the rainbow colormap
    rainbow = plt.colormaps.get_cmap('winter')

    # Layer colors
    layer_colors = ['green', 'blue', 'purple', 'pink', 'red']

    # Input-Arrows and Symbols
    for i, y in zip(input_data[0], np.linspace(0, 1, input_data.shape[1], endpoint=False) + v_spacing / 2.):
        nn_ax.arrow(-0.10, y, 0.05, 0, head_width=0.02, head_length=0.02, fc='green', ec='green')
        
        # Display the input value as an integer
        input_value = int(i)
        nn_ax.text(-0.12, y, f'{input_value}', ha='right', va='center', fontsize=10)

        # Conditional symbols next to the input value
        if i == 1.0:
            nn_ax.text(-0.17, y, 'X', ha='left', va='center', fontsize=20, color='red')
        elif i == -1.0:
            nn_ax.text(-0.17, y, 'O', ha='left', va='center', fontsize=20, color='green')

    # Neurons and Connections
    for n, layer_size in enumerate(layer_sizes):
        layer_x = n * h_spacing
        if layer_size > 16:
            displayed_neurons = 16
            middle_neurons = [6, 7, 8, 9]
        else:
            displayed_neurons = layer_size
            middle_neurons = []

        for i, neuron_y in enumerate(np.linspace(0, 1, displayed_neurons, endpoint=False) + v_spacing / 2.):
            neuron_color = 'white' if i in middle_neurons and layer_size > 16 else layer_colors[n % len(layer_colors)]
            circle = plt.Circle((layer_x, neuron_y), v_spacing/2. * 1.5, color=neuron_color, ec='k', zorder=4)
            nn_ax.add_artist(circle)

            if n > 0:
                for j, prev_neuron_y in enumerate(np.linspace(0, 1, layer_sizes[n - 1], endpoint=False) + v_spacing / 2.):
                    color = rainbow(float(i + j) / (displayed_neurons + layer_sizes[n - 1]))
                    line = plt.Line2D([layer_x - h_spacing, layer_x], [prev_neuron_y, neuron_y], c=color, alpha=0.7)
                    nn_ax.add_artist(line)

    # Output-Values
    for i, y in zip(output_data[0], np.linspace(0, 1, output_data.shape[1], endpoint=False) + v_spacing / 2.):
        nn_ax.arrow(1 - 0.18, y, 0.05, 0, head_width=0.02, head_length=0.02, fc='red', ec='red')
        nn_ax.text(0.90, y, f'{i:.2f}', ha='left', va='center', fontsize=10)

    # Adding layer names and neuron counts to the visualization
    for n, layer in enumerate(model.layers):
        layer_x = n * h_spacing
        layer_name = layer.name
        nn_ax.text(layer_x, 1.05, layer_name, ha='center', va='center', fontsize=12)
        neuron_count = layer_sizes[n]
        nn_ax.text(layer_x, 1.02, f'({neuron_count} neurons)', ha='center', va='center', fontsize=10)

    nn_ax.axis('off')
    plt.show()
    plt.pause(0.001)  # Pause to update the figure

global stats_fig, stats_ax, stats_bars

def plot_game_statistics(wins_for_X, wins_for_O, draws):
    global stats_fig, stats_ax

    labels = ['Wins for X', 'Wins for O', 'Draws']
    values = [wins_for_X, wins_for_O, draws]
    colors = ['red', 'green', 'yellow']

    if 'stats_fig' not in globals():
        stats_fig, stats_ax = plt.subplots(figsize=(8, 5))
    else:
        stats_ax.clear()  # Clear the axes for the new plot

    # Create a pie chart
    stats_ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    stats_ax.set_title('Game Outcomes')

    plt.draw()
    plt.pause(0.001)  # Pause to update the plot

global epsilon_fig, epsilon_ax, epsilon_line

def plot_epsilon_value(epsilon_value, game_number, total_games):
    global epsilon_fig, epsilon_ax, epsilon_line

    # Create the figure and axis if they don't exist
    if 'epsilon_fig' not in globals():
        epsilon_fig, epsilon_ax = plt.subplots(figsize=(10, 4))
        epsilon_line, = epsilon_ax.plot([], [], 'r-')  # Red line for epsilon value
        epsilon_ax.set_xlim(0, total_games)
        epsilon_ax.set_ylim(0, 1)  # Epsilon values are typically between 0 and 1
        epsilon_ax.set_xlabel('Game Number')
        epsilon_ax.set_ylabel('Epsilon Value')
        epsilon_ax.set_title('Epsilon Value Over Time')

    # Update the data
    x_data, y_data = epsilon_line.get_data()
    x_data = np.append(x_data, game_number)
    y_data = np.append(y_data, epsilon_value)
    epsilon_line.set_data(x_data, y_data)

    # Redraw the plot
    epsilon_fig.canvas.draw()
    plt.pause(0.001)  # Pause to update the plot

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

# Function to assign improved rewards based on game outcome
def assign_rewards(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    reward_for_block = 0.5  # Reward for blocking opponent's win

    # Determine the base reward based on game outcome
    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    # Decay factor for the rewards
    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward

        # If not the last move, check if the move was a blocking move
        if i < len(game_history) - 1:
            next_board_state, _ = game_history[i + 1]
            if check_potential_win(board_state, -1 * np.sign(board_state[move])) and \
                    not check_potential_win(next_board_state, -1 * np.sign(board_state[move])):
                # This means the AI blocked an opponent's win
                target[move] = reward_for_block

        # Update the game history with the new target
        game_history[i] = (board_state, target)

        # Update the reward for the next iteration
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
            visualize_output_layer(predictions, board)
            visualize_detailed_network(model, board_state , predictions)
        if (args.human_player == 'X') or (args.human_player == 'O') or (args.show_visuals):
            visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)

        # Determine the move based on player type
        if (args.human_player == 'X' and player == 1) or (args.human_player == 'O' and player == -1):
            move = get_human_move(board)
        else:
             # Use epsilon-greedy strategy for move selection  
            move = epsilon_greedy_move(model, board, epsilon)

        if args.delay:
            time.sleep(1)  # Pauses the program

        # Make the move
        valid_move_made = make_move(board, move, player)
        if not valid_move_made:
            continue  # Skip the rest if the move was invalid

        if (args.human_player == 'X') or (args.human_player == 'O'):
            visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
            time.sleep(1)  # Pauses the program

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

### Neural models
# MLP Model
# MLP Model
def create_mlp_model(input_shape, dense_units, dropout_rate):
    model = keras.Sequential([
        layers.Dense(dense_units, activation='relu', input_shape=input_shape, name='input_layer'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(dense_units, activation='relu', name='hidden_layer_1'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        layers.Dense(9, activation='linear', name='output_layer')  # 9 output nodes for Tic-Tac-Toe
    ], name='MLP_Model')
    return model

# CNN Model
def create_cnn_model(input_shape, dense_units, dropout_rate):
    model = keras.Sequential([
        layers.Reshape((3, 3, 1), input_shape=input_shape, name='input_layer'),  
        layers.Conv2D(32, (3, 3), activation='relu', name='conv_layer'),
        layers.Flatten(name='flatten_layer'),
        layers.Dense(dense_units, activation='relu', name='hidden_layer'),
        layers.Dropout(dropout_rate, name='dropout'),
        layers.Dense(9, activation='linear', name='output_layer')
    ], name='CNN_Model')
    return model

# RNN Model
def create_rnn_model(input_shape, dense_units, dropout_rate):
    model = keras.Sequential([
        layers.Reshape((9, 1), input_shape=input_shape, name='input_layer'), 
        layers.SimpleRNN(dense_units, activation='relu', name='rnn_layer'),
        layers.Dense(dense_units, activation='relu', name='hidden_layer'),
        layers.Dropout(dropout_rate, name='dropout'),
        layers.Dense(9, activation='linear', name='output_layer')
    ], name='RNN_Model')
    return model

# Create or load the model based on the type argument
if os.path.exists(args.model_name):
    model = tf.keras.models.load_model(args.model_name)
    print("Model loaded successfully.")
else:
    input_shape = (9,)
    if args.model_type == 'MLP':
        model = create_mlp_model(input_shape, args.dense_units, args.dropout_rate)
    elif args.model_type == 'CNN':
        model = create_cnn_model(input_shape, args.dense_units, args.dropout_rate)
    elif args.model_type == 'RNN':
        model = create_rnn_model(input_shape, args.dense_units, args.dropout_rate)
    else:
        raise ValueError("Invalid model type")

    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f"New {args.model_type} model created.")

model.summary()

initial_weights = model.get_weights()

for layer in model.layers:
    print(f"Layer name: {layer.name}, Type: {type(layer)}")
    if hasattr(layer, 'units'):
        print(f"  - Units: {layer.units}")

if args.show_visuals:
    # After model is created or loaded
    visualize_model_weights_and_biases(model)

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

# Define the number of games after which model will be updated
# Set batch_size to be a tenth of n_games
batch_size = max(1, n_games // 10)  # Ensures at least one game per batch

batch_game_history = []

for game_number in range(1, n_games + 1):
    current_game_history = simulate_game_and_train(model, epsilon)
    batch_game_history.append(current_game_history)  # Append the history of the current game

    # Check if it's time to update the model
    if game_number % batch_size == 0 or game_number == n_games:
        print(f"update model")
        update_model(model, batch_game_history)
        if args.show_visuals:
            visualize_model_weights_and_biases(model)

        batch_game_history = []  # Reset for the next batch

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    if args.show_visuals:
        plot_game_statistics(wins_for_X, wins_for_O, draws)
        # Update the epsilon plot
        plot_epsilon_value(epsilon, game_number, n_games)

    # Switch starting player for the next game
    starting_player = -starting_player

    # Apply a 3-second delay if either 'delay' is enabled or a human player is playing
    if args.delay or args.human_player in ['X', 'O']:
        time.sleep(3)  # Pauses the program for 3 seconds


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
