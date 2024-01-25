import os
import platform
from colorama import Fore, Back, Style
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
This script provides several utility functions to facilitate interactions with and visualization of a neural network model within a terminal, in the context of a tic-tac-toe game.

'clear_screen' and 'cursor_topleft' are functions to clear the screen and move the cursor to the top left, preparing for fresh screen output.

'print_board' displays the current state of the game board with coloured symbols for each player or an empty space.

'print_output_layer' prints the activation outputs of the final layer of the model, for each possible move on the board.

'plot_epsilon_value_text' prints a progress bar of gameplay along with the current epsilon value that determines the exploitation vs exploration ratio in the epsilon-greedy strategy.

'print_model_weights_and_biases' prints statistics, like mean and standard deviation, of the weights and biases of each layer in the given model. If the layer is of type SimpleRNN, it splits the weights into input weights and recurrent weights.

'visualize_detailed_network_text' prints the detailed structure of the neural network, displaying the type and number of neurons present in each layer. It also prints an example of the input and output data. In the context of a game, each neuron could represent a possible move. 
"""

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')
        
def cursor_topleft():
    print("\033[H", end='')

# Function to print the current state of the board
def print_board(board):
    print("\nBoard State:")
    symbols = {1: Fore.RED + 'X', -1: Fore.GREEN + 'O', 0: Style.RESET_ALL +' '}
    for i in range(3):
        print('\033[39m|'.join(symbols[board[i*3 + j]] for j in range(3)))
        if i < 2:
            print(Style.RESET_ALL + '-----')
    print(Style.RESET_ALL)
    print()

def print_output_layer(output_layer_activation, board):
    if output_layer_activation.size == 1:
        # Convert the numpy value to a Python scalar
        value = output_layer_activation.item()  # Extracts the scalar value from the array
        print(f"Value Prediction: {Fore.BLUE}{value:.2f}{Style.RESET_ALL}")
    else:
        # For Policy-based model (3x3 output)
        output_grid = output_layer_activation.reshape((3, 3))

        print("Neural Network Output Layer Activation:")
        for i in range(3):
            for j in range(3):
                value = output_grid[i, j]
                if board[i * 3 + j] != 0:  # Check if the move is taken
                    symbol = Fore.RED + 'X' if board[i * 3 + j] == 1 else Fore.GREEN + 'O'
                    print(f"[{i},{j}]: {symbol}   {Style.RESET_ALL}", end='  ')
                else:
                    print(f"[{i},{j}]: {Fore.BLUE}{value:.2f}{Style.RESET_ALL}", end='  ')
            print()  # New line after each row

def plot_epsilon_value_text(epsilon_value, game_number, total_games):
    # Define the width of the progress bar
    progress_bar_width = 50

    # Calculate the progress percentage
    progress_percentage = game_number / total_games

    # Calculate the number of filled positions in the progress bar
    filled_positions = int(progress_bar_width * progress_percentage)

    # Create the progress bar string
    progress_bar = '[' + '=' * filled_positions + ' ' * (progress_bar_width - filled_positions) + ']'

    # Print the progress bar with the current game number, total games, and epsilon value
    print(f"Game {game_number} of {total_games} {progress_bar} Epsilon: {epsilon_value:.4f}")

def print_model_weights_and_biases(model):
    for i, layer in enumerate(model.layers):
        weights_biases = layer.get_weights()
        if len(weights_biases) > 0:
            if type(layer) != layers.SimpleRNN:
                weights, biases = weights_biases

                print(f"Layer {i+1}: {layer.name}")
                print(f"  Weights: Mean = {weights.mean():.4f}, Std = {weights.std():.4f}, Min = {weights.min():.4f}, Max = {weights.max():.4f}")
                print(f"  Biases: Mean = {biases.mean():.4f}, Std = {biases.std():.4f}, Min = {biases.min():.4f}, Max = {biases.max():.4f}")

            elif type(layer) == layers.SimpleRNN:
                weights = weights_biases[0][:, :layer.units]
                recurrent_weights = weights_biases[0][:, layer.units:]
                biases = weights_biases[1]

                print(f"SimpleRNN Layer {i+1}: {layer.name}")
                print(f"  Input Weights: Mean = {weights.mean():.4f}, Std = {weights.std():.4f}, Min = {weights.min():.4f}, Max = {weights.max():.4f}")
                print(f"  Recurrent Weights: Mean = {recurrent_weights.mean():.4f}, Std = {recurrent_weights.std():.4f}, Min = {recurrent_weights.min():.4f}, Max = {recurrent_weights.max():.4f}")
                print(f"  Biases: Mean = {biases.mean():.4f}, Std = {biases.std():.4f}, Min = {biases.min():.4f}, Max = {biases.max():.4f}")

def visualize_detailed_network_text(model, input_data, output_data):
    # Initialize layer_sizes with the size of the input data
    layer_sizes = [np.prod(input_data.shape[1:])]

    for layer in model.layers:
        if hasattr(layer, 'units'):  # For Dense layers
            layer_sizes.append(layer.units)
        elif isinstance(layer, keras.layers.Conv2D):  # For Conv2D layers
            layer_sizes.append(layer.filters)
        elif isinstance(layer, keras.layers.Flatten) or isinstance(layer, keras.layers.Reshape):
            layer_sizes.append(np.prod(layer.output_shape[1:]))
        else:
            continue  # Skip other layer types in size calculation

    # Add the size of the output data
    layer_sizes.append(np.prod(output_data.shape[1:]))

    print("Detailed Neural Network Structure:")
    print("==================================")

    # Input Layer
    print(f"Input Layer: {layer_sizes[0]} neurons (features)")
    

    # Iterate over all layers to determine their sizes
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.SimpleRNN):
            layer_sizes.append(layer.units)
        elif isinstance(layer, keras.layers.Conv2D):
            layer_sizes.append(layer.filters)
        elif isinstance(layer, keras.layers.Flatten) or isinstance(layer, keras.layers.Reshape):
            layer_sizes.append(np.prod(layer.output_shape[1:]))
        else:
            # For layers like Dropout, use the size of the previous layer
            layer_sizes.append(layer_sizes[-1])

    # Ensure the output layer size is added
    layer_sizes.append(np.prod(output_data.shape[1:]))

    # Now, iterate over the layers for printing details
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        layer_description = f"{layer_type} Layer with {layer_sizes[i+1]} neurons"
        print(layer_description)

    # Output Layer
    print(f"Output Layer: {layer_sizes[-1]} neurons (output size)")

    # Displaying the first few input and output data values as examples
    print("\nExample Input Data: ", input_data[0][:5])
    print("Example Output Data: ", output_data[0][:5])