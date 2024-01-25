import os
import platform
from colorama import Fore, Back, Style
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

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
                    print(f"[{i},{j}]: {symbol} (Taken){Style.RESET_ALL}", end='  ')
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

def visualize_model_weights_and_biases_text(model):
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

