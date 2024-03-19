import os
import platform
from colorama import init, Fore, Style
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

init(autoreset=True)

"""
This script includes a variety of utility functions for handling and visualizing a neural network model within a terminal, particularly for a Tic-Tac-Toe game. These functions enhance user interaction with the model and the game, providing insights into the AI's decision-making process and the game's dynamics.

Utility Functions:

- 'clear_screen': Clears the terminal screen for fresh output, compatible with different operating systems.
- 'cursor_topleft': Positions the cursor at the top left of the terminal, preparing for new output.
- 'move_cursor(x, y)': Moves the cursor to a specified (x, y) position within the terminal.

Game Display Functions:

- 'print_board(board)': Visually displays the Tic-Tac-Toe board in the terminal, using colored symbols to represent each player's moves.
- 'print_output_layer(output_layer_activation, board)': Outputs the activation values of the final layer in the model, showing the model's assessment for each possible move on the current board.
- 'plot_epsilon_value_text(epsilon_value, game_number, total_games)': Displays a progress bar for the game along with the current epsilon value, illustrating the balance between exploitation and exploration in the AI's strategy.

Model Analysis Functions:

- 'print_model_weights_and_biases(model)': Prints detailed statistics (mean, standard deviation, min, max) of the weights and biases for each layer in the model. For SimpleRNN layers, it differentiates between input and recurrent weights.
- 'visualize_detailed_network_text(model, input_data, output_data)': Provides a comprehensive view of the neural network's structure, including the type and number of neurons in each layer and an example of input and output data. This function is instrumental in understanding the architecture and data flow within the model.

Together, these functions offer an extensive toolkit for interacting with and understanding the neural network model and the game of Tic-Tac-Toe, from basic gameplay visualization to in-depth analysis of the AI model.
"""

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')
        
def cursor_topleft():
    print("\033[H", end='')

def move_cursor(x, y):
    """
    Move the cursor to the specified (x, y) position.
    """
    print(f"\033[{y};{x}H", end='')

# Function to print the current state of the board
def print_board(board):
    print("\n🎮 Board State:")
    # Use Style.BRIGHT for bold 'X' and 'O'
    symbols = {1: Style.BRIGHT + Fore.RED + 'X' + Style.RESET_ALL, 
               -1: Style.BRIGHT + Fore.GREEN + 'O' + Style.RESET_ALL, 
               0: Style.DIM + '{}'}
    horizontal_line = '───┼───┼───'
    vertical_line = Style.RESET_ALL + '│'

    for i in range(3):
        row = ' ' + ' {} '.format(vertical_line).join(
            symbols[board[i*3 + j]].format(i*3 + j) if board[i*3 + j] == 0 else symbols[board[i*3 + j]] 
            for j in range(3))
        print(row)
        if i < 2:
            print(horizontal_line)
    print()

def print_output_layer(output_layer_activation, board):
    move_cursor(0, 16)
    if output_layer_activation.size == 1:
        # Convert the numpy value to a Python scalar
        value = output_layer_activation.item()  # Extracts the scalar value from the array
        print(f"🔮 Value Prediction: {Fore.BLUE}{value:.2f}{Style.RESET_ALL}")
    else:
        # For Policy-based model (3x3 output)
        output_grid = output_layer_activation.reshape((3, 3))

        print("🧠 Neural Network Output Layer Activation:")
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
    print(f"🎲 Game {game_number} of {total_games} {progress_bar} Epsilon (ε): {epsilon_value:.4f}")

def print_model_weights_and_biases(model):
    print("\033[28;0H", end='')

    for i, layer in enumerate(model.layers):
        weights_biases = layer.get_weights()
        if len(weights_biases) > 0:
            print(f"{Fore.MAGENTA}Layer {i + 1} - {layer.name}:{Style.RESET_ALL}")
            if type(layer) != layers.SimpleRNN:
                weights, biases = weights_biases
                print(f"{Fore.GREEN}  ⚖️ Weights{Style.RESET_ALL} - Mean: {weights.mean():.4f}  Std: {weights.std():.4f}  Min: {weights.min():.4f}  Max: {weights.max():.4f}")
                print(f"{Fore.CYAN}  ⚙️ Biases{Style.RESET_ALL}  - Mean: {biases.mean():.4f}  Std: {biases.std():.4f}  Min: {biases.min():.4f}  Max: {biases.max():.4f}")
            else:
                weights, recurrent_weights, biases = weights_biases
                print(f"{Fore.GREEN}  ⚖️ Input Weights{Style.RESET_ALL}     - Mean: {weights.mean():.4f}  Std: {weights.std():.4f}  Min: {weights.min():.4f}  Max: {weights.max():.4f}")
                print(f"{Fore.YELLOW}  ⚖️ Recurrent Weights{Style.RESET_ALL} - Mean: {recurrent_weights.mean():.4f}  Std: {recurrent_weights.std():.4f}  Min: {recurrent_weights.min():.4f}  Max: {recurrent_weights.max():.4f}")
                print(f"{Fore.CYAN}  ⚙️ Biases{Style.RESET_ALL}            - Mean: {biases.mean():.4f}  Std: {biases.std():.4f}  Min: {biases.min():.4f}  Max: {biases.max():.4f}")
        else:
            print(f"{Fore.RED}Layer {i + 1} - {layer.name} (No weights/biases){Style.RESET_ALL}")

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

def print_tensorflow_info():
    print(f"🔢 TensorFlow Version: {tf.__version__}")

    # Print TensorFlow build details if available
    build_info = tf.sysconfig.get_build_info()
    print("🏗️ TensorFlow Build Information:")
    for key, value in build_info.items():
        print(f"  ➡️ {key}: {value}")

    # List of physical devices
    devices = tf.config.list_physical_devices()
    if not devices:
        print("⚠️ No physical devices found.")
    else:
        print("🔍 Found the following physical devices:")
        for idx, device in enumerate(devices):
            print(f"  🖥️ Device {idx + 1}:")
            print(f"    📌 Type: {device.device_type}")
            print(f"    🔖 Name: {device.name}")

            if device.device_type == 'GPU':
                details = tf.config.experimental.get_device_details(device)
                print(f"    🚀 Compute Capability: {details.get('compute_capability', 'N/A')}")

                # Additional GPU details
                try:
                    gpu_memory = tf.config.experimental.get_memory_info(device.name)
                    print(f"    📊 Total Memory: {gpu_memory['total'] / (1024**3):.2f} GB")
                    print(f"    📉 Current Memory Usage: {gpu_memory['current'] / (1024**3):.2f} GB")
                except:
                    print("    ℹ️ GPU memory details not available.")

                # Check if the device has the attribute 'memory_limit_bytes' before accessing it
                if hasattr(device, 'memory_limit_bytes'):
                    print(f"    💾 Memory Limit: {device.memory_limit_bytes / (1024**3):.2f} GB")

            elif device.device_type == 'CPU':
                # Additional CPU details if needed
                pass  # Placeholder for CPU-specific information

            print()