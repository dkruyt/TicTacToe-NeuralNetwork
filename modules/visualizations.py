import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tensorflow import keras
from tensorflow.keras import layers

"""
This code includes a collection of visualization functions designed for analyzing and troubleshooting the performance of a neural network in the context of a Tic-Tac-Toe game. These visualizations aid in understanding the model's decision-making process and evaluating its performance throughout the game.

Visualization Functions:

- 'visualize_input_layer': Creates a colored grid representation of the neural network's input layer, which corresponds to the current state of the Tic-Tac-Toe game board.
- 'visualize_output_layer': Generates visualizations for the model's output layer activations, highlighting the cell with the highest activation as the model's chosen next move.
- 'visualize_model_weights_and_biases': Offers detailed visual insights into the weights and biases of all layers in the model, aiding in the understanding of the model's learning process.
- 'visualize_detailed_network': Provides an elaborate visual representation of the entire network's structure, including its layers, connections, and neuron counts, alongside sample input and output data.
- 'plot_game_statistics': Visualizes game outcomes (wins, losses, draws) as a pie chart, showcasing the performance distribution of the players.
- 'plot_epsilon_value': Displays the progression of the epsilon value over time, illustrating the exploration versus exploitation balance in the epsilon-greedy strategy.
- 'plot_cumulative_statistics': Shows cumulative statistics of game outcomes over time, offering a broader perspective on the model's performance across multiple games.

These visualization tools play a crucial role in diagnosing model behavior, understanding game dynamics, and guiding improvements in the AI's strategy for Tic-Tac-Toe.
"""

# Ensure interactive mode is on for live updating of plots
plt.ion()

global figi

# Define a function to visualize the input layer of the neural network
def visualize_input_layer(input_layer, game_number, wins_for_X, wins_for_O, draws):
    global figi
    if 'figi' not in globals():
        figi, axi = plt.subplots()
    else:
        axi = figi.axes[0]  # Access the existing axes object

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

global figo

# Define function to visualize activations in the output layer
def visualize_output_layer(output_layer_activation, board, colormap='autumn'):
    global figo
    if 'figo' not in globals():
        figo, axo = plt.subplots()
    else:
        axo = figo.axes[0]  # Access the existing axes object
    clear_output(wait=True)
    axo.clear()  # Clear the axes to remove old content

    if output_layer_activation.size == 1:
        # Convert the numpy value to a Python scalar
        value = output_layer_activation.item()  # Extracts the scalar value from the array

        # Use the scalar value for formatting
        axo.text(0.5, 0.5, f'{value:.2f}', ha='center', va='center', fontsize=20)
        axo.set_title("Value Prediction")
    else:
        # For Policy-based model (3x3 output)
        output_grid = output_layer_activation.reshape((3, 3))
        heatmap = axo.imshow(output_grid, cmap=colormap, interpolation='nearest')

        # Adding value annotations on each cell
        for (i, j), value in np.ndenumerate(output_grid):
            if board[i * 3 + j] != 0:  # Check if the move is taken
                text_color = 'lightgray'  # Gray out the text for taken moves
            else:
                text_color = 'blue'  # Use a different color for available moves
            axo.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)

        axo.set_title("Neural Network Output Layer Activation")
        axo.set_xlabel("Column in Tic-Tac-Toe Board")
        axo.set_ylabel("Row in Tic-Tac-Toe Board")
        axo.set_xticks(np.arange(0, 3, 1))
        axo.set_yticks(np.arange(0, 3, 1))

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

    max_neurons = 24

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
    def find_middle_neurons(max_neurons):

        midpoint = max_neurons // 2
        start = max(midpoint - 2, 0)
        end = min(midpoint + 2, max_neurons)

        return list(range(start, end))

    for n, layer_size in enumerate(layer_sizes):
        layer_x = n * h_spacing
        if layer_size > max_neurons:
            displayed_neurons = max_neurons
            middle_neurons = find_middle_neurons(max_neurons)
        else:
            displayed_neurons = layer_size
            middle_neurons = []

        for i, neuron_y in enumerate(np.linspace(0, 1, displayed_neurons, endpoint=False) + v_spacing / 3.):
            neuron_color = 'lightgray' if i in middle_neurons and layer_size > max_neurons else layer_colors[n % len(layer_colors)]
            circle = plt.Circle((layer_x, neuron_y), 0.012, color=neuron_color, ec='k', zorder=4)
            nn_ax.add_artist(circle)

            if n > 0:
                for j, prev_neuron_y in enumerate(np.linspace(0, 1, min(layer_sizes[n - 1], max_neurons), endpoint=False) + v_spacing / 3.):
                    color = rainbow(float(i + j) / (displayed_neurons + min(layer_sizes[n - 1], max_neurons)))
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

# Global variables for the figure, axes and lines
global plotstats_fig, stats_ax, stats_lines

def plot_cumulative_statistics(wins_for_X, wins_for_O, draws, total_games, batch_size):

    global plotstats_fig, stats_ax, stats_lines
    labels = ['Wins for X', 'Wins for O', 'Draws']

    # Create the figure and axes if they don't exist
    if 'plotstats_fig' not in globals():
        plotstats_fig, stats_ax = plt.subplots(figsize=(10, 5))
        stats_lines = stats_ax.plot([], [], 'r-',  # Line for Wins for X
                                    [], [], 'g-',  # Line for Wins for O
                                    [], [], 'b-')  # Line for Draws
        stats_ax.set_xlim(0, total_games)
        stats_ax.set_ylim(0, total_games)
        stats_ax.set_xlabel('Game Number')
        stats_ax.set_ylabel('Cumulative Count')
        stats_ax.set_title('Game Statistics Over Time')
        stats_ax.legend(labels)

    # Slow with small batch size
    # stats_ax.vlines = []
    # # Add vertical dotted lines every batch_size games
    # for x in range(0, total_games, batch_size):
    #     line = stats_ax.axvline(x, linestyle='dotted', color='grey')
    #     stats_ax.vlines.append(line)

    # Update the data
    x_data = range(len(wins_for_X))
    stats_lines[0].set_data(x_data, wins_for_X)  # Update Wins for X line
    stats_lines[1].set_data(x_data, wins_for_O)  # Update Wins for O line
    stats_lines[2].set_data(x_data, draws)  # Update Draws line

    # Redraw the plot
    plotstats_fig.canvas.draw()
    plt.pause(0.001)  # Pause to update the plot

