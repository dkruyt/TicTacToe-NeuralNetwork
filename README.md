# README.md for Tic-Tac-Toe Neural Network Training Script

## Overview
This script is designed by Dennis Kruyt just for fun, wanted to learn a little about training a neural network by playing Tic-Tac-Toe. It utilizes TensorFlow and Keras for neural network operations, and matplotlib for visualization. The script offers options for displaying game visuals and text, and includes a delay feature for better observation of the game process.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Colorama
- IPython (for `clear_output`)
- pickle

## Installation
Ensure you have Python installed, then install the required packages using pip:
```
pip install numpy tensorflow keras matplotlib colorama IPython
```

## Usage
Run the script from the command line. There are several optional arguments you can use:
- `--show-visuals`: Enable visual display of the game state and neural network layers. But makes it much slower, just for observation!
- `--show-text`: Enable text display of the game progress and results.
- `--delay`: Add a delay in the game loop for better observation.

Can also run without for max speed.

Example:
```
python tic_tac_toe.py --show-visuals --show-text
```

## Features
- **Neural Network Training**: Trains a neural network to learn playing Tic-Tac-Toe.
- **Game Visualization**: Visualizes the game board and neural network layers if enabled.
- **Command-Line Arguments**: Offers control over visuals, text display, and game speed.
- **Adaptive AI**: Utilizes an epsilon-greedy strategy for the AI's move selection.

## Functions
- `visualize_input_layer`: Visualizes the neural network's input layer.
- `visualize_output_layer`: Visualizes the neural network's output layer activations.
- `clear_screen`: Clears the terminal screen based on the operating system.
- `check_winner`: Checks for a winner in the game.
- `make_move`: Updates the game board with a player's move.
- `switch_player`: Switches the current player.
- `print_board`: Prints the current state of the game board.
- `epsilon_greedy_move`: Determines the AI's move using an epsilon-greedy strategy.
- `update_model`: Updates the neural network model based on game history.
- `summarize_game_history`: Summarizes the history of games played.
- `simulate_game_and_train`: Simulates a Tic-Tac-Toe game and trains the model.

## Neural Network Model
The neural network is a sequential model with dense layers and dropout for regularization. It predicts the next move based on the game board's state.

## Saving and Loading the Model
The model is automatically saved after training and can be loaded when the script is run again, allowing for continuous learning.

## Contributing
Contributions to the script are welcome. Please ensure to follow the existing coding style and add comments for any new features.

## License
This script is released under the MIT License. See `LICENSE` for more details.

---
