# README.md for Tic-Tac-Toe Neural Network Training Script

## Overview
This script is designed by Dennis Kruyt just for fun, wanted to learn a little about training a neural network by playing Tic-Tac-Toe. It utilizes TensorFlow to create and train a neural network, which can be either a Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), or Recurrent Neural Network (RNN). The game includes various visualization features for the neural network's layers, activations, and game statistics, as well as command-line arguments to customize the game and training settings.

## Neural Network Model
The neural network is a sequential model with dense layers and dropout for regularization. It predicts the next move based on the game board's state.

## Saving and Loading the Model
The model is automatically saved after training and can be loaded when the script is run again, allowing for continuous learning.

## Features
- **Neural Network Models**: Choose between MLP, CNN, and RNN models.
- **Game Visualization**: Visualize the neural network's input, output, weights, biases, and detailed network structure during gameplay.
- **Game Statistics**: Display statistics like wins, losses, and draws in a pie chart.
- **Epsilon-Greedy Strategy**: Implement an epsilon-greedy strategy for move selection, with visual representation of epsilon value over time.
- **Command-line Arguments**: Customize various aspects of the game and training process.
- **Human vs AI Gameplay**: Play as a human against the AI.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Colorama
- IPython (for interactive display features)

## Installation
1. **Install Python**: Download and install Python from [python.org](https://www.python.org/).
2. **Install Dependencies**: Run `pip install numpy tensorflow matplotlib colorama ipython`.
3. **Download the Script**: Clone or download the script from the repository.

## Usage
Run the script from the command line with desired arguments. For example:

```bash
python tic_tac_toe.py --show-visuals --games 100 --model-type MLP

python tic_tac_toe.py --games 1000 --model-type MLP --model-name mlp-model.keras --dense-units 16
```

### Available Arguments
- `--show-visuals`: Enable visualizations during the game.
- `--show-text`: Enable text output for game status.
- `--delay`: Add a delay after each move.
- `--human-player`: Specify whether to play as 'X', 'O', or let AI play against itself.
- `--games`: Set the number of games to play.
- `--model-name`: Specify the filename for saving/loading the model.
- `--dense-units`: Set the number of neurons in Dense layers.
- `--dropout-rate`: Set the dropout rate.
- `--epsilon-start`, `--epsilon-end`, `--epsilon-decay`: Configure the epsilon-greedy strategy.
- `--model-type`: Choose the type of neural network model (MLP, CNN, RNN).

## Notes
- Ensure that you have a suitable environment for running TensorFlow, especially if using GPU acceleration.
- The visuals are best viewed in a Jupyter Notebook or an environment that supports IPython display features live vscode.