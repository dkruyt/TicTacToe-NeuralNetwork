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

## Detailed Explanation of Arguments

### `--show-visuals`
- **Description**: Enable game visuals during gameplay.
- **Impact**: Enhances the interactive experience by displaying visualizations of the neural network and game statistics. Requires an appropriate display environment like Jupyter Notebook.

### `--show-text`
- **Description**: Enable text output for game status and progress.
- **Impact**: Useful for understanding the game's flow, especially when not using visualizations. Provides information about player turns, board state, and game outcomes.

### `--delay`
- **Description**: Add a delay after each move in the game.
- **Impact**: Slows down the gameplay, allowing users to observe changes and decisions made by the AI after each move.

### `--human-player`
- **Description**: Specify whether a human will play (as 'X' or 'O') or set to 'None' for AI vs AI games.
- **Impact**: Allows users to either engage directly with the AI or observe an AI vs. AI match, making the game interactive and versatile.

### `--games`
- **Description**: Set the number of games to be played in the simulation.
- **Impact**: Controls the length of the simulation, allowing for either extensive training sessions or quick demonstrations.

### `--model-name`
- **Description**: Specify the filename for saving/loading the neural network model.
- **Impact**: Enables persistence of the model's state across sessions, crucial for ongoing training and refinement of the AI.

### `--dense-units`
- **Description**: Determine the number of neurons in each Dense layer of the neural network.
- **Impact**: Influences the learning capacity of the model. More neurons can increase complexity and computational requirements.

### `--dropout-rate`
- **Description**: Set the dropout rate for regularization to prevent overfitting.
- **Impact**: Helps in generalizing the model. A high dropout rate might hinder the model's ability to learn, while a low rate might lead to overfitting.

### `--epsilon-start`, `--epsilon-end`, `--epsilon-decay`
- **Description**: Control the epsilon-greedy strategy for balancing exploration and exploitation.
- **Impact**: Critical for effective learning. The start value sets initial exploration, end value is the lowest boundary, and decay controls the reduction rate after each game.

### `--model-type`
- **Description**: Choose the type of neural network model: MLP (Multi-Layer Perceptron), CNN (Convolutional Neural Network), or RNN (Recurrent Neural Network).
- **Impact**: Each model type has different strengths and is suitable for various learning and pattern recognition tasks in the game.

Each argument significantly influences the game simulation and AI training, affecting performance, learning effectiveness, and user experience.

## Notes
- Ensure that you have a suitable environment for running TensorFlow, especially if using GPU acceleration.
- The visuals are best viewed in a Jupyter Notebook or an environment that supports IPython display features live vscode.