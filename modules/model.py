import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
This module contains a suite of neural network models designed for the Tic-Tac-Toe AI. Each model varies in complexity and structure, tailored for different aspects of learning and decision-making in the game.

Model Definitions:

1. create_mlp_model(input_shape, dense_units, dropout_rate)
   - Type: Multi-Layer Perceptron (MLP)
   - Description: A general-purpose neural network model suitable for predicting Tic-Tac-Toe moves.
   - Architecture: Consists of dense layers with ReLU activation and dropout layers to prevent overfitting. The final output layer has 9 units corresponding to each Tic-Tac-Toe cell, using linear activation.
   - Usage: Ideal for general move prediction and game state evaluation.

2. create_policy_mlp_model(input_shape, dense_units, dropout_rate)
   - Type: Policy MLP
   - Description: Tailored for classification tasks, predicting the probability distribution of moves.
   - Architecture: Similar to the MLP model but concludes with a softmax activation layer for multi-class classification.
   - Usage: Useful for selecting moves based on probability, enhancing decision-making strategies.

3. create_value_mlp_model(input_shape, dense_units, dropout_rate)
   - Type: Value MLP
   - Description: A value-based model predicting the overall value or desirability of the game state.
   - Architecture: Ends with a single neuron using tanh activation, ideal for value estimation tasks.
   - Usage: Best for scenarios where evaluating the worth of a game state is crucial.

4. create_cnn_model(input_shape, dense_units, dropout_rate)
   - Type: Convolutional Neural Network (CNN)
   - Description: Leverages the spatial structure of the Tic-Tac-Toe board.
   - Architecture: Incorporates convolutional layers to process the board's grid, followed by dense layers.
   - Usage: Effective in learning spatial patterns and relationships on the board.

5. create_rnn_model(input_shape, dense_units, dropout_rate)
   - Type: Recurrent Neural Network (RNN)
   - Description: Designed to capture the sequential nature of the game moves.
   - Architecture: Uses RNN layers followed by dense layers, focusing on sequence data processing.
   - Usage: Ideal for situations where the sequence of moves is a significant factor.

6. create_simple_mlp_model(input_shape, dense_units)
   - Type: Simple MLP
   - Description: A more straightforward version of the MLP model.
   - Architecture: Consists of fewer layers, omitting additional hidden layers and dropout layers.
   - Usage: Suitable for less complex scenarios or as a baseline model.

These models provide a comprehensive toolkit for developing and training AI for Tic-Tac-Toe, each catering to different aspects of the game's complexity.
"""

### Neural models
# MLP Model
def create_mlp_model(input_shape, dense_units, dropout_rate):
    model = keras.Sequential([
        layers.Dense(dense_units, activation='relu', input_shape=input_shape, name='input_layer'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(dense_units, activation='relu', name='hidden_layer_1'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        layers.Dense(9, activation='linear', name='output_layer')  # 9 output nodes for Tic-Tac-Toe
    ], name='MLP_Model')
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_policy_mlp_model(input_shape, dense_units, dropout_rate):
    model = keras.Sequential([
        layers.Dense(dense_units, activation='relu', input_shape=input_shape, name='input_layer'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(dense_units, activation='relu', name='hidden_layer_1'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        layers.Dense(9, activation='softmax', name='output_layer')  # softmax for multi-class classification
    ], name='Policy_MLP_Model')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_value_mlp_model(input_shape, dense_units, dropout_rate):
    model = keras.Sequential([
        layers.Dense(dense_units, activation='relu', input_shape=input_shape, name='input_layer'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(dense_units, activation='relu', name='hidden_layer_1'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        layers.Dense(1, activation='tanh', name='output_layer')  # tanh for regression (value-based)
    ], name='Value_MLP_Model')
    model.compile(optimizer='adam', loss='mean_squared_error')
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
    model.compile(optimizer='adam', loss='mean_squared_error')
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
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# simple mlp
def create_simple_mlp_model(input_shape, dense_units):
    model = keras.models.Sequential([
        keras.layers.Dense(dense_units, activation="relu", input_shape=input_shape, name='input_layer'),
        keras.layers.Dense(9, name='output_layer')
    ], name='SimpleMLP_Model')
    model.compile(loss="mse", optimizer="adam")
    return model