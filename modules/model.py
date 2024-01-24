import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
