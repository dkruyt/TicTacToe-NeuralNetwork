from colorama import Fore, Back, Style
import time
import platform
import argparse

## Local stuff
from modules.model import *
from modules.visualizations import *
from modules.game_logic import *
from modules.rewards import *
from modules.text import *

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

# Main 
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

# Main loop
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

# Save the model
try:
    model.save(args.model_name)  # Saves the model

    # Print success message
    print(f"Model saved successfully as {args.model_name}")

except IOError as e:
    # Handle I/O error such as directory not found or disk full
    print(f"Failed to save the model due to an I/O error: {e}")

except Exception as e:
    # Handle other possible exceptions
    print(f"An unexpected error occurred while saving the model: {e}")