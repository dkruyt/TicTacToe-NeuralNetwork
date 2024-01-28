import time
import argparse
from art import *
import sys

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
parser.add_argument('--alternate-moves', action='store_true', 
                    help='Alternate moves between X and O players (default: False)')
parser.add_argument('--games', type=int, default=10, 
                    help='Number of games to play (default: 10)')
parser.add_argument('--batch-size', type=int, default=None, 
                    help='Batch size for updating training the model, default a tenth from the number of games.')
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
parser.add_argument('--model-type', type=str, 
                    choices=['MLP', 'Policy', 'Value', 'CNN', 'RNN', 'Simple'], default='MLP', 
                    help='Define the type of AI model. Choices are Multilayer Perceptron (MLP), Policy, Value, Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN).')
parser.add_argument('--reward', type=str,
                    choices=['block', 'progress', 'penalty', 'simple', 'future', 'combined', 'win_moves', 'winning_sequence', 'opponent_penalty'], default='progress', 
                    help='Select the reward strategy for the AI agent.')
# Add new arguments for specifying strategies for Agent X and Agent O
parser.add_argument('--agent-x-strategy', type=str,
                    choices=['epsilon_greedy', 'random', 'softmax', 'ucb'], default='epsilon_greedy', 
                    help='Strategy for Agent X (default: epsilon_greedy)')
parser.add_argument('--agent-o-strategy', type=str,
                    choices=['epsilon_greedy', 'random', 'softmax', 'ucb'], default='epsilon_greedy', 
                    help='Strategy for Agent O (default: epsilon_greedy)')
args = parser.parse_args()

# Print the argument values
print("Configured settings for Tic-Tac-Toe game:")
print("Show Visuals:      ", args.show_visuals)
print("Show Text:         ", args.show_text)
print("Delay:             ", args.delay)
print("Human Player:      ", args.human_player)
print("Alternate Moves:   ", args.alternate_moves)
print("Number of Games:   ", args.games)
print("Model Name:        ", args.model_name)
print("Dense Units:       ", args.dense_units)
print("Dropout Rate:      ", args.dropout_rate)
print("Epsilon Start:     ", args.epsilon_start)
print("Epsilon End:       ", args.epsilon_end)
print("Epsilon Decay:     ", args.epsilon_decay)
print("Model Type:        ", args.model_type)
print("Reward Strategy:   ", args.reward)
print("Agent X Strategy:  ", args.agent_x_strategy)
print("Agent O Strategy:  ", args.agent_o_strategy)

print()

## Local stuff, load after arg, so help is display faster.

from modules.model import *
from modules.visualizations import *
from modules.game_logic import *
from modules.rewards import *
from modules.text import *

show_text = args.show_text

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
        if args.reward == 'progress':
            assign_rewards_progress(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'block':
            assign_rewards_block(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'simple':
            assign_rewards_simple(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'penalty':
            assign_reward_penalty(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'future':
            assign_rewards_future(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'combined':
            assign_rewards_combined(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'win_moves':
            assign_rewards_only_for_win(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'winning_sequence':
            assign_rewards_for_winning_sequence(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome
        elif args.reward == 'opponent_penalty':
            assign_rewards_and_opponent_penalty(game_history, check_winner(game_history[-1][0]))  # Assign rewards based on game outcome


        for board_state, target in game_history:
            X_train.append(board_state)
            y_train.append(target)

    model.fit(np.array(X_train), np.array(y_train), epochs=10, verbose=0, batch_size=32, callbacks=[tensorboard_callback])

# Main
def simulate_game_and_train(model, epsilon):
    board = [0]*9
    player = starting_player
    global game_history
    global wins_for_X, wins_for_O, draws

    current_game_history = []  # List to store moves of the current game

    while True:
        if args.show_text:
            cursor_topleft()
            plot_epsilon_value_text(epsilon, game_number, n_games)
            print(f"Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}\n")
            print(f"Starting Player: {Fore.RED + 'X' if starting_player == 1 else Fore.GREEN + 'O'}" + Style.RESET_ALL)
            print("Player " + (Fore.RED + 'O' if player == -1 else Fore.GREEN + 'X') + Style.RESET_ALL + "'s turn")
            print_board(board)

        # todo we need this? Correct!!!!
        #board_state = np.array([board])
        # Adjust board state based on current player
        board_state = np.array([[-x if player == -1 else x for x in board]])

        #predictions = model.predict(board_state, verbose=0)
        #predictions = predict_with_cache(board_state, player)

        if args.show_text:
            #visualize_detailed_network_text(model, board_state , predictions)
            print_output_layer(predictions, board)
            print()

        if args.show_visuals:
            visualize_output_layer(predictions, board)
            visualize_detailed_network(model, board_state , predictions)

        if (args.human_player == 'X') or (args.human_player == 'O') or (args.show_visuals):
            visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)

        # Determine the move based on player, strategy, and model type
        if (args.human_player == 'X' and player == 1) or (args.human_player == 'O' and player == -1):
            move = get_human_move(board)
        else:
            # For 'Value' model type, use epsilon_greedy_move_value strategy
            if args.model_type == 'Value':  
                move = epsilon_greedy_move_value(model, board, player, epsilon, show_text, board_state)
            else:
                # For other model types, use specified strategy for each agent
                current_strategy = args.agent_x_strategy if player == 1 else args.agent_o_strategy

                if current_strategy == 'epsilon_greedy':
                    move = epsilon_greedy_move_default(model, board, player, epsilon, show_text, board_state)
                elif current_strategy == 'random':
                    move = random_move_selection(board, show_text)
                elif current_strategy == 'softmax':
                    move = softmax_exploration(model, board, show_text, player, board_state)
                elif current_strategy == 'ucb':
                    move = ucb_move_selection(model, board, show_text, player, board_state, c_param=0.1)

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
                cursor_topleft()
                plot_epsilon_value_text(epsilon, game_number, n_games)
                print(f"Wins for X: {wins_for_X}, Wins for O: {wins_for_O}, Draws: {draws}\n")
                print(f"Starting Player: {Fore.RED + 'X' if starting_player == 1 else Fore.GREEN + 'O'}" + Style.RESET_ALL)
                print("Player " + (Fore.RED + 'O' if player == -1 else Fore.GREEN + 'X') + Style.RESET_ALL + "'s turn")
                print_board(board)
                print_output_layer(predictions, board)
                print()
                print()

            if args.show_visuals:
                visualize_input_layer(board, game_number, wins_for_X, wins_for_O, draws)
                visualize_output_layer(predictions, board)

            # Print winner
            print(f"Game {game_number}: Winner - {Fore.RED + 'X   ' if winner == 1 else Fore.GREEN + 'O   ' if winner == -1 else 'Draw'}" + Style.RESET_ALL)
            print()
            
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
    elif args.model_type == 'Policy':
        model = create_policy_mlp_model(input_shape, args.dense_units, args.dropout_rate)
    elif args.model_type == 'Value':
        model = create_value_mlp_model(input_shape, args.dense_units, args.dropout_rate)
    elif args.model_type == 'CNN':
        model = create_cnn_model(input_shape, args.dense_units, args.dropout_rate)
    elif args.model_type == 'RNN':
        model = create_rnn_model(input_shape, args.dense_units, args.dropout_rate)
    elif args.model_type == 'Simple':
        model = create_simple_mlp_model(input_shape, args.dense_units)
    else:
        raise ValueError("Invalid model type")

    print(f"New {args.model_type} model created.")

model.summary()

initial_weights = model.get_weights()

# Set up TensorBoard logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

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

list_X = []
list_O = []
list_draws = []

model_update_count = 0

epsilon_start = args.epsilon_start
epsilon_end = args.epsilon_end
epsilon_decay = args.epsilon_decay
epsilon = epsilon_start

# Define the number of games after which model will be updated
# Default batch_size to be a tenth of n_games
if args.batch_size is not None:
    batch_size = args.batch_size
else:
    batch_size = max(1, n_games // 10)

batch_game_history = []

# Generate ASCII art
print (text2art("Shall we play a game?"))
if args.delay:
    time.sleep(3)  # Pauses the program

if args.show_text:
    clear_screen()

# Main loop
for game_number in range(1, n_games + 1):
    try:
        current_game_history = simulate_game_and_train(model, epsilon)
        batch_game_history.append(current_game_history)  # Append the history of the current game

        # Check if it's time to update the model
        if game_number % batch_size == 0 or game_number == n_games:
            model_update_count += 1  # Increment the counter
            print(f"Updating model... (Update count: {model_update_count})")
            update_model(model, batch_game_history)
            if args.show_visuals:
                visualize_model_weights_and_biases(model)
            if args.show_text:
                print_model_weights_and_biases(model)

            batch_game_history = []  # Reset for the next batch

        # Update epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if args.show_visuals:
            plot_game_statistics(wins_for_X, wins_for_O, draws)
            # Update the epsilon plot
            plot_epsilon_value(epsilon, game_number, n_games)

        # Switch starting player for the next game
        if args.alternate_moves:
            starting_player = -starting_player

        # test
        list_X.append(wins_for_X)
        list_O.append(wins_for_O)
        list_draws.append(draws)
        if game_number % batch_size == 0 or game_number == n_games:
            plot_cumulative_statistics(list_X, list_O, list_draws, n_games, batch_size)

        # Apply a 3-second delay if either 'delay' is enabled or a human player is playing
        if args.delay or args.human_player in ['X', 'O']:
            time.sleep(3)  # Pauses the program for 3 seconds

    except KeyboardInterrupt:
        clear_screen()
        save_model = input("\nDetected KeyboardInterrupt. Do you want to save the model before exiting? (y/n): ")
        if save_model.lower() == 'y':
            print("Saving model...")
            # Save your model here
            model.save(args.model_name)  # Saves the model
            print("Model saved. Exiting.")
        else:
            print("Model not saved. Exiting.")
        sys.exit(0)

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

#if args.show_visuals:
print("We are done.")
input("Press Enter to exit images will be closed...")  