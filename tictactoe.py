import time
import argparse
import sys
import os

"""
This script facilitates running a Tic-Tac-Toe game with various customizable options, particularly for AI training and visualization. It includes argument parsing for configuring game settings and AI strategies, and integrates functions for game logic, AI model interactions, and visual outputs. Key features include the ability to toggle visuals, set AI strategies, choose model types, and define reward strategies.

Main Components:

- Argument Parsing: Enables customization of game settings such as visuals, AI strategies, number of games, and model parameters.
- Game Simulation: Manages the gameplay loop, alternating between human and AI players based on the configuration.
- AI Strategy Implementation: Employs various strategies (e.g., epsilon-greedy, softmax, UCB) for AI decision-making.
- Model Training: Periodically updates the AI model based on batches of game outcomes, applying specified reward strategies.
- Visualization: If enabled, displays the game board, model predictions, and training statistics in real-time.
- Model Saving: Offers the option to save the trained model at the end of the session or upon interruption.

The script is intended for users interested in AI training, game theory, and real-time visualization of AI decision processes in a controlled, turn-based game environment.
"""

# Set up the argument parser with an extended description
parser = argparse.ArgumentParser(description=(
    'Run a customizable Tic-Tac-Toe game simulation with AI training and visual analytics. '
    'This script allows users to configure game settings, AI strategies, and model parameters, '
    'and includes real-time visualizations for game progress and AI behavior analysis.'
))

# Define arguments with more descriptive help messages
parser.add_argument('--show-visuals', action='store_true', 
                    help='Enables real-time visualizations of the game board and AI predictions, enhancing the interactive experience.')
parser.add_argument('--show-text', action='store_true', 
                    help='Activates text-based output for game events and AI decisions, useful for detailed monitoring of game progress.')
parser.add_argument('--delay', action='store_true', 
                    help='Introduces a delay between moves, allowing more time to observe and analyze AI behavior and game dynamics.')
parser.add_argument('--human-player', type=str, choices=['X', 'O', 'None'], default='None', 
                    help='Allows a human player to participate as X or O against the AI, or set to None for AI vs AI games.')
parser.add_argument('--alternate-moves', action='store_true', 
                    help='Alternates the starting player between X and O in successive games, ensuring balanced gameplay.')
parser.add_argument('--games', type=int, default=10, 
                    help='Specifies the total number of games to be played in the simulation, controlling the length of the training session.')
parser.add_argument('--batch-size', type=int, default=None, 
                    help='Determines the batch size for model updates, with a default of one-tenth the total number of games.')
parser.add_argument('--model-name', type=str, default='tic_tac_toe_model.keras', 
                    help='Sets the filename for saving or loading the AI model, facilitating model reuse and continuous training.')
parser.add_argument('--dense-units', type=int, default=32, 
                    help='Defines the number of neurons in each Dense layer of the neural network, impacting the model\'s complexity.')
parser.add_argument('--dropout-rate', type=float, default=0.2, 
                    help='Sets the dropout rate in Dropout layers, a technique to prevent overfitting in the neural network.')
parser.add_argument('--epsilon-start', type=float, default=1.0, 
                    help='Initial value of epsilon in epsilon-greedy strategy, governing the balance between exploration and exploitation.')
parser.add_argument('--epsilon-end', type=float, default=0.1, 
                    help='Final value of epsilon after decay, indicating the strategy\'s shift towards more exploitation over time.')
parser.add_argument('--epsilon-decay', type=float, default=0.99, 
                    help='Epsilon decay rate after each game, controlling the rate at which the strategy moves from exploration to exploitation.')
parser.add_argument('--model-type', type=str, 
                    choices=['MLP', 'Policy', 'Value', 'CNN', 'RNN', 'Simple'], default='MLP', 
                    help='Selects the AI model type, with options including MLP, CNN, RNN, and others, each offering different learning capabilities.')
parser.add_argument('--use-cache', action='store_true', default=False,
                    help='Enables caching of model predictions to speed up the simulation.')
parser.add_argument('--reward', type=str,
                    choices=['block', 'progress', 'penalty', 'simple', 'future', 'combined', 'win_moves', 'winning_sequence', 'opponent_penalty'], default='progress', 
                    help='Chooses the reward strategy for training the AI, affecting how the model learns from game outcomes.')
parser.add_argument('--agent-x-strategy', type=str,
                    choices=['epsilon_greedy', 'random', 'softmax', 'ucb', 'minimax', 'epsilon_minimax'], default='epsilon_greedy', 
                    help='Determines the strategy for Agent X, influencing its decision-making process during the game.')
parser.add_argument('--agent-o-strategy', type=str,
                    choices=['epsilon_greedy', 'random', 'softmax', 'ucb', 'minimax', 'epsilon_minimax'], default='epsilon_greedy', 
                    help='Sets the strategy for Agent O, similarly impacting its gameplay tactics.')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Debug stuff.')
args = parser.parse_args()

# Print the argument values with Unicode icons
print("🛠 Configured settings for Tic-Tac-Toe game:")
print("👁️ Show Visuals:      ", args.show_visuals)
print("📜 Show Text:         ", args.show_text)
print("⏲️ Delay:             ", args.delay)
print("🎮 Human Player:      ", args.human_player)
print("🔄 Alternate Moves:   ", args.alternate_moves)
print("🎲 Number of Games:   ", args.games)
print("💾 Model Name:        ", args.model_name)
print("🧠 Dense Units:       ", args.dense_units)
print("💦 Dropout Rate:      ", args.dropout_rate)
print("📈 Epsilon Start:     ", args.epsilon_start)
print("📉 Epsilon End:       ", args.epsilon_end)
print("⏳ Epsilon Decay:     ", args.epsilon_decay)
print("🤖 Model Type:        ", args.model_type)
print("💾 Use Cache:         ", args.use_cache)
print("🏆 Reward Strategy:   ", args.reward)
print("⚔️ Agent X Strategy:  ", args.agent_x_strategy)
print("🛡️ Agent O Strategy:  ", args.agent_o_strategy)

print()

## Local stuff, load after arg, so help is display faster.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppresses informational messages

from modules.model import *
from modules.visualizations import *
from modules.game_logic import *
from modules.rewards import *
from modules.text import *

show_text = args.show_text

print_tensorflow_info()

# Update the model training function
def update_model(model, batch_game_history):
    X_train = []
    y_train = []

    for game_history in batch_game_history:

        total_reward = 0

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

            if args.debug:
                # Find the index of the move made in the board_state
                move_index = board_state.index(1) if 1 in board_state else board_state.index(-1)
                # Get the reward for the move
                move_reward = target[move_index]
            
                # Accumulate the reward
                total_reward += move_reward

                # Print the move and its reward
                print(f"Move at index {move_index}: Reward = {move_reward}")
    if args.debug:
        # Print the total reward for the game
        print(f"Total reward for this game: {total_reward}\n")

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

        #board_state = np.array([board])
        # Adjust board state based on current player
        board_state = np.array([[-x if player == -1 else x for x in board]])

        if args.show_text or args.show_visuals:
            #predictions = model.predict(board_state, verbose=0)
            predictions = predict_with_cache(model, board_state, player, show_text, args.use_cache)

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
            # For 'Value' model type, select strategy support
            if args.model_type == 'Value':
                current_strategy = args.agent_x_strategy if player == 1 else args.agent_o_strategy
                if current_strategy == 'random':
                    move = random_move_selection(board, show_text, player)
                elif current_strategy == 'minimax':
                    move = minimax_move(board, player, show_text)
                elif current_strategy == 'epsilon_minimax':
                    move = minimax_with_epsilon(board, player, epsilon, show_text)
                else:
                    move = epsilon_greedy_move_value(model, board, player, epsilon, show_text, board_state, args.use_cache)
            else:
                # For other model types, use specified strategy for each agent
                current_strategy = args.agent_x_strategy if player == 1 else args.agent_o_strategy

                if current_strategy == 'epsilon_greedy':
                    move = epsilon_greedy_move_default(model, board, player, epsilon, show_text, board_state, args.use_cache)
                elif current_strategy == 'random':
                    move = random_move_selection(board, show_text, player)
                elif current_strategy == 'softmax':
                    move = softmax_exploration(model, board, show_text, player, board_state, args.use_cache)
                elif current_strategy == 'ucb':
                    move = ucb_move_selection(model, board, show_text, player, board_state, args.use_cache, c_param=0.1)
                elif current_strategy == 'minimax':
                    move = minimax_move(board, player, show_text)
                elif current_strategy == 'epsilon_minimax':
                    move = minimax_with_epsilon(board, player, epsilon, show_text)

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
            if args.show_text:
                move_cursor(0, 14)
            print(f"Game {game_number}: Winner - {Fore.RED + 'X   ' if winner == 1 else Fore.GREEN + 'O   ' if winner == -1 else 'Draw'}" + Style.RESET_ALL)
            #print()
            
            #time.sleep(3)

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

    print(f"🧠 New {args.model_type} model created.")

model.summary()

initial_weights = model.get_weights()

# Set up TensorBoard logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

if args.show_visuals:
    # After model is created or loaded
    visualize_model_weights_and_biases(model)

# Train the model over multiple games
starting_player = 1  # Start with 'X' in the first game
n_games = args.games

# Initialize counters
wins_for_X = 0
wins_for_O = 0
draws = 0

list_X = []
list_O = []
list_draws = []
list_epsilon = []

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

if args.human_player:
    text = "🕹️ SHALL WE PLAY A GAME?"
    print()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.1)
    print()
print()

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
            if args.show_text:
                print(f"\033[3;0H", end='')
            print(f"Updating model... (Update count: {model_update_count})")
            update_model(model, batch_game_history)
            flush_cache()
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
            #plot_epsilon_value(epsilon, game_number, n_games)

        # Switch starting player for the next game
        if args.alternate_moves:
            starting_player = -starting_player

        # test
        list_X.append(wins_for_X)
        list_O.append(wins_for_O)
        list_draws.append(draws)
        list_epsilon.append(epsilon)
        if game_number % batch_size == 0 or game_number == n_games:
            plot_cumulative_statistics(list_X, list_O, list_draws, n_games, batch_size, list_epsilon)

        # Apply a 3-second delay if either 'delay' is enabled or a human player is playing
        if args.delay or args.human_player in ['X', 'O']:
            time.sleep(3)  # Pauses the program for 3 seconds

    except KeyboardInterrupt:
        clear_screen()
        save_model = input("\n🛑 Detected KeyboardInterrupt. Do you want to save the model before exiting? (y/n): ")
        if save_model.lower() == 'y':
            print("💾 Saving model...")
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
    print(f"✅ Model saved successfully as {args.model_name}")

except IOError as e:
    # Handle I/O error such as directory not found or disk full
    print(f"❌ Failed to save the model due to an I/O error: {e}")

except Exception as e:
    # Handle other possible exceptions
    print(f"❌ An unexpected error occurred while saving the model: {e}")

#if args.show_visuals:
print("We are done.")
input("Press Enter to exit images will be closed...")  