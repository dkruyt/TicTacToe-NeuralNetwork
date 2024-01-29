import random
import numpy as np
from colorama import Fore, Back, Style

"""
This script includes a variety of functions to support gameplay and analysis in Tic-Tac-Toe, each contributing to the game's artificial intelligence and user interaction capabilities.

Key Functions:

- 'check_winner': Determines the game's status by identifying wins for 'X' (-1), 'O' (1), draws (2), or ongoing games (0).
- 'summarize_game_history': Gathers win, loss, and draw statistics from played games stored in the game history.
- 'make_move': Updates the game board with a player's move after validating the move.
- 'get_human_move': Allows human players to input their move, ensuring it's a valid selection.
- 'switch_player': Alternates the current player between 'X' and 'O' after each turn.

Strategic AI Functions:

- 'epsilon_greedy_move_default': Implements the epsilon-greedy strategy, balancing between exploiting learned strategies and exploring new moves.
- 'epsilon_greedy_move_value': A variant of the epsilon-greedy strategy, tailored for value-based models, focusing on the most valuable moves.
- 'random_move_selection': Selects a move randomly, useful for exploration.
- 'softmax_exploration': Chooses moves based on a probability distribution derived from the softmax of predicted Q-values.
- 'ucb_move_selection': Uses the Upper Confidence Bound strategy, combining the value of moves with their uncertainty.
- 'minimax_move': Employs the Minimax algorithm, a classic decision rule for minimizing the possible loss in a worst-case scenario.
- 'minimax_with_epsilon': Combines the Minimax strategy with an epsilon-greedy approach for a balanced decision-making process.

Utility Functions:

- 'check_potential_win': Checks if a win is possible in the next move, aiding in blocking strategies.
- 'get_valid_moves': Returns a list of valid moves based on the current board state.
- 'flush_cache': Clears the prediction cache to maintain performance.
- 'predict_with_cache': Optimizes predictions using a caching mechanism, reducing computation for previously encountered states.
- 'print_cache_stats': Displays statistics about cache usage, including hit and miss ratios.

These functions collectively enable sophisticated gameplay, ranging from basic game mechanics to advanced AI strategies, providing a rich and interactive Tic-Tac-Toe experience.
"""

# Function to check if there is a winner or draw
def check_winner(board):
    for i in range(3):
        # Check rows
        if abs(sum(board[i*3:(i+1)*3])) == 3:
            return board[i*3]
        # Check columns
        if abs(sum(board[i::3])) == 3:
            return board[i]
    # Check diagonals
    if abs(board[0] + board[4] + board[8]) == 3:
        return board[0]
    if abs(board[2] + board[4] + board[6]) == 3:
        return board[2]
    # Check for draw
    if 0 not in board:
        return 2  # Draw
    return 0  # Game ongoing

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

# Function to update the board state with the player's move
def make_move(board, move, player):
    if board[move] == 0:
        board[move] = player
        return True
    return False

# Function for a human player
def get_human_move(board):
    valid_moves = [i for i in range(9) if board[i] == 0]
    move = None
    while move not in valid_moves:
        try:
            move = int(input("Enter your move (0-8): "))
            if move not in valid_moves:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return move

# Function to switch players between moves
def switch_player(player):
    return -player

# Function to select the next move using epsilon-greedy strategy
def epsilon_greedy_move_default(model, board, player, epsilon, show_text, board_state):
    if random.random() < epsilon:
        # Exploration: Choose a random move
        valid_moves = [i for i in range(9) if board[i] == 0]
        if show_text:
            print("\r\033[KAI is exploring: Chose a random move.", end='')
        return random.choice(valid_moves)
    else:
        # Exploitation: Choose the best move based on model prediction
        #predictions = model.predict(board_state, verbose=0)[0]
        predictions = predict_with_cache(model, board_state, player, show_text)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -1e7
        if show_text:
            print("\r\033[KAI is exploiting: Chose the best predicted move for.", end='')
        return np.argmax(predictions)

def epsilon_greedy_move_value(model, board, player, epsilon, show_text, board_state):
    if random.random() < epsilon:
        # Exploration: Choose a random move
        valid_moves = [i for i in range(9) if board[i] == 0]
        if show_text:
            print("\r\033[KAI is exploring: Chose a random move.", end='')
        return random.choice(valid_moves)
    else:
        # Exploitation: Choose the best move based on model prediction
        best_move = None
        best_value = -float('inf')
        for i in range(9):
            if board[i] == 0:
                new_board = board.copy()
                new_board[i] = player
                board_state = np.array([new_board])
                #predicted_value = model.predict(board_state, verbose=0)[0]
                predicted_value = predict_with_cache(model, board_state, player, show_text)[0]
                if predicted_value > best_value:
                    best_value = predicted_value
                    best_move = i
        if show_text:
            print("\r\033[KAI is exploiting: Chose the best predicted move.", end='')
        return best_move if best_move is not None else random.choice([i for i in range(9) if board[i] == 0])


def check_potential_win(board, player, show_text):
    for i in range(3):
        # Check rows and columns for potential win
        if sum(board[i*3:(i+1)*3]) == 2 * player or sum(board[i::3]) == 2 * player:
            return True
        # Check diagonals for potential win
        if i == 0 and (board[0] + board[4] + board[8] == 2 * player or 
                       board[2] + board[4] + board[6] == 2 * player):
            return True
    return False

# Function to select next move
def random_move_selection(board, show_text):
    valid_moves = [i for i in range(9) if board[i] == 0]
    if show_text:
        print("AI is choosing a random move.")
    return random.choice(valid_moves)

# Function to select next move using softmax exploration
def softmax_exploration(model, board, show_text, player, board_state):
    if show_text:
        print("AI is selecting a move using softmax exploration.")
    # Getting Q values from the model for current state
    #Q_values = model.predict(np.array([board]), verbose=0)[0]
    Q_values = predict_with_cache(model, board_state, player, show_text)[0]
    # Calculating policy probabilities using softmax
    policy = np.exp(Q_values) / np.sum(np.exp(Q_values))

    # Getting all the valid moves
    valid_moves = [i for i in range(9) if board[i] == 0]

    # Keeping only probabilities of valid moves
    policy = [policy[i] if i in valid_moves else 0 for i in range(9)]

    # Normalizing the policy again after excluding invalid moves
    policy = policy / np.sum(policy)

    # Choosing a move from valid moves according to the policy probabilities
    move = np.random.choice(range(9), p=policy)

    return move

# Initialize action counts
action_counts = [0]*9

def ucb_move_selection(model, board, show_text, player, board_state, c_param=0.1):
    global action_counts
    if show_text:
        print("AI is selecting a move using Upper Confidence Bound strategy.")
  
    # Get Q values for the board state from the model
    #Q_values = model.predict(np.array([board]), verbose=0)[0]
    Q_values = predict_with_cache(model, board_state, player, show_text)[0]
    
    # Get the count of total actions taken
    total_actions = sum(action_counts)
  
    # Compute UCB values for each action
    ucb_values = [Q_values[a] + c_param * np.sqrt(np.log(total_actions+1)/ (action_counts[a]+1)) 
                  if board[a] == 0 else -np.inf for a in range(9)]
        
    # Select the action with highest UCB value
    move = np.argmax(ucb_values)
  
    # Update the count of the selected move
    action_counts[move] += 1
  
    return move

### minimax_move and minimax_with_epsilon functions as well to pass alpha and beta.
def minimax(board, player, alpha=-float('inf'), beta=float('inf')):
    winner = check_winner_minimax(board)
    if winner is not None:
        return winner

    if player == 1:  # Maximizing player
        best_val = -float('inf')
        for move in get_valid_moves(board):
            board[move] = player
            val = minimax(board, -player, alpha, beta)
            board[move] = 0
            best_val = max(best_val, val)
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val
    else:  # Minimizing player
        best_val = float('inf')
        for move in get_valid_moves(board):
            board[move] = player
            val = minimax(board, -player, alpha, beta)
            board[move] = 0
            best_val = min(best_val, val)
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val

        
def check_winner_minimax(board):
    for i in range(3):
        # Check rows
        if sum(board[i*3:(i+1)*3]) == 3:
            return 1
        elif sum(board[i*3:(i+1)*3]) == -3:
            return -1

        # Check columns
        if sum(board[i::3]) == 3:
            return 1
        elif sum(board[i::3]) == -3:
            return -1

    # Check diagonals
    if board[0] + board[4] + board[8] == 3 or board[2] + board[4] + board[6] == 3:
        return 1
    elif board[0] + board[4] + board[8] == -3 or board[2] + board[4] + board[6] == -3:
        return -1

    # Check for draw
    if 0 not in board:
        return 0  # Draw

    return None  # Game ongoing
    
def minimax_move(board, player, show_text):
    #print(f"Minimax move for player {'X' if player == 1 else 'O'}")
    #best_val = -float('inf')
    best_val = -float('inf') if player == 1 else float('inf')
    best_move = None
    moves = get_valid_moves(board)

    for move in moves:
        board[move] = player
        val = minimax(board, -player)  # Recursively call minimax with the opposite player
        board[move] = 0
    
        #print(f"Move {move}: Score {val}")

        if player == 1:  # Maximize for player X
            if val > best_val:
                best_val = val
                best_move = move
        else:  # Minimize for player O
            if val < best_val:
                best_val = val
                best_move = move

    #print(f"Chosen move: {best_move}, Score: {best_val}")

    return best_move

def minimax_with_epsilon(board, player, epsilon, show_text):
    if random.random() < epsilon:
        # With probability epsilon, choose a random move
        valid_moves = [i for i in range(9) if board[i] == 0]
        return random.choice(valid_moves)
    else:
        # Otherwise, use the Minimax algorithm to choose the best move
        return minimax_move(board, player, show_text)

def get_valid_moves(board):
    return [i for i, cell in enumerate(board) if cell == 0]

# Cache board states, faster lookup
prediction_cache = {}
# Global variables for cache hit and miss counters
cache_hits = 0
cache_misses = 0

def ndarray_hash(array):
    """Create a hash for a numpy array."""
    return hash(array.tobytes())

# Function to flush the cache
def flush_cache():
    global prediction_cache
    prediction_cache.clear()
    #print(f"\033[20;0H", end='')
    #print("Cache has been flushed.")

# TODO: Implement via arg to enable or disable
def predict_with_cache(model, input_data, player, show_text, use_cache=True):
    global cache_hits, cache_misses
    # Create a hash for the input data
    input_hash = ndarray_hash(input_data)

    # Check if the result is in cache and if caching is enabled
    if use_cache and input_hash in prediction_cache:
        if show_text:
            print_cache_stats()
            #print("Prediction " + (Fore.GREEN + 'O' if player == -1 else Fore.RED + 'X') + Style.RESET_ALL + ": from cache")

        cache_hits += 1
        return prediction_cache[input_hash]

    # Compute and store the result if not in cache
    result = model.predict(input_data, verbose=0)
    
    prediction_cache[input_hash] = result
    cache_misses += 1

    if show_text:
        print_cache_stats()
        #print("Prediction " + (Fore.GREEN + 'O' if player == -1 else Fore.RED + 'X') + Style.RESET_ALL + ": neural net")
    return result

def print_cache_stats():
    global cache_hits, cache_misses
    total_accesses = cache_hits + cache_misses
    hit_miss_ratio = cache_hits / total_accesses if total_accesses > 0 else 0
    cache_size = len(prediction_cache)
    print(f"\033[22;0H", end='')
    print(f"Cache Hits: {cache_hits}")
    print(f"Cache Misses: {cache_misses}")
    print(f"Hit/Miss Ratio: {hit_miss_ratio:.2f}")  # formatted to two decimal place
    print(f"Number of Items in Cache: {cache_size}")