import numpy as np
import random

"""
This section of code includes a variety of reward assignment strategies for training a reinforcement learning agent in Tic-Tac-Toe. Each strategy has a unique approach to rewarding and penalizing game states and actions, designed to teach the AI optimal gameplay.

- 'assign_rewards_simple': Awards basic rewards for winning and penalties for losing. Rewards are discounted for each move away from the game's conclusion, emphasizing the value of moves leading to the outcome.

- 'assign_reward_penalty': Similar to 'assign_rewards_simple', but with an added penalty for each move to encourage faster wins.

- 'assign_rewards_block': Gives additional rewards for moves that prevent the opponent from winning. It complements the 'assign_rewards_simple' strategy by recognizing defensive plays.

- 'assign_rewards_progress': Builds on 'assign_rewards_simple' by adding incremental rewards for each move that advances the game towards a win.

- 'assign_rewards_future': Focuses on foresight by rewarding moves that can potentially lead to a win in future turns.

- 'assign_rewards_combined': A comprehensive approach that combines the aspects of blocking, progress, future prediction, and move penalties.

- 'assign_rewards_only_for_win': Assigns rewards exclusively for moves that directly contribute to winning, ignoring all other moves.

- 'assign_rewards_for_winning_sequence': Rewards all moves that are part of the final winning sequence, emphasizing effective strategies that lead to victory.

- 'assign_rewards_and_opponent_penalty': Similar to 'assign_rewards_for_winning_sequence', but also penalizes the opponent's last move if it failed to prevent the win.

Utility functions:

- 'check_potential_win': Evaluates if a player is about to win in their next move.
- 'check_future_win': Checks if a move can lead to a potential win in subsequent turns.
- 'find_winning_sequence_moves': Identifies the moves that formed the winning sequence in a completed game.

These strategies offer diverse ways to reinforce desirable behaviors in the AI, catering to different aspects of learning and decision-making in the game of Tic-Tac-Toe.
"""

# Function to assign improved rewards based on game outcome
def assign_rewards_simple(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5

    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward
        game_history[i] = (board_state, target)
        current_reward *= decay_factor

def assign_reward_penalty(game_history, winner):
    """Rewards are assigned from end of game backwards, with later
    rewards decayed by a factor. Also a penalty for each move"""
    reward_for_win = 10.0
    reward_for_loss = -10.0
    reward_for_draw = 0.0

    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward
        game_history[i] = (board_state, target)
        current_reward *= decay_factor
        current_reward -= 1  # penalty for each move made

# Function to assign improved rewards based on game outcome
def assign_rewards_block(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    reward_for_block = 0.5  # Reward for blocking opponent's win

    # Determine the base reward based on game outcome
    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    # Decay factor for the rewards
    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward

        # If not the last move, check if the move was a blocking move
        if i < len(game_history) - 1:
            next_board_state, _ = game_history[i + 1]
            if check_potential_win(board_state, -1 * np.sign(board_state[move])) and \
                    not check_potential_win(next_board_state, -1 * np.sign(board_state[move])):
                # This means the AI blocked an opponent's win
                target[move] = reward_for_block

        # Update the game history with the new target
        game_history[i] = (board_state, target)

        # Update the reward for the next iteration
        current_reward *= decay_factor

def assign_rewards_progress(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    reward_for_progress = 0.1  # Incremental reward for moves leading to a win

    #print(f"Game outcome: {'Win' if winner == 1 else 'Loss' if winner == -1 else 'Draw'}")

    # Determine the base reward based on game outcome
    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    # If the game was won, distribute rewards backwards from the winning move
    if winner in [1, -1]:
        decay_factor = 0.9
        current_reward = reward

        for i in range(len(game_history) - 1, -1, -1):
            board_state, move = game_history[i]
            target = np.zeros(9)

            # Assign reward to the move
            target[move] = current_reward

            # Update the game history with the new target
            game_history[i] = (board_state, target)

            #print(f"Move {i+1}, Board State: {board_state}, Move: {move}, Reward: {current_reward:.2f}")

            # Apply decay to the reward for the next (earlier) move
            current_reward *= decay_factor

    # If the game resulted in a draw, assign the draw reward
    elif winner == 2:
        for i in range(len(game_history)):
            board_state, move = game_history[i]
            target = np.zeros(9)
            target[move] = reward_for_draw
            game_history[i] = (board_state, target)

            #print(f"Move {i+1}, Board State: {board_state}, Move: {move}, Reward: {reward_for_draw}")

def assign_rewards_future(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    predictive_reward = 0.2

    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward
        # Additional reward for predictive move
        if check_future_win(board_state, move):
            #print('Future win detected - predictive reward applied')
            target[move] += predictive_reward
        game_history[i] = (board_state, target)
        current_reward *= decay_factor
        #print(f'End of iteration {i} current_reward:', current_reward)

    #print('Final game history:', game_history)

def check_potential_win(board, player):
    for i in range(3):
        # Check rows and columns for potential win
        if sum(board[i*3:(i+1)*3]) == 2 * player or sum(board[i::3]) == 2 * player:
            return True
        # Check diagonals for potential win
        if i == 0 and (board[0] + board[4] + board[8] == 2 * player or 
                       board[2] + board[4] + board[6] == 2 * player):
            return True
    return False

# Function to check a potential win in the next move
def check_future_win(board_state, move):
    temp_state = board_state.copy()
    temp_state[move] = 1  # Assuming the sign of the current player is 1
    for i in range(9):  
        if temp_state[i] == 0:
            temp_state[i] = 1
            if check_potential_win(temp_state, 1):  # Assuming check_win_condition function is there
                return True
            temp_state[i] = 0
    return False

def assign_rewards_combined(game_history, winner):
    reward_for_win = 2.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    reward_for_block = 0.2 
    reward_for_progress = 0.1 
    predictive_reward = 0.2
    penalty_for_each_move = 0.2

    if winner == 1:
        reward = reward_for_win
    elif winner == -1:
        reward = reward_for_loss
    elif winner == 2:
        reward = reward_for_draw
    else:
        raise ValueError("Invalid winner value")

    decay_factor = 0.9
    current_reward = reward

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)
        target[move] = current_reward

        # If not the last move, check if the move was a blocking move
        if i < len(game_history) - 1:
            next_board_state, _ = game_history[i + 1]
            if check_potential_win(board_state, -1 * np.sign(board_state[move])) and \
                    not check_potential_win(next_board_state, -1 * np.sign(board_state[move])):
                target[move] += reward_for_block

        # Additional reward for predictive move
        if check_future_win(board_state, move):
            target[move] += predictive_reward

        # Incremental reward for each move leading to a win
        current_reward += reward_for_progress

        game_history[i] = (board_state, target)

        # Apply decay to the reward for the next (earlier) move
        current_reward *= decay_factor
        # penalty for each move made
        current_reward -= penalty_for_each_move

def assign_rewards_only_for_win(game_history, winner):
    """
    Assign rewards only for moves that directly contribute to winning.
    No penalties are given for any moves.
    """
    reward_for_win = 1.0  # Reward for the winning move
    no_reward = 0.0  # No reward for other moves

    if winner == 1:
        # If the agent wins, assign rewards
        for i in range(len(game_history) - 1, -1, -1):
            board_state, move = game_history[i]
            target = np.zeros(9)
            
            if i == len(game_history) - 1:
                # Assign reward only to the winning move
                target[move] = reward_for_win
            else:
                # No reward for other moves
                target[move] = no_reward

            game_history[i] = (board_state, target)

    elif winner == -1 or winner == 2:
        # If the agent loses or the game is a draw, no rewards are assigned
        for i in range(len(game_history)):
            board_state, move = game_history[i]
            target = np.zeros(9)
            target[move] = no_reward
            game_history[i] = (board_state, target)

    else:
        raise ValueError("Invalid winner value")

    return game_history

def assign_rewards_for_winning_sequence(game_history, winner):
    """
    Assign rewards to all moves that contribute to the winning sequence.
    """
    reward_for_winning_sequence = 1.0  # Reward for moves that are part of the winning sequence

    if winner == 1:
        # Find the winning sequence moves
        winning_moves = find_winning_sequence_moves(game_history)
        
        for i in range(len(game_history)):
            board_state, move = game_history[i]
            target = np.zeros(9)

            # Assign reward if the move is part of the winning sequence
            if move in winning_moves:
                target[move] = reward_for_winning_sequence
            else:
                # No reward for other moves
                target[move] = 0.0

            game_history[i] = (board_state, target)

    else:
        # If the agent does not win, no rewards are assigned
        for i in range(len(game_history)):
            board_state, move = game_history[i]
            target = np.zeros(9)
            target[move] = 0.0
            game_history[i] = (board_state, target)

    return game_history

def assign_rewards_and_opponent_penalty(game_history, winner):
    """
    Assign rewards to all moves that contribute to the winning sequence and 
    penalize the opponent's last move if it failed to block a winning move.
    """
    reward_for_winning_sequence = 1.0  # Reward for moves that are part of the winning sequence
    penalty_for_not_blocking = -0.5  # Penalty for the opponent's failure to block the winning move

    if winner == 1:
        # Find the winning sequence moves
        winning_moves = find_winning_sequence_moves(game_history)

        for i in range(len(game_history)):
            board_state, move = game_history[i]
            target = np.zeros(9)

            # Assign reward if the move is part of the winning sequence
            if move in winning_moves:
                target[move] = reward_for_winning_sequence
            else:
                target[move] = 0.0

            game_history[i] = (board_state, target)

        # Apply penalty to the opponent's last move if it failed to block the winning move
        if len(game_history) > 1:
            last_opponent_move_index = len(game_history) - 2
            previous_board_state, _ = game_history[last_opponent_move_index]
            _, opponent_board_state = game_history[last_opponent_move_index + 1]

            opponent_move = np.where(opponent_board_state != previous_board_state)[0][0]
            target = np.zeros(9)
            target[opponent_move] = penalty_for_not_blocking
            game_history[last_opponent_move_index] = (previous_board_state, target)

    else:
        # If the agent does not win, no rewards or penalties are assigned
        for i in range(len(game_history)):
            board_state, move = game_history[i]
            target = np.zeros(9)
            target[move] = 0.0
            game_history[i] = (board_state, target)

    return game_history

def find_winning_sequence_moves(game_history):
    """
    Find the moves that are part of the winning sequence.
    """
    # Assuming the last board state in game_history is the winning state
    final_board_state, _ = game_history[-1]
    winning_moves = []

    # Check rows, columns, and diagonals for the winning sequence
    for i in range(3):
        # Check rows
        if final_board_state[i*3] == final_board_state[i*3+1] == final_board_state[i*3+2] != 0:
            winning_moves.extend([i*3, i*3+1, i*3+2])
        # Check columns
        if final_board_state[i] == final_board_state[i+3] == final_board_state[i+6] != 0:
            winning_moves.extend([i, i+3, i+6])

    # Check diagonals
    if final_board_state[0] == final_board_state[4] == final_board_state[8] != 0:
        winning_moves.extend([0, 4, 8])
    if final_board_state[2] == final_board_state[4] == final_board_state[6] != 0:
        winning_moves.extend([2, 4, 6])

    return list(set(winning_moves))  # Remove duplicates if any