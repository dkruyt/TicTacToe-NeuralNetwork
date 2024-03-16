import numpy as np
import random
import copy

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
    reward_for_block = 0.2  # Reward for blocking opponent's win

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
    decay_factor = 0.9

    print(f"Game outcome: {'Win' if winner == 1 else 'Loss' if winner == -1 else 'Draw'}")
    print(game_history)

    # Initialize rewards for both players
    current_reward_player1 = reward_for_win if winner == 1 else (reward_for_draw if winner == 2 else reward_for_loss)
    current_reward_player2 = reward_for_win if winner == -1 else (reward_for_draw if winner == 2 else reward_for_loss)

    # Determine player for each move
    move_player = 1  # Assuming player 1 starts and players alternate

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)

        # Determine the current reward and player making the move
        if move_player == 1:
            current_reward = current_reward_player1
            # Apply decay and update reward for player 1's next move
            current_reward_player1 = current_reward * decay_factor
        else:
            current_reward = current_reward_player2
            # Apply decay and update reward for player 2's next move
            current_reward_player2 = current_reward * decay_factor

        # Assign reward to the move based on the player
        target[move] = current_reward
        # Update the game history with the new target
        game_history[i] = (board_state, target)

        print(f"Move {i+1}, Board State: {board_state}, Move: {move}, Reward: {current_reward:.2f}")

        # Switch player for the next iteration (previous move)
        move_player *= -1  # Switch between 1 and -1

def assign_rewards_progress_test1(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    decay_factor = 0.9
    
    tmp_game_history = copy.deepcopy(game_history)
    
    current_reward_player1 = reward_for_win if winner == 1 else (reward_for_draw if winner == 2 else reward_for_loss)
    current_reward_player2 = reward_for_win if winner == -1 else (reward_for_draw if winner == 2 else reward_for_loss)
    #print(winner)
    #print(game_history)
    for i in reversed(range(len(game_history))):
        current_reward = 0  # Initialize current_reward for safety

        if i > 0:  # Ensure there is a previous state to apply the reward to
            board_state, move = game_history[i]
            prev_board_state, prev_move = game_history[i-1]
            prev_target = np.zeros(9)  # Initialize target with zeros for the previous state
            if board_state[move] == 1:
                #print("player X")
                current_reward = current_reward_player1
                current_reward_player1 *= decay_factor
            elif board_state[move] == -1:
                #print("player O")
                current_reward = current_reward_player2
                current_reward_player2 *= decay_factor
                    
            prev_target[move] = current_reward # Assign the calculated reward to the previous state's move
            tmp_game_history[i-1] = (prev_board_state, prev_target)
        
        #print(f"Move {i+1}, Board State prev: {prev_board_state}, Move: {move}, Reward: {current_reward:.2f}")
        #print(f"Move {i+1}, Board State curr: {board_state}, Move: {move}, Reward: {current_reward:.2f}")

    tmp_game_history.pop()
    game_history = tmp_game_history
    
    #print(game_history)

    
    #print("\nUpdated game history with rewards:")
    #for state, reward in game_history:
    #    print(state, reward)

def assign_rewards_progress_test2(game_history, winner):
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    decay_factor = 0.9
    
    current_reward_player1 = reward_for_win if winner == 1 else (reward_for_draw if winner == 0 else reward_for_loss)
    current_reward_player2 = reward_for_win if winner == -1 else (reward_for_draw if winner == 0 else reward_for_loss)
    
    for i in reversed(range(len(game_history) - 1)):
        board_state, move = game_history[i + 1]
        prev_board_state, prev_move = game_history[i]
        target = np.zeros(9)
        
        if board_state[move] == 1:
            current_reward = current_reward_player1
            current_reward_player1 *= decay_factor
        elif board_state[move] == -1:
            current_reward = current_reward_player2
            current_reward_player2 *= decay_factor
        else:
            raise ValueError("Invalid board state")
        
        target[move] = current_reward
        game_history[i] = (prev_board_state, target)
    
    # Remove the last state since it doesn't have a previous state to assign the reward to
    game_history.pop()
    
    return game_history

def assign_rewards_future(game_history, winner):
    # Initialize reward parameters
    reward_for_win = 1.0
    reward_for_loss = -1.0
    reward_for_draw = 0.5
    predictive_reward = 0.2
    decay_factor = 0.9

    #print(f"Game outcome: {'Win' if winner == 1 else 'Loss' if winner == -1 else 'Draw'}")

    # Initialize current rewards for both players
    current_reward_player1 = reward_for_win if winner == 1 else (reward_for_draw if winner == 2 else reward_for_loss)
    current_reward_player2 = reward_for_win if winner == -1 else (reward_for_draw if winner == 2 else reward_for_loss)
    # Determine player for each move
    move_player = 1  # Start with player 1

    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)

        # Determine current reward and apply predictive reward if applicable
        if move_player == 1:
            current_reward = current_reward_player1
            if check_future_win(board_state, move, player=1):
                current_reward += predictive_reward
            # Update reward for next player 1 move
            current_reward_player1 = current_reward * decay_factor
        else:
            current_reward = current_reward_player2
            if check_future_win(board_state, move, player=-1):
                current_reward += predictive_reward
            # Update reward for next player 2 move
            current_reward_player2 = current_reward * decay_factor

        # Assign the calculated reward to the move
        target[move] = current_reward
        game_history[i] = (board_state, target)

        print(f"Move {i+1}, Board State: {board_state}, Move: {move}, Reward: {current_reward:.2f}")

        # Switch player for the next iteration (previous move)
        move_player *= -1

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

def check_future_win(board_state, move, player):
    """
    Checks if a move leads to a potential win in future turns for the specified player.
    Player should be 1 or -1, indicating player 1 or player 2 respectively.
    """
    temp_state = board_state.copy()
    temp_state[move] = player  # Set the move for the current player
    for i in range(9):
        if temp_state[i] == 0:
            temp_state[i] = player
            if check_potential_win(temp_state, player):
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

def assign_rewards_leading_upto(game_history, winner):
    """
    Assign rewards to game moves based on the final outcome, with a decay factor applied
    to rewards for moves further from the game's conclusion. Rewards are calculated separately
    for each player based on their actions.
    """
    # Rewards and decay factors
    reward_for_win = 1.0
    reward_for_draw = 0.5
    decay_factor = 0.9  # Decay factor for the reward
    
    # Track rewards for each player separately
    current_reward_player1 = reward_for_win if winner == 1 else (reward_for_draw if winner == 0 else 0)
    current_reward_player2 = reward_for_win if winner == -1 else (reward_for_draw if winner == 0 else 0)

    # Determine player for each move
    move_player = 1  # Start with player 1

    # Iterate through game history backwards
    for i in range(len(game_history) - 1, -1, -1):
        board_state, move = game_history[i]
        target = np.zeros(9)

        # Apply the current reward to the move based on which player made the move
        if move_player == 1:
            target[move] = current_reward_player1
            # Update reward for next player 1 move
            current_reward_player1 *= decay_factor
        else:
            target[move] = current_reward_player2
            # Update reward for next player 2 move
            current_reward_player2 *= decay_factor

        # Update game history with the assigned rewards
        game_history[i] = (board_state, target)

        # Switch player for the next iteration (previous move)
        move_player *= -1  # Switch between 1 and -1
        
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