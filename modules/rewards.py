import numpy as np
import random

"""
This section of code defines several strategies of assigning rewards (and penalties) to the states and actions in a game of tic-tac-toe, for the purpose of training a reinforcement learning agent.

Firstly, 'assign_rewards_simple' is the most basic strategy that rewards winning and penalizes losing. The rewards are discounted back from the winning move - the closer the move is to the end of the game, the lower the discount. This is based on the premise that moves closer to winning are more valuable.

Secondly, 'assign_reward_penalty' has a similar approach, but additionally includes a small penalty for each move made to encourage quicker wins.

Next, 'assign_rewards_block' assigns an additional reward to moves that block the opponent from winning. If such a move is made, the reward for this move is updated to reward_for_block and other behaviors remain the same as 'assign_rewards_simple'.

Following 'assign_rewards_block' is 'assign_rewards_progress'. This strategy functions similarly to 'assign_rewards_simple', but includes incremental rewards for each move leading to a win, encouraging moves that progress the game towards a win.

Finally, 'assign_rewards_future' introduces predictive rewards for moves that can potentially lead to a win in the future - encouraging foresight in the agent's strategy.

The functions 'check_potential_win' and 'check_future_win' are utility functions used by reward assignment functions. 'check_potential_win' checks if either player has two in a row and thus can win in the next move, while 'check_future_win' checks if a given move can lead to a potential win in the next move by iterating through the possible following moves.

All of these strategies are different approaches towards teaching an AI to play the game optimally and can be used depending on the specific situation or the complexity of the game.
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
