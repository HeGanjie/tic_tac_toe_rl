import gymnasium as gym
import numpy as np
from typing import Optional
from gymnasium import spaces


class TicTacToeEnv(gym.Env):
    """
    Tic Tac Toe environment for reinforcement learning.
    
    The board is represented as a 3x3 grid with the following values:
    - 0: empty cell
    - 1: current player's piece
    - -1: opponent's piece
    
    The observation is from the perspective of the current player,
    so the current player always sees their pieces as +1 and opponent as -1.
    
    Actions are integers from 0-8 representing positions on the board:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(9)  # 9 possible positions
        # Observation space: 3x3 board with values 0 (empty), 1 (current player), -1 (opponent)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.int8
        )
        
        # Initialize the board
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # 1 for X (first player), -1 for O (second player)
        self.done = False
        self.winner = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self._get_observation(), {}  # Return perspective-based observation and info dict
    
    def step(self, action):
        """
        Execute one time step of the environment.
        
        Args:
            action: integer from 0-8 representing the position to play
            
        Returns:
            observation: current state of the board from current player's perspective
            reward: reward for the action
            done: whether the game has ended
            truncated: whether the episode was truncated
            info: additional information
        """
        # Check if action is valid
        if not self._is_valid_action(action):
            # Invalid action - penalize and end game with opponent winning
            self.winner = self.current_player * -1  # Opponent wins
            self.done = True
            return self._get_observation(), -20, True, False, {"error": "Invalid action", "winner": self.winner}
        
        # Apply the action to the board
        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player
        
        # Check for win or draw
        winner = self._check_winner()
        done = bool(winner) or self._is_board_full()
        
        reward = 0
        
        if winner == 1:  # Player 1 wins
            reward = 1 if self.current_player == 1 else -1  # Reward relative to current player
            self.winner = 1
            done = True
        elif winner == -1:  # Player -1 wins
            reward = 1 if self.current_player == -1 else -1  # Reward relative to current player
            self.winner = -1
            done = True
        elif done:  # Draw
            reward = 0
            self.winner = 0
        
        # Switch player for next turn
        self.current_player *= -1
        
        # Return observation, reward, done, truncated, and info
        return self._get_observation(), reward, done, False, {"winner": self.winner}
    
    def _get_observation(self):
        """
        Get the observation from the perspective of the current player.
        This ensures the current player always sees their pieces as +1 and opponent as -1.
        """
        # Return the board from the current player's perspective
        # This means: current player's pieces = +1, opponent's pieces = -1
        return (self.board * self.current_player).flatten().copy()
    
    def _is_valid_action(self, action):
        """Check if an action is valid (position is empty)."""
        if action < 0 or action > 8:
            return False
        row, col = divmod(action, 3)
        return self.board[row, col] == 0  # Check the actual board state, not the perspective
    
    def _check_winner(self):
        """Check if there's a winner. Returns 1 for X, -1 for O, 0 for no winner."""
        # Check rows
        for row in range(3):
            if abs(sum(self.board[row, :])) == 3:
                return self.board[row, 0]
        
        # Check columns
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]
        
        # Check diagonals
        if abs(sum(self.board.diagonal())) == 3:
            return self.board[0, 0]
        
        if abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return self.board[0, 2]
        
        return 0  # No winner
    
    def _is_board_full(self):
        """Check if the board is full."""
        return not (self.board == 0).any()
    
    def render(self, mode='human'):
        """Render the current state of the board to the CLI."""
        if mode != 'human':
            raise NotImplementedError("Only human mode is supported for rendering")
        
        # Print the board with X, O, and spaces
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        
        print()
        for i in range(3):
            row_str = " {} | {} | {} ".format(
                symbols[self.board[i, 0]], 
                symbols[self.board[i, 1]], 
                symbols[self.board[i, 2]]
            )
            print(row_str)
            if i < 2:
                print("-----------")
        print()
    
    def get_valid_actions(self):
        """Get list of valid actions (empty positions)."""
        return [i for i in range(9) if self._is_valid_action(i)]
    
    def close(self):
        """Clean up resources."""
        pass