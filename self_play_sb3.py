import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Tuple, List, Any
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tic_tac_toe_env import TicTacToeEnv


class TicTacToeSB3Env(gym.Env):
    """
    Wrapper for Tic Tac Toe environment to work better with Stable Baselines3.
    This version ensures all observations are properly shaped for neural networks.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(9)  # 9 possible positions
        # Observation space: 3x3 board with values 0 (empty), 1 (X), -1 (O)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.float32
        )
        
        # Initialize the board
        self.board = np.zeros((9,), dtype=np.float32)  # Flattened for neural network
        self.current_player = 1  # 1 for X (agent), -1 for O (opponent)
        self.done = False
        self.winner = None
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.board = np.zeros((9,), dtype=np.float32)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy(), {}  # Return flattened board and info dict
    
    def step(self, action):
        """
        Execute one time step of the environment.
        """
        # Check if action is valid
        if not self._is_valid_action(action):
            # Invalid action - penalize and end game
            return self.board.copy(), -10, True, False, {"error": "Invalid action"}
        
        # Apply the action to the board
        if self.board[action] != 0:
            # Invalid action - position already taken
            return self.board.copy(), -10, True, False, {"error": "Position already taken"}
        
        self.board[action] = self.current_player
        
        # Check for win or draw
        winner = self._check_winner()
        done = bool(winner) or self._is_board_full()
        
        reward = 0
        
        if winner == 1:  # Agent wins
            reward = 1
            self.winner = 1
            done = True
        elif winner == -1:  # Opponent wins
            reward = -1
            self.winner = -1
            done = True
        elif done:  # Draw
            reward = 0
            self.winner = 0
        
        # Switch player for next turn
        self.current_player *= -1
        
        # Return observation, reward, done, truncated, and info
        return self.board.copy(), reward, done, False, {"winner": self.winner}
    
    def _is_valid_action(self, action):
        """Check if an action is valid (position is empty)."""
        if action < 0 or action > 8:
            return False
        return self.board[action] == 0
    
    def _check_winner(self):
        """Check if there's a winner. Returns 1 for X, -1 for O, 0 for no winner."""
        board_2d = self.board.reshape(3, 3)
        
        # Check rows
        for row in range(3):
            if abs(sum(board_2d[row, :])) == 3:
                return board_2d[row, 0]
        
        # Check columns
        for col in range(3):
            if abs(sum(board_2d[:, col])) == 3:
                return board_2d[0, col]
        
        # Check diagonals
        if abs(sum(board_2d.diagonal())) == 3:
            return board_2d[0, 0]
        
        if abs(sum(np.fliplr(board_2d).diagonal())) == 3:
            return board_2d[0, 2]
        
        return 0  # No winner
    
    def _is_board_full(self):
        """Check if the board is full."""
        return not (self.board == 0).any()
    
    def render(self, mode='human'):
        """Render the current state of the board to the CLI."""
        if mode != 'human':
            raise NotImplementedError("Only human mode is supported for rendering")
        
        # Print the board with X, O, and spaces
        symbols = {0.0: ' ', 1.0: 'X', -1.0: 'O'}
        
        board_2d = self.board.reshape(3, 3)
        
        print()
        for i in range(3):
            row_str = " {} | {} | {} ".format(
                symbols[board_2d[i, 0]], 
                symbols[board_2d[i, 1]], 
                symbols[board_2d[i, 2]]
            )
            print(row_str)
            if i < 2:
                print("-----------")
        print()
    
    def get_valid_actions(self):
        """Get list of valid actions (empty positions)."""
        return [i for i in range(9) if self.board[i] == 0]


class SelfPlayWithSB3:
    """
    Self-play trainer using Stable Baselines3 algorithms.
    """
    
    def __init__(self, algorithm='DQN', learning_rate=1e-3):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.agent = None
        self.opponent_agent = None
        self.game_history = deque(maxlen=10000)
        
        # Create environment
        self.env = TicTacToeSB3Env()
        
        # Initialize the agent based on the selected algorithm
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the RL agent based on the selected algorithm."""
        if self.algorithm == 'DQN':
            self.agent = DQN(
                "MlpPolicy", 
                self.env, 
                learning_rate=self.learning_rate,
                buffer_size=10000,
                learning_starts=1000,
                target_update_interval=500,
                train_freq=1,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                verbose=0
            )
        elif self.algorithm == 'PPO':
            self.agent = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                clip_range=0.2,
                verbose=0
            )
        elif self.algorithm == 'A2C':
            self.agent = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=5,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # For self-play, we typically use the same agent model for both players initially
        self.opponent_agent = self.agent
    
    def play_game(self, render=False) -> Tuple[int, List]:
        """
        Play a single game between two agents (self-play).
        
        Returns:
            Tuple of (winner, game_experience)
        """
        obs, _ = self.env.reset()
        done = False
        game_experience = []
        step_count = 0
        
        while not done and step_count < 10:  # Max 9 moves in Tic Tac Toe
            if render:
                self.env.render()
            
            # Get action from current agent
            if step_count % 2 == 0:  # Agent 1's turn (X)
                action, _ = self.agent.predict(obs, deterministic=False)
            else:  # Agent 2's turn (O)
                # With some probability, use the latest agent or a random agent
                if random.random() < 0.7:  # 70% of the time use the learned agent
                    action, _ = self.opponent_agent.predict(obs, deterministic=False)
                else:  # 30% of the time use random moves for exploration
                    valid_actions = [i for i in range(9) if obs[i] == 0]
                    if valid_actions:
                        action = random.choice(valid_actions)
                    else:
                        action = 0  # Fallback
            
            # Store experience before taking action
            prev_obs = obs.copy()
            valid_actions = [i for i in range(9) if obs[i] == 0]
            
            # Take action
            obs, reward, done, truncated, info = self.env.step(action.item() if isinstance(action, np.ndarray) else action)
            step_count += 1
            
            # Store experience tuple
            game_experience.append({
                'state': prev_obs,
                'action': action,
                'reward': reward,
                'next_state': obs,
                'done': done,
                'valid_actions': valid_actions
            })
        
        winner = info.get('winner', 0)
        
        # Update the final reward for training (the reward is already set correctly in step())
        if game_experience:
            game_experience[-1]['final_winner'] = winner
        
        if render:
            self.env.render()
            if winner == 1:
                print("X (Agent 1) wins!")
            elif winner == -1:
                print("O (Agent 2) wins!")
            else:
                print("It's a draw!")
        
        return winner, game_experience
    
    def train_agents(self, num_episodes=10000):
        """
        Train agents using self-play.
        """
        print(f"Starting self-play training with {self.algorithm} algorithm...")
        print(f"Training for {num_episodes} episodes...")
        
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        
        for episode in range(num_episodes):
            winner, game_exp = self.play_game(render=(episode % 2000 == 0))
            
            # Update statistics
            if winner == 1:
                wins_agent1 += 1
            elif winner == -1:
                wins_agent2 += 1
            else:
                draws += 1
            
            # Perform learning step based on the game experience
            if self.algorithm == 'DQN' and len(game_exp) > 0:
                # For DQN, we'll update the agent periodically
                if episode > 100 and episode % 10 == 0:
                    # Learn from recent experiences
                    self.agent.learn(total_timesteps=100, reset_num_timesteps=False)
            elif self.algorithm in ['PPO', 'A2C'] and episode % 100 == 0:
                # For policy gradient methods
                self.agent.learn(total_timesteps=500, reset_num_timesteps=False)
            
            # Print progress
            if (episode + 1) % 1000 == 0:
                total_games = episode + 1
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"X wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
                print(f"O wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
                print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
                print("-" * 40)
        
        print("Training completed!")
        total_games = num_episodes
        print(f"Final Results:")
        print(f"X wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
        print(f"O wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        return {
            'wins_agent1': wins_agent1,
            'wins_agent2': wins_agent2,
            'draws': draws,
            'total_games': total_games
        }
    
    def evaluate_agents(self, num_games=1000):
        """
        Evaluate how well the trained agents perform.
        """
        print(f"\nEvaluating agents over {num_games} games...")
        
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        
        for i in range(num_games):
            winner, _ = self.play_game(render=False)
            
            if winner == 1:
                wins_agent1 += 1
            elif winner == -1:
                wins_agent2 += 1
            else:
                draws += 1
        
        total_games = num_games
        print(f"Evaluation Results:")
        print(f"Agent 1 wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
        print(f"Agent 2 wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        return {
            'wins_agent1': wins_agent1,
            'wins_agent2': wins_agent2,
            'draws': draws,
            'total_games': total_games
        }
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        if self.algorithm == 'DQN':
            self.agent = DQN.load(filepath, env=self.env)
        elif self.algorithm == 'PPO':
            self.agent = PPO.load(filepath, env=self.env)
        elif self.algorithm == 'A2C':
            self.agent = A2C.load(filepath, env=self.env)
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate self-play training.
    """
    print("Tic Tac Toe Self-Play Training with Stable Baselines3")
    print("=" * 50)
    
    # Choose algorithm (DQN, PPO, or A2C)
    algorithm = "DQN"  # You can change this to "PPO" or "A2C"
    
    # Create self-play trainer
    trainer = SelfPlayWithSB3(algorithm=algorithm)
    
    # Train the agents
    training_results = trainer.train_agents(num_episodes=5000)
    
    # Evaluate the agents
    evaluation_results = trainer.evaluate_agents(num_games=1000)
    
    # Save the trained model
    model_path = f"tic_tac_toe_{algorithm.lower()}_model.zip"
    trainer.save_model(model_path)
    
    # Play a demo game with rendering
    print("\nPlaying a demo game with the trained agents:")
    trainer.play_game(render=True)


if __name__ == "__main__":
    main()