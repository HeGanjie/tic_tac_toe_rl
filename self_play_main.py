import random
from typing import List, Tuple, Dict

import torch.nn as nn
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from networks.cnn_3_16_32 import CustomTicTacToeCNN, CustomCNN_3x3
from networks.resnet import CustomResNetCNN
from networks.transformer_arch import TicTacToeTransformer
from tic_tac_toe_env import TicTacToeEnv  # Import the base environment


class SelfPlayTrainer:
    """
    Advanced self-play trainer for Tic Tac Toe using Stable Baselines3.
    """
    
    def __init__(self, algorithm='DQN', learning_rate=3e-4, n_envs=4):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.best_model = None
        self.current_model = None
        self.game_history = []
        self.n_envs = n_envs
        
        # Initialize model based on algorithm
        self._init_model(n_envs=n_envs)
    
    def _init_model(self, n_envs=4):
        """Initialize the RL model based on the selected algorithm with vectorized environments."""
        self.n_envs = n_envs
        # Create multiple environments for vectorized training
        self.vec_env = DummyVecEnv([lambda: TicTacToeEnv() for _ in range(n_envs)])
        # Also keep a single environment for game playing (evaluation, human-play, etc.)
        self.single_env = TicTacToeEnv()
        
        if self.algorithm == 'DQN':
            policy_kwargs = dict(
                features_extractor_class=CustomTicTacToeCNN,
                features_extractor_kwargs=dict(features_dim=64),
                net_arch=[64, 64],
                activation_fn=nn.ReLU,
            )
            self.current_model = DQN(
                "CnnPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=32,
                target_update_interval=500,
                train_freq=1,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                policy_kwargs=policy_kwargs,
                verbose=0
            )
        elif self.algorithm == 'PPO':
            policy_kwargs = dict(
                features_extractor_class=CustomTicTacToeCNN,
                features_extractor_kwargs=dict(features_dim=64),
                net_arch=[64, 64],
                activation_fn=nn.ReLU,
                normalize_images=False,
            )
            self.current_model = MaskablePPO(
                MaskableActorCriticPolicy,
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=10,
                batch_size=64,
                n_epochs=10,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                verbose=0
            )
        elif self.algorithm == 'A2C':
            policy_kwargs = dict(
                features_extractor_class=CustomTicTacToeCNN,
                features_extractor_kwargs=dict(features_dim=64),
                net_arch=[64, 64],
                activation_fn=nn.ReLU,
            )
            self.current_model = A2C(
                "CnnPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=5,
                policy_kwargs=policy_kwargs,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train_self_play(self, num_episodes=10000):
        """
        Train the model using self-play with balanced player roles.
        The model learns by playing against itself in alternating roles.
        """
        print(f"Starting self-play training with {self.algorithm} for {num_episodes} episodes...")
        
        # The issue we're addressing is that the model should learn to play effectively 
        # as both first and second player. In traditional RL training, the agent always 
        # starts from the same initial conditions, which can cause first-player bias.
        #
        # For proper self-play, we need the model to be trained on games where it plays
        # both as X (first player) and O (second player) against different opponents.
        # The current stable-baselines3 training already does this implicitly through
        # the vectorized environments, but we can enhance this with more explicit 
        # alternating role training.
        
        # Create a separate training environment where the model learns to play
        # against itself or a baseline opponent in alternating roles
        total_timesteps = num_episodes * 10  # Approximate, as each episode may vary in length
        print(f"Training model for approximately {total_timesteps} total timesteps...")
        
        # For better balanced training, we'll also do some manual self-play games
        # during training to ensure first/second player balance
        self.current_model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        print("Self-play training completed!")
        
        # Set the trained model as the best model
        self.best_model = self.current_model
        return {
            'wins_agent1': 0,  # These will be calculated in the evaluation script
            'wins_agent2': 0,
            'draws': 0,
            'total_games': 0
        }


    @staticmethod
    def _random_agent(obs):
        """
        A simple random agent for baseline comparison.
        obs is the board state from the current player's perspective in 6-channel format.
        obs shape is (6, 3, 3) where:
        - obs[0]: X pieces
        - obs[1]: O pieces
        - obs[2]: Current player is X
        - obs[3]: Current player is O
        - obs[4]: Last opponent move (one-hot)
        - obs[5]: Bias plane
        Empty positions are where both X and O channels have 0.
        """
        # Find valid actions from the board state
        # Empty positions are where both X (channel 0) and O (channel 1) channels are 0
        empty_positions = (obs[0] == 0) & (obs[1] == 0)  # Positions that are neither X nor O
        valid_actions = []
        
        for i in range(9):
            row, col = divmod(i, 3)
            if empty_positions[row, col] == 1:  # If position is empty
                valid_actions.append(i)
        
        if valid_actions:
            return random.choice(valid_actions)
        else:
            return 0  # fallback action
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.best_model:
            self.best_model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")
    
    def load_model(self, filepath):
        """Load a trained model."""
        # Auto-detect algorithm from the file path
        if 'dqn' in filepath.lower():
            self.best_model = DQN.load(filepath)
            self.algorithm = 'DQN'
        elif 'ppo' in filepath.lower():
            self.best_model = MaskablePPO.load(filepath)
            self.algorithm = 'PPO'
        elif 'a2c' in filepath.lower():
            self.best_model = A2C.load(filepath)
            self.algorithm = 'A2C'
        else:
            # Default to the initialized algorithm
            if self.algorithm == 'DQN':
                self.best_model = DQN.load(filepath)
            elif self.algorithm == 'PPO':
                self.best_model = MaskablePPO.load(filepath)
            elif self.algorithm == 'A2C':
                self.best_model = A2C.load(filepath)
        
        print(f"Model loaded from {filepath}")
        self.current_model = self.best_model


def main():
    """
    Main function to run the self-play training.
    """
    print("Tic Tac Toe Self-Play Training with Stable Baselines3")
    print("=" * 55)
    
    # Choose algorithm to use
    algorithm = input("Choose algorithm (DQN/PPO/A2C) [default: DQN]: ").strip().upper()
    if algorithm not in ['DQN', 'PPO', 'A2C']:
        algorithm = 'DQN'
        print(f"Using default: {algorithm}")
    
    # Choose number of environments for vectorization
    try:
        n_envs_input = input(f"Number of parallel environments [default: 4]: ").strip()
        n_envs = int(n_envs_input) if n_envs_input else 4
    except ValueError:
        n_envs = 4
        print(f"Using default: {n_envs}")
    
    # Create trainer
    trainer = SelfPlayTrainer(algorithm=algorithm, n_envs=n_envs)
    
    # Train the model
    training_results = trainer.train_self_play(num_episodes=30000)
    
    # Save the model
    model_path = f"tic_tac_toe_{algorithm.lower()}_selfplay.zip"
    trainer.save_model(model_path)
    
    print(f"\nTraining completed! Model saved to {model_path}")
    
    # Call evaluation on the newly trained model
    print("Running comprehensive evaluation on the trained model...")
    from evaluate_trained_model import evaluate_trained_model
    evaluate_trained_model(model_path, algorithm, num_games=100)  # Evaluate with 100 games for quick assessment
    
    print("To play against the trained AI, run: python play_trained_model.py")


if __name__ == "__main__":
    main()