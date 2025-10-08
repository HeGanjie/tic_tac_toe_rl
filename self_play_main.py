import random
from typing import List, Tuple, Dict

import torch.nn as nn
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from networks.cnn_3_16_32 import CustomTicTacToeCNN
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
                features_extractor_kwargs=dict(features_dim=64, d_model=16, nhead=2, num_layers=1),
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
                net_arch=[],
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
                features_extractor_kwargs=dict(features_dim=64, d_model=16, nhead=2, num_layers=1),
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

    def play_game(self, model1, model2, render=False) -> Tuple[int, List[Dict]]:
        """
        Play a game between two models or agents using a single environment.
        
        Args:
            model1: First model/agent (player X)
            model2: Second model/agent (player O)
            render: Whether to render the game
            
        Returns:
            Tuple of (winner, game_experience)
        """
        env = self.single_env  # Use single environment for game playing
        obs, _ = env.reset()
        done = False
        game_experience = []
        player_turn = 1  # 1 for X (model1), -1 for O (model2)
        
        while not done:
            if render:
                env.render()
            
            # Get action based on current player
            if player_turn == 1:  # X's turn
                if hasattr(model1, 'predict'):  # It's an SB3 model
                    # Check if it's a maskable model that supports action masking (like MaskablePPO)
                    if hasattr(env, 'action_masks') and self.algorithm == 'PPO':
                        action_masks = env.action_masks()
                        action, _ = model1.predict(obs, deterministic=False, action_masks=action_masks)
                    else:
                        action, _ = model1.predict(obs, deterministic=False)
                else:  # It's a function (like random_agent)
                    action = model1(obs)
            else:  # O's turn
                if hasattr(model2, 'predict'):  # It's an SB3 model
                    # Check if it's a maskable model that supports action masking (like MaskablePPO)
                    if hasattr(env, 'action_masks') and self.algorithm == 'PPO':
                        action_masks = env.action_masks()
                        action, _ = model2.predict(obs, deterministic=False, action_masks=action_masks)
                    else:
                        action, _ = model2.predict(obs, deterministic=False)
                else:  # It's a function (like random_agent)
                    action = model2(obs)
            
            # Store experience before taking action
            prev_obs = obs.copy()
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            
            # Append to game experience
            game_experience.append({
                'state': prev_obs,
                'action': action,
                'reward': reward,
                'next_state': obs,
                'done': done,
                'player': player_turn
            })
            
            # Switch player only if game continues (no invalid action penalty that ends game)
            if not done:
                player_turn *= -1
        
        if render:
            env.render()
            winner = info.get('winner', 0)
            if winner == 1:
                print("X (Model 1) wins!")
            elif winner == -1:
                print("O (Model 2) wins!")
            else:
                print("It's a draw!")
        
        return info.get('winner', 0), game_experience

    def train_self_play(self, num_episodes=10000):
        """
        Train the model using self-play with balanced player roles.
        Uses vectorized environments for faster training.
        """
        print(f"Starting self-play training with {self.algorithm} for {num_episodes} episodes...")
        
        # Since we're using vectorized environments, we'll let the RL algorithm handle
        # the interaction with the environment and learning process directly.
        # The model will learn through the normal SB3 training process.
        
        # We'll run the training for the specified number of timesteps
        if self.algorithm == 'DQN':
            # For DQN, train for the equivalent timesteps
            self.current_model.learn(total_timesteps=num_episodes * 10, progress_bar=True)  # Approximate
        elif self.algorithm == 'PPO':
            # For PPO, train for the specified number of timesteps
            self.current_model.learn(total_timesteps=num_episodes * 10, progress_bar=True)  # Approximate
        elif self.algorithm == 'A2C':
            # For A2C, train for the specified number of timesteps
            self.current_model.learn(total_timesteps=num_episodes * 10, progress_bar=True)  # Approximate
        
        # For actual game statistics during training with self-play between two models, 
        # we can evaluate the model periodically during training or after training
        print("Self-play training completed!")
        
        # Evaluate the final model by playing games
        wins_agent1, wins_agent2, draws = self._evaluate_during_training()
        
        total_games = wins_agent1 + wins_agent2 + draws
        print(f"Final Results:")
        print(f"X wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
        print(f"O wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        self.best_model = self.current_model
        return {
            'wins_agent1': wins_agent1,
            'wins_agent2': wins_agent2,
            'draws': draws,
            'total_games': total_games
        }

    def _evaluate_during_training(self, num_eval_games=100):
        """
        Evaluate the current model by playing games against itself or a random agent,
        to get statistics on performance. This is run after training.
        """
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        
        for i in range(num_eval_games):
            # Alternate which model plays as X and O to balance the statistics
            if i % 2 == 0:
                winner, _ = self.play_game(self.current_model, self._random_agent)
                if winner == 1: wins_agent1 += 1
                elif winner == -1: wins_agent2 += 1
                else: draws += 1
            else:
                winner, _ = self.play_game(self._random_agent, self.current_model)
                if winner == 1: wins_agent2 += 1  # From the other player's perspective
                elif winner == -1: wins_agent1 += 1  # From the other player's perspective
                else: draws += 1
        
        return wins_agent1, wins_agent2, draws
    
    @staticmethod
    def _random_agent(obs):
        """
        A simple random agent for baseline comparison.
        obs is the board state from the current player's perspective in 3-channel format.
        obs shape is (3, 3, 3) where:
        - obs[0]: Current player's pieces
        - obs[1]: Opponent's pieces  
        - obs[2]: Empty positions
        """
        # Find valid actions from the empty positions channel (channel 2)
        empty_positions = obs[2]  # This contains 1 where positions are empty, 0 otherwise
        valid_actions = []
        
        for i in range(9):
            row, col = divmod(i, 3)
            if empty_positions[row, col] == 1:  # If position is empty
                valid_actions.append(i)
        
        if valid_actions:
            return random.choice(valid_actions)
        else:
            return 0  # fallback action
    
    def evaluate_model(self, num_games=1000):
        """
        Evaluate the trained model against various opponents.
        """
        print(f"\nEvaluating the trained model over {num_games} games...")
        
        # Test against random agent
        print("\nTesting against random agent:")
        random_wins = 0
        model_wins = 0
        draws = 0
        
        for i in range(num_games):
            winner, _ = self.play_game(self.best_model, self._random_agent)
            if winner == 1:
                model_wins += 1
            elif winner == -1:
                random_wins += 1
            else:
                draws += 1
        
        print(f"Model vs Random - Model wins: {model_wins}, Random wins: {random_wins}, Draws: {draws}")
        
        # Test self-play (model vs itself)
        print("\nTesting self-play (model vs model):")
        x_wins = 0
        o_wins = 0
        self_draws = 0
        
        for i in range(num_games):
            winner, _ = self.play_game(self.best_model, self.best_model)
            if winner == 1:
                x_wins += 1
            elif winner == -1:
                o_wins += 1
            else:
                self_draws += 1
        
        print(f"Self-play - X wins: {x_wins}, O wins: {o_wins}, Draws: {self_draws}")
    
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
    
    # Evaluate the model
    trainer.evaluate_model(num_games=100)
    
    # Save the model
    model_path = f"tic_tac_toe_{algorithm.lower()}_selfplay.zip"
    trainer.save_model(model_path)
    
    print(f"\nTraining completed! Model saved to {model_path}")
    print("To play against the trained AI, run: python play_trained_model.py")


if __name__ == "__main__":
    main()