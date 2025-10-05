import numpy as np
import random
from typing import List, Tuple, Dict
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from tic_tac_toe_env import TicTacToeEnv
import torch
import torch.nn as nn
import torch.optim as optim
import os


class TicTacToeSB3Env:
    """
    Wrapper for Tic Tac Toe environment compatible with Stable Baselines3.
    """
    
    def __init__(self):
        self.action_space = type('ActionSpace', (), {'n': 9})()  # Discrete(9)
        self.observation_space = type('ObservationSpace', (), {'shape': (9,), 'dtype': np.float32})()
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.board = np.zeros((9,), dtype=np.float32)
        self.current_player = 1  # 1 for X (agent), -1 for O (opponent)
        self.done = False
        self.winner = None
        return self.board.copy(), {}
    
    def step(self, action):
        """Execute one time step of the environment."""
        # Check if action is valid
        if action < 0 or action > 8 or self.board[action] != 0:
            # Invalid action - penalize and end game
            return self.board.copy(), -10, True, False, {"error": "Invalid action"}
        
        # Apply the action to the board
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
        
        return self.board.copy(), reward, done, False, {"winner": self.winner}
    
    def _check_winner(self):
        """Check if there's a winner."""
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
    
    def render(self):
        """Render the current state of the board to the CLI."""
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


class SelfPlayTrainer:
    """
    Advanced self-play trainer for Tic Tac Toe using Stable Baselines3.
    """
    
    def __init__(self, algorithm='DQN', learning_rate=1e-4):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.best_model = None
        self.current_model = None
        self.game_history = []
        
        # Initialize model based on algorithm
        self._init_model()
    
    def _init_model(self):
        """Initialize the RL model based on the selected algorithm."""
        # Create a temporary environment for model initialization
        temp_env = TicTacToeSB3Env()
        
        if self.algorithm == 'DQN':
            self.current_model = DQN(
                "MlpPolicy",
                temp_env,
                learning_rate=self.learning_rate,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=32,
                target_update_interval=500,
                train_freq=1,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                verbose=0
            )
        elif self.algorithm == 'PPO':
            self.current_model = PPO(
                "MlpPolicy",
                temp_env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                clip_range=0.2,
                verbose=0
            )
        elif self.algorithm == 'A2C':
            self.current_model = A2C(
                "MlpPolicy",
                temp_env,
                learning_rate=self.learning_rate,
                n_steps=5,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def play_game(self, model1, model2, render=False) -> Tuple[int, List[Dict]]:
        """
        Play a game between two models.
        
        Args:
            model1: First model (player X)
            model2: Second model (player O)
            render: Whether to render the game
            
        Returns:
            Tuple of (winner, game_experience)
        """
        env = TicTacToeSB3Env()
        obs, _ = env.reset()
        done = False
        game_experience = []
        player_turn = 1  # 1 for X (model1), -1 for O (model2)
        
        while not done:
            if render:
                env.render()
            
            # Get action based on current player
            if player_turn == 1:  # X's turn
                action, _ = model1.predict(obs, deterministic=False)
            else:  # O's turn
                action, _ = model2.predict(obs, deterministic=False)
            
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
            
            # Switch player
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
    
    def generate_training_data(self, num_games=1000):
        """
        Generate training data by playing games between models.
        """
        print(f"Generating training data from {num_games} games...")
        
        training_data = []
        wins_1, wins_2, draws = 0, 0, 0
        
        opponent_model = None
        
        for game_idx in range(num_games):
            # Create an opponent model (could be random, previous version, or same model)
            if game_idx < 200:  # First 200 games: random opponent
                opponent_model = "random"
            elif game_idx < 500:  # Next 300 games: mix of random and current model
                opponent_model = self.current_model if random.random() < 0.5 else "random"
            else:  # Remaining games: mostly current model
                opponent_model = self.current_model
            
            # For self-play, we'll make both players use the current model
            # but with some randomness to encourage exploration
            winner, game_exp = self.play_game(
                self.current_model, 
                self.current_model if opponent_model != "random" else self.current_model, 
                render=(game_idx % 1000 == 0)
            )
            
            # Adjust rewards for training
            for step in game_exp:
                # Convert player-specific rewards to self-play rewards if needed
                pass
            
            training_data.extend(game_exp)
            
            # Update stats
            if winner == 1:
                wins_1 += 1
            elif winner == -1:
                wins_2 += 1
            else:
                draws += 1
            
            # Train on experience periodically
            if len(training_data) >= 100 and game_idx % 10 == 0:
                # Limit the training data to last 1000 experiences
                recent_data = training_data[-1000:]
                
                # Train the model
                if self.algorithm == 'DQN':
                    # For DQN, we'll need to train on the collected experiences
                    # For now, we'll just train for a few steps
                    pass  # The actual training happens during the model's learning process
                elif self.algorithm in ['PPO', 'A2C']:
                    # Policy gradient methods update based on full episodes
                    pass
        
        print(f"Training data generated. Results: X:{wins_1}, O:{wins_2}, Draws:{draws}")
        return training_data
    
    def train_self_play(self, num_episodes=10000):
        """
        Train the model using self-play.
        """
        print(f"Starting self-play training with {self.algorithm} for {num_episodes} episodes...")
        
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        
        # The main training loop will play games and update the model
        for episode in range(num_episodes):
            # Determine opponent strategy based on training stage
            if episode < 500:  # Early training: mostly random opponents
                opponent = "random"
            elif episode < 2000:  # Mid training: mix of random and current model
                opponent = self.current_model if random.random() < 0.7 else "random"
            else:  # Later training: mostly current model with some exploration
                opponent = self.current_model if random.random() < 0.9 else "random"
            
            # Play a game
            if opponent == "random":
                # Play against a random player
                winner, _ = self.play_game(self.current_model, self._random_agent, 
                                          render=(episode % 2000 == 0))
            else:
                # Self-play against a copy of the current model
                winner, _ = self.play_game(self.current_model, self.current_model,
                                          render=(episode % 2000 == 0))
            
            # Update statistics
            if winner == 1:
                wins_agent1 += 1
            elif winner == -1:
                wins_agent2 += 1
            else:
                draws += 1
            
            # Periodically train the model
            if episode > 0 and episode % 50 == 0:
                # Train the model based on the algorithm
                if self.algorithm == 'DQN' and episode > 100:
                    self.current_model.learn(total_timesteps=500, reset_num_timesteps=False)
                elif self.algorithm in ['PPO', 'A2C'] and episode % 200 == 0:
                    # Policy gradient methods need more experiences before updating
                    pass  # Will be updated in batches
            
            # Print progress
            if (episode + 1) % 1000 == 0:
                total_games = episode + 1
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"X wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
                print(f"O wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
                print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
                print("-" * 40)
        
        print("Self-play training completed!")
        total_games = num_episodes
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
    
    def _random_agent(self, obs):
        """
        A simple random agent for baseline comparison.
        """
        valid_actions = [i for i in range(9) if obs[i] == 0]
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
        
        for i in range(num_games // 2):
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
        
        for i in range(num_games // 2):
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
        if self.algorithm == 'DQN':
            self.best_model = DQN.load(filepath)
        elif self.algorithm == 'PPO':
            self.best_model = PPO.load(filepath)
        elif self.algorithm == 'A2C':
            self.best_model = A2C.load(filepath)
        
        print(f"Model loaded from {filepath}")
        self.current_model = self.best_model


def play_human_vs_ai(trainer: SelfPlayTrainer):
    """
    Allow a human to play against the trained AI.
    """
    print("\nPlaying against the trained AI!")
    print("You are X, the AI is O.")
    print("Positions are numbered from 0-8 as follows:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 ")
    print()
    
    env = TicTacToeSB3Env()
    obs, _ = env.reset()
    done = False
    
    while not done:
        env.render()
        
        # Human's turn
        valid_actions = [i for i in range(9) if obs[i] == 0]
        print(f"Valid moves: {valid_actions}")
        
        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if move in valid_actions:
                    break
                else:
                    print("Invalid move! Please select from valid moves.")
            except ValueError:
                print("Please enter a number between 0-8.")
        
        obs, reward, done, truncated, info = env.step(move)
        
        if done:
            env.render()
            winner = info.get('winner', 0)
            if winner == 1:
                print("Congratulations! You won!")
            elif winner == -1:
                print("AI wins! Better luck next time.")
            else:
                print("It's a draw!")
            break
        
        if not done:
            # AI's turn
            print("AI is thinking...")
            ai_action, _ = trainer.best_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(ai_action)
            
            if done:
                env.render()
                winner = info.get('winner', 0)
                if winner == 1:
                    print("Congratulations! You won!")
                elif winner == -1:
                    print("AI wins! Better luck next time.")
                else:
                    print("It's a draw!")


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
    
    # Create trainer
    trainer = SelfPlayTrainer(algorithm=algorithm)
    
    # Train the model
    training_results = trainer.train_self_play(num_episodes=5000)
    
    # Evaluate the model
    trainer.evaluate_model(num_games=1000)
    
    # Save the model
    model_path = f"tic_tac_toe_{algorithm.lower()}_selfplay.zip"
    trainer.save_model(model_path)
    
    # Play against human
    play_choice = input("\nWould you like to play against the trained AI? (y/n): ").strip().lower()
    if play_choice == 'y':
        play_human_vs_ai(trainer)


if __name__ == "__main__":
    main()