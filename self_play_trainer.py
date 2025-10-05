import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
import pickle
import os
from tic_tac_toe_env import TicTacToeEnv


class SelfPlayTrainer:
    """
    Self-play trainer for Tic Tac Toe.
    Two agents (often the same model) play against each other to improve.
    """
    
    def __init__(self, model=None, num_episodes: int = 10000):
        self.model = model  # The neural network model (e.g., DQN, PPO)
        self.num_episodes = num_episodes
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        
    def play_game(self, agent1, agent2, render: bool = False) -> Tuple[List, List, int]:
        """
        Play a single game between two agents.
        
        Args:
            agent1: First agent (player X)
            agent2: Second agent (player O)
            render: Whether to render the game
            
        Returns:
            Tuple of (game_history_agent1, game_history_agent2, winner)
        """
        env = TicTacToeEnv()
        obs, _ = env.reset()
        done = False
        
        game_history_1 = []  # For agent 1 (X)
        game_history_2 = []  # For agent 2 (O)
        
        # Track which agent is playing at each step
        current_agent = 1  # 1 for agent 1 (X), -1 for agent 2 (O)
        
        while not done:
            if render:
                env.render()
            
            valid_actions = env.get_valid_actions()
            
            if current_agent == 1:  # Agent 1's turn (X)
                if self.model is not None:
                    # Use the model to select action
                    action = self.select_action_with_model(env, agent1, valid_actions)
                else:
                    # Random agent for now
                    action = random.choice(valid_actions)
                
                game_history_1.append({
                    'state': obs.copy(),
                    'action': action,
                    'valid_actions': valid_actions.copy()
                })
            else:  # Agent 2's turn (O)
                if self.model is not None:
                    # Use the model to select action
                    action = self.select_action_with_model(env, agent2, valid_actions)
                else:
                    # Random agent for now
                    action = random.choice(valid_actions)
                
                game_history_2.append({
                    'state': obs.copy(),
                    'action': action,
                    'valid_actions': valid_actions.copy()
                })
            
            obs, reward, done, truncated, info = env.step(action)
            current_agent *= -1  # Switch players
        
        winner = info.get('winner', 0)
        
        # Assign rewards based on the outcome
        if winner == 1:  # Agent 1 (X) wins
            reward_1, reward_2 = 1, -1
        elif winner == -1:  # Agent 2 (O) wins
            reward_1, reward_2 = -1, 1
        else:  # Draw
            reward_1, reward_2 = 0, 0
            
        # Add final rewards to histories
        if game_history_1:  # Agent 1 played at least one move
            game_history_1[-1]['reward'] = reward_1
            game_history_1[-1]['done'] = done
            game_history_1[-1]['next_state'] = obs.copy()
        
        if game_history_2:  # Agent 2 played at least one move
            game_history_2[-1]['reward'] = reward_2
            game_history_2[-1]['done'] = done
            game_history_2[-1]['next_state'] = obs.copy()
        
        # Add opponent's final state to each agent's history
        for step in game_history_1:
            step['opponent_state'] = obs.copy()
            step['opponent_reward'] = reward_2
            
        for step in game_history_2:
            step['opponent_state'] = obs.copy()
            step['opponent_reward'] = reward_1
        
        return game_history_1, game_history_2, winner
    
    def select_action_with_model(self, env, agent, valid_actions):
        """
        Select an action using the neural network model.
        This is a placeholder that needs to be implemented based on your specific algorithm.
        """
        # For now, we'll use epsilon-greedy approach if we have a model
        # This will be expanded later for specific RL algorithms
        if random.random() < 0.1:  # 10% chance of random move for exploration
            return random.choice(valid_actions)
        else:
            # Placeholder: random choice
            # In real implementation, this would use the model to predict the best action
            return random.choice(valid_actions)
    
    def train_episode(self, episode_num: int):
        """
        Run a single training episode with self-play.
        """
        # Create two copies of the current best agent
        # For the first few episodes, we might use random agents
        if episode_num < 100:
            agent1 = "random"
            agent2 = "random"
        elif episode_num < 500:
            # Mix of random and previous best agent
            agent1 = "random" if random.random() < 0.5 else self.model
            agent2 = "random" if random.random() < 0.5 else self.model
        else:
            # Mostly use the current best agent
            agent1 = self.model if self.model is not None else "random"
            agent2 = self.model if self.model is not None else "random"
        
        # Add some noise to agent2 to encourage exploration
        if episode_num >= 500 and self.model is not None:
            # Add some randomization to agent2 to encourage exploration
            # This helps to generate more diverse game experiences
            agent2 = self.model  # Could implement some noise here
        
        game_hist_1, game_hist_2, winner = self.play_game(agent1, agent2, 
                                                          render=(episode_num % 1000 == 0))
        
        # Store game experiences for training
        self.store_experience(game_hist_1, winner, 1)
        self.store_experience(game_hist_2, winner, -1)
        
        return winner
    
    def store_experience(self, game_history: List, winner: int, player: int):
        """
        Store game experience in memory for training.
        """
        for step in game_history:
            # Add the experience to replay buffer
            self.memory.append(step)
    
    def train(self):
        """
        Main training loop for self-play.
        """
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        
        print(f"Starting self-play training for {self.num_episodes} episodes...")
        
        for episode in range(self.num_episodes):
            winner = self.train_episode(episode)
            
            # Update statistics
            if winner == 1:
                wins_agent1 += 1
            elif winner == -1:
                wins_agent2 += 1
            else:
                draws += 1
            
            # Print progress
            if (episode + 1) % 1000 == 0:
                total_games = episode + 1
                print(f"Episode {episode + 1}/{self.num_episodes}")
                print(f"X wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
                print(f"O wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
                print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
                print("-" * 40)
        
        print("Training completed!")
        total_games = self.num_episodes
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
    
    def save_memory(self, filepath: str):
        """Save the experience replay buffer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory), f)
        print(f"Memory saved to {filepath}")
    
    def load_memory(self, filepath: str):
        """Load the experience replay buffer from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                loaded_memory = pickle.load(f)
            self.memory = deque(loaded_memory, maxlen=10000)
            print(f"Memory loaded from {filepath}, size: {len(self.memory)}")
        else:
            print(f"File {filepath} not found")


def evaluate_agents(trainer: SelfPlayTrainer, num_games: int = 1000):
    """
    Evaluate how well the trained agents perform.
    """
    print(f"\nEvaluating agents over {num_games} games...")
    
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    
    for i in range(num_games):
        game_hist_1, game_hist_2, winner = trainer.play_game(
            trainer.model if trainer.model is not None else "random",
            trainer.model if trainer.model is not None else "random"
        )
        
        if winner == 1:
            wins_agent1 += 1
        elif winner == -1:
            wins_agent2 += 1
        else:
            draws += 1
    
    total_games = num_games
    print(f"Evaluation Results:")
    print(f"Player 1 wins: {wins_agent1} ({wins_agent1/total_games*100:.1f}%)")
    print(f"Player 2 wins: {wins_agent2} ({wins_agent2/total_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
    
    return {
        'wins_agent1': wins_agent1,
        'wins_agent2': wins_agent2,
        'draws': draws,
        'total_games': total_games
    }


if __name__ == "__main__":
    # Create the trainer
    trainer = SelfPlayTrainer(num_episodes=5000)
    
    # Train the agents
    results = trainer.train()
    
    # Evaluate the agents
    eval_results = evaluate_agents(trainer, num_games=1000)
    
    # Save the experience memory
    trainer.save_memory("tic_tac_toe_experience.pkl")