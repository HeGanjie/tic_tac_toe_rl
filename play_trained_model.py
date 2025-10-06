#!/usr/bin/env python3
"""
Load Trained Model and Play Against Human

This script loads a previously trained Tic Tac Toe model and allows a human
to play against it. If no trained model exists, it will train a new one first.
"""

import os
import sys
from self_play_main import SelfPlayTrainer
from tic_tac_toe_env import TicTacToeEnv


def play_against_trained_model(model_path="tic_tac_toe_dqn_selfplay.zip", algorithm="DQN"):
    """
    Load a trained model and allow human to play against it.
    
    Args:
        model_path: Path to the saved model file
        algorithm: The algorithm used to train the model ('DQN', 'PPO', or 'A2C')
    """
    print("Tic Tac Toe: Human vs Trained AI")
    print("=" * 35)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Training a new model first...")
        
        # Train a new model
        trainer = SelfPlayTrainer(algorithm=algorithm)
        print(f"Training a {algorithm} model for 2000 episodes...")
        training_results = trainer.train_self_play(num_episodes=2000)
        trainer.save_model(model_path)
        
        # Set the trained model as the best model for play
        best_model = trainer.best_model
    else:
        print(f"Loading trained model from {model_path}...")
        trainer = SelfPlayTrainer(algorithm=algorithm)  # This will be updated by load_model
        trainer.load_model(model_path)
        print("Model loaded successfully!")
    
    print("\nYou are X, the AI is O.")
    print("Positions are numbered from 0-8 as follows:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 ")
    print()
    
    while True:
        # Start a new game
        env = TicTacToeEnv()
        obs, _ = env.reset()
        done = False
        
        print("\nNew game started!")
        env.render()
        
        while not done:
            # Human's turn
            valid_actions = [i for i in range(9) if obs[i] == 0]
            if not valid_actions:
                break
                
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
            env.render()
            
            if done:
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
                # Use action masks if available
                if hasattr(env, 'action_masks'):
                    action_masks = env.action_masks()
                    ai_action, _ = trainer.best_model.predict(obs, deterministic=True, action_masks=action_masks)
                else:
                    ai_action, _ = trainer.best_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(ai_action)
                env.render()
                
                if done:
                    winner = info.get('winner', 0)
                    if winner == 1:
                        print("Congratulations! You won!")
                    elif winner == -1:
                        print("AI wins! Better luck next time.")
                    else:
                        print("It's a draw!")
        
        # Ask if player wants to play again
        play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
        if play_again != 'y':
            break
    
    print("Thanks for playing!")


def main():
    """
    Main function to run the human vs trained AI game.
    """
    print("Loading Trained Model to Play Against Human")
    print("=" * 45)
    
    # Default model path and algorithm
    model_path = "tic_tac_toe_dqn_selfplay.zip"
    algorithm = "DQN"
    
    # Allow specifying model path as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        algorithm = sys.argv[2].upper()
        if algorithm not in ["DQN", "PPO", "A2C"]:
            print(f"Warning: Unsupported algorithm {algorithm}, using DQN instead")
            algorithm = "DQN"
    
    try:
        play_against_trained_model(model_path, algorithm)
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()