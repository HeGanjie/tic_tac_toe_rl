#!/usr/bin/env python3
"""
Tic Tac Toe Self-Play Training Demo

This script demonstrates how to implement and run self-play training 
for Tic Tac Toe using Stable Baselines3 reinforcement learning algorithms.

The system trains an AI agent by having it play against itself or other agents,
gradually improving its strategy through reinforcement learning.
"""

import sys
import os
from typing import Optional

def run_self_play_demo():
    """
    Run the complete self-play training demo.
    """
    print("Tic Tac Toe Self-Play Training Demo")
    print("=" * 40)
    
    try:
        # Import the main training module
        from self_play_main import main as self_play_main
        
        print("\nStarting self-play training...")
        print("This will train an AI agent using reinforcement learning.")
        print("The agent will play against itself and improve over time.\n")
        
        # Run the main training function
        self_play_main()
        
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Make sure all required files are in the same directory.")
        return False
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_game_demo():
    """
    Run a simple game demo to show the environment works.
    """
    print("\nTic Tac Toe Environment Demo")
    print("=" * 30)
    
    try:
        from tic_tac_toe_env import TicTacToeEnv
        import random
        
        print("Playing a demo game between two random agents:\n")
        
        env = TicTacToeEnv()
        obs, info = env.reset()
        env.render()
        
        done = False
        step_count = 0
        
        while not done and step_count < 9:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # Random agent move
            action = random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Random agent played position {action}")
            env.render()
            
            if done:
                if info["winner"] == 1:
                    print("X (Random Agent 1) wins!")
                elif info["winner"] == -1:
                    print("O (Random Agent 2) wins!")
                else:
                    print("It's a draw!")
                break
                
            step_count += 1
        
    except ImportError as e:
        print(f"Error importing environment: {e}")
        return False
    except Exception as e:
        print(f"An error occurred in demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def show_usage():
    """
    Show usage information.
    """
    print("Usage Instructions:")
    print("1. Run self-play training: python demo.py train")
    print("2. Run game demo: python demo.py demo") 
    print("3. Run with default (training): python demo.py")
    print()
    print("Requirements:")
    print("- Python 3.7+")
    print("- Stable Baselines3")
    print("- Gymnasium")
    print("- PyTorch")
    print()


def main():
    """
    Main function to run the demo based on command line arguments.
    """
    print("Welcome to Tic Tac Toe Self-Play Training System!")
    print()
    
    # Show usage if requested or if invalid arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
        return
    
    # Determine what to run based on arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ['train', 'training']:
            run_self_play_demo()
        elif mode in ['demo', 'd', 'game']:
            run_game_demo()
        else:
            print(f"Unknown mode: {mode}")
            show_usage()
    else:
        # Default: run training
        print("No mode specified, running self-play training...")
        run_self_play_demo()


if __name__ == "__main__":
    main()