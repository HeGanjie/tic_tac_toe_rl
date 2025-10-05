import numpy as np
from tic_tac_toe_env import TicTacToeEnv
import random


def play_against_random():
    """Play against a random agent to test the environment."""
    print("Welcome to Tic Tac Toe!")
    print("You are X, and the computer is O.")
    print("Positions are numbered as follows:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 ")
    print()
    
    env = TicTacToeEnv()
    obs, info = env.reset()
    env.render()
    
    while True:
        # Human's turn (X)
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
            
        print(f"Valid moves: {valid_actions}")
        
        while True:
            try:
                action = int(input("Enter your move (0-8): "))
                if action in valid_actions:
                    break
                else:
                    print("Invalid move! Please select an empty position from the valid moves.")
            except ValueError:
                print("Please enter a number between 0-8.")
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nYou played position {action}")
        env.render()
        
        if done:
            if info["winner"] == 1:  # Human wins
                print("Congratulations! You won!")
            elif info["winner"] == -1:  # Computer wins
                print("Computer wins! Better luck next time.")
            else:  # Draw
                print("It's a draw!")
            break
        
        # Computer's turn (O)
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        computer_action = random.choice(valid_actions)
        obs, reward, done, truncated, info = env.step(computer_action)
        print(f"Computer played position {computer_action}")
        env.render()
        
        if done:
            if info["winner"] == 1:  # Human wins
                print("Congratulations! You won!")
            elif info["winner"] == -1:  # Computer wins
                print("Computer wins! Better luck next time.")
            else:  # Draw
                print("It's a draw!")
            break


def demo_random_game():
    """Demonstrate a game between two random agents."""
    print("\n" + "="*50)
    print("Demo: Random Agent vs Random Agent")
    print("="*50)
    
    env = TicTacToeEnv()
    obs, info = env.reset()
    env.render()
    
    while True:
        valid_actions = env.get_valid_actions()
        if not valid_actions or info.get("done", False):
            break
            
        # Select random action
        action = random.choice(valid_actions)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Player {'X' if env.current_player == -1 else 'O'} played position {action}")
        env.render()
        
        if done:
            if info["winner"] == 1:
                print("X (Player 1) wins!")
            elif info["winner"] == -1:
                print("O (Player 2) wins!")
            else:
                print("It's a draw!")
            break


if __name__ == "__main__":
    choice = input("Choose an option:\n1. Play against random agent\n2. Watch random agents play\nEnter choice (1 or 2): ")
    
    if choice == "1":
        play_against_random()
    elif choice == "2":
        demo_random_game()
    else:
        print("Invalid choice. Running demo game...")
        demo_random_game()