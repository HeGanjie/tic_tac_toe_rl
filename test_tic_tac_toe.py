import numpy as np
from tic_tac_toe_env import TicTacToeEnv


def test_tic_tac_toe_env():
    """Test the Tic Tac Toe environment implementation."""
    print("Testing Tic Tac Toe Environment...")
    
    # Create environment
    env = TicTacToeEnv()
    
    # Test reset
    print("\n1. Testing reset method:")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial board shape: {env.board.shape}")
    print(f"Board after reset:\n{env.board}")
    
    # Test action space
    print(f"\n2. Testing action space: {env.action_space}")
    print(f"Action space sample: {env.action_space.sample()}")
    
    # Test observation space
    print(f"\n3. Testing observation space: {env.observation_space}")
    
    # Test a simple game sequence
    print("\n4. Testing game sequence:")
    env.reset()
    
    # Sample play: X in center (position 4), O in top-left (position 0)
    obs, reward, done, truncated, info = env.step(4)  # X in center
    print(f"Action 4 (X in center) - Reward: {reward}, Done: {done}")
    env.render()
    
    obs, reward, done, truncated, info = env.step(0)  # O in top-left
    print(f"Action 0 (O in top-left) - Reward: {reward}, Done: {done}")
    env.render()
    
    # X in top-right (position 2)
    obs, reward, done, truncated, info = env.step(2)  # X in top-right
    print(f"Action 2 (X in top-right) - Reward: {reward}, Done: {done}")
    env.render()
    
    # O in middle left (position 3)
    obs, reward, done, truncated, info = env.step(3)  # O in middle left
    print(f"Action 3 (O in middle left) - Reward: {reward}, Done: {done}")
    env.render()
    
    # X wins by completing top row
    obs, reward, done, truncated, info = env.step(1)  # X in top-middle
    print(f"Action 1 (X in top-middle) - Reward: {reward}, Done: {done}, Winner: {info.get('winner')}")
    env.render()
    
    # Test valid actions
    print(f"\n5. Testing valid actions: {env.get_valid_actions()}")
    
    # Test invalid action
    print("\n6. Testing invalid action:")
    env.reset()
    env.step(0)  # Place X at position 0
    obs, reward, done, truncated, info = env.step(0)  # Try to place again at position 0 (should be invalid)
    print(f"Invalid action result - Reward: {reward}, Done: {done}, Info: {info}")
    
    print("\nEnvironment testing completed successfully!")


def test_win_conditions():
    """Test different win conditions."""
    print("\nTesting win conditions...")
    env = TicTacToeEnv()
    
    # Test horizontal win (top row)
    print("\nHorizontal win (top row):")
    env.reset()
    env.board[0, 0] = 1  # X
    env.board[0, 1] = 1  # X 
    env.board[0, 2] = 1  # X
    env.render()
    winner = env._check_winner()
    print(f"Winner: {winner} (expected: 1)")
    
    # Test vertical win (middle column)
    print("\nVertical win (middle column):")
    env.reset()
    env.board[0, 1] = -1  # O
    env.board[1, 1] = -1  # O
    env.board[2, 1] = -1  # O
    env.render()
    winner = env._check_winner()
    print(f"Winner: {winner} (expected: -1)")
    
    # Test diagonal win (top-left to bottom-right)
    print("\nDiagonal win (top-left to bottom-right):")
    env.reset()
    env.board[0, 0] = 1  # X
    env.board[1, 1] = 1  # X
    env.board[2, 2] = 1  # X
    env.render()
    winner = env._check_winner()
    print(f"Winner: {winner} (expected: 1)")
    
    # Test draw
    print("\nTesting draw condition:")
    env.reset()
    # Fill the board without a winner
    env.board[0, 0] = 1
    env.board[0, 1] = -1
    env.board[0, 2] = 1
    env.board[1, 0] = 1
    env.board[1, 1] = -1
    env.board[1, 2] = -1
    env.board[2, 0] = -1
    env.board[2, 1] = 1
    env.board[2, 2] = 1
    env.render()
    winner = env._check_winner()
    is_full = env._is_board_full()
    print(f"Winner: {winner} (expected: 0), Is board full: {is_full} (expected: True)")
    

if __name__ == "__main__":
    test_tic_tac_toe_env()
    test_win_conditions()