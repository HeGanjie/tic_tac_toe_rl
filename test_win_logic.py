"""
Test the Tic Tac Toe environment to verify win detection logic
"""
import numpy as np
from tic_tac_toe_env import TicTacToeEnv


def test_win_detection():
    """Test if win detection works correctly."""
    print("Testing win detection logic...")
    
    env = TicTacToeEnv()
    
    # Test case 1: Diagonal win for X
    print("\nTest 1: Diagonal win for X (0, 4, 8)")
    env.reset()
    env.board[0, 0] = 1   # X at position 0
    env.board[1, 1] = 1   # X at position 4
    env.board[2, 2] = 1   # X at position 8
    print("Board:")
    env.render()
    winner = env._check_winner()
    print(f"Detected winner: {winner} (expected: 1 for X)")
    
    # Test case 2: Human about to win (0, 4) -> 8, should be blocked at 8
    print("\nTest 2: X has positions 0 and 4, next move at 8 should win")
    env.reset()
    env.board[0, 0] = 1   # X at position 0 (top-left)
    env.board[1, 1] = 1   # X at position 4 (center)
    env.board[2, 2] = 0   # Position 8 (bottom-right) is empty
    print("Board:")
    env.render()
    # Simulate X placing at position 8
    row, col = divmod(8, 3)
    env.board[row, col] = 1
    winner = env._check_winner()
    print(f"After X places at 8, winner: {winner} (expected: 1 for X)")
    print("Board after X at 8:")
    env.render()
    
    # Test case 3: Proper game sequence from the example
    print("\nTest 3: Reproducing the game sequence")
    env.reset()
    # Human places X at 4 (center)
    row, col = divmod(4, 3)
    env.board[row, col] = 1
    print("After human places X at 4:")
    env.render()
    
    # AI places O at 2 (top-right) 
    row, col = divmod(2, 3)
    env.board[row, col] = -1
    print("After AI places O at 2:")
    env.render()
    
    # Human places X at 0 (top-left)
    row, col = divmod(0, 3)
    env.board[row, col] = 1
    print("After human places X at 0:")
    env.render()
    
    # At this point human has X at 0 and 4, could win at 8
    # AI should place at 8 to block
    winner = env._check_winner()
    print(f"Current winner: {winner}")
    
    # If AI plays optimally, it should block at position 8
    # But if it doesn't, human can win on next move
    print("If human now plays at 8, they should win")
    row, col = divmod(8, 3)
    env.board[row, col] = 1
    winner = env._check_winner()
    print(f"After human places X at 8, winner: {winner} (expected: 1)")
    env.render()


def test_environment_with_manual_play():
    """Simulate the exact game scenario."""
    print("\n" + "="*50)
    print("Simulating the actual game scenario:")
    print("Human (X) vs AI (O)")
    print("="*50)
    
    env = TicTacToeEnv()
    obs, _ = env.reset()
    
    print("\nInitial board:")
    env.render()
    
    # Move 1: Human places X at 4 (center)
    obs, reward, done, truncated, info = env.step(4)  # Human places X at center
    print("Human places X at 4 (center):")
    env.render()
    
    # Move 2: AI places O
    # We need to simulate what the trained agent would do
    # For now, we'll manually place O at position 2 (top-right)
    obs, reward, done, truncated, info = env.step(2)  # AI places O at top-right
    print("AI places O at 2 (top-right):")
    env.render()
    
    # Move 3: Human places X at 0 (top-left)
    obs, reward, done, truncated, info = env.step(0)  # Human places X at top-left
    print("Human places X at 0 (top-left):")
    env.render()
    
    print("Current state - Human has positions 0 and 4")
    print("Human can win by placing at 8 (bottom-right) - the diagonal")
    print("AI should block at 8 to prevent the win")
    
    # At this point, if AI places anywhere other than 8, human wins
    # If AI places at 8, the game continues
    winner = env._check_winner()
    print(f"Current winner: {winner}")
    print(f"Game done: {done}")
    
    # The key question: where would the trained AI place?
    # The trained model should have learned to block


if __name__ == "__main__":
    test_win_detection()
    test_environment_with_manual_play()