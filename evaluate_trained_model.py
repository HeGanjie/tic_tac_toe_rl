#!/usr/bin/env python3
"""
评估已训练的井字棋模型

此脚本加载一个已训练的模型并进行多种评估
"""

import sys
from typing import Tuple, Dict, List
import random
import numpy as np
from self_play_main import SelfPlayTrainer
from stable_baselines3 import DQN, A2C
from sb3_contrib import MaskablePPO


class TicTacToeEvaluator(SelfPlayTrainer):
    """
    井字棋模型评估器 - 专门用于评估已训练模型的性能
    继承 SelfPlayTrainer 以复用游戏逻辑和环境
    """
    
    def __init__(self, model_path: str, algorithm: str = 'PPO'):
        # 调用父类构造函数初始化算法
        super().__init__(algorithm=algorithm)
        # 加载已训练的模型
        self.load_model(model_path)
        # 设置评估模型
        self.model = self.best_model

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

    def evaluate_first_vs_second_player(self, num_games=1000):
        """
        Evaluate how well the trained model performs as first player vs second player
        by playing against the same strength opponent (e.g., a copy of itself or random agent).
        """
        print(f"\nEvaluating first-player vs second-player performance over {num_games} games...")
        
        # Use a fixed opponent (e.g., random agent) to isolate first vs second player effects
        x_wins = 0  # Model as X (first player) wins
        o_wins = 0  # Model as O (second player) wins
        draws = 0
        
        for i in range(num_games):
            # Alternate which model plays as X to balance the evaluation
            if i % 2 == 0:
                # Model is player X (first)
                winner, _ = self.play_game(self.model, self._random_agent)
                if winner == 1:
                    x_wins += 1  # Model (as X) won
                elif winner == -1:
                    o_wins += 1  # Opponent (as O) won
                else:
                    draws += 1
            else:
                # Model is player O (second)
                winner, _ = self.play_game(self._random_agent, self.model)
                if winner == 1:
                    x_wins += 1  # Opponent (as X) won
                elif winner == -1:
                    o_wins += 1  # Model (as O) won
                else:
                    draws += 1
        
        total_games = x_wins + o_wins + draws
        print(f"Model as X (first): {x_wins} ({x_wins/total_games*100:.1f}%)")
        print(f"Model as O (second): {o_wins} ({o_wins/total_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        if x_wins + o_wins > 0:  # Avoid division by zero
            win_rate_as_x = x_wins / (x_wins + o_wins) if (x_wins + o_wins) > 0 else 0
            print(f"Model's win rate when playing first: {win_rate_as_x:.2f}")
        else:
            print("Not enough non-draw games to calculate win rate difference")
    
    def comprehensive_evaluation(self, num_games=1000):
        """
        Perform a comprehensive evaluation of the trained model.
        """
        print("="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # 1. Performance vs random agent
        print("\n1. PERFORMANCE AGAINST RANDOM AGENT:")
        model_as_x_wins = 0
        random_as_o_wins = 0
        x_draws = 0
        
        for i in range(num_games // 2):
            winner, _ = self.play_game(self.model, self._random_agent)
            if winner == 1:
                model_as_x_wins += 1
            elif winner == -1:
                random_as_o_wins += 1
            else:
                x_draws += 1
        
        random_as_x_wins = 0
        model_as_o_wins = 0
        o_draws = 0
        
        for i in range(num_games // 2):
            winner, _ = self.play_game(self._random_agent, self.model)
            if winner == 1:
                random_as_x_wins += 1
            elif winner == -1:
                model_as_o_wins += 1
            else:
                o_draws += 1
        
        total_model_wins = model_as_x_wins + model_as_o_wins
        total_random_wins = random_as_o_wins + random_as_x_wins
        total_draws = x_draws + o_draws
        
        print(f"  As X (first): {model_as_x_wins}/{num_games//2} wins ({model_as_x_wins/(num_games//2)*100:.1f}%)")
        print(f"  As O (second): {model_as_o_wins}/{num_games//2} wins ({model_as_o_wins/(num_games//2)*100:.1f}%)")
        print(f"  Overall: Model wins {total_model_wins}/{num_games}, Random wins {total_random_wins}/{num_games}, Draws {total_draws}/{num_games}")
        print(f"  Overall win rate: {total_model_wins/num_games*100:.1f}%")
        
        # 2. First vs second player with same opponent
        print("\n2. FIRST vs SECOND PLAYER ADVANTAGE (vs random opponent):")
        self.evaluate_first_vs_second_player(num_games)
        
        # 3. Consistency against itself
        print("\n3. CONSISTENCY AGAINST SAME MODEL:")
        same_model_x_wins = 0
        same_model_o_wins = 0
        same_model_draws = 0
        
        for i in range(num_games):
            if i % 2 == 0:  # Model as X
                winner, _ = self.play_game(self.model, self.model)
                if winner == 1:
                    same_model_x_wins += 1  # First player wins
                elif winner == -1:
                    same_model_o_wins += 1  # Second player wins
                else:
                    same_model_draws += 1
            else:  # Model as O (in practice, still same model vs same model)
                winner, _ = self.play_game(self.model, self.model)
                if winner == 1:
                    same_model_x_wins += 1  # First player wins
                elif winner == -1:
                    same_model_o_wins += 1  # Second player wins
                else:
                    same_model_draws += 1
        
        print(f"  First player wins: {same_model_x_wins}/{num_games} ({same_model_x_wins/num_games*100:.1f}%)")
        print(f"  Second player wins: {same_model_o_wins}/{num_games} ({same_model_o_wins/num_games*100:.1f}%)")
        print(f"  Draws: {same_model_draws}/{num_games} ({same_model_draws/num_games*100:.1f}%)")
        
        # Summary
        print("\n4. SUMMARY:")
        print(f"  Model is {(model_as_x_wins/(num_games//2) / (model_as_o_wins/(num_games//2 + 0.001))):.2f}x more likely to win as first player vs random opponent")
        print(f"  First player advantage: {((same_model_x_wins)/(num_games) - (same_model_o_wins)/(num_games)):.2f} win rate difference")
        print("="*60)

    
def evaluate_trained_model(model_path, algorithm='PPO', num_games=1000):
    """
    评估训练好的模型性能
    
    Args:
        model_path: 模型文件路径
        algorithm: 算法名称 ('DQN', 'PPO', 'A2C')
        num_games: 评估游戏局数
    """
    print(f"开始评估模型: {model_path}")
    print(f"算法: {algorithm}")
    print(f"评估局数: {num_games}")
    print("=" * 50)
    
    # 创建评估器
    evaluator = TicTacToeEvaluator(model_path, algorithm)
    
    print(f"模型加载成功！")
    print()
    
    # 使用内置的综合评估方法
    evaluator.comprehensive_evaluation(num_games=num_games)
    
    print("=" * 50)
    print("评估完成！")


def main():
    """
    主函数
    """
    print("井字棋模型评估工具")
    print("=" * 30)
    
    # 默认参数
    model_path = "tic_tac_toe_ppo_selfplay.zip"
    algorithm = "PPO"
    num_games = 1000
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        algorithm = sys.argv[2].upper()
        if algorithm not in ["DQN", "PPO", "A2C"]:
            print(f"警告: 不支持的算法 {algorithm}，使用默认 PPO")
            algorithm = "PPO"
    if len(sys.argv) > 3:
        try:
            num_games = int(sys.argv[3])
        except ValueError:
            print(f"警告: 游戏局数必须是数字，使用默认 1000")
            num_games = 1000
    
    try:
        evaluate_trained_model(model_path, algorithm, num_games)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确保模型文件存在，或提供正确的路径")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()