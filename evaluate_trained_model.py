#!/usr/bin/env python3
"""
评估已训练的井字棋模型

此脚本加载一个已训练的模型并进行多种评估
"""

import sys
from self_play_main import SelfPlayTrainer


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
    
    # 创建训练器
    trainer = SelfPlayTrainer(algorithm=algorithm)
    
    # 加载模型
    trainer.load_model(model_path)
    
    print(f"模型加载成功！")
    print()
    
    # 使用内置的评估方法
    trainer.evaluate_model(num_games=num_games)
    
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