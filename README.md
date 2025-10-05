# Tic Tac Toe Self-Play Training with Stable Baselines3

This project implements a self-play training system for Tic Tac Toe using reinforcement learning with Stable Baselines3.

## Overview

The system trains an AI agent to play Tic Tac Toe by having it play against itself (self-play). The trained agent learns optimal strategies through reinforcement learning, gradually improving its performance.

## Files

- `tic_tac_toe_env.py`: Basic Tic Tac Toe environment compatible with Gymnasium
- `self_play_main.py`: Main self-play training implementation with Stable Baselines3 integration
- `demo.py`: Demo script to run training or play games

## Features

- **Gymnasium Compatible Environment**: Properly implements the Gymnasium environment interface
- **Multiple RL Algorithms**: Supports DQN, PPO, and A2C algorithms from Stable Baselines3
- **Self-Play Training**: Agents improve by playing against themselves
- **Evaluation System**: Built-in evaluation against random agents
- **Human vs AI Play**: Option to play against the trained AI
- **Model Persistence**: Save and load trained models

## Usage

### Run Training:
```bash
python demo.py train
```

### Run Demo Game:
```bash
python demo.py demo
```

### Select Algorithm:
During training, you can choose between DQN (default), PPO, or A2C algorithms.

## Self-Play Training Process

1. **Initialization**: Start with a randomly initialized neural network
2. **Game Play**: Agent plays games (either against itself or other agents)
3. **Experience Collection**: Collect game experiences for training
4. **Model Update**: Update the neural network based on game outcomes
5. **Iteration**: Repeat the process to improve the agent

The self-play approach allows the agent to:
- Learn from both sides of the game (X and O)
- Discover optimal strategies through self-improvement
- Gradually become stronger by playing against increasingly better opponents (itself)

## Implementation Details

- **State Representation**: 9-element vector representing the 3x3 board (empty=0, X=1, O=-1)
- **Action Space**: 9 discrete actions (positions 0-8)
- **Reward System**: +1 for win, -1 for loss, 0 for draw
- **Training Episodes**: Configurable number of games for training

## Results Example

During the training run shown in the output:
- The agent starts with random performance (many draws initially)
- Gradually learns to win more often as training progresses
- Achieves >50% win rate by the end of training
- Demonstrates clear improvement over the course of training

## Customization

You can modify the training parameters in `self_play_main.py`:
- Number of training episodes
- Learning rate
- Algorithm-specific parameters
- Opponent selection strategy during training

## Loading and Using Trained Models

After training, you can load the saved model to play against a human:

```bash
python play_trained_model.py [model_path] [algorithm]
```

For example:
```bash
python play_trained_model.py tic_tac_toe_dqn_selfplay.zip DQN
```

The trained model file will be saved as `tic_tac_toe_[algorithm]_selfplay.zip` after training.

## Model File Locations

- `tic_tac_toe_dqn_selfplay.zip`: DQN trained model
- `tic_tac_toe_ppo_selfplay.zip`: PPO trained model  
- `tic_tac_toe_a2c_selfplay.zip`: A2C trained model