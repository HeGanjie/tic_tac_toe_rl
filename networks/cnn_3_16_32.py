import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn as nn


class CustomTicTacToeCNN(BaseFeaturesExtractor):
    """
    一个为3x3x3井字棋棋盘设计的轻量级CNN。
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        # 输入: (3, 3, 3) -> [Channels, Height, Width]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2), # 3x3 -> 2x2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2), # 2x2 -> 1x1
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算CNN输出维度
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # 调整维度顺序: [batch, H, W, C] -> [batch, C, H, W]
        # Actually, our observations are already in channel-first format (C, H, W)
        # So [batch, C, H, W] - no permutation needed
        return self.linear(self.cnn(observations))
