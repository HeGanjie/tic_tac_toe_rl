import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn as nn
import torch as th

class CustomTicTacToeCNN(BaseFeaturesExtractor):
    """
    一个为井字棋棋盘设计的轻量级CNN，支持不同通道数。
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        # 动态获取通道数 from observation space
        n_channels = observation_space.shape[0]  # Get number of channels from observation space
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=2), # 3x3 -> 2x2 (or adjusted for different input sizes)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), # 2x2 -> 1x1 (or adjusted)
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


class CustomCNN_3x3(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        # 动态获取通道数 from observation space
        n_channels = observation_space.shape[0]  # Get number of channels from observation space
        
        self.cnn = nn.Sequential(
            # 层1: 3x3 核，看到整个棋盘
            nn.Conv2d(n_channels, 128, kernel_size=3, padding=0), # (C,3,3) -> (32,1,1) where C is number of channels
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, obs):
        # This assumes input is [B, C, H, W] (batch, channels, height, width)
        # If it's [B, H, W, C], we need to permute before passing to CNN
        # Obs should be in (B, C, H, W) format from gymnasium environments
        return self.linear(self.cnn(obs))