import torch.nn as nn
import torch.nn.functional as F
import torch as th


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径：两个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径（Shortcut）路径：如果输入输出通道数不同，需要用1x1卷积进行变换
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径 forward
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 捷径路径 forward
        shortcut = self.shortcut(x)

        # 相加并通过激活函数
        out += shortcut
        out = F.relu(out)
        return out


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomResNetCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        # 动态获取通道数 from observation space
        n_channels = observation_space.shape[0]  # Get number of channels from observation space

        # 初始卷积层，将输入通道数映射到更多通道
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 堆叠残差块
        self.res_blocks = nn.Sequential(
            ResidualBlock(16, 16),  # 第一个残差块
            ResidualBlock(16, 32),  # 第二个残差块，扩大通道数
            # 可以继续添加更多块: ResidualBlock(32, 32),
        )

        # 全局平均池化：将 (C, H, W) 的特征图池化为 (C, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 计算最终线性层的输入维度
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            # For our environment, observations are in (C, H, W) format, 
            # so when batched they are (B, C, H, W) which is already NCHW
            n_flatten = self._forward_features(sample_input).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def _forward_features(self, x):
        # NCHW format
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        return self.flatten(x)

    def forward(self, observations):
        # 输入应该是 NCHW (Batch, Channels, Height, Width) format
        # Our observations are already in this format (C, H, W) per environment step
        features = self._forward_features(observations)
        return self.linear(features)
