import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import math


class PositionalEncoding(nn.Module):
    """Simple positional encoding for the 9 positions on the board"""
    def __init__(self, d_model, max_len=9):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class TicTacToeTransformer(BaseFeaturesExtractor):
    """
    Ultra-lightweight transformer-based feature extractor for Tic Tac Toe.
    Processes the 3x3x3 board representation using a minimal transformer architecture.
    """
    def __init__(self, observation_space, features_dim=64, d_model=16, nhead=2, num_layers=1):
        super().__init__(observation_space, features_dim)
        
        # Input: 3 channels of 3x3 = 9 positions, each with 3 features
        
        self.d_model = d_model
        self.num_patches = 9  # 9 positions on the board
        
        # Linear projection to map 3-channel input to d_model dimensions
        self.patch_embedding = nn.Linear(3, d_model)  # 3 features per position -> d_model
        
        # Positional encoding for the 9 board positions
        self.pos_encoder = PositionalEncoding(d_model, max_len=9)
        
        # Ultra-lightweight Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 32 for d_model=16
            dropout=0.0,  # No dropout to save parameters
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to output features
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 9, features_dim),  # Flatten all 9 positions then project to features_dim
            nn.ReLU()
        )
        
        # Calculate how many elements we have after transformer processing
        self.flatten_size = d_model * 9

    def forward(self, observations):
        """
        Forward pass of the transformer.
        
        Args:
            observations: tensor of shape [batch_size, 3, 3, 3] 
                         representing [current_player, opponent, empty] for each position
        """
        batch_size = observations.size(0)
        
        # Reshape from [batch_size, 3, 3, 3] to [batch_size, 9, 3]
        # where 9 is the 9 positions and 3 is [current_player, opponent, empty] for each
        x = observations.view(batch_size, 3, -1).transpose(1, 2)  # [batch_size, 9, 3]
        
        # Apply patch embedding: [batch_size, 9, 3] -> [batch_size, 9, d_model]
        x = self.patch_embedding(x) * math.sqrt(self.d_model)  # Scale as in original transformer
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, 9, d_model]
        
        # Flatten the sequence dimension to create a fixed-size representation
        x = x.contiguous().view(batch_size, self.flatten_size)  # [batch_size, 9 * d_model]
        
        # Final projection to features_dim
        x = self.output_projection(x)  # [batch_size, features_dim]
        
        return x