"""
EEGNet: A Compact Convolutional Neural Network for EEG Signal Processing

This script implements EEGNet, a lightweight convolutional neural network designed to process EEG (electroencephalography) signals. 
The model extracts temporal and spatial features using convolutional layers and applies non-linearity, pooling, 
and dropout for efficient feature learning.

EEGNet Architecture:
    - Temporal Convolution: Extracts temporal features.
    - Depthwise Spatial Convolution: Captures spatial dependencies across EEG channels.
    - Activation Function: ELU (Exponential Linear Unit) introduces non-linearity.
    - Pooling & Dropout: Average pooling reduces dimensionality, and dropout prevents overfitting.
    - Depthwise Convolution & Pointwise Convolution: Further feature extraction.
    - Fully Connected Layer

Author: Abhishek Mishra
Date: 02/12/2025
"""

import torch
from torch import nn

class EEGNet(nn.Module):
    def __init__(self,
                 num_temporal_filts: int = 8,
                 num_spatial_filts: int = 2,
                 num_chans: int = 32,
                 window_length: int = 100,
                 p_dropout: float = 0.5,
                 avgpool_factor: int = 4,
                 num_classes: int = 4) -> None:
        """
        :param num_temporal_filts: Number of temporal filters in the first convolutional layer
        :param num_spatial_filts: Number of spatial filters in the second convolutional layer
        :param num_chans: Number of channels in the input data
        :param p_dropout: Probability of dropout
        :param window_length: Length of the window in samples
        :param avgpool_factor: Factor for the first average pooling layer
        """
        super(EEGNet, self).__init__()
        self.F1 = num_temporal_filts
        self.D = num_spatial_filts
        self.C = num_chans
        self.F2 = self.D * self.F1

        self.p = p_dropout
        self.T = window_length
        self.avgpool_factor1 = avgpool_factor
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 9), padding='same'),
            nn.Conv2d(self.F1, self.D * self.F1, (self.C, 1), groups=self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, self.avgpool_factor1)),
            nn.Dropout(self.p)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F2,
                      self.F2,
                      (1, 2 * (self.T // (self.avgpool_factor1 * 2)) + 1),
                      groups=self.F2,
                      padding='same'),
            nn.Conv2d(self.F2, self.F2, (1, 1)),
            nn.ELU(),
            nn.Flatten(start_dim=1),  # Flatten along dimension 1
            nn.Linear((self.T // self.avgpool_factor1) * self.F2, self.num_classes)  # Fully connected layer
       
            # nn.Flatten(start_dim=2, end_dim=-1),
            # nn.Linear(self.T // self.avgpool_factor1, 1)
        )

    def forward(self, x):
        x = x.swapaxes(1, 2).unsqueeze(1)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        return block2.squeeze(1)  

if __name__ == "__main__":
    num_timesteps = 240
    batch_size = 16
    num_channels = 32

    net = EEGNet(num_temporal_filts=64,
                 num_spatial_filts=4,
                 num_chans=num_channels,
                 window_length=num_timesteps,
                 avgpool_factor=2)

    test_data = torch.rand(batch_size, num_timesteps, num_channels)
    print("input shape", test_data.shape)
    print("output shape", net(test_data).shape)