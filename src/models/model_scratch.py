
"""
Hybrid Scratch CNN for ISIC-2018
=================================

This module defines our from-scratch convolutional neural network
for the ISIC-2018 skin lesion classification task.

Our model combines:
- Depthwise separable convolutions (MobileNet-style)
- Residual connections (ResNet-style)
- Global average pooling
- Lightweight classifier head (7 classes)

This architecture is:
- Efficient enough for future mobile deployment
- Powerful enough for competitive accuracy
- Compatible with Grad-CAM and other explainability tools
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Our depthwise separable convolution block.

    We factorize a standard convolution into:
    - depthwise: groupwise conv over each channel
    - pointwise: 1x1 conv to mix channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Convolution stride for the depthwise part, by default 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of our depthwise separable block.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class HybridResidualBlock(nn.Module):
    """
    Our hybrid residual block using depthwise separable convolutions.

    We use:
    - DepthwiseSeparableConv for efficient feature extraction
    - Residual connection for stable training

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    stride : int, optional
        Convolution stride, by default 1. If stride != 1 or
        in_channels != out_channels, we use a 1x1 conv for residual.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)

        # Residual projection if channels/stride do not match
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of our hybrid residual block.
        """
        identity = self.shortcut(x)
        out = self.conv(x)
        out = out + identity
        out = self.act(out)
        return out


class ISICScratchCNN(nn.Module):
    """
    Our from-scratch hybrid CNN for ISIC-2018.

    Architecture overview
    ---------------------
    Input: 3 x H x W (we assume 224 x 224 after transforms)

    Stem:
        Conv(3 -> 32) -> BN -> ReLU

    Stages:
        Stage 1: HybridResidualBlock(32 -> 64)
        Stage 2: HybridResidualBlock(64 -> 128, stride=2)
                 HybridResidualBlock(128 -> 128)
        Stage 3: HybridResidualBlock(128 -> 256, stride=2)
                 HybridResidualBlock(256 -> 256)
        Stage 4: HybridResidualBlock(256 -> 512, stride=2)
                 HybridResidualBlock(512 -> 512)

    Head:
        Global Average Pooling
        Dropout
        Linear(512 -> num_classes)

    Parameters
    ----------
    num_classes : int
        Number of output classes (ISIC-2018 uses 7).
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1
        self.stage1 = HybridResidualBlock(32, 64, stride=1)

        # Stage 2
        self.stage2 = nn.Sequential(
            HybridResidualBlock(64, 128, stride=2),
            HybridResidualBlock(128, 128, stride=1),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            HybridResidualBlock(128, 256, stride=2),
            HybridResidualBlock(256, 256, stride=1),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            HybridResidualBlock(256, 512, stride=2),
            HybridResidualBlock(512, 512, stride=1),
        )

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through our hybrid CNN.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)         # (B, 512, 1, 1)
        x = torch.flatten(x, 1)         # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)                  # (B, num_classes)

        return x

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in our model.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Simple sanity check when running this module directly.
    model = ISICScratchCNN(num_classes=7)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Trainable parameters:", model.count_parameters())
