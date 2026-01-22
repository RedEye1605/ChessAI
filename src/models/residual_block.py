"""
Residual Block Implementation
=============================
Implementasi residual blocks untuk neural network catur.
Menggunakan berbagai teknik normalisasi untuk stabilitas training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class ResidualBlock(nn.Module):
    """
    Residual Block dengan normalization untuk stabilitas.
    
    Arsitektur:
        Input -> Conv1 -> Norm -> Activation -> Conv2 -> Norm -> Add(Input) -> Activation
    
    Attributes:
        conv1: First convolution layer
        conv2: Second convolution layer
        norm1: First normalization layer
        norm2: Second normalization layer
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        normalization: Literal['batch', 'layer', 'group', 'none'] = 'layer',
        activation: Literal['relu', 'gelu', 'silu'] = 'relu',
        dropout: float = 0.0,
        use_se: bool = False  # Squeeze-and-Excitation
    ):
        """
        Inisialisasi Residual Block.
        
        Args:
            channels: Jumlah input/output channels
            kernel_size: Ukuran kernel convolution
            normalization: Tipe normalization ('batch', 'layer', 'group', 'none')
            activation: Tipe aktivasi ('relu', 'gelu', 'silu')
            dropout: Dropout rate
            use_se: Gunakan Squeeze-and-Excitation block
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        # Convolution layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        
        # Normalization layers
        self.norm1 = self._get_norm_layer(normalization, channels)
        self.norm2 = self._get_norm_layer(normalization, channels)
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Squeeze-and-Excitation
        self.se = SEBlock(channels) if use_se else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        """Get normalization layer berdasarkan tipe."""
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'layer':
            # LayerNorm untuk Conv2D - normalize over C,H,W
            return nn.GroupNorm(1, channels)  # GroupNorm dengan 1 group = LayerNorm
        elif norm_type == 'group':
            num_groups = min(32, channels // 4)
            return nn.GroupNorm(num_groups, channels)
        else:
            return nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        """Initialize weights dengan He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor shape (batch, channels, 8, 8)
            
        Returns:
            Output tensor shape (batch, channels, 8, 8)
        """
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Squeeze-and-Excitation
        out = self.se(out)
        
        # Skip connection
        out = out + identity
        out = self.activation(out)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    
    Recalibrate channel-wise feature responses dengan:
    1. Global Average Pooling (Squeeze)
    2. FC -> ReLU -> FC -> Sigmoid (Excitation)
    3. Channel-wise multiplication
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Inisialisasi SE Block.
        
        Args:
            channels: Jumlah channels
            reduction: Reduction ratio untuk bottleneck
        """
        super().__init__()
        
        reduced_channels = max(channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor shape (batch, channels, H, W)
            
        Returns:
            Recalibrated tensor
        """
        batch, channels, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        y = self.squeeze(x).view(batch, channels)
        
        # Excitation: FC layers
        y = self.excitation(y).view(batch, channels, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class ResidualBlockPreAct(nn.Module):
    """
    Pre-activation Residual Block.
    
    Arsitektur (He et al., 2016 - Identity Mappings):
        Input -> Norm -> Activation -> Conv1 -> Norm -> Activation -> Conv2 -> Add(Input)
    
    Lebih stabil untuk deep networks.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        normalization: str = 'layer',
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        """Inisialisasi Pre-activation Residual Block."""
        super().__init__()
        
        padding = kernel_size // 2
        
        # Normalization (sebelum activation)
        self.norm1 = self._get_norm_layer(normalization, channels)
        self.norm2 = self._get_norm_layer(normalization, channels)
        
        # Activations
        self.act1 = self._get_activation(activation)
        self.act2 = self._get_activation(activation)
        
        # Convolutions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, channels)
        elif norm_type == 'group':
            return nn.GroupNorm(min(32, channels // 4), channels)
        return nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU(inplace=True)
        return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dengan pre-activation design."""
        identity = x
        
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        out = out + identity
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block untuk efisiensi.
    
    Arsitektur:
        Input -> Conv1x1 (reduce) -> Conv3x3 -> Conv1x1 (expand) -> Add(Input)
    
    Lebih efisien computation untuk deep networks.
    """
    
    def __init__(
        self,
        channels: int,
        bottleneck_ratio: int = 4,
        normalization: str = 'layer',
        activation: str = 'relu'
    ):
        """Inisialisasi Bottleneck Block."""
        super().__init__()
        
        bottleneck_channels = channels // bottleneck_ratio
        
        # Layers
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, 1, bias=False)
        self.norm1 = self._get_norm_layer(normalization, bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.norm2 = self._get_norm_layer(normalization, bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, 1, bias=False)
        self.norm3 = self._get_norm_layer(normalization, channels)
        
        self.activation = self._get_activation(activation)
        
        self._init_weights()
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, channels)
        return nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out
