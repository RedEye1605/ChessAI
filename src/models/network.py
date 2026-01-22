"""
Neural Network untuk Chess RL
=============================
Implementasi Policy-Value Network berbasis ResNet untuk agen catur.
Arsitektur terinspirasi dari AlphaZero dengan modifikasi untuk stabilitas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

from .residual_block import ResidualBlock, ResidualBlockPreAct, BottleneckBlock, SEBlock


class ChessNetwork(nn.Module):
    """
    Base Chess Neural Network.
    
    Arsitektur:
    - Input Encoder: Conv2D untuk mengubah input channels ke hidden channels
    - Backbone: Stack of Residual Blocks
    - Policy Head: Output distribusi probabilitas moves
    - Value Head: Output estimasi nilai posisi
    
    Input: (batch, input_channels, 8, 8)
    Output: 
        - policy: (batch, action_size)  
        - value: (batch, 1)
    """
    
    def __init__(
        self,
        input_channels: int = 14,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        action_size: int = 4672,
        normalization: str = 'layer',
        activation: str = 'relu',
        dropout: float = 0.1,
        use_se: bool = False,
        use_preact: bool = False
    ):
        """
        Inisialisasi Chess Network.
        
        Args:
            input_channels: Jumlah input channels (default 14 untuk standard encoding)
            num_filters: Jumlah filters di hidden layers
            num_residual_blocks: Jumlah residual blocks
            action_size: Ukuran action space
            normalization: Tipe normalization ('batch', 'layer', 'group')
            activation: Tipe aktivasi ('relu', 'gelu', 'silu')
            dropout: Dropout rate
            use_se: Gunakan Squeeze-and-Excitation
            use_preact: Gunakan pre-activation residual blocks
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.action_size = action_size
        
        # Input encoder
        self.input_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.input_norm = self._get_norm_layer(normalization, num_filters)
        self.input_activation = self._get_activation(activation)
        
        # Residual backbone
        block_class = ResidualBlockPreAct if use_preact else ResidualBlock
        self.residual_blocks = nn.ModuleList([
            block_class(
                channels=num_filters,
                normalization=normalization,
                activation=activation,
                dropout=dropout if i > 0 else 0,  # No dropout di block pertama
                use_se=use_se if hasattr(block_class, 'use_se') else False
            )
            for i in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 128, 1, bias=False)
        self.policy_norm = self._get_norm_layer(normalization, 128)
        self.policy_activation = self._get_activation(activation)
        self.policy_fc = nn.Linear(128 * 64, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.value_norm = self._get_norm_layer(normalization, 32)
        self.value_activation = self._get_activation(activation)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        """Get normalization layer."""
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, channels)
        elif norm_type == 'group':
            return nn.GroupNorm(min(32, channels // 4), channels)
        return nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU(inplace=True)
        return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Special initialization untuk output layers
        # Smaller initialization untuk stabilitas
        nn.init.xavier_uniform_(self.policy_fc.weight, gain=0.01)
        nn.init.zeros_(self.policy_fc.bias)
        nn.init.xavier_uniform_(self.value_fc2.weight, gain=0.01)
        nn.init.zeros_(self.value_fc2.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor shape (batch, input_channels, 8, 8)
            legal_action_mask: Optional boolean mask (batch, action_size)
            
        Returns:
            policy: Log probabilities shape (batch, action_size)
            value: Value estimates shape (batch, 1)
        """
        # Input encoding
        out = self.input_conv(x)
        out = self.input_norm(out)
        out = self.input_activation(out)
        
        # Residual backbone
        for block in self.residual_blocks:
            out = block(out)
        
        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_norm(policy)
        policy = self.policy_activation(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        
        # Apply legal action mask
        if legal_action_mask is not None:
            # Set illegal actions ke nilai sangat negatif
            policy = policy.masked_fill(~legal_action_mask, float('-inf'))
        
        # Log softmax untuk numerical stability
        policy = F.log_softmax(policy, dim=-1)
        
        # Value head
        value = self.value_conv(out)
        value = self.value_norm(value)
        value = self.value_activation(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  # Output range [-1, 1]
        
        return policy, value
    
    def get_action_probs(
        self, 
        x: torch.Tensor,
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get action probabilities (bukan log probabilities).
        
        Args:
            x: Input tensor
            legal_action_mask: Optional legal move mask
            
        Returns:
            Action probabilities shape (batch, action_size)
        """
        log_probs, _ = self.forward(x, legal_action_mask)
        return torch.exp(log_probs)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only.
        
        Args:
            x: Input tensor
            
        Returns:
            Value estimates shape (batch, 1)
        """
        _, value = self.forward(x)
        return value
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        return {
            'input_channels': self.input_channels,
            'num_filters': self.num_filters,
            'num_residual_blocks': self.num_residual_blocks,
            'action_size': self.action_size,
            'total_parameters': self.count_parameters()
        }


class ChessPolicyValueNetwork(ChessNetwork):
    """
    Extended Policy-Value Network dengan additional features.
    
    Tambahan:
    - Auxiliary policy head untuk intermediate supervision
    - Uncertainty estimation
    - Feature extraction methods
    """
    
    def __init__(self, *args, uncertainty_estimation: bool = False, **kwargs):
        """
        Inisialisasi extended network.
        
        Args:
            uncertainty_estimation: Enable uncertainty estimation
            *args, **kwargs: Arguments untuk ChessNetwork
        """
        super().__init__(*args, **kwargs)
        
        self.uncertainty_estimation = uncertainty_estimation
        
        if uncertainty_estimation:
            # Additional head untuk uncertainty
            self.value_log_var = nn.Linear(256, 1)
            nn.init.constant_(self.value_log_var.weight, 0)
            nn.init.constant_(self.value_log_var.bias, -2)  # Start dengan low variance
    
    def forward_with_uncertainty(
        self, 
        x: torch.Tensor,
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass dengan uncertainty estimation.
        
        Returns:
            policy: Log probabilities
            value: Value mean
            value_var: Value variance
        """
        # Input encoding
        out = self.input_conv(x)
        out = self.input_norm(out)
        out = self.input_activation(out)
        
        # Residual backbone
        for block in self.residual_blocks:
            out = block(out)
        
        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_norm(policy)
        policy = self.policy_activation(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        if legal_action_mask is not None:
            policy = policy.masked_fill(~legal_action_mask, float('-inf'))
        policy = F.log_softmax(policy, dim=-1)
        
        # Value head with uncertainty
        value_features = self.value_conv(out)
        value_features = self.value_norm(value_features)
        value_features = self.value_activation(value_features)
        value_features = value_features.view(value_features.size(0), -1)
        value_hidden = F.relu(self.value_fc1(value_features))
        
        value = torch.tanh(self.value_fc2(value_hidden))
        
        if self.uncertainty_estimation:
            log_var = self.value_log_var(value_hidden)
            value_var = torch.exp(log_var)
        else:
            value_var = torch.zeros_like(value)
        
        return policy, value, value_var
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features untuk analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor dari backbone
        """
        out = self.input_conv(x)
        out = self.input_norm(out)
        out = self.input_activation(out)
        
        for block in self.residual_blocks:
            out = block(out)
        
        return out


class SmallChessNetwork(nn.Module):
    """
    Smaller network untuk testing dan quick experiments.
    
    Lebih cepat untuk training tapi dengan capacity yang lebih rendah.
    """
    
    def __init__(
        self,
        input_channels: int = 14,
        num_filters: int = 64,
        num_residual_blocks: int = 4,
        action_size: int = 4672
    ):
        """Inisialisasi small network."""
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters, normalization='batch')
            for _ in range(num_residual_blocks)
        ])
        
        # Simple policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, action_size)
        )
        
        # Simple value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        out = self.input_conv(x)
        
        for block in self.residual_blocks:
            out = block(out)
        
        policy = self.policy_head(out)
        if legal_action_mask is not None:
            policy = policy.masked_fill(~legal_action_mask, float('-inf'))
        policy = F.log_softmax(policy, dim=-1)
        
        value = self.value_head(out)
        
        return policy, value


def create_network(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function untuk membuat network dari config.
    
    Args:
        config: Network configuration dictionary
        
    Returns:
        Initialized network
    """
    network_type = config.get('network_type', 'standard')
    
    if network_type == 'small':
        return SmallChessNetwork(
            input_channels=config.get('input_channels', 14),
            num_filters=config.get('num_filters', 64),
            num_residual_blocks=config.get('num_residual_blocks', 4),
            action_size=config.get('action_size', 4672)
        )
    else:
        return ChessPolicyValueNetwork(
            input_channels=config.get('input_channels', 14),
            num_filters=config.get('num_filters', 256),
            num_residual_blocks=config.get('num_residual_blocks', 10),
            action_size=config.get('action_size', 4672),
            normalization=config.get('normalization', 'layer'),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.1),
            use_se=config.get('use_se', False),
            use_preact=config.get('use_preact', False),
            uncertainty_estimation=config.get('uncertainty_estimation', False)
        )
