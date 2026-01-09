"""
CNN-based policy network for 2048 game.

This module implements a Convolutional Neural Network that processes
the 4x4 board using multiple convolutional layers with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm and ReLU activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True
    ):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            padding: Padding for convolution
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection.
    """
    
    def __init__(self, channels: int, dropout: float = 0.0):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            dropout: Dropout rate (0 for no dropout)
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class CNNPolicy(nn.Module):
    """
    CNN-based policy network for 2048.
    
    Architecture:
        1. Initial convolution (1 → base_channels)
        2. Residual blocks (num_blocks)
        3. Global average pooling
        4. Policy head → 4 action logits
    """
    
    def __init__(
        self,
        base_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
        head_dropout: float = 0.3
    ):
        """
        Initialize CNN policy network.
        
        Args:
            base_channels: Number of channels in convolutional layers
            num_blocks: Number of residual blocks
            dropout: Dropout rate in residual blocks
            head_dropout: Dropout rate in policy head
        """
        super().__init__()
        
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        
        # Initial convolution: 1 channel (board) → base_channels
        self.conv_input = ConvBlock(1, base_channels, kernel_size=3, padding=1)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, dropout) for _ in range(num_blocks)
        ])
        
        # Additional convolution for feature extraction
        self.conv_features = ConvBlock(base_channels, base_channels * 2, kernel_size=3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head: predict 4 actions (up, down, left, right)
        self.policy_head = nn.Sequential(
            nn.Linear(base_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 4)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            board: Board state tensor of shape (batch_size, 1, 4, 4)
            
        Returns:
            Action logits of shape (batch_size, 4)
        """
        # Add channel dimension if needed
        if board.dim() == 3:
            board = board.unsqueeze(1)  # (batch, 1, 4, 4)
        
        # Initial convolution
        x = self.conv_input(board)
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Feature extraction
        x = self.conv_features(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, channels*2)
        
        # Policy head
        action_logits = self.policy_head(x)
        
        return action_logits
    
    def get_num_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_action_probabilities(self, board: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities using softmax.
        
        Args:
            board: Board state tensor of shape (batch_size, 1, 4, 4)
            
        Returns:
            Action probabilities of shape (batch_size, 4)
        """
        logits = self.forward(board)
        return torch.softmax(logits, dim=-1)
    
    def predict_action(self, board: torch.Tensor) -> torch.Tensor:
        """
        Predict the best action.
        
        Args:
            board: Board state tensor of shape (batch_size, 1, 4, 4)
            
        Returns:
            Action indices of shape (batch_size,)
        """
        logits = self.forward(board)
        return logits.argmax(dim=-1)


class DualCNNPolicy(nn.Module):
    """
    Dual-head CNN with policy and value outputs.
    
    Similar to AlphaZero, this model outputs both action probabilities
    and state value estimation using a shared CNN backbone.
    """
    
    def __init__(
        self,
        base_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
        head_dropout: float = 0.3
    ):
        """
        Initialize dual-head CNN policy network.
        
        Args:
            base_channels: Number of channels in convolutional layers
            num_blocks: Number of residual blocks
            dropout: Dropout rate in residual blocks
            head_dropout: Dropout rate in heads
        """
        super().__init__()
        
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        
        # Shared backbone
        self.conv_input = ConvBlock(1, base_channels, kernel_size=3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, dropout) for _ in range(num_blocks)
        ])
        
        self.conv_features = ConvBlock(base_channels, base_channels * 2, kernel_size=3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head: 4 actions
        self.policy_head = nn.Sequential(
            nn.Linear(base_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 4)
        )
        
        # Value head: estimate state value
        self.value_head = nn.Sequential(
            nn.Linear(base_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual-head network.
        
        Args:
            board: Board state tensor of shape (batch_size, 1, 4, 4)
            
        Returns:
            Tuple of (policy_logits, value) where:
                policy_logits: shape (batch_size, 4)
                value: shape (batch_size, 1)
        """
        # Add channel dimension if needed
        if board.dim() == 3:
            board = board.unsqueeze(1)
        
        # Shared backbone
        x = self.conv_input(board)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.conv_features(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dual heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_num_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def predict(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value (inference mode).
        
        Args:
            board: Board state (batch, 4, 4) or (4, 4)
            
        Returns:
            (policy_probs, value) tuple
        """
        self.eval()
        with torch.no_grad():
            if board.dim() == 2:
                board = board.unsqueeze(0)
            
            policy_logits, value = self.forward(board)
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            return policy_probs, value


def test_cnn_policy():
    """Test CNN policy network."""
    print("Testing CNN Policy Network...\n")
    
    # Model hyperparameters
    batch_size = 4
    base_channels = 128
    num_blocks = 4
    
    print("1. Testing Single-Head CNN Policy")
    print("=" * 60)
    
    # Create model
    model = CNNPolicy(
        base_channels=base_channels,
        num_blocks=num_blocks
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Expected: ~300K-500K parameters")
    
    # Create dummy input (normalized board)
    board = torch.randn(batch_size, 1, 4, 4)
    print(f"\nInput shape: {board.shape}")
    
    # Forward pass
    logits = model(board)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 4), f"Expected shape ({batch_size}, 4), got {logits.shape}"
    print("✓ Output shape correct")
    
    # Test probabilities
    probs = model.get_action_probabilities(board)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0].tolist()}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "Probabilities should sum to 1"
    print("✓ Probabilities sum to 1")
    
    # Test action prediction
    actions = model.predict_action(board)
    print(f"\nPredicted actions shape: {actions.shape}")
    print(f"Predicted actions: {actions.tolist()}")
    assert actions.shape == (batch_size,), "Action shape mismatch"
    assert all(0 <= a < 4 for a in actions), "Actions should be in [0, 3]"
    print("✓ Action prediction correct")
    
    # Test gradients
    loss = logits.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Model should have gradients"
    print("✓ Gradients computed successfully")
    
    print("\n2. Testing Dual-Head CNN Policy")
    print("=" * 60)
    
    # Create dual model
    dual_model = DualCNNPolicy(
        base_channels=base_channels,
        num_blocks=num_blocks
    )
    
    print(f"Model: {dual_model.__class__.__name__}")
    print(f"Parameters: {dual_model.get_num_params():,}")
    
    # Forward pass
    policy_logits, value = dual_model(board)
    print(f"\nPolicy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    assert policy_logits.shape == (batch_size, 4), "Policy shape mismatch"
    assert value.shape == (batch_size, 1), "Value shape mismatch"
    print("✓ Output shapes correct")
    
    # Check value range
    assert torch.all((value >= -1) & (value <= 1)), "Value should be in [-1, 1]"
    print(f"Sample values: {value[:, 0].tolist()}")
    print("✓ Value range correct ([-1, 1])")
    
    # Test gradients
    total_loss = policy_logits.sum() + value.sum()
    total_loss.backward()
    has_grad = any(p.grad is not None for p in dual_model.parameters())
    assert has_grad, "Dual model should have gradients"
    print("✓ Gradients computed successfully")
    
    print("\n3. Model Size Comparison")
    print("=" * 60)
    print(f"CNNPolicy: {model.get_num_params():,} parameters")
    print(f"DualCNNPolicy: {dual_model.get_num_params():,} parameters")
    print(f"Difference: {dual_model.get_num_params() - model.get_num_params():,} parameters")
    
    # Test different configurations
    print("\n4. Testing Different Configurations")
    print("=" * 60)
    
    configs = [
        (64, 3, "Small"),
        (128, 4, "Medium"),
        (256, 6, "Large")
    ]
    
    for channels, blocks, name in configs:
        model = CNNPolicy(base_channels=channels, num_blocks=blocks)
        print(f"{name} (channels={channels}, blocks={blocks}): {model.get_num_params():,} parameters")
    
    print("\n" + "=" * 60)
    print("✓ All CNN Policy tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_cnn_policy()
