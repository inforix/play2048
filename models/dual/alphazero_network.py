"""
AlphaZero Dual Network for 2048.

Combines policy and value predictions using a shared ResNet backbone.
Architecture inspired by AlphaGo Zero and AlphaZero papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .resblock import ResBlock
except ImportError:
    from resblock import ResBlock


class AlphaZeroNetwork(nn.Module):
    """
    Dual network with shared backbone and separate policy/value heads.
    
    Architecture:
        Input (1, 4, 4)
        -> Initial Conv (1 -> channels)
        -> ResBlocks x num_blocks
        -> Policy Head -> (4,)
        -> Value Head -> (1,)
    """
    
    def __init__(self, num_blocks: int = 4, channels: int = 256, dropout: float = 0.3):
        """
        Initialize AlphaZero network.
        
        Args:
            num_blocks: Number of residual blocks
            channels: Number of channels in residual blocks
            dropout: Dropout rate for policy/value heads
        """
        super(AlphaZeroNetwork, self).__init__()
        
        self.num_blocks = num_blocks
        self.channels = channels
        
        # Initial convolution: 1 channel (board) -> channels
        self.conv_input = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower (shared backbone)
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, channels) for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 4 * 4, 4)  # 4 actions
        self.policy_dropout = nn.Dropout(dropout)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 4 * 4, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input board state (batch, 4, 4)
            
        Returns:
            (policy_logits, value) tuple
            - policy_logits: (batch, 4) - logits for 4 actions
            - value: (batch, 1) - state value in [-1, 1]
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, 4, 4)
        
        # Shared backbone
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc(p)
        
        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)  # Flatten
        v = self.value_dropout(v)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value
    
    def predict(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value (inference mode).
        
        Args:
            board: Board state (batch, 4, 4) or (4, 4)
            
        Returns:
            (policy_probs, value) tuple
            - policy_probs: (batch, 4) - action probabilities
            - value: (batch, 1) - state value
        """
        self.eval()
        with torch.no_grad():
            if board.dim() == 2:
                board = board.unsqueeze(0)  # Add batch dim
            
            policy_logits, value = self.forward(board)
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            return policy_probs, value


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_alphazero_network():
    """Test AlphaZeroNetwork implementation."""
    print("Testing AlphaZeroNetwork...")
    
    # Test different configurations
    configs = [
        (4, 128, "Small"),
        (4, 256, "Medium"),
        (6, 256, "Large")
    ]
    
    for num_blocks, channels, name in configs:
        print(f"\n{name} configuration (blocks={num_blocks}, channels={channels}):")
        
        model = AlphaZeroNetwork(num_blocks=num_blocks, channels=channels)
        
        # Test forward pass
        x = torch.randn(2, 4, 4)  # Batch of 2 boards
        policy_logits, value = model(x)
        
        assert policy_logits.shape == (2, 4), f"Expected (2, 4), got {policy_logits.shape}"
        assert value.shape == (2, 1), f"Expected (2, 1), got {value.shape}"
        assert value.min() >= -1 and value.max() <= 1, f"Value not in [-1, 1]: {value}"
        
        print(f"  ✓ Forward pass: (2, 4, 4) -> policy (2, 4), value (2, 1)")
        
        # Test predict method
        policy_probs, value = model.predict(x[0])
        assert policy_probs.shape == (1, 4)
        assert torch.abs(policy_probs.sum() - 1.0) < 1e-5, "Policy doesn't sum to 1"
        print(f"  ✓ Predict: policy sums to {policy_probs.sum().item():.6f}")
        
        # Test gradient flow
        x_grad = torch.randn(2, 4, 4, requires_grad=True)
        policy_logits, value = model(x_grad)
        loss = policy_logits.sum() + value.sum()
        loss.backward()
        assert x_grad.grad is not None
        print(f"  ✓ Gradients flow correctly")
        
        # Count parameters
        params = count_parameters(model)
        print(f"  ✓ Trainable parameters: {params:,}")
    
    print("\n✓ All AlphaZeroNetwork tests passed!")
    
    # Memory estimate
    print("\nMemory estimates (single forward pass):")
    model = AlphaZeroNetwork(num_blocks=4, channels=256)
    x = torch.randn(64, 4, 4)  # Batch of 64
    
    # Estimate memory
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    activation_size = 64 * 256 * 4 * 4 * 4 / (1024 ** 2)  # Rough estimate
    
    print(f"  Model size: ~{model_size:.1f} MB")
    print(f"  Activation memory (batch=64): ~{activation_size:.1f} MB")


if __name__ == "__main__":
    test_alphazero_network()
