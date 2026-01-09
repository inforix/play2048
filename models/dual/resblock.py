"""
Residual Block for AlphaZero network.

Used in the shared backbone to extract spatial features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    
    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU -> out
        |                                        |
        +----------> (1x1 conv if needed) -------+
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ResBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection - use 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, in_channels, H, W)
            
        Returns:
            Output tensor (batch, out_channels, H, W)
        """
        identity = self.skip(x)
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += identity
        out = F.relu(out)
        
        return out


def test_resblock():
    """Test ResBlock implementation."""
    print("Testing ResBlock...")
    
    # Test same channels
    block = ResBlock(128, 128)
    x = torch.randn(2, 128, 4, 4)
    out = block(x)
    assert out.shape == (2, 128, 4, 4), f"Expected (2, 128, 4, 4), got {out.shape}"
    print(f"✓ Same channels: {x.shape} -> {out.shape}")
    
    # Test different channels
    block = ResBlock(128, 256)
    x = torch.randn(2, 128, 4, 4)
    out = block(x)
    assert out.shape == (2, 256, 4, 4), f"Expected (2, 256, 4, 4), got {out.shape}"
    print(f"✓ Different channels: {x.shape} -> {out.shape}")
    
    # Test gradient flow
    block = ResBlock(64, 64)
    x = torch.randn(2, 64, 4, 4, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients not flowing through ResBlock"
    print(f"✓ Gradients flow correctly")
    
    # Count parameters
    params = sum(p.numel() for p in block.parameters())
    print(f"✓ Parameters: {params:,}")
    
    print("\n✓ All ResBlock tests passed!")


if __name__ == "__main__":
    test_resblock()
