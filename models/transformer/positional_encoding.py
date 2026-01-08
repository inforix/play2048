"""
2D Positional Encoding for Transformer-based 2048 AI.

This module implements learnable 2D positional encodings for the 4x4 grid,
allowing the model to distinguish between different tile positions.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for a 4x4 grid.
    
    Each of the 16 positions has a unique learnable embedding vector
    that is added to the tile embeddings to encode spatial structure.
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Initialize 2D positional encoding.
        
        Args:
            embed_dim: Embedding dimension (default: 128)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 16 learnable position embeddings for 4x4 grid
        # Initialized with small random values
        self.position_embeddings = nn.Parameter(
            torch.randn(16, embed_dim) * 0.02
        )
        
    def forward(self, x):
        """
        Add positional encodings to input.
        
        Args:
            x: Input tensor of shape (batch_size, 16, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, 16, embed_dim) with positional encoding added
        """
        # x shape: (batch_size, 16, embed_dim)
        # position_embeddings shape: (16, embed_dim)
        # Broadcasting automatically handles batch dimension
        return x + self.position_embeddings.unsqueeze(0)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'embed_dim={self.embed_dim}, num_positions=16 (4x4)'


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Fixed sinusoidal 2D positional encoding (alternative to learnable).
    
    Uses sin/cos functions at different frequencies to encode 2D positions,
    similar to the original Transformer paper but adapted for 2D grids.
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Initialize sinusoidal 2D positional encoding.
        
        Args:
            embed_dim: Embedding dimension (must be even, default: 128)
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal encoding"
        
        self.embed_dim = embed_dim
        
        # Create sinusoidal encoding for 4x4 grid
        position_encoding = self._create_encoding()
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('position_encoding', position_encoding)
    
    def _create_encoding(self):
        """
        Create sinusoidal position encodings for 4x4 grid.
        
        Returns:
            Tensor of shape (16, embed_dim)
        """
        # Create position encodings
        encoding = torch.zeros(16, self.embed_dim)
        
        # Compute sinusoidal encodings
        # Half dimensions for row, half for column
        div_term = torch.exp(
            torch.arange(0, self.embed_dim // 2, dtype=torch.float) *
            -(math.log(10000.0) / (self.embed_dim // 2))
        )
        
        pos = 0
        for row in range(4):
            for col in range(4):
                # Row encoding (first half of dimensions)
                encoding[pos, 0::4] = torch.sin(row * div_term[0::2])
                encoding[pos, 1::4] = torch.cos(row * div_term[0::2])
                
                # Column encoding (second half of dimensions)
                encoding[pos, 2::4] = torch.sin(col * div_term[1::2])
                encoding[pos, 3::4] = torch.cos(col * div_term[1::2])
                
                pos += 1
        
        return encoding
    
    def forward(self, x):
        """
        Add positional encodings to input.
        
        Args:
            x: Input tensor of shape (batch_size, 16, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, 16, embed_dim) with positional encoding added
        """
        return x + self.position_encoding.unsqueeze(0)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'embed_dim={self.embed_dim}, num_positions=16 (4x4), type=sinusoidal'


def test_positional_encoding():
    """Test positional encoding modules."""
    print("Testing Positional Encoding modules...\n")
    
    batch_size = 4
    embed_dim = 128
    seq_len = 16  # 4x4 grid
    
    # Test learnable encoding
    print("1. Testing Learnable 2D Positional Encoding")
    print("-" * 50)
    
    learnable_pe = PositionalEncoding2D(embed_dim=embed_dim)
    print(f"Module: {learnable_pe}")
    print(f"Parameter shape: {learnable_pe.position_embeddings.shape}")
    print(f"Number of parameters: {learnable_pe.position_embeddings.numel():,}")
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = learnable_pe(x)
    print(f"Output shape: {output.shape}")
    
    # Verify output
    assert output.shape == (batch_size, seq_len, embed_dim), "Unexpected output shape"
    assert not torch.allclose(x, output), "Output should differ from input"
    print("✓ Shape and modification check passed")
    
    # Test gradients
    loss = output.sum()
    loss.backward()
    assert learnable_pe.position_embeddings.grad is not None, "Gradients should exist"
    print("✓ Gradient check passed")
    
    # Test sinusoidal encoding
    print("\n2. Testing Sinusoidal 2D Positional Encoding")
    print("-" * 50)
    
    sinusoidal_pe = SinusoidalPositionalEncoding2D(embed_dim=embed_dim)
    print(f"Module: {sinusoidal_pe}")
    print(f"Encoding shape: {sinusoidal_pe.position_encoding.shape}")
    
    # Forward pass
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    output2 = sinusoidal_pe(x2)
    print(f"\nInput shape: {x2.shape}")
    print(f"Output shape: {output2.shape}")
    
    # Verify output
    assert output2.shape == (batch_size, seq_len, embed_dim), "Unexpected output shape"
    assert not torch.allclose(x2, output2), "Output should differ from input"
    print("✓ Shape and modification check passed")
    
    # Verify fixed encoding (no gradients)
    assert not sinusoidal_pe.position_encoding.requires_grad, "Should not require gradients"
    print("✓ Fixed encoding check passed")
    
    # Compare different positions
    print("\n3. Comparing Position Encodings")
    print("-" * 50)
    
    pe_values = learnable_pe.position_embeddings.detach()
    
    # Check that different positions have different encodings
    pos_0 = pe_values[0]
    pos_1 = pe_values[1]
    pos_15 = pe_values[15]
    
    diff_01 = torch.norm(pos_0 - pos_1).item()
    diff_015 = torch.norm(pos_0 - pos_15).item()
    
    print(f"Difference between position 0 and 1: {diff_01:.4f}")
    print(f"Difference between position 0 and 15: {diff_015:.4f}")
    assert diff_01 > 0, "Different positions should have different encodings"
    print("✓ Position uniqueness check passed")
    
    print("\n" + "="*50)
    print("✓ All positional encoding tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_positional_encoding()
