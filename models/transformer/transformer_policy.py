"""
Transformer-based policy network for 2048 game.

This module implements a Transformer encoder that treats the 16 tiles
as a sequence with 2D positional encoding, outputting action probabilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

from .positional_encoding import PositionalEncoding2D


class TransformerPolicy(nn.Module):
    """
    Transformer-based policy network for 2048.
    
    Architecture:
        1. Flatten 4x4 board to 16-tile sequence
        2. Tile embedding (1 → embed_dim)
        3. Add 2D positional encoding
        4. Transformer encoder (num_layers layers)
        5. Global pooling
        6. Policy head → 4 action logits
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Initialize Transformer policy network.
        
        Args:
            embed_dim: Embedding dimension (default: 128)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of transformer layers (default: 4)
            dim_feedforward: Dimension of feedforward network (default: 512)
            dropout: Dropout rate in transformer layers (default: 0.1)
            head_dropout: Dropout rate in policy head (default: 0.2)
            activation: Activation function ('relu' or 'gelu', default: 'relu')
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Tile embedding: map single tile value to embed_dim
        self.tile_embedding = nn.Linear(1, embed_dim)
        
        # 2D positional encoding for 4x4 grid
        self.positional_encoding = PositionalEncoding2D(embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Expect (batch, seq, feature) format
            norm_first=False   # Post-norm (LayerNorm after residual)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Policy head: classify into 4 actions (up, down, left, right)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        board: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            board: Board state tensor of shape (batch_size, 1, 4, 4)
            return_attention: If True, also return attention weights
            
        Returns:
            If return_attention=False:
                Action logits of shape (batch_size, 4)
            If return_attention=True:
                Tuple of (action_logits, attention_weights)
        """
        batch_size = board.shape[0]
        
        # Flatten spatial dimensions: (batch, 1, 4, 4) -> (batch, 16, 1)
        x = board.view(batch_size, 16, 1)
        
        # Tile embedding: (batch, 16, 1) -> (batch, 16, embed_dim)
        x = self.tile_embedding(x)
        
        # Add positional encoding: (batch, 16, embed_dim) -> (batch, 16, embed_dim)
        x = self.positional_encoding(x)
        
        # Transformer encoding: (batch, 16, embed_dim) -> (batch, 16, embed_dim)
        if return_attention:
            # Get attention weights from each layer
            attentions = []
            for layer in self.transformer_encoder.layers:
                x = layer(x)
                # Note: Extracting attention weights requires modifying the layer
                # For simplicity, we'll skip this in the initial implementation
            transformer_output = x
        else:
            transformer_output = self.transformer_encoder(x)
        
        # Global pooling: mean across sequence dimension
        # (batch, 16, embed_dim) -> (batch, embed_dim)
        pooled = transformer_output.mean(dim=1)
        
        # Policy head: (batch, embed_dim) -> (batch, 4)
        action_logits = self.policy_head(pooled)
        
        if return_attention:
            return action_logits, None  # Placeholder for attention weights
        else:
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


class DualTransformerPolicy(nn.Module):
    """
    Dual-head Transformer with policy and value outputs.
    
    Similar to AlphaZero, this model outputs both action probabilities
    and state value estimation.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Initialize dual-head Transformer policy network.
        
        Args:
            embed_dim: Embedding dimension (default: 128)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of transformer layers (default: 4)
            dim_feedforward: Dimension of feedforward network (default: 512)
            dropout: Dropout rate in transformer layers (default: 0.1)
            head_dropout: Dropout rate in heads (default: 0.2)
            activation: Activation function ('relu' or 'gelu', default: 'relu')
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Shared components (same as TransformerPolicy)
        self.tile_embedding = nn.Linear(1, embed_dim)
        self.positional_encoding = PositionalEncoding2D(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Policy head: classify into 4 actions
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 4)
        )
        
        # Value head: estimate state value
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
        batch_size = board.shape[0]
        
        # Flatten and embed
        x = board.view(batch_size, 16, 1)
        x = self.tile_embedding(x)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global pooling
        pooled = x.mean(dim=1)
        
        # Dual heads
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy_logits, value
    
    def get_num_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_transformer_policy():
    """Test TransformerPolicy model."""
    print("Testing Transformer Policy Network...\n")
    
    # Model hyperparameters
    batch_size = 4
    embed_dim = 128
    num_heads = 8
    num_layers = 4
    
    print("1. Testing Single-Head Transformer Policy")
    print("=" * 60)
    
    # Create model
    model = TransformerPolicy(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Expected: ~500K-800K parameters")
    
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
    
    print("\n2. Testing Dual-Head Transformer Policy")
    print("=" * 60)
    
    # Create dual model
    dual_model = DualTransformerPolicy(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
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
    print(f"TransformerPolicy: {model.get_num_params():,} parameters")
    print(f"DualTransformerPolicy: {dual_model.get_num_params():,} parameters")
    print(f"Difference: {dual_model.get_num_params() - model.get_num_params():,} parameters")
    
    print("\n" + "=" * 60)
    print("✓ All Transformer Policy tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_transformer_policy()
