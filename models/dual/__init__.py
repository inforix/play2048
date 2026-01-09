"""
Dual network architecture for AlphaZero-style training.
"""

from .resblock import ResBlock
from .alphazero_network import AlphaZeroNetwork

__all__ = ['ResBlock', 'AlphaZeroNetwork']
