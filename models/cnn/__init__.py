"""
CNN models for 2048 game.

This package contains Convolutional Neural Network implementations
for learning to play 2048.
"""

from .cnn_policy import CNNPolicy, DualCNNPolicy

__all__ = ['CNNPolicy', 'DualCNNPolicy']
