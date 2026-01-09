"""
AlphaZero Training Loop for 2048.

Implements the iterative self-play and training cycle:
1. Generate self-play data using current model
2. Train network on replay buffer
3. Evaluate new model vs best model
4. Update best model if new model wins
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import copy
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dual import AlphaZeroNetwork
from training.selfplay import ReplayBuffer, generate_self_play_data, self_play_game
from training.utils import get_device, format_time
import time


class AlphaZeroTrainer:
    """Main trainer for AlphaZero."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 0.5
    ):
        """
        Initialize trainer.
        
        Args:
            model: AlphaZero network
            device: Torch device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            policy_loss_weight: Weight for policy loss
            value_loss_weight: Weight for value loss
        """
        self.model = model
        self.device = device
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
    
    def train_epoch(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        num_batches: int = None
    ) -> dict:
        """
        Train for one epoch on replay buffer.
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Mini-batch size
            num_batches: Number of batches (None = full epoch)
            
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Determine number of batches
        if num_batches is None:
            num_batches = max(1, len(replay_buffer) // batch_size)
        
        for _ in range(num_batches):
            # Sample batch
            states, target_policies, target_values = replay_buffer.sample(batch_size)
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)
            
            # Forward pass
            policy_logits, values = self.model(states)
            
            # Calculate losses
            policy_loss = self.policy_criterion(policy_logits, target_policies)
            value_loss = self.value_criterion(values, target_values)
            
            # Combined loss
            loss = (self.policy_loss_weight * policy_loss + 
                   self.value_loss_weight * value_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }


def evaluate_models(
    new_model: nn.Module,
    best_model: nn.Module,
    device: torch.device,
    num_games: int = 50,
    mcts_simulations: int = 100
) -> float:
    """
    Evaluate new model against best model.
    
    Args:
        new_model: Newly trained model
        best_model: Current best model
        device: Torch device
        num_games: Number of evaluation games
        mcts_simulations: MCTS simulations per move
        
    Returns:
        Win rate of new model (0.0 to 1.0)
    """
    new_model.eval()
    best_model.eval()
    
    new_wins = 0
    draws = 0
    
    for _ in tqdm(range(num_games), desc="Evaluation"):
        # Play with new model
        _, new_stats = self_play_game(
            new_model, device, 
            mcts_simulations=mcts_simulations,
            add_noise=False
        )
        
        # Play with best model
        _, best_stats = self_play_game(
            best_model, device,
            mcts_simulations=mcts_simulations,
            add_noise=False
        )
        
        # Compare scores
        if new_stats['score'] > best_stats['score']:
            new_wins += 1
        elif new_stats['score'] == best_stats['score']:
            draws += 1
    
    win_rate = (new_wins + 0.5 * draws) / num_games
    return win_rate


def train_alphazero(
    iterations: int = 100,
    games_per_iteration: int = 100,
    mcts_simulations: int = 100,
    training_epochs: int = 10,
    batch_size: int = 256,
    replay_buffer_size: int = 500000,
    eval_interval: int = 5,
    eval_games: int = 50,
    save_interval: int = 10,
    device: str = 'auto',
    resume_from: str = None,
    output_dir: str = 'checkpoints/alphazero'
):
    """
    Main AlphaZero training loop.
    
    Args:
        iterations: Number of training iterations
        games_per_iteration: Self-play games per iteration
        mcts_simulations: MCTS simulations per move
        training_epochs: Training epochs per iteration
        batch_size: Mini-batch size
        replay_buffer_size: Max replay buffer size
        eval_interval: Evaluate every N iterations
        eval_games: Number of evaluation games
        save_interval: Save checkpoint every N iterations
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        resume_from: Path to checkpoint to resume from
        output_dir: Output directory for checkpoints
    """
    # Setup
    device = get_device() if device == 'auto' else torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    
    # Initialize model
    model = AlphaZeroNetwork(num_blocks=4, channels=256)
    model.to(device)
    best_model = copy.deepcopy(model)
    
    # Initialize trainer
    trainer = AlphaZeroTrainer(model, device)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        best_model.load_state_dict(checkpoint['best_model'])
        start_iteration = checkpoint['iteration'] + 1
        if 'replay_buffer' in checkpoint:
            replay_buffer.load(checkpoint['replay_buffer'])
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"AlphaZero Training - 2048")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Iterations: {iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"MCTS simulations: {mcts_simulations}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for iteration in range(start_iteration, iterations):
        iter_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"{'='*60}")
        
        # 1. Self-play data generation
        print(f"\n🎮 Self-play: Generating {games_per_iteration} games...")
        examples, stats = generate_self_play_data(
            model,
            device,
            num_games=games_per_iteration,
            mcts_simulations=mcts_simulations if iteration >= 10 else 50,  # Lower simulations early
            augment=True,
            verbose=True
        )
        
        # Add to replay buffer
        replay_buffer.add(examples)
        
        # Log self-play stats
        avg_score = np.mean([s['score'] for s in stats])
        avg_moves = np.mean([s['moves'] for s in stats])
        win_rate = np.mean([s['won'] for s in stats]) * 100
        max_tile_dist = {}
        for s in stats:
            max_tile_dist[s['max_tile']] = max_tile_dist.get(s['max_tile'], 0) + 1
        
        print(f"\nSelf-play results:")
        print(f"  Examples generated: {len(examples):,} (×8 augmented)")
        print(f"  Replay buffer size: {len(replay_buffer):,}")
        print(f"  Average score: {avg_score:.0f}")
        print(f"  Average moves: {avg_moves:.0f}")
        print(f"  Win rate (2048): {win_rate:.1f}%")
        print(f"  Max tile distribution: {dict(sorted(max_tile_dist.items()))}")
        
        writer.add_scalar('SelfPlay/AvgScore', avg_score, iteration)
        writer.add_scalar('SelfPlay/WinRate', win_rate, iteration)
        writer.add_scalar('SelfPlay/AvgMoves', avg_moves, iteration)
        
        # 2. Train network
        print(f"\n🔧 Training network for {training_epochs} epochs...")
        for epoch in range(training_epochs):
            metrics = trainer.train_epoch(
                replay_buffer,
                batch_size=batch_size,
                num_batches=min(100, len(replay_buffer) // batch_size)
            )
            
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{training_epochs}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Policy={metrics['policy_loss']:.4f}, "
                      f"Value={metrics['value_loss']:.4f}")
        
        # Update learning rate
        trainer.scheduler.step()
        
        # Log training metrics
        writer.add_scalar('Train/Loss', metrics['loss'], iteration)
        writer.add_scalar('Train/PolicyLoss', metrics['policy_loss'], iteration)
        writer.add_scalar('Train/ValueLoss', metrics['value_loss'], iteration)
        writer.add_scalar('Train/LearningRate', trainer.optimizer.param_groups[0]['lr'], iteration)
        
        # 3. Evaluate model
        if (iteration + 1) % eval_interval == 0:
            print(f"\n⚔️  Evaluating: New model vs Best model ({eval_games} games)...")
            win_rate = evaluate_models(
                model, best_model, device,
                num_games=eval_games,
                mcts_simulations=mcts_simulations
            )
            
            print(f"  New model win rate: {win_rate*100:.1f}%")
            writer.add_scalar('Eval/WinRate', win_rate * 100, iteration)
            
            # Update best model if better
            if win_rate > 0.55:
                print(f"  ✅ New model is better! Updating best model.")
                best_model = copy.deepcopy(model)
            else:
                print(f"  ❌ Best model retained.")
        
        # 4. Save checkpoint
        if (iteration + 1) % save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_iter{iteration+1}.pth'
            torch.save({
                'iteration': iteration,
                'model': model.state_dict(),
                'best_model': best_model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'scheduler': trainer.scheduler.state_dict()
            }, checkpoint_path)
            print(f"\n💾 Saved checkpoint: {checkpoint_path}")
        
        # Iteration summary
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time
        print(f"\n⏱️  Iteration time: {format_time(iter_time)}")
        print(f"⏱️  Total time: {format_time(total_time)}")
        print(f"⏱️  Est. remaining: {format_time(total_time / (iteration - start_iteration + 1) * (iterations - iteration - 1))}")
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'iteration': iterations - 1,
        'model': model.state_dict(),
        'best_model': best_model.state_dict()
    }, final_path)
    print(f"\n✅ Training complete! Final model saved to: {final_path}")
    
    writer.close()
    
    return best_model


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero for 2048')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--games', type=int, default=100,
                       help='Self-play games per iteration')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size')
    
    # Evaluation
    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Evaluate every N iterations')
    parser.add_argument('--eval-games', type=int, default=50,
                       help='Number of evaluation games')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='checkpoints/alphazero',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N iterations')
    
    args = parser.parse_args()
    
    # Run training
    train_alphazero(
        iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_simulations=args.mcts_sims,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        save_interval=args.save_interval,
        device=args.device,
        resume_from=args.resume,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
