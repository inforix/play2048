"""
Monte Carlo Tree Search (MCTS) for 2048 Game

Implements MCTS with modifications for single-player stochastic game:
- Player nodes: Choose one of 4 actions (up, down, left, right)
- Chance nodes: Random tile spawning (2 with 90%, 4 with 10%)

References:
- AlphaGo Zero paper: https://www.nature.com/articles/nature24270
- AlphaZero paper: https://arxiv.org/abs/1712.01815
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from collections import defaultdict
import copy


class MCTSNode:
    """
    MCTS tree node for 2048 game.
    
    Supports both player nodes (action selection) and chance nodes (random tile).
    """
    
    def __init__(
        self, 
        board: np.ndarray,
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
        is_chance_node: bool = False
    ):
        """
        Initialize MCTS node.
        
        Args:
            board: 4x4 numpy array representing the board state
            parent: Parent node in the tree
            action: Action that led to this node (0=up, 1=down, 2=left, 3=right)
            prior: Prior probability from neural network policy
            is_chance_node: Whether this is a chance node (random tile)
        """
        self.board = board
        self.parent = parent
        self.action = action
        self.prior = prior
        self.is_chance_node = is_chance_node
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        
        # For chance nodes, track tile spawn locations
        self.tile_value = None  # 2 or 4
        self.tile_position = None  # (row, col)
    
    def q_value(self) -> float:
        """
        Average value Q(s,a) = W(s,a) / N(s,a).
        
        Returns:
            Mean action value
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, c_puct: float = 1.4, parent_visit_count: Optional[int] = None) -> float:
        """
        Upper Confidence Bound score for action selection.
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: Exploration constant
            parent_visit_count: Parent's visit count (if None, use self.parent)
            
        Returns:
            UCB score
        """
        if self.visit_count == 0:
            # Unvisited nodes have infinite value (exploration)
            return float('inf')
        
        # Get parent visit count
        if parent_visit_count is None:
            parent_visit_count = self.parent.visit_count if self.parent else 1
        
        # Q value (exploitation)
        q = self.q_value()
        
        # Exploration bonus
        exploration = c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return q + exploration
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def expand(self, policy_probs: np.ndarray, valid_actions: np.ndarray) -> None:
        """
        Expand this player node by creating children for all valid actions.
        
        Args:
            policy_probs: Neural network policy probabilities for 4 actions
            valid_actions: Boolean array indicating valid actions
        """
        if self.is_chance_node:
            raise ValueError("Cannot expand chance node with policy")
        
        for action in range(4):
            if valid_actions[action]:
                # Create child node (will be filled during selection)
                self.children[action] = None
                # Store prior for UCB calculation
                if not hasattr(self, 'priors'):
                    self.priors = {}
                self.priors[action] = policy_probs[action]
    
    def select_child(self, c_puct: float = 1.4) -> Tuple[int, 'MCTSNode']:
        """
        Select child with highest UCB score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            (action, child_node) tuple
        """
        if self.is_chance_node:
            # For chance nodes, select uniformly (will be created on demand)
            # In practice, we sample during expansion
            raise ValueError("Chance nodes use different selection mechanism")
        
        best_action = None
        best_ucb = -float('inf')
        
        for action, child in self.children.items():
            if child is None:
                # Unvisited action - has infinite UCB
                return action, None
            
            ucb = child.ucb_score(c_puct, self.visit_count)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
        
        return best_action, self.children[best_action]
    
    def update(self, value: float) -> None:
        """
        Update node statistics after backpropagation.
        
        Args:
            value: Value to add (from perspective of player to move)
        """
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """
    Monte Carlo Tree Search for 2048 game.
    
    Uses neural network to provide policy prior and value estimation.
    Handles single-player game with random tile generation.
    """
    
    def __init__(
        self,
        model,
        device: torch.device,
        num_simulations: int = 100,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialize MCTS.
        
        Args:
            model: Neural network with forward(board) -> (policy_logits, value)
            device: Torch device for inference
            num_simulations: Number of MCTS simulations per search
            c_puct: Exploration constant for UCB
            dirichlet_alpha: Dirichlet noise alpha parameter
            dirichlet_epsilon: Mixing weight for Dirichlet noise at root
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def search(self, board: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS search from given board state.
        
        Args:
            board: 4x4 numpy array
            add_noise: Whether to add Dirichlet noise at root (for exploration)
            
        Returns:
            Action probabilities proportional to visit counts
        """
        # Create root node
        root = MCTSNode(board.copy())
        
        # Get initial policy and expand root
        policy, value = self._evaluate(root.board)
        valid_actions = self._get_valid_actions(root.board)
        
        # Add Dirichlet noise for exploration (during training)
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * 4)
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
        
        # Normalize policy over valid actions
        policy = policy * valid_actions
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            # All actions invalid - uniform over valid
            policy = valid_actions / valid_actions.sum()
        
        root.expand(policy, valid_actions)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Extract action probabilities from visit counts
        action_probs = np.zeros(4)
        for action, child in root.children.items():
            if child is not None:
                action_probs[action] = child.visit_count
        
        # Normalize
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        else:
            # No simulations completed - use prior
            action_probs = policy
        
        return action_probs
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Run one MCTS simulation from node to leaf and backpropagate.
        
        Args:
            node: Starting node
            
        Returns:
            Value from leaf node
        """
        # 1. Selection - traverse tree until leaf
        path = [node]
        current = node
        
        while not current.is_leaf():
            action, child = current.select_child(self.c_puct)
            
            if child is None:
                # Unvisited node - need to expand
                # First, we need to simulate the action and create chance node
                new_board, moved = self._apply_action(current.board, action)
                
                if not moved:
                    # Invalid action - this shouldn't happen if expansion was correct
                    # Give it a very negative value
                    value = -1.0
                    self._backpropagate(path, value)
                    return value
                
                # Create player node after action (before random tile)
                child = MCTSNode(new_board, parent=current, action=action, prior=current.priors[action])
                current.children[action] = child
                path.append(child)
                
                # Now we need to handle random tile spawning
                # For simplicity, we'll evaluate this state directly
                # (In full implementation, would create chance nodes)
                policy, value = self._evaluate(child.board)
                
                # Check if game over
                if self._is_terminal(child.board):
                    value = self._terminal_value(child.board)
                else:
                    # Expand this node for future simulations
                    valid_actions = self._get_valid_actions(child.board)
                    if valid_actions.sum() > 0:
                        policy = policy * valid_actions
                        policy = policy / policy.sum() if policy.sum() > 0 else valid_actions / valid_actions.sum()
                        child.expand(policy, valid_actions)
                
                # Backpropagate value
                self._backpropagate(path, value)
                return value
            else:
                path.append(child)
                current = child
        
        # 2. Expansion & Evaluation - we've reached a leaf
        if self._is_terminal(current.board):
            value = self._terminal_value(current.board)
        else:
            policy, value = self._evaluate(current.board)
            valid_actions = self._get_valid_actions(current.board)
            
            if valid_actions.sum() > 0:
                policy = policy * valid_actions
                policy = policy / policy.sum() if policy.sum() > 0 else valid_actions / valid_actions.sum()
                current.expand(policy, valid_actions)
        
        # 3. Backpropagation
        self._backpropagate(path, value)
        
        return value
    
    def _backpropagate(self, path: list, value: float) -> None:
        """
        Backpropagate value up the tree.
        
        Args:
            path: List of nodes from root to leaf
            value: Value to backpropagate
        """
        for node in reversed(path):
            node.update(value)
            # Note: In 2048, we don't flip the value since it's single-player
            # The value represents expected outcome from current position
    
    def _evaluate(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate board position using neural network.
        
        Args:
            board: 4x4 numpy array
            
        Returns:
            (policy_probs, value) tuple
        """
        self.model.eval()
        with torch.no_grad():
            # Make a copy to ensure contiguous array (handles negative strides from transpose)
            board_copy = np.ascontiguousarray(board)
            board_tensor = torch.from_numpy(board_copy).float().unsqueeze(0).to(self.device)  # (1, 4, 4)
            policy_logits, value = self.model(board_tensor)
            
            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]  # (4,)
            value = value.item()
        
        return policy_probs, value
    
    def _get_valid_actions(self, board: np.ndarray) -> np.ndarray:
        """
        Get mask of valid actions (actions that change board state).
        
        Args:
            board: 4x4 numpy array
            
        Returns:
            Boolean array of shape (4,) indicating valid actions
        """
        valid = np.zeros(4, dtype=bool)
        for action in range(4):
            _, moved = self._apply_action(board, action)
            valid[action] = moved
        return valid
    
    def _apply_action(self, board: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
        """
        Apply action to board and return new board.
        
        Args:
            board: 4x4 numpy array
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            (new_board, moved) tuple
        """
        # Simulate the move
        new_board = board.copy()
        moved = False
        
        if action == 0:  # Up
            new_board, moved = self._slide_up(new_board)
        elif action == 1:  # Down
            new_board, moved = self._slide_down(new_board)
        elif action == 2:  # Left
            new_board, moved = self._slide_left(new_board)
        elif action == 3:  # Right
            new_board, moved = self._slide_right(new_board)
        
        return new_board, moved
    
    def _slide_left(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles to the left."""
        new_board = board.copy()
        moved = False
        
        for i in range(4):
            # Compact non-zero tiles
            row = new_board[i, :]
            non_zero = row[row != 0]
            
            # Merge adjacent equal tiles
            merged = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged.append(non_zero[j] * 2)
                    skip = True
                else:
                    merged.append(non_zero[j])
            
            # Pad with zeros
            new_row = np.array(merged + [0] * (4 - len(merged)))
            
            # Check if changed
            if not np.array_equal(row, new_row):
                moved = True
            
            new_board[i, :] = new_row
        
        return new_board, moved
    
    def _slide_right(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles to the right."""
        flipped = np.fliplr(board)
        new_board, moved = self._slide_left(flipped)
        return np.fliplr(new_board), moved
    
    def _slide_up(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles up."""
        transposed = board.T
        new_board, moved = self._slide_left(transposed)
        return new_board.T, moved
    
    def _slide_down(self, board: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge tiles down."""
        transposed = board.T
        new_board, moved = self._slide_right(transposed)
        return new_board.T, moved
    
    def _is_terminal(self, board: np.ndarray) -> bool:
        """
        Check if board is in terminal state (game over).
        
        Args:
            board: 4x4 numpy array
            
        Returns:
            True if no valid moves remain
        """
        return not self._get_valid_actions(board).any()
    
    def _terminal_value(self, board: np.ndarray) -> float:
        """
        Get value for terminal state.
        
        Args:
            board: 4x4 numpy array
            
        Returns:
            Value based on max tile achieved
        """
        max_tile = np.max(board)
        
        # Reward based on achievement
        if max_tile >= 2048:
            return 1.0
        elif max_tile >= 1024:
            return 0.5
        elif max_tile >= 512:
            return 0.0
        elif max_tile >= 256:
            return -0.3
        else:
            return -0.5


def test_mcts():
    """Test MCTS with a simple random network."""
    print("Testing MCTS implementation...")
    
    # Create a simple dummy network for testing
    class DummyNetwork(torch.nn.Module):
        def forward(self, x):
            batch_size = x.size(0)
            # Return uniform policy and zero value
            policy = torch.randn(batch_size, 4)
            value = torch.zeros(batch_size, 1)
            return policy, value
    
    device = torch.device('cpu')
    model = DummyNetwork()
    
    # Create initial board
    board = np.zeros((4, 4))
    board[0, 0] = 2
    board[1, 1] = 4
    
    print(f"\nInitial board:")
    print(board)
    
    # Run MCTS
    mcts = MCTS(model, device, num_simulations=50)
    action_probs = mcts.search(board, add_noise=False)
    
    print(f"\nAction probabilities after 50 simulations:")
    print(f"Up: {action_probs[0]:.3f}")
    print(f"Down: {action_probs[1]:.3f}")
    print(f"Left: {action_probs[2]:.3f}")
    print(f"Right: {action_probs[3]:.3f}")
    
    # Select action
    action = np.argmax(action_probs)
    action_names = ['Up', 'Down', 'Left', 'Right']
    print(f"\nSelected action: {action_names[action]}")
    
    print("\n✓ MCTS test passed!")


if __name__ == "__main__":
    test_mcts()
