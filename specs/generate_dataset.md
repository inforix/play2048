# 2048 Dataset Generation Specification

## Overview

**Purpose**: Automated generation of high-quality training datasets for 2048 neural network models using Expectimax AI gameplay.

**Design Philosophy**: Create diverse, expert-level game trajectories through algorithmic play rather than manual data collection, ensuring consistency, scalability, and reproducibility.

**Target Use Case**: Generate 100K+ training samples (500-1000 games) for supervised learning of CNN, Dual Network, and Transformer models as specified in `train_spec.md`.

---

## Game Implementation Specification

### Core Game Logic

**Board Representation**:
- 4×4 NumPy array of int32 values
- Tile values: {0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, ...}
- Zero (0) represents empty cell

**Tile Spawning Rules**:
- New tiles appear in random empty cells after each valid move
- Probability distribution: 90% chance of 2, 10% chance of 4
- Initial state: 2 random tiles (following standard 2048 rules)

**Movement Mechanics**:
- Directions: {up, down, left, right}
- Tiles slide in direction until blocked by edge or another tile
- Adjacent identical tiles merge: `tile + tile = 2 × tile`
- Score increment: merged tile value added to total score
- Multiple merges in single move allowed (but each tile merges max once)
- Move validity: must change at least one tile position or merge

**Line Processing Algorithm** (for each row/column):
```
1. Extract non-zero tiles
2. Iterate left-to-right (or direction-dependent):
   - If current == next: merge, add to score, skip next
   - Else: keep current, continue
3. Pad result with zeros to original length
```

**Game Termination Conditions**:
- Win condition: 2048 tile reached (game continues)
- Loss condition: board full AND no valid moves (no merges possible)
- Safety limit: 5000 moves (prevents infinite loops in degenerate cases)

**State Tracking**:
- Current board configuration
- Cumulative score
- Move history (board_before, direction, score_before) for each move
- Game outcome (won, max_tile, final_score)

---

## Expectimax AI Specification

### Algorithm Design

**Strategy**: Expectimax search with comprehensive board evaluation heuristics.

**Search Parameters**:
- Default depth: 4 ply (2 player moves + 2 chance nodes)
- Configurable via `--depth` argument (range: 2-6)
- Player nodes: maximize evaluation score
- Chance nodes: expected value over tile placements

**Chance Node Sampling**:
- Full evaluation too expensive (16! possible placements)
- Sample up to 4 random empty cells per chance node
- For each sampled cell:
  - P(tile=2) = 0.9, P(tile=4) = 0.1
  - Expected value = 0.9 × V(state with 2) + 0.1 × V(state with 4)
- Average over sampled positions

**Move Selection**:
```
For each direction in {up, down, left, right}:
  If move is valid:
    Simulate move
    Evaluate resulting state with Expectimax(depth-1)
    Track best score
Return direction with highest score
```

### Evaluation Function

**Composite Heuristic** (weighted sum of 5 components):

```python
score = position_score + empty_bonus + monotonicity_score + smoothness_score + corner_bonus
```

**1. Position Score (Snake Pattern Weighting)**:
```
Weight Matrix:
  [4^15  4^14  4^13  4^12]     [≈10^9   ≈10^8   ≈10^8   ≈10^7 ]
  [4^8   4^9   4^10  4^11]  =  [≈10^5   ≈10^5   ≈10^6   ≈10^6 ]
  [4^7   4^6   4^5   4^4 ]     [≈10^4   ≈10^4   ≈10^3   ≈10^2 ]
  [4^0   4^1   4^2   4^3 ]     [1       4       16      64    ]

Position Score = Σ(board[i][j] × weight[i][j])
```
- Encourages high-value tiles in top-right corner
- S-shaped pattern descending down-left
- Dominant term in evaluation (can reach 10^12 for 2048 in corner)

**2. Empty Tiles Bonus**:
```
Empty Bonus = count(zeros) × 50,000
```
- Critical for maintaining maneuverability
- Prevents board lock-up
- Typical contribution: 0-800,000 (0-16 empty cells)

**3. Monotonicity Score**:
```
For each row:
  Count transitions where board[i][j] ≤ board[i][j+1]  (increasing)
  Count transitions where board[i][j] ≥ board[i][j+1]  (decreasing)
  Row monotonicity = max(increasing, decreasing)

Similarly for each column.

Monotonicity Score = (sum of all row/col monotonicity) × 10,000
```
- Rewards organized tile arrangement
- Best case: 24 (all rows/cols perfectly monotonic) → 240,000
- Typical: 15-20 → 150,000-200,000

**4. Smoothness Score**:
```
For each non-zero tile:
  Compare log2(tile) with log2(right_neighbor) if exists
  Compare log2(tile) with log2(bottom_neighbor) if exists
  Smoothness -= |log2 difference|

Smoothness Score = smoothness × 1,000
```
- Negative penalty for large jumps (e.g., 2 next to 512)
- Encourages similar-valued tiles to be adjacent
- Typical range: -50 to 0 → -50,000 to 0

**5. Corner Anchoring Bonus**:
```
If board[0][3] == max(board):  # Top-right corner
  Corner Bonus = max_tile × 10,000
Else:
  Corner Bonus = 0
```
- Strong incentive to keep largest tile in top-right
- For 2048 tile: bonus = 20,480,000
- Combined with position weights: ensures corner dominance

**Evaluation Function Characteristics**:
- Range: ~10^6 (poor position) to ~10^12 (2048 in corner with good structure)
- Position weights dominate late game
- Empty tiles critical in mid-game
- Monotonicity/smoothness fine-tune decisions

---

## Dataset Format Specification

### Output Format: JSONL (JSON Lines)

**File Structure**:
- Each line = one complete game (valid JSON object)
- Line-delimited for streaming processing
- Compatible with `torch.utils.data.Dataset` line-by-line reading

**Game Record Schema**:
```json
{
  "totalMoves": integer,          // Number of moves in game (50-500 typical)
  "finalScore": integer,          // Cumulative score at game end
  "maxTile": integer,             // Highest tile value achieved (256-4096)
  "won": boolean,                 // True if 2048+ tile reached
  "finalBoard": [[int]],          // 4×4 final board state
  "timestamp": string,            // ISO 8601 datetime of game completion
  "moves": [MoveRecord]           // Array of move records
}
```

**Move Record Schema**:
```json
{
  "board": [[int]],               // 4×4 board state BEFORE this move
  "direction": string,            // Action taken: "up"|"down"|"left"|"right"
  "score": integer                // Cumulative score BEFORE this move
}
```

**Critical Design Decision**: Board state stored **before** move
- Rationale: Neural network learns state → action mapping
- Input: pre-move board → Output: direction chosen
- Post-move board can be derived via game simulation (not needed for supervised learning)

### Alternative Format: JSON Array

**File Structure**:
```json
[
  { game1 },
  { game2 },
  ...
]
```
- Enabled via `.json` extension instead of `.jsonl`
- Entire file must be loaded to memory
- Not recommended for large datasets (1000+ games)

---

## Command-Line Interface Specification

### Invocation Syntax
```bash
# Using uv (recommended and enforced)
uv run python generate_dataset.py [OPTIONS]

# Or if in activated venv
python generate_dataset.py [OPTIONS]
```

### Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--games` | `-g` | int | 500 | Number of games to play |
| `--output` | `-o` | str | `data/training_games.jsonl` | Output file path |
| `--depth` | `-d` | int | 4 | Expectimax search depth (2-6) |
| `--verbose` | `-v` | flag | False | Print per-game progress |
| `--seed` | - | int | None | Random seed for reproducibility |

### Argument Specifications

**`--games` / `-g`**:
- Valid range: 1 to 10,000 (practical limit)
- Recommended values:
  - Quick test: 10-50 games
  - Development: 100-200 games
  - Training: 500-1000 games
  - Production: 2000-5000 games
- Estimated time: ~5-10 seconds per game at depth=4

**`--output` / `-o`**:
- Format auto-detected from extension:
  - `.jsonl` → JSON Lines (recommended)
  - `.json` → JSON array
- Directory created automatically if doesn't exist
- Overwrites existing file without warning

**`--depth` / `-d`**:
- Valid range: 2-6
- Performance vs. quality trade-off:
  - depth=2: Fast (~1s/game), poor play (~30% win rate)
  - depth=3: Medium (~2s/game), decent play (~50% win rate)
  - depth=4: Standard (~5-10s/game), good play (~70-80% win rate) ← **default**
  - depth=5: Slow (~30s/game), excellent play (~85% win rate)
  - depth=6: Very slow (~2min/game), marginal improvement
- Higher depth → better data quality but slower generation

**`--verbose` / `-v`**:
- Prints every 50 moves during each game
- Shows: move count, current score, max tile
- Useful for monitoring long runs
- Example output:
  ```
  🎯 Game 1/500
  Move 50: Score 1234, Max tile 128
  Move 100: Score 3456, Max tile 256
  ✓ Finished: Score 15000, Max tile 2048, Moves 178
  ```

**`--seed`**:
- Sets both Python `random` and NumPy `random` seeds
- Enables deterministic dataset generation
- Use cases:
  - Reproducible experiments
  - Consistent train/test splits
  - Debugging game logic

### Usage Examples

```bash
# Standard training dataset
uv run python generate_dataset.py --games 500 --output data/train.jsonl

# High-quality validation set (deeper search)
uv run python generate_dataset.py --games 100 --depth 5 --output data/val.jsonl

# Quick test with reproducibility
uv run python generate_dataset.py --games 10 --seed 42 --verbose

# Large production dataset
uv run python generate_dataset.py --games 2000 --output data/production.jsonl

# Development iteration
uv run python generate_dataset.py -g 50 -d 3 -o data/dev.jsonl -v
```

---

## Output Statistics Specification

### Real-Time Progress

**Progress Bar** (via tqdm):
```
Playing games: 100%|██████████| 500/500 [45:30<00:00,  5.46s/it]
```
- Shows: percentage, bar, games completed/total, elapsed time, speed

### Post-Generation Summary

**Statistics Report**:
```
📊 Dataset Statistics
============================================================
Total games: 500
Wins (2048+): 387 (77.4%)
Average score: 16,234
Average moves per game: 167.3
Max tile achieved: 4096

Max tile distribution:
  4096:   12 games (  2.4%)
  2048:  375 games ( 75.0%)
  1024:   98 games ( 19.6%)
   512:   13 games (  2.6%)
   256:    2 games (  0.4%)

Total training samples: 83,650
Average samples per game: 167

✅ Dataset generation complete!
Output saved to: data/training_games.jsonl
```

### Quality Metrics

**Expected Performance** (depth=4):
- **Win rate**: 70-80% (reaches 2048 tile)
- **Score distribution**:
  - Median: ~14,000
  - Mean: ~16,000
  - 75th percentile: ~20,000
  - 90th percentile: ~25,000
- **Move count**: 150-200 moves per game
- **Max tile distribution**:
  - 2048: ~75%
  - 1024: ~20%
  - 4096: ~2-5%
  - 512 or less: <5%

**Data Quality Indicators**:
- Low variance in win rate across runs (±5%) → consistent AI
- Few very short games (<50 moves) → no catastrophic failures
- Occasional 4096 tiles → AI explores high-value strategies
- Move diversity: all 4 directions used (not just up/right)

---

## Technical Specifications

### Dependencies

**Required**:
- Python 3.8+
- NumPy >= 1.20.0 (array operations, game board)
- tqdm >= 4.60.0 (progress bars)

**Package Manager**: `uv` (enforced)
```bash
# Install dependencies
uv pip install -r requirements.txt

# Or with pyproject.toml
uv sync
```

**Standard Library**:
- `argparse` (CLI parsing)
- `json` (output serialization)
- `random` (tile spawning, sampling)
- `datetime` (timestamps)
- `copy` (game state cloning)
- `os` (directory creation)
- `collections.Counter` (statistics)

**No ML Dependencies**: Intentionally lightweight for standalone use

### Performance Characteristics

**Time Complexity**:
- Expectimax: O(b^d × s) where:
  - b = branching factor (4 directions)
  - d = search depth
  - s = sampling factor (4 cells sampled)
- Per move: ~10^3 to 10^4 board evaluations at depth=4
- Per game: ~150 moves × 10^4 evals = ~1.5M evaluations

**Space Complexity**:
- Game state: O(1) - fixed 4×4 board
- Move history: O(n) where n = game length
- Expectimax recursion stack: O(d) - search depth
- Peak memory: <100 MB for 500 games

**Computational Cost** (estimated, depth=4):
- Moves per second: ~0.1-0.2 (5-10 seconds per move)
- Games per hour: ~50-100
- 500 games: ~40-60 minutes
- 1000 games: ~80-120 minutes

**Scaling**:
- Linear in number of games (parallelizable)
- Exponential in search depth (use depth=3 for speed)
- Memory usage grows linearly with games (stream to disk)

### Determinism & Reproducibility

**Random Components**:
1. Tile spawning (position and value)
2. Chance node sampling in Expectimax

**Reproducibility Requirements**:
- Use `--seed` argument
- Same Python/NumPy versions
- Same `--depth` parameter

**Non-Deterministic Factors**:
- Floating-point precision (negligible impact)
- Order of dictionary iteration (Python 3.7+ stable)

---

## Integration with Training Pipeline

### Compatibility Matrix

| Dataset Format | Training Method | Compatible |
|---------------|-----------------|------------|
| JSONL output | Method 1 (CNN) | ✅ Yes |
| JSONL output | Method 2 (Dual) | ✅ Yes |
| JSONL output | Method 3 (Transformer) | ✅ Yes |
| JSON output | All methods | ✅ Yes (slower loading) |

### Data Augmentation Synergy

**Generated Data** + **Augmentation** = 8× Effective Dataset Size

Example:
```
500 games × 167 moves/game = 83,500 base samples
× 8 (rotations + reflections) = 668,000 training samples
```

### Dataset Split Recommendations

**For 500 games**:
```
Training:   350 games (70%) → ~58,000 base samples → ~464,000 augmented
Validation:  75 games (15%) → ~12,500 base samples → ~100,000 augmented
Test:        75 games (15%) → ~12,500 base samples → ~100,000 augmented
```

**For 1000 games**:
```
Training:   700 games (70%) → ~116,000 samples → ~928,000 augmented
Validation: 150 games (15%) → ~25,000 samples → ~200,000 augmented
Test:       150 games (15%) → ~25,000 samples → ~200,000 augmented
```

### Quality vs. Quantity Trade-off

| Configuration | Games | Depth | Time | Quality | Use Case |
|--------------|-------|-------|------|---------|----------|
| Quick | 100 | 3 | ~20 min | Medium | Prototyping |
| Standard | 500 | 4 | ~60 min | High | Development |
| Production | 1000 | 4 | ~120 min | High | Training |
| Premium | 500 | 5 | ~4 hours | Excellent | Final model |

---

## Validation & Quality Assurance

### Automated Checks

**During Generation**:
- ✓ All moves are valid (change board state)
- ✓ No board states violate game rules
- ✓ Scores are monotonically increasing
- ✓ Max tile values follow power-of-2 sequence

**Post-Generation**:
```bash
# Verify JSONL syntax
python -c "import json; [json.loads(l) for l in open('data/train.jsonl')]"

# Check schema compliance
python -c "
data = [json.loads(l) for l in open('data/train.jsonl')]
assert all('moves' in g and 'finalScore' in g for g in data)
"
```

### Manual Spot Checks

**Sample Inspection**:
```python
import json

# Load first game
with open('data/train.jsonl') as f:
    game = json.loads(f.readline())

# Verify structure
print(f"Moves: {game['totalMoves']}")
print(f"Score: {game['finalScore']}")
print(f"Max tile: {game['maxTile']}")
print(f"First move: {game['moves'][0]['direction']}")

# Check board state
import numpy as np
board = np.array(game['moves'][0]['board'])
print(f"Initial board:\n{board}")
```

### Statistical Validation

**Expected Distributions**:
- Win rate: 70-80% (depth=4)
- Score: log-normal distribution (right-skewed)
- Move count: normal distribution (mean ~170, std ~40)
- Max tile: discrete (peaks at 2048)

**Red Flags**:
- ⚠️ Win rate < 50%: AI not performing well
- ⚠️ Average score < 10,000: Check evaluation function
- ⚠️ No 4096 tiles in 1000 games: Search depth too shallow
- ⚠️ Move count < 100: Games ending too early

---

## Limitations & Considerations

### Known Limitations

**1. Deterministic AI**:
- Same board state → always same move
- Reduces strategic diversity
- Mitigation: Multiple runs with different seeds

**2. Search Horizon**:
- Depth=4 only sees 2 moves ahead
- May miss long-term tactics
- Trade-off: deeper search = exponentially slower

**3. Sampling Bias**:
- Chance nodes sample only 4 cells
- May miss rare but important scenarios
- Impact: <5% difference in evaluation

**4. Evaluation Function**:
- Hand-crafted heuristics (not learned)
- May have blind spots
- Alternative: Use RL-trained policy (future work)

### Alternative AI Strategies (Not Implemented)

**Monte Carlo Tree Search (MCTS)**:
- Pros: Less prone to evaluation errors
- Cons: Slower, requires more games for convergence

**Reinforcement Learning Policy**:
- Pros: Could exceed expert performance
- Cons: Requires training first, not available for bootstrapping

**Human Play**:
- Pros: Natural strategic diversity
- Cons: Slow, inconsistent quality, labor-intensive

---

## Future Enhancements

### Planned Features

**1. Multi-Threading**:
```bash
python generate_dataset.py --games 1000 --workers 8
```
- Parallel game execution
- 8× speedup on multi-core CPUs

**2. Curriculum Datasets**:
```bash
python generate_dataset.py --curriculum easy --games 100
python generate_dataset.py --curriculum medium --games 300
python generate_dataset.py --curriculum hard --games 100
```
- Easy: depth=2, focus on basic moves
- Medium: depth=3, early-mid game
- Hard: depth=5, complex endgames

**3. Targeted Sampling**:
```bash
python generate_dataset.py --min-score 15000 --min-tile 2048
```
- Filter out low-quality games
- Focus on successful strategies

**4. Metadata Enrichment**:
```json
{
  "moves": [
    {
      "board": [[...]],
      "direction": "up",
      "score": 100,
      "evaluation_score": 1234567,     // NEW
      "alternative_moves": ["left"],    // NEW
      "nodes_evaluated": 3456           // NEW
    }
  ]
}
```

**5. Export Formats**:
- CSV: For non-Python tools
- HDF5: Fast numerical array access
- Parquet: Columnar storage for large datasets

---

## Appendix

### File Locations

```
play2048/
├── generate_dataset.py        # This script
├── generate_dataset.md         # This specification
├── train_spec.md              # Training specification
├── data/                      # Output directory (created automatically)
│   ├── training_games.jsonl   # Default output
│   ├── validation_games.jsonl
│   └── test_games.jsonl
└── tools/                     # Future utilities
    └── analyze_dataset.py     # Dataset statistics tool (future)
```

### Quick Reference

**Generate standard dataset**:
```bash
python generate_dataset.py --games 500
```

**Validate output**:
```bash
wc -l data/training_games.jsonl  # Should be 500
```

**Inspect first game**:
```bash
head -n 1 data/training_games.jsonl | python -m json.tool
```

**Count total moves**:
```bash
python -c "
import json
total = sum(json.loads(l)['totalMoves'] for l in open('data/training_games.jsonl'))
print(f'Total moves: {total:,}')
"
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-07  
**Companion Scripts**: `generate_dataset.py`  
**Status**: Production Ready
