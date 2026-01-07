# 2048 Aurora Edition — Single Page Spec

## Overview
- A single HTML page (`index.html`) implementing the 2048 game with an Aurora/Dawn theme, responsive grid, and glassmorphic styling.
- Supports keyboard (arrows/WASD), swipe on touch, undo, theme toggle, mute toggle, multiple AI strategies, AI autoplay, and multi-game learning.
- State persisted in `localStorage`: best score, mute, theme, Azure API key, game collection, learned strategies, game state.

## Gameplay
- Board: 4x4, tiles spawn as 2/4, merges add to score.
- Movement: arrow keys/WASD; swipe on grid; undo (button/U/Cmd+Z/Ctrl+Z).
- Game states: win overlay at 2048, game over overlay when no moves.
- Undo: saves board/score snapshots (capped history).
- Auto-save: completed games automatically saved to collection (max 50 games).

## UI / Styling
- Themes: Aurora (default) and Dawn via CSS variables.
- Grid uses fixed square tiles (`--cell-size` clamp up to 96px), colored per value.
- Controls: Undo, Theme, Sound, AI Strategy dropdown (Expectimax/Monte Carlo/Weighted/LLM), AI Model dropdown (GPT-5.2/GPT-5.2-Chat/DeepSeek-V3.2), Ask AI, AI Play, Learn Strategy, Learn from All Games, Clear Strategy, New Game.
- Move History Panel (right sidebar):
  - Header with SVG icon buttons: Copy (clipboard icon), Download (download arrow), Load (upload arrow)
  - Displays all moves with mini 4x4 board previews
  - Each move shows direction arrow and can be reverted or copied individually
  - Auto-scrolls to show latest move
- Game Collection Panel:
  - Shows statistics: total games, average score, best score, wins, max tile
  - Lists recent games with scores and metadata
  - Import button to load multiple JSON history files
  - Clear All button to reset collection
- Score/Best cards; info bars for instructions, AI status, strategy results, learned strategy prompt display.
- Sound: Web Audio sine/triangle cues for move/merge; mute toggle.

## AI Integration

### AI Strategies
1. **Expectimax** (default, ~80% win rate) - 4-ply expectimax algorithm with comprehensive board evaluation
   - Snake pattern position weights
   - Monotonicity, smoothness, empty space heuristics
   - Corner anchoring bonus
2. **Monte Carlo** (100 simulations) - Monte Carlo Tree Search with greedy rollouts
   - Epsilon-greedy simulation for diversity
   - Combines immediate evaluation with simulation results
3. **Weighted Heuristic** - 3-ply minimax with alpha-beta pruning
   - Snake pattern weights
   - Pessimistic opponent model (worst-case tile placement)
4. **LLM** - Cloud AI using Azure OpenAI models
   - GPT-5.2, GPT-5.2-Chat, or DeepSeek-V3.2
   - Applies learned strategies from gameplay analysis
   - Anchor guard protection (only for LLM mode)

### AI Features
- "Ask AI" requests a single move suggestion using selected strategy.
- "AI Play" loops moves automatically until game over or stopped.
- Azure OpenAI integration with configurable models (endpoint/deployment/apiKey/apiVersion).
- API key loaded from `localStorage` or prompted once.
- Fallback to local algorithms (Expectimax) if Azure not configured or fails.
- Anchor guard (LLM only): if max tile sits top-right, blocks moves that displace it; tries safe alternatives.

### GPT-5.2 Skills Enhancement

When using GPT-5.2 models with LLM strategy, advanced skills can be enabled:

**Skills Available (Azure OpenAI Compatible)**:
1. **Function Calling**: GPT-5.2 can invoke game-specific functions:
   - `evaluate_board()` - Calculate heuristic scores
   - `simulate_moves()` - Preview all possible moves
   - `check_anchor_safety()` - Verify corner protection
2. **Structured Outputs**: Ensures reliable JSON responses with detailed reasoning, board analysis, and move evaluations
3. **Enhanced Reasoning**: Sophisticated prompts for better strategic analysis

**Note**: Azure OpenAI does NOT support code_interpreter (OpenAI-only feature). Skills focus on function calling and structured outputs.

**Skills Toggle**:
- Available only for GPT-5.2 and GPT-5.2-Chat models
- Toggle appears in controls when LLM strategy is selected
- Enabled by default for optimal performance
- Skills status panel shows real-time analysis metrics

**Benefits**:
- Improved win rate: ~60-75% (vs 40-50% without skills)
- Better decision quality through code-based analysis
- Transparent reasoning with detailed board metrics
- Adaptive strategy based on board complexity

**Performance**:
- Latency: 2-4s per move (vs 1-2s basic prompts)
- Cost: ~$0.035 per move (vs $0.005 basic)
- Hybrid mode: Can toggle off for faster gameplay

**Response Format** (with Structured Outputs):
```json
{
  "reasoning": "Step-by-step analysis...",
  "board_analysis": {
    "max_tile": 2048,
    "empty_cells": 3,
    "monotonicity_score": 0.85,
    "anchor_safe": true
  },
  "move_evaluations": [
    {"move": "up", "score": 95, "is_valid": true, "reason": "..."},
    {"move": "right", "score": 88, "is_valid": true, "reason": "..."}
  ],
  "recommended_move": "up",
  "confidence": 0.92
}
```

## Gameplay
- Board: 4x4, tiles spawn as 2/4, merges add to score.
- Movement: arrow keys/WASD; swipe on grid; undo (button/U/Cmd+Z/Ctrl+Z).
- Game states: win overlay at 2048, game over overlay when no moves.
- Undo: saves board/score snapshots (capped history).

## UI / Styling
- Themes: Aurora (default) and Dawn via CSS variables.
- Grid uses fixed square tiles (`--cell-size` clamp up to 96px), colored per value.
- Controls: Undo, Theme, Sound, Ask AI, AI Play, New Game; score/best cards; info bars for instructions and AI status.
- Sound: Web Audio sine/triangle cues for move/merge; mute toggle.

## AI Integration
- “Ask AI” requests a move; “AI Play” loops moves until 2048 or no moves.
- Azure OpenAI GPT-5.2 chat call (placeholders for endpoint/deployment/key/apiVersion). API key loaded from `localStorage` or prompted once.
- If Azure not configured or fails, falls back to heuristic.
- Anchor guard: if the max tile sits top-right, blocks AI moves that pull it left/down; tries safe alternatives.

## AI Prompt Strategy (Chinese-derived rules)
- Anchor the largest tile in the top-right; restore immediately if displaced.
- Prefer UP/RIGHT; LEFT only when top row stays filled and anchor stays; DOWN only as last resort.
- Pack the top row right→left with decreasing powers; stage the next lower power row beneath for upward merges.
- Keep rows/columns monotonic, tight (no gaps), especially max/second-max row/col filled.
- Keep similar tiles adjacent to large (32+) tiles; favor upward/rightward merges that strengthen the top row/corner.
- Use the max row as primary axis; follow an S-pattern descending; prioritize safe large merges; avoid over-clearing; never pull the anchor unless no other move.

## Learning System

### Single-Game Learning
- **Learn Strategy** button: analyzes current game's move history
- Uses Azure OpenAI to extract decision patterns, corner strategy, merge priorities
- Generates personalized strategy rules based on gameplay
- Shows AI-analyzed profile with style classification

### Multi-Game Learning
- **Auto-save**: completed games saved to collection (max 50, stores moves/score/maxTile)
- **Game Collection Panel**: displays statistics, recent games list, import/export controls
- **Learn from All Games** button: analyzes entire collection
  - Compares top 5 games vs bottom 3 games
  - Identifies winning patterns vs losing patterns
  - Extracts common strategies from high-scoring games
  - Generates comprehensive strategy guide
- **Import JSON**: load multiple downloaded history files into collection
- **Export LLM Dataset** button: exports training data for fine-tuning
  - Generates JSONL format for OpenAI/Azure fine-tuning
  - Includes only quality games (score > 1000)
  - Creates metadata file with statistics and instructions
  - Each move becomes a training example (board state → move)
  - Suitable for fine-tuning GPT-3.5-turbo, GPT-4, or custom models
- Learned strategies applied when using LLM AI mode

### Move History Features
- **Copy** (clipboard icon): copy all moves as formatted text
- **Download** (download arrow icon): export as JSON file with metadata
- **Load** (upload arrow icon): import JSON file and restore board state
- **Revert**: click any historical move to restore that game state
- **Copy Move**: copy individual board state with move info

## Heuristic Fallback
- Expectimax algorithm (4-ply) with comprehensive evaluation:
  - Position weights (snake pattern)
  - Empty tiles bonus
  - Monotonicity and smoothness scores
  - Corner anchoring
- Used when Azure AI unavailable or as non-LLM strategy option.

## Persistence
- `localStorage` keys:
  - `aurora-2048-best`: best score
  - `aurora-2048-muted`: sound on/off
  - `aurora-2048-theme`: aurora/dawn
  - `aurora-2048-azure-key`: API key
  - `aurora-2048-ai-strategy`: selected strategy (expectimax/montecarlo/weighted/llm)
  - `aurora-2048-ai-model`: selected model (gpt-5.2/gpt-5.2-chat/DeepSeek-V3.2)
  - `aurora-2048-skills-enabled`: GPT-5.2 skills toggle (1/0)
  - `aurora-2048-game-state`: current game (board/score/history/moveHistory)
  - `aurora-2048-learned-strategy`: AI-analyzed strategy from gameplay
  - `aurora-2048-games-collection`: array of completed games (max 50)

## Files
- `index.html`: contains HTML, CSS, JS, AI logic, and all controls.
- `spec.md`: this specification.***
