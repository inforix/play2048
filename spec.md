# 2048 Aurora Edition — Single Page Spec

## Overview
- A single HTML page (`index.html`) implementing the 2048 game with an Aurora/Dawn theme, responsive grid, and glassmorphic styling.
- Supports keyboard (arrows/WASD), swipe on touch, undo, theme toggle, mute toggle, AI suggestion, and AI autoplay.
- State persisted in `localStorage`: best score, mute, theme, Azure API key.

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

## Heuristic Fallback
- Simple scoring: prefers moves that merge, create empty spaces, slight bias to left/down in fallback order; used when AI unavailable.

## Persistence
- `localStorage`: `aurora-2048-best`, `aurora-2048-muted`, `aurora-2048-theme`, `aurora-2048-azure-key`.

## Files
- `index.html`: contains HTML, CSS, JS, AI logic, and all controls.
- `spec.md`: this specification.***
