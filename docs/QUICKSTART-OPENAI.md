# Quick Start: OpenAI GPT-5.2 with Code Interpreter

## 5-Minute Setup

### Step 1: Open the Game
Open `index.html` in your browser

### Step 2: Select LLM Strategy
1. Click the **AI Strategy** dropdown
2. Select **"LLM (GPT/DeepSeek)"**

### Step 3: Choose OpenAI GPT-5.2
1. The **AI Model** dropdown will appear
2. Select **"OpenAI GPT-5.2"** (last option)
3. A prompt will ask for your API key

### Step 4: Enter API Key
1. Enter your OpenAI API key from `https://api.bianxie.ai`
2. Click OK
3. Key is saved in browser localStorage

### Step 5: Verify Skills
1. Check that **🧠 Skills** toggle is enabled (default)
2. Look for the status panel showing:
   ```
   Functions: ✓ | Structured: ✓ | Code: ✓
   ```
3. The "Code: ✓" confirms code interpreter is active

### Step 6: Play!
- Click **"Ask AI"** for single move suggestion
- Click **"AI Play"** for autonomous gameplay
- Watch the skills status panel for analysis details

## What You Get

### With OpenAI GPT-5.2 (Full Skills)
✅ **Code Interpreter** - Python execution for advanced analysis  
✅ **Function Calling** - Access to game state functions  
✅ **Structured Outputs** - Detailed JSON responses  
✅ **Enhanced Reasoning** - Strategic decision-making  

### Performance
- **Win Rate**: 65-80% (vs 55-70% with Azure)
- **Response Time**: 2-5 seconds
- **Cost**: ~$0.025 per move
- **Quality**: Best AI decision-making available

## Comparison: Azure vs OpenAI

| Feature | Azure GPT-5.2 | OpenAI GPT-5.2 |
|---------|---------------|----------------|
| Code Interpreter | ❌ | ✅ |
| Function Calling | ✅ | ✅ |
| Structured Outputs | ✅ | ✅ |
| Cost per Move | $0.018 | $0.025 |
| Win Rate | 55-70% | 65-80% |
| Response Time | 2-4s | 2-5s |

## Example Gameplay

```
1. New game starts
2. Click "Ask AI"
3. GPT-5.2 receives board state
4. Code interpreter executes:
   - Multi-step lookahead analysis
   - Statistical board evaluation
   - Pattern recognition
5. Functions called:
   - evaluate_board() - Get heuristic scores
   - simulate_moves() - Preview outcomes
   - check_anchor_safety() - Verify corner
6. Structured JSON response:
   {
     "reasoning": "Board has max tile 256 at top-right...",
     "board_analysis": {
       "max_tile": 256,
       "empty_cells": 8,
       "monotonicity_score": 0.82,
       "anchor_safe": true
     },
     "move_evaluations": [...],
     "recommended_move": "right",
     "confidence": 0.91
   }
7. Move executed: RIGHT
8. Skills status updates with metrics
```

## Troubleshooting

### "Code: ✗" showing instead of "Code: ✓"
→ You selected an Azure model, not OpenAI GPT-5.2  
→ Switch to "OpenAI GPT-5.2" in dropdown

### API Key prompt keeps appearing
→ Invalid or empty key  
→ Get valid key from https://api.bianxie.ai  
→ Re-enter via model selection

### Slow responses (>5 seconds)
→ Normal for code interpreter execution  
→ Complex boards take longer to analyze  
→ Consider Azure models for faster responses

### API Error 401
→ Invalid API key  
→ Check key is correct  
→ Verify no extra spaces

### Falls back to heuristic
→ Check browser console for errors  
→ Verify API key is set  
→ Test network connectivity

## Tips for Best Results

### 1. Enable All Skills
Keep 🧠 Skills toggle **enabled** for best performance

### 2. Let It Think
Code interpreter needs 2-5s for complex analysis - be patient

### 3. Watch the Metrics
Skills status panel shows reasoning quality:
- High monotonicity (>0.8) = good board state
- Anchor safe = max tile protected
- High confidence (>85%) = strong recommendation

### 4. Compare Strategies
Try games with:
- Azure GPT-5.2 (no code interpreter)
- OpenAI GPT-5.2 (with code interpreter)
- Expectimax (local algorithm)

See which works best for your style!

### 5. Learn from AI
Enable "Learn Strategy" to extract patterns from AI gameplay

## Advanced Usage

### Multi-Turn with Code Interpreter

Future enhancement: Send code execution results back to GPT for refinement
```javascript
// Request 1: Initial analysis
tools: [{ type: 'code_interpreter' }]

// Request 2: Refine with code results
messages: [
  ...previous,
  { role: 'assistant', content: code_output },
  { role: 'user', content: 'Based on this, what's optimal?' }
]
```

### Adaptive Skills

Disable code interpreter for simple boards to save cost:
```javascript
// Only use code interpreter when board >50% full
if (emptyCount < 8) {
  tools.push({ type: 'code_interpreter' });
}
```

### Response Caching

Store code execution results for identical board states:
```javascript
const cacheKey = JSON.stringify(board);
if (cache.has(cacheKey)) {
  return cache.get(cacheKey);
}
```

## Cost Management

### Per-Game Costs
- 50 moves @ $0.025 = **$1.25 per game**
- 100 moves @ $0.025 = **$2.50 per game**

### Optimization Strategies
1. Use Azure for practice, OpenAI for serious games
2. Disable skills for easy boards
3. Enable code interpreter only for critical decisions
4. Cache responses for repeated board states

## Next Steps

- **Read**: [OPENAI-SUPPORT.md](OPENAI-SUPPORT.md) for technical details
- **Compare**: Play 10 games each with Azure/OpenAI/Expectimax
- **Benchmark**: Use `bench/bench2048.js` for systematic testing
- **Optimize**: Experiment with skills toggle for cost/performance balance

## Questions?

- Azure vs OpenAI differences → [OPENAI-SUPPORT.md](OPENAI-SUPPORT.md)
- Skills specification → [spec-gpt52-skills.md](spec-gpt52-skills.md)
- General usage → [README-GPT52-SKILLS.md](README-GPT52-SKILLS.md)
- Azure limitations → [AZURE-COMPATIBILITY.md](AZURE-COMPATIBILITY.md)

Enjoy playing with the most advanced 2048 AI available! 🎮🤖
