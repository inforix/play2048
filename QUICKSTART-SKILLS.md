# Quick Start: GPT-5.2 Skills Testing

**Ready to test in 5 minutes!** 🚀

## Prerequisites

✅ Azure OpenAI account with GPT-5.2 access  
✅ API key and endpoint configured  
✅ Modern web browser (Chrome, Firefox, Edge, Safari)

## Steps to Test

### 1. Open the Game
```bash
# Navigate to project directory
cd /Users/wyp/develop/play2048

# Open in browser (Mac)
open index.html

# Or (Linux/Windows)
# Just double-click index.html
```

### 2. Configure Azure API (First Time Only)

When you first click "Ask AI":
1. Enter your **Azure OpenAI API key**
2. Key is stored locally in browser
3. You won't be asked again

### 3. Enable Skills

In the control panel:
1. Select **AI Strategy** → **"LLM (GPT/DeepSeek)"**
2. Select **AI Model** → **"GPT-5.2"**
3. Verify **🧠 Skills** checkbox appears
4. Ensure it's **checked** ✅

### 4. Start Playing

**Option A: Single Move**
- Click **"Ask AI"**
- Wait 2-4 seconds
- See skills analysis panel
- Move is suggested (or executed if AI Play)

**Option B: Full Game**
- Click **"AI Play"**
- Watch GPT-5.2 play automatically
- Skills status updates each move
- Stops at 2048 or game over

### 5. Verify Skills Are Working

Look for this panel:
```
🧠 GPT-5.2 Skills Active: Max: 2048 | Empty: 3 | Mono: 0.85 | Anchor: ✓ | Conf: 92%
```

If you see this → **Skills are working!** 🎉

## Quick Tests

### Test 1: Skills Toggle
1. **Enable skills** → Click "Ask AI"
2. Note the latency and analysis detail
3. **Disable skills** → Click "Ask AI" again
4. Should be faster but less detailed

### Test 2: Model Comparison
1. Play 5 games with **GPT-5.2 + Skills**
2. Note average score
3. Play 5 games with **GPT-5.2 - Skills**
4. Compare win rates

### Test 3: vs Local Algorithm
1. Select **Expectimax** strategy
2. Play 5 games (very fast)
3. Compare to GPT-5.2 skills performance

## What to Look For

### ✅ Success Indicators:
- Skills toggle visible when GPT-5.2 selected
- Skills status panel shows board analysis
- Moves seem strategic (maintains corner, avoids DOWN)
- Console shows structured JSON (F12 → Console)
- Win rate improves vs basic prompts

### ⚠️ Potential Issues:
- Toggle not visible → Check model selection
- No status panel → Skills may be disabled
- Schema errors → Azure requires strict schemas (all fields required)
- `max_tokens` error → Azure uses `max_completion_tokens` instead
- `code_interpreter` error → Not supported by Azure (already removed)
- Errors in console → Check API key/endpoint
- Slow (>5s) → Network issue or API throttling
- Random moves → Fallback to heuristic (API failed)

## Browser Console Inspection

Press **F12** to open developer tools, then:

1. **Console Tab**: See structured JSON responses
2. **Network Tab**: View API requests/responses
3. **Application Tab** → **Local Storage**: Verify settings

### Expected Console Output:
```javascript
AI Reasoning: Board 75% full. Max tile 1024 at top-right. UP creates merge chain...
Function simulate_moves called: {up: {...}, down: {...}, left: {...}, right: {...}}
Code interpreter output: [{"type": "result", "result": "Best move: up (score: 0.92)"}]
```

## Performance Benchmarks

After 10 games with skills enabled, you should see:

| Metric | Target |
|--------|--------|
| Games reaching 2048 | ≥6/10 (60%) |
| Average score | 18,000+ |
| Average moves | 150-200 |
| Max tile | 2048-4096 |
| API errors | 0-1 |

## Troubleshooting

### Skills Toggle Not Visible
```
Solution:
1. Select "LLM (GPT/DeepSeek)" strategy
2. Select "GPT-5.2" or "GPT-5.2-Chat" model
3. Refresh page if needed
```

### API Key Issues
```
Solution:
1. Open Console (F12)
2. Run: localStorage.removeItem('aurora-2048-azure-key')
3. Refresh page
4. Re-enter correct API key
```

### Slow Performance
```
Normal: 2-4s per move
Slow: >5s per move

Check:
- Network connection
- Azure region latency
- API rate limits
- Skills can be disabled for speed
```

### No Board Analysis
```
Solution:
1. Check skills toggle is ON
2. Open Console, look for errors
3. Verify GPT-5.2 is selected
4. Try "Ask AI" again
```

## Advanced Testing

### Test Code Interpreter
```javascript
// In browser console:
console.log(gpt52Skills);
// Should show: {codeInterpreter: true, ...}
```

### Test Function Calling
```javascript
// Call a function manually:
const board = [[0,0,2,4],[0,0,8,16],[2,4,32,64],[4,8,128,256]];
evaluateBoardFunction(board);
// Should return board metrics
```

### Test Structured Output
```javascript
// Enable verbose logging:
// Edit index.html, add to parseGPT52SkillsResponse:
console.log('Full response:', JSON.stringify(data, null, 2));
```

## Next Steps

### After Successful Testing:

1. **Play Multiple Games**: Build up game collection
2. **Learn Strategy**: Use "📚 Learn Strategy" button
3. **Compare Performance**: Skills vs No-Skills vs Expectimax
4. **Tune Settings**: Experiment with temperature, prompts
5. **Share Results**: Report win rates and insights

### Document Your Results:

```markdown
## My Test Results (Date: _______)

- Games Played: ___
- Win Rate: ___% (__ wins / __ games)
- Avg Score: _____
- Max Tile: _____
- Skills Enabled: Yes/No
- Model: GPT-5.2 / GPT-5.2-Chat
- Notes: ___________________________
```

## Support Resources

- **Full Spec**: `spec-gpt52-skills.md`
- **User Guide**: `README-GPT52-SKILLS.md`
- **Implementation**: `IMPLEMENTATION-SUMMARY.md`
- **Game Spec**: `spec.md`
- **Paper**: `paper_llm_focus.md`

## Ready? Let's Go! 🎮

```bash
1. Open index.html
2. Select LLM + GPT-5.2
3. Enable 🧠 Skills
4. Click "AI Play"
5. Watch the magic! ✨
```

**Good luck and have fun testing GPT-5.2 skills!** 🚀🧠

---

*Questions? Check the console (F12), review docs, or inspect the code in index.html*
