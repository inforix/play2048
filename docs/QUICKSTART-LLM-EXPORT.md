# Quick Start: LLM Dataset Export

**Create training data for fine-tuning your own 2048 AI!** 🧠

## Overview

The **LLM Dataset Export** feature converts your game collection into training data suitable for fine-tuning Large Language Models (LLMs) like GPT-3.5-turbo, GPT-4, or custom models.

### What You Get
- 📄 **Training Dataset** (JSONL format) - One training example per move
- 📊 **Metadata File** (JSON) - Statistics, instructions, and dataset info
- ✅ **OpenAI Compatible** - Ready for fine-tuning API
- ☁️ **Azure Compatible** - Works with Azure OpenAI Studio

## Quick Start (5 Minutes)

### Step 1: Build Your Game Collection

Play at least 10-20 games to build a quality dataset:

1. Open `index.html` in your browser
2. Play games until completion (reach 2048 or game over)
3. Games are **automatically saved** to collection
4. Aim for **high scores** (>1000) for best training quality

**Tip**: Use "AI Play" with Expectimax strategy to quickly generate high-quality games!

### Step 2: Export Dataset

In the **Game Collection** panel:

1. Verify you have games saved (check statistics)
2. Click **"🧠 Export LLM Dataset"** button
3. Two files download automatically:
   - `2048-training-dataset-YYYY-MM-DD.jsonl`
   - `2048-dataset-metadata-YYYY-MM-DD.json`

### Step 3: Review Your Dataset

Open the metadata file to see:
- Total training examples created
- Games used (only score > 1000)
- Dataset statistics
- Usage instructions

### Step 4: Fine-Tune Your Model

#### Option A: OpenAI API

```bash
# Install OpenAI CLI
pip install openai

# Upload and fine-tune
openai api fine_tunes.create \
  -t 2048-training-dataset-2026-01-06.jsonl \
  -m gpt-3.5-turbo \
  --suffix "2048-player"

# Check status
openai api fine_tunes.list
```

#### Option B: Azure OpenAI Studio

1. Log into [Azure OpenAI Studio](https://oai.azure.com/)
2. Navigate to **Fine-tuning** → **Create custom model**
3. Upload your `.jsonl` file
4. Configure training parameters
5. Start fine-tuning job
6. Deploy when complete

#### Option C: Other Platforms

The JSONL format works with:
- Anthropic Claude fine-tuning
- Google PaLM 2 fine-tuning
- Custom transformers training
- Any platform accepting chat completion format

## Dataset Format

### Training Example Structure

Each line in the JSONL file contains one training example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert 2048 game AI. Analyze the board state and suggest the best move. Respond with only one word: up, down, left, or right."
    },
    {
      "role": "user",
      "content": "Current board state (4x4 grid, . = empty):\n    .    .    2    4\n    .    .    8   16\n    2    4   32   64\n    4    8  128  256\n\nWhat move should I make?"
    },
    {
      "role": "assistant",
      "content": "up"
    }
  ]
}
```

### Metadata File Structure

```json
{
  "dataset_info": {
    "format": "OpenAI fine-tuning JSONL",
    "created": "2026-01-06T12:00:00.000Z",
    "total_examples": 2450,
    "games_used": 15,
    "total_games_in_collection": 20
  },
  "statistics": {
    "avg_score": 18500,
    "best_score": 45000,
    "total_moves": 2450,
    "win_rate": "75.0%"
  },
  "usage_instructions": {
    "openai": "Use this JSONL file with OpenAI's fine-tuning API",
    "azure": "Upload to Azure OpenAI Studio for custom model training",
    "format": "Each line is a complete training example"
  }
}
```

## Quality Filters

The export automatically filters for quality:

✅ **Included**:
- Games with final score > 1000
- All moves from qualifying games
- Valid board states

❌ **Excluded**:
- Very short games (< 5 moves)
- Low-scoring games (< 1000)
- Test/practice games

## Best Practices

### Building a Quality Dataset

1. **Play Strategically**: Focus on high-scoring games
2. **Use Winning Strategies**: Anchor tiles, avoid DOWN moves
3. **Diverse Scenarios**: Include early, mid, and late game states
4. **Sufficient Volume**: Aim for 2000+ training examples (20+ games)

### Recommended Game Count

| Purpose | Games Needed | Examples |
|---------|--------------|----------|
| Testing | 10-15 | 1000-1500 |
| Good Model | 20-30 | 2000-3000 |
| Optimal Model | 40-50 | 4000-5000 |

### Improving Dataset Quality

**Strategy 1: Generate with AI**
```
1. Select "Expectimax" strategy
2. Click "AI Play"
3. Let it play 20+ games
4. Export high-quality dataset
```

**Strategy 2: Learn and Apply**
```
1. Play 5-10 games manually
2. Click "📚 Learn from All Games"
3. Switch to LLM strategy (applies learned patterns)
4. Play 20+ more games
5. Export improved dataset
```

**Strategy 3: Import Expert Games**
```
1. Download expert game histories
2. Click "📥 Import JSON" 
3. Load multiple JSON files
4. Export combined dataset
```

## Using Your Fine-Tuned Model

After fine-tuning completes:

### OpenAI API
```python
import openai

response = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo:your-org:2048-player:abc123",
  messages=[
    {"role": "system", "content": "You are an expert 2048 game AI..."},
    {"role": "user", "content": "Current board state:\n..."}
  ]
)

move = response.choices[0].message.content
print(f"Suggested move: {move}")
```

### Azure OpenAI
```python
import openai

openai.api_type = "azure"
openai.api_base = "https://your-resource.openai.azure.com"
openai.api_key = "your-api-key"
openai.api_version = "2024-02-15-preview"

response = openai.ChatCompletion.create(
  deployment_id="your-fine-tuned-deployment",
  messages=[...]
)
```

### Integrate Back into Game

Modify `index.html` to add your custom model:

```javascript
const aiModels = {
  // ... existing models ...
  'my-custom-model': {
    endpoint: 'https://your-resource.openai.azure.com',
    deployment: 'your-fine-tuned-deployment',
    apiVersion: '2024-02-15-preview',
    type: 'azure'
  }
};
```

Then select it from the dropdown and play!

## Troubleshooting

### No Export Button Visible
**Solution**: The button is in the Game Collection panel. Scroll down to see it.

### "No games in collection" Error
**Solution**: Play some games first. Games are auto-saved when you reach game over.

### "No suitable training examples" Error
**Solution**: Your games need score > 1000. Play better games or use AI Play with Expectimax.

### File Not Downloading
**Solution**: 
- Check browser download settings
- Allow pop-ups from the page
- Try a different browser

### JSONL Format Issues
**Solution**: 
- Verify file ends in `.jsonl` not `.json`
- Each line should be a complete JSON object
- No comma between lines
- Validate with: `jq -c . file.jsonl`

### Fine-Tuning Fails
**Common Issues**:
- Insufficient examples (need 10+ minimum)
- Invalid JSON format
- Missing required fields
- File encoding issues (use UTF-8)

## Advanced Usage

### Combine Multiple Datasets

```bash
# Merge multiple exports
cat dataset1.jsonl dataset2.jsonl > combined.jsonl

# Remove duplicates and shuffle
sort -u combined.jsonl | shuf > final-dataset.jsonl
```

### Split Train/Validation

```bash
# Create 90/10 split
total=$(wc -l < dataset.jsonl)
train=$((total * 9 / 10))

head -n $train dataset.jsonl > train.jsonl
tail -n +$((train + 1)) dataset.jsonl > validation.jsonl
```

### Augment Dataset

Enhance your dataset by:
1. Rotating board states (4 rotations per game)
2. Mirroring board states (2 mirrors per game)
3. This 8× multiplies your training examples!

### Filter by Performance

```bash
# Extract only high-scoring games
# Add scoring logic to filter JSONL by game performance
```

## Performance Expectations

### Before Fine-Tuning
- **Generic GPT-3.5**: ~30-40% win rate
- **Generic GPT-4**: ~40-50% win rate

### After Fine-Tuning (20+ games)
- **Fine-tuned GPT-3.5**: ~60-75% win rate
- **Fine-tuned GPT-4**: ~75-85% win rate

### Cost Analysis

| Model | Training Cost | Inference Cost | Best For |
|-------|--------------|----------------|----------|
| GPT-3.5-turbo | ~$1-3 | $0.002/move | Budget, speed |
| GPT-4 | ~$10-30 | $0.01/move | Best accuracy |
| Custom (open-source) | Variable | Free | Research |

## Example Workflow

### Complete End-to-End Process

```bash
# 1. Generate training data
# (Open index.html, click AI Play, let it run 20 games)

# 2. Export dataset
# (Click "🧠 Export LLM Dataset")

# 3. Fine-tune model
openai api fine_tunes.create \
  -t 2048-training-dataset-2026-01-06.jsonl \
  -m gpt-3.5-turbo \
  --n_epochs 3 \
  --batch_size 16 \
  --learning_rate_multiplier 0.1

# 4. Wait for completion (monitor with)
openai api fine_tunes.follow -i ft-abc123

# 5. Test your model
openai api chat.completions.create \
  -m ft:gpt-3.5-turbo:your-org:2048:abc123 \
  -g system "You are a 2048 expert AI" \
  -g user "Board:\n0 0 2 4\n..."

# 6. Deploy and integrate
# (Update index.html with your model details)
```

## Resources

### Documentation
- OpenAI Fine-tuning Guide: https://platform.openai.com/docs/guides/fine-tuning
- Azure OpenAI Fine-tuning: https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning
- JSONL Format Spec: https://jsonlines.org/

### Tools
- JSONL Validator: https://jsonlint.com/
- OpenAI CLI: `pip install openai`
- jq (JSON processor): https://stedolan.github.io/jq/

### Related Features
- Game Collection: Auto-saves completed games
- Import JSON: Load external game histories  
- Learn from All Games: Extract winning patterns
- AI Strategies: Expectimax, Monte Carlo, LLM

## Support

**Questions or Issues?**
1. Check browser console (F12) for errors
2. Verify game collection has games
3. Ensure games have score > 1000
4. Review metadata file for dataset info
5. Validate JSONL format with `jq`

**Feature Requests?**
- Add more export formats (CSV, Parquet, HuggingFace)
- Include board evaluations in training data
- Support for different prompt styles
- Augmentation options

---

**Ready to create your own 2048-playing AI!** 🚀

Play games → Export dataset → Fine-tune model → Deploy → Win! 🏆
