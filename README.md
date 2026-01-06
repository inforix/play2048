# 2048 Aurora Edition 🎮✨

A beautiful, AI-powered implementation of the classic 2048 game with advanced features for gameplay, learning, and LLM fine-tuning.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Status](https://img.shields.io/badge/status-production-green)
![AI](https://img.shields.io/badge/AI-GPT--5.2%20Ready-purple)

## Features

### 🎨 Beautiful UI
- **Aurora & Dawn Themes** - Stunning gradient backgrounds with glassmorphic design
- **Responsive Grid** - Adapts to mobile and desktop
- **Smooth Animations** - Polished tile movements and transitions
- **Sound Effects** - Audio feedback for moves and merges

### 🤖 Advanced AI Strategies
1. **Expectimax** (80% win rate) - 4-ply search with comprehensive evaluation
2. **Monte Carlo** - MCTS with greedy rollouts (100 simulations)
3. **Weighted Heuristic** - Minimax with snake pattern optimization
4. **LLM** - Cloud AI powered by GPT-5.2, DeepSeek, or custom models

### 🧠 GPT-5.2 Skills Enhancement
- **Function Calling** - Game-specific analysis functions
- **Structured Outputs** - Reliable JSON responses with reasoning
- **Code Interpreter** - Python execution for advanced calculations (OpenAI only)
- **Enhanced Prompts** - Strategic decision-making with learned patterns

### 📚 Learning System
- **Single-Game Learning** - AI analyzes your gameplay to extract patterns
- **Multi-Game Learning** - Compares winning vs losing strategies across collection
- **Strategy Application** - Learned patterns applied to LLM decision-making
- **Game Collection** - Auto-saves completed games (max 50)

### 🧪 LLM Training Dataset Export ⭐ NEW
- **Export to JSONL** - Fine-tuning format for OpenAI/Azure
- **Quality Filtering** - Only exports games with score > 1000
- **Metadata Included** - Statistics and usage instructions
- **Use Cases**: Create custom AI models that learn from your gameplay

### 📊 Move History & Analysis
- **Visual History** - Mini board previews for each move
- **Export/Import** - Download/upload game histories as JSON
- **Revert to State** - Click any move to restore that position
- **Copy to Clipboard** - Share individual moves or entire games

## Quick Start

### Play Locally
```bash
# Clone or download the repository
cd play2048

# Open in browser (Mac)
open index.html

# Or (Linux/Windows)
# Just double-click index.html
```

### Configure AI
1. Click "Ask AI" (you'll be prompted for API key on first use)
2. Enter your **Azure OpenAI** or **OpenAI API** key
3. Key is stored locally in your browser
4. Select model: GPT-5.2, GPT-5.2-Chat, or DeepSeek-V3.2

### Export Training Data
1. Play 20+ games (or use "AI Play" to auto-generate)
2. Scroll to **Game Collection** panel
3. Click **"🧠 Export LLM Dataset"**
4. Download JSONL file + metadata
5. Use for fine-tuning GPT-3.5/GPT-4

## Documentation

### Getting Started
- **[spec.md](spec.md)** - Complete game specification and features
- **[QUICKSTART-SKILLS.md](QUICKSTART-SKILLS.md)** - Test GPT-5.2 skills in 5 minutes
- **[QUICKSTART-LLM-EXPORT.md](QUICKSTART-LLM-EXPORT.md)** - Export and fine-tune guide

### Features & Guides
- **[README-GPT52-SKILLS.md](README-GPT52-SKILLS.md)** - GPT-5.2 skills user guide
- **[FEATURE-LLM-EXPORT.md](FEATURE-LLM-EXPORT.md)** - Dataset export deep dive
- **[OPENAI-SUPPORT.md](OPENAI-SUPPORT.md)** - Azure vs OpenAI comparison

### Technical Documentation
- **[spec-gpt52-skills.md](spec-gpt52-skills.md)** - Technical specification (650+ lines)
- **[IMPLEMENTATION-SUMMARY.md](IMPLEMENTATION-SUMMARY.md)** - Code overview
- **[AZURE-COMPATIBILITY.md](AZURE-COMPATIBILITY.md)** - Azure OpenAI limitations
- **[CHANGELOG-OPENAI.md](CHANGELOG-OPENAI.md)** - Version history

### Research
- **[paper_llm_focus.md](paper_llm_focus.md)** - Academic paper on LLM strategies
- **[bench/](bench/)** - Benchmarking tools and results

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   index.html                        │
│  ┌──────────────────────────────────────────────┐  │
│  │         Game Engine (Pure JavaScript)        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │      AI Strategies                           │  │
│  │  • Expectimax (4-ply)                        │  │
│  │  • Monte Carlo (100 sims)                    │  │
│  │  • Weighted Heuristic                        │  │
│  │  • LLM (Azure/OpenAI)                        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │      GPT-5.2 Skills                          │  │
│  │  • Function Calling                          │  │
│  │  • Structured Outputs                        │  │
│  │  • Code Interpreter (OpenAI)                 │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │      Learning System                         │  │
│  │  • Strategy Analysis                         │  │
│  │  • Multi-Game Learning                       │  │
│  │  • Pattern Extraction                        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │      Dataset Export                          │  │
│  │  • JSONL Generation                          │  │
│  │  • Quality Filtering                         │  │
│  │  • Metadata Creation                         │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Key Technologies

- **Pure JavaScript** - No frameworks or build tools required
- **LocalStorage** - Persistent game state and settings
- **Web Audio API** - Dynamic sound generation
- **Fetch API** - Azure/OpenAI integration
- **CSS Variables** - Theme system
- **ES6+** - Modern JavaScript features

## Performance Benchmarks

### AI Win Rates (reaching 2048)
| Strategy | Win Rate | Avg Score | Speed |
|----------|----------|-----------|-------|
| Expectimax | 75-85% | 20-25K | Instant |
| Monte Carlo | 65-75% | 18-22K | Instant |
| Weighted | 60-70% | 15-20K | Instant |
| LLM (basic) | 40-50% | 12-18K | 1-2s/move |
| LLM + Skills | 60-75% | 18-25K | 2-4s/move |
| LLM + Fine-tuned | 75-85% | 22-30K | 2-3s/move |

### Cost Analysis
| Model | Cost/Move | 100 Games | Notes |
|-------|-----------|-----------|-------|
| Local (Expectimax) | Free | Free | Best performance |
| Azure GPT-5.2 | $0.005 | ~$1 | Basic prompts |
| Azure + Skills | $0.018 | ~$4 | Functions + Structured |
| OpenAI + Skills | $0.025 | ~$5 | + Code interpreter |
| Fine-tuned GPT-3.5 | $0.002 | ~$0.40 | After training |

## Use Cases

### 1. Playing for Fun
- Beautiful themes and smooth gameplay
- Undo support for learning
- Sound effects and animations
- Mobile-friendly interface

### 2. AI Experimentation
- Compare different algorithms
- Test LLM capabilities
- Benchmark strategies
- Learn AI decision-making

### 3. Research & Education
- Study game-playing AI
- Demonstrate fine-tuning
- Teach prompt engineering
- Analyze learning patterns

### 4. Custom Model Training
- Export gameplay datasets
- Fine-tune personal models
- Transfer playing styles
- Create competitive AI

## File Structure

```
play2048/
├── index.html                     # Main game (single file!)
├── README.md                      # This file
├── spec.md                        # Game specification
│
├── Documentation/
│   ├── QUICKSTART-SKILLS.md       # 5-min GPT-5.2 guide
│   ├── QUICKSTART-LLM-EXPORT.md   # Export & fine-tune guide
│   ├── README-GPT52-SKILLS.md     # Skills user manual
│   ├── FEATURE-LLM-EXPORT.md      # Export feature overview
│   ├── OPENAI-SUPPORT.md          # Provider comparison
│   ├── AZURE-COMPATIBILITY.md     # Azure limitations
│   └── CHANGELOG-OPENAI.md        # Version history
│
├── Technical/
│   ├── spec-gpt52-skills.md       # Technical spec (650+ lines)
│   ├── IMPLEMENTATION-SUMMARY.md  # Code overview
│   └── paper_llm_focus.md         # Research paper
│
├── Benchmarks/
│   ├── bench/bench2048.js         # Benchmark script
│   ├── bench/results.json         # Test results
│   └── bench/results_summary.md   # Analysis
│
└── Replay/
    └── replay/*.json              # Saved game histories
```

## Development

### Prerequisites
None! Just a modern web browser.

### Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd play2048

# Open in browser
open index.html

# That's it! No build step needed.
```

### Customization

#### Add Custom AI Strategy
```javascript
// In index.html, find the strategies section
function myCustomStrategy(boardState) {
  // Your algorithm here
  return 'up'; // or 'down', 'left', 'right'
}

// Add to strategy selector
selectedStrategy = 'mycustom';
```

#### Modify Export Format
```javascript
// In exportLLMDataset() function
const example = {
  messages: [
    // Customize system/user/assistant messages
  ]
};
```

#### Change Quality Threshold
```javascript
// Default: score > 1000
if (game.finalScore < 1000) continue;

// Change to:
if (game.finalScore < 2000) continue;
```

## Troubleshooting

### Common Issues

**AI not working**
- Ensure API key is entered correctly
- Check browser console (F12) for errors
- Verify endpoint URLs are correct
- Try fallback (Expectimax strategy)

**Export button not visible**
- Scroll to Game Collection panel
- Ensure games are in collection
- Check browser compatibility

**Fine-tuning fails**
- Validate JSONL format: `jq -c . file.jsonl`
- Check minimum examples (10+)
- Verify file encoding (UTF-8)
- Review OpenAI/Azure documentation

### Getting Help

1. Check documentation files
2. Review browser console (F12)
3. Verify API credentials
4. Test with local strategies first
5. Check [QUICKSTART-*.md] guides

## Contributing

This is a single-file application for simplicity. To contribute:

1. Fork the repository
2. Modify `index.html`
3. Test thoroughly in multiple browsers
4. Update relevant documentation
5. Submit pull request with description

## Future Roadmap

- [ ] Multi-player mode (compete with friends)
- [ ] Achievement system
- [ ] More export formats (CSV, Parquet)
- [ ] Dataset augmentation (rotations, mirrors)
- [ ] Direct fine-tuning API integration
- [ ] Real-time multiplayer
- [ ] Tournament mode
- [ ] Leaderboards

## Credits

- **Original 2048**: Gabriele Cirulli
- **Aurora Theme**: Custom design
- **AI Strategies**: Multiple algorithms implementation
- **GPT-5.2 Integration**: Azure OpenAI + OpenAI support
- **Fine-tuning Export**: Custom implementation

## License

This project is available for educational and research purposes.

## Links

- **Live Demo**: [Open index.html in browser]
- **Documentation**: See files above
- **Research Paper**: [paper_llm_focus.md](paper_llm_focus.md)
- **Benchmarks**: [bench/results_summary.md](bench/results_summary.md)

---

**Ready to play, learn, and create your own AI!** 🚀🧠

Start with: `open index.html` → Play games → Export dataset → Fine-tune model → Win! 🏆
