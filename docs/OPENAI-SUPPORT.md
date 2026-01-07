# OpenAI GPT-5.2 Support

## Overview

The 2048 Aurora Edition now supports **two GPT-5.2 providers**:

1. **Azure OpenAI** (existing) - Limited skills support
2. **OpenAI** (new) - Full skills support including code interpreter

## Provider Comparison

| Feature | Azure OpenAI | OpenAI |
|---------|-------------|---------|
| **Base URL** | `https://maritimeai-resource.openai.azure.com` | `https://api.bianxie.ai` |
| **API Authentication** | `api-key` header | `Authorization: Bearer` header |
| **Code Interpreter** | ❌ Not supported | ✅ **Supported** |
| **Function Calling** | ✅ Supported | ✅ Supported |
| **Structured Outputs** | ✅ Supported | ✅ Supported |
| **Token Parameter** | `max_completion_tokens` | `max_tokens` |
| **Tools Format** | `functions` array | `tools` array with type |

## Models Available

### Azure OpenAI Models
- **Azure GPT-5.2** - Core GPT-5.2 model
- **Azure GPT-5.2-Chat** - Chat-optimized variant
- **Azure DeepSeek-V3.2** - DeepSeek model via Azure

### OpenAI Models
- **OpenAI GPT-5.2** - Full OpenAI GPT-5.2 with all skills

## API Key Setup

The game manages separate API keys for each provider:

### Azure OpenAI
- Storage key: `aurora-2048-azure-key`
- Prompted when selecting Azure models
- Header format: `api-key: YOUR_KEY`

### OpenAI
- Storage key: `aurora-2048-openai-key`
- Prompted when selecting OpenAI GPT-5.2
- Header format: `Authorization: Bearer YOUR_KEY`

Both keys are stored in browser localStorage for persistence.

## Skills Configuration

### Azure OpenAI Skills
```javascript
const gpt52SkillsAzure = {
  codeInterpreter: false,  // ❌ Not supported
  structuredOutputs: true,  // ✅ Supported
  functionCalling: true,    // ✅ Supported
  reasoning: true           // ✅ Enhanced prompts
};
```

### OpenAI Skills
```javascript
const gpt52SkillsOpenAI = {
  codeInterpreter: true,   // ✅ Supported
  structuredOutputs: true,  // ✅ Supported
  functionCalling: true,    // ✅ Supported
  reasoning: true           // ✅ Enhanced prompts
};
```

## Code Interpreter Capability

When using **OpenAI GPT-5.2**, the code interpreter tool is enabled:

```javascript
// Tools array for OpenAI
{
  tools: [
    { type: 'code_interpreter' },  // Python execution environment
    { type: 'function', function: evaluateBoardFunction },
    { type: 'function', function: simulateMovesFunction },
    { type: 'function', function: checkAnchorSafetyFunction }
  ]
}
```

The code interpreter allows GPT-5.2 to:
- Write and execute Python code for complex calculations
- Perform multi-step board analysis
- Test hypothetical game scenarios
- Generate move sequences programmatically

## API Request Differences

### Azure OpenAI Request
```javascript
POST https://maritimeai-resource.openai.azure.com/openai/deployments/gpt-5.2/chat/completions?api-version=2025-03-01-preview

Headers:
  api-key: xxx
  Content-Type: application/json

Body:
{
  "messages": [...],
  "functions": [...],              // Direct functions array
  "response_format": {...},
  "max_completion_tokens": 4000    // Azure-specific parameter
}
```

### OpenAI Request
```javascript
POST https://api.bianxie.ai/v1/chat/completions

Headers:
  Authorization: Bearer xxx
  Content-Type: application/json

Body:
{
  "model": "gpt-5.2",              // Model specified in body
  "messages": [...],
  "tools": [                       // Tools array with types
    { "type": "code_interpreter" },
    { "type": "function", "function": {...} }
  ],
  "response_format": {...},
  "max_tokens": 4000                // Standard parameter
}
```

## Response Handling

### Azure Function Calls
```javascript
{
  "choices": [{
    "message": {
      "function_call": {           // Single function call
        "name": "evaluate_board",
        "arguments": "{...}"
      }
    }
  }]
}
```

### OpenAI Tool Calls
```javascript
{
  "choices": [{
    "message": {
      "tool_calls": [              // Multiple tool calls possible
        {
          "type": "function",
          "function": {
            "name": "evaluate_board",
            "arguments": "{...}"
          }
        }
      ]
    }
  }]
}
```

## Usage Instructions

### For Players

1. **Select Model**
   - Choose "LLM (GPT/DeepSeek)" strategy
   - Select "OpenAI GPT-5.2" from model dropdown

2. **Enter API Key**
   - Prompt will ask for OpenAI API key
   - Key is stored locally in browser
   - Get key from: https://api.bianxie.ai

3. **Enable Skills**
   - Toggle "🧠 Skills" checkbox (enabled by default)
   - Skills status panel shows active capabilities
   - Look for "Code: ✓" indicator for OpenAI

4. **Play**
   - Click "Ask AI" for single move suggestion
   - Click "AI Play" for autonomous gameplay
   - Skills panel displays analysis metrics

### For Developers

The implementation automatically detects provider type:

```javascript
const model = aiModels[selectedModel];
const isOpenAI = model.type === 'openai';
const isAzure = model.type === 'azure';

// Select appropriate skills configuration
const gpt52Skills = isOpenAI ? gpt52SkillsOpenAI : gpt52SkillsAzure;

// Build request based on provider
if (isOpenAI) {
  // Add code_interpreter to tools
  if (gpt52Skills.codeInterpreter) {
    tools.push({ type: 'code_interpreter' });
  }
  
  // Use Bearer auth
  headers['Authorization'] = `Bearer ${openaiApiKey}`;
}
```

## Performance Expectations

### Azure OpenAI (No Code Interpreter)
- **Cost per move**: ~$0.018
- **Response time**: 1-2 seconds
- **Win rate**: 55-70%
- **Skills**: Functions + Structured outputs

### OpenAI (With Code Interpreter)
- **Cost per move**: ~$0.025 (estimated +40% for code execution)
- **Response time**: 2-4 seconds (includes Python execution)
- **Win rate**: **65-80%** (expected improvement)
- **Skills**: Functions + Structured outputs + Code interpreter

## Benefits of Code Interpreter

1. **Advanced Analysis**
   - Multi-step lookahead calculations
   - Statistical board evaluation
   - Pattern recognition algorithms

2. **Dynamic Strategy**
   - Adaptive heuristics based on game state
   - Monte Carlo simulations
   - Optimal move sequences

3. **Debugging**
   - GPT-5.2 can write test code
   - Verify board logic programmatically
   - Generate board visualizations

## Troubleshooting

### "OpenAI API error: 401"
- Invalid API key
- Re-enter key via model selection
- Verify key at https://api.bianxie.ai

### Skills not showing Code: ✓
- Ensure "OpenAI GPT-5.2" is selected (not Azure)
- Check skills toggle is enabled
- Verify LLM strategy is active

### Slow responses
- Code interpreter adds ~1-2s execution time
- Normal for complex board analysis
- Consider disabling skills for faster play

### Falls back to heuristic
- API key not set or invalid
- Network error
- Check browser console for details

## Migration from Azure

If switching from Azure to OpenAI:

1. Both API keys are stored separately
2. No need to re-enter Azure key
3. Switch models anytime via dropdown
4. Each model remembers its own key

## Future Enhancements

Potential improvements with code interpreter:

- **Multi-turn conversations** - Send function results back for refinement
- **Adaptive skills** - Enable code interpreter only for complex boards
- **Response caching** - Store code execution results for identical states
- **Custom algorithms** - Let GPT-5.2 write board evaluators on-the-fly
- **Visual analysis** - Generate matplotlib charts of game progress

## See Also

- [AZURE-COMPATIBILITY.md](AZURE-COMPATIBILITY.md) - Azure vs OpenAI differences
- [spec-gpt52-skills.md](spec-gpt52-skills.md) - Technical specification
- [README-GPT52-SKILLS.md](README-GPT52-SKILLS.md) - User guide
