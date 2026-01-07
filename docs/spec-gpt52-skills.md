# GPT-5.2 Skills Integration Specification

**Project**: Aurora 2048  
**Version**: 1.0  
**Date**: January 6, 2026  
**Status**: Implementation Phase

## 1. Overview

This specification defines the integration of Azure OpenAI GPT-5.2 advanced skills into the Aurora 2048 game to improve AI decision-making quality and win rates when users select LLM-based strategies.

**Important**: This implementation is designed for **Azure OpenAI**, which has different capabilities than OpenAI's API. Azure OpenAI supports **Function Calling** and **Structured Outputs**, but does NOT support **Code Interpreter** (Python execution).

### 1.1 Objectives

- Leverage GPT-5.2's Function Calling and Structured Outputs capabilities (Azure-compatible)
- Increase LLM win rate from ~40-50% to 55-70%
- Provide transparent reasoning for AI decisions
- Maintain backward compatibility with existing AI strategies

### 1.2 Scope

**In Scope**:
- Code Interpreter integration for move evaluation
- Structured JSON output schema
- Function calling for board analysis
- Enhanced reasoning prompts
- UI toggle for skills enablement
- Cost and performance optimization

**Out of Scope**:
- Fine-tuning custom models
- Real-time learning during gameplay
- Multiplayer AI battles

## 2. Technical Architecture

### 2.1 Skills Components

#### 2.1.1 Function Calling (Azure OpenAI Supported) ✅

**Purpose**: Allow GPT-5.2 to invoke game-specific functions for validated board analysis

**Implementation**:
```javascript
functions: [
  {
    name: "evaluate_board",
    description: "...",
    parameters: { ... }
  },
  // ... more functions
]
```

**Supported by Azure OpenAI**: Yes

**Use Cases**:
- Query current board metrics (monotonicity, smoothness, empty cells)
- Simulate possible moves and get outcomes
- Check if moves are safe (anchor guard)
- Get validated heuristic scores

#### 2.1.2 Structured Outputs (Azure OpenAI Supported) ✅

**Purpose**: Ensure reliable, parsable JSON responses

**Implementation**:
```javascript
response_format: {
  type: "json_schema",
  json_schema: { ... }
}
```

**Supported by Azure OpenAI**: Yes

**Schema Definition**:
```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "move_decision",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "reasoning": {
          "type": "string",
          "description": "Step-by-step analysis of current board state"
        },
        "board_analysis": {
          "type": "object",
          "properties": {
            "max_tile": { "type": "number" },
            "max_tile_position": { "type": "string" },
            "empty_cells": { "type": "number" },
            "monotonicity_score": { "type": "number" },
            "smoothness_score": { "type": "number" },
            "anchor_safe": { "type": "boolean" }
          },
          "required": ["max_tile", "empty_cells", "anchor_safe"]
        },
        "move_evaluations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "move": { 
                "type": "string",
                "enum": ["up", "down", "left", "right"]
              },
              "score": { "type": "number" },
              "is_valid": { "type": "boolean" },
              "reason": { "type": "string" }
            },
            "required": ["move", "score", "is_valid"]
          }
        },
        "recommended_move": {
          "type": "string",
          "enum": ["up", "down", "left", "right"]
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      },
      "required": ["recommended_move", "confidence", "board_analysis"],
      "additionalProperties": false
    }
  }
}
```

#### 2.1.3 Enhanced Reasoning Prompts

**Purpose**: Improve decision quality through better prompt engineering

**Not a Skill**: This is prompt optimization, not an API feature

**Implementation**:
- Sophisticated system messages guiding strategic thinking
- Detailed board state descriptions
- Request step-by-step analysis
- Include learned strategies and game theory principles

### 2.2 Azure OpenAI Limitations

**What's NOT Supported**:
- ❌ **Code Interpreter** (`tools: [{"type": "code_interpreter"}]`)
  - This is OpenAI-only, not available in Azure OpenAI
  - Would allow Python code execution for calculations
  - Workaround: Use function calling to provide calculated data

**What IS Supported**:
- ✅ **Function Calling** - Invoke predefined functions
- ✅ **Structured Outputs** - JSON schema validation
- ✅ **Enhanced Prompts** - Better instructions and reasoning

### 2.3 Azure OpenAI API Specifics

**Parameter Differences from OpenAI**:

1. **Token Limits**: Use `max_completion_tokens` instead of `max_tokens`
   ```javascript
   // Azure OpenAI (correct)
   body.max_completion_tokens = 4000;
   
   // OpenAI API (incorrect for Azure)
   // body.max_tokens = 4000;  // Error!
   ```

2. **Strict Schema Mode**: Azure requires stricter JSON schemas
   - EVERY property must be in `required` array (no optional fields)
   - EVERY object must have `additionalProperties: false`
   - Nested objects have same requirements
   - Arrays of objects: each item schema needs full required list

3. **Supported Tools**: Only `function` type, not `code_interpreter`
   ```javascript
   // Supported
   body.functions = [...];
   
   // NOT supported
   // body.tools = [{type: "code_interpreter"}];  // Error!
   ```

### 2.4 System Architecture

1. **evaluate_board**
```json
{
  "name": "evaluate_board",
  "description": "Evaluate board state using multiple heuristics including monotonicity, smoothness, empty cells, and corner anchoring. Returns a composite score.",
  "parameters": {
    "type": "object",
    "properties": {
      "board": {
        "type": "array",
        "items": {
          "type": "array",
          "items": { "type": "number" }
        },
        "description": "4x4 board as 2D array"
      }
    },
    "required": ["board"]
  }
}
```

2. **simulate_moves**
```json
{
  "name": "simulate_moves",
  "description": "Simulate all 4 possible moves (up, down, left, right) and return outcomes including new board states, score gains, and whether merges occurred.",
  "parameters": {
    "type": "object",
    "properties": {
      "board": {
        "type": "array",
        "description": "Current 4x4 board state"
      }
    },
    "required": ["board"]
  }
}
```

3. **check_anchor_safety**
```json
{
  "name": "check_anchor_safety",
  "description": "Check if a proposed move would displace the maximum tile from the top-right corner (anchor position).",
  "parameters": {
    "type": "object",
    "properties": {
      "board": { "type": "array" },
      "move": {
        "type": "string",
        "enum": ["up", "down", "left", "right"]
      }
    },
    "required": ["board", "move"]
  }
}
```

### 2.2 System Architecture

```
User Input
    ↓
[AI Strategy Selector]
    ↓
Is GPT-5.2 + Skills Enabled?
    ↓ Yes                    ↓ No
[Skills-Enhanced Path]   [Basic Prompt Path]
    ↓
1. Build enhanced prompt with:
   - Code interpreter tool
   - Function definitions
   - Structured output schema
   - Reasoning instructions
    ↓
2. Call Azure OpenAI API
    ↓
3. GPT-5.2 may:
   - Execute Python code
   - Call game functions
   - Return structured JSON
    ↓
4. Parse response
    ↓
5. Extract recommended move
    ↓
6. Apply anchor guard
    ↓
7. Execute move
```

### 2.3 Enhanced Prompt Structure

```
System Message:
- Role: Expert 2048 AI strategist with access to code execution and analysis tools
- Capabilities: Can write Python code, call game functions, analyze board states
- Output format: Must return structured JSON with reasoning

User Message:
- Current board state (4x4 grid)
- Move analysis (which moves are valid)
- Anchor status
- Learned strategy context
- Request: Analyze and recommend best move using code interpreter and function calls

Tools Available:
- Code Interpreter (Python execution)
- evaluate_board() function
- simulate_moves() function
- check_anchor_safety() function

Expected Response:
- Structured JSON matching schema
- Reasoning chain
- Board analysis metrics
- Move evaluations with scores
- Final recommendation with confidence
```

## 3. Implementation Specifications

### 3.1 Code Changes

**File**: `index.html`

**3.1.1 Add Skills Configuration**

Location: After `aiModels` definition

```javascript
// GPT-5.2 Skills Configuration
const gpt52Skills = {
  codeInterpreter: true,
  structuredOutputs: true,
  functionCalling: true,
  reasoning: true
};

let skillsEnabled = true; // Default: enabled for GPT-5.2
```

**3.1.2 Add Structured Output Schema**

Location: After skills configuration

```javascript
const moveDecisionSchema = {
  type: "json_schema",
  json_schema: {
    name: "move_decision",
    strict: true,
    schema: { /* ... full schema ... */ }
  }
};
```

**3.1.3 Add Function Definitions**

```javascript
const gameFunctions = [
  { /* evaluate_board */ },
  { /* simulate_moves */ },
  { /* check_anchor_safety */ }
];
```

**3.1.4 Implement Function Handlers**

```javascript
function handleFunctionCall(functionName, args) {
  switch(functionName) {
    case 'evaluate_board':
      return evaluateBoardFunction(args.board);
    case 'simulate_moves':
      return simulateMovesFunction(args.board);
    case 'check_anchor_safety':
      return checkAnchorSafetyFunction(args.board, args.move);
  }
}
```

**3.1.5 Update fetchAzureMove()**

Changes:
- Detect GPT-5.2 model
- Add tools array with code_interpreter
- Add functions array
- Add response_format for structured output
- Handle function call responses
- Parse structured JSON
- Enhanced error handling

**3.1.6 Add UI Toggle**

Location: Controls section

```html
<div style="display: flex; align-items: center; gap: 8px;" id="skills-toggle-container">
  <label style="display: flex; align-items: center; gap: 6px; font-size: 14px;">
    <input type="checkbox" id="enable-skills" checked>
    <span>🧠 GPT-5.2 Skills</span>
  </label>
</div>
```

**3.1.7 Skills Status Display**

```html
<div class="info" id="skills-status" style="display: none;">
  <strong>🧠 Skills Active:</strong>
  <span id="skills-active-list"></span>
</div>
```

### 3.2 API Request Format (GPT-5.2 with Skills)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert 2048 AI strategist with access to code execution and analysis tools..."
    },
    {
      "role": "user",
      "content": "BOARD STATE:\n...\n\nAnalyze and recommend the best move."
    }
  ],
  "tools": [
    { "type": "code_interpreter" }
  ],
  "functions": [
    { "name": "evaluate_board", "..." },
    { "name": "simulate_moves", "..." },
    { "name": "check_anchor_safety", "..." }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": { "..." }
  },
  "temperature": 0.1,
  "max_completion_tokens": 4000
}
```

### 3.3 Response Parsing

```javascript
async function parseGPT52Response(data) {
  const message = data.choices[0].message;
  
  // Check for function calls
  if (message.function_call) {
    const result = handleFunctionCall(
      message.function_call.name,
      JSON.parse(message.function_call.arguments)
    );
    // Send function result back to GPT-5.2
    return await continueConversation(result);
  }
  
  // Check for tool calls (code interpreter)
  if (message.tool_calls) {
    // Handle code interpreter results
    return await handleToolCalls(message.tool_calls);
  }
  
  // Parse structured JSON response
  const decision = JSON.parse(message.content);
  return decision.recommended_move;
}
```

## 4. Performance Specifications

### 4.1 Expected Improvements

| Metric | Baseline (No Skills) | Target (With Skills) | Measurement |
|--------|---------------------|----------------------|-------------|
| Win Rate @ 2048 | 40-50% | 55-70% | 100 games |
| Average Score | 12,000-15,000 | 18,000-23,000 | 100 games |
| Max Tile Reached | 1024 (60%) | 2048 (65%) | Distribution |
| Decision Quality | 70% optimal | 80% optimal | Expert review |
| Reasoning Transparency | 0% | 100% | Qualitative |

**Note**: Targets revised based on Azure OpenAI capabilities (no code interpreter)

### 4.2 Cost Analysis

**Per Move Cost**:
- Basic Prompt: $0.005 (500 tokens @ $1.75/M input, 50 tokens @ $14/M output)
- With Functions: $0.015 (1K tokens input, 200 tokens output + function overhead)
- With Structured Output: $0.012 (800 tokens input, 300 tokens output)
- With Both: $0.018 (1.2K tokens input, 400 tokens output)

**Full Game Cost** (avg 150 moves to 2048):
- Basic: $0.75
- With Skills: $2.25 - $2.70

**Note**: Significantly lower than originally estimated since code interpreter (most expensive) is not used

**Optimization Strategies**:
1. Cache function results for identical board states
2. Use skills only for complex board states (>50% full)
3. Hybrid: Skills for critical decisions, local algorithm for simple moves
4. Batch API calls for multi-game analysis

### 4.3 Latency Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Basic LLM call | 1.5s | 3s |
| With Code Interpreter | 3s | 5s |
| With Function Calls | 2s | 4s |
| UI Update | 100ms | 200ms |

## 5. Testing Specifications

### 5.1 Unit Tests

- [ ] Skills toggle functionality
- [ ] Schema validation
- [ ] Function call handlers
- [ ] Response parsing (valid JSON)
- [ ] Response parsing (invalid JSON, fallback)
- [ ] Anchor guard integration

### 5.2 Integration Tests

- [ ] Complete game flow with skills enabled
- [ ] Complete game flow with skills disabled
- [ ] Model switching (GPT-5.2 ↔ GPT-5.2-Chat ↔ DeepSeek)
- [ ] Function call round-trip
- [ ] Code interpreter execution and parsing

### 5.3 Performance Tests

**Benchmark Configuration**:
```bash
# 100 games with GPT-5.2 + Skills
node bench/bench2048.js --agent llm_rerank --games 100 --seed 123 --skills

# 100 games with basic prompts
node bench/bench2048.js --agent llm_rerank --games 100 --seed 123 --no-skills

# Compare results
node bench/analyze.js --compare results_skills.json results_basic.json
```

**Success Criteria**:
- Win rate improvement: ≥15%
- Average score improvement: ≥30%
- No regression in move execution time
- API error rate: <1%

### 5.4 User Acceptance Tests

- [ ] Skills toggle is visible and functional
- [ ] Skills status shows active capabilities
- [ ] AI reasoning is displayed clearly
- [ ] Board analysis metrics are accurate
- [ ] Move recommendations make sense
- [ ] Fallback works when skills fail

## 6. Documentation Requirements

### 6.1 Code Documentation

- JSDoc comments for all new functions
- Inline comments explaining skills logic
- Error handling documentation

### 6.2 User Documentation

**In-Game Help**:
```
🧠 GPT-5.2 Skills

When enabled, GPT-5.2 uses advanced capabilities:
• Code Interpreter: Executes Python code to analyze moves
• Function Calling: Uses game-specific evaluation functions
• Structured Outputs: Provides detailed reasoning and metrics

This improves decision quality but may add 1-2s per move.
Toggle off for faster (but less optimal) decisions.
```

**spec.md Updates**:
- Add GPT-5.2 Skills section
- Document configuration options
- Explain when skills are active

### 6.3 Research Documentation

**paper_llm_focus.md Updates**:
- New section: "3.6 Strategy S5: GPT-5.2 Skills Enhancement"
- Benchmark comparison table
- Cost-benefit analysis
- Ablation study results

## 7. Deployment Checklist

- [ ] Implement skills configuration
- [ ] Add structured output schema
- [ ] Create function definitions and handlers
- [ ] Update fetchAzureMove() with skills logic
- [ ] Add UI toggle and status display
- [ ] Implement response parsing for all formats
- [ ] Add error handling and fallbacks
- [ ] Test with real Azure GPT-5.2 endpoint
- [ ] Validate API costs against budget
- [ ] Update documentation
- [ ] Run benchmark suite
- [ ] Deploy to production
- [ ] Monitor performance and costs

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Code interpreter fails | High | Medium | Fallback to function calling or basic prompt |
| Structured output invalid | Medium | Low | Robust JSON parsing with try-catch |
| Function call errors | Medium | Low | Validate inputs, handle exceptions |
| API rate limits | High | Medium | Implement exponential backoff, caching |
| High latency | Medium | Medium | Show progress indicators, timeout handling |

### 8.2 Cost Risks

| Risk | Mitigation |
|------|------------|
| Excessive API usage | Implement usage tracking, daily limits |
| Skills overuse | Hybrid mode: skills only when needed |
| Unexpected charges | Monitor spending, set budget alerts |

### 8.3 User Experience Risks

| Risk | Mitigation |
|------|------------|
| Slow response times | Async UI updates, loading indicators |
| Confusing reasoning output | Format for readability, summarize key points |
| Skills toggle confusion | Clear labeling, help tooltips |

## 9. Success Metrics

### 9.1 Quantitative Metrics

- **Win Rate**: ≥65% (vs 45% baseline)
- **Average Score**: ≥22,000 (vs 13,000 baseline)
- **2048 Tile Rate**: ≥70% (vs 45% baseline)
- **API Error Rate**: <1%
- **Average Response Time**: <3.5s

### 9.2 Qualitative Metrics

- User satisfaction with AI decisions
- Reasoning clarity and usefulness
- Skills toggle adoption rate
- Feature stickiness (continued use)

## 10. Future Enhancements

### 10.1 Phase 2 Features

- **Adaptive Skills**: Auto-enable skills for complex boards only
- **Learning from Games**: Fine-tune based on game collection
- **Multi-step Planning**: Look ahead 3-5 moves
- **Visualization**: Show AI reasoning as board overlay

### 10.2 Phase 3 Features

- **Custom Strategies**: User-defined heuristic weights
- **Tournament Mode**: AI vs AI battles
- **Reinforcement Learning**: Train on successful games
- **Real-time Coaching**: Suggest moves as user plays

---

## Appendix A: API Examples

### A.1 Request with Code Interpreter

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert 2048 AI strategist with Python code execution capabilities. Analyze the board state and recommend the best move."
    },
    {
      "role": "user",
      "content": "Board:\n0 0 2 4\n0 0 8 16\n2 4 32 64\n4 8 128 256\n\nWrite Python code to evaluate all moves and recommend the best one."
    }
  ],
  "tools": [{ "type": "code_interpreter" }],
  "temperature": 0.1
}
```

### A.2 Response with Code Execution

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'll analyze the board using Python...",
      "tool_calls": [{
        "type": "code_interpreter",
        "code_interpreter": {
          "input": "# Board evaluation code...",
          "outputs": [{
            "type": "result",
            "result": "Best move: right (score: 0.92)"
          }]
        }
      }]
    }
  }]
}
```

---

**Document Version**: 1.0  
**Last Updated**: January 6, 2026  
**Author**: GitHub Copilot  
**Status**: Ready for Implementation
