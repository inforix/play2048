# Specification Index

This directory contains all project specifications following spec-driven development principles.

## Core Specifications

### 1. Game Specification (`spec.md`)
**Purpose**: Defines the 2048 game implementation, UI/UX, and AI integration

**Key Sections**:
- Gameplay rules and mechanics
- UI components and styling
- AI strategies (Expectimax, Monte Carlo, Weighted, LLM)
- Learning system (single-game and multi-game analysis)
- Move history features
- Persistence (localStorage)

**Status**: ✅ Complete & Implemented  
**Implementation**: `src/game/index.html`

---

### 2. Training Specification (`train_spec.md`)
**Purpose**: Defines neural network architectures, training procedures, and evaluation metrics

**Key Sections**:
- **Method 1**: CNN-Based Policy Network
  - 4 conv layers + FC layers
  - Cross-entropy loss
  - 100 epochs training
  
- **Method 2**: Dual Network (AlphaZero-style)
  - ResNet backbone with 3 residual blocks
  - Policy head + Value head
  - Combined loss (policy + value)
  - 150 epochs training
  
- **Method 3**: Transformer-Based
  - 4 transformer encoder layers
  - 2D positional encoding
  - Multi-head attention (8 heads)
  - 200 epochs with warmup
  
- Data preprocessing and augmentation
- Hyperparameter configurations
- Evaluation metrics (offline + online)
- Model comparison framework

**Status**: 📋 Specification Complete, Implementation Pending  
**Implementation**: To be created in `models/`, `training/`, `evaluation/`

---

### 3. Dataset Generation Specification (`generate_dataset.md`)
**Purpose**: Defines automated dataset generation using Expectimax AI

**Key Sections**:
- Game logic implementation (tile mechanics, movement rules)
- Expectimax algorithm (4-ply search, evaluation function)
- Dataset format (JSONL schema with board-before-move)
- CLI interface (arguments, options, usage)
- Performance characteristics and quality metrics
- Integration with training pipeline

**Status**: ✅ Complete & Implemented  
**Implementation**: `scripts/generate_dataset.py`

---

## Specification Hierarchy

```
┌─────────────────────────────────────┐
│      spec.md (Game Rules)           │
│  Defines what to learn from         │
└──────────────┬──────────────────────┘
               │
               ├─────────────────────────┐
               ↓                         ↓
┌──────────────────────────┐  ┌─────────────────────────┐
│ generate_dataset.md      │  │  train_spec.md          │
│ (Data Collection)        │  │  (Model Training)       │
│                          │  │                         │
│ Expectimax AI plays →    │  │  ← Learns from data     │
│ generates expert data    │  │  3 model architectures  │
└──────────────────────────┘  └─────────────────────────┘
```

## Using These Specifications

### For Developers
1. **Before implementing**: Read the relevant specification
2. **During implementation**: Reference spec for exact requirements
3. **After implementation**: Validate against spec criteria
4. **On changes**: Update spec first, then code

### For Users
- **Play the game**: See `spec.md` for features and controls
- **Generate data**: See `generate_dataset.md` for usage and options
- **Train models**: See `train_spec.md` for architectures and procedures

### For Contributors
- All PRs must reference specification section
- Code changes may require spec updates
- New features require new specification sections

## Spec-Driven Development Workflow

```
1. Write Specification
   ↓
2. Review & Refine Spec
   ↓
3. Implement According to Spec
   ↓
4. Validate Implementation
   ↓
5. Update Spec if Deviations Found
   ↓
6. Document Learnings
```

## Version Control

| Specification | Version | Last Updated | Status |
|---------------|---------|--------------|--------|
| spec.md | 1.0 | 2026-01-07 | Stable |
| train_spec.md | 1.0 | 2026-01-07 | Stable |
| generate_dataset.md | 1.0 | 2026-01-07 | Stable |

## Related Documentation

- **Project README**: `../README.md` - Project overview
- **Implementation Docs**: `../docs/` - Implementation guides and notes
- **Code Documentation**: In-code comments and docstrings

---

**Specification Philosophy**: 
> "Write the specification first, code second. The spec is the source of truth."

All implementations must conform to these specifications. Deviations require specification updates with clear rationale.
