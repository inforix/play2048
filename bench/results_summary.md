# Benchmark Summary (from results.json)

Run configuration:
- Games per agent: 200
- Seed: 123

## Main table

| Agent | Win@2048 | Avg score | Avg steps |
|---|---:|---:|---:|
| llm_naive | 0.0% | 2196.0 | 200.8 |
| llm_guarded | 0.0% | 2196.0 | 200.8 |
| llm_rerank | 0.0% | 4352.1 | 326.3 |
| greedy_eval | 0.0% | 4352.1 | 326.3 |
| montecarlo | 0.0% | 1995.9 | 183.0 |
| expectimax | 49.0% | 23205.9 | 1234.3 |
| random | 0.0% | 1112.7 | 119.6 |

## Key deltas (LLM-focused)
- Anchor Guard (llm_guarded vs llm_naive):
  - Win@2048: +0.0 percentage points
  - Avg score: +0.0
  - Note: identical outcomes in this run.

- Hybrid reranking (llm_rerank vs llm_guarded):
  - Win@2048: +0.0 percentage points
  - Avg score: +2156.1 (+98.2%)

## Max-tile highlights
- llm_naive: P(max tile = 512) = 0.5%, P(1024+) = 0.0%
- llm_rerank: P(max tile = 1024) = 5.0%, P(2048+) = 0.0%
- expectimax: P(max tile >= 2048) = 49.0% (2048: 42.0%, 4096: 7.0%)

## Interpretation (short)
- Prompt-only proxy policies did not reach 2048 in 200 games.
- Guarding only matters when the policy would otherwise select unsafe moves; with this preference order and seed, the guard did not change decisions.
- Reranking roughly doubled the score but still did not achieve 2048; deeper lookahead (e.g., expectimax) is required to reach high win rates.
