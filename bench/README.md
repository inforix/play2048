# Aurora-2048 Offline Benchmarks (Node)

This folder contains a **Node.js** benchmark harness for the 2048 mechanics used in this repo.
It is designed to produce **reproducible** metrics (win rate, average score, max-tile distribution)
without requiring a browser UI.

## Why this exists
The paper focus is **how to improve an LLM-driven agent to win 2048**. In practice, the most reliable
way to do this is to combine LLM guidance with **hard constraints** and **local evaluation/search**.

To avoid fabricating results, this harness lets you actually measure performance.

## Run
From repo root:

```bash
node bench/bench2048.js --agent expectimax --games 200 --seed 123
node bench/bench2048.js --all --games 200 --seed 123
node bench/bench2048.js --agent montecarlo --games 200 --seed 123 --simulations 100
node bench/bench2048.js --agent expectimax --games 200 --seed 123 --depth 4
node bench/bench2048.js --agent llm_rerank --games 200 --seed 123
```

To get machine-readable output:

```bash
node bench/bench2048.js --all --games 500 --seed 123 --json > bench/results.json
```

## Agents included
- `random`: random valid move (lower bound)
- `greedy_eval`: 1-ply greedy using the same heuristic evaluation
- `expectimax`: depth-limited expectimax with chance-node sampling
- `montecarlo`: Monte Carlo rollouts with epsilon-greedy simulations
- `llm_naive`: proxy for a naive "prefer right/up" prompt (no search)
- `llm_guarded`: `llm_naive` + **anchor guard** constraint
- `llm_rerank`: proxy for **LLM as reranker**: candidates + anchor guard + local heuristic scoring

> Important: `llm_*` agents here **do not call an actual LLM**.
> They are algorithmic stand-ins for prompt+constraint pipelines, so the benchmark can run offline.

## How to use in the paper
- Use `llm_naive` as a baseline representing "prompt-only" behavior.
- Use `llm_guarded` to quantify the benefit of hard constraints.
- Use `llm_rerank` to quantify the benefit of hybridizing with local evaluation.
- Report `expectimax`/`montecarlo` as strong non-LLM baselines.

If you want to benchmark a *real* Azure LLM, we can extend the harness with an API client (cost/latency applies).
