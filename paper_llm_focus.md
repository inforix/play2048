# Paper Draft (LLM-Focused): Improving LLM Win Rate in 2048 via Constraints and Hybrid Reranking

> English draft text suitable to be copied into an SCI-style manuscript. Numbers should be filled from `bench/results.json` (see `bench/README.md`).

## 3. LLM-Centric Methodology

### 3.1 Problem: Why prompt-only LLMs underperform
A prompt-only large language model (LLM) controller for 2048 typically exhibits two failure modes: (i) **catastrophic invariant violations** (e.g., pulling the maximum tile away from the anchor corner), and (ii) **short-horizon greed** (selecting moves that merge immediately but reduce long-term survivability by decreasing free cells or breaking monotonicity). These behaviors arise because the LLM is not inherently optimizing the stochastic long-horizon objective of 2048 and is sensitive to ambiguous board descriptions.

### 3.2 Strategy S1: Hard safety constraints (Anchor Guard)
We enforce a deterministic guard that rejects moves likely to destroy the core invariant: keeping the maximum tile anchored in the designated corner (top-right).

Let $M=\max(B)$ and define an anchor indicator
$$
\mathbb{I}_{\text{anchor}}(B) = \mathbb{I}[B_{0,3}=M].
$$
If $\mathbb{I}_{\text{anchor}}(B)=1$, we disallow a subset of actions (notably \texttt{down} unless the rightmost column is fully occupied) and fall back to the next-best safe action. This simple post-processing step converts unconstrained natural-language suggestions into **safe action selection**.

### 3.3 Strategy S2: Candidate move generation + local reranking
Instead of asking the LLM to directly output the final action, we treat it as a **candidate generator** and combine it with a local evaluator $E(B)$:

1) Enumerate valid actions $\mathcal{A}_\text{valid}(B)$.
2) Ask the LLM to suggest a short ranked list (or apply a prompt preference ordering).
3) Apply Anchor Guard to each candidate.
4) Choose
$$
\pi(B)=\arg\max_{a\in\mathcal{A}_\text{valid}(B)} \; \Big(E(f(B,a)) + \lambda\,\Delta\sigma(B,a)\Big).
$$

This hybrid approach reduces the decision problem for the LLM (from global planning to local preference expression) while guaranteeing that the final decision respects survival-oriented heuristics.

### 3.4 Strategy S3: Retrieval-augmented prompting from move-history and game collection
Aurora-2048 records move-by-move boards and automatically saves completed games. We exploit these logs to build retrieval prompts:

- **Single-game learning:** summarize patterns that led to success/failure within the current episode.
- **Multi-game learning:** compare top-performing games vs low-performing games to extract discriminative rules (e.g., when \texttt{down} is safe, how to recover the anchor, how to maintain 4+ empty cells).

The retrieved rule set is injected into the LLM prompt as a compact policy prior.

### 3.5 Strategy S4: LLM as parameter tuner (optional)
An additional approach is to let the LLM tune coefficients in $E(B)$ (e.g., empty-cell weight vs monotonicity weight) based on observed failures, while the action selection remains deterministic. This improves stability and keeps runtime predictable.

## 4. Experimental Setup

### 4.1 Protocol
We evaluate each agent over $N$ self-play games with fixed random seeds. Each game ends when no moves are available.

**Metrics:** Win@2048 (%), average score, average steps, and max-tile distribution.

### 4.2 Baselines and variants
- **Prompt-only proxy:** `llm_naive`
- **Constraint-only:** `llm_guarded` (prompt proxy + Anchor Guard)
- **Hybrid reranking:** `llm_rerank` (prompt proxy + Anchor Guard + local evaluation)
- **Strong non-LLM baselines:** `expectimax`, `montecarlo`, `greedy_eval`

### 4.3 Reproducibility
All results in this manuscript should be generated with:

```bash
node bench/bench2048.js --all --games N --seed S --json > bench/results.json
```

## 5. Results (Fill from Benchmarks)

### 5.1 Overall performance
Create a table from `bench/results.json`:

| Agent | Win@2048 | Avg score | Avg steps |
|---|---:|---:|---:|
| llm_naive | (fill) | (fill) | (fill) |
| llm_guarded | (fill) | (fill) | (fill) |
| llm_rerank | (fill) | (fill) | (fill) |
| greedy_eval | (fill) | (fill) | (fill) |
| montecarlo | (fill) | (fill) | (fill) |
| expectimax | (fill) | (fill) | (fill) |

### 5.2 Effect of constraints and hybridization
Report relative improvements:

- Anchor Guard gain in win rate:
$$
\Delta_{\text{guard}} = \text{WinRate}(\texttt{llm_guarded}) - \text{WinRate}(\texttt{llm_naive}).
$$

- Reranking gain over guard:
$$
\Delta_{\text{rerank}} = \text{WinRate}(\texttt{llm_rerank}) - \text{WinRate}(\texttt{llm_guarded}).
$$

### 5.3 Max-tile distribution
Plot or tabulate the probability of reaching $\{512, 1024, 2048, 4096\}$.

## 6. Discussion (LLM win-rate mechanisms)
- **Hard constraints** prevent rare but catastrophic errors, often producing a disproportionate increase in Win@2048.
- **Hybrid reranking** reduces myopic merges and better preserves empty space and monotonic gradients.
- **Retrieval prompts** help the LLM learn *when exceptions apply* (e.g., when \texttt{down} is safe) without relying on general knowledge.

## 7. Limitations
- The offline benchmark includes `llm_*` proxy agents that do not measure network latency/cost and do not capture true language-model variability.
- A real LLM benchmark should include rate limits, timeouts, and a consistent decoding policy.
