# Camera-ready LaTeX

Main manuscript:
- `paper/aurora2048_llm.tex`

## Build
From repo root:

```bash
cd paper
latexmk -pdf aurora2048_llm.tex
```

If you don't have `latexmk`:

```bash
cd paper
pdflatex aurora2048_llm.tex
pdflatex aurora2048_llm.tex
```

## Notes
- The results table is filled from your measured `bench/results.json` (200 games/agent, seed=123).
- The `llm_*` agents in the offline benchmark are proxies and do not invoke an API.
