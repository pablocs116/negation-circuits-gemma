# negation-circuits-gemma

![Status](https://img.shields.io/badge/status-in_progress-orange)
![Model](https://img.shields.io/badge/model-Gemma--2--2b-blue)
![Method](https://img.shields.io/badge/method-circuit--tracing-purple)

Mechanistic interpretability study of how Gemma-2-2b handles negation,
using attribution graphs from Anthropic's open-source circuit-tracer.

## Hypothesis

When processing negated factual statements (e.g. "Paris is NOT the capital
of Germany"), Gemma-2-2b uses a distinct internal circuit that actively
suppresses the semantically associated but incorrect answer token. I expect
to find dedicated features that encode negation scope and causally intervene
on the factual retrieval pathway — and that disabling these features causes
the model to output the wrong (unnegated) answer.

## Research questions

1. Which features activate specifically on negation tokens, and at what layers?
2. Does negation suppress the incorrect answer directly, or reroute through
   alternative features?
3. Is the negation circuit the same across surface forms (NOT / never /
   no / isn't)?
4. Does the circuit generalise across domains (geography, arithmetic, facts)?

## Structure

notebooks/
├── 01_exploration.ipynb        # first attribution graphs, intuition building
├── 02_negation_analysis.ipynb  # main experiment: negation circuit mapping
└── 03_patching_experiments.ipynb  # causal patching to test hypotheses

experiments/
└── log.md                      # dated lab notebook

graphs/
└── *.json                      # saved attribution graphs

## Setup

```bash
git clone https://github.com/pablocs116/negation-circuits-gemma
cd negation-circuits-gemma
pip install circuit-tracer
```

All experiments run on a free Colab T4 GPU. See notebooks/01_exploration.ipynb
to get started.

## Key findings

🔜 In progress — check back in July 2025

## References

- Ameisen et al. (2025) — Attribution Graphs
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Lindsey et al. (2024) — Crosscoders / Transcoders
  https://transformer-circuits.pub/2024/crosscoders/index.html
- circuit-tracer library
  https://github.com/decoderesearch/circuit-tracer

## Author

Pablo Cabriada Sierra — ML Engineer, Multiverse Computing
https://linkedin.com/in/pablo-cabriada-sierra
