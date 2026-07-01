# negation-circuits-gemma

![Status](https://img.shields.io/badge/status-in_progress-orange)
![Model](https://img.shields.io/badge/model-Gemma--2--2b-blue)
![Method](https://img.shields.io/badge/method-circuit--tracing-purple)

Mechanistic interpretability study of how Gemma-2-2b handles negation,
using attribution graphs generated via the [Neuronpedia](https://www.neuronpedia.org)
circuit-tracing API — no local GPU required.

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

```
notebooks/
├── 01_exploration.ipynb        # first attribution graphs, intuition building
├── 02_negation_analysis.ipynb  # main experiment: negation circuit mapping (stub)
└── 03_patching_experiments.ipynb  # causal patching to test hypotheses (stub)

scripts/
├── generate_graph.py           # generate one attribution graph via Neuronpedia API
├── batch_generate.py           # generate the 5 core negation prompts
└── analyze_graphs.py           # fetch full graphs, reproduce overlap/control analysis

experiments/
└── log.md                      # dated lab notebook (experiments 1–5)

graphs/
└── *.json                      # saved graph metadata (gitignored; regenerate via scripts)
```

## Setup

```bash
git clone https://github.com/pablocs116/negation-circuits-gemma
cd negation-circuits-gemma
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then paste your Neuronpedia API key into .env
python scripts/generate_graph.py
```

Graphs are generated server-side by the Neuronpedia API — no local model, GPU,
or CUDA needed. Get a free API key at https://www.neuronpedia.org. See CLAUDE.md
for the full step-by-step setup.

## Key findings (experiments 1–5)

- **A domain-general negation circuit exists.** Across 5 domains (geography,
  physics, astronomy, history, tech), a small set of late-layer features fires
  on negated prompts.
- **The circuit is narrow and late.** Only ~3 features are truly negation-specific
  (present in negation graphs, absent in affirmative controls), all at layers
  24–25: `L25/167884`, `L24/16207946`, `L24/18147275`. `L25/167884` is the single
  most robust — 5/5 negation graphs, 0/5 controls.
- **Mid-layer features are shared with controls** — they handle factual retrieval,
  not negation. The negation circuit acts as a late-layer suppression gate on the
  retrieval pathway, not a broad mid-layer mechanism.
- **Surface-form invariant.** The same L24–25 features fire for NOT / NEVER /
  ISN'T / NO, suggesting they encode semantic negation rather than a lexical
  pattern.

Full details and Neuronpedia graph URLs in [`experiments/log.md`](experiments/log.md).
Next step: ablation — disable the L24–25 features and check whether the model
reverts to the un-negated answer.

## References

- Ameisen et al. (2025) — Attribution Graphs
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Lindsey et al. (2024) — Crosscoders / Transcoders
  https://transformer-circuits.pub/2024/crosscoders/index.html
- Neuronpedia circuit tracer
  https://www.neuronpedia.org/gemma-2-2b/graph

## Author

Pablo Cabriada Sierra — ML Engineer, Multiverse Computing
https://linkedin.com/in/pablo-cabriada-sierra
