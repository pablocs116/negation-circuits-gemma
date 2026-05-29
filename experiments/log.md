## 2026-05-29

**Prompt:** "The capital of France is Paris. The capital of Germany is NOT Paris, it is"
**Hypothesis:** Negation features suppress "Paris" as output token; expect dedicated suppression features at mid-to-late layers.
**Method:** Neuronpedia API, gemma-2-2b, `scripts/generate_graph.py`
**Slug:** `negation-geography-germany-01`
**Neuronpedia URL:** https://neuronpedia.org/gemma-2-2b/graph?slug=negation-geography-germany-01
**Graph JSON:** https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmpqwi2000004ezxf1zdbkh29/negation-geography-germany-01-1780058041008.json
**Nodes found:** [inspect on Neuronpedia UI]
**Top features:** [inspect on Neuronpedia UI]
**Surprise / anomaly:** None yet — first graph
**Next step:** Run `scripts/batch_generate.py` for all 5 prompts, compare node overlap
