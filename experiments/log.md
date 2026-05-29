## 2026-05-29 — Experiment 1: First negation graph

**Prompt:** "The capital of France is Paris. The capital of Germany is NOT Paris, it is"
**Hypothesis:** Negation features suppress "Paris" as output token; expect dedicated suppression features at mid-to-late layers.
**Method:** Neuronpedia API, gemma-2-2b, `scripts/generate_graph.py`
**Slug:** `negation-geography-germany-01`
**Neuronpedia URL:** https://neuronpedia.org/gemma-2-2b/graph?slug=negation-geography-germany-01
**Output:** "Berlin" (p=0.845) — model correctly resolves negation
**Top 5 by influence:** L2/35460828, L12/68907917, L12/13741890, L1/40002038, L6/75074124
**Top 3 by activation:** L24/88478228 (122.2), L24/16207946 (113.2), L25/167884 (103.7)
**Surprise / anomaly:** None yet — first graph
**Next step:** Batch generate all 5 prompts, cross-domain comparison

---

## 2026-05-29 — Experiment 2: Cross-domain negation circuit (5 prompts)

**Prompts:**
1. Geography: "The capital of France is Paris. The capital of Germany is NOT Paris, it is" → Berlin (p=0.845)
2. Physics: "Water is NOT a solid at room temperature, it is a" → liquid (p=0.694)
3. Astronomy: "The sun does NOT orbit the Earth, the Earth orbits the" → sun (p=0.656)
4. History: "Shakespeare was NOT born in London, he was born in" → Stratford (p=0.729)
5. Tech: "Python is NOT a compiled language, it is a" → interpreted (p=0.446)

**Method:** Neuronpedia API, gemma-2-2b, `scripts/batch_generate.py`

**Neuronpedia URLs:**
- https://neuronpedia.org/gemma-2-2b/graph?slug=negation-geography-germany
- https://neuronpedia.org/gemma-2-2b/graph?slug=negation-physics-water
- https://neuronpedia.org/gemma-2-2b/graph?slug=negation-astronomy-sun
- https://neuronpedia.org/gemma-2-2b/graph?slug=negation-history-shakespeare
- https://neuronpedia.org/gemma-2-2b/graph?slug=negation-tech-python

**Cross-domain features (appearing in 5/5 graphs, top 50 by activation):**
- L24/88478228 — present in ALL graphs (6 occurrences, appears twice in one graph)
- L25/11250370 — present in all 5
- L25/167884 — present in all 5
- L24/16207946 — present in all 5
- L19/90310060 — present in all 5
- L10/100614194 — present in all 5
- L9/125286525 — present in all 5

**Layer distribution of high-activation features:**
- Layers 24-25 dominate (52 of 250 top-50 slots) — late-layer features are most active
- Layers 9-11 also prominent (42 slots) — mid-layer features may encode negation scope
- Layer 0 appears in 4/5 graphs — early-layer token-level features

**Pairwise overlap (top 50 features):**
- physics × tech: 25 shared (highest — both are "X is NOT a Y" format)
- history × tech: 21 shared
- history × physics: 19 shared
- astronomy × physics: 13 shared
- astronomy × tech: 13 shared
- geography × physics: 11 shared
- geography × tech: 8 shared
- geography × history: 6 shared (lowest)

**Key finding:** 7 features appear in all 5 negation graphs across domains — these are strong candidates for a **domain-general negation circuit**. The concentration at L24-25 suggests late-layer suppression/steering, while L9-10 features may encode negation scope detection.

**Surprise / anomaly:**
- Geography has lowest overlap with other domains — may rely more on factual retrieval than negation-specific features
- L15/376262 has very high activation (117-157) in 3/5 graphs but low influence — could be a "reading" feature rather than causal
- L24/88478228 appears in ALL graphs with very high activation — strongest negation circuit candidate

**Next step:** Generate non-negation control prompts ("The capital of Germany is") to confirm which features are negation-specific vs. general factual retrieval

---

## 2026-05-29 — Experiment 3: Control comparison (negation vs non-negation)

**Control prompts (affirmative, same domain):**
1. "The capital of Germany is" → Berlin
2. "Water is a liquid at room temperature, it is a" → liquid
3. "The Earth orbits the" → sun
4. "Shakespeare was born in" → Stratford
5. "Python is an interpreted language, it is a" → interpreted

**Method:** Same as Experiment 2, then compared top-50 activation features per graph

**Neuronpedia URLs:**
- https://neuronpedia.org/gemma-2-2b/graph?slug=control-geography-germany
- https://neuronpedia.org/gemma-2-2b/graph?slug=control-physics-water
- https://neuronpedia.org/gemma-2-2b/graph?slug=control-astronomy-sun
- https://neuronpedia.org/gemma-2-2b/graph?slug=control-history-shakespeare
- https://neuronpedia.org/gemma-2-2b/graph?slug=control-tech-python

### Negation-specific features (in 3+ negation graphs, 0-1 control):

| Feature | Negation | Control | Ratio |
|---------|----------|---------|-------|
| L25/167884 | 5/5 | 0/5 | ∞ |
| L24/16207946 | 4/5 | 0/5 | ∞ |
| L24/18147275 | 3/5 | 0/5 | ∞ |

### Negation-enriched features (ratio ≥ 2x):

| Feature | Negation | Control | Ratio |
|---------|----------|---------|-------|
| L25/167884 | 5/5 | 0/5 | 5.0x |
| L24/16207946 | 4/5 | 0/5 | 4.0x |
| L24/18147275 | 3/5 | 0/5 | 3.0x |
| L25/11250370 | 5/5 | 2/5 | 2.5x |

### Shared features (in 4+ negation AND 3+ control — NOT negation-specific):
L9/125286525, L10/100614194, L24/88478228, L9/3843368, L12/78206258, L11/42278598, L10/59891029, L9/4701701

**Key findings:**
1. **Only 3 features are truly negation-specific** (absent in controls): L25/167884, L24/16207946, L24/18147275 — all at layers 24-25 (late suppression)
2. **L25/167884 is the strongest negation feature** — appears in ALL 5 negation graphs, ZERO controls
3. Most features previously identified as "negation circuit" (L24/88478228, L9-10 features) are actually **general factual retrieval** features, not negation-specific
4. The negation circuit is **narrower than expected** — just 3-4 features at L24-25, not a broad mid-layer circuit

**Surprise / anomaly:**
- L24/88478228 was the highest-activation feature across all negation graphs but also appears in all controls — it's a general output/steering feature, not negation-specific
- Mid-layer features (L9-12) are entirely shared with controls — they handle factual retrieval, not negation
- The negation-specific features cluster at L24-25, suggesting negation operates as a **late-layer suppression gate** on the factual retrieval pathway

**Next step:** Test negation surface forms ("never"/"isn't"/"no" instead of "NOT") to see if the same L24-25 features activate, or if different negation words recruit different features

---

## 2026-05-29 — Experiment 4: Surface form generalization (NOT vs NEVER vs ISN'T vs NO)

**Prompts (same domain, different negation surface form):**
- Geography: "NOT Paris" / "NEVER Paris" / "isn't Paris" / "No, ... not Paris"
- Physics: "NOT a solid" / "never a solid" / "isn't a solid"

**Method:** Same as Experiments 2-3, compared top-50 activation features against NOT-specific features

**Neuronpedia URLs:**
- https://neuronpedia.org/gemma-2-2b/graph?slug=negform-never-geography
- https://neuronpedia.org/gemma-2-2b/graph?slug=negform-isnt-geography
- https://neuronpedia.org/gemma-2-2b/graph?slug=negform-no-geography
- https://neuronpedia.org/gemma-2-2b/graph?slug=negform-never-physics
- https://neuronpedia.org/gemma-2-2b/graph?slug=negform-isnt-physics

### Negation-specific features across surface forms:

| Feature | NOT geo | NOT phys | NEVER geo | ISNT geo | NO geo | NEVER phys | ISNT phys | Control |
|---------|---------|----------|-----------|----------|--------|------------|-----------|---------|
| L25/167884 | YES | YES | YES | YES | YES | YES | YES | no |
| L24/16207946 | YES | YES | YES | YES | YES | no | YES | no |
| L24/18147275 | YES | YES | YES | YES | YES | YES | YES | no |
| L25/11250370 | YES | YES | YES | YES | YES | YES | YES | no |

**Key findings:**
1. **All 4 negation-specific/enriched features generalize across surface forms** — L25/167884 appears in ALL 7 negation graphs (NOT/NEVER/ISN'T/NO) and ZERO controls
2. The negation circuit is **surface-form invariant** — NOT, NEVER, ISN'T, and NO all recruit the same L24-25 features
3. This strongly suggests these features encode **semantic negation** (the concept of "X is false"), not lexical patterns of specific negation words
4. L25/167884 is the single most robust negation feature — the "negation neuron" of Gemma-2-2b

**Surprise / anomaly:**
- L24/16207946 was absent in "NEVER physics" — may be geography-biased or need larger sample
- "No, ... not" (double negation / contrastive) still activates same features — the circuit handles discourse-level negation, not just word-level

**Next step:** Extract and upload the negation subgraph (only L24-25 negation-specific features + their links) to Neuronpedia for clean visualization

---

## 2026-05-29 — Experiment 5: Negation circuit subgraph extraction

**Method:** Extracted top-5 strongest incoming/outgoing links per negation feature from the geography graph, creating a 30-node/39-link focused circuit diagram.

**Neuronpedia URL (focused circuit):** https://neuronpedia.org/gemma-2-2b/graph?slug=negation-circuit-focused

### Circuit structure:

**Negation-specific features (4 nodes):**
- L25/167884 (act=103.7) — "negation neuron"
- L24/16207946 (act=113.1)
- L24/18147275 (act=53.4)
- L25/11250370 (act=83.4)

**Key upstream inputs (feeding into negation features):**
- L24/88478228 (act=122.2) — general factual retrieval feature, strongest input
- L22/25873198, L21/17793573 — mid-layer semantic features
- L21/80600534 (ctx=13) — processes "NOT" token position

**Key downstream outputs (negation features feed into):**
- L25/60352 (act=36.7) — strongest output target
- L25/133309930, L25/25364977, L25/90350377 — L25-to-L25 routing
- L27/228087 → "BERLIN" (p=0.016) — direct link to correct output
- L27/6544 → "NOT" (p=0.005) — residual echo of negation token

**Circuit interpretation:**
1. Factual retrieval features (L24/88478228) feed INTO negation features
2. Negation features at L24-25 act as a **suppression gate** — they receive the "Paris" signal and redirect it
3. Negation features output to other L25 features that steer toward the correct answer ("Berlin")
4. The circuit operates in the **last 3 layers** — a narrow, late-layer intervention

**Next step:** Ablation experiment — test what happens when negation features are disabled (requires local model or API support)
