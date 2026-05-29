# negation-circuits-gemma — Claude Code Setup Plan

This file tells Claude Code exactly what to do to get the project running.
Execute steps in order. Stop and report any error before continuing.

---

## Context

We are studying how Gemma-2-2b handles negation using mechanistic interpretability
and attribution graphs. We use the **Neuronpedia API** to generate graphs — no local
GPU, no CUDA, no dependency hell. Everything runs from a clean Python environment
on Mac M4 with Cursor IDE.

Model: `gemma-2-2b`
Method: Attribution graphs via Neuronpedia API + Python library
Hardware: Apple M4, 24GB unified RAM, Mac OS, Cursor IDE

---

## Architecture Overview

```
Cursor (Mac) — write all code, edit notebooks, GLM assists
      ↓
Neuronpedia API — generates attribution graphs on their servers
      ↓
Your repo — stores graphs as JSON, notebooks, and experiment log
      ↓
Neuronpedia UI — visualize and share graphs via URL
```

No Colab. No local model loading. No CUDA. No dependency conflicts.

---

## Step 0 — Create Neuronpedia Account (manual, in browser)

1. Go to https://www.neuronpedia.org
2. Click Sign In → create account (Google or email)
3. Once logged in, find your API key in profile/account settings
4. Copy the key — long alphanumeric string
5. Also try the Circuit Tracer UI manually first:
   https://www.neuronpedia.org/gemma-2-2b/graph
   Type a prompt, click Generate Graph — this is what the API automates.

---

## Step 1 — Verify local environment

```bash
pwd
# Should print: /Users/pcabriada/Projects/negation-circuits-gemma

which python
# Should point to .venv/bin/python

python --version
# Needs 3.10+
```

If venv not active:
```bash
source .venv/bin/activate
```

If venv does not exist:
```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Step 2 — Set Neuronpedia API key

```bash
# Add to ~/.zshrc permanently (replace with your real key)
echo 'export NP_API_KEY="YOUR_NEURONPEDIA_KEY_HERE"' >> ~/.zshrc
source ~/.zshrc

# Verify
echo $NP_API_KEY
# Should print your actual key
```

---

## Step 3 — Install dependencies (minimal, no conflicts)

```bash
pip install requests pandas matplotlib numpy jupyter notebook
pip freeze > requirements.txt
```

No transformers. No huggingface_hub. No torch. The Neuronpedia API handles
everything server-side — your Mac just sends HTTP requests.

---

## Step 4 — Verify API auth

```bash
python - <<'PYEOF'
import os, requests
api_key = os.environ.get("NP_API_KEY")
if not api_key or "YOUR_" in api_key:
    raise ValueError("NP_API_KEY not set correctly")
headers = {"x-api-key": api_key}
resp = requests.get("https://www.neuronpedia.org/api/ping", headers=headers)
print(f"Status: {resp.status_code} — {resp.text}")
PYEOF
```

Expected: Status 200. If 401: wrong API key. If connection error: check internet.

---

## Step 5 — Build project folder structure

```bash
mkdir -p notebooks graphs experiments scripts
touch experiments/log.md
touch scripts/generate_graph.py
touch scripts/batch_generate.py
touch scripts/analyze_graphs.py
```

---

## Step 6 — Create scripts/generate_graph.py

Write this file:

```python
# scripts/generate_graph.py
import os, json, requests, time

API_KEY = os.environ.get("NP_API_KEY")
BASE_URL = "https://www.neuronpedia.org/api"
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def generate_graph(prompt, slug=None):
    """Generate an attribution graph via Neuronpedia API."""
    payload = {
        "prompt": prompt,
        "model": "gemma-2-2b",
        "slug": slug or prompt[:40].replace(" ", "-").lower()
    }
    resp = requests.post(
        f"{BASE_URL}/graph/generate",
        headers=headers,
        json=payload,
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()

def save_graph(graph, name):
    """Save graph JSON to graphs/ directory."""
    path = f"graphs/{name}.json"
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
    slug = graph.get("slug", name)
    url = f"https://www.neuronpedia.org/gemma-2-2b/graph?slug={slug}"
    print(f"Saved: {path}")
    print(f"View:  {url}")
    return url

if __name__ == "__main__":
    prompt = "The capital of France is Paris. The capital of Germany is NOT Paris, it is"
    print(f"Generating graph...")
    graph = generate_graph(prompt, slug="negation-geography-germany-01")
    save_graph(graph, "negation_geography_germany_01")
```

Run it:
```bash
cd /Users/pcabriada/Projects/negation-circuits-gemma
python scripts/generate_graph.py
```

Expected output:
- JSON file saved in graphs/
- A Neuronpedia URL to view the graph visually

---

## Step 7 — Create scripts/batch_generate.py

This script generates all 5 core negation prompts at once:

```python
# scripts/batch_generate.py
import os, json, time, requests
from generate_graph import generate_graph, save_graph

PROMPTS = [
    ("The capital of France is Paris. The capital of Germany is NOT Paris, it is",
     "negation-geography-germany"),
    ("Water is NOT a solid at room temperature, it is a",
     "negation-physics-water"),
    ("The sun does NOT orbit the Earth, the Earth orbits the",
     "negation-astronomy-sun"),
    ("Shakespeare was NOT born in London, he was born in",
     "negation-history-shakespeare"),
    ("Python is NOT a compiled language, it is a",
     "negation-tech-python"),
]

if __name__ == "__main__":
    urls = []
    for prompt, slug in PROMPTS:
        print(f"\n[{slug}]")
        try:
            graph = generate_graph(prompt, slug=slug)
            url = save_graph(graph, slug)
            urls.append((slug, url))
            time.sleep(2)  # be polite to the API
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- All graphs ---")
    for slug, url in urls:
        print(f"{slug}: {url}")
```

---

## Step 8 — Scaffold notebooks/01_exploration.ipynb

Cells in order:

### Cell 1 — Setup
```python
import os, json, requests, pandas as pd, matplotlib.pyplot as plt
API_KEY = os.environ.get("NP_API_KEY")
assert API_KEY and "YOUR_" not in API_KEY, "Set NP_API_KEY in ~/.zshrc"
BASE_URL = "https://www.neuronpedia.org/api"
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
print("Setup OK")
```

### Cell 2 — Load saved graphs
```python
import glob
graph_files = glob.glob("../graphs/*.json")
graphs = {}
for f in graph_files:
    name = f.split("/")[-1].replace(".json", "")
    with open(f) as fh:
        graphs[name] = json.load(fh)
print(f"Loaded {len(graphs)} graphs: {list(graphs.keys())}")
```

### Cell 3 — Basic analysis
```python
for name, graph in graphs.items():
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    print(f"\n{name}")
    print(f"  Nodes: {len(nodes)} | Edges: {len(edges)}")
    if nodes:
        df = pd.DataFrame(nodes)
        if "attribution" in df.columns:
            top = df.nlargest(3, "attribution")[["label","attribution"]]
            print(f"  Top nodes:\n{top.to_string(index=False)}")
```

### Cell 4 — Neuronpedia URLs for visual inspection
```python
for name in graphs:
    slug = graphs[name].get("slug", name)
    url = f"https://www.neuronpedia.org/gemma-2-2b/graph?slug={slug}"
    print(f"{name}:\n  {url}\n")
```

---

## Step 9 — Log first experiment entry

After first graph runs, append to experiments/log.md:

```markdown
## YYYY-MM-DD

**Prompt:** "The capital of France is Paris. The capital of Germany is NOT Paris, it is"
**Hypothesis:** Negation features suppress "Paris" as output token; expect dedicated
suppression features at mid-to-late layers.
**Method:** Neuronpedia API, gemma-2-2b, generate_graph.py
**Nodes found:** [fill in]
**Top features:** [paste top 3 node labels from graph JSON or Neuronpedia UI]
**Neuronpedia URL:** [paste URL]
**Surprise / anomaly:** [anything unexpected]
**Next step:** Run batch_generate.py for all 5 prompts, compare node overlap
```

---

## Step 10 — Request higher API limits (optional but useful)

Default limits may throttle bulk experiments. Email Neuronpedia:
- Email: contact@neuronpedia.org
- Subject: "API whitelist request — negation circuit research"
- Body: one sentence about your project
- They do this for free per their blog post

---

## Step 11 — First git commit

```bash
git add .
git commit -m "day-1: Neuronpedia API setup, scripts, first negation graphs"
git push
```

---

## Weekly milestones

| Week | Dates       | Goal |
|------|-------------|------|
| 1    | May 27–Jun 2  | Setup, first 5 graphs, intuition building on Neuronpedia UI |
| 2    | Jun 3–Jun 9   | 20+ graphs across prompt variants, identify consistent nodes |
| 3    | Jun 10–Jun 16 | Compare negation vs non-negation graphs, find suppression features |
| 4    | Jun 17–Jun 23 | Deeper feature analysis, cross-domain generalization tests |
| 5    | Jun 24–Jun 30 | Write-up, README polish, embed Neuronpedia URLs in notebook |

---

## Ongoing: experiments/log.md entry format

```markdown
## YYYY-MM-DD

**Prompt:** "..."
**Hypothesis:** ...
**Method:** ...
**Nodes found:** ...
**Top features:** ...
**Neuronpedia URL:** ...
**Surprise / anomaly:** ...
**Next step:** ...
```

---

## Known issues and fixes

| Issue | Fix |
|-------|-----|
| `NP_API_KEY` blank in Python | `source ~/.zshrc` then restart Cursor kernel |
| API returns 401 | Wrong/expired key — regenerate at neuronpedia.org |
| API returns 429 | Rate limited — add `time.sleep(2)` between calls, or email for whitelist |
| Graph has no nodes | Prompt too short — use full sentence with clear factual claim |
| `graph/generate` not found | Check current endpoint at https://www.neuronpedia.org/api-doc |
| Connection timeout | Increase timeout param: `timeout=120` |

---

## DO NOT DO

- Do NOT try to install circuit-tracer locally — dependency conflicts are unresolvable
- Do NOT try to run on Colab — same dependency issue with huggingface_hub versions
- Do NOT run models locally via MPS — not needed, API is faster and simpler
- Do NOT upgrade huggingface_hub or transformers — not used in this approach

---

## Key URLs

| Resource | URL |
|----------|-----|
| Neuronpedia Circuit Tracer UI | https://www.neuronpedia.org/gemma-2-2b/graph |
| API documentation | https://www.neuronpedia.org/api-doc |
| Graph generation example notebook | https://github.com/hijohnnylin/neuronpedia/blob/main/packages/python/neuronpedia-webapp-client/neuronpedia/examples/generate_graph.ipynb |
| Attribution graphs paper | https://transformer-circuits.pub/2025/attribution-graphs/methods.html |
| Neuronpedia Slack | https://join.slack.com/t/opensourcemechanistic |
