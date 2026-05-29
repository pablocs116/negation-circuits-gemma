# negation-circuits-gemma — Claude Code Setup Plan

This file tells Claude Code exactly what to do to get the project running.
Execute steps in order. Stop and report any error before continuing.

---

## Context

We are studying how Gemma-2-2b handles negation using mechanistic interpretability
and attribution graphs from Anthropic's open-source circuit-tracer library.
Model: `google/gemma-2-2b`
Transcoder set: `gemma` (resolves to `mwhanna/gemma-scope-transcoders`)
Hardware: Apple M4, 24GB unified RAM, Mac OS

---

## Step 0 — Verify environment

```bash
# Confirm we are inside the project directory
pwd
# Should print something like /Users/pcabriada/Projects/negation-circuits-gemma

# Confirm venv is active
which python
# Should point to .venv/bin/python inside the project

# Confirm Python version (needs 3.10+, NOT 3.14 if circuit-tracer has issues)
python --version
```

If venv is not active:
```bash
source .venv/bin/activate
```

---

## Step 1 — Kill stale kernels and processes

```bash
pkill -f jupyter
pkill -f python
sleep 2
```

---

## Step 2 — Verify HuggingFace token is set

```bash
echo $HF_TOKEN
```

If this prints blank or `hf_YOUR_TOKEN_HERE`:
```bash
# Open zshrc and fix the token manually
nano ~/.zshrc
# Find the HF_TOKEN line, replace with real token
# Save: Ctrl+X → Y → Enter
source ~/.zshrc
echo $HF_TOKEN
# Should now print the real token starting with hf_
```

---

## Step 3 — Verify HuggingFace auth in Python

```bash
python - <<'EOF'
import os
from huggingface_hub import login, whoami
token = os.environ.get("HF_TOKEN")
if not token or token == "hf_YOUR_TOKEN_HERE":
    raise ValueError("HF_TOKEN is not set correctly in environment")
login(token=token)
user = whoami()
print(f"Logged in as: {user['name']}")
EOF
```

Expected output: `Logged in as: YOUR_HF_USERNAME`
If this hangs for more than 30 seconds, there is a network issue — report it.

---

## Step 4 — Verify Gemma-2-2b license is accepted

Go to https://huggingface.co/google/gemma-2-2b and confirm the license
has been accepted under the logged-in account. This must be done manually
in the browser — cannot be automated.

---

## Step 5 — Install dependencies

```bash
pip install circuit-tracer
pip install jupyter notebook matplotlib pandas numpy
pip install transformer-lens
pip freeze > requirements.txt
```

---

## Step 6 — Test model download in Python directly (not notebook)

```bash
python - <<'EOF'
import os
from huggingface_hub import login, whoami
from circuit_tracer import ReplacementModel

# Auth
login(token=os.environ.get("HF_TOKEN"))
print(f"Auth OK: {whoami()['name']}")

# Load model — this will download Gemma-2-2b (~5GB) on first run
# Expected time: 10-30 min on first run, 2-5 min on subsequent runs
print("Loading model... (this will take a while on first run)")
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",
    transcoder_set="gemma",
    device="mps",
    dtype="float16",
)
tokenizer = model.tokenizer
print("Model loaded successfully")

# Test tokenizer
prompt = "The capital of France is Paris. The capital of Germany is NOT Paris, it is"
tokens = tokenizer.encode(prompt)
print(f"Prompt tokenized: {len(tokens)} tokens")
print(tokens)
EOF
```

Watch the terminal — you should see a download progress bar.
If it hangs silently for more than 2 minutes without any output, interrupt
with Ctrl+C and report the error.

---

## Step 7 — Verify MPS (M4 GPU) is available

```bash
python - <<'EOF'
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
EOF
```

Expected: `MPS available: True`

---

## Step 8 — Build project folder structure

```bash
mkdir -p notebooks graphs experiments
touch experiments/log.md

# Create notebook files if they don't exist
[ ! -f notebooks/01_exploration.ipynb ] && touch notebooks/01_exploration.ipynb
[ ! -f notebooks/02_negation_analysis.ipynb ] && touch notebooks/02_negation_analysis.ipynb
[ ! -f notebooks/03_patching_experiments.ipynb ] && touch notebooks/03_patching_experiments.ipynb
```

---

## Step 9 — Scaffold 01_exploration.ipynb

Write the following cells into `notebooks/01_exploration.ipynb` as a valid
Jupyter notebook JSON. Cells in order:

### Cell 1 — Auth
```python
import os
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))
print("Auth OK")
```

### Cell 2 — Imports
```python
import torch
from circuit_tracer import attribute, Graph, ReplacementModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
```

### Cell 3 — Load model
```python
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",
    transcoder_set="gemma",
    device=device,
    dtype="float16",
)
tokenizer = model.tokenizer
print("Model loaded")
```

### Cell 4 — First negation prompts
```python
# Core negation prompt set — Day 1 exploration
prompts = [
    "The capital of France is Paris. The capital of Germany is NOT Paris, it is",
    "Water is NOT a solid at room temperature, it is",
    "The sun does NOT orbit the Earth, the Earth orbits the",
    "Shakespeare was NOT born in London, he was born in",
    "Python is NOT a compiled language, it is a",
]

for p in prompts:
    tokens = tokenizer.encode(p)
    print(f"{len(tokens):3d} tokens | {p[:60]}...")
```

### Cell 5 — First attribution graph
```python
# Run attribution on first prompt — this is the core experiment
prompt = prompts[0]
graph = attribute(
    prompt=prompt,
    model=model,
    verbose=True,
)
print(f"Graph nodes: {len(graph.nodes)}")
print(f"Graph edges: {len(graph.edges)}")
```

---

## Step 10 — First git commit

```bash
git add .
git commit -m "day-1: project scaffold, env setup, first exploration notebook"
git push
```

---

## Ongoing: experiments/log.md format

Every time an experiment is run, append an entry to `experiments/log.md`
in this format:

```markdown
## YYYY-MM-DD

**Prompt:** "..."
**Hypothesis:** ...
**What I ran:** ...
**What I found:** ...
**Surprise / anomaly:** ...
**Next step:** ...
```

---

## Known issues and fixes

| Issue | Fix |
|---|---|
| `waiter.acquire()` hang | HF token not passed — run login() cell first |
| `!echo $HF_TOKEN` blank in notebook | Kernel doesn't inherit shell env — use `os.environ.get()` in Python instead |
| Model loads on CPU not MPS | Pass `device="mps"` explicitly to `from_pretrained` |
| `transcoder_set="gemma"` fails | Replace with `transcoder_set="mwhanna/gemma-scope-transcoders"` |
| Python 3.14 compatibility issues | Downgrade venv to Python 3.11 — `brew install python@3.11` |

---

## Resources

- circuit-tracer repo: https://github.com/safety-research/circuit-tracer
- Attribution graphs paper: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Neuronpedia (feature explorer): https://www.neuronpedia.org
- Gemma-2-2b model card: https://huggingface.co/google/gemma-2-2b
- mwhanna transcoders: https://huggingface.co/mwhanna/gemma-scope-transcoders
