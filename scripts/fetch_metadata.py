"""Rebuild local graph metadata files from Neuronpedia by slug.

The graphs generated in experiments 1-5 already live on Neuronpedia under the
account tied to NP_API_KEY. This script fetches their metadata by slug and writes
graphs/<slug>.json in the same schema generate_graph.py produces, so a fresh clone
can run scripts/analyze_graphs.py without regenerating anything.

Usage:
  python scripts/fetch_metadata.py                 # fetch the experiment 1-5 slugs
  python scripts/fetch_metadata.py my-slug other   # fetch specific slugs
"""

import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_ID = "gemma-2-2b"
API = "https://www.neuronpedia.org/api/graph"

# The full dataset behind experiments/log.md (experiments 2-4).
DEFAULT_SLUGS = [
    # negation (cross-domain) — experiment 2
    "negation-geography-germany",
    "negation-physics-water",
    "negation-astronomy-sun",
    "negation-history-shakespeare",
    "negation-tech-python",
    # affirmative controls — experiment 3
    "control-geography-germany",
    "control-physics-water",
    "control-astronomy-sun",
    "control-history-shakespeare",
    "control-tech-python",
    # negation surface forms — experiment 4
    "negform-never-geography",
    "negform-isnt-geography",
    "negform-no-geography",
    "negform-never-physics",
    "negform-isnt-physics",
]


def fetch_metadata(slug, api_key):
    url = f"{API}/{MODEL_ID}/{slug}"
    req = urllib.request.Request(
        url, headers={"x-api-key": api_key, "User-Agent": "negation-circuits/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        m = json.loads(resp.read())
    # Neuronpedia's `url` field is the S3 graph-data JSON; map it to json_url.
    return {
        "id": m.get("id", ""),
        "slug": m.get("slug", slug),
        "url": f"https://www.neuronpedia.org/{MODEL_ID}/graph?slug={slug}",
        "url_embed": "",
        "json_url": m.get("url", ""),
        "model_id": MODEL_ID,
        "prompt": m.get("prompt", ""),
    }


def main():
    api_key = os.environ.get("NP_API_KEY")
    if not api_key or "YOUR_" in api_key:
        print("NP_API_KEY not set — add it to .env (see .env.example).")
        return

    slugs = sys.argv[1:] or DEFAULT_SLUGS
    os.makedirs("graphs", exist_ok=True)
    ok = 0
    for slug in slugs:
        try:
            data = fetch_metadata(slug, api_key)
            path = f"graphs/{slug}.json"
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"  saved {path}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {slug}: {e}")
    print(f"\n{ok}/{len(slugs)} metadata files written to graphs/.")


if __name__ == "__main__":
    main()
