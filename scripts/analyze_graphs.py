"""Reproducible analysis of saved negation attribution graphs.

The files in graphs/*.json are *metadata* (id, slug, url, json_url, prompt) written
by generate_graph.py. The actual attribution graph — nodes and links — lives at each
metadata's `json_url` (a public S3 file on Neuronpedia). This script fetches those
(caching to graphs/full/), extracts the feature nodes, and reproduces the three
analyses recorded in experiments/log.md:

  1. per-graph summary  — predicted output token + top features by influence
  2. cross-domain overlap — features shared across the negation graphs
  3. negation vs control — features present in negation graphs but absent in controls

Graph JSON schema (confirmed against a live Neuronpedia graph):
  top-level: {"metadata", "qParams", "nodes", "links"}
  node: node_id, feature (int), layer (str), ctx_idx (int),
        feature_type ("cross layer transcoder" | "mlp reconstruction error" |
                      "embedding" | "logit"),
        activation (float | None), influence (float), clerp (str), token_prob (float)

Ranking defaults to `activation` (this reproduces experiments/log.md). Freshly
generated graphs carry per-feature activations; some publicly-served graph JSONs
have them nulled, in which case ranking falls back to `influence`. Pass
--by influence to force influence ranking.

Usage:
  python scripts/analyze_graphs.py                # all analyses, rank by influence
  python scripts/analyze_graphs.py --top-n 50     # match the log's top-50 window
  python scripts/analyze_graphs.py --no-fetch     # only use already-cached graphs
"""

import argparse
import glob
import json
import os
import urllib.request
from collections import defaultdict

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; only needed if json_url requires the key
    pass

CACHE_DIR = "graphs/full"
FEATURE_TYPE = "cross layer transcoder"


# --------------------------------------------------------------------------- IO


def load_metadata(graphs_dir="graphs"):
    """Load metadata files written by generate_graph.py (skips the cache subdir)."""
    metas = {}
    for path in sorted(glob.glob(os.path.join(graphs_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as fh:
            metas[name] = json.load(fh)
    return metas


def fetch_graph_json(meta, fetch=True):
    """Return the full graph JSON for a metadata entry, caching to graphs/full/.

    Reads from cache if present. Otherwise downloads meta['json_url']. Returns None
    if the graph cannot be obtained (missing url, offline, or --no-fetch).
    """
    slug = meta.get("slug") or meta.get("id") or "graph"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{slug}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)

    url = meta.get("json_url")
    if not url or not fetch:
        return None

    headers = {"User-Agent": "negation-circuits/1.0"}
    api_key = os.environ.get("NP_API_KEY")
    if api_key and "YOUR_" not in api_key:
        headers["x-api-key"] = api_key
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        # Some graph files are served gzip-compressed (magic bytes 1f 8b).
        if raw[:2] == b"\x1f\x8b":
            import gzip

            raw = gzip.decompress(raw)
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001 — surface, don't crash the whole run
        print(f"  ! could not fetch graph for '{slug}': {e}")
        return None

    with open(cache_path, "w") as fh:
        json.dump(data, fh)
    return data


# ---------------------------------------------------------------- graph parsing


def feature_nodes(graph):
    """Extract cross-layer-transcoder feature nodes as normalised dicts.

    key is 'L{layer}/{feature}' — the identifier used throughout experiments/log.md.
    """
    out = []
    for n in graph.get("nodes", []):
        if n.get("feature_type") != FEATURE_TYPE:
            continue
        layer = n.get("layer")
        feature = n.get("feature")
        if layer is None or feature is None:
            continue
        out.append(
            {
                "key": f"L{int(layer)}/{feature}",
                "layer": int(layer),
                "feature": feature,
                "ctx_idx": n.get("ctx_idx"),
                "activation": n.get("activation"),
                "influence": n.get("influence") or 0.0,
            }
        )
    return out


def predicted_output(graph):
    """Return (token, prob) of the highest-probability logit node, or (None, None)."""
    best = None
    for n in graph.get("nodes", []):
        if n.get("feature_type") != "logit":
            continue
        prob = n.get("token_prob") or 0.0
        if best is None or prob > best[1]:
            token = (n.get("clerp") or "").replace("Output ", "").strip()
            best = (token, prob)
    return best or (None, None)


def top_feature_keys(graph, by="influence", top_n=50):
    """Return the set of feature keys ranked in the top-N by `by` (influence/activation).

    Falls back to influence if activation is requested but null throughout.
    """
    feats = feature_nodes(graph)
    if by == "activation" and not any(f["activation"] for f in feats):
        by = "influence"
    ranked = sorted(feats, key=lambda f: (f.get(by) or 0.0), reverse=True)
    # dedupe on key, preserving best rank
    seen, keys = set(), []
    for f in ranked:
        if f["key"] not in seen:
            seen.add(f["key"])
            keys.append(f["key"])
        if len(keys) >= top_n:
            break
    return keys


# ------------------------------------------------------------------ classifying


def is_negation(name):
    return name.startswith(("negation", "negform"))


def is_control(name):
    return name.startswith("control")


# -------------------------------------------------------------------- analyses


def summarize(graphs, by, top_n):
    rows = []
    for name, g in graphs.items():
        token, prob = predicted_output(g)
        feats = feature_nodes(g)
        top = top_feature_keys(g, by=by, top_n=3)
        rows.append(
            {
                "name": name,
                "predicted": token,
                "p": round(prob, 3) if prob is not None else None,
                "n_features": len(feats),
                f"top3_by_{by}": ", ".join(top),
            }
        )
    return pd.DataFrame(rows)


def cross_domain(graphs, by, top_n):
    """Count, for each feature, how many negation graphs have it in their top-N."""
    neg = {n: g for n, g in graphs.items() if is_negation(n)}
    if not neg:
        return pd.DataFrame(), 0
    counts = defaultdict(int)
    for g in neg.values():
        for key in top_feature_keys(g, by=by, top_n=top_n):
            counts[key] += 1
    rows = [{"feature": k, "n_negation_graphs": c} for k, c in counts.items()]
    df = pd.DataFrame(rows).sort_values(
        ["n_negation_graphs", "feature"], ascending=[False, True]
    )
    return df, len(neg)


def negation_vs_control(graphs, by, top_n):
    """Presence of each feature across negation vs control graphs, with ratio."""
    neg = {n: g for n, g in graphs.items() if is_negation(n)}
    ctrl = {n: g for n, g in graphs.items() if is_control(n)}
    if not neg or not ctrl:
        return pd.DataFrame(), len(neg), len(ctrl)

    def presence(group):
        c = defaultdict(int)
        for g in group.values():
            for key in top_feature_keys(g, by=by, top_n=top_n):
                c[key] += 1
        return c

    nc, cc = presence(neg), presence(ctrl)
    rows = []
    for key in set(nc) | set(cc):
        n_neg, n_ctrl = nc.get(key, 0), cc.get(key, 0)
        rows.append(
            {
                "feature": key,
                "negation": f"{n_neg}/{len(neg)}",
                "control": f"{n_ctrl}/{len(ctrl)}",
                "neg_frac": n_neg / len(neg),
                # +epsilon avoids div-by-zero; inf => fully negation-specific
                "ratio": float("inf") if n_ctrl == 0 and n_neg > 0
                else round((n_neg / len(neg)) / (n_ctrl / len(ctrl)), 2)
                if n_ctrl
                else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    # negation-specific / enriched first: high neg fraction, high ratio
    df = df.sort_values(["neg_frac", "ratio"], ascending=[False, False])
    return df, len(neg), len(ctrl)


# ------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # activation reproduces experiments/log.md; falls back to influence if a graph's
    # activations are null (as in some publicly-served graph JSONs).
    ap.add_argument("--by", choices=["influence", "activation"], default="activation")
    ap.add_argument("--top-n", type=int, default=50, help="ranking window (log used 50)")
    ap.add_argument("--no-fetch", action="store_true", help="use cached graphs only")
    ap.add_argument("--graphs-dir", default="graphs")
    args = ap.parse_args()

    metas = load_metadata(args.graphs_dir)
    if not metas:
        print(f"No metadata in {args.graphs_dir}/. Run scripts/generate_graph.py first.")
        return

    print(f"Loading {len(metas)} graph(s)...")
    graphs = {}
    for name, meta in metas.items():
        g = fetch_graph_json(meta, fetch=not args.no_fetch)
        if g is not None:
            graphs[name] = g
    if not graphs:
        print("No graph JSON available (offline and nothing cached).")
        return
    print(f"Analysing {len(graphs)} graph(s), ranking by {args.by}, top-{args.top_n}.\n")

    print("=" * 70)
    print("1. PER-GRAPH SUMMARY")
    print("=" * 70)
    print(summarize(graphs, args.by, args.top_n).to_string(index=False))

    print("\n" + "=" * 70)
    print("2. CROSS-DOMAIN OVERLAP (features by # of negation graphs in top-N)")
    print("=" * 70)
    cd, n_neg = cross_domain(graphs, args.by, args.top_n)
    if cd.empty:
        print("No negation graphs found (names starting with negation*/negform*).")
    else:
        shared = cd[cd["n_negation_graphs"] == n_neg]
        print(f"Features in ALL {n_neg} negation graphs: {len(shared)}")
        print(cd.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("3. NEGATION vs CONTROL")
    print("=" * 70)
    nvc, n_neg, n_ctrl = negation_vs_control(graphs, args.by, args.top_n)
    if nvc.empty:
        print("Need both negation* and control* graphs to compare.")
    else:
        specific = nvc[(nvc["ratio"] == float("inf")) & (nvc["neg_frac"] >= 0.6)]
        print(f"Negation-specific (>=60% of negation graphs, 0 controls): {len(specific)}")
        print(nvc.head(15).to_string(index=False))

    # persist a reproducible artifact next to the lab notebook
    out = "experiments/analysis_summary.md"
    os.makedirs("experiments", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"# Analysis summary (auto-generated)\n\n")
        fh.write(f"Ranked by **{args.by}**, top-{args.top_n}. "
                 f"{len(graphs)} graphs.\n\n")
        fh.write("## Per-graph\n\n")
        fh.write(summarize(graphs, args.by, args.top_n).to_markdown(index=False))
        if not cd.empty:
            fh.write("\n\n## Cross-domain overlap\n\n")
            fh.write(cd.head(20).to_markdown(index=False))
        if not nvc.empty:
            fh.write("\n\n## Negation vs control\n\n")
            fh.write(nvc.head(20).to_markdown(index=False))
        fh.write("\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
