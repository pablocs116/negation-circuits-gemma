import json
import glob
import pandas as pd


def load_graphs():
    graph_files = glob.glob("graphs/*.json")
    graphs = {}
    for f in graph_files:
        name = f.split("/")[-1].replace(".json", "")
        with open(f) as fh:
            graphs[name] = json.load(fh)
    return graphs


def summarize_graphs(graphs):
    rows = []
    for name, data in graphs.items():
        rows.append({
            "name": name,
            "graph_id": data.get("graph_id", ""),
            "url": data.get("url", ""),
            "model_id": data.get("model_id", ""),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    graphs = load_graphs()
    print(f"Loaded {len(graphs)} graphs: {list(graphs.keys())}")
    df = summarize_graphs(graphs)
    print(df.to_string(index=False))
