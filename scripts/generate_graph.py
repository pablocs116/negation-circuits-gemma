import os
import json
import time

from dotenv import load_dotenv
from neuronpedia.np_graph_metadata import NPGraphMetadata

load_dotenv()

MODEL_ID = "gemma-2-2b"


def generate_graph(prompt, slug=None):
    graph_metadata = NPGraphMetadata.generate(
        model_id=MODEL_ID,
        prompt=prompt,
        graph_id=slug or f"negation-{int(time.time() * 1000)}",
    )
    return graph_metadata


def save_graph_metadata(graph_metadata, name):
    os.makedirs("graphs", exist_ok=True)
    path = f"graphs/{name}.json"
    data = {
        "id": graph_metadata.id,
        "slug": graph_metadata.slug,
        "url": graph_metadata.url,
        "url_embed": graph_metadata.url_embed,
        "json_url": graph_metadata.json_url,
        "model_id": MODEL_ID,
        "prompt": graph_metadata.prompt,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")
    print(f"View:  {graph_metadata.url}")
    return data


if __name__ == "__main__":
    prompt = "The capital of France is Paris. The capital of Germany is NOT Paris, it is"
    slug = "negation-geography-germany-01"
    print(f"Generating graph for: {prompt[:60]}...")
    meta = generate_graph(prompt, slug=slug)
    save_graph_metadata(meta, "negation_geography_germany_01")
