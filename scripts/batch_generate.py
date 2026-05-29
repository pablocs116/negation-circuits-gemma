import time

from generate_graph import generate_graph, save_graph_metadata

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
    results = []
    for prompt, slug in PROMPTS:
        print(f"\n[{slug}]")
        try:
            meta = generate_graph(prompt, slug=slug)
            data = save_graph_metadata(meta, slug)
            results.append((slug, data.get("url", "")))
            time.sleep(2)
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- All graphs ---")
    for slug, url in results:
        print(f"{slug}: {url}")
