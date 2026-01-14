"""
Phase 2.3: Cluster Labeling with Claude

Uses Claude API to generate human-readable labels for each cluster
based on keywords and representative samples.

Usage:
    python label_clusters.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "clusters_prioritized.json"
OUTPUT_FILE = DATA_DIR / "clusters_labeled.json"

# Config
MAX_CLUSTERS_TO_LABEL = 50  # Only label top clusters to save API calls
MODEL = "claude-3-5-haiku-20241022"


def generate_label(client: anthropic.Anthropic, cluster: dict) -> dict:
    """Generate a human-readable label for a cluster using Claude."""
    keywords = cluster.get("keywords", [])[:15]
    samples = cluster.get("representative_samples", [])[:5]
    size = cluster.get("size", 0)
    problem_data = cluster.get("problem_data", {})

    # Truncate samples
    truncated_samples = [s[:300] + "..." if len(s) > 300 else s for s in samples]

    prompt = f"""Analyze this cluster of running community discussions and provide a concise label.

**Keywords:** {', '.join(keywords)}

**Sample discussions:**
{chr(10).join(f'- {s}' for s in truncated_samples)}

**Stats:**
- Cluster size: {size} discussions
- Frustration level: {problem_data.get('frustration_percentage', 0):.0f}%
- High-value signals: {problem_data.get('high_value_percentage', 0):.0f}%

Based on this data, provide:
1. A short label (2-5 words) that describes the main topic
2. A one-sentence description of what people in this cluster are discussing

Respond in this exact JSON format:
{{"label": "Your Short Label", "description": "One sentence description."}}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        text = response.content[0].text.strip()
        # Extract JSON from response
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}")+1]
            result = json.loads(json_str)
            return {
                "label": result.get("label", cluster.get("topic_label", "Unknown")),
                "description": result.get("description", ""),
            }
    except Exception as e:
        print(f"    Error: {e}")

    return {
        "label": cluster.get("topic_label", "Unknown"),
        "description": "",
    }


def main():
    print("=" * 60)
    print("Phase 2.3: Cluster Labeling with Claude")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Initialize client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)
    print(f"Model: {MODEL}")

    # Load clusters
    print("\nLoading clusters...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    clusters = data["clusters"]
    print(f"  Loaded {len(clusters)} clusters")

    # Label top clusters
    print(f"\nLabeling top {MAX_CLUSTERS_TO_LABEL} clusters...")

    labeled_count = 0
    for i, cluster in enumerate(clusters[:MAX_CLUSTERS_TO_LABEL]):
        print(f"  [{i+1}/{MAX_CLUSTERS_TO_LABEL}] Cluster {cluster['cluster_id']}...", end=" ")

        result = generate_label(client, cluster)
        cluster["claude_label"] = result["label"]
        cluster["claude_description"] = result["description"]

        print(f"→ {result['label']}")
        labeled_count += 1

    # Keep existing labels for remaining clusters
    for cluster in clusters[MAX_CLUSTERS_TO_LABEL:]:
        cluster["claude_label"] = cluster.get("topic_label", "General Discussion")
        cluster["claude_description"] = ""

    # Save
    print("\nSaving labeled clusters...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"clusters": clusters}, f, indent=2)
    print(f"  Saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("Top 15 Labeled Clusters")
    print("=" * 60)

    for i, cluster in enumerate(clusters[:15]):
        print(f"\n{i+1}. {cluster.get('claude_label', 'N/A')}")
        print(f"   {cluster.get('claude_description', 'N/A')}")
        print(f"   Size: {cluster['size']:,} | Priority: {cluster.get('priority_tier', 'N/A')}")

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 60)
    print("Phase 2.3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
