"""
Phase 2: Topic Clustering

Applies UMAP dimensionality reduction and HDBSCAN clustering
to discover natural topic groupings in the data.

Usage:
    python clustering.py
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "unified_data.parquet"

# Config
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_COMPONENTS_CLUSTER = 50  # For clustering
UMAP_COMPONENTS_VIZ = 2       # For visualization

HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = 20


def load_embeddings(df: pl.DataFrame) -> np.ndarray:
    """Extract embeddings as numpy array."""
    print("  Converting embeddings to numpy array...")
    embeddings = np.array(df["embedding"].to_list())
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def run_umap(embeddings: np.ndarray, n_components: int, purpose: str) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    print(f"\n  Running UMAP ({purpose})...")
    print(f"    Input dims: {embeddings.shape[1]}")
    print(f"    Output dims: {n_components}")
    print(f"    n_neighbors: {UMAP_N_NEIGHBORS}")
    print(f"    min_dist: {UMAP_MIN_DIST}")

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=n_components,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42,
        verbose=True
    )

    projections = reducer.fit_transform(embeddings)
    print(f"    Output shape: {projections.shape}")
    return projections


def run_hdbscan(projections: np.ndarray) -> np.ndarray:
    """Run HDBSCAN clustering."""
    print(f"\n  Running HDBSCAN clustering...")
    print(f"    min_cluster_size: {HDBSCAN_MIN_CLUSTER_SIZE}")
    print(f"    min_samples: {HDBSCAN_MIN_SAMPLES}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    labels = clusterer.fit_predict(projections)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = (labels == -1).sum()
    noise_pct = noise_count / len(labels) * 100

    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points: {noise_count:,} ({noise_pct:.1f}%)")

    return labels


def extract_cluster_keywords(df: pl.DataFrame, labels: np.ndarray, top_n: int = 20) -> dict:
    """Extract top TF-IDF keywords for each cluster."""
    print("\n  Extracting cluster keywords...")

    cluster_keywords = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        # Get texts for this cluster
        mask = labels == label
        texts = df.filter(pl.lit(mask))["text"].to_list()

        if len(texts) < 10:
            continue

        # TF-IDF on cluster texts
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Get average TF-IDF scores
            avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()

            # Top keywords
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]

            cluster_keywords[int(label)] = keywords
        except Exception as e:
            print(f"    Warning: Could not extract keywords for cluster {label}: {e}")
            cluster_keywords[int(label)] = []

    return cluster_keywords


def get_representative_samples(df: pl.DataFrame, labels: np.ndarray,
                               projections: np.ndarray, n_samples: int = 10) -> dict:
    """Get representative samples closest to each cluster centroid."""
    print("\n  Getting representative samples...")

    cluster_samples = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        # Get cluster points
        mask = labels == label
        cluster_projections = projections[mask]
        cluster_texts = df.filter(pl.lit(mask))["text"].to_list()

        if len(cluster_texts) < n_samples:
            cluster_samples[int(label)] = cluster_texts
            continue

        # Calculate centroid
        centroid = cluster_projections.mean(axis=0)

        # Find closest points to centroid
        distances = np.linalg.norm(cluster_projections - centroid, axis=1)
        closest_indices = distances.argsort()[:n_samples]

        samples = [cluster_texts[i] for i in closest_indices]
        cluster_samples[int(label)] = samples

    return cluster_samples


def build_cluster_metadata(df: pl.DataFrame, labels: np.ndarray,
                           keywords: dict, samples: dict) -> list:
    """Build metadata for each cluster."""
    print("\n  Building cluster metadata...")

    clusters = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        cluster_df = df.filter(pl.lit(mask))

        # Source distribution
        source_counts = cluster_df.group_by("channel").agg(
            pl.len().alias("count")
        ).sort("count", descending=True)

        sources = {row["channel"]: row["count"] for row in source_counts.iter_rows(named=True)}

        # Average engagement
        avg_engagement = cluster_df["engagement_score"].mean()

        clusters.append({
            "cluster_id": int(label),
            "size": int(mask.sum()),
            "label": f"Cluster {label}",  # Placeholder, will be updated by Claude
            "keywords": keywords.get(int(label), []),
            "representative_samples": samples.get(int(label), []),
            "sources": sources,
            "avg_engagement": float(avg_engagement) if avg_engagement else 0.0,
        })

    # Sort by size
    clusters.sort(key=lambda x: x["size"], reverse=True)

    return clusters


def main():
    print("=" * 60)
    print("Phase 2: Topic Clustering")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    df = pl.read_parquet(str(INPUT_FILE))
    print(f"  Loaded {df.height:,} chunks")

    # Extract embeddings
    embeddings = load_embeddings(df)

    # UMAP for clustering (768 -> 50 dims)
    projections_50d = run_umap(embeddings, UMAP_COMPONENTS_CLUSTER, "clustering")

    # UMAP for visualization (768 -> 2 dims)
    projections_2d = run_umap(embeddings, UMAP_COMPONENTS_VIZ, "visualization")

    # HDBSCAN clustering
    labels = run_hdbscan(projections_50d)

    # Extract keywords
    keywords = extract_cluster_keywords(df, labels)

    # Get representative samples
    samples = get_representative_samples(df, labels, projections_50d)

    # Build cluster metadata
    clusters = build_cluster_metadata(df, labels, keywords, samples)

    # Save results
    print("\nSaving results...")

    # Save projections with labels
    print("  Saving projections...")
    projections_df = pl.DataFrame({
        "id": df["id"],
        "umap_x": projections_2d[:, 0],
        "umap_y": projections_2d[:, 1],
        "cluster_label": labels,
    })
    projections_df.write_parquet(str(DATA_DIR / "umap_projections.parquet"))

    # Save cluster metadata
    print("  Saving cluster metadata...")
    with open(DATA_DIR / "clusters.json", "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_chunks": df.height,
            "total_clusters": len(clusters),
            "noise_count": int((labels == -1).sum()),
            "noise_percentage": float((labels == -1).sum() / len(labels) * 100),
            "clusters": clusters
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total chunks: {df.height:,}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Noise points: {(labels == -1).sum():,} ({(labels == -1).sum() / len(labels) * 100:.1f}%)")

    print("\nTop 10 clusters by size:")
    for i, cluster in enumerate(clusters[:10]):
        print(f"  {i+1}. Cluster {cluster['cluster_id']}: {cluster['size']:,} chunks")
        print(f"      Keywords: {', '.join(cluster['keywords'][:5])}")

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
