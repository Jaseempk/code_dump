"""
Phase 3: Problem Signal Detection

Detects questions, frustration signals, and problem patterns in the data.
Enriches clusters with problem scores for prioritization.

Usage:
    python problem_detection.py
"""

import re
import json
from pathlib import Path
import polars as pl
import numpy as np
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "unified_data.parquet"
CLUSTERS_FILE = DATA_DIR / "clusters.json"
PROJECTIONS_FILE = DATA_DIR / "umap_projections.parquet"

# Question patterns (regex)
QUESTION_PATTERNS = [
    r'\?',                                  # Direct questions
    r'(?i)\bhow (do|can|should|did|would) I\b',  # How-to questions
    r'(?i)\banyone (else|know|have|tried|experience)\b',  # Community validation
    r'(?i)\bhelp (with|me|please)\b',       # Help requests
    r'(?i)\bwhat (is|are|should|would|does)\b',  # Clarification questions
    r'(?i)\bis it (normal|okay|ok|bad|good|safe)\b',  # Validation seeking
    r'(?i)\btips (for|on|about)\b',         # Advice seeking
    r'(?i)\badvice (for|on|about)\b',       # Advice seeking
    r'(?i)\brecommend\b',                   # Recommendations
    r'(?i)\bsuggestions?\b',                # Suggestions
    r'(?i)\bshould I\b',                    # Decision questions
    r'(?i)\bwhy (do|does|is|am|are)\b',     # Why questions
    r'(?i)\bwhen (should|do|does|is)\b',    # When questions
]

# Frustration/problem signals
FRUSTRATION_SIGNALS = [
    # Struggle words
    "struggling", "struggle", "can't", "cannot", "unable", "failing", "failed",
    "stuck", "difficult", "hard time", "trouble", "problem", "issue",
    # Negative emotions
    "hate", "frustrated", "frustrating", "annoying", "annoyed", "sucks",
    "worst", "terrible", "awful", "painful", "hurts", "pain", "injury",
    "injured", "sore", "ache", "aching",
    # Seeking validation
    "anyone else", "is it just me", "am i the only", "does anyone",
    "has anyone", "normal to", "is this normal",
    # Wish/want patterns
    "wish i could", "wish there was", "if only", "i want to but",
    # Confusion
    "confused", "confusing", "don't understand", "makes no sense",
    "lost", "no idea", "clueless",
    # Fear/anxiety
    "worried", "anxious", "scared", "nervous", "afraid", "fear",
    # Failure indicators
    "gave up", "quit", "stopped", "can't seem to", "keep failing",
    "not working", "doesn't work", "won't work",
]

# High-value problem indicators (stronger signals)
HIGH_VALUE_SIGNALS = [
    "beginner", "new to running", "just started", "first time",
    "overweight", "obese", "heavy", "out of shape",
    "motivation", "lazy", "procrastinate", "excuse",
    "shin splints", "knee pain", "plantar", "it band", "achilles",
    "breathing", "out of breath", "can't breathe", "side stitch",
    "slow", "too slow", "getting faster", "improve pace",
    "weight loss", "lose weight", "burn calories",
    "couch to 5k", "c25k", "week 1", "week 2", "week 3",
]


def count_pattern_matches(text: str, patterns: list) -> int:
    """Count how many patterns match in the text."""
    count = 0
    text_lower = text.lower()
    for pattern in patterns:
        if isinstance(pattern, str):
            if pattern in text_lower:
                count += 1
        else:
            # Regex pattern
            if re.search(pattern, text):
                count += 1
    return count


def detect_questions(texts: list) -> tuple:
    """Detect question patterns in texts."""
    is_question = []
    question_counts = []

    for text in texts:
        count = count_pattern_matches(text, QUESTION_PATTERNS)
        is_question.append(count > 0)
        question_counts.append(count)

    return is_question, question_counts


def detect_frustration(texts: list) -> tuple:
    """Detect frustration signals in texts."""
    frustration_counts = []
    high_value_counts = []

    for text in texts:
        frust_count = count_pattern_matches(text, FRUSTRATION_SIGNALS)
        hv_count = count_pattern_matches(text, HIGH_VALUE_SIGNALS)
        frustration_counts.append(frust_count)
        high_value_counts.append(hv_count)

    return frustration_counts, high_value_counts


def calculate_problem_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate problem scores for each chunk."""
    print("  Detecting question patterns...")
    is_question, question_counts = detect_questions(df["text"].to_list())

    print("  Detecting frustration signals...")
    frustration_counts, high_value_counts = detect_frustration(df["text"].to_list())

    # Add columns
    df = df.with_columns([
        pl.Series("is_question", is_question),
        pl.Series("question_count", question_counts),
        pl.Series("frustration_count", frustration_counts),
        pl.Series("high_value_count", high_value_counts),
    ])

    # Normalize engagement score (0-1)
    max_engagement = df["engagement_score"].max()
    if max_engagement and max_engagement > 0:
        df = df.with_columns([
            (pl.col("engagement_score") / max_engagement).alias("engagement_normalized")
        ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("engagement_normalized")
        ])

    # Calculate composite problem score
    df = df.with_columns([
        (
            pl.col("question_count") * 1.0 +
            pl.col("frustration_count") * 2.0 +
            pl.col("high_value_count") * 3.0 +
            pl.col("engagement_normalized") * 0.5
        ).alias("problem_score")
    ])

    return df


def enrich_clusters(df: pl.DataFrame, clusters: dict, projections: pl.DataFrame) -> dict:
    """Enrich clusters with problem signal data."""
    print("\n  Enriching clusters with problem data...")

    # Merge cluster labels into df
    label_map = dict(zip(projections["id"].to_list(), projections["cluster_label"].to_list()))
    df = df.with_columns([
        pl.col("id").replace(label_map, default=-1).alias("cluster_label")
    ])

    enriched_clusters = []

    for cluster in clusters["clusters"]:
        cluster_id = cluster["cluster_id"]

        # Get cluster data
        cluster_df = df.filter(pl.col("cluster_label") == cluster_id)

        if cluster_df.height == 0:
            cluster["problem_data"] = {
                "avg_problem_score": 0,
                "question_percentage": 0,
                "frustration_percentage": 0,
                "high_value_percentage": 0,
                "top_problem_chunks": [],
            }
            enriched_clusters.append(cluster)
            continue

        # Calculate aggregates
        avg_problem_score = cluster_df["problem_score"].mean()
        question_pct = cluster_df.filter(pl.col("is_question"))["is_question"].len() / cluster_df.height * 100
        frustration_pct = cluster_df.filter(pl.col("frustration_count") > 0).height / cluster_df.height * 100
        high_value_pct = cluster_df.filter(pl.col("high_value_count") > 0).height / cluster_df.height * 100

        # Get top problem chunks
        top_chunks = cluster_df.sort("problem_score", descending=True).head(5)
        top_problem_chunks = [
            {
                "text": row["text"][:500],
                "problem_score": row["problem_score"],
                "engagement": row["engagement_score"],
            }
            for row in top_chunks.iter_rows(named=True)
        ]

        cluster["problem_data"] = {
            "avg_problem_score": float(avg_problem_score) if avg_problem_score else 0,
            "question_percentage": float(question_pct),
            "frustration_percentage": float(frustration_pct),
            "high_value_percentage": float(high_value_pct),
            "top_problem_chunks": top_problem_chunks,
        }

        enriched_clusters.append(cluster)

    clusters["clusters"] = enriched_clusters
    return clusters


def main():
    print("=" * 60)
    print("Phase 3: Problem Signal Detection")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    df = pl.read_parquet(str(INPUT_FILE))
    print(f"  Loaded {df.height:,} chunks")

    # Load clusters
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
    print(f"  Loaded {len(clusters['clusters'])} clusters")

    # Load projections (for cluster labels)
    projections = pl.read_parquet(str(PROJECTIONS_FILE))

    # Calculate problem scores
    print("\nCalculating problem scores...")
    df = calculate_problem_scores(df)

    # Statistics
    print("\n" + "=" * 60)
    print("Problem Detection Statistics")
    print("=" * 60)

    total_questions = df.filter(pl.col("is_question")).height
    total_frustrated = df.filter(pl.col("frustration_count") > 0).height
    total_high_value = df.filter(pl.col("high_value_count") > 0).height

    print(f"\nQuestion chunks: {total_questions:,} ({total_questions/df.height*100:.1f}%)")
    print(f"Frustration chunks: {total_frustrated:,} ({total_frustrated/df.height*100:.1f}%)")
    print(f"High-value chunks: {total_high_value:,} ({total_high_value/df.height*100:.1f}%)")

    print(f"\nProblem score distribution:")
    print(f"  Min: {df['problem_score'].min():.2f}")
    print(f"  Max: {df['problem_score'].max():.2f}")
    print(f"  Mean: {df['problem_score'].mean():.2f}")
    print(f"  Median: {df['problem_score'].median():.2f}")

    # Enrich clusters
    clusters = enrich_clusters(df, clusters, projections)

    # Sort clusters by problem score
    clusters["clusters"].sort(
        key=lambda x: x["problem_data"]["avg_problem_score"],
        reverse=True
    )

    # Save enriched data
    print("\nSaving results...")

    # Save enriched clusters
    with open(DATA_DIR / "clusters_enriched.json", "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"  Saved enriched clusters to clusters_enriched.json")

    # Save problem scores to parquet (for temporal analysis)
    problem_df = df.select([
        "id", "source", "channel", "timestamp", "month_bucket",
        "is_question", "question_count", "frustration_count",
        "high_value_count", "problem_score", "engagement_score"
    ])
    problem_df.write_parquet(str(DATA_DIR / "problem_scores.parquet"))
    print(f"  Saved problem scores to problem_scores.parquet")

    # Top problem clusters
    print("\n" + "=" * 60)
    print("Top 15 High-Problem Clusters")
    print("=" * 60)

    for i, cluster in enumerate(clusters["clusters"][:15]):
        pd = cluster["problem_data"]
        print(f"\n{i+1}. Cluster {cluster['cluster_id']}: {cluster['size']:,} chunks")
        print(f"   Keywords: {', '.join(cluster['keywords'][:5])}")
        print(f"   Avg Problem Score: {pd['avg_problem_score']:.2f}")
        print(f"   Questions: {pd['question_percentage']:.1f}%")
        print(f"   Frustration: {pd['frustration_percentage']:.1f}%")
        print(f"   High-Value: {pd['high_value_percentage']:.1f}%")

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
