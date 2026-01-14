"""
Phase 1: Data Extraction & Preparation

Extracts all data from 6 ChromaDB collections (Reddit + Discord),
unifies schema, cleans, and saves to parquet for analysis.

Usage:
    python extract_data.py
"""

import sys
from pathlib import Path
import polars as pl
import chromadb
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # fitbet root
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# ChromaDB sources
REDDIT_SOURCES = [
    ("r_running", BASE_DIR / "reddit_pipeline" / "r_running" / "chroma_db"),
    ("r_C25K", BASE_DIR / "reddit_pipeline" / "r_C25K" / "chroma_db"),
    ("r_beginnerrunning", BASE_DIR / "reddit_pipeline" / "r_beginnerrunning" / "chroma_db"),
]

DISCORD_SOURCES = [
    ("beginner_running", BASE_DIR / "discord_pipeline" / "beginner_running" / "chroma_db"),
    ("running_questions", BASE_DIR / "discord_pipeline" / "running_questions" / "chroma_db"),
    ("running_science", BASE_DIR / "discord_pipeline" / "running_science" / "chroma_db"),
]

# Config
MIN_CHUNK_LENGTH = 25


def extract_reddit_collection(name: str, chroma_path: Path) -> pl.DataFrame:
    """Extract data from a Reddit ChromaDB collection."""
    print(f"  Extracting {name}...")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=name)

    # Get all data
    results = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    rows = []
    for i, doc_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i]

        # Parse date (normalize to naive datetime for consistency)
        date_str = metadata.get("date", "")
        try:
            if date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                # Convert to naive datetime (remove timezone)
                timestamp = dt.replace(tzinfo=None) if dt.tzinfo else dt
            else:
                timestamp = None
        except:
            timestamp = None

        rows.append({
            "id": doc_id,
            "source": "reddit",
            "channel": name,
            "text": results["documents"][i],
            "embedding": results["embeddings"][i],
            "timestamp": timestamp,
            "engagement_score": float(metadata.get("score", 0)),
            "author": metadata.get("author", ""),
            "post_title": metadata.get("post_title", ""),
            "num_comments": int(metadata.get("num_comments", 0)),
        })

    print(f"    Extracted {len(rows):,} chunks")
    return pl.DataFrame(rows)


def extract_discord_collection(name: str, chroma_path: Path) -> pl.DataFrame:
    """Extract data from a Discord ChromaDB collection."""
    print(f"  Extracting {name}...")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=name)

    # Get all data
    results = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    rows = []
    for i, doc_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i]

        # Parse session_start timestamp (normalize to naive datetime)
        session_start = metadata.get("session_start", "")
        try:
            if session_start:
                dt = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
                # Convert to naive datetime (remove timezone)
                timestamp = dt.replace(tzinfo=None) if dt.tzinfo else dt
            else:
                timestamp = None
        except:
            timestamp = None

        rows.append({
            "id": doc_id,
            "source": "discord",
            "channel": name,
            "text": results["documents"][i],
            "embedding": results["embeddings"][i],
            "timestamp": timestamp,
            "engagement_score": 1.0,  # Discord has no engagement metrics
            "author": metadata.get("user_name", ""),
            "post_title": "",  # Discord doesn't have titles
            "num_comments": 0,
        })

    print(f"    Extracted {len(rows):,} chunks")
    return pl.DataFrame(rows)


def add_time_buckets(df: pl.DataFrame) -> pl.DataFrame:
    """Add month, quarter, and year bucket columns."""
    return df.with_columns([
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month_bucket"),
        pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8).alias("quarter_bucket"),
        pl.col("timestamp").dt.year().cast(pl.Utf8).alias("year_bucket"),
    ])


def main():
    print("=" * 60)
    print("Phase 1: Data Extraction & Preparation")
    print("=" * 60)

    all_dfs = []

    # Extract Reddit data
    print("\nExtracting Reddit collections...")
    for name, path in REDDIT_SOURCES:
        if path.exists():
            df = extract_reddit_collection(name, path)
            all_dfs.append(df)
        else:
            print(f"  WARNING: {path} not found, skipping")

    # Extract Discord data
    print("\nExtracting Discord collections...")
    for name, path in DISCORD_SOURCES:
        if path.exists():
            df = extract_discord_collection(name, path)
            all_dfs.append(df)
        else:
            print(f"  WARNING: {path} not found, skipping")

    # Combine all data
    print("\nCombining data...")
    unified_df = pl.concat(all_dfs)
    print(f"  Total chunks before cleaning: {unified_df.height:,}")

    # Clean data
    print("\nCleaning data...")

    # Remove short chunks
    unified_df = unified_df.filter(pl.col("text").str.len_chars() >= MIN_CHUNK_LENGTH)
    print(f"  After removing chunks < {MIN_CHUNK_LENGTH} chars: {unified_df.height:,}")

    # Remove null timestamps
    null_count = unified_df.filter(pl.col("timestamp").is_null()).height
    print(f"  Chunks with null timestamps: {null_count:,}")

    # Add time buckets (only for non-null timestamps)
    print("\nAdding time buckets...")
    unified_df = add_time_buckets(unified_df)

    # Statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    print(f"\nTotal chunks: {unified_df.height:,}")

    print("\nBy source:")
    for source in ["reddit", "discord"]:
        count = unified_df.filter(pl.col("source") == source).height
        print(f"  {source}: {count:,}")

    print("\nBy channel:")
    channel_counts = unified_df.group_by("channel").agg(pl.len().alias("count")).sort("count", descending=True)
    for row in channel_counts.iter_rows(named=True):
        print(f"  {row['channel']}: {row['count']:,}")

    # Date range
    valid_timestamps = unified_df.filter(pl.col("timestamp").is_not_null())
    if valid_timestamps.height > 0:
        min_date = valid_timestamps.select(pl.col("timestamp").min()).item()
        max_date = valid_timestamps.select(pl.col("timestamp").max()).item()
        print(f"\nDate range: {min_date} to {max_date}")

    # Save
    print("\nSaving data...")
    output_path = OUTPUT_DIR / "unified_data.parquet"

    # Convert embeddings list to proper format for parquet
    # Polars handles list columns natively
    unified_df.write_parquet(str(output_path))
    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
