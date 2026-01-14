"""
Phase 4: Temporal Trend Analysis

Analyzes how clusters change over time to identify growing vs dying topics.

Usage:
    python temporal_analysis.py
"""

import json
from pathlib import Path
import polars as pl
import numpy as np
from scipy import stats
from datetime import datetime
from collections import defaultdict

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROBLEM_SCORES_FILE = DATA_DIR / "problem_scores.parquet"
CLUSTERS_FILE = DATA_DIR / "clusters_enriched.json"
PROJECTIONS_FILE = DATA_DIR / "umap_projections.parquet"


def calculate_trend(monthly_counts: dict) -> dict:
    """Calculate trend statistics for a time series."""
    if len(monthly_counts) < 3:
        return {
            "trend": "insufficient_data",
            "growth_rate": 0.0,
            "p_value": 1.0,
            "slope": 0.0,
        }

    # Sort by month
    sorted_months = sorted(monthly_counts.keys())
    values = [monthly_counts[m] for m in sorted_months]

    # Convert to numpy
    x = np.arange(len(values))
    y = np.array(values)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate growth rate (slope relative to mean)
    mean_val = np.mean(y)
    if mean_val > 0:
        growth_rate = slope / mean_val
    else:
        growth_rate = 0.0

    # Classify trend
    if p_value > 0.1:  # Not statistically significant
        trend = "stable"
    elif growth_rate > 0.20:
        trend = "rapidly_growing"
    elif growth_rate > 0.05:
        trend = "growing"
    elif growth_rate < -0.20:
        trend = "dying"
    elif growth_rate < -0.05:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "growth_rate": float(growth_rate),
        "p_value": float(p_value),
        "slope": float(slope),
        "r_squared": float(r_value ** 2),
    }


def detect_seasonality(monthly_counts: dict) -> bool:
    """Simple seasonality detection based on month patterns."""
    if len(monthly_counts) < 12:
        return False

    # Group by month-of-year
    month_groups = defaultdict(list)
    for month_key, count in monthly_counts.items():
        try:
            month_num = int(month_key.split("-")[1])
            month_groups[month_num].append(count)
        except:
            continue

    if len(month_groups) < 6:
        return False

    # Calculate variance between month averages
    month_avgs = [np.mean(v) for v in month_groups.values() if v]
    if len(month_avgs) < 6:
        return False

    # If variance is high relative to mean, likely seasonal
    cv = np.std(month_avgs) / np.mean(month_avgs) if np.mean(month_avgs) > 0 else 0
    return cv > 0.3  # 30% coefficient of variation threshold


def analyze_cluster_temporally(cluster_df: pl.DataFrame, cluster_id: int) -> dict:
    """Analyze temporal patterns for a cluster."""
    # Filter to this cluster
    if cluster_df.height == 0:
        return {
            "monthly_counts": {},
            "quarterly_counts": {},
            "first_seen": None,
            "last_seen": None,
            "total_count": 0,
            "trend_stats": {"trend": "no_data"},
            "is_seasonal": False,
        }

    # Count by month
    monthly = cluster_df.filter(
        pl.col("month_bucket").is_not_null()
    ).group_by("month_bucket").agg(
        pl.len().alias("count")
    ).sort("month_bucket")

    monthly_counts = {
        row["month_bucket"]: row["count"]
        for row in monthly.iter_rows(named=True)
    }

    # Count by quarter
    quarterly_counts = defaultdict(int)
    for month, count in monthly_counts.items():
        try:
            year, m = month.split("-")
            quarter = (int(m) - 1) // 3 + 1
            quarterly_counts[f"{year}-Q{quarter}"] += count
        except:
            continue

    # Date range
    sorted_months = sorted(monthly_counts.keys()) if monthly_counts else []
    first_seen = sorted_months[0] if sorted_months else None
    last_seen = sorted_months[-1] if sorted_months else None

    # Recent trend (last 24 months only for trend calculation)
    recent_months = sorted_months[-24:] if len(sorted_months) > 24 else sorted_months
    recent_counts = {m: monthly_counts[m] for m in recent_months}

    # Calculate trend
    trend_stats = calculate_trend(recent_counts)

    # Check seasonality
    is_seasonal = detect_seasonality(monthly_counts)

    return {
        "monthly_counts": monthly_counts,
        "quarterly_counts": dict(quarterly_counts),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "total_count": cluster_df.height,
        "trend_stats": trend_stats,
        "is_seasonal": bool(is_seasonal),  # Convert numpy bool to Python bool
    }


def main():
    print("=" * 60)
    print("Phase 4: Temporal Trend Analysis")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    problem_df = pl.read_parquet(str(PROBLEM_SCORES_FILE))
    print(f"  Loaded {problem_df.height:,} chunks with problem scores")

    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
    print(f"  Loaded {len(clusters['clusters'])} clusters")

    projections = pl.read_parquet(str(PROJECTIONS_FILE))

    # Merge cluster labels
    label_map = dict(zip(projections["id"].to_list(), projections["cluster_label"].to_list()))
    problem_df = problem_df.with_columns([
        pl.col("id").map_elements(lambda x: label_map.get(x, -1), return_dtype=pl.Int64).alias("cluster_label")
    ])

    # Analyze each cluster
    print("\nAnalyzing temporal patterns...")
    temporal_data = {}
    trend_counts = defaultdict(int)

    for i, cluster in enumerate(clusters["clusters"]):
        cluster_id = cluster["cluster_id"]

        # Get cluster data
        cluster_df = problem_df.filter(pl.col("cluster_label") == cluster_id)

        # Analyze temporally
        temporal = analyze_cluster_temporally(cluster_df, cluster_id)
        temporal_data[cluster_id] = temporal

        # Count trends
        trend_counts[temporal["trend_stats"]["trend"]] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(clusters['clusters'])} clusters")

    # Add temporal data to clusters
    for cluster in clusters["clusters"]:
        cluster["temporal"] = temporal_data.get(cluster["cluster_id"], {})

    # Summary statistics
    print("\n" + "=" * 60)
    print("Temporal Analysis Summary")
    print("=" * 60)

    print("\nTrend Distribution:")
    for trend, count in sorted(trend_counts.items(), key=lambda x: -x[1]):
        print(f"  {trend}: {count} clusters")

    # Sort clusters by different criteria
    growing_clusters = sorted(
        [c for c in clusters["clusters"] if c.get("temporal", {}).get("trend_stats", {}).get("trend") in ["growing", "rapidly_growing"]],
        key=lambda x: x.get("temporal", {}).get("trend_stats", {}).get("growth_rate", 0),
        reverse=True
    )

    declining_clusters = sorted(
        [c for c in clusters["clusters"] if c.get("temporal", {}).get("trend_stats", {}).get("trend") in ["declining", "dying"]],
        key=lambda x: x.get("temporal", {}).get("trend_stats", {}).get("growth_rate", 0)
    )

    # Top growing
    print("\n" + "=" * 60)
    print("Top 10 GROWING Clusters")
    print("=" * 60)

    for i, cluster in enumerate(growing_clusters[:10]):
        temp = cluster.get("temporal", {})
        ts = temp.get("trend_stats", {})
        pd = cluster.get("problem_data", {})

        print(f"\n{i+1}. Cluster {cluster['cluster_id']}: {cluster['size']:,} chunks")
        print(f"   Keywords: {', '.join(cluster['keywords'][:5])}")
        print(f"   Trend: {ts.get('trend', 'N/A')} (growth: {ts.get('growth_rate', 0)*100:.1f}%)")
        print(f"   Problem Score: {pd.get('avg_problem_score', 0):.2f}")
        print(f"   Active: {temp.get('first_seen', 'N/A')} → {temp.get('last_seen', 'N/A')}")

    # Top declining
    print("\n" + "=" * 60)
    print("Top 10 DECLINING Clusters")
    print("=" * 60)

    for i, cluster in enumerate(declining_clusters[:10]):
        temp = cluster.get("temporal", {})
        ts = temp.get("trend_stats", {})
        pd = cluster.get("problem_data", {})

        print(f"\n{i+1}. Cluster {cluster['cluster_id']}: {cluster['size']:,} chunks")
        print(f"   Keywords: {', '.join(cluster['keywords'][:5])}")
        print(f"   Trend: {ts.get('trend', 'N/A')} (growth: {ts.get('growth_rate', 0)*100:.1f}%)")
        print(f"   Problem Score: {pd.get('avg_problem_score', 0):.2f}")
        print(f"   Active: {temp.get('first_seen', 'N/A')} → {temp.get('last_seen', 'N/A')}")

    # Save results
    print("\nSaving results...")

    # Save full temporal data
    with open(DATA_DIR / "temporal_trends.json", "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "trend_summary": dict(trend_counts),
            "cluster_temporal_data": temporal_data,
        }, f, indent=2)
    print(f"  Saved temporal trends to temporal_trends.json")

    # Save enriched clusters with temporal data
    with open(DATA_DIR / "clusters_final.json", "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"  Saved final clusters to clusters_final.json")

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 60)
    print("Phase 4 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
