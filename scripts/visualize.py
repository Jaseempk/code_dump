"""
Phase 6: Visualization

Creates interactive visualizations for cluster analysis.

Usage:
    python visualize.py
"""

import json
from pathlib import Path
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
VIZ_DIR = Path(__file__).parent.parent / "viz"
PROJECTIONS_FILE = DATA_DIR / "umap_projections.parquet"
CLUSTERS_FILE = DATA_DIR / "clusters_labeled.json"
TEMPORAL_FILE = DATA_DIR / "temporal_trends.json"


def create_cluster_map(projections: pl.DataFrame, clusters: dict) -> go.Figure:
    """Create 2D UMAP cluster visualization."""
    print("  Creating cluster map...")

    # Create label mapping
    label_map = {}
    for c in clusters["clusters"][:50]:  # Top 50 clusters
        label_map[c["cluster_id"]] = c.get("claude_label", f"Cluster {c['cluster_id']}")

    # Add labels to projections
    df = projections.to_pandas()
    df["label"] = df["cluster_label"].apply(
        lambda x: label_map.get(x, "Other") if x != -1 else "Noise"
    )

    # Sample for performance (max 50k points)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    # Create scatter plot
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="label",
        title="Topic Clusters - Running Community Discussions",
        hover_data=["cluster_label"],
        opacity=0.6,
    )

    fig.update_layout(
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
    )

    fig.update_traces(marker=dict(size=3))

    return fig


def create_problem_heatmap(clusters: dict) -> go.Figure:
    """Create heatmap of problem signals by cluster."""
    print("  Creating problem heatmap...")

    # Get top 20 clusters by problem score
    sorted_clusters = sorted(
        clusters["clusters"],
        key=lambda x: x.get("problem_data", {}).get("avg_problem_score", 0),
        reverse=True
    )[:20]

    labels = [c.get("claude_label", f"Cluster {c['cluster_id']}")[:30] for c in sorted_clusters]
    problem_scores = [c.get("problem_data", {}).get("avg_problem_score", 0) for c in sorted_clusters]
    frustration = [c.get("problem_data", {}).get("frustration_percentage", 0) for c in sorted_clusters]
    high_value = [c.get("problem_data", {}).get("high_value_percentage", 0) for c in sorted_clusters]
    sizes = [c.get("size", 0) for c in sorted_clusters]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Problem Score", "Frustration %", "High-Value %", "Cluster Size"],
        shared_yaxes=True,
    )

    # Problem score
    fig.add_trace(
        go.Bar(y=labels, x=problem_scores, orientation='h', name="Problem Score", marker_color='#ef4444'),
        row=1, col=1
    )

    # Frustration
    fig.add_trace(
        go.Bar(y=labels, x=frustration, orientation='h', name="Frustration %", marker_color='#f97316'),
        row=1, col=2
    )

    # High-value
    fig.add_trace(
        go.Bar(y=labels, x=high_value, orientation='h', name="High-Value %", marker_color='#22c55e'),
        row=1, col=3
    )

    # Size
    fig.add_trace(
        go.Bar(y=labels, x=sizes, orientation='h', name="Size", marker_color='#3b82f6'),
        row=1, col=4
    )

    fig.update_layout(
        title="Top 20 Problem Clusters - Pain Point Analysis",
        height=700,
        width=1400,
        showlegend=False,
    )

    return fig


def create_temporal_chart(clusters: dict, temporal: dict) -> go.Figure:
    """Create temporal trend visualization."""
    print("  Creating temporal trends chart...")

    # Get clusters with significant trends
    growing = []
    declining = []

    for c in clusters["clusters"][:100]:
        ts = c.get("temporal", {}).get("trend_stats", {})
        trend = ts.get("trend", "stable")
        growth = ts.get("growth_rate", 0)

        if trend in ["growing", "rapidly_growing"]:
            growing.append((c.get("claude_label", f"Cluster {c['cluster_id']}"), growth * 100))
        elif trend in ["declining", "dying"]:
            declining.append((c.get("claude_label", f"Cluster {c['cluster_id']}"), growth * 100))

    # Sort
    growing.sort(key=lambda x: x[1], reverse=True)
    declining.sort(key=lambda x: x[1])

    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Growing Topics", "Declining Topics"],
    )

    # Growing
    if growing:
        labels, values = zip(*growing[:10])
        fig.add_trace(
            go.Bar(x=list(values), y=list(labels), orientation='h', marker_color='#22c55e', name="Growing"),
            row=1, col=1
        )

    # Declining
    if declining:
        labels, values = zip(*declining[:10])
        fig.add_trace(
            go.Bar(x=list(values), y=list(labels), orientation='h', marker_color='#ef4444', name="Declining"),
            row=1, col=2
        )

    fig.update_layout(
        title="Topic Trends - Growing vs Declining",
        height=500,
        width=1200,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=2)

    return fig


def create_priority_matrix(clusters: dict) -> go.Figure:
    """Create priority matrix visualization."""
    print("  Creating priority matrix...")

    # Prepare data
    data = []
    for c in clusters["clusters"][:50]:
        pd = c.get("problem_data", {})
        ts = c.get("temporal", {}).get("trend_stats", {})

        data.append({
            "label": c.get("claude_label", f"Cluster {c['cluster_id']}")[:25],
            "problem_score": pd.get("avg_problem_score", 0),
            "growth_rate": ts.get("growth_rate", 0) * 100,
            "size": c.get("size", 0),
            "priority": c.get("priority_tier", "P2-Monitor"),
        })

    df = pl.DataFrame(data).to_pandas()

    # Color by priority
    color_map = {
        "P0-Critical": "#dc2626",
        "P1-Important": "#f97316",
        "P2-Monitor": "#3b82f6",
        "P3-Deprioritize": "#9ca3af",
    }

    fig = px.scatter(
        df,
        x="growth_rate",
        y="problem_score",
        size="size",
        color="priority",
        hover_name="label",
        color_discrete_map=color_map,
        title="Priority Matrix: Problem Score vs Growth Rate",
        labels={
            "growth_rate": "Growth Rate (%)",
            "problem_score": "Problem Score",
        },
    )

    # Add quadrant lines
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    fig.add_annotation(x=10, y=8, text="🎯 HIGH PRIORITY", showarrow=False, font=dict(size=12, color="green"))
    fig.add_annotation(x=-10, y=8, text="⚠️ ADDRESS NOW", showarrow=False, font=dict(size=12, color="orange"))
    fig.add_annotation(x=10, y=2, text="📈 MONITOR", showarrow=False, font=dict(size=12, color="blue"))
    fig.add_annotation(x=-10, y=2, text="⏸️ DEPRIORITIZE", showarrow=False, font=dict(size=12, color="gray"))

    fig.update_layout(
        width=1000,
        height=700,
    )

    return fig


def main():
    print("=" * 60)
    print("Phase 6: Visualization")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Create viz directory
    VIZ_DIR.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")

    # Check if labeled clusters exist, otherwise use prioritized
    if CLUSTERS_FILE.exists():
        with open(CLUSTERS_FILE, "r") as f:
            clusters = json.load(f)
    else:
        with open(DATA_DIR / "clusters_prioritized.json", "r") as f:
            clusters = json.load(f)
    print(f"  Loaded {len(clusters['clusters'])} clusters")

    projections = pl.read_parquet(str(PROJECTIONS_FILE))
    print(f"  Loaded {projections.height:,} projections")

    with open(TEMPORAL_FILE, "r") as f:
        temporal = json.load(f)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Cluster map
    fig1 = create_cluster_map(projections, clusters)
    fig1.write_html(str(VIZ_DIR / "cluster_map.html"))
    print(f"  Saved cluster_map.html")

    # 2. Problem heatmap
    fig2 = create_problem_heatmap(clusters)
    fig2.write_html(str(VIZ_DIR / "problem_heatmap.html"))
    print(f"  Saved problem_heatmap.html")

    # 3. Temporal trends
    fig3 = create_temporal_chart(clusters, temporal)
    fig3.write_html(str(VIZ_DIR / "temporal_trends.html"))
    print(f"  Saved temporal_trends.html")

    # 4. Priority matrix
    fig4 = create_priority_matrix(clusters)
    fig4.write_html(str(VIZ_DIR / "priority_matrix.html"))
    print(f"  Saved priority_matrix.html")

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 60)
    print("Phase 6 Complete!")
    print("=" * 60)
    print(f"\nVisualizations saved to: {VIZ_DIR}")


if __name__ == "__main__":
    main()
