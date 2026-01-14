"""
Topic Analysis Dashboard - Streamlit App

Interactive visualization of running community topic analysis.

Usage:
    streamlit run dashboard.py
"""

import json
from pathlib import Path
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Page config
st.set_page_config(
    page_title="Running Community Insights",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
DATA_DIR = Path(__file__).parent / "data"
CLUSTERS_FILE = DATA_DIR / "clusters_labeled.json"
PROJECTIONS_FILE = DATA_DIR / "umap_projections.parquet"
TEMPORAL_FILE = DATA_DIR / "temporal_trends.json"
PRIORITY_FILE = DATA_DIR / "priority_matrix.csv"


@st.cache_data
def load_data():
    """Load all data files."""
    # Clusters
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)

    # Projections (sample for performance)
    projections = pl.read_parquet(str(PROJECTIONS_FILE))
    if projections.height > 50000:
        projections = projections.sample(n=50000, seed=42)

    # Temporal
    with open(TEMPORAL_FILE, "r") as f:
        temporal = json.load(f)

    # Priority matrix
    priority_df = pd.read_csv(PRIORITY_FILE)

    return clusters, projections, temporal, priority_df


def render_sidebar(clusters: dict):
    """Render sidebar with filters and stats."""
    st.sidebar.title("🏃 Running Community Insights")
    st.sidebar.markdown("---")

    # Key metrics
    st.sidebar.subheader("📊 Dataset Overview")
    st.sidebar.metric("Total Chunks Analyzed", "505,211")
    st.sidebar.metric("Topic Clusters", len(clusters["clusters"]))

    # Priority counts
    priority_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    for c in clusters["clusters"]:
        tier = c.get("priority_tier", "P2-Monitor")[:2]
        priority_counts[tier] = priority_counts.get(tier, 0) + 1

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Priority Distribution")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("P0 Critical", priority_counts["P0"])
    col2.metric("P1 Important", priority_counts["P1"])
    col1.metric("P2 Monitor", priority_counts["P2"])
    col2.metric("P3 Deprioritize", priority_counts["P3"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Filters")

    # Priority filter
    selected_priorities = st.sidebar.multiselect(
        "Priority Tiers",
        ["P0-Critical", "P1-Important", "P2-Monitor", "P3-Deprioritize"],
        default=["P1-Important", "P2-Monitor"],
    )

    # Min cluster size
    min_size = st.sidebar.slider("Minimum Cluster Size", 50, 1000, 100)

    # Frustration threshold
    min_frustration = st.sidebar.slider("Min Frustration %", 0, 100, 50)

    return {
        "priorities": selected_priorities,
        "min_size": min_size,
        "min_frustration": min_frustration,
    }


def filter_clusters(clusters: dict, filters: dict) -> list:
    """Filter clusters based on sidebar selections."""
    filtered = []
    for c in clusters["clusters"]:
        tier = c.get("priority_tier", "P2-Monitor")
        size = c.get("size", 0)
        frustration = c.get("problem_data", {}).get("frustration_percentage", 0)

        if tier in filters["priorities"] and size >= filters["min_size"] and frustration >= filters["min_frustration"]:
            filtered.append(c)

    return filtered


def render_overview_tab(clusters: dict, filtered_clusters: list):
    """Render overview statistics."""
    st.header("📈 Overview")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    # Calculate aggregate stats
    total_chunks = sum(c.get("size", 0) for c in filtered_clusters)
    avg_frustration = (
        sum(c.get("problem_data", {}).get("frustration_percentage", 0) for c in filtered_clusters) / len(filtered_clusters)
        if filtered_clusters else 0
    )
    avg_high_value = (
        sum(c.get("problem_data", {}).get("high_value_percentage", 0) for c in filtered_clusters) / len(filtered_clusters)
        if filtered_clusters else 0
    )

    col1.metric("Filtered Clusters", len(filtered_clusters))
    col2.metric("Total Discussions", f"{total_chunks:,}")
    col3.metric("Avg Frustration", f"{avg_frustration:.1f}%")
    col4.metric("Avg High-Value", f"{avg_high_value:.1f}%")

    st.markdown("---")

    # Top pain points table
    st.subheader("🔥 Top Pain Points")

    sorted_by_pain = sorted(
        filtered_clusters,
        key=lambda x: x.get("problem_data", {}).get("avg_problem_score", 0),
        reverse=True,
    )[:10]

    pain_data = []
    for c in sorted_by_pain:
        pd_data = c.get("problem_data", {})
        pain_data.append({
            "Topic": c.get("claude_label", c.get("topic_label", "Unknown"))[:40],
            "Size": c.get("size", 0),
            "Problem Score": f"{pd_data.get('avg_problem_score', 0):.1f}",
            "Frustration %": f"{pd_data.get('frustration_percentage', 0):.0f}%",
            "High-Value %": f"{pd_data.get('high_value_percentage', 0):.0f}%",
            "Priority": c.get("priority_tier", "N/A"),
        })

    st.dataframe(pd.DataFrame(pain_data), use_container_width=True, hide_index=True)


def render_cluster_map_tab(projections: pl.DataFrame, clusters: dict):
    """Render 2D cluster visualization."""
    st.header("🗺️ Topic Cluster Map")
    st.caption("2D UMAP projection of 505K discussion embeddings")

    # Create label mapping for top clusters
    label_map = {}
    for c in clusters["clusters"][:30]:
        label_map[c["cluster_id"]] = c.get("claude_label", f"Cluster {c['cluster_id']}")

    # Prepare data
    df = projections.to_pandas()
    df["label"] = df["cluster_label"].apply(
        lambda x: label_map.get(x, "Other") if x != -1 else "Noise"
    )

    # Color options
    color_by = st.radio(
        "Color by:",
        ["Topic Label", "Cluster ID"],
        horizontal=True,
    )

    color_col = "label" if color_by == "Topic Label" else "cluster_label"

    # Create scatter plot
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color=color_col,
        opacity=0.5,
        hover_data=["cluster_label"],
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=9),
        ),
    )
    fig.update_traces(marker=dict(size=3))

    st.plotly_chart(fig, use_container_width=True)


def render_problem_heatmap_tab(filtered_clusters: list):
    """Render problem signals heatmap."""
    st.header("🔴 Problem Signal Analysis")

    # Sort by problem score
    sorted_clusters = sorted(
        filtered_clusters,
        key=lambda x: x.get("problem_data", {}).get("avg_problem_score", 0),
        reverse=True,
    )[:25]

    if not sorted_clusters:
        st.warning("No clusters match the current filters.")
        return

    labels = [c.get("claude_label", f"Cluster {c['cluster_id']}")[:35] for c in sorted_clusters]
    problem_scores = [c.get("problem_data", {}).get("avg_problem_score", 0) for c in sorted_clusters]
    frustration = [c.get("problem_data", {}).get("frustration_percentage", 0) for c in sorted_clusters]
    high_value = [c.get("problem_data", {}).get("high_value_percentage", 0) for c in sorted_clusters]
    sizes = [c.get("size", 0) for c in sorted_clusters]

    # Create subplot
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Problem Score", "Frustration %", "High-Value %", "Cluster Size"],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Bar(y=labels, x=problem_scores, orientation='h', marker_color='#ef4444', name="Problem"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(y=labels, x=frustration, orientation='h', marker_color='#f97316', name="Frustration"),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(y=labels, x=high_value, orientation='h', marker_color='#22c55e', name="High-Value"),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(y=labels, x=sizes, orientation='h', marker_color='#3b82f6', name="Size"),
        row=1, col=4
    )

    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_trends_tab(clusters: dict):
    """Render temporal trends visualization."""
    st.header("📅 Topic Trends Over Time")

    # Separate growing and declining
    growing = []
    declining = []

    for c in clusters["clusters"]:
        ts = c.get("temporal", {}).get("trend_stats", {})
        trend = ts.get("trend", "stable")
        growth = ts.get("growth_rate", 0) * 100
        label = c.get("claude_label", c.get("topic_label", f"Cluster {c['cluster_id']}"))

        if trend in ["growing", "rapidly_growing"]:
            growing.append((label, growth, c.get("size", 0)))
        elif trend in ["declining", "dying"]:
            declining.append((label, growth, c.get("size", 0)))

    growing.sort(key=lambda x: x[1], reverse=True)
    declining.sort(key=lambda x: x[1])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Growing Topics")
        if growing:
            labels, values, sizes = zip(*growing[:10])
            fig = go.Figure(go.Bar(
                y=list(labels),
                x=list(values),
                orientation='h',
                marker_color='#22c55e',
                text=[f"+{v:.1f}%" for v in values],
                textposition='outside',
            ))
            fig.update_layout(height=400, xaxis_title="Growth Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significantly growing topics found.")

    with col2:
        st.subheader("📉 Declining Topics")
        if declining:
            labels, values, sizes = zip(*declining[:10])
            fig = go.Figure(go.Bar(
                y=list(labels),
                x=list(values),
                orientation='h',
                marker_color='#ef4444',
                text=[f"{v:.1f}%" for v in values],
                textposition='outside',
            ))
            fig.update_layout(height=400, xaxis_title="Growth Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significantly declining topics found.")

    # Trend summary
    st.markdown("---")
    st.subheader("📊 Trend Summary")

    trend_counts = {"stable": 0, "growing": 0, "declining": 0, "dying": 0, "rapidly_growing": 0}
    for c in clusters["clusters"]:
        trend = c.get("temporal", {}).get("trend_stats", {}).get("trend", "stable")
        trend_counts[trend] = trend_counts.get(trend, 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stable", trend_counts["stable"])
    col2.metric("Growing", trend_counts["growing"] + trend_counts["rapidly_growing"])
    col3.metric("Declining", trend_counts["declining"])
    col4.metric("Dying", trend_counts["dying"])


def render_priority_matrix_tab(filtered_clusters: list):
    """Render priority matrix scatter plot."""
    st.header("🎯 Priority Matrix")
    st.caption("Problem Score vs Growth Rate - sized by cluster volume")

    # Prepare data
    data = []
    for c in filtered_clusters[:60]:
        pd_data = c.get("problem_data", {})
        ts = c.get("temporal", {}).get("trend_stats", {})

        data.append({
            "label": c.get("claude_label", c.get("topic_label", f"Cluster {c['cluster_id']}"))[:30],
            "problem_score": pd_data.get("avg_problem_score", 0),
            "growth_rate": ts.get("growth_rate", 0) * 100,
            "size": c.get("size", 100),
            "priority": c.get("priority_tier", "P2-Monitor"),
            "frustration": pd_data.get("frustration_percentage", 0),
        })

    if not data:
        st.warning("No clusters match the current filters.")
        return

    df = pd.DataFrame(data)

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
        hover_data=["frustration", "size"],
        color_discrete_map=color_map,
        labels={
            "growth_rate": "Growth Rate (%)",
            "problem_score": "Problem Score",
            "frustration": "Frustration %",
        },
    )

    # Quadrant lines
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Quadrant labels
    fig.add_annotation(x=8, y=9, text="HIGH PRIORITY", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=-8, y=9, text="ADDRESS NOW", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=8, y=1, text="MONITOR", showarrow=False, font=dict(size=14, color="blue"))
    fig.add_annotation(x=-8, y=1, text="DEPRIORITIZE", showarrow=False, font=dict(size=14, color="gray"))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Legend explanation
    with st.expander("📖 How to Read This Chart"):
        st.markdown("""
        - **X-axis (Growth Rate)**: Positive = topic gaining popularity, Negative = declining
        - **Y-axis (Problem Score)**: Higher = more frustration signals and questions
        - **Bubble Size**: Number of discussions in the cluster
        - **Color**: Priority tier based on combination of signals

        **Quadrants:**
        - **Top-Right (High Priority)**: High pain + Growing = Build features here
        - **Top-Left (Address Now)**: High pain + Declining = Fix urgent issues
        - **Bottom-Right (Monitor)**: Low pain + Growing = Watch for opportunities
        - **Bottom-Left (Deprioritize)**: Low pain + Declining = Avoid investing here
        """)


def render_recommendations_tab(clusters: dict):
    """Render Dasher feature recommendations."""
    st.header("💡 Dasher Feature Recommendations")

    recommendations = [
        {
            "feature": "Injury Prevention Challenges",
            "source": "Knee Pain & Injuries",
            "description": "Challenges focused on low-impact running, strength exercises, and gradual mileage increase to prevent knee injuries",
            "complexity": "Medium",
            "impact": "High",
        },
        {
            "feature": "Beginner-Friendly Challenges",
            "source": "Beginner Running",
            "description": "Low-stake, short-duration challenges designed for absolute beginners with walk/run intervals",
            "complexity": "Low",
            "impact": "High",
        },
        {
            "feature": "Shin Splint Recovery Program",
            "source": "Shin Splints",
            "description": "Guided recovery challenges with rest days, cross-training, and gradual return-to-running plans",
            "complexity": "Medium",
            "impact": "High",
        },
        {
            "feature": "C25K Challenge Mode",
            "source": "C25K Discussions",
            "description": "Stake-based Couch to 5K program with weekly milestones and community accountability",
            "complexity": "Low",
            "impact": "High",
        },
        {
            "feature": "Race Prep Challenges",
            "source": "Race Pacing",
            "description": "Goal-race specific challenges with pace targets, taper weeks, and race day readiness tracking",
            "complexity": "Medium",
            "impact": "High",
        },
        {
            "feature": "Weight Loss Running Program",
            "source": "Weight Loss Discussions",
            "description": "Challenges combining running goals with calorie awareness for runners focused on weight management",
            "complexity": "Medium",
            "impact": "High",
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{i}. {rec['feature']}")
                st.write(rec["description"])
                st.caption(f"Based on: {rec['source']}")
            with col2:
                impact_color = "🟢" if rec["impact"] == "High" else "🟡"
                complexity_color = "🟢" if rec["complexity"] == "Low" else "🟡" if rec["complexity"] == "Medium" else "🔴"
                st.metric("Impact", f"{impact_color} {rec['impact']}")
                st.metric("Complexity", f"{complexity_color} {rec['complexity']}")
            st.markdown("---")


def render_cluster_explorer_tab(clusters: dict, filtered_clusters: list):
    """Render detailed cluster explorer."""
    st.header("🔍 Cluster Explorer")

    # Cluster selector
    cluster_options = {
        c.get("claude_label", c.get("topic_label", f"Cluster {c['cluster_id']}")): c
        for c in filtered_clusters[:50]
    }

    if not cluster_options:
        st.warning("No clusters match the current filters.")
        return

    selected_label = st.selectbox("Select a cluster to explore:", list(cluster_options.keys()))
    cluster = cluster_options[selected_label]

    st.markdown("---")

    # Cluster details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Stats")
        st.metric("Cluster Size", f"{cluster.get('size', 0):,}")
        st.metric("Priority", cluster.get("priority_tier", "N/A"))
        trend = cluster.get("temporal", {}).get("trend_stats", {}).get("trend", "stable")
        st.metric("Trend", trend.title())

    with col2:
        st.subheader("😤 Problem Signals")
        pd_data = cluster.get("problem_data", {})
        st.metric("Problem Score", f"{pd_data.get('avg_problem_score', 0):.1f}")
        st.metric("Frustration %", f"{pd_data.get('frustration_percentage', 0):.0f}%")
        st.metric("High-Value %", f"{pd_data.get('high_value_percentage', 0):.0f}%")

    with col3:
        st.subheader("📈 Temporal")
        ts = cluster.get("temporal", {}).get("trend_stats", {})
        growth = ts.get("growth_rate", 0) * 100
        st.metric("Growth Rate", f"{growth:+.1f}%")
        st.metric("Significant", "Yes" if ts.get("is_significant", False) else "No")

    # Keywords
    st.markdown("---")
    st.subheader("🏷️ Keywords")
    keywords = cluster.get("keywords", [])[:20]
    st.write(" | ".join(keywords) if keywords else "No keywords available")

    # Sample discussions
    st.markdown("---")
    st.subheader("💬 Sample Discussions")
    samples = cluster.get("representative_samples", [])[:5]
    for i, sample in enumerate(samples, 1):
        with st.expander(f"Sample {i}"):
            st.write(sample[:500] + "..." if len(sample) > 500 else sample)


def main():
    # Load data
    try:
        clusters, projections, temporal, priority_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure all data files exist in the topic_analysis/data/ directory.")
        return

    # Sidebar
    filters = render_sidebar(clusters)

    # Filter clusters
    filtered_clusters = filter_clusters(clusters, filters)

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview",
        "🗺️ Cluster Map",
        "🔴 Problem Signals",
        "📅 Trends",
        "🎯 Priority Matrix",
        "💡 Recommendations",
        "🔍 Explorer",
    ])

    with tab1:
        render_overview_tab(clusters, filtered_clusters)

    with tab2:
        render_cluster_map_tab(projections, clusters)

    with tab3:
        render_problem_heatmap_tab(filtered_clusters)

    with tab4:
        render_trends_tab(clusters)

    with tab5:
        render_priority_matrix_tab(filtered_clusters)

    with tab6:
        render_recommendations_tab(clusters)

    with tab7:
        render_cluster_explorer_tab(clusters, filtered_clusters)

    # Footer
    st.markdown("---")
    st.caption("Topic Analysis Dashboard | Data from Reddit & Discord running communities | 505K+ discussions analyzed")


if __name__ == "__main__":
    main()
