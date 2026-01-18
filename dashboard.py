"""
Topic Analysis Dashboard - Premium Edition

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

# Custom CSS for premium look
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Custom header */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .dashboard-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .dashboard-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }

    /* Cards for content */
    .content-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    /* Priority badges */
    .priority-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .priority-p0 { background: #fef2f2; color: #dc2626; }
    .priority-p1 { background: #fff7ed; color: #ea580c; }
    .priority-p2 { background: #eff6ff; color: #2563eb; }
    .priority-p3 { background: #f1f5f9; color: #64748b; }

    /* Trend indicators */
    .trend-up { color: #16a34a; }
    .trend-down { color: #dc2626; }
    .trend-stable { color: #64748b; }

    /* Data tables */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }

    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }

    .dataframe td {
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }

    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Impact badges */
    .impact-high { background: #dcfce7; color: #16a34a; }
    .impact-medium { background: #fef3c7; color: #d97706; }
    .impact-low { background: #f1f5f9; color: #64748b; }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_DIR = Path(__file__).parent / "data"
CLUSTERS_FILE = DATA_DIR / "clusters_labeled.json"
PROJECTIONS_FILE = DATA_DIR / "umap_projections.parquet"
TEMPORAL_FILE = DATA_DIR / "temporal_trends.json"
PRIORITY_FILE = DATA_DIR / "priority_matrix.csv"

# Color palette
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
    "info": "#2563eb",
    "muted": "#64748b",
    "gradient": ["#667eea", "#764ba2", "#9333ea", "#c026d3"],
}

# Plotly theme
PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, sans-serif"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": ["#667eea", "#764ba2", "#16a34a", "#d97706", "#dc2626", "#2563eb"],
    }
}


@st.cache_data
def load_data():
    """Load all data files."""
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)

    projections = pl.read_parquet(str(PROJECTIONS_FILE))
    if projections.height > 50000:
        projections = projections.sample(n=50000, seed=42)

    with open(TEMPORAL_FILE, "r") as f:
        temporal = json.load(f)

    priority_df = pd.read_csv(PRIORITY_FILE)

    return clusters, projections, temporal, priority_df


def render_header():
    """Render premium header."""
    st.markdown("""
    <div class="dashboard-header">
        <h1>🏃 Running Community Insights</h1>
        <p>AI-powered analysis of 505,211 discussions from Reddit & Discord</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(clusters: dict):
    """Render styled sidebar."""
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        st.markdown("---")

        # Dataset stats
        st.markdown("#### Dataset Overview")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", "505K")
        col2.metric("Clusters", len(clusters["clusters"]))

        # Priority counts
        priority_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for c in clusters["clusters"]:
            tier = c.get("priority_tier", "P2-Monitor")[:2]
            priority_counts[tier] = priority_counts.get(tier, 0) + 1

        st.markdown("---")
        st.markdown("#### Priority Distribution")

        for tier, count in priority_counts.items():
            color = {"P0": "🔴", "P1": "🟠", "P2": "🔵", "P3": "⚪"}[tier]
            st.markdown(f"{color} **{tier}**: {count} clusters")

        st.markdown("---")
        st.markdown("#### Filters")

        selected_priorities = st.multiselect(
            "Priority Tiers",
            ["P0-Critical", "P1-Important", "P2-Monitor", "P3-Deprioritize"],
            default=["P1-Important", "P2-Monitor"],
        )

        min_size = st.slider("Min Cluster Size", 50, 1000, 100)
        min_frustration = st.slider("Min Frustration %", 0, 100, 40)

        st.markdown("---")
        st.markdown("#### Data Sources")
        st.markdown("""
        - r/running (297K)
        - r/C25K (82K)
        - r/beginnerrunning (28K)
        - Discord: 3 servers (98K)
        """)

    return {
        "priorities": selected_priorities,
        "min_size": min_size,
        "min_frustration": min_frustration,
    }


def filter_clusters(clusters: dict, filters: dict) -> list:
    """Filter clusters based on selections."""
    filtered = []
    for c in clusters["clusters"]:
        tier = c.get("priority_tier", "P2-Monitor")
        size = c.get("size", 0)
        frustration = c.get("problem_data", {}).get("frustration_percentage", 0)

        if tier in filters["priorities"] and size >= filters["min_size"] and frustration >= filters["min_frustration"]:
            filtered.append(c)
    return filtered


def render_metrics(clusters: dict, filtered_clusters: list):
    """Render key metrics as cards."""
    total_chunks = sum(c.get("size", 0) for c in filtered_clusters)
    avg_frustration = (
        sum(c.get("problem_data", {}).get("frustration_percentage", 0) for c in filtered_clusters) / len(filtered_clusters)
        if filtered_clusters else 0
    )
    avg_problem = (
        sum(c.get("problem_data", {}).get("avg_problem_score", 0) for c in filtered_clusters) / len(filtered_clusters)
        if filtered_clusters else 0
    )

    # Count trends
    growing = sum(1 for c in clusters["clusters"] if c.get("temporal", {}).get("trend_stats", {}).get("trend") in ["growing", "rapidly_growing"])
    declining = sum(1 for c in clusters["clusters"] if c.get("temporal", {}).get("trend_stats", {}).get("trend") in ["declining", "dying"])

    cols = st.columns(5)

    metrics = [
        ("Filtered Clusters", len(filtered_clusters), "📊"),
        ("Total Discussions", f"{total_chunks:,}", "💬"),
        ("Avg Frustration", f"{avg_frustration:.0f}%", "😤"),
        ("Growing Topics", growing, "📈"),
        ("Declining Topics", declining, "📉"),
    ]

    for col, (label, value, icon) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


def render_overview_tab(clusters: dict, filtered_clusters: list):
    """Render overview with metrics and top pain points."""
    render_metrics(clusters, filtered_clusters)

    st.markdown('<div class="section-header">🔥 Top Pain Points</div>', unsafe_allow_html=True)

    sorted_by_pain = sorted(
        filtered_clusters,
        key=lambda x: x.get("problem_data", {}).get("avg_problem_score", 0),
        reverse=True,
    )[:10]

    # Create styled table
    for i, c in enumerate(sorted_by_pain, 1):
        pd_data = c.get("problem_data", {})
        ts = c.get("temporal", {}).get("trend_stats", {})
        trend = ts.get("trend", "stable")
        growth = ts.get("growth_rate", 0) * 100
        priority = c.get("priority_tier", "P2-Monitor")

        trend_icon = {"growing": "📈", "rapidly_growing": "🚀", "declining": "📉", "dying": "💀"}.get(trend, "➡️")
        priority_class = {"P0-Critical": "priority-p0", "P1-Important": "priority-p1", "P2-Monitor": "priority-p2", "P3-Deprioritize": "priority-p3"}.get(priority, "priority-p2")

        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

        with col1:
            st.markdown(f"**{i}. {c.get('claude_label', c.get('topic_label', 'Unknown'))[:45]}**")
        with col2:
            st.markdown(f"<span class='priority-badge {priority_class}'>{priority[:2]}</span>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"😤 {pd_data.get('frustration_percentage', 0):.0f}%")
        with col4:
            st.markdown(f"💬 {c.get('size', 0):,}")
        with col5:
            st.markdown(f"{trend_icon} {growth:+.1f}%")

        st.markdown("---")


def render_cluster_map_tab(projections: pl.DataFrame, clusters: dict):
    """Render enhanced cluster visualization."""
    st.markdown('<div class="section-header">🗺️ Topic Landscape</div>', unsafe_allow_html=True)
    st.caption("50,000 discussions projected into 2D space using UMAP. Each point is a discussion, colors represent topic clusters.")

    label_map = {}
    for c in clusters["clusters"][:25]:
        label_map[c["cluster_id"]] = c.get("claude_label", f"Cluster {c['cluster_id']}")

    df = projections.to_pandas()
    df["Topic"] = df["cluster_label"].apply(
        lambda x: label_map.get(x, "Other Topics") if x != -1 else "Unclustered"
    )

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="Topic",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        height=650,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        margin=dict(l=20, r=20, t=20, b=100),
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))

    st.plotly_chart(fig, use_container_width=True)


def render_problem_heatmap_tab(filtered_clusters: list):
    """Render enhanced problem signals visualization."""
    st.markdown('<div class="section-header">🔴 Problem Signal Analysis</div>', unsafe_allow_html=True)
    st.caption("Clusters ranked by problem intensity. Higher scores indicate more frustrated users seeking solutions.")

    sorted_clusters = sorted(
        filtered_clusters,
        key=lambda x: x.get("problem_data", {}).get("avg_problem_score", 0),
        reverse=True,
    )[:20]

    if not sorted_clusters:
        st.warning("No clusters match the current filters. Try adjusting the sidebar filters.")
        return

    labels = [c.get("claude_label", f"Cluster {c['cluster_id']}")[:30] for c in sorted_clusters]
    problem_scores = [c.get("problem_data", {}).get("avg_problem_score", 0) for c in sorted_clusters]
    frustration = [c.get("problem_data", {}).get("frustration_percentage", 0) for c in sorted_clusters]
    high_value = [c.get("problem_data", {}).get("high_value_percentage", 0) for c in sorted_clusters]
    sizes = [c.get("size", 0) for c in sorted_clusters]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Problem Score", "Frustration %", "High-Value %", "Volume"],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    colors = ["#dc2626", "#f97316", "#16a34a", "#667eea"]
    data = [problem_scores, frustration, high_value, sizes]

    for i, (vals, color) in enumerate(zip(data, colors), 1):
        fig.add_trace(
            go.Bar(
                y=labels,
                x=vals,
                orientation='h',
                marker_color=color,
                marker=dict(
                    line=dict(width=0),
                    cornerradius=4,
                ),
            ),
            row=1, col=i
        )

    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=0, r=20, t=40, b=20),
    )

    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", row=1, col=i)

    st.plotly_chart(fig, use_container_width=True)


def render_trends_tab(clusters: dict):
    """Render enhanced temporal trends."""
    st.markdown('<div class="section-header">📅 Topic Momentum</div>', unsafe_allow_html=True)
    st.caption("Topics gaining or losing community attention over time. Growth rates based on 6-month trend analysis.")

    growing = []
    declining = []

    for c in clusters["clusters"]:
        ts = c.get("temporal", {}).get("trend_stats", {})
        trend = ts.get("trend", "stable")
        growth = ts.get("growth_rate", 0) * 100
        label = c.get("claude_label", c.get("topic_label", f"Cluster {c['cluster_id']}"))
        size = c.get("size", 0)

        if trend in ["growing", "rapidly_growing"]:
            growing.append((label, growth, size))
        elif trend in ["declining", "dying"]:
            declining.append((label, growth, size))

    growing.sort(key=lambda x: x[1], reverse=True)
    declining.sort(key=lambda x: x[1])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Rising Topics")
        if growing:
            labels, values, sizes = zip(*growing[:8])
            fig = go.Figure(go.Bar(
                y=list(labels),
                x=list(values),
                orientation='h',
                marker=dict(
                    color=list(values),
                    colorscale=[[0, "#86efac"], [1, "#16a34a"]],
                    cornerradius=6,
                ),
                text=[f"+{v:.1f}%" for v in values],
                textposition='outside',
                textfont=dict(size=11, color="#16a34a"),
            ))
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=60, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Growth Rate (%)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significantly growing topics in current selection.")

    with col2:
        st.markdown("### 📉 Declining Topics")
        if declining:
            labels, values, sizes = zip(*declining[:8])
            fig = go.Figure(go.Bar(
                y=list(labels),
                x=list(values),
                orientation='h',
                marker=dict(
                    color=[abs(v) for v in values],
                    colorscale=[[0, "#fca5a5"], [1, "#dc2626"]],
                    cornerradius=6,
                ),
                text=[f"{v:.1f}%" for v in values],
                textposition='outside',
                textfont=dict(size=11, color="#dc2626"),
            ))
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=60, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Growth Rate (%)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significantly declining topics in current selection.")

    # Trend summary cards
    st.markdown("---")
    trend_counts = {"stable": 0, "growing": 0, "declining": 0, "dying": 0, "rapidly_growing": 0}
    for c in clusters["clusters"]:
        trend = c.get("temporal", {}).get("trend_stats", {}).get("trend", "stable")
        trend_counts[trend] = trend_counts.get(trend, 0) + 1

    cols = st.columns(4)
    trend_data = [
        ("Stable", trend_counts["stable"], "➡️", "#64748b"),
        ("Growing", trend_counts["growing"] + trend_counts["rapidly_growing"], "📈", "#16a34a"),
        ("Declining", trend_counts["declining"], "📉", "#f97316"),
        ("Dying", trend_counts["dying"], "💀", "#dc2626"),
    ]

    for col, (label, count, icon, color) in zip(cols, trend_data):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="font-size: 2rem; font-weight: 700; color: {color};">{count}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


def render_priority_matrix_tab(filtered_clusters: list):
    """Render enhanced priority matrix."""
    st.markdown('<div class="section-header">🎯 Strategic Priority Matrix</div>', unsafe_allow_html=True)
    st.caption("Position topics by pain intensity (Y) and growth trajectory (X) to identify where to focus product efforts.")

    data = []
    for c in filtered_clusters[:50]:
        pd_data = c.get("problem_data", {})
        ts = c.get("temporal", {}).get("trend_stats", {})

        data.append({
            "Topic": c.get("claude_label", c.get("topic_label", f"Cluster {c['cluster_id']}"))[:25],
            "Problem Score": pd_data.get("avg_problem_score", 0),
            "Growth Rate": ts.get("growth_rate", 0) * 100,
            "Volume": c.get("size", 100),
            "Priority": c.get("priority_tier", "P2-Monitor"),
            "Frustration": pd_data.get("frustration_percentage", 0),
        })

    if not data:
        st.warning("No clusters match the current filters.")
        return

    df = pd.DataFrame(data)

    color_map = {
        "P0-Critical": "#dc2626",
        "P1-Important": "#f97316",
        "P2-Monitor": "#667eea",
        "P3-Deprioritize": "#94a3b8",
    }

    fig = px.scatter(
        df,
        x="Growth Rate",
        y="Problem Score",
        size="Volume",
        color="Priority",
        hover_name="Topic",
        hover_data=["Frustration", "Volume"],
        color_discrete_map=color_map,
        size_max=50,
    )

    # Quadrant styling
    fig.add_shape(type="rect", x0=0, x1=20, y0=5, y1=12, fillcolor="rgba(22, 163, 74, 0.08)", line_width=0)
    fig.add_shape(type="rect", x0=-20, x1=0, y0=5, y1=12, fillcolor="rgba(249, 115, 22, 0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=20, y0=0, y1=5, fillcolor="rgba(102, 126, 234, 0.08)", line_width=0)
    fig.add_shape(type="rect", x0=-20, x1=0, y0=0, y1=5, fillcolor="rgba(148, 163, 184, 0.08)", line_width=0)

    fig.add_hline(y=5, line_dash="dot", line_color="#cbd5e1", line_width=2)
    fig.add_vline(x=0, line_dash="dot", line_color="#cbd5e1", line_width=2)

    # Quadrant labels
    annotations = [
        dict(x=10, y=10.5, text="<b>🎯 BUILD HERE</b><br>High pain + Growing", showarrow=False, font=dict(size=12, color="#16a34a")),
        dict(x=-10, y=10.5, text="<b>⚠️ FIX NOW</b><br>High pain + Declining", showarrow=False, font=dict(size=12, color="#f97316")),
        dict(x=10, y=1.5, text="<b>👀 MONITOR</b><br>Low pain + Growing", showarrow=False, font=dict(size=12, color="#667eea")),
        dict(x=-10, y=1.5, text="<b>⏸️ SKIP</b><br>Low pain + Declining", showarrow=False, font=dict(size=12, color="#94a3b8")),
    ]

    for ann in annotations:
        fig.add_annotation(**ann)

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Growth Rate (%)", range=[-20, 20], showgrid=False, zeroline=False),
        yaxis=dict(title="Problem Score", range=[0, 12], showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 How to Read This Matrix"):
        st.markdown("""
        | Quadrant | Description | Action |
        |----------|-------------|--------|
        | **🎯 Build Here** | High frustration + Growing interest | Prioritize features for these topics |
        | **⚠️ Fix Now** | High frustration + Declining interest | Address urgent pain points before users leave |
        | **👀 Monitor** | Low frustration + Growing interest | Watch for emerging opportunities |
        | **⏸️ Skip** | Low frustration + Declining interest | Don't invest resources here |
        """)


def render_recommendations_tab(clusters: dict):
    """Render enhanced feature recommendations."""
    st.markdown('<div class="section-header">💡 Dasher Feature Opportunities</div>', unsafe_allow_html=True)
    st.caption("Actionable feature recommendations based on community pain points and trends.")

    recommendations = [
        {
            "title": "Injury Prevention Challenges",
            "icon": "🦵",
            "source": "Knee Pain & Injuries (92% frustration)",
            "desc": "Stake-based challenges with low-impact running plans, strength exercises, and gradual mileage progression. Include injury risk assessments and recovery protocols.",
            "impact": "High",
            "complexity": "Medium",
        },
        {
            "title": "Beginner-First Onboarding",
            "icon": "🌱",
            "source": "Beginner Running (68% frustration, +5% growth)",
            "desc": "Dedicated beginner mode with walk/run intervals, achievement celebrations, and curated low-stake challenges. Reduce intimidation barrier for new runners.",
            "impact": "High",
            "complexity": "Low",
        },
        {
            "title": "C25K Challenge Mode",
            "icon": "🎯",
            "source": "C25K Discussions (82K chunks)",
            "desc": "9-week progressive challenge aligned with the C25K program. Stake increases as users progress, with milestone rewards and community support.",
            "impact": "High",
            "complexity": "Low",
        },
        {
            "title": "Shin Splint Recovery Program",
            "icon": "🩹",
            "source": "Shin Splints (76% frustration)",
            "desc": "Guided recovery challenges with mandatory rest days, cross-training alternatives, and gradual return-to-running plans. Partner with physio content creators.",
            "impact": "High",
            "complexity": "Medium",
        },
        {
            "title": "Race Prep Challenges",
            "icon": "🏅",
            "source": "Race Pacing (630 discussions)",
            "desc": "Goal-race specific training blocks with pace targets, taper weeks, and race-day readiness checklists. Support 5K through marathon distances.",
            "impact": "High",
            "complexity": "Medium",
        },
        {
            "title": "Weight Management Integration",
            "icon": "⚖️",
            "source": "Weight Loss Running (100% high-value)",
            "desc": "Running + nutrition awareness challenges for users with weight loss goals. Avoid pure calorie-counting; focus on sustainable habits and non-scale victories.",
            "impact": "Medium",
            "complexity": "Medium",
        },
    ]

    for rec in recommendations:
        impact_class = f"impact-{rec['impact'].lower()}"

        st.markdown(f"""
        <div class="feature-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="feature-title">{rec['icon']} {rec['title']}</div>
                    <div style="color: #667eea; font-size: 0.8rem; margin-bottom: 0.5rem;">Based on: {rec['source']}</div>
                    <div class="feature-desc">{rec['desc']}</div>
                </div>
                <div style="text-align: right; min-width: 100px;">
                    <span class="priority-badge {impact_class}" style="margin-bottom: 0.5rem; display: inline-block;">{rec['impact']} Impact</span><br>
                    <span style="color: #64748b; font-size: 0.8rem;">{rec['complexity']} Complexity</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_explorer_tab(clusters: dict, filtered_clusters: list):
    """Render enhanced cluster explorer."""
    st.markdown('<div class="section-header">🔍 Deep Dive Explorer</div>', unsafe_allow_html=True)

    cluster_options = {
        c.get("claude_label", c.get("topic_label", f"Cluster {c['cluster_id']}")): c
        for c in filtered_clusters[:50]
    }

    if not cluster_options:
        st.warning("No clusters match the current filters.")
        return

    selected_label = st.selectbox("Select a topic to explore:", list(cluster_options.keys()))
    cluster = cluster_options[selected_label]

    st.markdown("---")

    # Info cards row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="content-card">
            <h4 style="margin-top: 0; color: #667eea;">📊 Statistics</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Cluster Size", f"{cluster.get('size', 0):,}")
        priority = cluster.get("priority_tier", "N/A")
        st.markdown(f"**Priority:** {priority}")
        trend = cluster.get("temporal", {}).get("trend_stats", {}).get("trend", "stable")
        st.markdown(f"**Trend:** {trend.title()}")

    with col2:
        st.markdown("""
        <div class="content-card">
            <h4 style="margin-top: 0; color: #dc2626;">😤 Problem Signals</h4>
        </div>
        """, unsafe_allow_html=True)
        pd_data = cluster.get("problem_data", {})
        st.metric("Problem Score", f"{pd_data.get('avg_problem_score', 0):.1f}")
        st.metric("Frustration", f"{pd_data.get('frustration_percentage', 0):.0f}%")
        st.metric("High-Value", f"{pd_data.get('high_value_percentage', 0):.0f}%")

    with col3:
        st.markdown("""
        <div class="content-card">
            <h4 style="margin-top: 0; color: #16a34a;">📈 Temporal Data</h4>
        </div>
        """, unsafe_allow_html=True)
        ts = cluster.get("temporal", {}).get("trend_stats", {})
        growth = ts.get("growth_rate", 0) * 100
        st.metric("Growth Rate", f"{growth:+.1f}%")
        st.markdown(f"**Significant:** {'Yes' if ts.get('is_significant', False) else 'No'}")
        st.markdown(f"**Seasonal:** {'Yes' if ts.get('is_seasonal', False) else 'No'}")

    # Keywords
    st.markdown("---")
    st.markdown("#### 🏷️ Top Keywords")
    keywords = cluster.get("keywords", [])[:15]
    if keywords:
        keyword_html = " ".join([f'<span style="background: #f1f5f9; padding: 0.3rem 0.7rem; border-radius: 15px; margin: 0.2rem; display: inline-block; font-size: 0.85rem;">{kw}</span>' for kw in keywords])
        st.markdown(keyword_html, unsafe_allow_html=True)

    # Sample discussions
    st.markdown("---")
    st.markdown("#### 💬 Sample Discussions")
    samples = cluster.get("representative_samples", [])[:5]
    for i, sample in enumerate(samples, 1):
        with st.expander(f"Discussion {i}"):
            st.markdown(f"_{sample[:600]}{'...' if len(sample) > 600 else ''}_")


def main():
    try:
        clusters, projections, temporal, priority_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure all data files exist in the `topic_analysis/data/` directory.")
        return

    render_header()
    filters = render_sidebar(clusters)
    filtered_clusters = filter_clusters(clusters, filters)

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
        render_explorer_tab(clusters, filtered_clusters)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;">
        Topic Analysis Dashboard • 505,211 discussions from Reddit & Discord • Powered by UMAP + HDBSCAN + Claude
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
