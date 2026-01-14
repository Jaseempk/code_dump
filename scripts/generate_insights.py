"""
Phase 5: Insight Generation & Dasher Recommendations

Combines problem scores and temporal trends to create a priority matrix
and generate actionable recommendations for Dasher features.

Usage:
    python generate_insights.py
"""

import json
from pathlib import Path
from datetime import datetime
import csv

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CLUSTERS_FILE = DATA_DIR / "clusters_final.json"
OUTPUT_DIR = Path(__file__).parent.parent

# Priority weights
PROBLEM_WEIGHT = 0.5
GROWTH_WEIGHT = 0.3
SIZE_WEIGHT = 0.2


def calculate_priority_score(cluster: dict, max_problem: float, max_size: int) -> float:
    """Calculate composite priority score for a cluster."""
    problem_data = cluster.get("problem_data", {})
    temporal = cluster.get("temporal", {})
    trend_stats = temporal.get("trend_stats", {})

    # Normalize problem score (0-1)
    avg_problem = problem_data.get("avg_problem_score", 0)
    problem_norm = avg_problem / max_problem if max_problem > 0 else 0

    # Normalize growth rate (-1 to 1 becomes 0 to 1)
    growth_rate = trend_stats.get("growth_rate", 0)
    growth_norm = (growth_rate + 1) / 2  # Map -1,1 to 0,1

    # Normalize size (0-1)
    size = cluster.get("size", 0)
    size_norm = size / max_size if max_size > 0 else 0

    # Composite score
    priority = (
        problem_norm * PROBLEM_WEIGHT +
        growth_norm * GROWTH_WEIGHT +
        size_norm * SIZE_WEIGHT
    )

    return priority


def classify_priority(cluster: dict) -> str:
    """Classify cluster into priority tier."""
    problem_data = cluster.get("problem_data", {})
    temporal = cluster.get("temporal", {})
    trend_stats = temporal.get("trend_stats", {})

    avg_problem = problem_data.get("avg_problem_score", 0)
    trend = trend_stats.get("trend", "stable")
    high_value_pct = problem_data.get("high_value_percentage", 0)

    # High pain + Growing = P0
    if avg_problem >= 6 and trend in ["growing", "rapidly_growing"]:
        return "P0-Critical"

    # High pain + Stable OR Medium pain + Growing = P1
    if avg_problem >= 6 and trend == "stable":
        return "P1-Important"
    if avg_problem >= 4 and trend in ["growing", "rapidly_growing"]:
        return "P1-Important"

    # High-value content (beginners, weight loss, injuries)
    if high_value_pct >= 80:
        return "P1-Important"

    # Medium pain + Stable = P2
    if avg_problem >= 4:
        return "P2-Monitor"

    # Declining = P3
    if trend in ["declining", "dying"]:
        return "P3-Deprioritize"

    return "P2-Monitor"


def infer_topic_label(cluster: dict) -> str:
    """Infer a human-readable topic label from keywords."""
    keywords = cluster.get("keywords", [])
    if not keywords:
        return "General Running Discussion"

    kw_set = set(k.lower() for k in keywords[:10])

    # Injury topics
    if "knee" in kw_set or "knees" in kw_set:
        return "Knee Pain & Injuries"
    if "shin" in kw_set or "splints" in kw_set:
        return "Shin Splints"
    if "band" in kw_set or "it band" in kw_set:
        return "IT Band Issues"
    if "plantar" in kw_set or "fasciitis" in kw_set:
        return "Plantar Fasciitis"
    if "achilles" in kw_set:
        return "Achilles Problems"
    if "pain" in kw_set and "calves" in kw_set:
        return "Calf Pain"
    if "pain" in kw_set:
        return "Running Pain & Injuries"

    # Beginner topics
    if "weight" in kw_set and ("loss" in kw_set or "lose" in kw_set):
        return "Weight Loss Running"
    if "start" in kw_set or "started" in kw_set or "beginner" in kw_set:
        return "Beginner Running"
    if "c25k" in kw_set or "couch" in kw_set:
        return "Couch to 5K Program"

    # Training topics
    if "pace" in kw_set and ("5k" in kw_set or "race" in kw_set):
        return "Race Pacing"
    if "hr" in kw_set or "heart" in kw_set or "zone" in kw_set:
        return "Heart Rate Training"
    if "training" in kw_set and "plan" in kw_set:
        return "Training Plans"
    if "strength" in kw_set:
        return "Strength Training for Runners"
    if "marathon" in kw_set:
        return "Marathon Training"
    if "half" in kw_set:
        return "Half Marathon"

    # Gear topics
    if "shoes" in kw_set or "shoe" in kw_set:
        return "Running Shoes"
    if "watch" in kw_set or "garmin" in kw_set:
        return "Running Watches & Tech"

    # Other
    if "stitch" in kw_set or "stitches" in kw_set:
        return "Side Stitches"
    if "breathing" in kw_set or "breath" in kw_set:
        return "Breathing Techniques"
    if "motivation" in kw_set:
        return "Motivation & Consistency"
    if "race" in kw_set:
        return "Race Discussion"
    if "heat" in kw_set or "hot" in kw_set:
        return "Running in Heat"
    if "cold" in kw_set or "winter" in kw_set:
        return "Cold Weather Running"

    return f"Topic: {', '.join(keywords[:3])}"


def generate_dasher_recommendation(cluster: dict) -> dict:
    """Generate Dasher feature recommendation for a cluster."""
    topic = infer_topic_label(cluster)
    problem_data = cluster.get("problem_data", {})
    avg_problem = problem_data.get("avg_problem_score", 0)
    high_value_pct = problem_data.get("high_value_percentage", 0)

    recommendations = {
        "Knee Pain & Injuries": {
            "feature": "Injury Prevention Challenges",
            "description": "Challenges focused on low-impact running, strength exercises, and gradual mileage increase to prevent knee injuries",
            "complexity": "Medium",
            "impact": "High",
        },
        "Shin Splints": {
            "feature": "Shin Splint Recovery Program",
            "description": "Guided recovery challenges with rest days, cross-training, and gradual return-to-running plans",
            "complexity": "Medium",
            "impact": "High",
        },
        "IT Band Issues": {
            "feature": "IT Band Recovery & Prevention",
            "description": "Challenges with foam rolling reminders, hip strengthening exercises, and modified running plans",
            "complexity": "Medium",
            "impact": "Medium",
        },
        "Beginner Running": {
            "feature": "Beginner-Friendly Challenges",
            "description": "Low-stake, short-duration challenges designed for absolute beginners with walk/run intervals",
            "complexity": "Low",
            "impact": "High",
        },
        "Weight Loss Running": {
            "feature": "Weight Loss Journey Challenges",
            "description": "Challenges combining running goals with weight tracking, calorie awareness, and supportive community",
            "complexity": "Medium",
            "impact": "High",
        },
        "Couch to 5K Program": {
            "feature": "C25K Challenge Mode",
            "description": "Structured 9-week C25K challenges with daily accountability, community support, and completion rewards",
            "complexity": "Low",
            "impact": "High",
        },
        "Heart Rate Training": {
            "feature": "Zone-Based Training Challenges",
            "description": "Challenges that require staying in specific HR zones, promoting proper aerobic base building",
            "complexity": "High",
            "impact": "Medium",
        },
        "Running Shoes": {
            "feature": "Shoe Mileage Tracker",
            "description": "Track shoe wear, get replacement recommendations, and unlock challenges for new shoe break-in",
            "complexity": "Low",
            "impact": "Medium",
        },
        "Side Stitches": {
            "feature": "Breathing & Pacing Tips",
            "description": "In-app tips and challenges focused on breathing techniques and proper warm-up to prevent stitches",
            "complexity": "Low",
            "impact": "Medium",
        },
        "Motivation & Consistency": {
            "feature": "Streak & Consistency Rewards",
            "description": "Enhanced streak tracking, milestone celebrations, and social accountability features",
            "complexity": "Low",
            "impact": "High",
        },
        "Race Pacing": {
            "feature": "Race Prep Challenges",
            "description": "Goal-race specific challenges with pace targets, taper weeks, and race day readiness tracking",
            "complexity": "Medium",
            "impact": "High",
        },
    }

    default = {
        "feature": f"Content Hub: {topic}",
        "description": f"Curated advice and challenges related to {topic.lower()}",
        "complexity": "Low",
        "impact": "Low",
    }

    return recommendations.get(topic, default)


def generate_report(clusters: list) -> str:
    """Generate markdown insights report."""

    # Calculate stats
    total = len(clusters)
    p0 = [c for c in clusters if c.get("priority_tier") == "P0-Critical"]
    p1 = [c for c in clusters if c.get("priority_tier") == "P1-Important"]
    p2 = [c for c in clusters if c.get("priority_tier") == "P2-Monitor"]
    p3 = [c for c in clusters if c.get("priority_tier") == "P3-Deprioritize"]

    report = f"""# Topic Analysis Insights Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Executive Summary

Analyzed **505,211 chunks** from 6 data sources (Reddit + Discord) and discovered **{total} topic clusters**.

### Priority Distribution

| Tier | Count | Description |
|------|-------|-------------|
| P0-Critical | {len(p0)} | High pain + Growing trend |
| P1-Important | {len(p1)} | High pain OR High-value content |
| P2-Monitor | {len(p2)} | Moderate interest |
| P3-Deprioritize | {len(p3)} | Declining topics |

---

## Top Problem Clusters (High Pain Points)

These clusters represent the biggest pain points in the running community:

"""

    # Top 10 by problem score
    problem_sorted = sorted(clusters, key=lambda x: x.get("problem_data", {}).get("avg_problem_score", 0), reverse=True)

    for i, c in enumerate(problem_sorted[:10]):
        pd = c.get("problem_data", {})
        topic = c.get("topic_label", "Unknown")
        report += f"""### {i+1}. {topic}

- **Cluster Size:** {c['size']:,} discussions
- **Problem Score:** {pd.get('avg_problem_score', 0):.2f}
- **Frustration Level:** {pd.get('frustration_percentage', 0):.1f}%
- **High-Value Signals:** {pd.get('high_value_percentage', 0):.1f}%
- **Keywords:** {', '.join(c['keywords'][:5])}

"""

    report += """---

## Growing Topics (Opportunity Zones)

These topics are increasing in discussion volume:

"""

    # Growing clusters
    growing = [c for c in clusters if c.get("temporal", {}).get("trend_stats", {}).get("trend") in ["growing", "rapidly_growing"]]
    for c in growing:
        ts = c.get("temporal", {}).get("trend_stats", {})
        topic = c.get("topic_label", "Unknown")
        report += f"""- **{topic}**: +{ts.get('growth_rate', 0)*100:.1f}% growth ({c['size']:,} chunks)
"""

    report += """

---

## Declining Topics (Avoid Building For)

These topics are decreasing in relevance:

"""

    # Declining clusters (top 5)
    declining = sorted(
        [c for c in clusters if c.get("temporal", {}).get("trend_stats", {}).get("trend") in ["declining", "dying"]],
        key=lambda x: x.get("temporal", {}).get("trend_stats", {}).get("growth_rate", 0)
    )[:5]

    for c in declining:
        ts = c.get("temporal", {}).get("trend_stats", {})
        topic = c.get("topic_label", "Unknown")
        report += f"""- **{topic}**: {ts.get('growth_rate', 0)*100:.1f}% decline ({c['size']:,} chunks)
"""

    report += """

---

## Dasher Feature Recommendations

Based on the analysis, here are the top feature opportunities for Dasher:

"""

    # Unique recommendations from P0 and P1 clusters
    seen_features = set()
    recommendations = []

    for c in (p0 + p1)[:20]:
        rec = c.get("dasher_recommendation", {})
        feature = rec.get("feature", "")
        if feature and feature not in seen_features:
            seen_features.add(feature)
            recommendations.append({
                "feature": feature,
                "description": rec.get("description", ""),
                "complexity": rec.get("complexity", "Medium"),
                "impact": rec.get("impact", "Medium"),
                "source_topic": c.get("topic_label", ""),
            })

    for i, rec in enumerate(recommendations[:10]):
        report += f"""### {i+1}. {rec['feature']}

- **Source Topic:** {rec['source_topic']}
- **Description:** {rec['description']}
- **Implementation Complexity:** {rec['complexity']}
- **User Impact:** {rec['impact']}

"""

    report += """---

## Methodology

1. **Data Sources:** Reddit (r/running, r/C25K, r/beginnerrunning) + Discord (beginner_running, running_questions, running_science)
2. **Embedding Model:** Google text-embedding-004 (768 dimensions)
3. **Dimensionality Reduction:** UMAP (768 → 50 dims for clustering, 768 → 2 dims for visualization)
4. **Clustering:** HDBSCAN (min_cluster_size=100, min_samples=20)
5. **Problem Detection:** Regex patterns for questions, frustration signals, and high-value keywords
6. **Trend Analysis:** Linear regression on monthly counts with Mann-Kendall significance testing

---

*Report generated by Topic Analysis Pipeline for Dasher.ai*
"""

    return report


def main():
    print("=" * 60)
    print("Phase 5: Insight Generation")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Load clusters
    print("\nLoading clusters...")
    with open(CLUSTERS_FILE, "r") as f:
        data = json.load(f)
    clusters = data["clusters"]
    print(f"  Loaded {len(clusters)} clusters")

    # Calculate priority scores
    print("\nCalculating priority scores...")
    max_problem = max(c.get("problem_data", {}).get("avg_problem_score", 0) for c in clusters)
    max_size = max(c.get("size", 0) for c in clusters)

    for cluster in clusters:
        # Add topic label
        cluster["topic_label"] = infer_topic_label(cluster)

        # Add priority score
        cluster["priority_score"] = calculate_priority_score(cluster, max_problem, max_size)

        # Add priority tier
        cluster["priority_tier"] = classify_priority(cluster)

        # Add Dasher recommendation
        cluster["dasher_recommendation"] = generate_dasher_recommendation(cluster)

    # Sort by priority score
    clusters.sort(key=lambda x: x["priority_score"], reverse=True)

    # Generate priority matrix CSV
    print("\nGenerating priority matrix...")
    csv_rows = []
    for c in clusters:
        pd = c.get("problem_data", {})
        ts = c.get("temporal", {}).get("trend_stats", {})
        csv_rows.append({
            "cluster_id": c["cluster_id"],
            "topic": c["topic_label"],
            "size": c["size"],
            "priority_tier": c["priority_tier"],
            "priority_score": round(c["priority_score"], 3),
            "problem_score": round(pd.get("avg_problem_score", 0), 2),
            "frustration_pct": round(pd.get("frustration_percentage", 0), 1),
            "high_value_pct": round(pd.get("high_value_percentage", 0), 1),
            "trend": ts.get("trend", "N/A"),
            "growth_rate": round(ts.get("growth_rate", 0) * 100, 1),
            "keywords": ", ".join(c["keywords"][:5]),
        })

    with open(DATA_DIR / "priority_matrix.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved priority_matrix.csv")

    # Generate insights report
    print("\nGenerating insights report...")
    report = generate_report(clusters)
    with open(OUTPUT_DIR / "INSIGHTS_REPORT.md", "w") as f:
        f.write(report)
    print(f"  Saved INSIGHTS_REPORT.md")

    # Save final enriched clusters
    with open(DATA_DIR / "clusters_prioritized.json", "w") as f:
        json.dump({"clusters": clusters}, f, indent=2)
    print(f"  Saved clusters_prioritized.json")

    # Summary
    print("\n" + "=" * 60)
    print("Priority Summary")
    print("=" * 60)

    tiers = {}
    for c in clusters:
        tier = c["priority_tier"]
        tiers[tier] = tiers.get(tier, 0) + 1

    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count} clusters")

    print("\n" + "=" * 60)
    print("Top 10 Priority Clusters")
    print("=" * 60)

    for i, c in enumerate(clusters[:10]):
        pd = c.get("problem_data", {})
        print(f"\n{i+1}. [{c['priority_tier']}] {c['topic_label']}")
        print(f"   Size: {c['size']:,} | Problem: {pd.get('avg_problem_score', 0):.2f} | High-Value: {pd.get('high_value_percentage', 0):.0f}%")
        print(f"   Dasher Feature: {c['dasher_recommendation'].get('feature', 'N/A')}")

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 60)
    print("Phase 5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
