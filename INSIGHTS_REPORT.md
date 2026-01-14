# Topic Analysis Insights Report

**Generated:** 2026-01-14 05:53

## Executive Summary

Analyzed **505,211 chunks** from 6 data sources (Reddit + Discord) and discovered **383 topic clusters**.

### Priority Distribution

| Tier | Count | Description |
|------|-------|-------------|
| P0-Critical | 0 | High pain + Growing trend |
| P1-Important | 24 | High pain OR High-value content |
| P2-Monitor | 312 | Moderate interest |
| P3-Deprioritize | 47 | Declining topics |

---

## Top Problem Clusters (High Pain Points)

These clusters represent the biggest pain points in the running community:

### 1. Knee Pain & Injuries

- **Cluster Size:** 163 discussions
- **Problem Score:** 10.77
- **Frustration Level:** 91.9%
- **High-Value Signals:** 97.5%
- **Keywords:** knee, pain, week, running, run

### 2. Beginner Running

- **Cluster Size:** 151 discussions
- **Problem Score:** 9.04
- **Frustration Level:** 68.2%
- **High-Value Signals:** 98.7%
- **Keywords:** running, start, run, weight, walking

### 3. Calf Pain

- **Cluster Size:** 163 discussions
- **Problem Score:** 8.91
- **Frustration Level:** 88.3%
- **High-Value Signals:** 79.6%
- **Keywords:** running, calves, run, week, pain

### 4. Topic: loss, running, calories

- **Cluster Size:** 156 discussions
- **Problem Score:** 8.77
- **Frustration Level:** 64.7%
- **High-Value Signals:** 100.0%
- **Keywords:** loss, running, calories, weight loss, diet

### 5. Topic: week, run, just

- **Cluster Size:** 374 discussions
- **Problem Score:** 8.71
- **Frustration Level:** 75.7%
- **High-Value Signals:** 100.0%
- **Keywords:** week, run, just, day, running

### 6. Knee Pain & Injuries

- **Cluster Size:** 142 discussions
- **Problem Score:** 8.07
- **Frustration Level:** 95.8%
- **High-Value Signals:** 67.6%
- **Keywords:** knee, pain, knees, run, week

### 7. Race Pacing

- **Cluster Size:** 630 discussions
- **Problem Score:** 7.37
- **Frustration Level:** 47.0%
- **High-Value Signals:** 98.0%
- **Keywords:** run, running, pace, pts, 5k

### 8. Beginner Running

- **Cluster Size:** 295 discussions
- **Problem Score:** 7.32
- **Frustration Level:** 70.1%
- **High-Value Signals:** 80.9%
- **Keywords:** run, weight, start, just, walking

### 9. Topic: run, finished, running

- **Cluster Size:** 246 discussions
- **Problem Score:** 7.23
- **Frustration Level:** 59.8%
- **High-Value Signals:** 97.6%
- **Keywords:** run, finished, running, today, just

### 10. Running Pain & Injuries

- **Cluster Size:** 345 discussions
- **Problem Score:** 7.12
- **Frustration Level:** 78.8%
- **High-Value Signals:** 87.9%
- **Keywords:** pts, pain, running, run, just

---

## Growing Topics (Opportunity Zones)

These topics are increasing in discussion volume:

- **Topic: pts, running, run**: +8.5% growth (242 chunks)
- **Beginner Running**: +5.0% growth (518 chunks)
- **Training Plans**: +5.8% growth (233 chunks)


---

## Declining Topics (Avoid Building For)

These topics are decreasing in relevance:

- **Topic: pts, comments, com**: -26.8% decline (154 chunks)
- **Topic: bot_metric, www, https www**: -13.5% decline (154 chunks)
- **Topic: miles, mi, week**: -13.2% decline (145 chunks)
- **Topic: pts, day, vaccine**: -12.3% decline (101 chunks)
- **Topic: running, people, run**: -11.2% decline (250 chunks)


---

## Dasher Feature Recommendations

Based on the analysis, here are the top feature opportunities for Dasher:

### 1. Injury Prevention Challenges

- **Source Topic:** Knee Pain & Injuries
- **Description:** Challenges focused on low-impact running, strength exercises, and gradual mileage increase to prevent knee injuries
- **Implementation Complexity:** Medium
- **User Impact:** High

### 2. Beginner-Friendly Challenges

- **Source Topic:** Beginner Running
- **Description:** Low-stake, short-duration challenges designed for absolute beginners with walk/run intervals
- **Implementation Complexity:** Low
- **User Impact:** High

### 3. Content Hub: Calf Pain

- **Source Topic:** Calf Pain
- **Description:** Curated advice and challenges related to calf pain
- **Implementation Complexity:** Low
- **User Impact:** Low

### 4. Content Hub: Topic: loss, running, calories

- **Source Topic:** Topic: loss, running, calories
- **Description:** Curated advice and challenges related to topic: loss, running, calories
- **Implementation Complexity:** Low
- **User Impact:** Low

### 5. Content Hub: Topic: week, run, just

- **Source Topic:** Topic: week, run, just
- **Description:** Curated advice and challenges related to topic: week, run, just
- **Implementation Complexity:** Low
- **User Impact:** Low

### 6. Shin Splint Recovery Program

- **Source Topic:** Shin Splints
- **Description:** Guided recovery challenges with rest days, cross-training, and gradual return-to-running plans
- **Implementation Complexity:** Medium
- **User Impact:** High

### 7. Race Prep Challenges

- **Source Topic:** Race Pacing
- **Description:** Goal-race specific challenges with pace targets, taper weeks, and race day readiness tracking
- **Implementation Complexity:** Medium
- **User Impact:** High

### 8. Content Hub: Topic: run, running, week

- **Source Topic:** Topic: run, running, week
- **Description:** Curated advice and challenges related to topic: run, running, week
- **Implementation Complexity:** Low
- **User Impact:** Low

### 9. Content Hub: Running Pain & Injuries

- **Source Topic:** Running Pain & Injuries
- **Description:** Curated advice and challenges related to running pain & injuries
- **Implementation Complexity:** Low
- **User Impact:** Low

### 10. Content Hub: Topic: run, finished, running

- **Source Topic:** Topic: run, finished, running
- **Description:** Curated advice and challenges related to topic: run, finished, running
- **Implementation Complexity:** Low
- **User Impact:** Low

---

## Methodology

1. **Data Sources:** Reddit (r/running, r/C25K, r/beginnerrunning) + Discord (beginner_running, running_questions, running_science)
2. **Embedding Model:** Google text-embedding-004 (768 dimensions)
3. **Dimensionality Reduction:** UMAP (768 → 50 dims for clustering, 768 → 2 dims for visualization)
4. **Clustering:** HDBSCAN (min_cluster_size=100, min_samples=20)
5. **Problem Detection:** Regex patterns for questions, frustration signals, and high-value keywords
6. **Trend Analysis:** Linear regression on monthly counts with Mann-Kendall significance testing

---

*Report generated by Topic Analysis Pipeline for Dasher.ai*
