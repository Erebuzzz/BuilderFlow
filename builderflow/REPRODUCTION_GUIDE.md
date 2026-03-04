# BuilderFlow — Reproduction Guide

> Step-by-step instructions to reproduce the BuilderFlow analysis on the Zerve platform.

## Prerequisites

1. **Zerve Account**: Sign up at [zerve.ai](https://zerve.ai) or use existing account
2. **Hackathon Registration**: Register at [zerve2026.hackerearth.com](https://zerve2026.hackerearth.com) to receive credits
3. **Dataset**: `zerve_hackathon_for_reviewc8fa7c7.csv` (available in Zerve's Example Datasets)

---

## 1. Project Setup

1. Create a **New Canvas** named `BuilderFlow`
2. Navigate to the **Data** tab and locate `zerve_hackathon_for_reviewc8fa7c7.csv`
3. Confirm the CSV is accessible (it should be pre-mounted in the project root)

---

## 2. Canvas Block Execution Order

The canvas contains **20 Python blocks + 1 Markdown block** organized in a DAG. Execute blocks in the order below. Each block's code can be found in the `Development/` folder.

### Phase 1: Data Loading & Feature Engineering

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 1 | `project_config_setup` | `project_config_setup.py` | Defines constants (cohort cutoff, early window, temporal splits) and loads raw CSV |
| 2 | `load_and_prepare_cohort` | `load_and_prepare_cohort.py` | Parses timestamps, creates user IDs, computes per-user timelines, filters to cohort |
| 3 | `feature_engineering_7d_window` | `feature_engineering_7d_window.py` | Engineers 51 features from first 7 days, adds labels, removes zero-variance and high-correlation features, target-encodes categoricals |
| 4 | `feature_schema_and_heatmap` | `feature_schema_and_heatmap.py` | Logs feature schema to JSON, renders correlation heatmap |

### Phase 2: EDA (run in any order after Phase 1)

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 5a | `eda_event_taxonomy` | `eda_event_taxonomy.py` | Bar charts of top 30 event types |
| 5b | `eda_user_timelines` | `eda_user_timelines.py` | Weekly arrivals, active days distribution |
| 5c | `eda_retention_by_behavior` | `eda_retention_by_behavior.py` | Retention curves by AI agent use, block runs, etc. |

### Phase 3: Clustering & Modeling (parallel branches)

**Branch A — Archetype Clustering:**

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 6 | `kmeans_archetype_clustering` | `kmeans_archetype_clustering.py` | KMeans k∈{4,5,6}, selects k=6 via silhouette, assigns archetype names |

**Branch B — XGBoost Model:**

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 7 | `xgboost_bayesian_opt_model` | `xgboost_bayesian_opt_model.py` | 30-trial Bayesian HPO, isotonic calibration, rolling CV, baselines |

**Branch C — Baseline Models (runs from `compute_labels_and_features`):**

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 6c | `compute_labels_and_features` | `compute_labels_and_features.py` | Creates `modeling_df` with labels and features |
| 7c | `train_baseline_and_main_models` | `train_baseline_and_main_models.py` | Trains constant, LR, and GBT baselines across 3 targets |
| 7d | `behavioral_clustering` | `behavioral_clustering.py` | Alternative clustering on full feature matrix |

### Phase 4: Interpretability

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 8 | `shap_advanced_analysis` | `shap_advanced_analysis.py` | Global SHAP, beeswarm, interactions, per-user force plots, bootstrap stability, ablation study |
| 9 | `shap_analysis` | `shap_analysis.py` | SHAP values for baseline GBT model |

### Phase 5: Business Actionability

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 10 | `uplift_intervention_scoring` | `uplift_intervention_scoring.py` | Scores all users, estimates uplift for 4 interventions, ranks by priority |
| 11 | `propensity_impact_analysis` | `propensity_impact_analysis.py` | Propensity score stratification for causal analysis |

### Phase 6: Reporting

| Step | Block Name | File | Description |
|------|-----------|------|-------------|
| 12 | `executive_summary_charts` | `executive_summary_charts.py` | One-page executive dashboard with all key charts |
| 13 | `calibration_and_comparison_charts` | `calibration_and_comparison_charts.py` | Model calibration plots |
| 14 | `executive_narrative` | `executive_narrative.md` | Written findings report (Markdown block) |

---

## 3. DAG Edges (Key Dependencies)

Variables flow across blocks via Zerve's shared namespace. Critical connections:

```
project_config_setup → load_and_prepare_cohort → feature_engineering_7d_window
    → feature_schema_and_heatmap → {
        kmeans_archetype_clustering → xgboost_bayesian_opt_model → {
            shap_advanced_analysis
            uplift_intervention_scoring
        }
        compute_labels_and_features → {
            train_baseline_and_main_models → {
                calibration_and_comparison_charts
                shap_analysis → executive_summary_charts
            }
            behavioral_clustering → executive_summary_charts
            propensity_impact_analysis
        }
        eda_event_taxonomy
        eda_user_timelines
        eda_retention_by_behavior
    }
```

---

## 4. Activating the Scheduled Job

1. Go to the **Add Layer** tab in the Canvas
2. Click **Scheduled Jobs**
3. Create a new Scheduled Job layer with the code from `ScheduledJob/retention_scoring_job.py`
4. Configure schedule: **Daily at 06:00 UTC**
5. **Activate** the job layer
6. Verify the first run produces `scored_users_latest.csv`

---

## 5. Expected Outputs

After running all blocks, you should see:

| Output | Description |
|--------|-------------|
| `feature_schema_7d.json` | Feature schema with 51 features |
| `scored_user_table.csv` | 1,472 users with retention predictions and uplift scores |
| `scored_users_latest.csv` | Latest scheduled job output |
| 15+ inline plots | PR/ROC curves, SHAP charts, cluster analysis, uplift heatmaps |
| Console metrics | PR-AUC 0.269, Lift@10% 2.56×, 6 archetypes, 4 ranked interventions |

---

## 6. Submission Checklist

- [ ] All blocks execute without error (top-to-bottom)
- [ ] Scheduled Job layer is activated and ran at least once
- [ ] `executive_narrative.md` is visible as final report in the Canvas
- [ ] Share canvas link via Zerve's **Publish** button
- [ ] Submit link on HackerEarth
- [ ] (Optional) Record 3-min video walkthrough
- [ ] (Optional) Share on LinkedIn/X with @ZerveAI
