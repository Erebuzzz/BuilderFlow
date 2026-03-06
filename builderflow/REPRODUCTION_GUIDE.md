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

## 2. Recreating the Canvas (Manual Block Creation)

Since Zerve currently requires manual setup via the UI, you will need to create the blocks one by one using the `+` icon in the canvas.

1. Hover below the active area and click the **`+` icon**
2. Select **Python** for code blocks, or **Markdown** for the executive narrative
3. Paste the code from the corresponding file in the repository
4. Name the block using the three dots `...` menu on the top right of the block
5. Run the block (Shift+Enter or Play button) before creating the next one

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
| 12 | `executive_summary_charts` | `executive_summary_charts.py` | One-page executive dashboard with all key charts (Python block) |
| 13 | `calibration_and_comparison_charts` | `calibration_and_comparison_charts.py` | Model calibration plots (Python block) |
| 14 | `executive_narrative` | `executive_narrative.md` | Written findings report (**Markdown block**) |

---

## 3. DAG Edges (Key Dependencies)

Because Zerve uses a shared namespace, you don't need to manually wire edges right away — simply creating and running the blocks in the order above will implicitly resolve the dependencies.

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

## 4. Deploying the Streamlit Application

To fulfill the "deployed functionality" requirement, we use Zerve's Web Application deployment feature.

1. In Zerve, click the **Deployments** (rocket icon) on the left sidebar
2. Click **Create Deployment**
3. Choose **Streamlit** (under Web Applications)
4. Name the deployment `BuilderFlow-Dashboard`
5. Paste the code from `streamlit_app.py` into the deployment's code editor
6. Click **Deploy**
7. Once active, open the URL to view the interactive retention risk dashboard!

*(Note: The Streamlit app relies on `scored_user_table.csv`, which is generated by the `uplift_intervention_scoring` block in Phase 5. Ensure that block has run before deploying.)*

---

## 5. Expected Outputs

After running all blocks and deploying, you should see:

| Output | Description |
|--------|-------------|
| `feature_schema_7d.json` | Feature schema with 51 features |
| `scored_user_table.csv` | 1,472 users with retention predictions and uplift scores |
| Streamlit Dashboard | Live web application showing interactive risk distributions and priority targets |
| 15+ inline plots | PR/ROC curves, SHAP charts, cluster analysis, uplift heatmaps |
| Console metrics | PR-AUC 0.269, Lift@10% 2.56×, 6 archetypes, 4 ranked interventions |

---

## 6. Submission Checklist

- [ ] All Canvas blocks execute without error (top-to-bottom)
- [ ] Zerve Streamlit Deployment is active and loads the dashboard
- [ ] `executive_narrative.md` is visible as the final written report in the Canvas
- [ ] Share canvas link via Zerve's **Publish** button
- [ ] Submit link on HackerEarth
- [ ] (Optional) Record 3-min video walkthrough of the Canvas and Streamlit dashboard
- [ ] (Optional) Share on LinkedIn/X with @ZerveAI
