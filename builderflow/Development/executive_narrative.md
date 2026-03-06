# 📋 BuilderFlow — Executive Summary: Zerve User Activation & Retention Analysis

## Study Design
**Cohort:** 1,472 users observed during their **first 7 days** on the Zerve platform, with outcomes tracked at 30-day and 90-day retention, plus 60-day upgrade horizons. Features were engineered from 226.6K behavioral events across 51 leakage-free dimensions including session patterns, feature adoption ratios, time-to-first actions, and platform mix. Models were evaluated on a **temporal hold-out test set** (Nov 1–8 2025) to simulate real deployment conditions.

**Primary Objective:** 30-Day Retention (1.1% base rate on test set — extremely sparse target)

**Zero Leakage Guarantee:** All features are computed strictly from the first 7 days of user activity. Retention labels are defined as any activity in days 8+. No future-looking features were used.

---

## 🏆 Top 5 Drivers of Retention (by SHAP importance)

| Rank | Driver | Mean \|SHAP\| | Interpretation |
|------|--------|-------------|----------------|
| **#1** | **Active Days** | 0.229 | Number of distinct active days in first week — strongest engagement signal |
| **#2** | **Max Gap Days** | 0.093 | Longest dormancy between sessions — proxy for habit formation |
| **#3** | **Agent Usage Ratio** | 0.085 | Proportion of AI agent events — high values indicate AI-dependency without productive workflow |
| **#4** | **Time-of-Day Entropy** | 0.073 | Diversity of activity timing — regular users show lower entropy |
| **#5** | **Number of Sessions** | 0.063 | Session count correlates with active engagement and multi-day return |

> **Cross-Target Strong Signals** (top-10 in ≥2 targets): Active Days, Max Gap, Agent Ratio, Session Entropy, Event Count, Sessions, Onboarding Completed — these features are robust predictors of both retention and upgrade.

> **Bootstrap Stability:** All top 15 features classified as "Very Stable" (CV < 0.25 across 20 bootstrap resamples). Strongest SHAP interaction: `active_days × max_gap_days` (0.983 correlation).

---

## 📊 Model Performance

| Metric | XGB Calibrated (Primary) | L2 LogReg | Naive Baseline |
|--------|--------------------------|-----------|----------------|
| **PR-AUC** | **0.2685** | 0.0418 | 0.0109 |
| **ROC-AUC** | 0.7572 | 0.8036 | 0.5000 |
| **Brier Score** | **0.0128** | — | — |
| **Lift @ Top 10%** | **2.56×** | — | — |
| **Lift @ Top 20%** | **2.52×** | — | — |

The isotonic-calibrated XGBoost model captures meaningful signal despite the extremely low base rate (1.1%). **The top 10% of model-scored users captures 2.56× the baseline retention rate** — a usable targeting signal for lifecycle campaigns.

**Rolling CV stability:** PR-AUC 0.2488 ± 0.0577 across 3 temporal windows, confirming generalizability.

> We optimized for **PR-AUC** rather than ROC-AUC because the positive class is rare (~1%), making precision-recall the appropriate evaluation metric. The L2 LogReg has higher ROC-AUC (0.804) but much lower PR-AUC (0.042), meaning it cannot effectively identify the sparse positive class.

---

## 🧩 Behavioral Archetypes & Outcome Rates

KMeans identified **6 stable behavioral segments** (silhouette = 0.518, ARI consistency = 0.997):

| Archetype | Users | Share | Ret 30d | Ret 90d | Upgrade 60d | Risk Level |
|-----------|-------|-------|---------|---------|-------------|------------|
| **Hands-On Builder** | 16 | 1.1% | 81.2% | 81.2% | 25.0% | 🟢 Low |
| **Power User** | 88 | 6.0% | 44.3% | 53.4% | 11.4% | 🟢 Low |
| **Mid-Tier Engager** | 44 | 3.0% | 57.0% | 11.9% | 2.9% | 🟡 Medium |
| **Onboarding Visitor** | 297 | 20.2% | 5.7% | 7.1% | 1.0% | 🟠 High |
| **Casual Browser** | 626 | 42.5% | 4.2% | 5.3% | 1.0% | 🔴 Critical |
| **OVERALL** | 1,472 | 100% | — | — | — | — |

**Key Insight:** The **Hands-On Builder** archetype (1.1% of users) delivers **81% retention** and **25% upgrade rate** — 19× and 25× the Casual Browser majority. These users combine multi-day engagement, block execution, canvas creation, and high session entropy. They are Zerve's power users and the archetype other segments should be guided toward.

---

## 🔬 Ablation Study — Feature Group Impact

| Group Dropped | PR-AUC Change | Lift@10% Change | Impact |
|--------------|---------------|-----------------|--------|
| Advanced Usage | **-0.0263** | -0.302× | ⚠️ Largest impact — agent/block/canvas ratios are critical |
| Metadata | -0.0091 | -0.051× | Geographic and signup-time signals matter |
| Collaboration | -0.0053 | -0.025× | Canvas sharing/edge creation adds marginal signal |
| Intensity | -0.0024 | +0.107× | Session/event counts are resilient — other groups compensate |

---

## 🎯 Intervention Priority Ranking

Based on uplift estimation with archetype-level calibration, persuadability scoring, and composite priority weighting:

| # | Intervention | Priority Score | Addressable Users | Avg Uplift | Expected New Retentions |
|---|-------------|---------------|-------------------|------------|------------------------|
| **1** | **Agent→Block Conversion UI** | 0.831 | 1,071 (73%) | 2.85pp | +21 |
| **2** | **Day 1/3/7 Email Drip** | 0.820 | 923 (63%) | 3.71pp | +20 |
| **3** | **Session Milestone Checklist** | 0.771 | 742 (50%) | 3.51pp | +16 |
| **4** | **Onboarding→Build Nudge** | 0.734 | 594 (40%) | 2.42pp | +8 |

**Recommended focus:** Agent→Block Conversion UI flow targets 73% of user base with highest composite priority. Both top interventions are low-engineering-cost, fast-to-market.

---

## 🏗️ Deployed Functionality

**Interactive Streamlit Dashboard:** A live web application deployed on Zerve that:
- Reads the scored user retention data and calculated uplift metrics
- Visualizes the retention risk distribution across behavioral archetypes using interactive Plotly charts
- Displays the overall risk tier breakdown (Low / Medium / High / Critical)
- Provides a prioritized targeting list of the top 50 users showing the highest potential uplift for the "Agent → Block UI" intervention

This enables the product and marketing teams to monitor user health dynamically and immediately action high-priority interventions.

---

## 📌 Product Recommendations

### 1. Agent→Build Bridge (Priority #1)
When AI Agent generates code, add a prominent **"Run & See Results"** button that auto-creates a block from agent output. Track `agent_to_block_conversion` as a product KPI — currently 37% of users show this behavior.

### 2. Day 1/3/7 Email Drip (Priority #2)
Timed email sequence segmented by archetype: Casual Browsers get "Getting Started" content; Power Users get "Advanced Agent Workflows."

### 3. Activation Checklist (Priority #3)
Visible **progress checklist**: ☐ Create canvas ☐ Add a block ☐ Run a block ☐ Connect to data ☐ Return on Day 2. Gamify with credit incentives at each milestone.

### 4. Onboarding-to-Build Nudge (Priority #4)
After completing the product tour, prompt with a **"Build Your First Canvas"** wizard with pre-populated sample data + one block.

### 5. Geographic GTM Optimization
Country is a SHAP driver — investigate pricing localization for high-potential markets.

---

## Closing Thesis

> **The strongest predictor of retention is not feature exposure, but the early transition from passive exploration to active construction.** Users who execute blocks, create canvases, and return across multiple days form the Hands-On Builder archetype — achieving 81% retention vs. 4% for Casual Browsers. The path from churn to retention is bridging AI-generated insights → executable workflows within the first 7 days.
