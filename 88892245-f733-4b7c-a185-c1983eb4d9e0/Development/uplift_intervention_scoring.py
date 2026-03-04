
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
_BG   = "#1D1D20"; _TXT  = "#fbfbff"; _TXT2 = "#909094"
_COLS = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
         "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
_HL = "#ffd400"; _GREEN = "#17b26a"; _WARN = "#f04438"

print("═"*75)
print("🎯  CALIBRATED USER SCORING + UPLIFT INTERVENTION RANKING")
print("═"*75)

# ═══════════════════════════════════════════════════════════════
# 0. ASSEMBLE FULL USER DATASET
#    feature_matrix_with_archetypes: (1.5k, 58) — has archetype, cluster_id,
#    all 51 features + ret30d, ret90d, upg60d, split
#    gbt_best_model + gbt_isotonic_calib from xgboost_bayesian_opt_model
# ═══════════════════════════════════════════════════════════════
_LABEL_COLS = ["ret30d", "ret90d", "upg60d"]
_META_COLS  = ["user_id_canon", "split", "cluster_id", "archetype"]
_feat_cols  = [c for c in feature_matrix_with_archetypes.columns
               if c not in _LABEL_COLS + _META_COLS]

_all_X = feature_matrix_with_archetypes[_feat_cols].values.astype(np.float32)

# ── Calibrated retention probability (upgrade/retention risk) ──
# Use isotonic-calibrated XGBoost: gbt_best_model → gbt_isotonic_calib
_raw_proba_all   = gbt_best_model.predict_proba(_all_X)[:, 1]
_calib_proba_all = gbt_isotonic_calib.predict(_raw_proba_all)

# Invert: model predicts retention; predicted_risk = 1 - calib_retention_prob
# (risk of churn / not retaining)
_predicted_risk_all = 1.0 - _calib_proba_all

# Also get upgrade probability proxy — use upg60d label rates by archetype
# as a group-level calibrated upgrade probability
_arch_upgrade_rate = (
    feature_matrix_with_archetypes.groupby("archetype")["upg60d"]
    .mean()
    .to_dict()
)

scoring_df = feature_matrix_with_archetypes[[
    "user_id_canon", "archetype", "cluster_id",
    "feat_ratio_agent", "feat_ratio_block_ops", "feat_ratio_onboarding",
    "feat_n_sessions", "feat_active_days", "feat_event_count",
    "feat_onboarding_completed", "feat_onboarding_skipped",
    "feat_agent_block_conversion", "feat_ttf_run_block",
    "feat_ttf_agent_use", "feat_signup_hour", "feat_te_country",
    "ret30d", "ret90d", "upg60d", "split"
]].copy()

scoring_df["predicted_retention_prob"] = _calib_proba_all
scoring_df["predicted_risk"]           = _predicted_risk_all  # 1 - retention
scoring_df["agent_usage_ratio"]        = scoring_df["feat_ratio_agent"]
scoring_df["predicted_upgrade_prob"]   = scoring_df["archetype"].map(_arch_upgrade_rate)

print(f"\n   Total users scored: {len(scoring_df):,}")
print(f"   Calibrated retention prob: mean={_calib_proba_all.mean():.4f}, "
      f"std={_calib_proba_all.std():.4f}")
print(f"   Predicted risk (churn):    mean={_predicted_risk_all.mean():.4f}, "
      f"std={_predicted_risk_all.std():.4f}")

# Print scored distribution by archetype
print("\n   ── Scores by Archetype ──")
print(f"   {'Archetype':25s} {'N':>5s}  {'Ret_Prob':>9s}  {'Risk':>7s}  {'Upg_Prob':>9s}  {'AgentRatio':>10s}")
print("   " + "─"*73)
for _arch in scoring_df["archetype"].value_counts().index:
    _sub = scoring_df[scoring_df["archetype"] == _arch]
    print(f"   {_arch:25s} {len(_sub):5d}  "
          f"{_sub['predicted_retention_prob'].mean():9.4f}  "
          f"{_sub['predicted_risk'].mean():7.4f}  "
          f"{_sub['predicted_upgrade_prob'].mean():9.4f}  "
          f"{_sub['agent_usage_ratio'].mean()*100:9.1f}%")

# ═══════════════════════════════════════════════════════════════
# 1. ARCHETYPE BASELINES (from cluster_outcome_rates)
# ═══════════════════════════════════════════════════════════════
_ARCHETYPE_BASELINES = {
    row["archetype"]: {
        "n_users": int(row["n_users"]),
        "ret30d":  float(row["ret30d"]),
        "ret90d":  float(row["ret90d"]),
        "upg60d":  float(row["upg60d"]),
    }
    for _, row in cluster_outcome_rates.iterrows()
}

# Observational uplift signals (from propensity_impact_analysis style estimate)
# Based on archetype outcome data + discount for observational → causal
_CAUSAL_DISCOUNT = 0.40  # 40% discount: observational estimates are optimistic

# Calibrated uplift signals per behavior from cluster_outcome_rates and domain reasoning
_exec_uplift_ret30_pp = 18.0  # High execution behavior → +18pp ret30d (empirical)
_exec_uplift_ret90_pp = 25.0  # High execution → +25pp ret90d
_collab_uplift_ret90_pp = 25.2  # Early collaboration → +25.2pp ret90d

print("\n\n═"*75)
print("📊 ARCHETYPE BASELINES")
print("═"*75)
print(f"{'Archetype':30s} {'N':>5s}  {'ret30d':>7s}  {'ret90d':>7s}  {'upg60d':>7s}")
print("─"*60)
for _arch, _b in sorted(_ARCHETYPE_BASELINES.items(), key=lambda x: -x[1]["ret90d"]):
    print(f"{_arch:30s} {_b['n_users']:5d}  {_b['ret30d']*100:6.1f}%  "
          f"{_b['ret90d']*100:6.1f}%  {_b['upg60d']*100:6.1f}%")

# ═══════════════════════════════════════════════════════════════
# 2. INTERVENTION DEFINITIONS
#    4 interventions, each with archetype-level uplift estimates
# ═══════════════════════════════════════════════════════════════
INTERVENTIONS = [
    {
        "name": "onboarding_to_build_nudge",
        "label": "Onboarding→Build Nudge",
        "description": "In-product prompt at onboarding completion guiding users to run their first block",
        "mechanism": "Bridges onboarding→execution gap (feat_ttf_run_block acceleration)",
        "primary_target": "Onboarding Visitor",
        "secondary_target": "Casual Browser",
        "target_metric": "ret30d",
        "uplift_by_archetype": {
            "Onboarding Visitor": _exec_uplift_ret30_pp * _CAUSAL_DISCOUNT * 0.80,  # 5.8pp
            "Casual Browser":     _exec_uplift_ret30_pp * _CAUSAL_DISCOUNT * 0.50,  # 3.6pp
            "Mid-Tier Engager":   _exec_uplift_ret30_pp * _CAUSAL_DISCOUNT * 0.20,  # 1.4pp
            "Hands-On Builder":   0.5,   # already building
            "Power User":         0.3,   # already retained
        },
        "engineering_cost": 1,    # Low (in-product nudge)
        "time_to_impact": 1,      # Fast (<1 week to ship)
        "confidence": 0.72,
    },
    {
        "name": "agent_to_block_conversion_UI_flow",
        "label": "Agent→Block Conversion UI",
        "description": "UI flow surfacing block creation from agent output (targets 85%+ agent-usage users)",
        "mechanism": "Converts AI-only sessions into block execution; reduces feat_ratio_agent dominance",
        "primary_target": "Casual Browser",
        "secondary_target": "Mid-Tier Engager",
        "target_metric": "ret90d",
        "uplift_by_archetype": {
            "Casual Browser":     _exec_uplift_ret30_pp * _CAUSAL_DISCOUNT * 1.00,  # 7.2pp
            "Mid-Tier Engager":   _exec_uplift_ret30_pp * _CAUSAL_DISCOUNT * 0.40,  # 2.9pp
            "Onboarding Visitor": _exec_uplift_ret30_pp * _CAUSAL_DISCOUNT * 0.20,  # 1.4pp
            "Hands-On Builder":   0.5,
            "Power User":         0.3,
        },
        "engineering_cost": 2,    # Medium (UI/UX flow)
        "time_to_impact": 2,      # Med (2–4 weeks)
        "confidence": 0.68,
    },
    {
        "name": "session_milestone_checklist",
        "label": "Session Milestone Checklist",
        "description": "Per-session progress checklist: run block, connect data, deploy, collaborate",
        "mechanism": "Drives multi-session habit (feat_n_sessions ↑, feat_active_days ↑)",
        "primary_target": "Mid-Tier Engager",
        "secondary_target": "Onboarding Visitor",
        "target_metric": "ret90d",
        "uplift_by_archetype": {
            "Mid-Tier Engager":   _exec_uplift_ret90_pp * _CAUSAL_DISCOUNT * 0.70,  # 7.0pp
            "Onboarding Visitor": _exec_uplift_ret90_pp * _CAUSAL_DISCOUNT * 0.35,  # 3.5pp
            "Casual Browser":     _exec_uplift_ret90_pp * _CAUSAL_DISCOUNT * 0.25,  # 2.5pp
            "Hands-On Builder":   0.5,
            "Power User":         0.3,
        },
        "engineering_cost": 1,    # Low (frontend checklist)
        "time_to_impact": 2,      # Med (habit formation)
        "confidence": 0.65,
    },
    {
        "name": "day1_day3_day7_email_drip",
        "label": "Day 1/3/7 Email Drip",
        "description": "Timed email sequence: Day 1 (build nudge), Day 3 (deploy), Day 7 (collaborate)",
        "mechanism": "Re-engagement across at-risk windows; surfaces collaboration + deployment",
        "primary_target": "Onboarding Visitor",
        "secondary_target": "Casual Browser",
        "target_metric": "ret90d",
        "uplift_by_archetype": {
            "Onboarding Visitor": _collab_uplift_ret90_pp * _CAUSAL_DISCOUNT * 0.65,  # 6.6pp
            "Casual Browser":     _collab_uplift_ret90_pp * _CAUSAL_DISCOUNT * 0.55,  # 5.5pp
            "Mid-Tier Engager":   _collab_uplift_ret90_pp * _CAUSAL_DISCOUNT * 0.35,  # 3.5pp
            "Hands-On Builder":   1.0,
            "Power User":         0.5,
        },
        "engineering_cost": 1,    # Low (email drip)
        "time_to_impact": 1,      # Fast (auto-scheduled)
        "confidence": 0.60,
    },
]

# ═══════════════════════════════════════════════════════════════
# 3. PER-USER UPLIFT SCORING
#    uplift_pp = archetype_base_pp × confidence × persuadability(risk) × agent_bonus
#    persuadability = 4 * risk * (1-risk) — parabola peaking at risk=0.5
# ═══════════════════════════════════════════════════════════════
def _compute_user_uplift(row, intervention):
    _arch        = row["archetype"]
    _risk        = float(row["predicted_risk"])
    _arch_up_pp  = intervention["uplift_by_archetype"].get(_arch, 0.5)
    _persuade    = 4.0 * _risk * (1.0 - _risk)  # max 1.0 at risk=0.5
    _agent_bonus = 1.0 + float(row["agent_usage_ratio"]) * 0.5 \
                   if "agent" in intervention["name"] else 1.0
    return max(0.0, _arch_up_pp * intervention["confidence"] * _persuade * _agent_bonus)

for _iv in INTERVENTIONS:
    _col = "uplift_" + _iv["name"]
    scoring_df[_col] = scoring_df.apply(lambda r: _compute_user_uplift(r, _iv), axis=1)

print("\n\n═"*75)
print("📊 MEAN UPLIFT SCORES BY ARCHETYPE × INTERVENTION (pp)")
print("═"*75)
_header = f"{'Archetype':25s}" + "".join(f"{iv['label'][:18]:>20s}" for iv in INTERVENTIONS)
print(_header)
print("─"*105)
for _arch in sorted(scoring_df["archetype"].dropna().unique()):
    _sub = scoring_df[scoring_df["archetype"] == _arch]
    _row_str = f"{_arch:25s}"
    for _iv in INTERVENTIONS:
        _upcol = "uplift_" + _iv["name"]
        _row_str += f"{_sub[_upcol].mean():>20.2f}"
    print(_row_str)

# ═══════════════════════════════════════════════════════════════
# 4. PRIORITY FORMULA
#    score = 25%×segment_size + 45%×uplift_potential + 20%×eng_efficiency + 10%×speed
# ═══════════════════════════════════════════════════════════════
_WEIGHTS = {"segment_size": 0.25, "total_uplift": 0.45, "eng_efficiency": 0.20, "speed": 0.10}
_COST_LBL = {1: "Low", 2: "Med", 3: "High"}
_TIME_LBL  = {1: "Fast (<1wk)", 2: "Med (2-4wk)", 3: "Slow (>1mo)"}

_intervention_scores = []
_N_TOTAL = len(scoring_df)

for _iv in INTERVENTIONS:
    _upcol     = "uplift_" + _iv["name"]
    _prim_mask = scoring_df["archetype"] == _iv["primary_target"]
    _sec_mask  = scoring_df["archetype"] == _iv["secondary_target"]
    _n_addr    = (_prim_mask | _sec_mask).sum()

    _prim_n  = _ARCHETYPE_BASELINES.get(_iv["primary_target"],   {}).get("n_users", 0)
    _sec_n   = _ARCHETYPE_BASELINES.get(_iv["secondary_target"],  {}).get("n_users", 0)
    _prim_up = _iv["uplift_by_archetype"].get(_iv["primary_target"], 0)
    _sec_up  = _iv["uplift_by_archetype"].get(_iv["secondary_target"], 0)
    _expected_add_ret = (
        _prim_n * (_prim_up / 100) * _iv["confidence"] +
        _sec_n  * (_sec_up  / 100) * _iv["confidence"]
    )

    _intervention_scores.append({
        "Intervention":            _iv["label"],
        "name":                    _iv["name"],
        "description":             _iv["description"],
        "mechanism":               _iv["mechanism"],
        "primary_target":          _iv["primary_target"],
        "secondary_target":        _iv["secondary_target"],
        "n_addressable":           int(_n_addr),
        "seg_pct":                 _n_addr / _N_TOTAL * 100,
        "mean_uplift_pp":          float(scoring_df[_upcol].mean()),
        "median_uplift_pp":        float(scoring_df[_upcol].median()),
        "p75_uplift_pp":           float(scoring_df[_upcol].quantile(0.75)),
        "total_uplift_sum":        float(scoring_df[_upcol].sum()),
        "expected_add_retentions": float(_expected_add_ret),
        "eng_cost":                _iv["engineering_cost"],
        "time_to_impact":          _iv["time_to_impact"],
        "confidence":              _iv["confidence"],
        "segment_size_score":      _n_addr / _N_TOTAL,
        "eng_efficiency_score":    (4 - _iv["engineering_cost"]) / 3.0,
        "speed_score":             (4 - _iv["time_to_impact"]) / 3.0,
    })

uplift_priority_df = pd.DataFrame(_intervention_scores)
_max_up = uplift_priority_df["total_uplift_sum"].max()
uplift_priority_df["total_uplift_score_norm"] = uplift_priority_df["total_uplift_sum"] / (_max_up + 1e-9)

uplift_priority_df["priority_score"] = (
    _WEIGHTS["segment_size"]   * uplift_priority_df["segment_size_score"] +
    _WEIGHTS["total_uplift"]   * uplift_priority_df["total_uplift_score_norm"] +
    _WEIGHTS["eng_efficiency"] * uplift_priority_df["eng_efficiency_score"] +
    _WEIGHTS["speed"]          * uplift_priority_df["speed_score"]
)

uplift_priority_df = uplift_priority_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
uplift_priority_df["priority_rank"] = uplift_priority_df.index + 1

# ═══════════════════════════════════════════════════════════════
# 5. PRINT PRIORITY RANK TABLE
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "═"*120)
print("🏆  INTERVENTION PRIORITY RANKING")
print(f"    Weights: segment×{_WEIGHTS['segment_size']} | uplift×{_WEIGHTS['total_uplift']} | eng×{_WEIGHTS['eng_efficiency']} | speed×{_WEIGHTS['speed']}")
print("═"*120)
print(f"\n  {'#':>3}  {'Intervention':30s}  {'Score':>6}  {'N Users':>7}  {'Seg%':>5}  "
      f"{'AvgUp(pp)':>9}  {'Exp+Ret':>8}  {'Cost':>4}  {'Time':>12}  {'Conf':>5}")
print("─"*110)

for _, _r in uplift_priority_df.iterrows():
    print(
        f"  #{int(_r['priority_rank']):2d}  "
        f"{_r['Intervention']:30s}  "
        f"{_r['priority_score']:6.4f}  "
        f"{int(_r['n_addressable']):7d}  "
        f"{_r['seg_pct']:5.1f}%  "
        f"{_r['mean_uplift_pp']:>8.2f}pp  "
        f"{_r['expected_add_retentions']:>8.1f}  "
        f"{_COST_LBL[int(_r['eng_cost'])]:>4s}  "
        f"{_TIME_LBL[int(_r['time_to_impact'])]:>12s}  "
        f"{_r['confidence']:>5.0%}"
    )

print("\n\n📍 TOP SEGMENTS PER INTERVENTION (by total expected impact):")
print("═"*90)
for _iv in INTERVENTIONS:
    _iv_row = uplift_priority_df[uplift_priority_df["name"] == _iv["name"]].iloc[0]
    _rank   = int(_iv_row["priority_rank"])
    _upcol  = "uplift_" + _iv["name"]
    print(f"\n  #{_rank}: {_iv['label']}  [Priority Score: {_iv_row['priority_score']:.4f}]")
    print(f"     Desc: {_iv['description']}")
    print(f"     How:  {_iv['mechanism']}")
    _segs = (
        scoring_df.groupby("archetype")[_upcol]
        .agg(n_users="count", mean_uplift="mean", total_uplift="sum")
        .reset_index().sort_values("total_uplift", ascending=False).head(3)
    )
    for _, _s in _segs.iterrows():
        print(f"     ▸ {_s['archetype']:25s}: {int(_s['n_users']):4d} users | "
              f"mean_uplift={_s['mean_uplift']:.2f}pp | total_impact={_s['total_uplift']:.0f}pp×users")

# ═══════════════════════════════════════════════════════════════
# 6. SEGMENT-LEVEL METRICS: ARCHETYPE
# ═══════════════════════════════════════════════════════════════
_uplift_cols = ["uplift_" + iv["name"] for iv in INTERVENTIONS]

segment_metrics_archetype = (
    scoring_df.groupby("archetype")
    .agg(
        n_users               = ("user_id_canon", "count"),
        ret30d_actual         = ("ret30d",  "mean"),
        ret90d_actual         = ("ret90d",  "mean"),
        upg60d_actual         = ("upg60d",  "mean"),
        predicted_retention   = ("predicted_retention_prob", "mean"),
        predicted_risk        = ("predicted_risk", "mean"),
        predicted_upgrade     = ("predicted_upgrade_prob", "mean"),
        agent_usage_ratio     = ("agent_usage_ratio", "mean"),
    )
    .reset_index()
    .sort_values("n_users", ascending=False)
)

for _col in _uplift_cols:
    _iv_name  = _col.replace("uplift_", "")
    _iv_label = next((iv["label"][:20] for iv in INTERVENTIONS if iv["name"] == _iv_name), _iv_name[:20])
    segment_metrics_archetype["best_uplift_" + _iv_name[:15]] = (
        scoring_df.groupby("archetype")[_col].mean().values
    )

print("\n\n" + "═"*100)
print("📊 SEGMENT METRICS BY ARCHETYPE")
print("═"*100)
print(f"{'Archetype':25s} {'N':>5}  {'ret30d%':>7}  {'ret90d%':>7}  {'upg60d%':>7}  "
      f"{'PredRet%':>8}  {'Risk%':>6}  {'Agent%':>7}")
print("─"*83)
_ov = scoring_df  # overall
print(f"{'OVERALL':25s} {len(_ov):5}  {_ov['ret30d'].mean()*100:6.1f}%  "
      f"{_ov['ret90d'].mean()*100:6.1f}%  {_ov['upg60d'].mean()*100:6.1f}%  "
      f"{_ov['predicted_retention_prob'].mean()*100:7.1f}%  "
      f"{_ov['predicted_risk'].mean()*100:5.1f}%  "
      f"{_ov['agent_usage_ratio'].mean()*100:6.1f}%")
print("─"*83)
for _, _r in segment_metrics_archetype.iterrows():
    print(f"{_r['archetype']:25s} {int(_r['n_users']):5}  "
          f"{_r['ret30d_actual']*100:6.1f}%  {_r['ret90d_actual']*100:6.1f}%  "
          f"{_r['upg60d_actual']*100:6.1f}%  {_r['predicted_retention']*100:7.1f}%  "
          f"{_r['predicted_risk']*100:5.1f}%  {_r['agent_usage_ratio']*100:6.1f}%")

# ═══════════════════════════════════════════════════════════════
# 7. SEGMENT-LEVEL METRICS: REGION (via feat_te_country proxy)
#    feat_te_country is a smoothed target-encoded country value
#    Bucket into quintiles as proxy for regional risk tiers
# ═══════════════════════════════════════════════════════════════
_q_labels = ["Region-Q1 (Low Risk)", "Region-Q2", "Region-Q3", "Region-Q4", "Region-Q5 (High Risk)"]
scoring_df["region_bucket"] = pd.qcut(
    scoring_df["feat_te_country"].rank(method="first"),
    q=5, labels=_q_labels
)

segment_metrics_region = (
    scoring_df.groupby("region_bucket")
    .agg(
        n_users             = ("user_id_canon", "count"),
        ret30d_actual       = ("ret30d",  "mean"),
        ret90d_actual       = ("ret90d",  "mean"),
        upg60d_actual       = ("upg60d",  "mean"),
        predicted_retention = ("predicted_retention_prob", "mean"),
        predicted_risk      = ("predicted_risk", "mean"),
        agent_usage_ratio   = ("agent_usage_ratio", "mean"),
        avg_te_country      = ("feat_te_country", "mean"),
    )
    .reset_index()
)

print("\n\n" + "═"*100)
print("📊 SEGMENT METRICS BY REGION BUCKET (feat_te_country quintile proxy)")
print("   (Q1 = countries with lowest retention rates, Q5 = highest)")
print("═"*100)
print(f"{'Region Bucket':25s} {'N':>5}  {'ret30d%':>7}  {'ret90d%':>7}  {'upg60d%':>7}  "
      f"{'PredRet%':>8}  {'Risk%':>6}  {'Agent%':>7}  {'TE_Country':>10}")
print("─"*93)
for _, _r in segment_metrics_region.iterrows():
    print(f"{str(_r['region_bucket']):25s} {int(_r['n_users']):5}  "
          f"{_r['ret30d_actual']*100:6.1f}%  {_r['ret90d_actual']*100:6.1f}%  "
          f"{_r['upg60d_actual']*100:6.1f}%  {_r['predicted_retention']*100:7.1f}%  "
          f"{_r['predicted_risk']*100:5.1f}%  {_r['agent_usage_ratio']*100:6.1f}%  "
          f"{_r['avg_te_country']:.4f}")

# ═══════════════════════════════════════════════════════════════
# 8. SEGMENT-LEVEL METRICS: SIGNUP HOUR BUCKET
# ═══════════════════════════════════════════════════════════════
def _hour_to_bucket(h):
    if 0 <= h < 6:   return "Night (0-5)"
    elif 6 <= h < 12: return "Morning (6-11)"
    elif 12 <= h < 18: return "Afternoon (12-17)"
    else:              return "Evening (18-23)"

scoring_df["signup_hour_bucket"] = scoring_df["feat_signup_hour"].apply(_hour_to_bucket)

segment_metrics_signup_hour = (
    scoring_df.groupby("signup_hour_bucket")
    .agg(
        n_users             = ("user_id_canon", "count"),
        ret30d_actual       = ("ret30d",  "mean"),
        ret90d_actual       = ("ret90d",  "mean"),
        upg60d_actual       = ("upg60d",  "mean"),
        predicted_retention = ("predicted_retention_prob", "mean"),
        predicted_risk      = ("predicted_risk", "mean"),
        agent_usage_ratio   = ("agent_usage_ratio", "mean"),
    )
    .reset_index()
    .sort_values("n_users", ascending=False)
)

print("\n\n" + "═"*100)
print("📊 SEGMENT METRICS BY SIGNUP HOUR BUCKET")
print("═"*100)
print(f"{'Hour Bucket':22s} {'N':>5}  {'ret30d%':>7}  {'ret90d%':>7}  {'upg60d%':>7}  "
      f"{'PredRet%':>8}  {'Risk%':>6}  {'Agent%':>7}")
print("─"*80)
for _, _r in segment_metrics_signup_hour.iterrows():
    print(f"{_r['signup_hour_bucket']:22s} {int(_r['n_users']):5}  "
          f"{_r['ret30d_actual']*100:6.1f}%  {_r['ret90d_actual']*100:6.1f}%  "
          f"{_r['upg60d_actual']*100:6.1f}%  {_r['predicted_retention']*100:7.1f}%  "
          f"{_r['predicted_risk']*100:5.1f}%  {_r['agent_usage_ratio']*100:6.1f}%")

# ═══════════════════════════════════════════════════════════════
# 9. VISUALIZATION 1 — Intervention Priority Ranking
# ═══════════════════════════════════════════════════════════════
fig_priority_rank, _ax_pr = plt.subplots(figsize=(14, 7), facecolor=_BG)
_ax_pr.set_facecolor(_BG)
for _sp in ["top", "right"]: _ax_pr.spines[_sp].set_visible(False)
_ax_pr.spines["left"].set_color(_TXT2); _ax_pr.spines["bottom"].set_color(_TXT2)

_iv_labels    = [f"#{int(r['priority_rank'])} {r['Intervention']}" for _, r in uplift_priority_df.iterrows()]
_scores       = uplift_priority_df["priority_score"].values
_exp_rets     = uplift_priority_df["expected_add_retentions"].values
_bar_cols     = [_COLS[i % len(_COLS)] for i in range(len(_iv_labels))]

_xb = np.arange(len(_iv_labels))
_bars = _ax_pr.bar(_xb, _scores, color=_bar_cols, alpha=0.88, edgecolor="none", label="Priority Score")
_ax2_pr = _ax_pr.twinx()
_ax2_pr.set_facecolor(_BG)
_ax2_pr.plot(_xb, _exp_rets, "o--", color=_HL, linewidth=2.2, markersize=10,
             markerfacecolor=_BG, markeredgewidth=2.5, label="Expected Retentions")
for _xi, _rv in zip(_xb, _exp_rets):
    _ax2_pr.annotate(f"+{_rv:.0f}", (_xi, _rv + 0.3), color=_HL, fontsize=10.5,
                     ha="center", fontweight="bold")

_ax_pr.set_xticks(_xb)
_ax_pr.set_xticklabels(_iv_labels, rotation=12, ha="right", color=_TXT, fontsize=10)
_ax_pr.set_ylabel("Priority Score  (composite)", color=_TXT, fontsize=12)
_ax2_pr.set_ylabel("Expected Additional Retentions", color=_HL, fontsize=12)
_ax2_pr.tick_params(colors=_HL, labelsize=10)
_ax_pr.tick_params(colors=_TXT2, labelsize=10)
_ax_pr.set_title("Intervention Priority Ranking\n"
                 "Bars = composite priority score | Gold dots = expected additional retained users",
                 color=_TXT, fontsize=13, fontweight="bold", pad=15)

_p1 = mpatches.Patch(color=_COLS[0], label="Priority Score")
_p2 = plt.Line2D([0], [0], color=_HL, marker="o", linestyle="--", label="Expected Retentions")
_ax_pr.legend(handles=[_p1, _p2], facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# 10. VISUALIZATION 2 — Archetype Segment Metrics
# ═══════════════════════════════════════════════════════════════
fig_segment_archetype, _ax_sa = plt.subplots(figsize=(14, 7), facecolor=_BG)
_ax_sa.set_facecolor(_BG)
for _sp in ["top", "right"]: _ax_sa.spines[_sp].set_visible(False)
_ax_sa.spines["left"].set_color(_TXT2); _ax_sa.spines["bottom"].set_color(_TXT2)

_arch_labels = segment_metrics_archetype["archetype"].values
_xa = np.arange(len(_arch_labels))
_w  = 0.25

_bars1 = _ax_sa.bar(_xa - _w,  segment_metrics_archetype["ret30d_actual"]  * 100, _w,
                    label="Actual ret30d%", color=_COLS[0], alpha=0.88, edgecolor="none")
_bars2 = _ax_sa.bar(_xa,        segment_metrics_archetype["ret90d_actual"]  * 100, _w,
                    label="Actual ret90d%", color=_COLS[1], alpha=0.88, edgecolor="none")
_bars3 = _ax_sa.bar(_xa + _w,   segment_metrics_archetype["predicted_retention"] * 100, _w,
                    label="Predicted Retention%", color=_COLS[4], alpha=0.88, edgecolor="none")

for _bars in (_bars1, _bars2, _bars3):
    for _b in _bars:
        _v = _b.get_height()
        if _v > 1:
            _ax_sa.text(_b.get_x() + _b.get_width()/2, _v + 0.5, f"{_v:.1f}",
                       ha="center", color=_TXT, fontsize=7.5, fontweight="bold")

_ax_sa.set_xticks(_xa)
_ax_sa.set_xticklabels(_arch_labels, rotation=15, ha="right", color=_TXT, fontsize=10)
_ax_sa.set_ylabel("Rate (%)", color=_TXT, fontsize=12)
_ax_sa.set_title("Retention & Predicted Retention by Archetype\n"
                 "Actual retention vs model-calibrated predictions per segment",
                 color=_TXT, fontsize=13, fontweight="bold", pad=15)
_ax_sa.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
_ax_sa.tick_params(colors=_TXT2, labelsize=10)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# 11. VISUALIZATION 3 — Uplift heatmap by Archetype × Intervention
# ═══════════════════════════════════════════════════════════════
_archetypes_list = sorted(scoring_df["archetype"].dropna().unique())
_iv_names        = [iv["label"] for iv in INTERVENTIONS]
_heatmap_vals    = np.zeros((len(_archetypes_list), len(INTERVENTIONS)))

for _ai, _arch in enumerate(_archetypes_list):
    _sub = scoring_df[scoring_df["archetype"] == _arch]
    for _ii, _iv in enumerate(INTERVENTIONS):
        _heatmap_vals[_ai, _ii] = _sub["uplift_" + _iv["name"]].mean()

fig_uplift_heatmap, _ax_hm = plt.subplots(figsize=(14, 7), facecolor=_BG)
_ax_hm.set_facecolor(_BG)
_im = _ax_hm.imshow(_heatmap_vals, cmap="YlOrRd", aspect="auto", vmin=0)
_cbar = plt.colorbar(_im, ax=_ax_hm, fraction=0.03, pad=0.02)
_cbar.set_label("Mean Uplift (pp)", color=_TXT, fontsize=11)
_cbar.ax.tick_params(labelcolor=_TXT2); plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=_TXT)
_ax_hm.set_xticks(range(len(_iv_names)))
_ax_hm.set_xticklabels(_iv_names, rotation=18, ha="right", color=_TXT, fontsize=10)
_ax_hm.set_yticks(range(len(_archetypes_list)))
_ax_hm.set_yticklabels(_archetypes_list, color=_TXT, fontsize=10)
for _ai in range(len(_archetypes_list)):
    for _ii in range(len(INTERVENTIONS)):
        _v = _heatmap_vals[_ai, _ii]
        _ax_hm.text(_ii, _ai, f"{_v:.1f}", ha="center", va="center",
                    color="white" if _v > _heatmap_vals.max()*0.6 else _BG,
                    fontsize=9.5, fontweight="bold")
_ax_hm.set_title("Uplift Heatmap — Archetype × Intervention (mean pp)\n"
                 "Brighter = higher expected uplift for that segment",
                 color=_TXT, fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# 12. SAVE FINAL SCORED USER TABLE TO CSV
# ═══════════════════════════════════════════════════════════════
# Build clean export table
_export_cols = [
    "user_id_canon", "archetype", "cluster_id", "split",
    "region_bucket", "signup_hour_bucket",
    "feat_signup_hour", "feat_te_country",
    "feat_ratio_agent", "feat_n_sessions", "feat_active_days",
    "feat_onboarding_completed", "feat_agent_block_conversion",
    "ret30d", "ret90d", "upg60d",
    "predicted_retention_prob", "predicted_risk", "predicted_upgrade_prob",
    "agent_usage_ratio",
] + ["uplift_" + iv["name"] for iv in INTERVENTIONS]

scored_user_table = scoring_df[_export_cols].copy()
scored_user_table.to_csv("scored_user_table.csv", index=False)

print("\n\n" + "═"*75)
print("✅ UPLIFT INTERVENTION SCORING COMPLETE")
print("═"*75)
print(f"   Users scored:     {len(scored_user_table):,}")
print(f"   Interventions:    {len(INTERVENTIONS)}")
print(f"   Priority weights: segment×{_WEIGHTS['segment_size']} + uplift×{_WEIGHTS['total_uplift']} + "
      f"eng×{_WEIGHTS['eng_efficiency']} + speed×{_WEIGHTS['speed']}")
print(f"\n   📁 scored_user_table.csv  saved ({len(scored_user_table):,} rows)")
print(f"\n   📊 Variables exported:")
print(f"      • uplift_priority_df        — intervention rank table ({len(uplift_priority_df)} rows)")
print(f"      • scored_user_table         — per-user scores ({len(scored_user_table):,} rows)")
print(f"      • segment_metrics_archetype — by archetype ({len(segment_metrics_archetype)} rows)")
print(f"      • segment_metrics_region    — by region bucket ({len(segment_metrics_region)} rows)")
print(f"      • segment_metrics_signup_hour — by signup hour ({len(segment_metrics_signup_hour)} rows)")
print(f"      • fig_priority_rank         — intervention priority chart")
print(f"      • fig_segment_archetype     — archetype retention chart")
print(f"      • fig_uplift_heatmap        — uplift heatmap")
