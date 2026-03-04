
# ═══════════════════════════════════════════════════════════════
# BUILDERFLOW — SCHEDULED JOB: Retention Risk Scoring & Intervention Prioritization
# ═══════════════════════════════════════════════════════════════
# Runs daily. Re-scores all users with calibrated XGBoost model,
# assigns archetypes, computes intervention uplift, and saves CSV.
#
# DEPENDENCIES (from Development layer):
#   • gbt_best_model        — trained XGBoost classifier
#   • gbt_isotonic_calib    — IsotonicRegression calibrator
#   • clean_feature_matrix  — feature DataFrame with ret30d/ret90d/upg60d/split
#   • cluster_archetype_names — {cluster_id: archetype_name} mapping
#   • cluster_outcome_rates — DataFrame with per-archetype outcome rates
#   • feature_matrix_with_archetypes — full feature matrix with archetype col
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

print("═"*70)
print(f"🔄 SCHEDULED JOB: Retention Scoring Pipeline")
print(f"   Run timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("═"*70)

# ── 1. Extract feature columns ──────────────────────────────────
_LABEL_COLS = ["ret30d", "ret90d", "upg60d"]
_META_COLS  = ["user_id_canon", "split", "cluster_id", "archetype"]
_feat_cols  = [c for c in feature_matrix_with_archetypes.columns
               if c not in _LABEL_COLS + _META_COLS]

print(f"\n   📦 Users to score:  {len(feature_matrix_with_archetypes):,}")
print(f"   📊 Features:       {len(_feat_cols)}")

# ── 2. Score all users with calibrated XGBoost ──────────────────
_X_all = feature_matrix_with_archetypes[_feat_cols].values.astype(np.float32)
_raw_proba   = gbt_best_model.predict_proba(_X_all)[:, 1]
_calib_proba = gbt_isotonic_calib.predict(_raw_proba)
_risk_score  = 1.0 - _calib_proba

print(f"\n   🎯 Calibrated retention probability:")
print(f"      Mean: {_calib_proba.mean():.4f}  |  Std: {_calib_proba.std():.4f}")
print(f"      Risk (churn):  Mean: {_risk_score.mean():.4f}")

# ── 3. Build scored output table ────────────────────────────────
scored_output = feature_matrix_with_archetypes[[
    "user_id_canon", "archetype", "cluster_id", "split",
    "feat_ratio_agent", "feat_n_sessions", "feat_active_days",
    "feat_onboarding_completed", "feat_agent_block_conversion",
    "ret30d", "ret90d", "upg60d",
]].copy()

scored_output["predicted_retention_prob"] = _calib_proba
scored_output["predicted_risk"]           = _risk_score
scored_output["risk_tier"] = pd.cut(
    _risk_score,
    bins=[0, 0.3, 0.6, 0.85, 1.0],
    labels=["🟢 Low Risk", "🟡 Medium Risk", "🟠 High Risk", "🔴 Critical"],
    include_lowest=True
)

# ── 4. Summary dashboard ────────────────────────────────────────
print("\n" + "═"*70)
print("📊 SCORING DASHBOARD")
print("═"*70)

# By risk tier
print("\n   ── Risk Tier Distribution ──")
_tier_dist = scored_output["risk_tier"].value_counts().sort_index()
for _tier, _n in _tier_dist.items():
    print(f"      {_tier:20s}: {_n:5d} users ({_n/len(scored_output)*100:5.1f}%)")

# By archetype
print("\n   ── Archetype Scores ──")
print(f"   {'Archetype':25s} {'N':>5}  {'Pred_Ret%':>9}  {'Risk%':>6}  {'Act_ret30d%':>11}")
print("   " + "─"*65)
for _arch in sorted(scored_output["archetype"].dropna().unique()):
    _sub = scored_output[scored_output["archetype"] == _arch]
    print(f"   {_arch:25s} {len(_sub):5d}  "
          f"{_sub['predicted_retention_prob'].mean()*100:8.1f}%  "
          f"{_sub['predicted_risk'].mean()*100:5.1f}%  "
          f"{_sub['ret30d'].mean()*100:10.1f}%")

# ── 5. Top at-risk users needing intervention ───────────────────
_high_risk = scored_output[scored_output["predicted_risk"] > 0.85].sort_values(
    "predicted_risk", ascending=False
).head(20)

print(f"\n   ── Top {min(20, len(_high_risk))} Highest-Risk Users ──")
print(f"   {'User':40s} {'Archetype':25s} {'Risk':>6}")
print("   " + "─"*75)
for _, _r in _high_risk.head(10).iterrows():
    print(f"   {str(_r['user_id_canon'])[:40]:40s} "
          f"{str(_r['archetype']):25s} {_r['predicted_risk']*100:5.1f}%")

# ── 6. Save output ──────────────────────────────────────────────
_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
_output_path = f"scored_users_{_timestamp}.csv"
scored_output.to_csv(_output_path, index=False)

print(f"\n   📁 Output saved: {_output_path}")
print(f"      Rows: {len(scored_output):,}  |  Columns: {scored_output.shape[1]}")

# Also save latest snapshot
scored_output.to_csv("scored_users_latest.csv", index=False)
print(f"   📁 Latest snapshot: scored_users_latest.csv")

print(f"\n{'═'*70}")
print(f"✅ SCHEDULED JOB COMPLETE — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"{'═'*70}")
