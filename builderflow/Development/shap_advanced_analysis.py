
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import average_precision_score
import xgboost as xgb

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
_BG   = "#1D1D20"; _TXT  = "#fbfbff"; _TXT2 = "#909094"
_COLS = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
         "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
_HL = "#ffd400"; _SUCCESS = "#17b26a"; _WARN = "#f04438"

# ═══════════════════════════════════════════════════════════════
# 0. SETUP — feature columns & arrays from upstream GBT block
# ═══════════════════════════════════════════════════════════════
_LABEL_COLS = ["ret30d", "ret90d", "upg60d"]
_META_COLS  = ["user_id_canon", "split", "first_event_ts"]
_feat_cols  = [c for c in gbt_train_df_out.columns
               if c not in _LABEL_COLS + _META_COLS]

_X_train = X_gbt_train.astype(np.float32)
_X_test  = X_gbt_test.astype(np.float32)
_y_test  = y_gbt_test
_model   = gbt_best_model

print("═"*65)
print("🔍 SHAP ADVANCED ANALYSIS — XGBoost GBT (built-in SHAP)")
print("═"*65)
print(f"   Model:    XGBClassifier (Bayesian HPO, {_model.n_estimators} estimators)")
print(f"   Train:    {len(_X_train):,}  |  Test: {len(_X_test):,}")
print(f"   Features: {len(_feat_cols)}")
print(f"   Method:   XGBoost native pred_contribs (exact tree SHAP)")

# ═══════════════════════════════════════════════════════════════
# 1. COMPUTE SHAP VALUES via XGBoost's native pred_contribs
#    Returns (n_samples, n_features+1) — last col is bias/base
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("⚙️  Computing SHAP values (XGBoost native pred_contribs)...")
print("═"*65)

_dtest  = xgb.DMatrix(_X_test,  feature_names=_feat_cols)
_dtrain = xgb.DMatrix(_X_train, feature_names=_feat_cols)

_shap_raw  = _model.get_booster().predict(_dtest, pred_contribs=True)
# Shape: (n_test, n_feats + 1)  — last column is the bias
_shap_vals = _shap_raw[:, :-1].astype(np.float64)   # (n_test, n_feats)
_shap_bias = float(_shap_raw[0, -1])

print(f"   SHAP matrix: {_shap_vals.shape}  |  bias (base value): {_shap_bias:.4f}")

# ═══════════════════════════════════════════════════════════════
# 2. CHART 1: Global Feature Importance — SHAP bar chart (top 20)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📊 CHART 1: Global SHAP Feature Importance (bar chart, top 20)")
print("═"*65)

_mean_abs_shap = np.abs(_shap_vals).mean(axis=0)
_sort_idx  = np.argsort(_mean_abs_shap)[::-1]
_TOP_N     = 20
_top_idx   = _sort_idx[:_TOP_N]
_top_vals  = _mean_abs_shap[_top_idx]
_top_names = [_feat_cols[i] for i in _top_idx]
_top_short = [n.replace("feat_","").replace("_"," ")[:38] for n in _top_names]

_bar_colors = [_COLS[0] if i < 5 else (_COLS[4] if i < 10 else _TXT2)
               for i in range(_TOP_N)]

fig_shap_bar, _ax_bar = plt.subplots(figsize=(13, 10), facecolor=_BG)
_ax_bar.set_facecolor(_BG)
for _sp in ["top","right"]: _ax_bar.spines[_sp].set_visible(False)
_ax_bar.spines["left"].set_color(_TXT2); _ax_bar.spines["bottom"].set_color(_TXT2)

_bars = _ax_bar.barh(
    _top_short[::-1], _top_vals[::-1],
    color=_bar_colors[::-1], alpha=0.9, edgecolor="none"
)
for _b, _v in zip(_bars, _top_vals[::-1]):
    _ax_bar.text(_v + 0.0003, _b.get_y() + _b.get_height()/2,
                 f"{_v:.4f}", va="center", color=_TXT, fontsize=8)

_ax_bar.set_xlabel("Mean |SHAP Value|  (average impact on model output)", color=_TXT, fontsize=11)
_ax_bar.set_title(f"Global SHAP Feature Importance — Top {_TOP_N} Features\n"
                  "(XGBoost native TreeSHAP, exact values, test set)",
                  color=_TXT, fontsize=13, fontweight="bold", pad=15)
_ax_bar.tick_params(axis="y", labelcolor=_TXT, labelsize=9)
_ax_bar.tick_params(axis="x", labelcolor=_TXT2, labelsize=9)
_p1 = mpatches.Patch(color=_COLS[0], label="Top 5 features")
_p2 = mpatches.Patch(color=_COLS[4], label="Top 6–10")
_p3 = mpatches.Patch(color=_TXT2,   label="Top 11–20")
_ax_bar.legend(handles=[_p1,_p2,_p3], facecolor=_BG, edgecolor=_TXT2,
               labelcolor=_TXT, fontsize=9, loc="lower right")
plt.tight_layout(); plt.show()

print("   Top 15 global SHAP importances (mean |SHAP value|):")
for _i, (_f, _v) in enumerate(zip(_top_names[:15], _top_vals[:15])):
    print(f"      {_i+1:2d}. {_f:45s}  {_v:.5f}")

# ═══════════════════════════════════════════════════════════════
# 3. CHART 2: Beeswarm Plot (top 15 features)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🐝 CHART 2: SHAP Beeswarm Plot (top 15 features, test set)")
print("═"*65)

_N_BEE   = 15
_bee_idx = _sort_idx[:_N_BEE]
_bee_names = [_feat_cols[i].replace("feat_","").replace("_"," ")[:30] for i in _bee_idx]

fig_shap_beeswarm, _ax_bee = plt.subplots(figsize=(13, 9), facecolor=_BG)
_ax_bee.set_facecolor(_BG)
for _sp in ["top","right"]: _ax_bee.spines[_sp].set_visible(False)
_ax_bee.spines["left"].set_color(_TXT2); _ax_bee.spines["bottom"].set_color(_TXT2)

_rng = np.random.RandomState(42)
for _rank, _fi in enumerate(_bee_idx[::-1]):
    _sv_col = _shap_vals[:, _fi]
    _fv_col = _X_test[:, _fi].astype(float)
    _fmin, _fmax = _fv_col.min(), _fv_col.max()
    _norm = (_fv_col - _fmin) / (_fmax - _fmin + 1e-8)
    _colors_bee = plt.cm.RdYlBu_r(_norm)
    _jitter = _rng.uniform(-0.3, 0.3, len(_sv_col))
    _ax_bee.scatter(_sv_col, _rank + _jitter, c=_colors_bee, s=5, alpha=0.6, edgecolors="none")

_ax_bee.set_yticks(range(_N_BEE))
_ax_bee.set_yticklabels(list(reversed(_bee_names)), fontsize=9, color=_TXT)
_ax_bee.set_xlabel("SHAP Value (log-odds contribution to 30-day retention)", color=_TXT, fontsize=11)
_ax_bee.axvline(0, color=_TXT2, linewidth=0.9, linestyle="--", alpha=0.6)
_ax_bee.set_title(f"SHAP Beeswarm — Top {_N_BEE} Features\n"
                  "Color = feature value (red=high, blue=low)",
                  color=_TXT, fontsize=13, fontweight="bold", pad=15)
_ax_bee.tick_params(colors=_TXT2, labelsize=9)

_sm_bee = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0,1))
_sm_bee.set_array([])
_cbar_bee = plt.colorbar(_sm_bee, ax=_ax_bee, fraction=0.025, pad=0.02)
_cbar_bee.set_label("Feature value (normalized)", color=_TXT, fontsize=10)
_cbar_bee.ax.tick_params(labelcolor=_TXT2, labelsize=8)
_cbar_bee.set_ticks([0,0.5,1]); _cbar_bee.set_ticklabels(["Low","Mid","High"])
plt.setp(_cbar_bee.ax.yaxis.get_ticklabels(), color=_TXT)
plt.tight_layout(); plt.show()
print("   ✅ Beeswarm rendered")

# ═══════════════════════════════════════════════════════════════
# 4. CHART 3: Top Interaction Pairs (SHAP vector cross-correlation)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔗 CHART 3: Top SHAP Interaction Pairs (top 15 features)")
print("═"*65)

_TOP_INT   = 15
_int_idx   = _sort_idx[:_TOP_INT]
_int_names = [_feat_cols[i] for i in _int_idx]
_shap_sub  = _shap_vals[:, _int_idx]
_interact_mat = np.abs(np.corrcoef(_shap_sub.T))
np.fill_diagonal(_interact_mat, 0)
_int_short = [n.replace("feat_","").replace("_"," ")[:20] for n in _int_names]

fig_shap_interact, _ax_im = plt.subplots(figsize=(13, 11), facecolor=_BG)
_ax_im.set_facecolor(_BG)
_im = _ax_im.imshow(_interact_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
_ax_im.set_xticks(range(_TOP_INT)); _ax_im.set_yticks(range(_TOP_INT))
_ax_im.set_xticklabels(_int_short, rotation=45, ha="right", color=_TXT, fontsize=8)
_ax_im.set_yticklabels(_int_short, color=_TXT, fontsize=8)
_cbar_im = plt.colorbar(_im, ax=_ax_im, fraction=0.046, pad=0.04)
_cbar_im.set_label("|SHAP correlation|  (0=independent, 1=highly interactive)", color=_TXT, fontsize=10)
_cbar_im.ax.tick_params(labelcolor=_TXT2); plt.setp(_cbar_im.ax.yaxis.get_ticklabels(), color=_TXT)

_thresh_annot = _interact_mat.mean() + _interact_mat.std()
for _ii in range(_TOP_INT):
    for _jj in range(_TOP_INT):
        if _ii != _jj and _interact_mat[_ii,_jj] >= _thresh_annot:
            _ax_im.text(_jj, _ii, f"{_interact_mat[_ii,_jj]:.2f}",
                        ha="center", va="center", color="white", fontsize=6, fontweight="bold")

_ax_im.set_title("SHAP Interaction Heatmap — Top 15 Features\n"
                 "(|Pearson correlation of SHAP value vectors| across test set users)",
                 color=_TXT, fontsize=13, fontweight="bold", pad=15)
plt.tight_layout(); plt.show()

_pairs_list = []
for _ii in range(_TOP_INT):
    for _jj in range(_ii+1, _TOP_INT):
        _pairs_list.append({"feature_1": _int_names[_ii], "feature_2": _int_names[_jj],
                            "shap_interaction": round(float(_interact_mat[_ii,_jj]), 4)})
shap_interaction_df = pd.DataFrame(_pairs_list).sort_values("shap_interaction", ascending=False).reset_index(drop=True)

print(f"\n   Top 10 interacting feature pairs (SHAP correlation):")
for _, _r in shap_interaction_df.head(10).iterrows():
    print(f"      {_r['feature_1']:38s} × {_r['feature_2']:38s} = {_r['shap_interaction']:.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. CHART 4: Per-User Force Plots — sample of 20 users
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("👤 CHART 4: Per-User SHAP Force Plots (20 users sampled)")
print("═"*65)

_proba_test  = _model.predict_proba(_X_test)[:,1]
_p60         = np.percentile(_proba_test, 60)
_p40         = np.percentile(_proba_test, 40)
_pos_idx_all = np.where(_proba_test >= _p60)[0]
_neg_idx_all = np.where(_proba_test <  _p40)[0]

_rng2 = np.random.RandomState(7)
_n_each = 10
_s_pos  = _rng2.choice(_pos_idx_all, size=min(_n_each, len(_pos_idx_all)), replace=False)
_s_neg  = _rng2.choice(_neg_idx_all, size=min(_n_each, len(_neg_idx_all)), replace=False)
_sample_20 = np.concatenate([_s_pos, _s_neg])[:20]

_N_FORCE_FEATS = 7
fig_shap_force, _axs_f = plt.subplots(4, 5, figsize=(22, 14), facecolor=_BG)
_axs_f = _axs_f.flatten()

for _plot_i, _ui in enumerate(_sample_20):
    _ax = _axs_f[_plot_i]
    _ax.set_facecolor(_BG)
    for _sp in ["top","right"]: _ax.spines[_sp].set_visible(False)
    _ax.spines["left"].set_color(_TXT2); _ax.spines["bottom"].set_color(_TXT2)

    _user_shap = _shap_vals[_ui]
    _user_pred = float(_proba_test[_ui])
    _user_y    = int(_y_test[_ui])

    _abs_idx = np.argsort(np.abs(_user_shap))[::-1][:_N_FORCE_FEATS]
    _sv_top  = _user_shap[_abs_idx]
    _nm_top  = [_feat_cols[j].replace("feat_","").replace("_"," ")[:16] for j in _abs_idx]
    _fv_top  = [f"{float(_X_test[_ui,j]):.2f}" for j in _abs_idx]
    _labels  = [f"{n}\n={v}" for n,v in zip(_nm_top, _fv_top)]

    _c_bars = [_SUCCESS if v > 0 else _WARN for v in _sv_top]
    _ax.barh(range(len(_sv_top)), _sv_top[::-1], color=_c_bars[::-1], alpha=0.9, edgecolor="none")
    _ax.set_yticks(range(len(_sv_top)))
    _ax.set_yticklabels(_labels[::-1], fontsize=5.5, color=_TXT)
    _ax.axvline(0, color=_TXT2, linewidth=0.7, linestyle="--")

    _label = "✅ Kept" if _user_y == 1 else "❌ Churn"
    _pc    = _SUCCESS if _user_pred >= 0.5 else _WARN
    _ax.set_title(f"User {_plot_i+1}  P={_user_pred:.3f}\n{_label}",
                  color=_pc, fontsize=7.5, fontweight="bold", pad=4)
    _ax.tick_params(axis="x", labelsize=5.5, colors=_TXT2)

plt.suptitle("Per-User SHAP Force Plots — 20 Sampled Users\n"
             "🟢 Green = drives toward retention  🔴 Red = drives toward churn  |  Feature value shown",
             color=_TXT, fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(); plt.show()
print(f"   ✅ Force plots rendered for {len(_sample_20)} users")

# ═══════════════════════════════════════════════════════════════
# 6. BOOTSTRAP SHAP VARIANCE — 20 resamples (test set)
#    Bootstrap: resample test rows, recompute mean |SHAP| each time
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔁 BOOTSTRAP SHAP VARIANCE — 20 resamples (test set)")
print("═"*65)
print("   Resampling test set 20× — measuring importance stability...")

_N_BOOT = 20
_n_feats_stab = len(_feat_cols)
_boot_means = np.zeros((_N_BOOT, _n_feats_stab))

for _b in range(_N_BOOT):
    _rng_b = np.random.RandomState(100 + _b)
    _boot_idx = _rng_b.choice(len(_X_test), size=len(_X_test), replace=True)
    _boot_shap = _shap_vals[_boot_idx]
    _boot_means[_b] = np.abs(_boot_shap).mean(axis=0)
    if (_b + 1) % 5 == 0:
        print(f"   Resample {_b+1:2d}/20 done")

_boot_mean_overall = _boot_means.mean(axis=0)
_boot_std_overall  = _boot_means.std(axis=0)
_boot_cv_overall   = _boot_std_overall / (np.abs(_boot_mean_overall) + 1e-8)

shap_stability_df = pd.DataFrame({
    "feature":   _feat_cols,
    "mean_shap": _boot_mean_overall.round(6),
    "std_shap":  _boot_std_overall.round(6),
    "cv_shap":   _boot_cv_overall.round(4),
}).sort_values("mean_shap", ascending=False).reset_index(drop=True)

print(f"\n   Bootstrap SHAP Stability — Top 20 features (20 resamples, test set):")
print(f"\n{'Feature':45s}  {'Mean|SHAP|':>11s}  {'Std':>9s}  {'CV':>8s}  {'Stability':>12s}")
print("─"*92)
for _, _r in shap_stability_df.head(20).iterrows():
    _sl = "Very Stable" if _r["cv_shap"] < 0.10 else ("Stable" if _r["cv_shap"] < 0.25 else "Variable")
    print(f"{_r['feature']:45s}  {_r['mean_shap']:>11.5f}  {_r['std_shap']:>9.5f}  {_r['cv_shap']:>8.4f}  {_sl:>12s}")

# Bootstrap stability chart — top 15 with error bars
_stab_top15 = shap_stability_df.head(15)
fig_shap_bootstrap, _ax_boot = plt.subplots(figsize=(14, 7), facecolor=_BG)
_ax_boot.set_facecolor(_BG)
for _sp in ["top","right"]: _ax_boot.spines[_sp].set_visible(False)
_ax_boot.spines["left"].set_color(_TXT2); _ax_boot.spines["bottom"].set_color(_TXT2)

_xb = np.arange(len(_stab_top15))
_ax_boot.bar(_xb, _stab_top15["mean_shap"].values, color=_COLS[0], alpha=0.85, edgecolor="none", label="Mean |SHAP|")
_ax_boot.errorbar(_xb, _stab_top15["mean_shap"].values, yerr=_stab_top15["std_shap"].values,
                  fmt="none", color=_HL, elinewidth=2, capsize=5, capthick=2, label="±1 Std (20 bootstraps)")
_ax_boot.set_xticks(_xb)
_ax_boot.set_xticklabels([n.replace("feat_","").replace("_"," ")[:22] for n in _stab_top15["feature"]],
                         rotation=35, ha="right", color=_TXT, fontsize=8)
_ax_boot.set_ylabel("Mean |SHAP Value|", color=_TXT, fontsize=11)
_ax_boot.set_title("SHAP Stability — Bootstrap Variance (20 resamples, test set)\n"
                   "Short error bars = stable importance; long bars = high sampling variance",
                   color=_TXT, fontsize=13, fontweight="bold", pad=15)
_ax_boot.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
_ax_boot.tick_params(colors=_TXT2)
plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════
# 7. ABLATION STUDY — Drop feature groups one at a time
#    Groups: intensity, advanced_usage, collaboration, metadata
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔬 ABLATION STUDY — Drop feature groups one at a time")
print("═"*65)

_GROUPS = {
    "intensity": [
        "feat_event_count","feat_active_days","feat_n_sessions",
        "feat_rampup_slope","feat_max_gap_days","feat_events_per_day",
        "feat_mean_events_per_session","feat_median_events_per_session",
        "feat_max_events_per_session","feat_mean_session_duration_min",
        "feat_max_session_duration_min","feat_mean_distinct_events_per_session",
        "feat_day_gap_variance",
    ],
    "advanced_usage": [
        "feat_ratio_agent","feat_ratio_block_ops","feat_ratio_canvas",
        "feat_ratio_credits","feat_ratio_deploy","feat_ratio_files",
        "feat_early_deploy_count","feat_early_schedule_count",
        "feat_session_entropy","feat_tod_entropy",
        "feat_execution_to_agent_ratio","feat_agent_block_conversion",
        "feat_distinct_events","feat_distinct_categories",
        "feat_ttf_run_block","feat_ttf_canvas_create","feat_ttf_agent_use",
        "feat_ttf_file_upload","feat_ttf_credits_used","feat_ttf_edge_create",
        "feat_ttf_block_create","feat_credit_amount_sum",
    ],
    "collaboration": [
        "feat_ratio_collab","feat_collab_actions","feat_distinct_canvases",
    ],
    "metadata": [
        "feat_signup_dow","feat_signup_hour",
        "feat_onboarding_completed","feat_onboarding_skipped","feat_tour_finished",
        "feat_te_country","feat_te_device","feat_te_os","feat_te_browser",
        "feat_ratio_onboarding",
    ],
}
for _g in _GROUPS:
    _GROUPS[_g] = [c for c in _GROUPS[_g] if c in _feat_cols]

def _compute_lift10_abl(y_true, y_proba):
    _n_top = max(1, int(len(y_true) * 0.10))
    _top   = np.argsort(y_proba)[::-1][:_n_top]
    _base  = float(y_true.mean())
    return float(y_true[_top].mean()) / _base if _base > 0 else 1.0

def _retrain_xgb_abl(X_tr, y_tr, X_te, y_te, best_params_dict, feat_names):
    """Retrain XGBoost with same best params on reduced feature set."""
    _spw = (len(y_tr) - int(y_tr.sum())) / max(int(y_tr.sum()), 1)
    _m = xgb.XGBClassifier(**best_params_dict, scale_pos_weight=_spw,
                            random_state=42, eval_metric="aucpr",
                            verbosity=0, use_label_encoder=False)
    _m.fit(X_tr, y_tr, verbose=False)
    _p = _m.predict_proba(X_te)[:,1]
    _prauc  = average_precision_score(y_te, _p) if y_te.sum() > 0 else 0.0
    _lift10 = _compute_lift10_abl(y_te, _p)
    return _prauc, _lift10

# Baseline
_base_proba_abl  = _model.predict_proba(_X_test)[:,1]
_base_prauc_abl  = average_precision_score(_y_test, _base_proba_abl) if _y_test.sum() > 0 else 0.0
_base_lift10_abl = _compute_lift10_abl(_y_test, _base_proba_abl)

print(f"\n   BASELINE (all {len(_feat_cols)} features): PR-AUC={_base_prauc_abl:.4f}  Lift@10%={_base_lift10_abl:.3f}×")
for _g, _gc in _GROUPS.items():
    print(f"   Group '{_g}': {len(_gc)} features")

_abl_rows = [{"group_dropped":"none (baseline)","pr_auc":_base_prauc_abl,
              "lift_10":_base_lift10_abl,"n_features":len(_feat_cols),
              "pr_auc_drop":0.0,"lift_drop":0.0}]

for _gname, _gcols in _GROUPS.items():
    _keep_cols = [c for c in _feat_cols if c not in _gcols]
    _keep_idx  = [_feat_cols.index(c) for c in _keep_cols]
    print(f"\n   Dropping '{_gname}' ({len(_gcols)} feats → {len(_keep_cols)} remain)...")
    _pr, _lift = _retrain_xgb_abl(
        _X_train[:,_keep_idx], y_gbt_train,
        _X_test[:,_keep_idx],  _y_test,
        dict(gbt_best_params), _keep_cols
    )
    _pr_d  = _base_prauc_abl  - _pr
    _lft_d = _base_lift10_abl - _lift
    print(f"      PR-AUC={_pr:.4f} (Δ={_pr_d:+.4f})  Lift@10%={_lift:.3f}× (Δ={_lft_d:+.3f})")
    _abl_rows.append({"group_dropped":_gname,"pr_auc":_pr,"lift_10":_lift,
                      "n_features":len(_keep_cols),"pr_auc_drop":_pr_d,"lift_drop":_lft_d})

shap_ablation_df = pd.DataFrame(_abl_rows)

print("\n" + "═"*65)
print("📋 ABLATION IMPACT TABLE")
print("═"*65)
print(f"\n{'Group Dropped':22s}  {'PR-AUC':>8s}  {'ΔPRAUC':>8s}  {'Lift@10%':>9s}  {'ΔLift':>8s}  {'N Feats':>8s}")
print("─"*75)
for _, _r in shap_ablation_df.iterrows():
    _mk = "  ← baseline" if _r["group_dropped"] == "none (baseline)" else ""
    print(f"{_r['group_dropped']:22s}  {_r['pr_auc']:>8.4f}  {_r['pr_auc_drop']:>+8.4f}  "
          f"{_r['lift_10']:>9.3f}  {_r['lift_drop']:>+8.3f}  {_r['n_features']:>8.0f}{_mk}")

# Ablation charts
_non_base_abl = shap_ablation_df[shap_ablation_df["group_dropped"] != "none (baseline)"]
fig_shap_ablation, (_ax_a1, _ax_a2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=_BG)
for _ax_a in (_ax_a1, _ax_a2):
    _ax_a.set_facecolor(_BG)
    for _sp in ["top","right"]: _ax_a.spines[_sp].set_visible(False)
    _ax_a.spines["left"].set_color(_TXT2); _ax_a.spines["bottom"].set_color(_TXT2)

_xg = np.arange(len(_non_base_abl))
_c_pr  = [_WARN if v > 0 else _SUCCESS for v in _non_base_abl["pr_auc_drop"].values]
_c_lft = [_WARN if v > 0 else _SUCCESS for v in _non_base_abl["lift_drop"].values]

_ax_a1.bar(_xg, _non_base_abl["pr_auc_drop"].values, color=_c_pr, alpha=0.9, edgecolor="none")
_ax_a1.axhline(0, color=_TXT2, linewidth=0.8, linestyle="--")
for _bi, _bv in enumerate(_non_base_abl["pr_auc_drop"].values):
    _off = max(abs(_bv)*0.1, 0.001) * (1 if _bv >= 0 else -1)
    _ax_a1.text(_bi, _bv + _off, f"{_bv:+.4f}", ha="center", color=_TXT, fontsize=12, fontweight="bold")
_ax_a1.set_xticks(_xg)
_ax_a1.set_xticklabels(_non_base_abl["group_dropped"].values, color=_TXT, fontsize=11, rotation=10)
_ax_a1.set_ylabel("PR-AUC Degradation  (↑ = hurts more)", color=_TXT, fontsize=11)
_ax_a1.set_title("Ablation — PR-AUC Impact\n🔴 Red = dropping hurts performance", color=_TXT, fontsize=12, fontweight="bold")
_ax_a1.tick_params(colors=_TXT2)

_ax_a2.bar(_xg, _non_base_abl["lift_drop"].values, color=_c_lft, alpha=0.9, edgecolor="none")
_ax_a2.axhline(0, color=_TXT2, linewidth=0.8, linestyle="--")
for _bi, _bv in enumerate(_non_base_abl["lift_drop"].values):
    _off = max(abs(_bv)*0.1, 0.005) * (1 if _bv >= 0 else -1)
    _ax_a2.text(_bi, _bv + _off, f"{_bv:+.3f}×", ha="center", color=_TXT, fontsize=12, fontweight="bold")
_ax_a2.set_xticks(_xg)
_ax_a2.set_xticklabels(_non_base_abl["group_dropped"].values, color=_TXT, fontsize=11, rotation=10)
_ax_a2.set_ylabel("Lift@10% Degradation  (↑ = hurts more)", color=_TXT, fontsize=11)
_ax_a2.set_title("Ablation — Lift@10% Impact\n🔴 Red = dropping hurts performance", color=_TXT, fontsize=12, fontweight="bold")
_ax_a2.tick_params(colors=_TXT2)

plt.suptitle("Feature Group Ablation Study — Retention Model (XGBoost GBT)\n"
             "Shows which feature groups matter most for predictive power",
             color=_TXT, fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════
# 8. EXPORT
# ═══════════════════════════════════════════════════════════════
shap_global_importance   = pd.DataFrame({"feature":_top_names,"mean_abs_shap":_top_vals})
shap_bootstrap_stability = shap_stability_df
shap_ablation_results    = shap_ablation_df

print("\n" + "═"*65)
print("✅ SHAP ADVANCED ANALYSIS COMPLETE")
print("═"*65)
print("   ✅ fig_shap_bar          — Global SHAP importance (top 20)")
print("   ✅ fig_shap_beeswarm     — Beeswarm plot (top 15 features)")
print("   ✅ fig_shap_interact     — Interaction heatmap (top 15)")
print("   ✅ fig_shap_force        — Per-user force plots (20 users)")
print("   ✅ fig_shap_bootstrap    — Bootstrap stability (20 resamples)")
print("   ✅ shap_stability_df     — Variance table (mean, std, CV)")
print("   ✅ fig_shap_ablation     — Ablation charts (PR-AUC & Lift)")
print("   ✅ shap_ablation_df      — Ablation impact table (4 groups)")
