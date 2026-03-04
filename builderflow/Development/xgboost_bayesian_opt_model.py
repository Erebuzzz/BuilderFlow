
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve, fbeta_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
_BG   = "#1D1D20"; _TXT  = "#fbfbff"; _TXT2 = "#909094"
_COLS = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
         "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
_HL   = "#ffd400"; _SUCCESS = "#17b26a"; _WARN = "#f04438"

# ═══════════════════════════════════════════════════════════════
# 0. TEMPORAL SPLIT: train Sep–Oct 2025, test Nov 1–8 2025
# ═══════════════════════════════════════════════════════════════
TARGET_LABEL = "ret30d"

_LABEL_COLS = ["ret30d", "ret90d", "upg60d"]
_META_COLS  = ["user_id_canon", "split"]
_feat_cols  = [c for c in clean_feature_matrix.columns
               if c not in _LABEL_COLS + _META_COLS]

_mdf = clean_feature_matrix.merge(
    cohort_users[["user_id_canon", "first_event_ts"]], on="user_id_canon", how="left"
)

_TRAIN_START = pd.Timestamp("2025-09-01", tz="UTC")
_TRAIN_END   = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
_TEST_START  = pd.Timestamp("2025-11-01", tz="UTC")
_TEST_END    = pd.Timestamp("2025-11-08 23:59:59", tz="UTC")

_train_mask = (_mdf["first_event_ts"] >= _TRAIN_START) & (_mdf["first_event_ts"] <= _TRAIN_END)
_test_mask  = (_mdf["first_event_ts"] >= _TEST_START)  & (_mdf["first_event_ts"] <= _TEST_END)
_train_df   = _mdf[_train_mask].copy().reset_index(drop=True)
_test_df    = _mdf[_test_mask].copy().reset_index(drop=True)

print("═"*65)
print("📅 TEMPORAL SPLIT  (Sep–Oct train | Nov 1–8 test)")
print("═"*65)
print(f"   Train: {len(_train_df):,} users  ({_train_df[TARGET_LABEL].mean()*100:.1f}% positive)")
print(f"   Test:  {len(_test_df):,}  users  ({_test_df[TARGET_LABEL].mean()*100:.1f}% positive)")
print(f"   Features: {len(_feat_cols)}")

if len(_test_df) < 20:
    print("\n   ⚠️  Nov 1-8 test set too small — falling back to temporal holdout (top 25%)")
    _sorted_ts  = _mdf.sort_values("first_event_ts").reset_index(drop=True)
    _split_q    = _sorted_ts["first_event_ts"].quantile(0.75)
    _train_mask = _mdf["first_event_ts"] <= _split_q
    _test_mask  = _mdf["first_event_ts"] > _split_q
    _train_df   = _mdf[_train_mask].copy().reset_index(drop=True)
    _test_df    = _mdf[_test_mask].copy().reset_index(drop=True)
    print(f"   Fallback train: {len(_train_df):,} | test: {len(_test_df):,}")

X_gbt_train = _train_df[_feat_cols].values.astype(np.float32)
y_gbt_train = _train_df[TARGET_LABEL].values.astype(int)
X_gbt_test  = _test_df[_feat_cols].values.astype(np.float32)
y_gbt_test  = _test_df[TARGET_LABEL].values.astype(int)

_n_pos = int(y_gbt_train.sum())
_n_neg = len(y_gbt_train) - _n_pos
_spw   = _n_neg / max(_n_pos, 1)
print(f"\n   scale_pos_weight = {_spw:.2f}  ({_n_pos} pos / {_n_neg} neg in train)")

# ═══════════════════════════════════════════════════════════════
# 1. 3-WINDOW ROLLING CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔁 3-WINDOW ROLLING CV  (temporal, expanding window)")
print("═"*65)

_sorted_all = _mdf.sort_values("first_event_ts").reset_index(drop=True)
_n_total    = len(_sorted_all)
_block_size = _n_total // 4
_blocks     = [_sorted_all.iloc[i*_block_size:(i+1)*_block_size] for i in range(3)]
_blocks.append(_sorted_all.iloc[3*_block_size:])

_rolling_cv_results = []
for _w_idx in range(3):
    _cv_train = pd.concat(_blocks[:_w_idx+1], ignore_index=True)
    _cv_val   = _blocks[_w_idx+1].copy()
    _Xtr = _cv_train[_feat_cols].values.astype(np.float32)
    _ytr = _cv_train[TARGET_LABEL].values.astype(int)
    _Xvl = _cv_val[_feat_cols].values.astype(np.float32)
    _yvl = _cv_val[TARGET_LABEL].values.astype(int)

    if _ytr.sum() < 3 or _yvl.sum() < 2:
        print(f"   Window {_w_idx+1}: skipped (positives: train={_ytr.sum()}, val={_yvl.sum()})")
        continue

    _spw_cv = (len(_ytr) - int(_ytr.sum())) / max(int(_ytr.sum()), 1)
    _cv_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
        scale_pos_weight=_spw_cv, random_state=42, eval_metric="aucpr",
        verbosity=0, use_label_encoder=False, early_stopping_rounds=20,
    )
    _cv_xgb.fit(_Xtr, _ytr, eval_set=[(_Xvl, _yvl)], verbose=False)
    _cv_proba  = _cv_xgb.predict_proba(_Xvl)[:, 1]
    _cv_prauc  = average_precision_score(_yvl, _cv_proba)
    _cv_rocauc = roc_auc_score(_yvl, _cv_proba)
    _rolling_cv_results.append({
        "window":  _w_idx+1, "n_train": len(_Xtr), "n_val": len(_Xvl),
        "pr_auc":  _cv_prauc, "roc_auc": _cv_rocauc,
    })
    print(f"   Window {_w_idx+1}: train={len(_Xtr):,}  val={len(_Xvl):,}  "
          f"PR-AUC={_cv_prauc:.4f}  ROC-AUC={_cv_rocauc:.4f}")

rolling_cv_df = pd.DataFrame(_rolling_cv_results)
if len(rolling_cv_df) > 0:
    print(f"\n   Mean PR-AUC  = {rolling_cv_df['pr_auc'].mean():.4f} ± {rolling_cv_df['pr_auc'].std():.4f}")
    print(f"   Mean ROC-AUC = {rolling_cv_df['roc_auc'].mean():.4f} ± {rolling_cv_df['roc_auc'].std():.4f}")

# ═══════════════════════════════════════════════════════════════
# 2. BAYESIAN HPO
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔍 BAYESIAN HPO  (30 trials: 15 explore + 15 exploit)")
print("═"*65)
print("   Model: XGBClassifier  |  Param space (ticket spec):")

_PARAM_SPACE = {
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.02, 0.05, 0.1],
    "n_estimators":     [200, 400, 600],
    "subsample":        [0.7, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda":       [1.0, 5.0, 10.0],
}
for _k, _v in _PARAM_SPACE.items():
    print(f"      {_k:25s} = {_v}")

_cv_hpo            = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
_rng               = np.random.RandomState(42)
_best_params_bayes = None
_best_score_bayes  = 0.0
_hpo_history       = []

def _score_xgb_params(params):
    _scores = []
    for _tr_idx, _vl_idx in _cv_hpo.split(X_gbt_train, y_gbt_train):
        _Xtr_cv = X_gbt_train[_tr_idx]; _ytr_cv = y_gbt_train[_tr_idx]
        _Xvl_cv = X_gbt_train[_vl_idx]; _yvl_cv = y_gbt_train[_vl_idx]
        _spw_inner = (len(_ytr_cv) - int(_ytr_cv.sum())) / max(int(_ytr_cv.sum()), 1)
        _m = xgb.XGBClassifier(**params, scale_pos_weight=_spw_inner,
                                random_state=42, eval_metric="aucpr",
                                verbosity=0, use_label_encoder=False)
        _m.fit(_Xtr_cv, _ytr_cv, verbose=False)
        _scores.append(average_precision_score(_yvl_cv, _m.predict_proba(_Xvl_cv)[:, 1]))
    return float(np.mean(_scores))

_N_EXPLORE = 15
print(f"\n   Phase 1: {_N_EXPLORE} exploration trials...")
for _i in range(_N_EXPLORE):
    _p = {k: _rng.choice(v).item() for k, v in _PARAM_SPACE.items()}
    _sc = _score_xgb_params(_p)
    _hpo_history.append(_sc)
    if _sc > _best_score_bayes:
        _best_score_bayes = _sc
        _best_params_bayes = dict(_p)
    if (_i + 1) % 5 == 0:
        print(f"      Trial {_i+1:2d}: best = {_best_score_bayes:.4f}")

_N_EXPLOIT   = 15
_best_so_far = dict(_best_params_bayes)
print(f"\n   Phase 2: {_N_EXPLOIT} exploitation trials...")
for _i in range(_N_EXPLOIT):
    _p = dict(_best_so_far)
    _keys_to_perturb = _rng.choice(list(_PARAM_SPACE.keys()), size=_rng.choice([1, 2]), replace=False)
    for _k in _keys_to_perturb:
        _p[_k] = _rng.choice(_PARAM_SPACE[_k]).item()
    _sc = _score_xgb_params(_p)
    _hpo_history.append(_sc)
    if _sc > _best_score_bayes:
        _best_score_bayes = _sc
        _best_params_bayes = dict(_p)
        _best_so_far = dict(_p)
    if (_i + 1) % 5 == 0:
        print(f"      Trial {_N_EXPLORE+_i+1:2d}: best = {_best_score_bayes:.4f}")

print(f"\n   ✅ Best CV PR-AUC = {_best_score_bayes:.4f}  (30 trials)")
print(f"\n   🏆 Best Hyperparameters:")
for _k, _v in _best_params_bayes.items():
    print(f"      {_k:25s} = {_v}")

# ═══════════════════════════════════════════════════════════════
# 3. TRAIN FINAL XGBoost (early stopping on 15% holdout)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🌲 FINAL XGBoost  (scale_pos_weight + early stopping)")
print("═"*65)

_n_val_es    = max(1, int(0.15 * len(X_gbt_train)))
_val_es_mask = np.zeros(len(X_gbt_train), dtype=bool)
_val_es_mask[-_n_val_es:] = True
_X_es_tr = X_gbt_train[~_val_es_mask]; _y_es_tr = y_gbt_train[~_val_es_mask]
_X_es_vl = X_gbt_train[_val_es_mask];  _y_es_vl = y_gbt_train[_val_es_mask]

gbt_model_final = xgb.XGBClassifier(
    **_best_params_bayes, scale_pos_weight=_spw,
    random_state=42, eval_metric="aucpr",
    verbosity=0, use_label_encoder=False, early_stopping_rounds=30,
)
gbt_model_final.fit(_X_es_tr, _y_es_tr, eval_set=[(_X_es_vl, _y_es_vl)], verbose=False)
_actual_iters = getattr(gbt_model_final, 'best_iteration', _best_params_bayes["n_estimators"]) + 1
print(f"   Best iteration: {_actual_iters}  (early stopping applied)")
print(f"   scale_pos_weight = {_spw:.2f}")

# ═══════════════════════════════════════════════════════════════
# 4. BASELINES
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📏 BASELINES")
print("═"*65)

_naive_rate          = float(y_gbt_train.mean())
gbt_naive_proba_test = np.full(len(y_gbt_test), _naive_rate)
print(f"   Naive positive rate: {_naive_rate*100:.2f}%")

_scaler_lr        = StandardScaler()
_Xtr_lr           = _scaler_lr.fit_transform(X_gbt_train.astype(np.float64))
_Xte_lr           = _scaler_lr.transform(X_gbt_test.astype(np.float64))
lr_gbt_model      = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                        random_state=42, class_weight="balanced")
lr_gbt_model.fit(_Xtr_lr, y_gbt_train)
gbt_lr_proba_test = lr_gbt_model.predict_proba(_Xte_lr)[:, 1]
print(f"   L2 LogReg trained  (C=1.0, balanced, max_iter=2000)")

# ═══════════════════════════════════════════════════════════════
# 5. ISOTONIC REGRESSION CALIBRATION
#    Fit IsotonicRegression on training raw probabilities → maps to calibrated
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📐 ISOTONIC CALIBRATION")
print("═"*65)

gbt_raw_proba_train = gbt_model_final.predict_proba(X_gbt_train)[:, 1]
gbt_raw_proba_test  = gbt_model_final.predict_proba(X_gbt_test)[:, 1]

# Fit isotonic regression on train raw proba → train labels
_iso_reg = IsotonicRegression(out_of_bounds="clip")
_iso_reg.fit(gbt_raw_proba_train, y_gbt_train)

gbt_calib_proba_test = _iso_reg.predict(gbt_raw_proba_test)

print(f"   Raw  GBT: [{gbt_raw_proba_test.min():.4f}, {gbt_raw_proba_test.max():.4f}]  "
      f"mean={gbt_raw_proba_test.mean():.4f}")
print(f"   Calibrated: [{gbt_calib_proba_test.min():.4f}, {gbt_calib_proba_test.max():.4f}]  "
      f"mean={gbt_calib_proba_test.mean():.4f}")
print(f"   Actual test rate: {y_gbt_test.mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# 6. METRIC HELPERS
# ═══════════════════════════════════════════════════════════════
def compute_lift(y_true, y_proba, pct=0.10):
    _n_top   = max(1, int(len(y_true) * pct))
    _top_idx = np.argsort(y_proba)[::-1][:_n_top]
    _base    = float(y_true.mean())
    return float(y_true[_top_idx].mean()) / _base if _base > 0 else 1.0

def tune_threshold_fbeta(y_true, y_proba, beta=2.0):
    _best_thresh, _best_fb = 0.5, 0.0
    for _t in np.linspace(0.01, 0.99, 200):
        _preds = (y_proba >= _t).astype(int)
        if _preds.sum() == 0: continue
        _fb = fbeta_score(y_true, _preds, beta=beta, zero_division=0)
        if _fb > _best_fb: _best_fb, _best_thresh = _fb, float(_t)
    return _best_thresh, _best_fb

def evaluate_model(name, y_true, y_proba):
    _pr_auc  = average_precision_score(y_true, y_proba) if y_true.sum() > 0 else 0.0
    _roc_auc = roc_auc_score(y_true, y_proba)           if y_true.sum() > 0 else 0.5
    _brier   = brier_score_loss(y_true, y_proba)
    _lift10  = compute_lift(y_true, y_proba, 0.10)
    _lift20  = compute_lift(y_true, y_proba, 0.20)
    _thresh, _fb = tune_threshold_fbeta(y_true, y_proba, beta=2.0)
    _preds   = (y_proba >= _thresh).astype(int)
    return {
        "model":               name,
        "pr_auc":              round(_pr_auc, 4),
        "roc_auc":             round(_roc_auc, 4),
        "brier":               round(_brier, 4),
        "lift_10":             round(_lift10, 3),
        "lift_20":             round(_lift20, 3),
        "f2_score":            round(_fb, 4),
        "opt_threshold":       round(_thresh, 3),
        "precision_at_thresh": round(precision_score(y_true, _preds, zero_division=0), 4),
        "recall_at_thresh":    round(recall_score(y_true, _preds, zero_division=0), 4),
    }

# ═══════════════════════════════════════════════════════════════
# 7. METRICS TABLE — all models on test set
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📊 TEST SET METRICS  — PR-AUC / ROC-AUC / Lift@10% / Lift@20%")
print("═"*65)

gbt_metrics_naive  = evaluate_model("Naive Rate",        y_gbt_test, gbt_naive_proba_test)
gbt_metrics_lr     = evaluate_model("L2 LogReg",         y_gbt_test, gbt_lr_proba_test)
gbt_metrics_raw    = evaluate_model("XGB GBT (raw)",     y_gbt_test, gbt_raw_proba_test)
gbt_metrics_calib  = evaluate_model("XGB GBT (calib.)",  y_gbt_test, gbt_calib_proba_test)

gbt_all_metrics = [gbt_metrics_naive, gbt_metrics_lr, gbt_metrics_raw, gbt_metrics_calib]
gbt_results_df  = pd.DataFrame(gbt_all_metrics)

print(f"\n{'Model':26s}  {'PR-AUC':>7s}  {'ROC-AUC':>8s}  {'Brier':>7s}  "
      f"{'Lift@10%':>9s}  {'Lift@20%':>9s}  {'F2(β=2)':>8s}  {'Thresh':>7s}")
print("-"*95)
for _, _r in gbt_results_df.iterrows():
    _star = " ← BEST" if _r["model"] == "XGB GBT (calib.)" else ""
    print(f"{_r['model']:26s}  {_r['pr_auc']:>7.4f}  {_r['roc_auc']:>8.4f}  "
          f"{_r['brier']:>7.4f}  {_r['lift_10']:>9.3f}  {_r['lift_20']:>9.3f}  "
          f"{_r['f2_score']:>8.4f}  {_r['opt_threshold']:>7.3f}{_star}")

# ═══════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

# ── Chart 1: PR + ROC Curves ──
fig_xgb_curves, (_ax_pr, _ax_roc) = plt.subplots(1, 2, figsize=(16, 7), facecolor=_BG)
for _ax in (_ax_pr, _ax_roc):
    _ax.set_facecolor(_BG)
    for _sp in ["top","right"]: _ax.spines[_sp].set_visible(False)
    _ax.spines["left"].set_color(_TXT2); _ax.spines["bottom"].set_color(_TXT2)
    _ax.tick_params(colors=_TXT2, labelsize=10)

for _nm, _yp, _col, _ls, _lw in [
    ("Naive Rate",        gbt_naive_proba_test, _COLS[3], "--", 1.5),
    ("L2 LogReg",         gbt_lr_proba_test,    _COLS[1], "-.", 2.0),
    ("XGB GBT (raw)",     gbt_raw_proba_test,   _COLS[0], "-",  2.0),
    ("XGB GBT (calib.)",  gbt_calib_proba_test, _HL,      "-",  2.8),
]:
    if y_gbt_test.sum() < 2: continue
    _prec, _rec, _ = precision_recall_curve(y_gbt_test, _yp)
    _fpr,  _tpr, _ = roc_curve(y_gbt_test, _yp)
    _ap  = average_precision_score(y_gbt_test, _yp)
    _auc = roc_auc_score(y_gbt_test, _yp)
    _ax_pr.plot(_rec, _prec, color=_col, linestyle=_ls, linewidth=_lw,
                label=f"{_nm} (AP={_ap:.3f})")
    _ax_roc.plot(_fpr, _tpr, color=_col, linestyle=_ls, linewidth=_lw,
                 label=f"{_nm} (AUC={_auc:.3f})")

_ax_pr.axhline(y_gbt_test.mean(), color=_TXT2, linestyle=":", linewidth=1, label="Random")
_ax_pr.set_xlabel("Recall", color=_TXT, fontsize=12)
_ax_pr.set_ylabel("Precision", color=_TXT, fontsize=12)
_ax_pr.set_title("Precision-Recall Curve", color=_TXT, fontsize=14, fontweight="bold", pad=12)
_ax_pr.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=9)
_ax_pr.set_xlim([0,1]); _ax_pr.set_ylim([0,1.02])
_ax_roc.plot([0,1],[0,1], color=_TXT2, linestyle=":", linewidth=1.2, label="Random (0.500)")
_ax_roc.set_xlabel("False Positive Rate", color=_TXT, fontsize=12)
_ax_roc.set_ylabel("True Positive Rate",  color=_TXT, fontsize=12)
_ax_roc.set_title("ROC Curve", color=_TXT, fontsize=14, fontweight="bold", pad=12)
_ax_roc.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=9)
_ax_roc.set_xlim([-0.01,1.01]); _ax_roc.set_ylim([-0.01,1.02])
plt.suptitle("XGBoost GBT vs Baselines — PR & ROC (Test: Nov 1–8 2025)",
             color=_TXT, fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(); plt.show()

# ── Chart 2: Reliability Diagram ──
fig_xgb_calibration, _ax_cal = plt.subplots(figsize=(10, 8), facecolor=_BG)
_ax_cal.set_facecolor(_BG)
for _sp in ["top","right"]: _ax_cal.spines[_sp].set_visible(False)
_ax_cal.spines["left"].set_color(_TXT2); _ax_cal.spines["bottom"].set_color(_TXT2)
_ax_cal.tick_params(colors=_TXT2)
_ax_cal.plot([0,1],[0,1], color=_TXT2, linestyle="--", linewidth=1.5, label="Perfect calibration")

for _nm, _yp, _col, _ls in [
    ("L2 LogReg",        gbt_lr_proba_test,    _COLS[1], "-."),
    ("XGB GBT (raw)",    gbt_raw_proba_test,   _COLS[0], "-"),
    ("XGB GBT (calib.)", gbt_calib_proba_test, _HL,      "-"),
]:
    _n_bins = max(3, min(8, int(y_gbt_test.sum() // 2)))
    _fp, _mp = calibration_curve(y_gbt_test, np.clip(_yp, 1e-6, 1-1e-6),
                                  n_bins=_n_bins, strategy="quantile")
    _ax_cal.plot(_mp, _fp, marker="o", color=_col, linestyle=_ls,
                 linewidth=2.2, markersize=7, label=_nm)

_ax_cal.set_xlabel("Mean Predicted Probability", color=_TXT, fontsize=13)
_ax_cal.set_ylabel("Fraction of Positives",      color=_TXT, fontsize=13)
_ax_cal.set_title("Reliability Diagram — Calibration Curve (Test Set)",
                   color=_TXT, fontsize=14, fontweight="bold", pad=15)
_ax_cal.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=11)
_ax_cal.set_xlim([-0.02,1.02]); _ax_cal.set_ylim([-0.02,1.02])
plt.tight_layout(); plt.show()

# ── Chart 3: HPO Convergence ──
fig_xgb_hpo, _ax_hpo = plt.subplots(figsize=(12, 6), facecolor=_BG)
_ax_hpo.set_facecolor(_BG)
for _sp in ["top","right"]: _ax_hpo.spines[_sp].set_visible(False)
_ax_hpo.spines["left"].set_color(_TXT2); _ax_hpo.spines["bottom"].set_color(_TXT2)
_ax_hpo.tick_params(colors=_TXT2)
_trials     = np.arange(1, len(_hpo_history)+1)
_best_curve = np.maximum.accumulate(_hpo_history)
_ax_hpo.scatter(_trials, _hpo_history, color=_COLS[0], alpha=0.55, s=35, label="Trial PR-AUC")
_ax_hpo.plot(_trials, _best_curve, color=_HL, linewidth=2.5, label="Best so far")
_ax_hpo.axvline(_N_EXPLORE + 0.5, color=_TXT2, linestyle=":", linewidth=1.2,
                label="Exploitation phase →")
_ax_hpo.set_xlabel("Trial", color=_TXT, fontsize=12)
_ax_hpo.set_ylabel("CV PR-AUC", color=_TXT, fontsize=12)
_ax_hpo.set_title("Bayesian HPO Convergence (Explore → Exploit)",
                   color=_TXT, fontsize=14, fontweight="bold", pad=12)
_ax_hpo.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
plt.tight_layout(); plt.show()

# ── Chart 4: Rolling CV ──
if len(rolling_cv_df) > 0:
    fig_xgb_rolling_cv, _ax_cv = plt.subplots(figsize=(10, 6), facecolor=_BG)
    _ax_cv.set_facecolor(_BG)
    for _sp in ["top","right"]: _ax_cv.spines[_sp].set_visible(False)
    _ax_cv.spines["left"].set_color(_TXT2); _ax_cv.spines["bottom"].set_color(_TXT2)
    _ax_cv.tick_params(colors=_TXT2)
    _cv_ws  = rolling_cv_df["window"].values
    _cv_pr  = rolling_cv_df["pr_auc"].values
    _cv_roc = rolling_cv_df["roc_auc"].values
    _ax_cv.plot(_cv_ws, _cv_pr,  "o-", color=_COLS[0], linewidth=2.5, markersize=9,
                markerfacecolor=_BG, markeredgewidth=2.5, label="PR-AUC")
    _ax_cv.plot(_cv_ws, _cv_roc, "s-", color=_COLS[1], linewidth=2.5, markersize=9,
                markerfacecolor=_BG, markeredgewidth=2.5, label="ROC-AUC")
    for _w, _pr, _roc in zip(_cv_ws, _cv_pr, _cv_roc):
        _ax_cv.annotate(f"{_pr:.3f}",  (_w, _pr + 0.012),  color=_COLS[0], fontsize=10, ha="center")
        _ax_cv.annotate(f"{_roc:.3f}", (_w, _roc + 0.012), color=_COLS[1], fontsize=10, ha="center")
    _ax_cv.set_xticks(_cv_ws)
    _ax_cv.set_xticklabels([f"Window {w}" for w in _cv_ws], color=_TXT, fontsize=11)
    _ax_cv.set_ylabel("Score", color=_TXT, fontsize=12)
    _ax_cv.set_title("3-Window Rolling CV (XGBoost, temporal expanding)",
                      color=_TXT, fontsize=14, fontweight="bold", pad=12)
    _ax_cv.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=11)
    _ax_cv.set_ylim([0, 1.05])
    plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════
# 9. BEST MODEL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🏆 BEST MODEL SELECTION SUMMARY")
print("═"*65)

_best_model_name = max(gbt_all_metrics, key=lambda m: m["pr_auc"])["model"]
_best_row        = next(m for m in gbt_all_metrics if m["model"] == _best_model_name)

print(f"\n   ✅ SELECTED: {_best_model_name}")
print(f"\n   PR-AUC:             {_best_row['pr_auc']:.4f}")
print(f"   ROC-AUC:            {_best_row['roc_auc']:.4f}")
print(f"   Brier Score:        {_best_row['brier']:.4f}  (lower = better)")
print(f"   Lift @ 10%:         {_best_row['lift_10']:.3f}×")
print(f"   Lift @ 20%:         {_best_row['lift_20']:.3f}×")
print(f"   F2-Score (β=2):     {_best_row['f2_score']:.4f}  (recall-weighted)")
print(f"   Optimal Threshold:  {_best_row['opt_threshold']:.3f}  (F-beta=2 tuned)")
print(f"   Precision @ thresh: {_best_row['precision_at_thresh']:.4f}")
print(f"   Recall @ thresh:    {_best_row['recall_at_thresh']:.4f}")
print(f"\n   🔑 Best XGBoost Hyperparameters (Bayesian HPO):")
for _k, _v in _best_params_bayes.items():
    print(f"      {_k:25s} = {_v}")
print(f"\n   scale_pos_weight:             {_spw:.2f}")
print(f"   Isotonic calibration:         applied (IsotonicRegression)")
print(f"   Decision threshold (F-β=2):   {_best_row['opt_threshold']:.3f}")
print(f"\n   Training: Sep–Oct 2025  ({len(X_gbt_train):,} users)")
print(f"   Test:     Nov 1–8 2025  ({len(X_gbt_test):,} users)")
print(f"   Features: {X_gbt_train.shape[1]}")

if len(rolling_cv_df) > 0:
    print(f"\n   3-Window Rolling CV:")
    print(f"      Mean PR-AUC  = {rolling_cv_df['pr_auc'].mean():.4f} ± {rolling_cv_df['pr_auc'].std():.4f}")
    print(f"      Mean ROC-AUC = {rolling_cv_df['roc_auc'].mean():.4f} ± {rolling_cv_df['roc_auc'].std():.4f}")

# Export
gbt_best_model      = gbt_model_final
gbt_isotonic_calib  = _iso_reg
gbt_best_params     = _best_params_bayes
gbt_train_df_out    = _train_df
gbt_test_df_out     = _test_df
gbt_calib_proba_out = gbt_calib_proba_test
gbt_lr_proba_out    = gbt_lr_proba_test
gbt_naive_proba_out = gbt_naive_proba_test
gbt_y_test_out      = y_gbt_test
gbt_hpo_history     = _hpo_history
gbt_results_df_out  = gbt_results_df

print("\n✅ XGBoost Bayesian HPO pipeline complete.")
