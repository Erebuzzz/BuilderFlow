
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")

_BG    = "#1D1D20"
_TXT   = "#fbfbff"
_TXT2  = "#909094"
_COLS  = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
          "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL    = "#ffd400"

# ═════════════════════════════════════════════════════════════════
# 0. BUILD CLUSTERING FEATURE MATRIX — 5 behavioral features
# ═════════════════════════════════════════════════════════════════
_CLUSTER_COLS = [
    "feat_n_sessions",
    "feat_ratio_block_ops",
    "feat_ratio_agent",
    "feat_ratio_onboarding",
    "feat_active_days",
]

_df = clean_feature_matrix[
    ["user_id_canon", "ret30d", "ret90d", "upg60d"] + _CLUSTER_COLS
].copy().reset_index(drop=True)

print("📊 Clustering on 5 behavioral features:")
for _c in _CLUSTER_COLS:
    print(f"   • {_c}: mean={_df[_c].mean():.3f}, std={_df[_c].std():.3f}")
print(f"   Total users: {len(_df):,}")

_X_raw    = _df[_CLUSTER_COLS].values.copy()
_scaler   = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_raw)

# ═════════════════════════════════════════════════════════════════
# 1. SILHOUETTE SCORES FOR k=4,5,6  (5 seeds for stability)
# ═════════════════════════════════════════════════════════════════
_K_VALUES    = [4, 5, 6]
_N_SEEDS     = 5
_sil_matrix   = {}
_label_matrix = {}

print("\n" + "="*65)
print("📐 SILHOUETTE SCORE + STABILITY ANALYSIS")
print("="*65)

for _k in _K_VALUES:
    _sil_scores   = []
    _label_arrays = []
    for _seed in range(42, 42 + _N_SEEDS):
        _km  = KMeans(n_clusters=_k, random_state=_seed, n_init=15, max_iter=500)
        _lbl = _km.fit_predict(_X_scaled)
        _sil = silhouette_score(_X_scaled, _lbl)
        _sil_scores.append(_sil)
        _label_arrays.append(_lbl)
    _sil_matrix[_k]   = _sil_scores
    _label_matrix[_k] = _label_arrays
    print(f"   k={_k}: silhouette = {np.mean(_sil_scores):.4f} ± {np.std(_sil_scores):.4f}"
          f"  (min={np.min(_sil_scores):.4f}, max={np.max(_sil_scores):.4f})")

# ═════════════════════════════════════════════════════════════════
# 2. CLUSTER ASSIGNMENT CONSISTENCY (ARI across seed pairs)
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("🔁 CLUSTER ASSIGNMENT CONSISTENCY (ARI across seed pairs)")
print("="*65)

_ari_scores = {}
for _k in _K_VALUES:
    _lbls  = _label_matrix[_k]
    _pairs = [(i, j) for i in range(len(_lbls)) for j in range(i+1, len(_lbls))]
    _aris  = [adjusted_rand_score(_lbls[a], _lbls[b]) for a, b in _pairs]
    _ari_scores[_k] = _aris
    print(f"   k={_k}: ARI = {np.mean(_aris):.4f} ± {np.std(_aris):.4f}"
          f"  (1.0 = perfectly consistent)")

# ═════════════════════════════════════════════════════════════════
# 3. SELECT BEST k — composite (60% silhouette + 40% ARI)
# ═════════════════════════════════════════════════════════════════
_mean_sil = {k: float(np.mean(v)) for k, v in _sil_matrix.items()}
_mean_ari = {k: float(np.mean(v)) for k, v in _ari_scores.items()}

_sil_vals = np.array([_mean_sil[k] for k in _K_VALUES], dtype=float)
_ari_vals = np.array([_mean_ari[k] for k in _K_VALUES], dtype=float)

_sil_rng = float(_sil_vals.max() - _sil_vals.min())
_ari_rng = float(_ari_vals.max() - _ari_vals.min())

_sil_norm  = (_sil_vals - _sil_vals.min()) / (_sil_rng + 1e-9)
_ari_norm  = (_ari_vals - _ari_vals.min()) / (_ari_rng + 1e-9)
_composite = 0.6 * _sil_norm + 0.4 * _ari_norm

_best_idx      = int(np.argmax(_composite))
cluster_best_k = _K_VALUES[_best_idx]

print("\n" + "="*65)
print("🏆 MODEL SELECTION SUMMARY")
print("="*65)
print(f"   {'k':>4s}  {'Silhouette':>12s}  {'ARI':>8s}  {'Composite':>10s}")
print(f"   {'-'*40}")
for _i, _k in enumerate(_K_VALUES):
    _marker = " ← BEST" if _k == cluster_best_k else ""
    print(f"   {_k:>4d}  {_mean_sil[_k]:>12.4f}  {_mean_ari[_k]:>8.4f}  {float(_composite[_i]):>10.4f}{_marker}")

# ═════════════════════════════════════════════════════════════════
# 4. FIT FINAL KMEANS WITH BEST k
# ═════════════════════════════════════════════════════════════════
_km_final     = KMeans(n_clusters=cluster_best_k, random_state=42, n_init=30, max_iter=500)
_labels_final = _km_final.fit_predict(_X_scaled)
_df["cluster_id"] = _labels_final

# ═════════════════════════════════════════════════════════════════
# 5. CENTROID PROFILES — in original feature space
# ═════════════════════════════════════════════════════════════════
_centroids_orig = _scaler.inverse_transform(_km_final.cluster_centers_)
_centroid_df    = pd.DataFrame(_centroids_orig, columns=_CLUSTER_COLS)
_centroid_df["cluster_id"] = list(range(cluster_best_k))
_centroid_df["n_users"]    = [int((_labels_final == _c).sum()) for _c in range(cluster_best_k)]
_centroid_df = _centroid_df.reset_index(drop=True)

print("\n" + "="*65)
print("📊 CENTROID PROFILES (original scale)")
print("="*65)
_feat_labels = ["Sessions", "Exec Ratio", "Agent Ratio", "Onboard Ratio", "Active Days"]
print("   " + f"{'Cluster':>8s}  {'N':>6s}  " + "  ".join(f"{l:>12s}" for l in _feat_labels))
print("   " + "-"*80)
for _c in range(cluster_best_k):
    _row  = _centroid_df[_centroid_df["cluster_id"] == _c].iloc[0]
    _vals = [float(_row[col]) for col in _CLUSTER_COLS]
    _n    = int(_row["n_users"])
    print("   " + f"{'C' + str(_c):>8s}  {_n:>6d}  " + "  ".join(f"{v:>12.3f}" for v in _vals))

# ═════════════════════════════════════════════════════════════════
# 6. ASSIGN ARCHETYPE LABELS — data-driven from centroid profiles
# ═════════════════════════════════════════════════════════════════
_engagement = _centroid_df["feat_n_sessions"] * _centroid_df["feat_active_days"]
_sorted_ids = list(_engagement.argsort().values)

cluster_archetype_names = {}
for _rank, _c in enumerate(_sorted_ids):
    _row       = _centroid_df[_centroid_df["cluster_id"] == _c].iloc[0]
    _sessions  = float(_row["feat_n_sessions"])
    _exec_r    = float(_row["feat_ratio_block_ops"])
    _agent_r   = float(_row["feat_ratio_agent"])
    _onboard_r = float(_row["feat_ratio_onboarding"])
    _days      = float(_row["feat_active_days"])
    _n         = int(_row["n_users"])

    if _sessions <= 1.5 and _days <= 1.1 and _agent_r > 0.5:
        _name = "AI Sampler"
    elif _sessions <= 2.5 and _onboard_r > 0.25:
        _name = "Onboarding Visitor"
    elif _sessions <= 2.5 and _days <= 1.3:
        _name = "Casual Browser"
    elif _agent_r > 0.45 and _sessions > 2:
        _name = "AI-First Explorer"
    elif _exec_r > 0.12 and _days >= 2:
        _name = "Hands-On Builder"
    elif _days >= 3:
        _name = "Power User"
    else:
        _name = "Mid-Tier Engager"

    cluster_archetype_names[int(_c)] = _name
    _pct = _n / len(_df) * 100
    print(f"\n🏷️  Cluster {_c} → '{_name}'  (n={_n}, {_pct:.1f}%)")
    print(f"     sessions={_sessions:.1f}, exec%={_exec_r*100:.0f}%, agent%={_agent_r*100:.0f}%, "
          f"onboard%={_onboard_r*100:.0f}%, days={_days:.1f}")

_df["archetype"] = _df["cluster_id"].map(cluster_archetype_names)

# Add archetype column to the full feature matrix
feature_matrix_with_archetypes = clean_feature_matrix.copy()
feature_matrix_with_archetypes = feature_matrix_with_archetypes.merge(
    _df[["user_id_canon", "cluster_id", "archetype"]], on="user_id_canon", how="left"
)
print(f"\n✅ 'archetype' column added to feature matrix  ({len(feature_matrix_with_archetypes):,} rows)")
print(f"   Archetype distribution:\n{feature_matrix_with_archetypes['archetype'].value_counts().to_string()}")

# ═════════════════════════════════════════════════════════════════
# 7. OUTCOME RATES BY ARCHETYPE
# ═════════════════════════════════════════════════════════════════
cluster_outcome_rates = _df.groupby("archetype").agg(
    n_users = ("user_id_canon", "count"),
    ret30d  = ("ret30d",  "mean"),
    ret90d  = ("ret90d",  "mean"),
    upg60d  = ("upg60d",  "mean"),
).reset_index()

_archetype_to_cluster = {v: k for k, v in cluster_archetype_names.items()}
cluster_outcome_rates["cluster_id"]        = cluster_outcome_rates["archetype"].map(_archetype_to_cluster)
cluster_outcome_rates["silhouette_k_best"] = round(_mean_sil[cluster_best_k], 4)
cluster_outcome_rates = cluster_outcome_rates.sort_values("ret90d", ascending=False).reset_index(drop=True)

_ov30 = float(_df["ret30d"].mean())
_ov90 = float(_df["ret90d"].mean())
_ovup = float(_df["upg60d"].mean())

print("\n" + "="*75)
print("📊 PER-CLUSTER OUTCOME RATES")
print("="*75)
print(f"{'Archetype':30s} {'N':>6s} {'ret30d':>8s} {'ret90d':>8s} {'upg60d':>8s}")
print("-"*75)
for _, _r in cluster_outcome_rates.iterrows():
    print(f"{_r['archetype']:30s} {int(_r['n_users']):>6d} "
          f"{_r['ret30d']*100:>7.1f}% {_r['ret90d']*100:>7.1f}% {_r['upg60d']*100:>7.1f}%")
print("-"*75)
print(f"{'OVERALL':30s} {len(_df):>6d} {_ov30*100:>7.1f}% {_ov90*100:>7.1f}% {_ovup*100:>7.1f}%")

# ═════════════════════════════════════════════════════════════════
# 8. CHART 1 — Silhouette & ARI stability across k
# ═════════════════════════════════════════════════════════════════
fig_silhouette_stability, (_ax_sil, _ax_ari) = plt.subplots(1, 2, figsize=(14, 6), facecolor=_BG)

for _ax in (_ax_sil, _ax_ari):
    _ax.set_facecolor(_BG)
    for _sp in ["top", "right"]:
        _ax.spines[_sp].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)
    _ax.tick_params(colors=_TXT2, labelsize=11)

_sil_data = [_sil_matrix[k] for k in _K_VALUES]
_bp = _ax_sil.boxplot(_sil_data, labels=[f"k={k}" for k in _K_VALUES],
                      patch_artist=True, medianprops=dict(color=_HL, linewidth=2.5))
for _patch, _col in zip(_bp["boxes"], [_COLS[0], _COLS[1], _COLS[2]]):
    _patch.set_facecolor(_col)
    _patch.set_alpha(0.75)
for _wh in _bp["whiskers"] + _bp["caps"] + _bp["fliers"]:
    _wh.set_color(_TXT2)

_best_x = _K_VALUES.index(cluster_best_k) + 1
_ax_sil.axvline(_best_x, color=_HL, linewidth=1.5, linestyle="--", alpha=0.7)
_ax_sil.set_title("Silhouette Score by k (5 seeds)", color=_TXT, fontsize=13, fontweight="bold", pad=12)
_ax_sil.set_ylabel("Silhouette Score", color=_TXT, fontsize=12)
_ax_sil.set_xlabel("Number of Clusters", color=_TXT, fontsize=12)

_ari_data = [_ari_scores[k] for k in _K_VALUES]
_bp2 = _ax_ari.boxplot(_ari_data, labels=[f"k={k}" for k in _K_VALUES],
                       patch_artist=True, medianprops=dict(color=_HL, linewidth=2.5))
for _patch, _col in zip(_bp2["boxes"], [_COLS[0], _COLS[1], _COLS[2]]):
    _patch.set_facecolor(_col)
    _patch.set_alpha(0.75)
for _wh in _bp2["whiskers"] + _bp2["caps"] + _bp2["fliers"]:
    _wh.set_color(_TXT2)

_ax_ari.axvline(_best_x, color=_HL, linewidth=1.5, linestyle="--", alpha=0.7)
_ax_ari.set_title("Assignment Consistency (ARI)", color=_TXT, fontsize=13, fontweight="bold", pad=12)
_ax_ari.set_ylabel("Adjusted Rand Index", color=_TXT, fontsize=12)
_ax_ari.set_xlabel("Number of Clusters", color=_TXT, fontsize=12)

plt.suptitle(f"KMeans Stability Evaluation — Best k={cluster_best_k}  (sil={_mean_sil[cluster_best_k]:.4f})",
             color=_TXT, fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════════════════
# 9. CHART 2 — Outcome rates by archetype
# ═════════════════════════════════════════════════════════════════
fig_archetype_outcomes, _axo = plt.subplots(figsize=(14, 7), facecolor=_BG)
_axo.set_facecolor(_BG)

_archetypes = cluster_outcome_rates["archetype"].values
_x          = np.arange(len(_archetypes))
_width      = 0.26

_metrics_map = {"ret30d": "30-Day Retention", "ret90d": "90-Day Retention", "upg60d": "60-Day Upgrade"}
for _mi, (_col_name, _label) in enumerate(_metrics_map.items()):
    _vals   = cluster_outcome_rates[_col_name].values * 100
    _offset = (_mi - 1) * _width
    _bars   = _axo.bar(_x + _offset, _vals, _width, label=_label,
                       color=_COLS[_mi], edgecolor="none", alpha=0.9)
    for _bar, _v in zip(_bars, _vals):
        if _v > 0.5:
            _axo.text(_bar.get_x() + _bar.get_width() / 2, float(_bar.get_height()) + 0.4,
                      f"{_v:.1f}%", ha="center", va="bottom", color=_TXT, fontsize=8, fontweight="bold")

_axo.set_xticks(_x)
_axo.set_xticklabels(_archetypes, rotation=20, ha="right", fontsize=10, color=_TXT)
_axo.set_ylabel("Rate (%)", color=_TXT, fontsize=12)
_axo.set_title(f"Retention & Upgrade Rates by User Archetype  (k={cluster_best_k})",
               color=_TXT, fontsize=14, fontweight="bold", pad=15)
_axo.tick_params(colors=_TXT2, labelsize=10)
_axo.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
for _sp in ["top", "right"]:
    _axo.spines[_sp].set_visible(False)
_axo.spines["bottom"].set_color(_TXT2)
_axo.spines["left"].set_color(_TXT2)
_axo.axhline(_ov30 * 100, color=_COLS[0], linestyle="--", alpha=0.35, linewidth=1)
_axo.axhline(_ov90 * 100, color=_COLS[1], linestyle="--", alpha=0.35, linewidth=1)
_axo.axhline(_ovup * 100, color=_COLS[2], linestyle="--", alpha=0.35, linewidth=1)
plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════════════════
# 10. CHART 3 — Normalized feature profiles per archetype
# ═════════════════════════════════════════════════════════════════
_feat_display = ["Sessions", "Exec Ratio", "Agent Ratio", "Onboard Ratio", "Active Days"]

_centroids_norm = _centroid_df[_CLUSTER_COLS].copy().values.astype(float)
for _fi in range(_centroids_norm.shape[1]):
    _cmin = _centroids_norm[:, _fi].min()
    _cmax = _centroids_norm[:, _fi].max()
    _centroids_norm[:, _fi] = (_centroids_norm[:, _fi] - _cmin) / (_cmax - _cmin + 1e-9)

fig_archetype_profiles, _axp = plt.subplots(figsize=(14, 6), facecolor=_BG)
_axp.set_facecolor(_BG)

_n_archetypes = cluster_best_k
_bar_width    = 0.8 / _n_archetypes
_x_feat       = np.arange(len(_feat_display))

for _ci in range(cluster_best_k):
    _arch_name = cluster_archetype_names.get(_ci, f"Cluster {_ci}")
    _norm_vals = _centroids_norm[_ci]
    _offset    = (_ci - _n_archetypes / 2 + 0.5) * _bar_width
    _axp.bar(_x_feat + _offset, _norm_vals, _bar_width,
             label=_arch_name, color=_COLS[_ci % len(_COLS)], edgecolor="none", alpha=0.88)

_axp.set_xticks(_x_feat)
_axp.set_xticklabels(_feat_display, fontsize=11, color=_TXT)
_axp.set_ylabel("Normalized Centroid Value (0-1)", color=_TXT, fontsize=11)
_axp.set_title("Behavioral Feature Profiles by Archetype", color=_TXT, fontsize=14, fontweight="bold", pad=15)
_axp.tick_params(colors=_TXT2, labelsize=10)
_axp.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=9, ncol=2)
for _sp in ["top", "right"]:
    _axp.spines[_sp].set_visible(False)
_axp.spines["bottom"].set_color(_TXT2)
_axp.spines["left"].set_color(_TXT2)
plt.tight_layout()
plt.show()

print(f"\n✅ KMeans archetype clustering complete.")
print(f"   • Best k = {cluster_best_k}  (silhouette = {_mean_sil[cluster_best_k]:.4f})")
print(f"   • Archetype labels assigned to all {len(_df):,} users")
print(f"   • feature_matrix_with_archetypes: {feature_matrix_with_archetypes.shape}")
print(f"   • cluster_outcome_rates: {len(cluster_outcome_rates)} rows × {len(cluster_outcome_rates.columns)} cols")
