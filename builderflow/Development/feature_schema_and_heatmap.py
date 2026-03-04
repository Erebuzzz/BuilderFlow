
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE SCHEMA LOGGING + CORRELATION HEATMAP
# Consumes clean_feature_matrix → saves schema as JSON, renders heatmap
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────
# Zerve Design System
# ─────────────────────────────────────────────────────
BG       = "#1D1D20"
TXT      = "#fbfbff"
TXT2     = "#909094"
ZERVE_C  = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
            "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
HL       = "#ffd400"
SUCCESS  = "#17b26a"
WARN     = "#f04438"

# ─────────────────────────────────────────────────────
# 1. IDENTIFY FEATURE COLUMNS
# ─────────────────────────────────────────────────────
_label_cols = ["ret30d", "ret90d", "upg60d"]
_meta_cols  = ["user_id_canon", "split"]
_feat_cols  = [c for c in clean_feature_matrix.columns if c not in _label_cols + _meta_cols]

print(f"{'='*65}")
print(f"📦  CLEAN FEATURE MATRIX SCHEMA")
print(f"{'='*65}")
print(f"   Shape            : {clean_feature_matrix.shape[0]:,} users × {len(_feat_cols)} features")
print(f"   Labels           : {_label_cols}")
print(f"   Split dist       : {clean_feature_matrix['split'].value_counts().to_dict()}")
print()

# ─────────────────────────────────────────────────────
# 2. SCHEMA — per-feature metadata
# ─────────────────────────────────────────────────────
_schema_rows = []
for _f in _feat_cols:
    _s = clean_feature_matrix[_f]
    _row = {
        "feature"  : _f,
        "dtype"    : str(_s.dtype),
        "n_null"   : int(_s.isnull().sum()),
        "mean"     : round(float(_s.mean()), 6),
        "std"      : round(float(_s.std()),  6),
        "min"      : round(float(_s.min()),  6),
        "max"      : round(float(_s.max()),  6),
        "n_unique" : int(_s.nunique()),
        "is_binary": bool(set(_s.dropna().unique()).issubset({0, 1, 0.0, 1.0})),
        "group"    : (
            "derived"       if any(_f.startswith(p) for p in [
                "feat_session_entropy", "feat_tod_entropy", "feat_day_gap_variance",
                "feat_execution_to_agent_ratio", "feat_canvas_creation_rate",
                "feat_agent_block_conversion"]) else
            "target_encoded" if _f.startswith("feat_te_") else
            "ttf"            if _f.startswith("feat_ttf_") else
            "ratio"          if "ratio" in _f else
            "session"        if "session" in _f else
            "intensity"      if _f in [
                "feat_event_count","feat_active_days","feat_n_sessions",
                "feat_events_per_day","feat_rampup_slope","feat_max_gap_days"] else
            "breadth"        if "distinct" in _f or "diversity" in _f else
            "onboarding"     if "onboarding" in _f or "tour" in _f else
            "deploy"         if "deploy" in _f or "schedule" in _f else
            "collab"         if "collab" in _f or "canvas" in _f else
            "metadata"       if _f.startswith("feat_signup") or "credit" in _f else
            "other"
        ),
    }
    _schema_rows.append(_row)

feature_schema = pd.DataFrame(_schema_rows)

# Save schema to JSON
_schema_json = feature_schema.to_dict(orient="records")
with open("feature_schema_7d.json", "w") as _fh:
    json.dump(_schema_json, _fh, indent=2)
print(f"✅  Feature schema saved → feature_schema_7d.json")

# Print schema summary
print(f"\n📋  SCHEMA SUMMARY BY GROUP:")
for _grp, _gdf in feature_schema.groupby("group"):
    _binary_n = _gdf["is_binary"].sum()
    print(f"   {_grp:20s}: {len(_gdf):3d} features  (binary={_binary_n})")

print(f"\n📋  ALL FEATURES:")
_schema_disp = feature_schema[["feature","dtype","mean","std","min","max","group"]].copy()
_schema_disp["mean"] = _schema_disp["mean"].map("{:.4f}".format)
_schema_disp["std"]  = _schema_disp["std"].map("{:.4f}".format)
_schema_disp["min"]  = _schema_disp["min"].map("{:.4f}".format)
_schema_disp["max"]  = _schema_disp["max"].map("{:.4f}".format)
print(_schema_disp.to_string(index=False))

# ─────────────────────────────────────────────────────
# 3. HIGHLIGHT: 6 DERIVED FEATURES
# ─────────────────────────────────────────────────────
_derived_features = [
    "feat_session_entropy",
    "feat_tod_entropy",
    "feat_day_gap_variance",
    "feat_execution_to_agent_ratio",
    "feat_canvas_creation_rate",
    "feat_agent_block_conversion",
]
print(f"\n✅  DERIVED FEATURES (6/6 confirmed in matrix):")
for _df in _derived_features:
    _present = _df in _feat_cols
    _row = feature_schema[feature_schema["feature"] == _df]
    if _present and len(_row):
        _r = _row.iloc[0]
        print(f"   {'✅' if _present else '❌'} {_df:45s}  mean={_r['mean']:.4f}  std={_r['std']:.4f}")
    else:
        print(f"   ❌ {_df}  (NOT FOUND)")

# ─────────────────────────────────────────────────────
# 4. CORRELATION HEATMAP — numeric features only
# ─────────────────────────────────────────────────────
_num_feats = clean_feature_matrix[_feat_cols].select_dtypes(include=[np.number]).columns.tolist()
_corr_mat  = clean_feature_matrix[_num_feats].corr()

# For readability, use shortened labels
_short_labels = [
    c.replace("feat_", "")
     .replace("_per_", "/")
     .replace("primary_", "")
     .replace("_count", "_cnt")
     .replace("_duration", "_dur")
     .replace("_distinct", "_dist")
     .replace("early_", "")
     .replace("onboarding", "ob")
     .replace("execution_to_agent", "exec/agent")
     .replace("canvas_creation_rate", "canvas_rate")
     .replace("session_entropy", "sess_ent")
     .replace("tod_entropy", "tod_ent")
     .replace("day_gap_variance", "gap_var")
     .replace("agent_block_conversion", "agent_conv")
    for c in _num_feats
]

_n = len(_num_feats)
_fig_h = max(14, _n * 0.35)
_fig_w = max(16, _n * 0.38)

fig_corr_heatmap, ax_hm = plt.subplots(figsize=(_fig_w, _fig_h), facecolor=BG)
ax_hm.set_facecolor(BG)

# Custom diverging colormap: dark blue → white → orange
from matplotlib.colors import LinearSegmentedColormap
_cmap = LinearSegmentedColormap.from_list(
    "zerve_div",
    ["#1F77B4", BG, "#FFB482"],
    N=256
)

_im = ax_hm.imshow(_corr_mat.values, cmap=_cmap, vmin=-1, vmax=1, aspect="auto")

# Colorbar
_cb = plt.colorbar(_im, ax=ax_hm, fraction=0.025, pad=0.01)
_cb.ax.tick_params(colors=TXT2, labelsize=8)
_cb.set_label("Pearson r", color=TXT2, fontsize=10)

# Axis labels
ax_hm.set_xticks(range(_n))
ax_hm.set_yticks(range(_n))
ax_hm.set_xticklabels(_short_labels, rotation=75, ha="right", fontsize=6.5, color=TXT2)
ax_hm.set_yticklabels(_short_labels, fontsize=6.5, color=TXT2)

# Mark high-correlation pairs (> 0.92 in abs) — should be none after filtering
_above_thresh = np.abs(_corr_mat.values) > 0.92
np.fill_diagonal(_above_thresh, False)
_hi_pairs = list(zip(*np.where(_above_thresh)))
if _hi_pairs:
    for _r, _c in _hi_pairs:
        ax_hm.add_patch(plt.Rectangle((_c - 0.5, _r - 0.5), 1, 1,
                                       fill=False, edgecolor=WARN, lw=1.5))
    print(f"\n⚠️  {len(_hi_pairs)//2} high-corr pairs (|r|>0.92) found in final matrix (highlighted in red)")
else:
    print(f"\n✅  No high-correlation pairs (|r|>0.92) remain in final matrix")

# Highlight derived features with golden tick marks
for _i, _lbl in enumerate(_short_labels):
    _derived_short = [
        "sess_ent", "tod_ent", "gap_var",
        "exec/agent_ratio", "canvas_rate", "agent_conv",
        "te_country", "te_device", "te_os", "te_browser",
    ]
    if any(_d in _lbl for _d in _derived_short):
        ax_hm.get_xticklabels()[_i].set_color(HL)
        ax_hm.get_yticklabels()[_i].set_color(HL)

ax_hm.set_title(
    "Feature Correlation Matrix — 7-Day Window\n(gold labels = derived/target-encoded features)",
    color=TXT, fontsize=13, fontweight="bold", pad=14
)

plt.tight_layout()
plt.show()

print(f"\n✅  Correlation heatmap rendered with {_n} numeric features")
print(f"   Max |r| (off-diag): {_corr_mat.where(~np.eye(_n, dtype=bool)).abs().max().max():.4f}")
