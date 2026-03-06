import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob

st.set_page_config(page_title="BuilderFlow Dashboard", page_icon="📈", layout="wide")

st.title("📈 BuilderFlow: Retention Risk Dashboard")
st.markdown("Monitor user retention risk, explore behavioral archetypes, and target interventions.")

# ── Data Loading ────────────────────────────────────────────────
# Try multiple possible locations for the CSV since Zerve's working
# directory may differ between canvas blocks and deployments.
@st.cache_data
def load_data():
    search_paths = [
        "scored_user_table.csv",
        "../scored_user_table.csv",
        "/mnt/user/scored_user_table.csv",
        os.path.expanduser("~/scored_user_table.csv"),
    ]
    # Also search recursively from common roots
    for root in ["/mnt", os.path.expanduser("~"), "."]:
        if os.path.isdir(root):
            found = glob.glob(os.path.join(root, "**", "scored_user_table.csv"), recursive=True)
            search_paths.extend(found)

    for path in search_paths:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            st.success(f"✅ Loaded data from `{path}` ({len(df):,} users)")
            return df

    # ── Fallback: use the scoring_df variable from Zerve shared namespace ──
    try:
        # In Zerve canvas context, globals may contain scoring_df
        if 'scored_user_table' in dir():
            st.success("✅ Loaded data from Zerve shared namespace")
            return scored_user_table
    except:
        pass

    # ── Fallback: generate demo data so the dashboard still renders ──
    st.warning("⚠️ `scored_user_table.csv` not found. Displaying **demo data** so you can preview the dashboard layout. "
               "Run all Canvas blocks first to populate real data.")
    np.random.seed(42)
    n = 1472
    archetypes = ["Casual Browser", "Onboarding Visitor", "Mid-Tier Engager", "Power User", "Hands-On Builder"]
    probs = [0.425, 0.202, 0.20, 0.06, 0.011]
    # Normalize
    probs = [p/sum(probs) for p in probs]
    arch = np.random.choice(archetypes, n, p=probs)

    risk = np.random.beta(2, 5, n)
    # Make risk correlate with archetype
    for i, a in enumerate(arch):
        if a == "Casual Browser":      risk[i] = np.clip(risk[i] + 0.45, 0, 1)
        elif a == "Onboarding Visitor": risk[i] = np.clip(risk[i] + 0.30, 0, 1)
        elif a == "Mid-Tier Engager":   risk[i] = np.clip(risk[i] + 0.10, 0, 1)
        elif a == "Hands-On Builder":   risk[i] = np.clip(risk[i] - 0.15, 0, 1)

    uplift = np.random.uniform(0.01, 0.12, n)
    priority = 0.4 * (1 - risk) + 0.4 * uplift + 0.2 * np.random.uniform(0, 1, n)

    return pd.DataFrame({
        "user_id_canon": [f"demo_user_{i:04d}" for i in range(n)],
        "archetype": arch,
        "predicted_risk": risk,
        "uplift_agent_to_block_conversion_UI_flow": uplift,
        "priority_agent_to_block_conversion_UI_flow": priority,
    })

df = load_data()

# ── Detect column names (handle both naming conventions) ──
uplift_col = next((c for c in df.columns if "uplift" in c.lower() and "agent" in c.lower()), None)
priority_col = next((c for c in df.columns if "priority" in c.lower() and "agent" in c.lower()), None)

if uplift_col is None:
    uplift_col = next((c for c in df.columns if "uplift" in c.lower()), "predicted_risk")
if priority_col is None:
    priority_col = next((c for c in df.columns if "priority" in c.lower()), "predicted_risk")

# ── Top-Level KPI Metrics ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users Scored", f"{len(df):,}")

high_risk = len(df[df["predicted_risk"] > 0.85])
col2.metric("🔴 Critical Risk (>85%)", f"{high_risk:,}",
            f"{(high_risk/len(df))*100:.1f}% of users", delta_color="inverse")

avg_uplift = df[uplift_col].mean() * 100
col3.metric("Avg Potential Uplift", f"+{avg_uplift:.2f}pp")

if "archetype" in df.columns:
    builders = len(df[df["archetype"] == "Hands-On Builder"])
    col4.metric("🟢 Hands-On Builders", f"{builders:,}")

st.markdown("---")

# ── Layout ──────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Retention Risk by Behavioral Archetype")
    fig1 = px.box(
        df, x="archetype", y="predicted_risk", color="archetype",
        title="Churn Risk Distribution Across Archetypes",
        labels={"predicted_risk": "Predicted Churn Risk", "archetype": "Behavioral Archetype"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(showlegend=False, template="plotly_dark",
                       plot_bgcolor="#1D1D20", paper_bgcolor="#1D1D20")
    st.plotly_chart(fig1, use_container_width=True)

with right_col:
    st.subheader("Risk Tier Breakdown")
    bins = [0, 0.3, 0.6, 0.85, 1.0]
    labels = ["🟢 Low (<30%)", "🟡 Medium (30-60%)", "🟠 High (60-85%)", "🔴 Critical (>85%)"]
    df["risk_tier"] = pd.cut(df["predicted_risk"], bins=bins, labels=labels, include_lowest=True)
    tier_counts = df["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Count"]

    fig2 = px.pie(
        tier_counts, values="Count", names="Risk Tier", hole=0.4,
        color="Risk Tier",
        color_discrete_map={
            "🟢 Low (<30%)": "#17b26a",
            "🟡 Medium (30-60%)": "#fadb14",
            "🟠 High (60-85%)": "#fa8c16",
            "🔴 Critical (>85%)": "#f5222d",
        },
    )
    fig2.update_layout(template="plotly_dark",
                       plot_bgcolor="#1D1D20", paper_bgcolor="#1D1D20")
    st.plotly_chart(fig2, use_container_width=True)

# ── Priority Intervention Table ─────────────────────────────────
st.markdown("---")
st.subheader("🎯 Top 50 Users for Priority Intervention (Agent → Block UI)")

top_targets = df.sort_values(by=priority_col, ascending=False).head(50)

display_df = top_targets[["user_id_canon", "archetype", "predicted_risk", uplift_col, priority_col]].copy()
display_df["predicted_risk"] = (display_df["predicted_risk"] * 100).round(1).astype(str) + "%"
display_df[uplift_col] = "+" + (display_df[uplift_col] * 100).round(2).astype(str) + "pp"
display_df[priority_col] = display_df[priority_col].round(3)

display_df.columns = ["User ID", "Archetype", "Churn Risk", "Expected Uplift", "Priority Score"]
display_df["User ID"] = display_df["User ID"].astype(str).str[:16] + "..."

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption("Powered by the BuilderFlow Canvas pipeline. Deploy on Zerve for live scoring.")
