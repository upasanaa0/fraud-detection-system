import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Detection Dashboard")
st.caption("Synthetic transaction data with planted fraud scenarios. Metrics below measure "
           "detection of those scenarios, not real-world fraud performance.")

OUTPUT_PATH = "data/final_output.csv"
REPORTS_PATH = "reports"
THRESHOLDS_PATH = "model/thresholds.json"
METRICS_PATH = f"{REPORTS_PATH}/metrics.json"

if not os.path.exists(OUTPUT_PATH) or not os.path.exists(THRESHOLDS_PATH):
    st.error("Run the pipeline first: python main.py")
    st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data(OUTPUT_PATH)
T = json.load(open(THRESHOLDS_PATH))          # thresholds chosen on validation data

# ── KPI cards ─────────────────────────────────────────────────────────────────
total = len(df)
blocked = int((df["decision"] == "BLOCK").sum())
challenged = int((df["decision"] == "CHALLENGE").sum())
allowed = int((df["decision"] == "ALLOW").sum())
fraud_rate = df["is_fraud"].mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Rate", f"{fraud_rate:.2f}%")
c3.metric("BLOCKED", f"{blocked:,}", delta=f"{blocked / total * 100:.1f}%")
c4.metric("CHALLENGED", f"{challenged:,}", delta=f"{challenged / total * 100:.1f}%")
c5.metric("ALLOWED", f"{allowed:,}", delta=f"{allowed / total * 100:.1f}%")

# ── Measured test-set performance ─────────────────────────────────────────────
if os.path.exists(METRICS_PATH):
    M = json.load(open(METRICS_PATH))
    st.subheader("Test-set performance (unseen later period)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", f"{M['roc_auc']:.3f}")
    m2.metric("PR-AUC", f"{M['pr_auc']:.3f}", help=f"Random baseline = {M['pr_auc_random_baseline']:.3f}")
    m3.metric("Recall @ 1% FPR", f"{M['recall_at_1pct_fpr']:.1%}")
    m4.metric("BLOCK precision", f"{M['block_precision']:.1%}")
    m5.metric("Legit blocked", f"{M['false_block_rate_of_legit']:.2%}")

st.divider()

# ── High Risk Alerts (BLOCK tier) ─────────────────────────────────────────────
st.subheader("🚨 High Risk Alerts")

block_df = df[df["decision"] == "BLOCK"].sort_values("final_risk_score", ascending=False)
top_alerts = block_df.head(10)

if len(block_df) > 0:
    st.download_button(
        label=f"⬇️ Download All {len(block_df):,} BLOCKED Transactions",
        data=block_df.to_csv(index=False).encode("utf-8"),
        file_name="blocked_transactions.csv",
        mime="text/csv",
    )
    st.write(f"**Showing the top {len(top_alerts)} of {len(block_df):,} blocked transactions:**")

    for _, row in top_alerts.iterrows():
        with st.expander(
            f"🚫 BLOCK | User: {int(row['user_id'])} | "
            f"₹{row['amount']:,.2f} | Score: {row['final_risk_score']:.4f}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Transaction Info**")
                st.write(f"User ID: `{int(row['user_id'])}`")
                st.write(f"Amount: `₹{row['amount']:,.2f}`")
                st.write(f"Merchant ID: `{int(row['merchant_id'])}`")
                if "city" in df.columns:
                    st.write(f"City: `{row['city']}`")
                if "payment_mode" in df.columns:
                    st.write(f"Payment Mode: `{row['payment_mode']}`")

            with col2:
                st.write("**Risk Signals**")
                st.write(f"Risk Score: `{row['final_risk_score']:.4f}`")
                st.write(f"Decision: `{row['decision']}`")
                st.write(f"Actual Fraud: `{'Yes ⚠️' if row['is_fraud'] == 1 else 'No'}`")
                if "amount_to_prior_mean" in df.columns:
                    st.write(f"Amount vs. User Average: `{row['amount_to_prior_mean']:.1f}x`")
                if "merchant_risk" in df.columns:
                    st.write(f"Merchant Risk: `{row['merchant_risk']:.3f}`")

            with col3:
                st.write("**Flags**")
                yes = lambda v: "Yes 🚩" if v == 1 else "No"
                if "new_device_flag" in df.columns:
                    st.write(f"New Device: `{yes(row['new_device_flag'])}`")
                if "device_other_users" in df.columns:
                    st.write(f"Other Users on Device: `{int(row['device_other_users'])}`")
                if "state_changed" in df.columns:
                    st.write(f"State Changed: `{yes(row['state_changed'])}`")
                if "impossible_travel_flag" in df.columns:
                    st.write(f"Impossible Travel: `{yes(row['impossible_travel_flag'])}`")
                if "velocity_risk_flag" in df.columns:
                    st.write(f"Velocity Risk: `{yes(row['velocity_risk_flag'])}`")
else:
    st.success("✅ No blocked transactions found")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Decision Distribution")
    order = ["ALLOW", "CHALLENGE", "BLOCK"]
    colors = {"ALLOW": "#2ecc71", "CHALLENGE": "#f39c12", "BLOCK": "#e74c3c"}
    counts = df["decision"].value_counts().reindex(order).fillna(0)
    counts = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(
        counts, labels=None, autopct="%1.1f%%", colors=[colors[k] for k in counts.index],
        startangle=90, pctdistance=0.75, explode=[0.05] * len(counts),
    )
    ax.legend(wedges, [f"{k} ({int(v):,})" for k, v in counts.items()], title="Decision",
              loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10)
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Fraud Score Distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = [i / 50 for i in range(51)]
    ax.hist(df.loc[df["is_fraud"] == 0, "final_risk_score"], bins=bins, alpha=0.6, color="#3498db", label="Legit")
    ax.hist(df.loc[df["is_fraud"] == 1, "final_risk_score"], bins=bins, alpha=0.6, color="#e74c3c", label="Fraud")
    ax.axvline(T["challenge"], color="orange", linestyle="--", linewidth=2, label=f"CHALLENGE ({T['challenge']:.2f})")
    ax.axvline(T["block"], color="red", linestyle="--", linewidth=2, label=f"BLOCK ({T['block']:.2f})")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count (log scale)")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Report images ─────────────────────────────────────────────────────────────
if os.path.exists(REPORTS_PATH):
    st.subheader("Model Evaluation Reports")
    col1, col2, col3 = st.columns(3)
    images = [
        ("Confusion Matrix", "confusion_matrix.png", col1),
        ("ROC Curve", "roc_curve.png", col2),
        ("Precision-Recall", "precision_recall.png", col3),
        ("Feature Importance", "feature_importance.png", col1),
        ("Score Distribution", "score_distribution.png", col2),
        ("Decision Distribution", "decision_distribution.png", col3),
    ]
    for title, fname, col in images:
        path = f"{REPORTS_PATH}/{fname}"
        if os.path.exists(path):
            with col:
                st.image(path, caption=title)

st.divider()

# ── Data table ────────────────────────────────────────────────────────────────
st.subheader("Transaction Data")
col1, _ = st.columns([1, 3])
with col1:
    decision_filter = st.selectbox("Filter by Decision", ["ALL", "BLOCK", "CHALLENGE", "ALLOW"])

filtered = df if decision_filter == "ALL" else df[df["decision"] == decision_filter]
filtered = filtered.sort_values("final_risk_score", ascending=False)

_, col2 = st.columns([3, 1])
with col2:
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"transactions_{decision_filter.lower()}.csv",
        mime="text/csv",
    )

show_cols = [c for c in ["user_id", "timestamp", "city", "amount", "payment_mode",
                         "final_risk_score", "decision", "is_fraud"] if c in filtered.columns]
st.dataframe(filtered[show_cols].head(500))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Fraud Detection System | XGBoost with causal behavioural features "
           "(Isolation Forest evaluated as an ablation)")