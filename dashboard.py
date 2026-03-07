import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection Dashboard")

OUTPUT_PATH  = "data/final_output.csv"
REPORTS_PATH = "reports"

if not os.path.exists(OUTPUT_PATH):
    st.error("Run the pipeline first: python main.py")
    st.stop()

df = pd.read_csv(OUTPUT_PATH)

# ── KPI cards ─────────────────────────────────────────────────────────────────
total      = len(df)
blocked    = len(df[df["decision"] == "BLOCK"])
challenged = len(df[df["decision"] == "CHALLENGE"])
allowed    = len(df[df["decision"] == "ALLOW"])
fraud_rate = df["is_fraud"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{total:,}")
col2.metric("Fraud Rate",         f"{fraud_rate:.2f}%")
col3.metric("BLOCKED",            f"{blocked:,}",    delta=f"{blocked/total*100:.1f}%")
col4.metric("CHALLENGED",         f"{challenged:,}", delta=f"{challenged/total*100:.1f}%")
col5.metric("ALLOWED",            f"{allowed:,}",    delta=f"{allowed/total*100:.1f}%")

st.divider()

# ── High Risk Alerts ──────────────────────────────────────────────────────────
st.subheader("🚨 High Risk Alerts")

high_risk = df[df["final_risk_score"] >= 0.8].sort_values(
    "final_risk_score", ascending=False
).head(10)

if len(high_risk) > 0:
    csv = high_risk.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download All High Risk Transactions",
        data=csv,
        file_name="high_risk_transactions.csv",
        mime="text/csv"
    )

    st.write(f"**{len(high_risk)} high risk transactions found:**")

    for i, (_, row) in enumerate(high_risk.iterrows()):
        with st.expander(
            f"🚫 BLOCK | User: {int(row['user_id'])} | "
            f"₹{row['amount']:,.2f} | "
            f"Score: {row['final_risk_score']:.4f}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Transaction Info**")
                st.write(f"User ID: `{int(row['user_id'])}`")
                st.write(f"Amount: `₹{row['amount']:,.2f}`")
                st.write(f"Merchant ID: `{int(row['merchant_id'])}`")

            with col2:
                st.write("**Risk Signals**")
                st.write(f"Risk Score: `{row['final_risk_score']:.4f}`")
                st.write(f"Decision: `{row['decision']}`")
                st.write(f"Actual Fraud: `{'Yes ⚠️' if row['is_fraud']==1 else 'No'}`")

            with col3:
                st.write("**Flags**")
                if "new_device_flag" in df.columns:
                    st.write(f"New Device: `{'Yes 🚩' if row['new_device_flag']==1 else 'No'}`")
                if "state_changed" in df.columns:
                    st.write(f"State Changed: `{'Yes 🚩' if row['state_changed']==1 else 'No'}`")
                if "device_location_risk" in df.columns:
                    st.write(f"Device+Location Risk: `{'Yes 🚩' if row['device_location_risk']==1 else 'No'}`")
                if "impossible_travel_flag" in df.columns:
                    st.write(f"Impossible Travel: `{'Yes 🚩' if row['impossible_travel_flag']==1 else 'No'}`")
                if "velocity_risk_flag" in df.columns:
                    st.write(f"Velocity Risk: `{'Yes 🚩' if row['velocity_risk_flag']==1 else 'No'}`")
else:
    st.success("✅ No high risk transactions found")

st.divider()

# ── Charts row 1 ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Decision Distribution")
    counts = df["decision"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct="%1.1f%%",
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        startangle=90,
        pctdistance=0.75,
        explode=[0.05] * len(counts)
    )
    ax.legend(
        wedges,
        [f"{label} ({val:,})" for label, val in zip(counts.index, counts.values)],
        title="Decision",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=10
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Fraud Score Distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    fraud_df     = df[df["is_fraud"] == 1]["final_risk_score"]
    not_fraud_df = df[df["is_fraud"] == 0]["final_risk_score"]
    ax.hist(not_fraud_df, bins=50, alpha=0.6, color="#3498db", label="Legit")
    ax.hist(fraud_df,     bins=50, alpha=0.6, color="#e74c3c", label="Fraud")
    ax.axvline(0.6, color="orange", linestyle="--", linewidth=2, label="CHALLENGE (0.6)")
    ax.axvline(0.8, color="red",    linestyle="--", linewidth=2, label="BLOCK (0.8)")
    ax.legend()
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Report images ─────────────────────────────────────────────────────────────
if os.path.exists(REPORTS_PATH):
    st.subheader("Model Evaluation Reports")
    col1, col2, col3 = st.columns(3)

    images = [
        ("Confusion Matrix",      "confusion_matrix.png",      col1),
        ("ROC Curve",             "roc_curve.png",             col2),
        ("Precision-Recall",      "precision_recall.png",      col3),
        ("Feature Importance",    "feature_importance.png",    col1),
        ("Score Distribution",    "score_distribution.png",    col2),
        ("Decision Distribution", "decision_distribution.png", col3),
    ]

    for title, fname, col in images:
        path = f"{REPORTS_PATH}/{fname}"
        if os.path.exists(path):
            with col:
                st.image(path, caption=title, use_container_width=True)

st.divider()

# ── Data table ────────────────────────────────────────────────────────────────
st.subheader("Transaction Data")
col1, col2 = st.columns([1, 3])

with col1:
    decision_filter = st.selectbox(
        "Filter by Decision",
        ["ALL", "BLOCK", "CHALLENGE", "ALLOW"]
    )

filtered = df if decision_filter == "ALL" else df[df["decision"] == decision_filter]

col1, col2 = st.columns([3, 1])
with col2:
    csv_full = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv_full,
        file_name=f"transactions_{decision_filter.lower()}.csv",
        mime="text/csv"
    )

st.dataframe(
    filtered[["user_id", "amount", "final_risk_score", "decision", "is_fraud"]].head(500),
    use_container_width=True
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Fraud Detection System | XGBoost + Isolation Forest Ensemble")