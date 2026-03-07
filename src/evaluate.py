import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)


def evaluate(model_dir="model", data_dir="data", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading models and data...")
    xgb_model = joblib.load(f"{model_dir}/xgb_model.pkl")

    with open(f"{model_dir}/feature_columns.json") as f:
        feature_cols = json.load(f)

    df     = pd.read_csv(f"{data_dir}/final_output.csv")
    y_true = df["is_fraud"]
    y_pred = (df["final_risk_score"] >= 0.6).astype(int)
    y_prob = df["final_risk_score"]

    # ── 1. Classification report ─────────────────────────────────────
    print("\n── Classification Report ──────────────────────")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    print(f"AUC-ROC Score:      {auc:.4f}")
    print(f"Average Precision:  {ap:.4f}")

    # ── 2. Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # ── 3. ROC curve ─────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=150)
    plt.close()
    print("Saved: roc_curve.png")

    # ── 4. Precision-Recall curve ────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#9b59b6", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall.png", dpi=150)
    plt.close()
    print("Saved: precision_recall.png")

    # ── 5. Score distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fraud_scores     = df[df["is_fraud"] == 1]["final_risk_score"]
    not_fraud_scores = df[df["is_fraud"] == 0]["final_risk_score"]
    ax.hist(not_fraud_scores, bins=50, alpha=0.6, color="#3498db", label="Legit")
    ax.hist(fraud_scores,     bins=50, alpha=0.6, color="#e74c3c", label="Fraud")
    ax.axvline(0.6, color="orange", linestyle="--", label="CHALLENGE (0.6)")
    ax.axvline(0.8, color="red",    linestyle="--", label="BLOCK (0.8)")
    ax.set_xlabel("Final Risk Score")
    ax.set_ylabel("Count")
    ax.set_title("Fraud Score Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distribution.png", dpi=150)
    plt.close()
    print("Saved: score_distribution.png")

    # ── 6. Feature importance ────────────────────────────────────────
    importance = pd.Series(
        xgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="bar", color="#2ecc71", ax=ax)
    ax.set_title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
    ax.set_ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

    # ── 7. Decision distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = df["decision"].value_counts()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
        explode=[0.05] * len(counts)
    )
    ax.legend(
        wedges,
        [f"{label} ({val:,})" for label, val in zip(counts.index, counts.values)],
        title="Decision",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=11
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
    ax.set_title("Decision Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/decision_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: decision_distribution.png")

    print(f"\nAll reports saved to: {output_dir}/")
    print(f"AUC-ROC: {auc:.4f}  |  Average Precision: {ap:.4f}")


if __name__ == "__main__":
    evaluate()