import json, os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)


def evaluate(model_dir="model", data_dir="data", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    cfg = json.load(open(f"{model_dir}/config.json"))
    df = pd.read_csv(f"{data_dir}/final_output.csv")
    y, s = df["is_fraud"].to_numpy(), df["final_risk_score"].to_numpy()
    flagged = df["decision"].isin(["CHALLENGE", "BLOCK"]).to_numpy()
    blocked = (df["decision"] == "BLOCK").to_numpy()

    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    r_at_1 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])
    r_at_01 = float(tpr[np.searchsorted(fpr, 0.001, side="right") - 1])
    m = {
        "test_rows": int(len(df)), "test_fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
        "pr_auc_random_baseline": float(y.mean()),
        "recall_at_1pct_fpr": r_at_1, "recall_at_0.1pct_fpr": r_at_01,
        "flagged_precision": float(y[flagged].mean()), "flagged_recall": float(y[flagged].sum() / y.sum()),
        "block_precision": float(y[blocked].mean()) if blocked.any() else None,
        "block_recall": float(y[blocked].sum() / y.sum()),
        "false_block_rate_of_legit": float(((y == 0) & blocked).sum() / (y == 0).sum()),
        "selected_model": "xgb_plus_iso" if cfg["use_iso"] else "xgb_only",
        "ablation_validation_pr_auc": {k: v["val_pr_auc"] for k, v in cfg["validation"].items()},
        "shuffled_label_auc": cfg["shuffled_label_auc"],
        "thresholds": cfg["thresholds"],
        "single_feature_auc_max": None,
    }
    feats = ["amount_to_prior_mean", "device_other_users", "away_from_home", "txn_count_1h", "travel_speed_kmph",
             "new_device_flag", "merchant_risk", "amount_sum_1h"]
    fa = {f: float(roc_auc_score(y, df[f])) for f in feats if f in df}
    m["single_feature_auc"], m["single_feature_auc_max"] = fa, max(fa.values())
    json.dump(m, open(f"{output_dir}/metrics.json", "w"), indent=2)

    tiers = df.groupby("decision").agg(txns=("is_fraud", "size"), fraud=("is_fraud", "sum"))
    tiers["fraud_rate"] = tiers["fraud"] / tiers["txns"]
    print(f"\nTEST SET (unseen later period): {len(df):,} txns, fraud {y.mean():.2%}")
    print(f"ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} (random = {y.mean():.4f})")
    print(f"Recall @1% FPR {r_at_1:.3f} | @0.1% FPR {r_at_01:.3f}")
    print(f"CHALLENGE+BLOCK: precision {m['flagged_precision']:.3f}, recall {m['flagged_recall']:.3f}")
    print(f"BLOCK only     : precision {m['block_precision']}, recall {m['block_recall']:.3f}, "
          f"false-block rate {m['false_block_rate_of_legit']:.4%}")
    print(f"Ablation (val PR-AUC): {m['ablation_validation_pr_auc']}")
    print(f"Shuffled-label AUC {m['shuffled_label_auc']:.3f} | max single-feature AUC {m['single_feature_auc_max']:.3f}")
    print(tiers.to_string())

    # ---- charts (same filenames the dashboard already loads) ----
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y, flagged), annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set(xlabel="Predicted (CHALLENGE/BLOCK = fraud)", ylabel="Actual", title="Confusion Matrix")
    fig.tight_layout(); fig.savefig(f"{output_dir}/confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{output_dir}/roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {m['pr_auc']:.3f}"); ax.axhline(y.mean(), color="k", ls="--", label="random")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{output_dir}/precision_recall.png", dpi=150); plt.close(fig)

    model = joblib.load(f"{model_dir}/xgb_model.pkl")
    imp = pd.Series(model.feature_importances_, index=cfg["features"]).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6)); imp.plot.barh(ax=ax, color="#e74c3c")
    ax.set_title("Feature Importance (top 15)"); fig.tight_layout()
    fig.savefig(f"{output_dir}/feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(s[y == 0], bins=bins, alpha=0.6, label="Legit", log=True); ax.hist(s[y == 1], bins=bins, alpha=0.6, label="Fraud", log=True)
    for name, c in (("challenge", "orange"), ("block", "red")):
        ax.axvline(cfg["thresholds"][name], color=c, ls="--", label=f"{name.upper()} ({cfg['thresholds'][name]:.2f})")
    ax.set(xlabel="Risk score", ylabel="Count (log)", title="Score Distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{output_dir}/score_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    df["decision"].value_counts().reindex(["ALLOW", "CHALLENGE", "BLOCK"]).fillna(0).plot.bar(ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_title("Decision Distribution"); ax.set_yscale("log"); fig.tight_layout()
    fig.savefig(f"{output_dir}/decision_distribution.png", dpi=150); plt.close(fig)
    return m


if __name__ == "__main__":
    evaluate()