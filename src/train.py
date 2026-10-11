import json, os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from src.features import BASE_FEATURES, apply_merchant_risk, build_features, merchant_table, oof_merchant_risk

SEED = 42
XGB_PARAMS = dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="aucpr", early_stopping_rounds=40, tree_method="hist", random_state=SEED, n_jobs=-1)


def pick_thresholds(y, p, target_precision=0.90, challenge_fpr=0.01):
    """Choose tier thresholds on VALIDATION data only."""
    fpr, tpr, thr = roc_curve(y, p)
    t_challenge = float(thr[np.searchsorted(fpr, challenge_fpr, side="right") - 1])
    prec, rec, pthr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target_precision)[0]
    t_block = float(pthr[ok[0]]) if len(ok) else float(pthr[-1])
    return {"challenge": min(t_challenge, t_block), "block": t_block}


def train(data_path="data/train.csv", model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)
    df = build_features(pd.read_csv(data_path))                       # time-sorted, causal
    cut = int(len(df) * 0.8)                                          # time-based split
    fit, val = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"fit {len(fit):,} ({fit.timestamp.min():%Y-%m-%d}..{fit.timestamp.max():%Y-%m-%d}) | "
          f"val {len(val):,} ({val.timestamp.min():%Y-%m-%d}..{val.timestamp.max():%Y-%m-%d}) | fraud rate {fit.is_fraud.mean():.2%}")

    # merchant risk: out-of-fold on fit rows, fit-table on val rows (no row sees its own label)
    fit["merchant_risk"] = oof_merchant_risk(fit)
    table, prior = merchant_table(fit)
    val["merchant_risk"] = apply_merchant_risk(val, table, prior)

    # Isolation Forest (unsupervised) -> anomaly score used as an extra feature
    iso = IsolationForest(n_estimators=150, random_state=SEED, n_jobs=-1).fit(fit[BASE_FEATURES])
    fit["iso_score"] = -iso.score_samples(fit[BASE_FEATURES])
    val["iso_score"] = -iso.score_samples(val[BASE_FEATURES])

    results, models = {}, {}
    for name, cols in {"xgb_only": BASE_FEATURES, "xgb_plus_iso": BASE_FEATURES + ["iso_score"]}.items():
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(fit[cols], fit["is_fraud"], eval_set=[(val[cols], val["is_fraud"])], verbose=False)
        p = m.predict_proba(val[cols])[:, 1]
        results[name] = {"val_pr_auc": float(average_precision_score(val.is_fraud, p)),
                         "val_roc_auc": float(roc_auc_score(val.is_fraud, p))}
        models[name] = (m, cols, p)
        print(f"{name:13s} val PR-AUC {results[name]['val_pr_auc']:.4f}  ROC-AUC {results[name]['val_roc_auc']:.4f}")

    best = max(results, key=lambda k: results[k]["val_pr_auc"])
    model, cols, pval = models[best]
    print(f"selected: {best}")

    # leak check: a regularised model trained on shuffled labels must score ~0.5 AUC (mean of 3 seeds)
    aucs = []
    for sd in range(3):
        ys = np.random.default_rng(sd).permutation(fit["is_fraud"].to_numpy())
        chk = xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, min_child_weight=20,
                                tree_method="hist", n_jobs=-1, random_state=sd)
        chk.fit(fit[cols], ys)
        aucs.append(roc_auc_score(val["is_fraud"], chk.predict_proba(val[cols])[:, 1]))
    leak_auc = float(np.mean(aucs))
    print(f"shuffled-label sanity check: val ROC-AUC {leak_auc:.3f} (mean of 3 seeds; should be ~0.5)")

    thresholds = pick_thresholds(val.is_fraud.to_numpy(), pval)
    print(f"thresholds (from validation): {thresholds}")

    joblib.dump(model, f"{model_dir}/xgb_model.pkl")
    joblib.dump(iso, f"{model_dir}/iso_model.pkl")
    joblib.dump({"table": table, "prior": prior}, f"{model_dir}/merchant_risk.pkl")
    json.dump({"features": cols, "use_iso": best == "xgb_plus_iso", "thresholds": thresholds,
               "validation": results, "shuffled_label_auc": leak_auc},
              open(f"{model_dir}/config.json", "w"), indent=2)
    json.dump(thresholds, open(f"{model_dir}/thresholds.json", "w"))
    print("saved models + config")


if __name__ == "__main__":
    train()