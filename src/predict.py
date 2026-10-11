import json
import joblib
import numpy as np
import pandas as pd
from src.features import BASE_FEATURES, apply_merchant_risk, build_features


def decide(score, t):
    return np.where(score >= t["block"], "BLOCK", np.where(score >= t["challenge"], "CHALLENGE", "ALLOW"))


def predict(data_path="data/production.csv", model_dir="model", output_path="data/final_output.csv",
            history_path="data/train.csv"):
    cfg = json.load(open(f"{model_dir}/config.json"))
    model, iso = joblib.load(f"{model_dir}/xgb_model.pkl"), joblib.load(f"{model_dir}/iso_model.pkl")
    mr = joblib.load(f"{model_dir}/merchant_risk.pkl")

    hist, prod = pd.read_csv(history_path), pd.read_csv(data_path)
    n_prod = len(prod)
    # features are causal, so computing on history+production gives production rows their real past
    allf = build_features(pd.concat([hist, prod], ignore_index=True))
    cutoff = pd.to_datetime(prod["timestamp"]).min()
    out = allf[allf["timestamp"] >= cutoff].copy()
    assert len(out) == n_prod, "history/production boundary mismatch"

    out["merchant_risk"] = apply_merchant_risk(out, mr["table"], mr["prior"])
    out["iso_score"] = -iso.score_samples(out[BASE_FEATURES])
    out["final_risk_score"] = model.predict_proba(out[cfg["features"]])[:, 1]
    out["decision"] = decide(out["final_risk_score"].to_numpy(), cfg["thresholds"])

    keep = ["user_id", "timestamp", "city", "state", "merchant_id", "device_id", "amount", "payment_mode", "is_fraud",
            "new_device_flag", "state_changed", "impossible_travel_flag", "velocity_risk_flag",
            "device_other_users", "amount_to_prior_mean", "merchant_risk", "final_risk_score", "decision"]
    out = out[keep]
    out.to_csv(output_path, index=False)
    wrong = out[((out.is_fraud == 1) & (out.decision == "ALLOW")) | ((out.is_fraud == 0) & (out.decision == "BLOCK"))]
    wrong.to_csv("data/misclassified.csv", index=False)
    print(out["decision"].value_counts().to_string())
    return out


if __name__ == "__main__":
    predict()