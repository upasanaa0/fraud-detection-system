import json
import os
import pandas as pd
import joblib
from src.feature_engineering import (
    add_lat_long, detect_state_change, detect_new_device,
    detect_os_change, device_location_combo_risk,
    detect_impossible_travel, velocity_features,
    add_user_profile_features, add_merchant_risk,
    feature_engineering, drop_helper_columns, clean_nans, encode_objects,
)

ML_DROP_COLS = [
    "travel_distance_km", "travel_speed_kmph", "impossible_travel_flag",
    "txn_count_1hr", "amount_sum_1hr", "velocity_risk_flag",
    "user_mean_amount", "user_std_amount", "user_txn_count", "user_max_amount",
]


def build_prod_features(df):
    print("  Adding geo coordinates...")
    df = add_lat_long(df)
    print("  Detecting behavioural flags...")
    df = detect_state_change(df)
    df = detect_new_device(df)
    df = detect_os_change(df)
    df = device_location_combo_risk(df)
    df = detect_impossible_travel(df)
    df = velocity_features(df)
    print("  Adding user profile features...")
    df = add_user_profile_features(df)
    print("  Adding merchant risk...")
    df = add_merchant_risk(df)
    print("  Feature engineering...")
    df = feature_engineering(df)
    print("  Dropping helper columns...")
    df = drop_helper_columns(df)
    for col in ["state", "city"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = encode_objects(df)
    df = clean_nans(df)
    return df


def ensemble_score(xgb_model, iso_model, X):
    xgb_prob  = xgb_model.predict_proba(X)[:, 1]
    iso_score = -iso_model.decision_function(X)
    return (0.7 * xgb_prob + 0.3 * iso_score).tolist()


def fraud_risk_engine(df):
    df = df.copy()
    final_scores, decisions = [], []

    for _, r in df.iterrows():
        s = r["fraud_score"]

        if "impossible_travel_flag" in df.columns and r["impossible_travel_flag"] == 1:
            s += 0.20
        if "device_location_risk" in df.columns and r["device_location_risk"] == 1:
            s += 0.15
        if "new_device_flag" in df.columns and r["new_device_flag"] == 1:
            s += 0.10
        if "velocity_risk_flag" in df.columns and r["velocity_risk_flag"] == 1:
            s += 0.15
        if "state_changed" in df.columns and r["state_changed"] == 1:
            s += 0.05

        s = min(s, 1.0)

        if s >= 0.8:
            decision = "BLOCK"
        elif s >= 0.6:
            decision = "CHALLENGE"
        else:
            decision = "ALLOW"

        final_scores.append(round(s, 6))
        decisions.append(decision)

    df["final_risk_score"] = final_scores
    df["decision"]         = decisions
    return df


def save_misclassified(df, output_path="data/misclassified.csv"):
    wrong = df[
        ((df["is_fraud"] == 1) & (df["decision"] == "ALLOW")) |
        ((df["is_fraud"] == 0) & (df["decision"] == "BLOCK"))
    ]
    wrong.to_csv(output_path, index=False)
    print(f"  Missed fraud:  {len(wrong[wrong['is_fraud']==1])}")
    print(f"  False blocks:  {len(wrong[wrong['is_fraud']==0])}")
    print(f"  Saved to: {output_path}")


def predict(data_path="data/production.csv", model_dir="model",
            output_path="data/final_output.csv"):

    print(f"Loading production data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")

    print("Building features...")
    prod_df = build_prod_features(df)

    print("Loading models...")
    xgb_model = joblib.load(f"{model_dir}/xgb_model.pkl")
    iso_model  = joblib.load(f"{model_dir}/iso_model.pkl")

    with open(f"{model_dir}/feature_columns.json") as f:
        feature_cols = json.load(f)

    X_prod = prod_df.drop(["is_fraud", "timestamp"], axis=1, errors="ignore")

    for col in ML_DROP_COLS:
        if col in X_prod.columns:
            X_prod.drop(col, axis=1, inplace=True)

    for col in feature_cols:
        if col not in X_prod.columns:
            X_prod[col] = 0
    X_prod = X_prod[feature_cols]

    print("Scoring with ensemble model...")
    prod_df["fraud_score"] = ensemble_score(xgb_model, iso_model, X_prod)

    print("Applying risk engine...")
    prod_df = fraud_risk_engine(prod_df)

    print("Saving misclassified transactions...")
    save_misclassified(prod_df)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    prod_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    print("\n── Decision Summary ──────────────────")
    print(prod_df["decision"].value_counts().to_string())
    print("──────────────────────────────────────")
    return prod_df


if __name__ == "__main__":
    predict()