import os
import json
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from src.feature_engineering import (
    add_lat_long, detect_state_change, detect_new_device,
    detect_os_change, device_location_combo_risk,
    add_user_profile_features, add_merchant_risk,
    feature_engineering, drop_helper_columns, clean_nans, encode_objects,
)

TRAIN_DROP_COLS = [
    "travel_distance_km", "travel_speed_kmph", "impossible_travel_flag",
    "txn_count_1hr", "amount_sum_1hr", "velocity_risk_flag",
    "user_mean_amount", "user_std_amount", "user_txn_count", "user_max_amount",
]


def build_train_features(df):
    print("  Adding geo coordinates...")
    df = add_lat_long(df)
    print("  Detecting behavioural flags...")
    df = detect_state_change(df)
    df = detect_new_device(df)
    df = detect_os_change(df)
    df = device_location_combo_risk(df)
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
    X = df.drop(["is_fraud", "timestamp"], axis=1)
    y = df["is_fraud"]
    for col in TRAIN_DROP_COLS:
        if col in X.columns:
            X.drop(col, axis=1, inplace=True)
    X = encode_objects(X)
    return X, y


def train(data_path="data/train.csv", model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")

    print("Building features...")
    X, y = build_train_features(df)
    print(f"Features: {len(X.columns)} columns")
    print(f"Fraud rate: {y.mean():.2%}")

    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    print("Training Isolation Forest...")
    iso_model = IsolationForest(
        n_estimators=150,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_train)

    print("Running cross validation...")
    cv_scores = cross_val_score(
        xgb_model, X_res, y_res,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )
    print(f"CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("Saving models...")
    joblib.dump(xgb_model, f"{model_dir}/xgb_model.pkl")
    joblib.dump(iso_model,  f"{model_dir}/iso_model.pkl")
    with open(f"{model_dir}/feature_columns.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)

    print("Models saved successfully.")
    print(f"CV AUC Score: {cv_scores.mean():.4f}")


if __name__ == "__main__":
    train()