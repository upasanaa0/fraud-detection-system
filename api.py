from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json

app = FastAPI(title="Fraud Detection API", version="1.0")

xgb_model = joblib.load("model/xgb_model.pkl")
iso_model  = joblib.load("model/iso_model.pkl")
with open("model/feature_columns.json") as f:
    FEATURE_COLS = json.load(f)

REGION_MAP = {
    "Tamil Nadu": "South", "Kerala": "South", "Karnataka": "South",
    "Telangana": "South", "Andhra Pradesh": "South",
    "Maharashtra": "West", "Gujarat": "West", "Goa": "West", "Rajasthan": "West",
    "Delhi": "North", "Punjab": "North", "Haryana": "North",
    "Uttar Pradesh": "North", "Uttarakhand": "North",
    "Jammu and Kashmir": "North", "Ladakh": "North",
    "West Bengal": "East", "Odisha": "East", "Bihar": "East", "Jharkhand": "East",
    "Madhya Pradesh": "Central", "Chhattisgarh": "Central",
    "Assam": "NorthEast", "Manipur": "NorthEast", "Meghalaya": "NorthEast",
    "Mizoram": "NorthEast", "Nagaland": "NorthEast", "Tripura": "NorthEast",
    "Arunachal Pradesh": "NorthEast", "Sikkim": "NorthEast",
}

STATE_COORDS = {
    "Tamil Nadu": (13.08, 80.27), "Kerala": (8.52, 76.93),
    "Karnataka": (12.97, 77.59), "Maharashtra": (19.07, 72.87),
    "Delhi": (28.70, 77.10), "West Bengal": (22.57, 88.36),
    "Gujarat": (23.02, 72.57), "Rajasthan": (26.91, 75.78),
    "Uttar Pradesh": (26.84, 80.94), "Punjab": (30.73, 76.77),
    "Telangana": (17.38, 78.48), "Andhra Pradesh": (16.50, 80.64),
    "Madhya Pradesh": (23.25, 77.41), "Bihar": (25.59, 85.13),
    "Odisha": (20.29, 85.82), "Assam": (26.14, 91.73),
    "Jharkhand": (23.34, 85.30), "Chhattisgarh": (21.25, 81.62),
    "Haryana": (29.05, 76.08), "Uttarakhand": (30.06, 79.01),
    "Himachal Pradesh": (31.10, 77.17), "Goa": (15.29, 74.12),
    "Tripura": (23.94, 91.98), "Manipur": (24.81, 93.93),
    "Meghalaya": (25.57, 91.89), "Nagaland": (25.67, 94.10),
    "Mizoram": (23.16, 92.93), "Sikkim": (27.53, 88.51),
    "Arunachal Pradesh": (27.08, 93.60),
    "Jammu and Kashmir": (34.08, 74.79), "Ladakh": (34.15, 77.57),
}


class Transaction(BaseModel):
    user_id: int
    amount: float
    merchant_id: int
    state: str
    payment_mode: str
    device_id: int
    device_type: str
    os_version: str
    app_version: str
    hour: int = 12
    dayofweek: int = 0
    is_weekend: int = 0
    account_age_days: int = 30
    state_changed: int = 0
    new_device_flag: int = 0
    os_changed_flag: int = 0
    device_location_risk: int = 0
    amount_zscore: float = 0.0
    is_high_volume_user: int = 0
    merchant_fraud_rate: float = 0.0
    high_risk_merchant: int = 0


def build_features(t: Transaction) -> pd.DataFrame:
    lat, lon = STATE_COORDS.get(t.state, (0, 0))
    region   = REGION_MAP.get(t.state, "Other")

    row = {
        "user_id": t.user_id,
        "amount": t.amount,
        "merchant_id": t.merchant_id,
        "device_id": t.device_id,
        "lat": lat,
        "lon": lon,
        "state_changed": t.state_changed,
        "new_device_flag": t.new_device_flag,
        "os_changed_flag": t.os_changed_flag,
        "device_location_risk": t.device_location_risk,
        "hour": t.hour,
        "dayofweek": t.dayofweek,
        "is_night": 1 if t.hour < 6 else 0,
        "is_weekend": t.is_weekend,
        "amount_log": np.log1p(t.amount),
        "amount_to_avg_ratio": 1.0,
        "account_age_days": t.account_age_days,
        "amount_zscore": t.amount_zscore,
        "is_high_volume_user": t.is_high_volume_user,
        "merchant_fraud_rate": t.merchant_fraud_rate,
        "high_risk_merchant": t.high_risk_merchant,
    }

    for r in ["East", "North", "NorthEast", "Other", "South", "West"]:
        row[f"region_{r}"] = 1 if region == r else 0
    for m in ["RuPay", "UPI", "Wallet"]:
        row[f"payment_mode_{m}"] = 1 if t.payment_mode == m else 0
    row["device_type_iOS"] = 1 if t.device_type == "iOS" else 0
    for v in ["11", "12", "13", "14"]:
        row[f"os_version_{v}"] = 1 if t.os_version == v else 0
    for v in ["1.1", "1.2", "2.0", "2.1"]:
        row[f"app_version_{v}"] = 1 if t.app_version == v else 0

    df = pd.DataFrame([row])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]
    return df


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost + IsolationForest"}


@app.post("/predict")
def predict(transaction: Transaction):
    X = build_features(transaction)

    xgb_prob  = float(xgb_model.predict_proba(X)[0][1])
    iso_score = float(-iso_model.decision_function(X)[0])
    s = round(0.7 * xgb_prob + 0.3 * iso_score, 6)

    if transaction.state_changed == 1:          s += 0.05
    if transaction.new_device_flag == 1:         s += 0.10
    if transaction.device_location_risk == 1:    s += 0.15
    s = min(s, 1.0)

    if s >= 0.8:
        decision = "BLOCK"
    elif s >= 0.6:
        decision = "CHALLENGE"
    else:
        decision = "ALLOW"

    return {
        "user_id": transaction.user_id,
        "amount": transaction.amount,
        "fraud_score": round(s, 6),
        "decision": decision,
        "breakdown": {
            "xgb_probability": round(xgb_prob, 4),
            "isolation_score": round(iso_score, 4),
            "rule_boosts_applied": {
                "state_changed": transaction.state_changed == 1,
                "new_device": transaction.new_device_flag == 1,
                "device_location_risk": transaction.device_location_risk == 1,
            }
        }
    }