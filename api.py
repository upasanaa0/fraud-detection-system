import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import BASE_FEATURES

CFG = json.load(open("model/config.json"))
MODEL = joblib.load("model/xgb_model.pkl")
ISO = joblib.load("model/iso_model.pkl")
MR = joblib.load("model/merchant_risk.pkl")
T = CFG["thresholds"]

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    merchant_id: int
    payment_mode: str = Field(pattern="^(UPI|RuPay|NetBanking|Wallet)$")
    hour: int = Field(ge=0, le=23)
    dayofweek: int = Field(ge=0, le=6)
    # history-derived context (from a feature store)
    user_txn_index: int = 0
    new_device_flag: int = 0
    device_other_users: int = 0
    state_changed: int = 0
    away_from_home: int = 0
    travel_dist_km: float = 0.0
    travel_speed_kmph: float = 0.0
    txn_count_1h: int = 1
    amount_sum_1h: float | None = None
    amount_to_prior_mean: float = 1.0
    amount_to_prior_max: float = 1.0


def build_row(t: Transaction) -> pd.DataFrame:
    r = {
        "amount_log": np.log1p(t.amount), "hour": t.hour, "is_night": int(t.hour < 5),
        "dayofweek": t.dayofweek, "is_weekend": int(t.dayofweek >= 5),
        "user_txn_index": t.user_txn_index, "new_device_flag": t.new_device_flag,
        "device_other_users": t.device_other_users, "state_changed": t.state_changed,
        "away_from_home": t.away_from_home, "travel_dist_km": t.travel_dist_km,
        "travel_speed_kmph": min(t.travel_speed_kmph, 3000),
        "impossible_travel_flag": int(t.travel_speed_kmph > 500 and t.travel_dist_km > 100),
        "txn_count_1h": t.txn_count_1h,
        "amount_sum_1h": t.amount if t.amount_sum_1h is None else t.amount_sum_1h,
        "amount_to_prior_mean": min(t.amount_to_prior_mean, 100),
        "amount_to_prior_max": min(t.amount_to_prior_max, 100),
        "merchant_risk": MR["table"].get(t.merchant_id, MR["prior"]),
    }
    for m in ["UPI", "RuPay", "NetBanking", "Wallet"]:
        r[f"mode_{m}"] = int(t.payment_mode == m)
    r["velocity_risk_flag"] = int(r["txn_count_1h"] >= 5 or r["amount_sum_1h"] > 20000)
    X = pd.DataFrame([r])[BASE_FEATURES]
    if CFG["use_iso"]:
        X["iso_score"] = -ISO.score_samples(X[BASE_FEATURES])
    return X[CFG["features"]]


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgb_plus_iso" if CFG["use_iso"] else "xgb_only", "thresholds": T}


@app.post("/predict")
def predict(t: Transaction):
    score = float(MODEL.predict_proba(build_row(t))[0, 1])
    decision = "BLOCK" if score >= T["block"] else "CHALLENGE" if score >= T["challenge"] else "ALLOW"
    return {"risk_score": round(score, 6), "decision": decision}