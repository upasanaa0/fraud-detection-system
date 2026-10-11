"""Leak-free, causal feature engineering.

Every behavioural feature for a transaction uses only that user's / device's
history up to and including that transaction. No label is ever read here.
Merchant risk is the only label-derived feature; it is built from TRAIN labels
only (see merchant_encoding) and applied to later data via a saved table.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

LATLON = {  # city -> (lat, lon)
    "Mumbai": (19.08, 72.88), "Delhi": (28.70, 77.10), "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.38, 78.48), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36),
    "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.78),
    "Lucknow": (26.85, 80.95),
}
MODES = ["UPI", "RuPay", "NetBanking", "Wallet"]
EPOCH = pd.Timestamp("2026-01-01")

BASE_FEATURES = [
    "amount_log", "hour", "is_night", "dayofweek", "is_weekend",
    "mode_UPI", "mode_RuPay", "mode_NetBanking", "mode_Wallet",
    "user_txn_index", "new_device_flag", "device_other_users",
    "state_changed", "away_from_home", "travel_dist_km", "travel_speed_kmph", "impossible_travel_flag",
    "txn_count_1h", "amount_sum_1h", "velocity_risk_flag",
    "amount_to_prior_mean", "amount_to_prior_max", "merchant_risk",
]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df):
    """Return df (sorted by time) with causal behavioural features. merchant_risk is left at 0."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts"] = (df["timestamp"] - EPOCH).dt.total_seconds()
    df = df.sort_values(["user_id", "ts"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("user_id", sort=False)

    df["lat"] = df["city"].map(lambda c: LATLON[c][0]); df["lon"] = df["city"].map(lambda c: LATLON[c][1])
    df["user_txn_index"] = g.cumcount()
    first = df["user_txn_index"] == 0

    # location / travel
    gap_h = (df["ts"] - g["ts"].shift(1)) / 3600
    dist = haversine_km(df["lat"], df["lon"], g["lat"].shift(1), g["lon"].shift(1))
    speed = dist / gap_h.clip(lower=1 / 60)
    df["travel_dist_km"] = dist.fillna(0)
    df["travel_speed_kmph"] = speed.clip(upper=3000).fillna(0)
    df["impossible_travel_flag"] = ((speed > 500) & (dist > 100)).astype(int)
    df["state_changed"] = ((df["state"] != g["state"].shift(1)) & ~first).astype(int)
    df["away_from_home"] = ((df["state"] != g["state"].transform("first")) & ~first).astype(int)

    # device
    df["new_device_flag"] = ((df.groupby(["user_id", "device_id"], sort=False).cumcount() == 0) & ~first).astype(int)
    pair = df.groupby(["device_id", "user_id"], sort=False)["ts"].min().reset_index().sort_values(["ts"])
    pair["cum"] = pair.groupby("device_id").cumcount() + 1
    left = df[["device_id", "ts"]].reset_index().sort_values("ts")
    m = pd.merge_asof(left, pair[["device_id", "ts", "cum"]], on="ts", by="device_id", direction="backward")
    df["device_other_users"] = (m.set_index("index")["cum"].reindex(df.index).fillna(1) - 1).values

    # velocity (1-hour window per user, vectorised)
    key = g.ngroup().to_numpy().astype("float64") * 1e9 + df["ts"].to_numpy()
    start = np.searchsorted(key, key - 3600, side="left")
    idx = np.arange(len(df))
    pref = np.concatenate([[0.0], np.cumsum(df["amount"].to_numpy())])
    df["txn_count_1h"] = idx - start + 1
    df["amount_sum_1h"] = pref[idx + 1] - pref[start]
    df["velocity_risk_flag"] = ((df["txn_count_1h"] >= 5) | (df["amount_sum_1h"] > 20000)).astype(int)

    # amount vs the user's own past
    prior_sum = g["amount"].cumsum() - df["amount"]
    prior_mean = prior_sum / df["user_txn_index"].replace(0, np.nan)
    prior_max = g["amount"].cummax().groupby(df["user_id"]).shift(1)
    df["amount_to_prior_mean"] = (df["amount"] / prior_mean).fillna(1.0).clip(upper=100)
    df["amount_to_prior_max"] = (df["amount"] / prior_max).fillna(1.0).clip(upper=100)

    # time / amount / mode
    df["amount_log"] = np.log1p(df["amount"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_night"] = (df["hour"] < 5).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    for mname in MODES:
        df[f"mode_{mname}"] = (df["payment_mode"] == mname).astype(int)

    df["merchant_risk"] = 0.0
    return df.sort_values(["ts", "user_id"], kind="mergesort").reset_index(drop=True)


# ---- merchant target encoding (the only label-derived feature) -------------
def merchant_table(df_labeled, smoothing=20):
    prior = float(df_labeled["is_fraud"].mean())
    agg = df_labeled.groupby("merchant_id")["is_fraud"].agg(["sum", "count"])
    rate = (agg["sum"] + smoothing * prior) / (agg["count"] + smoothing)
    return rate.to_dict(), prior


def apply_merchant_risk(df, table, prior):
    return df["merchant_id"].map(table).fillna(prior).to_numpy()


def oof_merchant_risk(df_labeled, n_splits=5, smoothing=20, seed=42):
    """Out-of-fold encoding for training rows, so a row never sees its own label."""
    out = np.zeros(len(df_labeled))
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(df_labeled):
        tab, prior = merchant_table(df_labeled.iloc[tr], smoothing)
        out[va] = apply_merchant_risk(df_labeled.iloc[va], tab, prior)
    return out