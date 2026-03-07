import math
import pandas as pd
import numpy as np

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


def add_lat_long(df):
    df = df.copy()
    df["lat"] = df["state"].map(lambda x: STATE_COORDS.get(x, (0, 0))[0])
    df["lon"] = df["state"].map(lambda x: STATE_COORDS.get(x, (0, 0))[1])
    return df


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_state_change(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    df["prev_state"] = df.groupby("user_id")["state"].shift(1)
    df["state_changed"] = (df["state"] != df["prev_state"]).astype(int).fillna(0)
    return df


def detect_new_device(df):
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"])
    df["prev_device"] = df.groupby("user_id")["device_id"].shift(1)
    df["new_device_flag"] = (df["device_id"] != df["prev_device"]).astype(int).fillna(0)
    return df


def detect_os_change(df):
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"])
    df["prev_os"] = df.groupby("user_id")["os_version"].shift(1)
    df["os_changed_flag"] = (df["os_version"] != df["prev_os"]).astype(int).fillna(0)
    return df


def device_location_combo_risk(df):
    df = df.copy()
    df["device_location_risk"] = (
        (df["new_device_flag"] == 1) & (df["state_changed"] == 1)
    ).astype(int)
    return df


def detect_impossible_travel(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    df["prev_lat"]  = df.groupby("user_id")["lat"].shift(1)
    df["prev_lon"]  = df.groupby("user_id")["lon"].shift(1)
    df["prev_time"] = df.groupby("user_id")["timestamp"].shift(1)

    dist_list, speed_list = [], []
    for _, r in df.iterrows():
        if pd.isna(r["prev_lat"]) or pd.isna(r["prev_time"]):
            dist_list.append(0)
            speed_list.append(0)
            continue
        dist  = haversine(r["lat"], r["lon"], r["prev_lat"], r["prev_lon"])
        tdiff = (r["timestamp"] - r["prev_time"]).total_seconds() / 3600
        speed = dist / tdiff if tdiff > 0 else 0
        dist_list.append(dist)
        speed_list.append(speed)

    df["travel_distance_km"]     = dist_list
    df["travel_speed_kmph"]      = speed_list
    df["impossible_travel_flag"] = (df["travel_speed_kmph"] > 500).astype(int)
    return df


def velocity_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    df.set_index("timestamp", inplace=True)
    df["txn_count_1hr"]  = (df.groupby("user_id")["amount"]
                              .rolling("1h").count()
                              .reset_index(level=0, drop=True))
    df["amount_sum_1hr"] = (df.groupby("user_id")["amount"]
                              .rolling("1h").sum()
                              .reset_index(level=0, drop=True))
    df["velocity_risk_flag"] = (
        (df["txn_count_1hr"] >= 5) | (df["amount_sum_1hr"] > 20000)
    ).astype(int)
    df.reset_index(inplace=True)
    return df


def add_user_profile_features(df):
    df = df.copy()
    user_stats = df.groupby("user_id")["amount"].agg(
        user_mean_amount="mean",
        user_std_amount="std",
        user_txn_count="count",
        user_max_amount="max"
    ).reset_index()
    df = df.merge(user_stats, on="user_id", how="left")
    df["amount_zscore"] = (
        (df["amount"] - df["user_mean_amount"]) /
        (df["user_std_amount"] + 1)
    )
    df["is_high_volume_user"] = (
        df["user_txn_count"] > df["user_txn_count"].quantile(0.9)
    ).astype(int)
    return df


def add_merchant_risk(df):
    df = df.copy()
    merchant_fraud_rate = df.groupby("merchant_id")["is_fraud"].mean()
    df["merchant_fraud_rate"] = df["merchant_id"].map(merchant_fraud_rate)
    df["high_risk_merchant"]  = (df["merchant_fraud_rate"] > 0.1).astype(int)
    return df


def add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    first_txn = df.groupby("user_id")["timestamp"].transform("min")
    df["account_age_days"] = (df["timestamp"] - first_txn).dt.days
    return df


def feature_engineering(df):
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["dayofweek"]   = df["timestamp"].dt.dayofweek
    df["is_night"]    = (df["hour"] < 6).astype(int)
    df["is_weekend"]  = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df["amount_log"]  = np.log1p(df["amount"])
    df["amount_to_avg_ratio"] = df["amount"] / (
        df.groupby("user_id")["amount"].transform("mean") + 1
    )
    first_txn = df.groupby("user_id")["timestamp"].transform("min")
    df["account_age_days"] = (df["timestamp"] - first_txn).dt.days
    df["region"] = df["state"].map(REGION_MAP).fillna("Other")
    df = pd.get_dummies(
        df,
        columns=["region", "payment_mode", "device_type", "os_version", "app_version"],
        drop_first=True
    )
    return df


HELPER_COLS = ["prev_state", "prev_device", "prev_os", "prev_lat", "prev_lon", "prev_time"]


def drop_helper_columns(df):
    df = df.copy()
    for col in HELPER_COLS:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"  Dropped: {col}")
    return df


def clean_nans(df):
    before = df.isna().sum().sum()
    df = df.fillna(0)
    print(f"  NaN before: {before}  after: {df.isna().sum().sum()}")
    return df


def encode_objects(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df