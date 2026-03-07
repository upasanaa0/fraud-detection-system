import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh"
]

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

PAYMENT_MODES = ["UPI", "RuPay", "NetBanking", "Wallet"]
DEVICE_TYPES  = ["Android", "iOS"]
OS_VERSIONS   = ["10", "11", "12", "13", "14"]
APP_VERSIONS  = ["1.0", "1.1", "1.2", "2.0", "2.1"]


def is_fraud_logic(amount, payment_mode, hour):
    score = 0

    if amount > 7000 and payment_mode == "Wallet":
        score += 0.6
    if hour < 5:
        score += 0.2
    if amount > 15000:
        score += 0.3
    if random.random() < 0.02:
        score += 0.8

    return 1 if score >= 0.8 else 0


def generate_data(rows):
    data = []
    for _ in range(rows):
        user_id      = random.randint(1000, 300000)
        amount       = round(np.random.exponential(scale=1800), 2)
        timestamp    = datetime.now() - timedelta(minutes=random.randint(0, 500000))
        merchant_id  = random.randint(200, 8000)
        state        = random.choice(INDIAN_STATES)
        city         = random.choice(INDIAN_CITIES)
        payment_mode = random.choice(PAYMENT_MODES)
        device_id    = random.randint(5000, 100000)
        device_type  = random.choice(DEVICE_TYPES)
        os_version   = random.choice(OS_VERSIONS)
        app_version  = random.choice(APP_VERSIONS)
        hour         = timestamp.hour

        is_fraud = is_fraud_logic(amount, payment_mode, hour)

        data.append([
            user_id, amount, timestamp, merchant_id, state, city,
            payment_mode, device_id, device_type, os_version, app_version, is_fraud
        ])

    return pd.DataFrame(data, columns=[
        "user_id", "amount", "timestamp", "merchant_id", "state", "city",
        "payment_mode", "device_id", "device_type", "os_version", "app_version", "is_fraud"
    ])


def create_datasets(train_rows=500000, prod_rows=100000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating training data ({train_rows} rows)...")
    generate_data(train_rows).to_csv(f"{output_dir}/train.csv", index=False)
    print(f"Generating production data ({prod_rows} rows)...")
    generate_data(prod_rows).to_csv(f"{output_dir}/production.csv", index=False)
    print("Datasets created successfully.")


if __name__ == "__main__":
    create_datasets()