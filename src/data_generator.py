"""Synthetic Indian payment-transaction generator with *planted, detectable* fraud patterns.

Everything here is simulated. Fraud labels come from explicit attack scenarios
(account takeover, velocity burst, impossible travel, wallet drain, subtle),
NOT from a random coin flip, and users/devices persist so behavioural features
have real history to work with.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_USERS = 30_000
N_MERCHANTS = 6_000
N_DAYS = 120
FRAUD_RATE = 0.02
START = pd.Timestamp("2026-01-01")

CITIES = {  # city: (lat, lon, state)
    "Mumbai": (19.08, 72.88, "Maharashtra"), "Delhi": (28.70, 77.10, "Delhi"),
    "Bangalore": (12.97, 77.59, "Karnataka"), "Hyderabad": (17.38, 78.48, "Telangana"),
    "Chennai": (13.08, 80.27, "Tamil Nadu"), "Kolkata": (22.57, 88.36, "West Bengal"),
    "Pune": (18.52, 73.86, "Maharashtra"), "Ahmedabad": (23.02, 72.57, "Gujarat"),
    "Jaipur": (26.91, 75.78, "Rajasthan"), "Lucknow": (26.85, 80.95, "Uttar Pradesh"),
}
CITY_NAMES = list(CITIES)
LAT = np.array([CITIES[c][0] for c in CITY_NAMES])
LON = np.array([CITIES[c][1] for c in CITY_NAMES])
MODES = np.array(["UPI", "RuPay", "NetBanking", "Wallet"])


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _frame(rng, user, ts, city, device, amount, mode, merchant, fraud):
    return pd.DataFrame({
        "user_id": user, "timestamp": ts,
        "city": np.array(CITY_NAMES)[city],
        "state": [CITIES[CITY_NAMES[c]][2] for c in city],
        "merchant_id": merchant, "device_id": device,
        "amount": np.round(amount, 2), "payment_mode": mode, "is_fraud": fraud,
    })


def generate_stream(total_rows, seed=SEED):
    rng = np.random.default_rng(seed)
    n_fraud = int(total_rows * FRAUD_RATE)
    n_legit = total_rows - n_fraud

    # ---- user profiles ---------------------------------------------------
    home = rng.integers(0, len(CITY_NAMES), N_USERS)
    mu = rng.normal(6.8, 0.6, N_USERS)                      # log-amount level per user
    pref_mode = rng.integers(0, 4, N_USERS)
    has_second = rng.random(N_USERS) < 0.2
    night_owl = rng.random(N_USERS) < 0.10
    activity = rng.gamma(2.0, 1.0, N_USERS); activity /= activity.sum()
    risky_merchants = rng.choice(N_MERCHANTS, int(N_MERCHANTS * 0.04), replace=False)

    # ---- legit transactions ---------------------------------------------
    u = rng.choice(N_USERS, n_legit, p=activity)
    day = rng.integers(0, N_DAYS, n_legit)
    hour = np.where(night_owl[u], (rng.normal(23, 3, n_legit)) % 24,
                    np.clip(rng.normal(14, 4, n_legit), 7, 23)).astype(int)
    ts = START + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") \
        + pd.to_timedelta(rng.integers(0, 3600, n_legit), "s")
    city = np.where(rng.random(n_legit) < 0.06, rng.integers(0, len(CITY_NAMES), n_legit), home[u])
    device = np.where(has_second[u] & (rng.random(n_legit) < 0.25), 200_000 + u, 100_000 + u)
    new_dev = rng.random(n_legit) < 0.03                    # legit device changes
    device = np.where(new_dev, 300_000 + rng.integers(0, 10**6, n_legit), device)
    amount = np.exp(rng.normal(mu[u], 0.7))
    big = rng.random(n_legit) < 0.05                         # legit big purchases
    amount = np.where(big, amount * rng.uniform(2, 5, n_legit), amount)
    mode = MODES[np.where(rng.random(n_legit) < 0.7, pref_mode[u], rng.integers(0, 4, n_legit))]
    merchant = np.where(rng.random(n_legit) < 0.02, rng.choice(risky_merchants, n_legit),
                        rng.integers(0, N_MERCHANTS, n_legit))
    legit = _frame(rng, u, ts, city, device, amount, mode, merchant, 0)

    # ---- fraud scenarios --------------------------------------------------
    fraud_pool = 900_000 + np.arange(400)                    # fraudster devices, reused across victims
    parts = []

    def pick_merchant(n):
        return np.where(rng.random(n) < 0.30, rng.choice(risky_merchants, n), rng.integers(0, N_MERCHANTS, n))

    def other_city(c):
        return (c + rng.integers(1, len(CITY_NAMES), len(c))) % len(CITY_NAMES)

    def rand_time(n, night=False):
        d = rng.integers(0, N_DAYS, n)
        h = rng.integers(0, 5, n) if night else rng.integers(0, 24, n)
        return START + pd.to_timedelta(d, "D") + pd.to_timedelta(h, "h") + pd.to_timedelta(rng.integers(0, 3600, n), "s")

    # account takeover: unseen shared fraud device, different city, big amount, often night
    n = int(n_fraud * 0.25); v = rng.integers(0, N_USERS, n)
    night = rng.random(n) < 0.6
    t = np.where(night, rand_time(n, True), rand_time(n, False))
    parts.append(_frame(rng, v, pd.DatetimeIndex(t), other_city(home[v]), rng.choice(fraud_pool, n),
                        np.exp(mu[v]) * rng.uniform(1.5, 6, n), MODES[rng.choice([0, 2, 3], n)], pick_merchant(n), 1))

    # velocity burst / card testing: 5-10 txns within minutes, small then larger
    made, rows = 0, []
    target = int(n_fraud * 0.15)
    while made < target:
        k = int(rng.integers(5, 11)); vv = int(rng.integers(0, N_USERS))
        t0 = rand_time(1)[0]
        gaps = np.cumsum(rng.integers(30, 480, k))
        amts = np.exp(mu[vv]) * np.where(np.arange(k) < 3, rng.uniform(0.01, 0.1, k), rng.uniform(0.5, 3, k))
        own = rng.random() < 0.5
        dev = np.full(k, 100_000 + vv) if own else np.full(k, rng.choice(fraud_pool))
        c = np.full(k, home[vv] if rng.random() < 0.5 else (home[vv] + 1) % len(CITY_NAMES))
        rows.append(_frame(rng, np.full(k, vv), pd.DatetimeIndex(t0 + pd.to_timedelta(gaps, "s")), c, dev, amts,
                           np.full(k, MODES[int(rng.integers(0, 4))]), pick_merchant(k), 1))
        made += k
    parts += rows

    # impossible travel: far-away txn minutes after a genuine one
    n = int(n_fraud * 0.15); a = legit.sample(n, random_state=seed)
    ac = np.array([CITY_NAMES.index(c) for c in a.city])
    far = np.array([rng.choice(np.where(_hav(LAT[x], LON[x], LAT, LON) > 800)[0]) for x in ac])
    parts.append(_frame(rng, a.user_id.values, pd.DatetimeIndex(a.timestamp + pd.to_timedelta(rng.integers(180, 2700, n), "s")),
                        far, np.where(rng.random(n) < 0.5, a.device_id.values, rng.choice(fraud_pool, n)),
                        a.amount.values * rng.uniform(1, 3, n), a.payment_mode.values, pick_merchant(n), 1))

    # wallet drain: big wallet amount at night, usually new device
    n = int(n_fraud * 0.10); v = rng.integers(0, N_USERS, n)
    parts.append(_frame(rng, v, pd.DatetimeIndex(rand_time(n, True)), home[v],
                        np.where(rng.random(n) < 0.7, rng.choice(fraud_pool, n), 100_000 + v),
                        np.exp(mu[v]) * rng.uniform(5, 15, n), np.full(n, "Wallet"), pick_merchant(n), 1))

    # subtle: own device + home city, only the amount is off
    n = n_fraud - sum(len(p) for p in parts); n = max(n, 0); v = rng.integers(0, N_USERS, n)
    parts.append(_frame(rng, v, pd.DatetimeIndex(rand_time(n)), home[v], 100_000 + v,
                        np.exp(mu[v]) * rng.uniform(1.8, 3.5, n), MODES[pref_mode[v]], pick_merchant(n), 1))

    df = pd.concat([legit] + parts, ignore_index=True)
    df = df.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)
    return df


def create_datasets(train_rows=500_000, prod_rows=100_000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    df = generate_stream(train_rows + prod_rows)
    df = df.iloc[: train_rows + prod_rows]
    train, prod = df.iloc[:train_rows], df.iloc[train_rows:]     # older -> train, newer -> production
    train.to_csv(f"{output_dir}/train.csv", index=False)
    prod.to_csv(f"{output_dir}/production.csv", index=False)
    print(f"train {len(train):,} rows (fraud {train.is_fraud.mean():.2%}) | "
          f"production {len(prod):,} rows (fraud {prod.is_fraud.mean():.2%}) | "
          f"split at {prod.timestamp.iloc[0]}")


if __name__ == "__main__":
    create_datasets()