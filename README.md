# 🛡️ Fraud Detection System

End-to-end machine learning pipeline for detecting fraudulent Indian payment transactions.
Processes 600K+ transactions using an ensemble of XGBoost + Isolation Forest
with real-time scoring via FastAPI and a live Streamlit analytics dashboard.

---

## 🔴 Live Demo

- **Dashboard** → https://fraud-detection-upasana.streamlit.app

---

## 📊 Results

| Metric            | Score                |
| ----------------- | -------------------- |
| AUC-ROC           | 0.89605              |
| Precision (Fraud) | 1.00                 |
| Recall (Fraud)    | 0.05                 |
| F1 Score (Fraud)  | 0.09                 |
| Training Data     | 500,000 transactions |
| Production Data   | 100,000 transactions |

---

## 🌟 What Makes This Unique

- **India-specific** — UPI, RuPay, NetBanking, Wallet payment modes with all 31 Indian states
- **Impossible travel detection** — Haversine distance formula to flag physically impossible transactions
- **Hybrid ensemble** — XGBoost (supervised) + Isolation Forest (unsupervised) combined
- **Three-tier decisions** — ALLOW / CHALLENGE / BLOCK mirrors real payment processors
- **Multi-layer risk scoring** — ML score + rule-based boosts on top

---

## 🔍 How It Works

```
User makes a payment
        ↓
Feature Engineering
(new device? impossible travel? spending anomaly?)
        ↓
XGBoost (70%) + Isolation Forest (30%)
        ↓
Rule-based boosts applied
        ↓
Final Risk Score (0 to 1)
        ↓
ALLOW (<0.6) / CHALLENGE (0.6-0.8) / BLOCK (>0.8)
```

---

## ⚙️ Features Engineered

| Feature                  | What it Detects                             |
| ------------------------ | ------------------------------------------- |
| Impossible Travel        | User in Mumbai then Chennai in 10 mins      |
| Velocity Check           | 5+ transactions or ₹20,000+ in 1 hour       |
| Device Fingerprinting    | Transaction from unknown device             |
| Device + Location Risk   | New device AND new location simultaneously  |
| State Change Detection   | Sudden transaction from different state     |
| OS Change Detection      | OS version changed between transactions     |
| Night Transaction        | Transaction between 12am and 6am            |
| Amount Anomaly           | Transaction way above user's usual spending |
| Merchant Risk Scoring    | Merchants with high historical fraud rates  |
| User Behaviour Profiling | Each user's spending patterns tracked       |

---

## 🛠️ Tech Stack

| Category        | Technology                |
| --------------- | ------------------------- |
| Language        | Python 3.10+              |
| ML Models       | XGBoost, Isolation Forest |
| Imbalanced Data | SMOTE (imbalanced-learn)  |
| Data Processing | Pandas, NumPy             |
| API             | FastAPI + Uvicorn         |
| Dashboard       | Streamlit                 |
| Visualisation   | Matplotlib, Seaborn       |
| Model Saving    | Joblib                    |
| Geo Distance    | Haversine Formula         |

---

## 📁 Project Structure

```
<<<<<<< HEAD
fraud_detect/
├── main.py                    ← Entry point (run this)
=======
fraud_detection/
├── main.py                    ← Run full pipeline
├── api.py                     ← FastAPI REST endpoint
├── dashboard.py               ← Streamlit dashboard
>>>>>>> 72fa564 (Add complete README)
├── requirements.txt
├── src/
│   ├── data_generator.py      ← Synthetic data generation
│   ├── feature_engineering.py ← All feature functions
│   ├── train.py               ← Model training
│   ├── predict.py             ← Inference + risk engine
│   └── evaluate.py            ← Metrics + charts
├── data/
│   └── final_output.csv       ← Scored transactions
├── model/
│   ├── xgb_model.pkl
│   ├── iso_model.pkl
│   └── feature_columns.json
└── reports/                   ← Auto-generated charts
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall.png
    ├── feature_importance.png
    ├── score_distribution.png
    └── decision_distribution.png
```

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/upasanaa0/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run full pipeline

```bash
python main.py
```

### 5. Generate evaluation reports

```bash
python -m src.evaluate
```

### 6. Start API

```bash
uvicorn api:app --reload
# Visit http://localhost:8000/docs
```

### 7. Start dashboard

```bash
streamlit run dashboard.py
# Visit http://localhost:8501
```

---

## 📡 API Usage

Send a transaction to score:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "user_id": 12345,
  "amount": 9500.00,
  "merchant_id": 456,
  "state": "Maharashtra",
  "payment_mode": "Wallet",
  "device_id": 78901,
  "device_type": "Android",
  "os_version": "13",
  "app_version": "2.0",
  "new_device_flag": 1,
  "state_changed": 1,
  "device_location_risk": 1
}'
```

Response:

```json
{
  "user_id": 12345,
  "amount": 9500.0,
  "fraud_score": 0.87,
  "decision": "BLOCK",
  "breakdown": {
    "xgb_probability": 0.76,
    "isolation_score": 0.43,
    "rule_boosts_applied": {
      "state_changed": true,
      "new_device": true,
      "device_location_risk": true
    }
  }
}
```

---

## 📈 Dashboard Features

- **KPI Cards** — Total transactions, fraud rate, BLOCK/CHALLENGE/ALLOW counts
- **High Risk Alerts** — Top 10 riskiest transactions with expandable details
- **Decision Distribution** — Pie chart of all decisions
- **Fraud Score Distribution** — Histogram comparing legit vs fraud scores
- **Model Evaluation** — Confusion matrix, ROC curve, precision-recall, feature importance
- **Transaction Table** — Filterable table with CSV download

---

## 🏗️ Architecture

```
Transaction Data
      ↓
Feature Engineering (15+ features)
      ↓
┌─────────────────────────────────┐
│  XGBoost Classifier    (70%)    │
│  Isolation Forest      (30%)    │
└─────────────────────────────────┘
      ↓
Ensemble Score
      ↓
Rule-Based Risk Engine
(impossible travel, velocity, device, location boosts)
      ↓
Final Risk Score (0.0 → 1.0)
      ↓
ALLOW / CHALLENGE / BLOCK
```

---

## 👩‍💻 Author

**Upasana**

- GitHub → https://github.com/upasanaa0

---

_Built as a portfolio project demonstrating end-to-end ML engineering for financial fraud detection_
