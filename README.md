# 🛡️ Fraud Detection Pipeline

A production-style ML pipeline for detecting fraudulent Indian payment transactions.  
Converted from a Colab notebook into a clean, modular VS Code project.

---

## Project Structure

```
fraud_detect/
├── main.py                    ← Entry point (run this)
├── requirements.txt
├── .vscode/
│   ├── launch.json            ← Run configs (F5 dropdown)
│   └── settings.json
├── src/
│   ├── data_generator.py      ← Synthetic data generation
│   ├── feature_engineering.py ← All feature / flag functions
│   ├── train.py               ← Model training (XGBoost + IsolationForest)
│   └── predict.py             ← Inference + risk engine
├── data/                      ← Created at runtime
│   ├── train.csv
│   ├── production.csv
│   └── final_output.csv       ← Final scored output
└── model/                     ← Created at runtime
    ├── xgb_model.pkl
    ├── iso_model.pkl
    └── feature_columns.json
```

---

## Quick Start

### 1. Create & activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Select Python interpreter in VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → choose `./venv`

### 4. Run the pipeline

**Option A – Terminal (full run):**
```bash
python main.py
```

**Option B – Fast dev run (smaller dataset):**
```bash
python main.py --train-rows 50000 --prod-rows 10000
```

**Option C – Individual steps:**
```bash
python main.py --step generate
python main.py --step train
python main.py --step predict
```

**Option D – VS Code debugger:**  
Press `F5` and pick a config from the dropdown.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| **Generate** | Creates 500k training + 100k production synthetic transactions |
| **Train** | Applies SMOTE, trains XGBoost + Isolation Forest, saves models |
| **Predict** | Scores production data, applies rule boosts, outputs decisions |

---

## Features Used

| Feature | Description |
|---------|-------------|
| `state_changed` | User transacted from a different state |
| `new_device_flag` | Transaction from a previously unseen device |
| `os_changed_flag` | OS version changed since last transaction |
| `device_location_risk` | New device AND new location simultaneously |
| `impossible_travel_flag` | Speed > 500 km/h between consecutive transactions |
| `velocity_risk_flag` | ≥5 transactions or >₹20,000 in last 1 hour |
| `amount_to_avg_ratio` | Transaction amount vs user's historical average |
| `is_night` | Transaction between midnight and 6 AM |
| Region / Payment dummies | One-hot encoded categorical variables |

---

## Output

`data/final_output.csv` contains every production transaction with:

| Column | Description |
|--------|-------------|
| `fraud_score` | Raw ensemble score (XGBoost 70% + IsoForest 30%) |
| `final_risk_score` | Score after rule-based boosts (capped at 1.0) |
| `decision` | **ALLOW** / **CHALLENGE** / **BLOCK** |

### Decision thresholds

| Decision | Score range |
|----------|-------------|
| ALLOW | < 0.5 |
| CHALLENGE | 0.5 – 0.79 |
| BLOCK | ≥ 0.8 |

---

## Performance Notes

Full run (500k + 100k rows) takes **10–25 minutes** depending on your machine.  
Use `--train-rows 50000 --prod-rows 10000` for a fast test run (~2–3 min).
