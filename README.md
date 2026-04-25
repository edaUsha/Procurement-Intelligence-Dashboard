# 🏭 Procurement Intelligence Dashboard

A real-time ML-powered procurement analytics dashboard built with Python, PostgreSQL, and Streamlit.

---

## 📁 Project Structure

```
procurement_dashboard/
├── data/
│   ├── generate_data.py        # Synthetic dataset generator (matches Kaggle schema)
│   └── procurement_data.csv    # Generated CSV (2000 rows)
├── database/
│   └── setup_db.py             # DB schema creation + CSV loader
├── pipeline/
│   └── live_pipeline.py        # Live data fetchers (TTL cache) + insert simulator
├── models/                     # ML models (next phase)
│   ├── price_forecaster.py     # Prophet — unit price trend forecasting
│   ├── compliance_predictor.py # XGBoost — compliance risk classifier
│   └── defect_predictor.py     # Regression — defective units predictor
├── dashboard/                  # Streamlit app (next phase)
│   └── app.py
├── config/
│   └── .env.example            # Environment variable template
└── requirements.txt
```

---

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL
```bash
# Create the database
createdb procurement_db

# Or via psql:
psql -U postgres -c "CREATE DATABASE procurement_db;"
```

### 3. Configure Environment
```bash
cp config/.env.example config/.env
# Edit .env with your actual DB credentials
```

### 4. Run DB Setup (Schema + Load Data)
```bash
python database/setup_db.py
```

This will:
- Create `purchase_orders` table with indexes
- Create `live_updates_log` table
- Create `supplier_kpi` materialized view
- Load 2000 rows from CSV

### 5. Run the Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔴 Live Data Updates

The dashboard auto-refreshes every **60 seconds** using Streamlit's `st.cache_data(ttl=60)`.

**How it works:**
1. Your ERP/production system inserts new rows into `purchase_orders`
2. Streamlit cache expires → re-queries PostgreSQL → dashboard updates
3. The `supplier_kpi` materialized view is refreshed on every insert

**To simulate live inserts (demo mode):**
- Enable via `ENABLE_LIVE_SIMULATOR=true` in `.env`
- Click the "➕ Simulate New PO" button in the dashboard sidebar

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI cards: total spend, savings, defect rate, on-time delivery |
| **Supplier Scorecard** | Risk ratings, compliance heatmap per supplier |
| **Price Forecasting** | Prophet model: actual vs. forecasted price by category |
| **Order Tracker** | Live PO status table, delay flags |
| **Anomaly Alerts** | Outliers in pricing or defects auto-flagged |

---

## 🧠 ML Models (Phase 2)

| Model | Algorithm | Target |
|-------|-----------|--------|
| Price Forecaster | Facebook Prophet | Future unit prices by category |
| Compliance Risk | XGBoost Classifier | Predict non-compliance probability |
| Defect Predictor | XGBoost Regressor | Predict defective units per order |
