"""
Model 1: Price Forecaster
Algorithm : Facebook Prophet (time-series)
Target    : Forecast future Unit Price per Item Category
Input     : order_date, avg unit_price per month per category
Output    : next 6-month price forecast + trend + seasonality
"""

import pandas as pd
import numpy as np
import pickle
import os
from prophet import Prophet

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/procurement_data.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & Prepare Data ────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Order_Date", "Delivery_Date"])
    df.columns = [c.lower() for c in df.columns]
    return df


def prepare_prophet_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prophet requires exactly two columns: ds (date) and y (value).
    We aggregate avg unit price per month per category.
    """
    cat_df = df[df["item_category"] == category].copy()
    monthly = (
        cat_df.groupby(pd.Grouper(key="order_date", freq="ME"))["unit_price"]
        .mean()
        .reset_index()
    )
    monthly.columns = ["ds", "y"]
    monthly = monthly.dropna()
    return monthly


# ── 2. Train Model per Category ───────────────────────────────────────────────
def train_forecaster(df: pd.DataFrame, category: str) -> Prophet:
    """
    Train a Prophet model for a single category.
    - yearly_seasonality: captures annual price cycles
    - changepoint_prior_scale: sensitivity to trend changes (inflation)
    """
    prophet_df = prepare_prophet_data(df, category)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # monthly data — no weekly pattern
        daily_seasonality=False,
        changepoint_prior_scale=0.3,  # higher = more flexible trend (good for inflation)
        interval_width=0.95,          # 95% confidence interval bands
    )
    model.fit(prophet_df)
    return model


# ── 3. Generate Forecast ──────────────────────────────────────────────────────
def forecast(model: Prophet, periods: int = 6) -> pd.DataFrame:
    """
    Predict next `periods` months.
    Returns df with: ds, yhat (forecast), yhat_lower, yhat_upper
    """
    future = model.make_future_dataframe(periods=periods, freq="ME")
    forecast_df = model.predict(future)
    return forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]


# ── 4. Train & Save All Category Models ──────────────────────────────────────
def train_all(df: pd.DataFrame) -> dict:
    categories = df["item_category"].unique()
    models = {}

    for cat in categories:
        print(f"   Training Prophet for: {cat}...")
        model = train_forecaster(df, cat)
        models[cat] = model

        # Save each model as a pickle file
        model_path = os.path.join(MODEL_DIR, f"prophet_{cat.replace(' ', '_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"   ✅ Saved → {model_path}")

    return models


# ── 5. Load Saved Model ───────────────────────────────────────────────────────
def load_model(category: str) -> Prophet:
    model_path = os.path.join(MODEL_DIR, f"prophet_{category.replace(' ', '_')}.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── 6. Evaluate Model (MAPE) ──────────────────────────────────────────────────
def evaluate(model: Prophet, df: pd.DataFrame, category: str) -> float:
    """
    Mean Absolute Percentage Error — lower is better.
    Splits data: 80% train, 20% test.
    """
    prophet_df = prepare_prophet_data(df, category)
    split = int(len(prophet_df) * 0.8)
    test = prophet_df.iloc[split:].copy()

    future = model.make_future_dataframe(periods=len(test), freq="ME")
    preds  = model.predict(future).tail(len(test))

    mape = np.mean(np.abs((test["y"].values - preds["yhat"].values) / test["y"].values)) * 100
    return round(mape, 2)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  🔮 Price Forecaster — Training")
    print("=" * 50)

    df = load_data()
    print(f"Loaded {len(df)} rows | Categories: {df['item_category'].nunique()}\n")

    models = train_all(df)

    print("\n📊 Model Evaluation (MAPE):")
    for cat, model in models.items():
        mape = evaluate(model, df, cat)
        print(f"   {cat:<20} MAPE: {mape}%")

    print("\n✅ All price forecaster models trained and saved!")

    # Quick forecast preview
    print("\n📈 Sample Forecast — Raw Materials (next 3 months):")
    fc = forecast(models["Raw Materials"], periods=3)
    print(fc.tail(3)[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_string(index=False))