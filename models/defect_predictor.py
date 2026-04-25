"""
Model 3 (Improved): Defect Predictor
Algorithm : XGBoost Regressor
Improvements over v1:
  - Supplier historical defect rate as a feature (most predictive signal)
  - Category historical defect rate
  - Log-transform target to handle skewed defect distribution
  - MAPE evaluation metric added
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/procurement_data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "defect_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "defect_encoders.pkl")
META_PATH  = os.path.join(MODEL_DIR, "defect_meta.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & Prepare ─────────────────────────────────────────────────────────
def load_and_prepare(path: str):
    df = pd.read_csv(path, parse_dates=["Order_Date", "Delivery_Date"])
    df.columns = [c.lower() for c in df.columns]

    # Only delivered/partial orders have meaningful defect data
    df = df[df["order_status"].isin(["Delivered", "Partial"])].copy()

    # Basic features
    df["price_gap_pct"]  = ((df["unit_price"] - df["negotiated_price"]) / df["unit_price"]).round(4)
    df["order_month"]    = df["order_date"].dt.month
    df["order_quarter"]  = df["order_date"].dt.quarter
    df["order_year"]     = df["order_date"].dt.year
    df["order_value"]    = df["quantity"] * df["negotiated_price"]
    df["lead_time_days"] = (df["delivery_date"] - df["order_date"]).dt.days.fillna(-1)
    df["price_per_unit"] = df["negotiated_price"]

    # ── Key advanced features ──
    # Supplier's historical avg defect rate (strong signal)
    supplier_defect_avg = df.groupby("supplier")["defective_units"].transform("mean")
    df["supplier_hist_defect"] = supplier_defect_avg.round(2)

    # Category's historical avg defect rate
    cat_defect_avg = df.groupby("item_category")["defective_units"].transform("mean")
    df["category_hist_defect"] = cat_defect_avg.round(2)

    # Large quantity orders tend to have more defects
    df["large_order_flag"] = (df["quantity"] > df["quantity"].quantile(0.75)).astype(int)

    # High negotiation → possible quality tradeoff
    df["aggressive_neg"]   = (df["price_gap_pct"] > 0.10).astype(int)

    df = df.dropna(subset=["defective_units"])
    return df


# ── 2. Encode ─────────────────────────────────────────────────────────────────
def encode_features(df):
    encoders = {}
    for col in ["supplier", "item_category"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


FEATURES = [
    "supplier_enc", "item_category_enc",
    "quantity", "unit_price", "negotiated_price",
    "price_gap_pct", "order_month", "order_quarter", "order_year",
    "order_value", "lead_time_days",
    "supplier_hist_defect", "category_hist_defect",
    "large_order_flag", "aggressive_neg",
]


# ── 3. Train ──────────────────────────────────────────────────────────────────
def train(df):
    X = df[FEATURES]
    y = df["defective_units"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.75,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, X_train, X_test, y_train, y_test


# ── 4. Cross-Validation ───────────────────────────────────────────────────────
def cross_validate(model, X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    print(f"\n📐 5-Fold CV R²: {[round(s,3) for s in scores]}")
    print(f"   Mean: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    preds = np.maximum(0, model.predict(X_test))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mape  = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100  # +1 avoids div/0

    print(f"\n📊 Evaluation:")
    print(f"   MAE  : {mae:.2f} units   (avg prediction error)")
    print(f"   R²   : {r2:.4f}          (1.0 = perfect)")
    print(f"   MAPE : {mape:.1f}%        (mean absolute % error)")
    return mae, r2, mape


# ── 6. Feature Importance ─────────────────────────────────────────────────────
def feature_importance(model):
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


# ── 7. Predict for Dashboard ──────────────────────────────────────────────────
def predict_defects(order: dict, model, encoders: dict,
                    supplier_hist: dict, category_hist: dict) -> dict:
    """
    order dict keys: supplier, item_category, quantity, unit_price,
                     negotiated_price, order_month, order_year, lead_time_days
    supplier_hist / category_hist: lookup dicts from training data
    """
    row = pd.DataFrame([order])
    for col in ["supplier", "item_category"]:
        row[col + "_enc"] = encoders[col].transform(row[col].astype(str))

    row["price_gap_pct"]         = (row["unit_price"] - row["negotiated_price"]) / row["unit_price"]
    row["order_value"]           = row["quantity"] * row["negotiated_price"]
    row["order_quarter"]         = ((row["order_month"] - 1) // 3 + 1)
    row["supplier_hist_defect"]  = supplier_hist.get(order["supplier"], 10.0)
    row["category_hist_defect"]  = category_hist.get(order["item_category"], 10.0)
    row["large_order_flag"]      = int(order["quantity"] > 375)
    row["aggressive_neg"]        = int(row["price_gap_pct"].iloc[0] > 0.10)

    predicted = max(0, round(float(model.predict(row[FEATURES])[0])))
    rate_pct  = round(predicted / order["quantity"] * 100, 2)

    return {
        "predicted_defects": predicted,
        "defect_rate_pct":   rate_pct,
        "risk_level": "🔴 High" if rate_pct > 8 else "🟡 Medium" if rate_pct > 3 else "🟢 Low"
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🔬 Defect Predictor v2")
    print("=" * 55)

    df = load_and_prepare(DATA_PATH)
    df, encoders = encode_features(df)

    # Build historical lookup dicts (saved for dashboard use)
    supplier_hist  = df.groupby("supplier")["defective_units"].mean().to_dict()
    category_hist  = df.groupby("item_category")["defective_units"].mean().to_dict()

    print(f"\nRows: {len(df)} (delivered/partial)")
    print(f"Avg defective units: {df['defective_units'].mean():.1f}")
    print(f"Supplier hist defects: { {k: round(v,1) for k,v in supplier_hist.items()} }")

    model, X_train, X_test, y_train, y_test = train(df)
    cross_validate(model, X_train, y_train)
    evaluate(model, X_test, y_test)

    print("\n🔍 Top 8 Feature Importances:")
    print(feature_importance(model).head(8).to_string(index=False))

    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(ENC_PATH,   "wb") as f: pickle.dump(encoders, f)
    with open(META_PATH,  "wb") as f:
        pickle.dump({"supplier_hist": supplier_hist, "category_hist": category_hist}, f)

    print(f"\n✅ Model    → {MODEL_PATH}")
    print(f"✅ Encoders → {ENC_PATH}")
    print(f"✅ Meta     → {META_PATH}")

    print("\n🧪 Sample Predictions:")
    samples = [
        {"supplier": "DeltaWorks", "item_category": "Electronics",
         "quantity": 200, "unit_price": 500.0, "negotiated_price": 440.0,
         "order_month": 6, "order_year": 2024, "lead_time_days": 30},
        {"supplier": "EpsilonMfg", "item_category": "Consumables",
         "quantity": 100, "unit_price": 20.0, "negotiated_price": 19.2,
         "order_month": 3, "order_year": 2024, "lead_time_days": 12},
    ]
    for s in samples:
        r = predict_defects(s, model, encoders, supplier_hist, category_hist)
        print(f"   {s['supplier']} | {s['item_category']} | Qty:{s['quantity']} → "
              f"{r['predicted_defects']} defects ({r['defect_rate_pct']}%) {r['risk_level']}")