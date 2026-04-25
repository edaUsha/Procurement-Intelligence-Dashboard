"""
Model 2 (Improved): Compliance Risk Predictor
Algorithm : XGBoost Classifier + threshold tuning + cross-validation
Improvements over v1:
  - Class imbalance handled via scale_pos_weight
  - Threshold tuning to maximize F1 on Non-Compliant class
  - 5-fold cross-validation for robust evaluation
  - 5 additional engineered features
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_curve, f1_score, confusion_matrix)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/procurement_data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "compliance_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "compliance_encoders.pkl")
META_PATH  = os.path.join(MODEL_DIR, "compliance_meta.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & Feature Engineer ────────────────────────────────────────────────
def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Order_Date", "Delivery_Date"])
    df.columns = [c.lower() for c in df.columns]

    df["price_gap"]        = df["unit_price"] - df["negotiated_price"]
    df["price_gap_pct"]    = (df["price_gap"] / df["unit_price"]).round(4)
    df["lead_time_days"]   = (df["delivery_date"] - df["order_date"]).dt.days.fillna(-1)
    df["defect_rate"]      = (df["defective_units"] / df["quantity"]).round(4)
    df["order_month"]      = df["order_date"].dt.month
    df["order_quarter"]    = df["order_date"].dt.quarter
    df["order_year"]       = df["order_date"].dt.year
    df["order_value"]      = df["quantity"] * df["negotiated_price"]

    # Advanced engineered features
    df["high_defect_flag"]  = (df["defect_rate"] > 0.10).astype(int)
    df["missing_delivery"]  = df["delivery_date"].isna().astype(int)
    df["large_order_flag"]  = (df["quantity"] > df["quantity"].quantile(0.75)).astype(int)
    df["aggressive_neg"]    = (df["price_gap_pct"] > 0.10).astype(int)

    # Target: 1 = Non-Compliant
    df["target"] = (df["compliance"] == "Non-Compliant").astype(int)
    return df


# ── 2. Encode ─────────────────────────────────────────────────────────────────
def encode_features(df):
    encoders = {}
    for col in ["supplier", "item_category", "order_status"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


FEATURES = [
    "supplier_enc", "item_category_enc", "order_status_enc",
    "quantity", "unit_price", "negotiated_price",
    "price_gap", "price_gap_pct", "defect_rate",
    "order_month", "order_quarter", "order_year", "order_value",
    "lead_time_days", "high_defect_flag", "missing_delivery",
    "large_order_flag", "aggressive_neg",
]


# ── 3. Train ──────────────────────────────────────────────────────────────────
def train(df):
    X, y = df[FEATURES], df["target"]
    neg, pos = (y == 0).sum(), (y == 1).sum()
    imbalance_ratio = neg / pos
    print(f"   Class ratio — Compliant:{neg}  Non-Compliant:{pos}  Ratio:{imbalance_ratio:.1f}x")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.75,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=imbalance_ratio,
        eval_metric="aucpr",
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, X_train, X_test, y_train, y_test


# ── 4. Threshold Tuning ───────────────────────────────────────────────────────
def tune_threshold(model, X_test, y_test) -> float:
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    default_f1 = f1_score(y_test, (proba >= 0.5).astype(int))
    print(f"\n🎯 Threshold Tuning:")
    print(f"   Default (0.50) F1: {default_f1:.3f}")
    print(f"   Best    ({best_threshold:.2f}) F1: {f1_scores[best_idx]:.3f}")
    return round(best_threshold, 3)


# ── 5. Cross-Validation ───────────────────────────────────────────────────────
def cross_validate(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n📐 5-Fold CV AUC: {[round(s,3) for s in scores]}")
    print(f"   Mean: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


# ── 6. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    auc   = roc_auc_score(y_test, proba)

    print(f"\n📊 Evaluation (threshold={threshold}):")
    print(classification_report(y_test, preds, target_names=["Compliant", "Non-Compliant"]))
    print(f"   ROC-AUC: {auc:.4f}")
    cm = confusion_matrix(y_test, preds)
    print(f"   Confusion Matrix → TN:{cm[0,0]} FP:{cm[0,1]} | FN:{cm[1,0]} TP:{cm[1,1]}")
    return auc


# ── 7. Feature Importance ─────────────────────────────────────────────────────
def feature_importance(model):
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


# ── 8. Predict for Dashboard ──────────────────────────────────────────────────
def predict_compliance_risk(order: dict, model, encoders: dict, threshold=0.5) -> dict:
    row = pd.DataFrame([order])
    for col in ["supplier", "item_category", "order_status"]:
        row[col + "_enc"] = encoders[col].transform(row[col].astype(str))

    row["price_gap"]       = row["unit_price"] - row["negotiated_price"]
    row["price_gap_pct"]   = row["price_gap"] / row["unit_price"]
    row["order_value"]     = row["quantity"] * row["negotiated_price"]
    row["high_defect_flag"]= int(row.get("defect_rate", pd.Series([0]))[0] > 0.10)
    row["missing_delivery"]= 0
    row["large_order_flag"]= int(row["quantity"].iloc[0] > 375)
    row["aggressive_neg"]  = int(row["price_gap_pct"].iloc[0] > 0.10)
    row["order_quarter"]   = ((row["order_month"] - 1) // 3 + 1)

    prob = float(model.predict_proba(row[FEATURES])[0][1])
    risk = "High" if prob >= threshold else "Medium" if prob >= threshold * 0.6 else "Low"
    return {"probability": round(prob, 4), "risk_level": risk, "threshold_used": threshold}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🛡️  Compliance Risk Predictor v2")
    print("=" * 55)

    df = load_and_prepare(DATA_PATH)
    df, encoders = encode_features(df)
    print(f"\nRows: {len(df)} | Non-compliant: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")

    model, X_train, X_test, y_train, y_test = train(df)
    cross_validate(model, X_train, y_train)
    best_threshold = tune_threshold(model, X_test, y_test)
    evaluate(model, X_test, y_test, threshold=best_threshold)

    print("\n🔍 Top 8 Feature Importances:")
    print(feature_importance(model).head(8).to_string(index=False))

    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(ENC_PATH,   "wb") as f: pickle.dump(encoders, f)
    with open(META_PATH,  "wb") as f: pickle.dump({"threshold": best_threshold}, f)

    print(f"\n✅ Model    → {MODEL_PATH}")
    print(f"✅ Encoders → {ENC_PATH}")
    print(f"✅ Meta     → {META_PATH}  (threshold={best_threshold})")

    print("\n🧪 Sample Predictions:")
    samples = [
        {"supplier": "DeltaWorks", "item_category": "Electronics", "order_status": "Pending",
         "quantity": 400, "unit_price": 520.0, "negotiated_price": 455.0,
         "lead_time_days": -1, "defect_rate": 0.15, "order_month": 6, "order_year": 2024},
        {"supplier": "EpsilonMfg", "item_category": "Consumables", "order_status": "Delivered",
         "quantity": 50, "unit_price": 20.0, "negotiated_price": 19.0,
         "lead_time_days": 10, "defect_rate": 0.01, "order_month": 3, "order_year": 2024},
    ]
    for s in samples:
        r = predict_compliance_risk(s, model, encoders, best_threshold)
        print(f"   {s['supplier']} | {s['item_category']} → {r['risk_level']} ({r['probability']*100:.1f}%)")