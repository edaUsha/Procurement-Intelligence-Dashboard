"""
Synthetic Procurement KPI Dataset Generator
Matches Kaggle schema: PO_ID, Supplier, Order_Date, Delivery_Date,
Item_Category, Order_Status, Quantity, Unit_Price, Negotiated_Price,
Defective_Units, Compliance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# --- Config ---
N_ROWS = 2000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "procurement_data.csv")

SUPPLIERS = ["AlphaSupplies", "BetaCorp", "GammaTrade", "DeltaWorks", "EpsilonMfg"]
CATEGORIES = ["Raw Materials", "Packaging", "Electronics", "Machinery", "Consumables"]
STATUSES = ["Delivered", "Pending", "Partial", "Cancelled"]

# Supplier profiles: (reliability, compliance_rate, defect_rate)
SUPPLIER_PROFILES = {
    "AlphaSupplies": (0.90, 0.95, 0.02),
    "BetaCorp":      (0.75, 0.80, 0.07),
    "GammaTrade":    (0.85, 0.88, 0.04),
    "DeltaWorks":    (0.60, 0.65, 0.12),
    "EpsilonMfg":    (0.95, 0.98, 0.01),
}

# Base unit price per category (with inflation applied later)
BASE_PRICES = {
    "Raw Materials": 120,
    "Packaging":     35,
    "Electronics":   480,
    "Machinery":     2200,
    "Consumables":   18,
}

START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 6, 30)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days


def random_date(start, days_range):
    return start + timedelta(days=random.randint(0, days_range))


def apply_inflation(base_price, order_date):
    """Simulate ~8% annual inflation from 2022 onwards."""
    years_elapsed = (order_date - START_DATE).days / 365
    return round(base_price * (1.08 ** years_elapsed), 2)


rows = []
for i in range(1, N_ROWS + 1):
    supplier = random.choice(SUPPLIERS)
    reliability, compliance_rate, defect_rate = SUPPLIER_PROFILES[supplier]
    category = random.choice(CATEGORIES)

    order_date = random_date(START_DATE, DATE_RANGE_DAYS)
    base_price = BASE_PRICES[category]
    unit_price = apply_inflation(base_price, order_date)

    # Negotiated price: 3–12% savings depending on supplier
    savings_pct = round(random.uniform(0.03, 0.12), 4)
    negotiated_price = round(unit_price * (1 - savings_pct), 2)

    quantity = random.randint(10, 500)

    # Delivery date: None if not delivered, else lead time 7–45 days
    status_rand = random.random()
    if status_rand < reliability:
        status = "Delivered"
        lead_days = random.randint(7, 45)
        delivery_date = order_date + timedelta(days=lead_days)
        delivery_date_str = delivery_date.strftime("%Y-%m-%d")
    elif status_rand < reliability + 0.10:
        status = "Partial"
        lead_days = random.randint(20, 60)
        delivery_date = order_date + timedelta(days=lead_days)
        delivery_date_str = delivery_date.strftime("%Y-%m-%d")
    elif status_rand < reliability + 0.15:
        status = "Cancelled"
        delivery_date_str = None
    else:
        status = "Pending"
        delivery_date_str = None

    # Defective units
    defective = int(quantity * random.uniform(0, defect_rate * 2))
    defective = min(defective, quantity)

    # Compliance: introduce ~missing delivery date as non-compliance
    if delivery_date_str is None and status not in ("Cancelled", "Pending"):
        compliance = "Non-Compliant"
    else:
        compliance = "Compliant" if random.random() < compliance_rate else "Non-Compliant"

    # Introduce ~3% missing delivery dates even for delivered (data quality issue)
    if status == "Delivered" and random.random() < 0.03:
        delivery_date_str = None

    rows.append({
        "PO_ID":             f"PO-{i:05d}",
        "Supplier":          supplier,
        "Order_Date":        order_date.strftime("%Y-%m-%d"),
        "Delivery_Date":     delivery_date_str,
        "Item_Category":     category,
        "Order_Status":      status,
        "Quantity":          quantity,
        "Unit_Price":        unit_price,
        "Negotiated_Price":  negotiated_price,
        "Defective_Units":   defective,
        "Compliance":        compliance,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Generated {len(df)} rows → {OUTPUT_PATH}")
print(df.head(3).to_string())
print(f"\nStatus distribution:\n{df['Order_Status'].value_counts()}")
print(f"\nCompliance distribution:\n{df['Compliance'].value_counts()}")
