"""
Live Data Pipeline
- Connects to PostgreSQL
- Exposes cached data-fetching functions for Streamlit (TTL-based refresh)
- Simulates live ERP inserts for demo/testing purposes
- Provides a `get_connection()` helper for all dashboard queries
"""

import os
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import streamlit as st

# ── DB Config ─────────────────────────────────────────────────────────────────
def get_db_url():
    return (
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'password')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'procurement_db')}"
    )


@st.cache_resource
def get_engine():
    """Single SQLAlchemy engine shared across all Streamlit reruns."""
    return create_engine(get_db_url(), pool_pre_ping=True, pool_size=5)


# ── Live-refreshing data fetchers (TTL = 60 seconds) ─────────────────────────

@st.cache_data(ttl=60)
def fetch_all_orders() -> pd.DataFrame:
    """Full purchase_orders table — refreshes every 60s."""
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM purchase_orders ORDER BY order_date DESC",
        engine,
        parse_dates=["order_date", "delivery_date"],
    )
    return df


@st.cache_data(ttl=60)
def fetch_supplier_kpi() -> pd.DataFrame:
    """Pre-aggregated supplier KPIs from materialized view."""
    engine = get_engine()
    return pd.read_sql("SELECT * FROM supplier_kpi ORDER BY total_spend DESC", engine)


@st.cache_data(ttl=60)
def fetch_monthly_spend() -> pd.DataFrame:
    """Monthly aggregated spend + savings for trend charts."""
    engine = get_engine()
    query = """
        SELECT
            DATE_TRUNC('month', order_date)::date          AS month,
            item_category,
            SUM(quantity * negotiated_price)               AS total_spend,
            SUM((unit_price - negotiated_price) * quantity) AS total_savings,
            COUNT(*)                                        AS order_count,
            ROUND(AVG(negotiated_price), 2)                AS avg_price
        FROM purchase_orders
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return pd.read_sql(query, engine, parse_dates=["month"])


@st.cache_data(ttl=60)
def fetch_defect_trend() -> pd.DataFrame:
    """Monthly defect rate per supplier for anomaly view."""
    engine = get_engine()
    query = """
        SELECT
            DATE_TRUNC('month', order_date)::date AS month,
            supplier,
            SUM(defective_units)                  AS defective_units,
            SUM(quantity)                         AS total_units,
            ROUND(100.0 * SUM(defective_units) / NULLIF(SUM(quantity), 0), 2) AS defect_rate_pct
        FROM purchase_orders
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return pd.read_sql(query, engine, parse_dates=["month"])


@st.cache_data(ttl=60)
def fetch_recent_updates(limit: int = 20) -> pd.DataFrame:
    """Latest entries from live_updates_log for the live feed widget."""
    engine = get_engine()
    query = f"""
        SELECT po_id, event_type, old_value, new_value,
               updated_at AT TIME ZONE 'UTC' AS updated_at
        FROM live_updates_log
        ORDER BY updated_at DESC
        LIMIT {limit}
    """
    return pd.read_sql(query, engine, parse_dates=["updated_at"])


# ── Live Insert Simulator (for demo/testing) ──────────────────────────────────

SUPPLIERS   = ["AlphaSupplies", "BetaCorp", "GammaTrade", "DeltaWorks", "EpsilonMfg"]
CATEGORIES  = ["Raw Materials", "Packaging", "Electronics", "Machinery", "Consumables"]
STATUSES    = ["Delivered", "Pending", "Partial"]

def simulate_live_insert():
    """
    Inserts a synthetic new PO into purchase_orders and logs it.
    Call this from a scheduler or the dashboard's 'Simulate Live Data' button.
    """
    engine = get_engine()

    with engine.connect() as conn:
        # Get latest PO number to auto-increment
        max_id = conn.execute(
            text("SELECT MAX(CAST(SUBSTRING(po_id FROM 4) AS INTEGER)) FROM purchase_orders")
        ).scalar() or 0
        new_id = f"PO-{max_id + 1:05d}"

        supplier = random.choice(SUPPLIERS)
        category = random.choice(CATEGORIES)
        order_date = datetime.now().date()
        quantity = random.randint(10, 300)
        unit_price = round(random.uniform(20, 2500), 2)
        negotiated_price = round(unit_price * random.uniform(0.88, 0.97), 2)
        defective = random.randint(0, max(1, int(quantity * 0.05)))
        status = random.choice(STATUSES)
        delivery_date = (order_date + timedelta(days=random.randint(7, 30))) if status == "Delivered" else None
        compliance = "Compliant" if random.random() > 0.15 else "Non-Compliant"

        conn.execute(text("""
            INSERT INTO purchase_orders
                (po_id, supplier, order_date, delivery_date, item_category,
                 order_status, quantity, unit_price, negotiated_price,
                 defective_units, compliance)
            VALUES
                (:po_id, :supplier, :order_date, :delivery_date, :item_category,
                 :order_status, :quantity, :unit_price, :negotiated_price,
                 :defective_units, :compliance)
        """), {
            "po_id": new_id, "supplier": supplier, "order_date": order_date,
            "delivery_date": delivery_date, "item_category": category,
            "order_status": status, "quantity": quantity,
            "unit_price": unit_price, "negotiated_price": negotiated_price,
            "defective_units": defective, "compliance": compliance,
        })

        # Log the event
        conn.execute(text("""
            INSERT INTO live_updates_log (po_id, event_type, new_value)
            VALUES (:po_id, 'INSERT', :summary)
        """), {
            "po_id": new_id,
            "summary": f"{supplier} | {category} | {status} | Qty:{quantity} | ${negotiated_price}"
        })

        conn.commit()

    # Refresh materialized view
    with engine.connect() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY supplier_kpi"))
        conn.commit()

    # Invalidate Streamlit cache so next render fetches fresh data
    fetch_all_orders.clear()
    fetch_supplier_kpi.clear()
    fetch_monthly_spend.clear()
    fetch_defect_trend.clear()
    fetch_recent_updates.clear()

    return new_id
