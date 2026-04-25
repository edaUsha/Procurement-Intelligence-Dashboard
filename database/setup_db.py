"""
Database Setup Script
- Creates PostgreSQL schema for procurement data
- Loads CSV into the DB
- Creates indexes for fast dashboard queries
- Sets up a 'live_updates' log table for simulating real-time inserts

Usage:
    python database/setup_db.py

Env vars (or edit DB_CONFIG below):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# ── DB Configuration ──────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
    "dbname":   os.getenv("DB_NAME",     "procurement_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
}

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/procurement_data.csv")

# ── Schema DDL ────────────────────────────────────────────────────────────────
CREATE_TABLES_SQL = """
-- Main procurement orders table
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id               VARCHAR(20)    PRIMARY KEY,
    supplier            VARCHAR(100)   NOT NULL,
    order_date          DATE           NOT NULL,
    delivery_date       DATE,
    item_category       VARCHAR(100)   NOT NULL,
    order_status        VARCHAR(50)    NOT NULL,
    quantity            INTEGER        NOT NULL,
    unit_price          NUMERIC(12, 2) NOT NULL,
    negotiated_price    NUMERIC(12, 2) NOT NULL,
    defective_units     INTEGER        DEFAULT 0,
    compliance          VARCHAR(50)    NOT NULL,
    inserted_at         TIMESTAMP      DEFAULT NOW()
);

-- Indexes for dashboard filter performance
CREATE INDEX IF NOT EXISTS idx_po_supplier      ON purchase_orders(supplier);
CREATE INDEX IF NOT EXISTS idx_po_order_date    ON purchase_orders(order_date);
CREATE INDEX IF NOT EXISTS idx_po_category      ON purchase_orders(item_category);
CREATE INDEX IF NOT EXISTS idx_po_status        ON purchase_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_po_compliance    ON purchase_orders(compliance);

-- Live updates log (simulates streaming inserts from ERP/production systems)
CREATE TABLE IF NOT EXISTS live_updates_log (
    id              SERIAL         PRIMARY KEY,
    po_id           VARCHAR(20),
    event_type      VARCHAR(50),   -- 'INSERT', 'STATUS_CHANGE', 'DELIVERY_UPDATE'
    old_value       TEXT,
    new_value       TEXT,
    updated_at      TIMESTAMP      DEFAULT NOW()
);

-- Materialized view for supplier KPI summary (refreshed by pipeline)
CREATE MATERIALIZED VIEW IF NOT EXISTS supplier_kpi AS
SELECT
    supplier,
    COUNT(*)                                                        AS total_orders,
    ROUND(AVG(unit_price - negotiated_price), 2)                   AS avg_savings_per_unit,
    ROUND(SUM((unit_price - negotiated_price) * quantity), 2)      AS total_savings,
    ROUND(100.0 * SUM(CASE WHEN order_status = 'Delivered' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                    AS on_time_delivery_pct,
    ROUND(100.0 * SUM(defective_units) / NULLIF(SUM(quantity), 0), 2)
                                                                    AS defect_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN compliance = 'Compliant' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                    AS compliance_rate_pct,
    SUM(quantity * negotiated_price)                               AS total_spend
FROM purchase_orders
GROUP BY supplier;

CREATE UNIQUE INDEX IF NOT EXISTS idx_supplier_kpi ON supplier_kpi(supplier);
"""

REFRESH_MATVIEW_SQL = "REFRESH MATERIALIZED VIEW CONCURRENTLY supplier_kpi;"


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def setup_schema(engine):
    print("📐 Creating schema...")
    with engine.connect() as conn:
        conn.execute(text(CREATE_TABLES_SQL))
        conn.commit()
    print("   ✅ Tables and indexes created.")


def load_csv(engine):
    print("📥 Loading CSV data...")
    df = pd.read_csv(CSV_PATH, parse_dates=["Order_Date", "Delivery_Date"])
    df.columns = [c.lower() for c in df.columns]

    # Check if already loaded
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM purchase_orders")).scalar()

    if count > 0:
        print(f"   ℹ️  Table already has {count} rows — skipping bulk load.")
        print("      To reload, run: TRUNCATE TABLE purchase_orders RESTART IDENTITY;")
        return count

    df.to_sql("purchase_orders", engine, if_exists="append", index=False,
              method="multi", chunksize=500)
    print(f"   ✅ Loaded {len(df)} rows into purchase_orders.")
    return len(df)


def refresh_materialized_views(engine):
    print("🔄 Refreshing materialized views...")
    with engine.connect() as conn:
        conn.execute(text(REFRESH_MATVIEW_SQL))
        conn.commit()
    print("   ✅ supplier_kpi view refreshed.")


def verify_setup(engine):
    print("\n📊 Verification:")
    queries = {
        "Total POs":        "SELECT COUNT(*) FROM purchase_orders",
        "Suppliers":        "SELECT COUNT(DISTINCT supplier) FROM purchase_orders",
        "Date range":       "SELECT MIN(order_date)::text || ' → ' || MAX(order_date)::text FROM purchase_orders",
        "KPI rows":         "SELECT COUNT(*) FROM supplier_kpi",
    }
    with engine.connect() as conn:
        for label, q in queries.items():
            val = conn.execute(text(q)).scalar()
            print(f"   {label}: {val}")


if __name__ == "__main__":
    print("=" * 55)
    print("  🏭 Procurement DB Setup")
    print(f"  Target: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print("=" * 55)

    try:
        engine = get_engine()
        setup_schema(engine)
        load_csv(engine)
        refresh_materialized_views(engine)
        verify_setup(engine)
        print("\n✅ Database setup complete! Ready for dashboard.\n")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure PostgreSQL is running")
        print("  2. Create the DB:  createdb procurement_db")
        print("  3. Set env vars:   DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        raise
