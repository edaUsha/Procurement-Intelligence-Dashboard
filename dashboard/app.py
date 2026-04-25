"""
Procurement Intelligence Dashboard
Main Streamlit App — app.py
Run: streamlit run dashboard/app.py
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Path setup so imports work from any directory ─────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DATA_PATH   = os.path.join(ROOT, "data", "procurement_data.csv")
MODEL_DIR   = os.path.join(ROOT, "models", "saved_models")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background-color: #0d0f14;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #2a2d35;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #161922 0%, #1c2030 100%);
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.green::before  { background: linear-gradient(90deg, #22c55e, #16a34a); }
.kpi-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #1d4ed8); }
.kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #d97706); }
.kpi-card.red::before    { background: linear-gradient(90deg, #ef4444, #dc2626); }

.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #f1ede6;
    line-height: 1;
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    margin-top: 6px;
}
.kpi-delta.up   { color: #22c55e; }
.kpi-delta.down { color: #ef4444; }
.kpi-delta.neutral { color: #6b7280; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #e8e4dc;
    letter-spacing: 0.5px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2d35;
    margin-bottom: 16px;
}

/* Risk badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
}
.badge-red    { background: #3f1212; color: #f87171; border: 1px solid #7f1d1d; }
.badge-amber  { background: #3f2a00; color: #fbbf24; border: 1px solid #78350f; }
.badge-green  { background: #052e16; color: #4ade80; border: 1px solid #14532d; }

/* Live feed */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Divider */
hr { border-color: #2a2d35 !important; }

/* Plotly charts background */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Streamlit metric override */
[data-testid="stMetric"] {
    background: #161922;
    border: 1px solid #2a2d35;
    border-radius: 10px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Data Loaders (CSV-based, TTL cache for live refresh) ──────────────────────
@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Order_Date", "Delivery_Date"])
    df.columns = [c.lower() for c in df.columns]
    df["price_gap"]     = df["unit_price"] - df["negotiated_price"]
    df["savings_total"] = df["price_gap"] * df["quantity"]
    df["order_value"]   = df["quantity"] * df["negotiated_price"]
    df["defect_rate"]   = (df["defective_units"] / df["quantity"]).round(4)
    df["lead_time_days"]= (df["delivery_date"] - df["order_date"]).dt.days
    df["order_month"]   = df["order_date"].dt.to_period("M").astype(str)
    return df


@st.cache_resource
def load_ml_models():
    models = {}
    try:
        with open(os.path.join(MODEL_DIR, "compliance_model.pkl"), "rb") as f:
            models["compliance"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "compliance_encoders.pkl"), "rb") as f:
            models["compliance_enc"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "defect_model.pkl"), "rb") as f:
            models["defect"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "defect_encoders.pkl"), "rb") as f:
            models["defect_enc"] = pickle.load(f)

        # Load prophet models per category
        models["prophet"] = {}
        for cat in ["Raw_Materials", "Packaging", "Electronics", "Machinery", "Consumables"]:
            path = os.path.join(MODEL_DIR, f"prophet_{cat}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models["prophet"][cat.replace("_", " ")] = pickle.load(f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Some models not loaded: {e}")
    return models


# ── Plotly Theme ──────────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono", color="#9ca3af", size=11),
    xaxis=dict(gridcolor="#1f2333", linecolor="#2a2d35", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1f2333", linecolor="#2a2d35", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
)
COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a78bfa"]
SUPPLIER_COLORS = {
    "AlphaSupplies": "#3b82f6",
    "BetaCorp":      "#f59e0b",
    "GammaTrade":    "#22c55e",
    "DeltaWorks":    "#ef4444",
    "EpsilonMfg":    "#a78bfa",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style='padding: 8px 0 20px 0;'>
            <div style='font-family:Syne;font-size:20px;font-weight:800;color:#f1ede6;'>
                🏭 PROCUREMENT
            </div>
            <div style='font-family:DM Mono;font-size:10px;letter-spacing:3px;color:#6b7280;margin-top:2px;'>
                INTELLIGENCE DASHBOARD
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        page = st.radio(
            "NAVIGATE",
            ["📊 Overview", "🏢 Supplier Scorecard", "📈 Price Forecast",
             "📦 Order Tracker", "🚨 Anomaly Alerts", "🤖 ML Predictor"],
            label_visibility="visible"
        )

        st.markdown("---")
        st.markdown('<div style="font-family:DM Mono;font-size:10px;color:#6b7280;letter-spacing:2px;">FILTERS</div>', unsafe_allow_html=True)

        # Filters
        suppliers = ["All"] + sorted(df["supplier"].unique().tolist())
        sel_supplier = st.selectbox("Supplier", suppliers)

        categories = ["All"] + sorted(df["item_category"].unique().tolist())
        sel_category = st.selectbox("Category", categories)

        date_min = df["order_date"].min().date()
        date_max = df["order_date"].max().date()
        date_range = st.date_input("Date Range", value=(date_min, date_max),
                                   min_value=date_min, max_value=date_max)

        st.markdown("---")

        # Live refresh indicator
        st.markdown("""
        <div style='font-family:DM Mono;font-size:10px;color:#6b7280;'>
            <span class='live-dot'></span>AUTO-REFRESH: 60s
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(f"""
        <div style='font-family:DM Mono;font-size:9px;color:#4b5563;margin-top:8px;'>
            Last loaded: {datetime.now().strftime("%H:%M:%S")}
        </div>
        """, unsafe_allow_html=True)

    # Apply filters
    filtered = df.copy()
    if sel_supplier != "All":
        filtered = filtered[filtered["supplier"] == sel_supplier]
    if sel_category != "All":
        filtered = filtered[filtered["item_category"] == sel_category]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["order_date"].dt.date >= date_range[0]) &
            (filtered["order_date"].dt.date <= date_range[1])
        ]

    return page, filtered


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview(df):
    st.markdown('<div class="section-header">📊 Procurement Overview</div>', unsafe_allow_html=True)

    # ── KPI Cards ──
    total_spend    = df["order_value"].sum()
    total_savings  = df["savings_total"].sum()
    otd_pct        = (df["order_status"] == "Delivered").mean() * 100
    defect_rate    = (df["defective_units"].sum() / df["quantity"].sum()) * 100
    compliance_pct = (df["compliance"] == "Compliant").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "TOTAL SPEND",       f"${total_spend/1e6:.2f}M", "blue",   "↑ YTD", "neutral"),
        (c2, "COST SAVINGS",      f"${total_savings/1e3:.0f}K","green",  f"↑ {total_savings/total_spend*100:.1f}% savings rate", "up"),
        (c3, "ON-TIME DELIVERY",  f"{otd_pct:.1f}%",           "green" if otd_pct > 80 else "amber", "", "neutral"),
        (c4, "DEFECT RATE",       f"{defect_rate:.1f}%",       "red" if defect_rate > 5 else "amber", "↓ target <3%", "down"),
        (c5, "COMPLIANCE RATE",   f"{compliance_pct:.1f}%",    "green" if compliance_pct > 90 else "amber", "", "neutral"),
    ]
    for col, label, val, color, delta, delta_cls in cards:
        with col:
            st.markdown(f"""
            <div class='kpi-card {color}'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-delta {delta_cls}'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Monthly Spend Trend ──
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Monthly Spend & Savings</div>', unsafe_allow_html=True)
        monthly = df.groupby(df["order_date"].dt.to_period("M").astype(str)).agg(
            spend=("order_value", "sum"),
            savings=("savings_total", "sum"),
            orders=("po_id", "count")
        ).reset_index()
        monthly.columns = ["month", "spend", "savings", "orders"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["spend"]/1000,
                             name="Spend ($K)", marker_color="#3b82f6", opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["savings"]/1000,
                                 name="Savings ($K)", line=dict(color="#22c55e", width=2),
                                 mode="lines+markers", marker=dict(size=4)), secondary_y=True)
        fig.update_layout(**CHART_THEME, height=280, showlegend=True,
                          yaxis_title="Spend ($K)", yaxis2_title="Savings ($K)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Spend by Category</div>', unsafe_allow_html=True)
        cat_spend = df.groupby("item_category")["order_value"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=cat_spend["item_category"],
            values=cat_spend["order_value"],
            hole=0.6,
            marker_colors=COLORS,
            textfont=dict(family="DM Mono", size=10),
        ))
        fig2.update_layout(**CHART_THEME, height=280, showlegend=True,
                           annotations=[dict(text="SPEND", x=0.5, y=0.5,
                                            font=dict(size=12, family="DM Mono", color="#6b7280"),
                                            showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Order Status + Compliance ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Order Status Distribution</div>', unsafe_allow_html=True)
        status_counts = df["order_status"].value_counts().reset_index()
        status_colors = {"Delivered": "#22c55e", "Pending": "#f59e0b",
                         "Partial": "#3b82f6", "Cancelled": "#ef4444"}
        fig3 = px.bar(status_counts, x="order_status", y="count",
                      color="order_status",
                      color_discrete_map=status_colors)
        fig3.update_layout(**CHART_THEME, height=220, showlegend=False,
                           xaxis_title="", yaxis_title="Orders")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Compliance by Supplier</div>', unsafe_allow_html=True)
        comp = df.groupby("supplier").apply(
            lambda x: round((x["compliance"] == "Compliant").mean() * 100, 1)
        ).reset_index()
        comp.columns = ["supplier", "compliance_pct"]
        comp = comp.sort_values("compliance_pct")
        colors_bar = ["#ef4444" if v < 75 else "#f59e0b" if v < 90 else "#22c55e"
                      for v in comp["compliance_pct"]]
        fig4 = go.Figure(go.Bar(
            x=comp["compliance_pct"], y=comp["supplier"],
            orientation="h", marker_color=colors_bar,
            text=[f"{v}%" for v in comp["compliance_pct"]],
            textposition="outside", textfont=dict(family="DM Mono", size=10),
        ))
        fig4.update_layout(**CHART_THEME, height=220, xaxis_title="Compliance %",
                           xaxis_range=[0, 110], yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SUPPLIER SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_supplier_scorecard(df):
    st.markdown('<div class="section-header">🏢 Supplier Scorecard</div>', unsafe_allow_html=True)

    # Supplier KPI table
    supplier_kpi = df.groupby("supplier").agg(
        total_orders=("po_id", "count"),
        total_spend=("order_value", "sum"),
        total_savings=("savings_total", "sum"),
        avg_defect_rate=("defect_rate", "mean"),
        on_time_pct=("order_status", lambda x: round((x == "Delivered").mean() * 100, 1)),
        compliance_pct=("compliance", lambda x: round((x == "Compliant").mean() * 100, 1)),
        avg_lead_time=("lead_time_days", "mean"),
    ).reset_index()

    supplier_kpi["savings_rate"] = (supplier_kpi["total_savings"] / supplier_kpi["total_spend"] * 100).round(1)
    supplier_kpi["avg_defect_rate"] = (supplier_kpi["avg_defect_rate"] * 100).round(2)
    supplier_kpi["avg_lead_time"] = supplier_kpi["avg_lead_time"].round(1)

    # Risk score (simple composite)
    supplier_kpi["risk_score"] = (
        (100 - supplier_kpi["on_time_pct"]) * 0.35 +
        (100 - supplier_kpi["compliance_pct"]) * 0.35 +
        supplier_kpi["avg_defect_rate"] * 2
    ).round(1)

    def risk_badge(score):
        if score > 25:   return "🔴 High"
        elif score > 12: return "🟡 Medium"
        else:            return "🟢 Low"

    supplier_kpi["risk_level"] = supplier_kpi["risk_score"].apply(risk_badge)

    # Radar chart — multi-supplier comparison
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Performance Radar</div>', unsafe_allow_html=True)
        categories_radar = ["On-Time %", "Compliance %", "Savings Rate", "Quality Score"]
        fig = go.Figure()
        for _, row in supplier_kpi.iterrows():
            quality_score = max(0, 100 - row["avg_defect_rate"] * 10)
            values = [row["on_time_pct"], row["compliance_pct"],
                      min(row["savings_rate"] * 5, 100), quality_score]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories_radar + [categories_radar[0]],
                fill="toself", name=row["supplier"],
                line_color=SUPPLIER_COLORS.get(row["supplier"], "#888"),
                opacity=0.6,
            ))
        fig.update_layout(**CHART_THEME, height=340,
                          polar=dict(
                              bgcolor="rgba(0,0,0,0)",
                              radialaxis=dict(visible=True, range=[0, 100],
                                             gridcolor="#2a2d35", tickfont=dict(size=9)),
                              angularaxis=dict(gridcolor="#2a2d35")
                          ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Spend vs. Defect Rate</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            supplier_kpi, x="total_spend", y="avg_defect_rate",
            size="total_orders", color="supplier",
            color_discrete_map=SUPPLIER_COLORS,
            text="supplier", size_max=50,
        )
        fig2.update_traces(textposition="top center",
                           textfont=dict(family="DM Mono", size=9))
        fig2.update_layout(**CHART_THEME, height=340,
                           xaxis_title="Total Spend ($)", yaxis_title="Avg Defect Rate (%)",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # KPI Table
    st.markdown('<div class="section-header">Supplier KPI Summary</div>', unsafe_allow_html=True)
    display_cols = ["supplier", "total_orders", "total_spend", "savings_rate",
                    "on_time_pct", "compliance_pct", "avg_defect_rate", "avg_lead_time", "risk_level"]
    display_df = supplier_kpi[display_cols].copy()
    display_df["total_spend"] = display_df["total_spend"].apply(lambda x: f"${x:,.0f}")
    display_df["savings_rate"] = display_df["savings_rate"].apply(lambda x: f"{x}%")
    display_df["on_time_pct"] = display_df["on_time_pct"].apply(lambda x: f"{x}%")
    display_df["compliance_pct"] = display_df["compliance_pct"].apply(lambda x: f"{x}%")
    display_df["avg_defect_rate"] = display_df["avg_defect_rate"].apply(lambda x: f"{x}%")
    display_df["avg_lead_time"] = display_df["avg_lead_time"].apply(lambda x: f"{x} days")
    display_df.columns = ["Supplier", "Orders", "Spend", "Savings Rate",
                          "On-Time", "Compliance", "Defect Rate", "Avg Lead Time", "Risk"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRICE FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
def page_price_forecast(df, models):
    st.markdown('<div class="section-header">📈 Price Trend Forecasting</div>', unsafe_allow_html=True)

    if not models.get("prophet"):
        st.warning("⚠️ Prophet models not found. Run `python models/price_forecaster.py` first.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_cat = st.selectbox("Select Category", list(models["prophet"].keys()))
        forecast_months = st.slider("Forecast Months", 3, 12, 6)

    with col2:
        model = models["prophet"][selected_cat]
        future = model.make_future_dataframe(periods=forecast_months, freq="ME")
        fc = model.predict(future)

        # Actual data
        actual = df[df["item_category"] == selected_cat].groupby(
            df["order_date"].dt.to_period("M").apply(lambda x: x.to_timestamp())
        )["unit_price"].mean().reset_index()
        actual.columns = ["ds", "actual_price"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual["ds"], y=actual["actual_price"],
            name="Actual Price", mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"],
            name="Forecast", mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
        ))
        cutoff = df["order_date"].max()
        fig.add_vline(x=cutoff, line_dash="dash", line_color="#6b7280",
                      annotation_text="Forecast Start", annotation_font_color="#6b7280")
        fig.update_layout(**CHART_THEME, height=350,
                          title=dict(text=f"{selected_cat} — Unit Price Forecast",
                                     font=dict(family="Syne", size=14, color="#e8e4dc")),
                          yaxis_title="Avg Unit Price ($)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Trend summary
    last_actual = actual["actual_price"].iloc[-1]
    forecast_end = fc["yhat"].iloc[-1]
    change_pct = (forecast_end - last_actual) / last_actual * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current Avg Price", f"${last_actual:.2f}")
    with c2:
        st.metric(f"Forecast ({forecast_months}m)", f"${forecast_end:.2f}",
                  delta=f"{change_pct:+.1f}%")
    with c3:
        inflation_flag = "🔴 High Inflation Risk" if change_pct > 8 else \
                         "🟡 Moderate" if change_pct > 3 else "🟢 Stable"
        st.metric("Inflation Signal", inflation_flag)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ORDER TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
def page_order_tracker(df):
    st.markdown('<div class="section-header">📦 Live Order Tracker</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total POs", f"{len(df):,}")
    with col2: st.metric("Pending", f"{(df['order_status']=='Pending').sum():,}")
    with col3: st.metric("Partial", f"{(df['order_status']=='Partial').sum():,}")
    with col4: st.metric("Cancelled", f"{(df['order_status']=='Cancelled').sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters row
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        status_filter = st.multiselect("Filter by Status",
            ["Delivered", "Pending", "Partial", "Cancelled"],
            default=["Pending", "Partial", "Cancelled"])
    with col_b:
        search = st.text_input("Search PO ID / Supplier", "")
    with col_c:
        sort_by = st.selectbox("Sort by", ["order_date", "order_value", "defective_units", "lead_time_days"])

    view = df.copy()
    if status_filter:
        view = view[view["order_status"].isin(status_filter)]
    if search:
        view = view[
            view["po_id"].str.contains(search, case=False) |
            view["supplier"].str.contains(search, case=False)
        ]
    view = view.sort_values(sort_by, ascending=False)

    # Flag delayed orders
    view["delayed"] = (
        (view["order_status"] == "Pending") &
        ((datetime.now().date() - view["order_date"].dt.date).apply(lambda x: x.days if hasattr(x, 'days') else 0) > 30)
    )

    display = view[["po_id", "supplier", "order_date", "delivery_date", "item_category",
                     "order_status", "quantity", "order_value", "defective_units", "compliance"]].copy()
    display["order_date"]    = display["order_date"].dt.strftime("%Y-%m-%d")
    display["delivery_date"] = display["delivery_date"].dt.strftime("%Y-%m-%d").fillna("—")
    display["order_value"]   = display["order_value"].apply(lambda x: f"${x:,.0f}")
    display.columns = ["PO ID", "Supplier", "Order Date", "Delivery Date", "Category",
                       "Status", "Qty", "Value", "Defects", "Compliance"]

    st.dataframe(display.head(200), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(200, len(display))} of {len(view)} records")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANOMALY ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
def page_anomaly_alerts(df):
    st.markdown('<div class="section-header">🚨 Anomaly Alerts</div>', unsafe_allow_html=True)

    # Detect anomalies using IQR method
    def detect_outliers(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

    df_flagged = df.copy()
    df_flagged["price_anomaly"]  = detect_outliers(df["unit_price"])
    df_flagged["defect_anomaly"] = detect_outliers(df["defective_units"])
    df_flagged["qty_anomaly"]    = detect_outliers(df["quantity"])
    df_flagged["any_anomaly"]    = df_flagged[["price_anomaly", "defect_anomaly", "qty_anomaly"]].any(axis=1)

    anomalies = df_flagged[df_flagged["any_anomaly"]].copy()

    # Summary
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🔴 Total Anomalies", len(anomalies))
    with c2: st.metric("💰 Price Outliers",  df_flagged["price_anomaly"].sum())
    with c3: st.metric("⚠️ Defect Outliers", df_flagged["defect_anomaly"].sum())

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Price Anomalies Over Time</div>', unsafe_allow_html=True)
        fig = go.Figure()
        normal = df_flagged[~df_flagged["price_anomaly"]]
        outlier = df_flagged[df_flagged["price_anomaly"]]
        fig.add_trace(go.Scatter(x=normal["order_date"], y=normal["unit_price"],
                                 mode="markers", name="Normal",
                                 marker=dict(color="#3b82f6", size=3, opacity=0.5)))
        fig.add_trace(go.Scatter(x=outlier["order_date"], y=outlier["unit_price"],
                                 mode="markers", name="Anomaly",
                                 marker=dict(color="#ef4444", size=7, symbol="x")))
        fig.update_layout(**CHART_THEME, height=280,
                          yaxis_title="Unit Price ($)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Defect Anomalies by Supplier</div>', unsafe_allow_html=True)
        defect_anom = df_flagged[df_flagged["defect_anomaly"]].groupby("supplier").size().reset_index()
        defect_anom.columns = ["supplier", "anomaly_count"]
        fig2 = px.bar(defect_anom, x="supplier", y="anomaly_count",
                      color="supplier", color_discrete_map=SUPPLIER_COLORS)
        fig2.update_layout(**CHART_THEME, height=280, showlegend=False,
                           xaxis_title="", yaxis_title="Anomaly Count")
        st.plotly_chart(fig2, use_container_width=True)

    # Anomaly table
    st.markdown('<div class="section-header">Flagged Orders</div>', unsafe_allow_html=True)
    anomaly_display = anomalies[["po_id", "supplier", "order_date", "item_category",
                                  "unit_price", "defective_units", "quantity",
                                  "price_anomaly", "defect_anomaly"]].copy()
    anomaly_display["order_date"] = anomaly_display["order_date"].dt.strftime("%Y-%m-%d")
    anomaly_display["flags"] = anomaly_display.apply(
        lambda r: " | ".join(filter(None, [
            "💰 Price" if r["price_anomaly"] else "",
            "🔬 Defect" if r["defect_anomaly"] else "",
        ])), axis=1
    )
    anomaly_display = anomaly_display[["po_id","supplier","order_date","item_category",
                                       "unit_price","defective_units","quantity","flags"]]
    anomaly_display.columns = ["PO ID","Supplier","Date","Category",
                                "Unit Price","Defects","Qty","Flags"]
    st.dataframe(anomaly_display.head(100), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
def page_ml_predictor(df, models):
    st.markdown('<div class="section-header">🤖 ML Predictor</div>', unsafe_allow_html=True)
    st.caption("Enter PO details to get ML-powered compliance risk and defect predictions.")

    if not models.get("compliance") or not models.get("defect"):
        st.warning("⚠️ ML models not found. Run the model training scripts first.")
        return

    suppliers   = sorted(df["supplier"].unique().tolist())
    categories  = sorted(df["item_category"].unique().tolist())
    statuses    = ["Delivered", "Pending", "Partial"]

    with st.form("predictor_form"):
        st.markdown("**Order Details**")
        c1, c2, c3 = st.columns(3)
        with c1:
            supplier  = st.selectbox("Supplier", suppliers)
            category  = st.selectbox("Item Category", categories)
            status    = st.selectbox("Order Status", statuses)
        with c2:
            quantity  = st.number_input("Quantity", min_value=1, max_value=10000, value=200)
            unit_price= st.number_input("Unit Price ($)", min_value=1.0, value=150.0, step=0.5)
            neg_price = st.number_input("Negotiated Price ($)", min_value=1.0, value=140.0, step=0.5)
        with c3:
            lead_days = st.number_input("Lead Time (days)", min_value=-1, max_value=180, value=20)
            order_month = st.selectbox("Order Month", list(range(1, 13)),
                                       format_func=lambda x: datetime(2024, x, 1).strftime("%B"))
            order_year  = st.selectbox("Order Year", [2022, 2023, 2024])

        submitted = st.form_submit_button("🔮 Run Predictions", use_container_width=True)

    if submitted:
        order_dict = {
            "supplier": supplier, "item_category": category, "order_status": status,
            "quantity": quantity, "unit_price": unit_price, "negotiated_price": neg_price,
            "lead_time_days": lead_days, "order_month": order_month, "order_year": order_year,
            "defect_rate": 0.05,
        }

        # ── Compliance Risk ──
        def predict_compliance(order, model, encoders):
            row = pd.DataFrame([order])
            for col in ["supplier", "item_category", "order_status"]:
                le = encoders[col]
                row[col + "_enc"] = le.transform(row[col].astype(str))
            row["price_gap"]      = row["unit_price"] - row["negotiated_price"]
            row["price_gap_pct"]  = row["price_gap"] / row["unit_price"]
            row["order_value"]    = row["quantity"] * row["negotiated_price"]
            features = ["supplier_enc","item_category_enc","order_status_enc",
                        "quantity","unit_price","negotiated_price","price_gap",
                        "price_gap_pct","defect_rate","order_month","order_year",
                        "order_value","lead_time_days"]
            return float(model.predict_proba(row[features])[0][1])

        def predict_defects(order, model, encoders):
            row = pd.DataFrame([order])
            for col in ["supplier", "item_category"]:
                le = encoders[col]
                row[col + "_enc"] = le.transform(row[col].astype(str))
            row["price_gap_pct"] = (row["unit_price"] - row["negotiated_price"]) / row["unit_price"]
            row["order_value"]   = row["quantity"] * row["negotiated_price"]
            features = ["supplier_enc","item_category_enc","quantity","unit_price",
                        "negotiated_price","price_gap_pct","order_month","order_year",
                        "order_value","lead_time_days"]
            pred = max(0, round(float(model.predict(row[features])[0])))
            return pred, round(pred / order["quantity"] * 100, 2)

        comp_risk  = predict_compliance(order_dict, models["compliance"], models["compliance_enc"])
        def_count, def_rate = predict_defects(order_dict, models["defect"], models["defect_enc"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔮 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            risk_label = "🔴 HIGH" if comp_risk > 0.6 else "🟡 MEDIUM" if comp_risk > 0.3 else "🟢 LOW"
            st.metric("Compliance Risk", risk_label)
        with r2:
            st.metric("Non-Compliance Probability", f"{comp_risk*100:.1f}%")
        with r3:
            st.metric("Predicted Defective Units", f"{def_count}")
        with r4:
            risk_lvl = "🔴 High" if def_rate > 8 else "🟡 Medium" if def_rate > 3 else "🟢 Low"
            st.metric("Defect Rate", f"{def_rate}% — {risk_lvl}")

        # Gauge chart for compliance risk
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=comp_risk * 100,
            title={"text": "Non-Compliance Risk Score", "font": {"family": "Syne", "color": "#e8e4dc"}},
            number={"suffix": "%", "font": {"family": "DM Mono", "color": "#e8e4dc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar": {"color": "#ef4444" if comp_risk > 0.6 else "#f59e0b" if comp_risk > 0.3 else "#22c55e"},
                "bgcolor": "#161922",
                "steps": [
                    {"range": [0, 30],   "color": "#052e16"},
                    {"range": [30, 60],  "color": "#3f2a00"},
                    {"range": [60, 100], "color": "#3f1212"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": comp_risk * 100},
            }
        ))
        fig.update_layout(**CHART_THEME, height=280)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    df     = load_data()
    models = load_ml_models()
    page, filtered_df = render_sidebar(df)

    if page == "📊 Overview":
        page_overview(filtered_df)
    elif page == "🏢 Supplier Scorecard":
        page_supplier_scorecard(filtered_df)
    elif page == "📈 Price Forecast":
        page_price_forecast(filtered_df, models)
    elif page == "📦 Order Tracker":
        page_order_tracker(filtered_df)
    elif page == "🚨 Anomaly Alerts":
        page_anomaly_alerts(filtered_df)
    elif page == "🤖 ML Predictor":
        page_ml_predictor(filtered_df, models)


if __name__ == "__main__":
    main()