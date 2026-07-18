"""
Page 6: Alerts — Price and indicator-based alerts.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from utils.database import add_alert, remove_alert, get_active_alerts, get_triggered_alerts
from utils.helpers import format_currency

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")
st.title("🔔 Price & Indicator Alerts")

# ── Create Alert Form ──
st.markdown("### ➕ Create New Alert")
with st.form("create_alert"):
    col1, col2, col3 = st.columns(3)
    with col1:
        alert_symbol = st.text_input("Stock Symbol", placeholder="AAPL / RELIANCE.NS")
    with col2:
        alert_market = "INDIA"
    with col3:
        alert_type = st.selectbox("Alert Type", [
            "price_above", "price_below",
            "rsi_oversold", "rsi_overbought",
        ])

    target_value = 0.0
    if alert_type in ["price_above", "price_below"]:
        target_value = st.number_input("Target Price", min_value=0.01, value=100.00, step=0.01)
    elif alert_type == "rsi_oversold":
        st.info("Alerts when RSI drops below 30")
        target_value = 30
    elif alert_type == "rsi_overbought":
        st.info("Alerts when RSI rises above 70")
        target_value = 70

    if st.form_submit_button("🔔 Create Alert", type="primary"):
        if alert_symbol:
            sym = alert_symbol.upper()
            if alert_market == "INDIA" and not sym.endswith((".NS", ".BO")):
                sym = f"{sym}.NS"
            add_alert(sym, alert_market, alert_type, target_value)
            st.success(f"Alert created for **{sym}** ({alert_type})")
            st.rerun()

st.markdown("---")

# ── Active Alerts ──
active_alerts = get_active_alerts()

if active_alerts:
    st.markdown(f"### 🔔 Active Alerts ({len(active_alerts)})")

    for alert in active_alerts:
        sym = alert["symbol"]
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            st.markdown(f"**{sym}**")
        with col2:
            type_display = alert["alert_type"].replace("_", " ").title()
            st.markdown(f"{type_display}")
        with col3:
            if "price" in alert["alert_type"]:
                st.markdown(f"Target: {format_currency(alert['target_value'])}")
            else:
                st.markdown(f"Threshold: {alert['target_value']}")
        with col4:
            if st.button("🗑️", key=f"rm_alert_{alert['id']}"):
                remove_alert(alert["id"])
                st.rerun()

    # Check alerts
    st.markdown("---")
    if st.button("🔍 Check Alerts Now", type="secondary"):
        triggered = []
        for alert in active_alerts:
            sym = alert["symbol"]
            try:
                df = get_stock_data(sym, period="5d")
                if df.empty:
                    continue

                price = df["Close"].iloc[-1]
                should_trigger = False

                if alert["alert_type"] == "price_above" and price >= alert["target_value"]:
                    should_trigger = True
                elif alert["alert_type"] == "price_below" and price <= alert["target_value"]:
                    should_trigger = True
                elif alert["alert_type"] in ["rsi_oversold", "rsi_overbought"]:
                    df_ind = calculate_all_indicators(df)
                    rsi = df_ind["RSI"].iloc[-1]
                    if alert["alert_type"] == "rsi_oversold" and rsi < 30:
                        should_trigger = True
                    elif alert["alert_type"] == "rsi_overbought" and rsi > 70:
                        should_trigger = True

                if should_trigger:
                    triggered.append(alert)
                    from utils.database import mark_alert_triggered
                    mark_alert_triggered(alert["id"])
            except Exception:
                continue

        if triggered:
            st.success(f"🔔 {len(triggered)} alert(s) triggered!")
            for t in triggered:
                st.balloons()
                st.warning(f"**{t['symbol']}** — {t['alert_type'].replace('_', ' ').title()}")
        else:
            st.info("No alerts triggered right now.")

else:
    st.info("No active alerts. Create one above to get notified when conditions are met!")

# ── Triggered Alerts History ──
triggered_alerts = get_triggered_alerts()
if triggered_alerts:
    st.markdown("---")
    st.markdown("### ✅ Triggered Alerts History")
    df_triggered = pd.DataFrame([{
        "Symbol": a["symbol"],
        "Type": a["alert_type"].replace("_", " ").title(),
        "Target": a["target_value"],
        "Created": a["created_date"],
    } for a in triggered_alerts[:20]])
    st.dataframe(df_triggered, use_container_width=True, hide_index=True)
