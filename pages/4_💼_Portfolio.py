"""
Page 4: Portfolio — Track holdings, P&L, ROI, and allocation.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from core.portfolio import get_portfolio_summary, get_trade_stats
from core.data_fetcher import get_stock_data
from utils.database import add_holding, remove_holding, get_portfolio, add_trade
from utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("💼 Portfolio Tracker")

# ── Get current prices ──
holdings = get_portfolio()
current_prices = {}

if holdings:
    with st.spinner("Fetching current prices..."):
        for h in holdings:
            try:
                df = get_stock_data(h["symbol"], period="5d")
                if not df.empty:
                    current_prices[h["symbol"]] = df["Close"].iloc[-1]
            except Exception:
                pass

summary = get_portfolio_summary(current_prices)
stats = get_trade_stats()

# ── Summary Metrics ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("💰 Total Invested", format_currency(summary["total_invested"]))
with c2:
    st.metric("📊 Current Value", format_currency(summary["current_value"]))
with c3:
    st.metric("📈 Total P&L", format_currency(summary["pnl"]),
              delta=format_percentage(summary["roi_pct"]))
with c4:
    st.metric("📦 Holdings", f"{summary['holdings_count']}")

st.markdown("---")

# ── Holdings Table ──
if summary["holdings"]:
    st.markdown("### 📋 Current Holdings")

    rows = []
    for h in summary["holdings"]:
        rows.append({
            "ID": h["id"],
            "Symbol": h["symbol"],
            "Market": h["market"],
            "Qty": h["quantity"],
            "Buy Price": format_currency(h["buy_price"]),
            "Current": format_currency(h["current_price"]),
            "Invested": format_currency(h["invested"]),
            "Value": format_currency(h["current_value"]),
            "P&L": f"{format_currency(h['pnl'])} ({format_percentage(h['pnl_pct'])})",
            "Stop Loss": format_currency(h["stop_loss"]) if h["stop_loss"] else "—",
            "Target": format_currency(h["target_price"]) if h["target_price"] else "—",
        })

    df_holdings = pd.DataFrame(rows)
    st.dataframe(df_holdings.drop(columns=["ID"]), use_container_width=True, hide_index=True)

    # Remove holding
    st.markdown("#### Remove Holding")
    remove_id = st.selectbox("Select holding to remove",
                             [(h["id"], f"{h['symbol']} ({h['quantity']} shares)") for h in summary["holdings"]],
                             format_func=lambda x: x[1])
    if st.button("🗑️ Remove", type="secondary"):
        remove_holding(remove_id[0])
        st.success(f"Removed {remove_id[1]}")
        st.rerun()

else:
    st.info("No holdings yet. Add your first position below!")

st.markdown("---")

# ── Add Holding Form ──
st.markdown("### ➕ Add New Holding")
with st.form("add_holding"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbol = st.text_input("Symbol", placeholder="AAPL / RELIANCE.NS")
    with col2:
        market = "INDIA"
    with col3:
        quantity = st.number_input("Quantity", min_value=1, value=10)
    with col4:
        buy_price = st.number_input("Buy Price", min_value=0.01, value=100.00, step=0.01)

    col5, col6, col7 = st.columns(3)
    with col5:
        buy_date = st.date_input("Buy Date", value=datetime.now())
    with col6:
        stop_loss = st.number_input("Stop Loss (optional)", min_value=0.0, value=0.0, step=0.01)
    with col7:
        target_price = st.number_input("Target Price (optional)", min_value=0.0, value=0.0, step=0.01)

    notes = st.text_input("Notes (optional)")

    submitted = st.form_submit_button("➕ Add Holding", type="primary")
    if submitted:
        if symbol and quantity > 0 and buy_price > 0:
            sym = symbol.upper()
            if market == "INDIA" and not sym.endswith((".NS", ".BO")):
                sym = f"{sym}.NS"
            add_holding(sym, market, quantity, buy_price, str(buy_date), stop_loss, target_price, notes)
            st.success(f"Added {quantity} shares of {sym} at {format_currency(buy_price)}")
            st.rerun()
        else:
            st.error("Please fill in all required fields.")

# ── Charts ──
if summary["holdings"]:
    st.markdown("---")
    st.markdown("### 📊 Portfolio Allocation")

    col_a, col_b = st.columns(2)

    with col_a:
        # Market allocation pie chart
        alloc = summary["allocation"]
        if alloc:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(alloc.keys()),
                values=list(alloc.values()),
                hole=0.4,
                marker=dict(colors=["#00D4AA", "#00A3FF", "#FFD93D", "#FF6B6B"]),
            )])
            fig_pie.update_layout(
                title="Market Allocation",
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                height=350,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # P&L bar chart
        pnl_data = [(h["symbol"], h["pnl"]) for h in summary["holdings"]]
        if pnl_data:
            fig_pnl = go.Figure(data=[go.Bar(
                x=[d[0] for d in pnl_data],
                y=[d[1] for d in pnl_data],
                marker_color=["#00D4AA" if d[1] >= 0 else "#FF4757" for d in pnl_data],
            )])
            fig_pnl.update_layout(
                title="P&L by Holding",
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                height=350,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

# ── Trade History ──
if stats["total_trades"] > 0:
    st.markdown("---")
    st.markdown("### 📜 Trade Statistics")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Trades", stats["total_trades"])
    with s2:
        st.metric("Win Rate", f"{stats['win_rate']}%")
    with s3:
        st.metric("Avg Profit", format_percentage(stats["avg_profit"]))
    with s4:
        st.metric("Total P&L", format_currency(stats["total_pnl"]))
