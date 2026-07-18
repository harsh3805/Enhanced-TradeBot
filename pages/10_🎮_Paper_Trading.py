"""
Page 10: Paper Trading — Trade with virtual money before risking real capital.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from core.paper_trading import (
    get_account, get_open_positions, get_trade_journal,
    paper_buy, paper_sell, get_paper_performance, reset_account,
    check_paper_stops
)
from core.data_fetcher import get_stock_data, get_stock_info
from core.analyzer import calculate_all_indicators
from core.strategies import combined_signal, get_stop_loss_target
from core.advanced_risk import score_position_risk, get_risk_label, portfolio_correlation_check
from utils.helpers import format_currency, format_percentage, signal_emoji
from utils.config import DEFAULT_CAPITAL

st.set_page_config(page_title="Paper Trading", page_icon="🎮", layout="wide")
st.title("🎮 Paper Trading Simulator")

st.markdown("""
> **Rule #1:** If you can't make money with fake money, you **will** lose real money.
>
> **Rule #2:** Paper trade for at least 3 months before going live.
""")

# ── Account Summary ──
account = get_account()
perf = get_paper_performance()

# ── Auto-check stops on page load ──
positions = get_open_positions()
if positions:
    current_prices = {}
    for pos in positions:
        try:
            df = get_stock_data(pos["symbol"], period="5d")
            if not df.empty:
                current_prices[pos["symbol"]] = df["Close"].iloc[-1]
        except Exception:
            pass
    if current_prices:
        actions = check_paper_stops(current_prices)
        for action in actions:
            if action["result"].get("success"):
                emoji = "🟢" if "profit" in action["result"].get("message", "") else "🔴"
                st.toast(action["result"]["message"], icon=emoji)

# ── Account Header ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Cash Balance", format_currency(account["current_capital"]))
with col2:
    total_open_value = sum(pos["quantity"] * pos["entry_price"] for pos in positions) if positions else 0
    st.metric("📊 In Positions", format_currency(total_open_value))
with col3:
    total_equity = account["current_capital"] + total_open_value
    st.metric("💎 Total Equity", format_currency(total_equity))
with col4:
    roi = ((total_equity - account["initial_capital"]) / account["initial_capital"]) * 100
    st.metric("📈 ROI", format_percentage(roi), delta_color="off")

st.markdown("---")

# ── Two Column Layout: Trade Entry + Positions ──
col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown("### ➡️ New Trade")
    with st.form("new_trade"):
        trade_symbol = st.text_input("Symbol", placeholder="AAPL / RELIANCE.NS")
        trade_market = "INDIA"
        trade_action = st.radio("Action", ["BUY"], horizontal=True)

        with st.expander("⚙️ Advanced", expanded=True):
            trade_qty = st.number_input("Quantity", min_value=1, value=10)
            trade_price = st.number_input("Price (leave 0 for market price)", min_value=0.0, value=0.0, step=0.01)
            cols_sl_tp = st.columns(2)
            trade_sl = cols_sl_tp[0].number_input("Stop Loss", min_value=0.0, value=0.0, step=0.01)
            trade_tp = cols_sl_tp[1].number_input("Target", min_value=0.0, value=0.0, step=0.01)
            trade_strategy = st.text_input("Strategy", placeholder="e.g., RSI Oversold", value="")
            trade_notes = st.text_area("Notes", placeholder="Why this trade?", height=60)

        submitted = st.form_submit_button("📈 Execute BUY", type="primary")
        if submitted and trade_symbol:
            sym = trade_symbol.upper()
            if trade_market == "INDIA" and not sym.endswith((".NS", ".BO")):
                sym = f"{sym}.NS"

            # Auto-get current price if not specified
            if trade_price <= 0:
                try:
                    df = get_stock_data(sym, period="5d")
                    if not df.empty:
                        trade_price = df["Close"].iloc[-1]
                except Exception:
                    st.error("Could not fetch price. Enter manually.")
                    st.stop()

            result = paper_buy(sym, trade_qty, trade_price, stop_loss=trade_sl,
                              target_price=trade_tp, strategy=trade_strategy, notes=trade_notes)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result["message"])
                st.balloons()
                st.rerun()

with col_b:
    # ── Open Positions ──
    st.markdown("### 📋 Open Positions")
    if positions:
        # Fetch current prices for all positions
        position_rows = []
        for pos in positions:
            try:
                df = get_stock_data(pos["symbol"], period="5d")
                if not df.empty:
                    current_price = df["Close"].iloc[-1]
                    pnl = (current_price - pos["entry_price"]) * pos["quantity"]
                    pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
                    position_value = pos["quantity"] * current_price

                    # Calculate risk score
                    total_account = account["current_capital"] + sum(
                        p["quantity"] * current_price for p in positions
                    )
                    risk_score = score_position_risk(
                        pos["symbol"], pos["entry_price"], pos["stop_loss"] or pos["entry_price"] * 0.95,
                        position_value, total_account
                    )

                    position_rows.append({
                        "ID": pos["id"],
                        "Symbol": pos["symbol"].replace(".NS", ""),
                        "Qty": pos["quantity"],
                        "Entry": pos["entry_price"],
                        "Current": current_price,
                        "Value": round(position_value, 2),
                        "P&L": round(pnl, 2),
                        "P&L%": round(pnl_pct, 2),
                        "SL": pos.get("stop_loss", 0) or "—",
                        "TP": pos.get("target_price", 0) or "—",
                        "Risk": get_risk_label(risk_score),
                        "Status": "🟢" if pnl >= 0 else "🔴",
                    })
                else:
                    position_rows.append({
                        "ID": pos["id"],
                        "Symbol": pos["symbol"].replace(".NS", ""),
                        "Qty": pos["quantity"],
                        "Entry": pos["entry_price"],
                        "Current": "N/A", "Value": 0,
                        "P&L": 0, "P&L%": 0,
                        "SL": pos.get("stop_loss", 0) or "—",
                        "TP": pos.get("target_price", 0) or "—",
                        "Risk": "N/A", "Status": "⚪",
                    })
            except Exception:
                continue

        if position_rows:
            df_positions = pd.DataFrame(position_rows)
            st.dataframe(df_positions, use_container_width=True, hide_index=True)

            # Close position selector
            st.markdown("#### Close Position")
            close_col1, close_col2 = st.columns([3, 1])
            with close_col1:
                close_id = st.selectbox(
                    "Select position to close:",
                    options=[r["ID"] for r in position_rows],
                    format_func=lambda x: f"{next(r['Symbol'] for r in position_rows if r['ID'] == x)} "
                                f"(Entry: {next(r['Entry'] for r in position_rows if r['ID'] == x)})"
                )
            with close_col2:
                close_reason = st.selectbox("Reason", ["Manual Close", "Take Profit", "Stop Loss", "Strategy Change"])

            if st.button("🔴 Sell / Close Position", type="secondary"):
                # Get current price
                pos_info = next((r for r in position_rows if r["ID"] == close_id), None)
                if pos_info and pos_info["Current"] != "N/A":
                    result = paper_sell(close_id, pos_info["Current"], reason=close_reason)
                    st.toast(result.get("message", "Position closed"))
                    st.rerun()
                else:
                    # Try to fetch price
                    pos_data = next((p for p in positions if p["id"] == close_id), None)
                    if pos_data:
                        df = get_stock_data(pos_data["symbol"], period="5d")
                        if not df.empty:
                            result = paper_sell(close_id, df["Close"].iloc[-1], reason=close_reason)
                            st.toast(result.get("message", "Position closed"))
                            st.rerun()
    else:
        st.info("No open positions. Start trading above!")

st.markdown("---")

# ── Performance Dashboard ──
st.markdown("### 📊 Performance Dashboard")

if perf["total_trades"] > 0:
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("📊 Total Trades", perf["total_trades"])
    with m2:
        st.metric("✅ Win Rate", f"{perf['win_rate']}%")
    with m3:
        st.metric("💰 Total P&L", format_currency(perf["total_pnl"]))
    with m4:
        st.metric("📈 Profit Factor", perf["profit_factor"])
    with m5:
        st.metric("💧 Max DD", f"{perf['max_drawdown']}%")

    # Trade Journal
    st.markdown("### 📜 Trade Journal")
    trades = get_trade_journal()
    if trades:
        df_trades = pd.DataFrame(trades)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)

else:
    st.info("No trades completed yet. Start trading to see performance here.")

# ── Reset Account ──
st.markdown("---")
with st.expander("⚙️ Account Settings"):
    col1, col2 = st.columns([1, 3])
    with col1:
        new_capital = st.number_input("Starting Capital", min_value=10000, value=100000, step=10000)
    with col2:
        if st.button("🔄 Reset Account (Clear All Trades)", type="secondary"):
            reset_account(new_capital)
            st.success(f"Account reset with {format_currency(new_capital)}!")
            st.rerun()
