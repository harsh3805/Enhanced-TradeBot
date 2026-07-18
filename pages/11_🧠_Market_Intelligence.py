"""
Page 11: Market Intelligence + Advanced Risk — FII/DII flows, earnings calendar, trailing stops, correlation.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from core.institutional import get_fii_dii_flow_summary, get_fii_dii_daily, get_fii_signal
from core.earnings import get_upcoming_earnings, get_earnings_dates, is_near_earnings
from core.advanced_risk import (
    check_daily_loss_limit, check_drawdown, calculate_trailing_stop,
    portfolio_correlation_check, calculate_portfolio_var, smart_position_size,
    score_position_risk, get_risk_label,
)
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from utils.database import get_portfolio, get_watchlist
from utils.config import POPULAR_INDIAN as POPULAR_INDIAN_STOCKS
from utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Market Intelligence", page_icon="🧠", layout="wide")
st.title("🧠 Market Intelligence & Advanced Risk")

# ────────────────────────────────────────────────────────────────
# TAB 1: FII/DII Flow
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ FII/DII Flow (India)",
    "📅 Earnings Calendar",
    "🛡️ Advanced Risk Tools",
    "📊 Portfolio Risk"
])

with tab1:
    st.markdown("### 🏛️ Institutional Flow Tracker (Indian Markets)")
    st.markdown("Track FII (Foreign) and DII (Domestic) investment flows — the smart money moves.")

    col_fii, col_dii, col_signal = st.columns(3)

    # Get FII data
    summary = get_fii_dii_flow_summary()
    flow_df = get_fii_dii_daily()

    if summary and not flow_df.empty:
        with col_fii:
            fii_net = summary.get("fii_net", 0)
            fii_color = "inverse" if fii_net >= 0 else "normal"
            st.metric("💰 FII Net Flow (10 days)",
                     f"₹{fii_net:,.0f} Cr",
                     delta=f"{'Buying +' if fii_net >= 0 else 'Selling'}",
                     delta_color=fii_color)

        with col_dii:
            dii_net = summary.get("dii_net", 0)
            dii_color = "inverse" if dii_net >= 0 else "normal"
            st.metric("🏛️ DII Net Flow (10 days)",
                     f"₹{dii_net:,.0f} Cr",
                     delta=f"{'Buying +' if dii_net >= 0 else 'Selling'}",
                     delta_color=dii_color)

        with col_signal:
            signal_text, confidence, reason = get_fii_signal()
            sig_emoji = "🟢" if signal_text == "BULLISH" else "🔴" if signal_text == "BEARISH" else "🟡"
            st.metric("🎯 FII Signal", f"{sig_emoji} {signal_text}",
                     delta=f"Confidence: {confidence}%")

        # FII/DII Flow Chart
        st.markdown("### 📊 FII/DII Daily Net Flow")
        if "FII Net" in flow_df.columns and "DII Net" in flow_df.columns:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=flow_df["Date"] if "Date" in flow_df.columns else flow_df.index,
                y=flow_df["FII Net"],
                name="FII Net",
                marker_color="#00D4AA",
                opacity=0.7,
            ))
            fig.add_trace(go.Bar(
                x=flow_df["Date"] if "Date" in flow_df.columns else flow_df.index,
                y=flow_df["DII Net"],
                name="DII Net",
                marker_color="#FFD93D",
                opacity=0.7,
            ))

            # Zero line
            fig.add_hline(y=0, line_color="white", opacity=0.3)

            fig.update_layout(
                title="Daily FII vs DII Net Flow (₹ Crores)",
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=400,
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"📊 FII: Buy ₹{summary.get('fii_buy', 0):,.0f} | Sell ₹{summary.get('fii_sell', 0):,.0f} | Net ₹{summary.get('fii_net', 0):,.0f} Cr")
        st.info(f"📊 DII: Buy ₹{summary.get('dii_buy', 0):,.0f} | Sell ₹{summary.get('dii_sell', 0):,.0f} | Net ₹{summary.get('dii_net', 0):,.0f} Cr")

        st.warning("""
        **How to use FII/DII data:**
        - When **FIIs buy heavily** (>₹5000 Cr/week) → Bullish signal for Indian market
        - When **FIIs sell heavily** (<₹-5000 Cr/week) → Bearish signal
        - When **DIIs buy while FIIs sell** → DIIs are providing support
        - **Best used with:** Technical analysis on individual stocks
        """)
    else:
        st.warning("FII/DII data unavailable. NSE may have blocked the endpoint or the market is closed.")
        st.info("💡 This data refreshes every hour. Only available during Indian market hours (9:15-15:30 IST).")

# ────────────────────────────────────────────────────────────────
# TAB 2: Earnings Calendar
# ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📅 Earnings Calendar")

    col_market = st.columns(2)
    with col_market[0]:
        source = st.radio("Check earnings for:", ["Watchlist", "Popular Stocks"], horizontal=True)

    watchlist = get_watchlist()
    if source == "Watchlist" and watchlist:
        symbols = [w["symbol"] for w in watchlist]
    else:
        symbols = POPULAR_INDIAN_STOCKS[:15]

    with st.spinner("Fetching upcoming earnings..."):
        earnings = get_upcoming_earnings(symbols)

    if earnings:
        st.success(f"Found {len(earnings)} upcoming earnings events")

        df_earnings = pd.DataFrame(earnings)
        # Format
        if "days_until" in df_earnings.columns:
            df_earnings["Status"] = df_earnings["days_until"].apply(lambda x:
                "🔴 This Week!" if 0 < x <= 5 else
                "🟡 Next Week" if 5 < x <= 14 else
                "🟢 Upcoming" if x > 14 else
                "📊 Recent"
            )

        st.dataframe(df_earnings.sort_values("days_until"), use_container_width=True, hide_index=True)

        # Highlight near-term earnings
        near_term = [e for e in earnings if 0 < e.get("days_until", 999) <= 7]
        if near_term:
            st.warning(f"⚠️ **{len(near_term)} stocks with earnings this week!**")
            for e in near_term:
                st.warning(f"🔴 **{e['symbol']}** — Earnings in {e['days_until']} day(s)")
    else:
        st.info("No upcoming earnings found for your selected stocks. Try a different set of stocks.")

    # Check individual stock
    st.markdown("#### Check Specific Stock")
    check_sym = st.text_input("Enter symbol to check earnings", value="AAPL")
    if st.button("Check") and check_sym:
        dates = get_earnings_dates(check_sym)
        if not dates.empty:
            st.write(dates)
        else:
            st.info(f"No earnings data for {check_sym}")

    st.info("""
    **💡 Earnings Tips:**
    - **Avoid holding through earnings** unless you have a specific catalyst thesis
    - IV (implied volatility) tends to be high before earnings
    - Consider waiting for post-earnings drift
    - A single earnings miss can wipe out 6 months of gains
    """)

# ────────────────────────────────────────────────────────────────
# TAB 3: Advanced Risk Tools
# ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🛡️ Advanced Risk Management Tools")

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🎯 Trailing Stop-Loss", "📐 Smart Position Sizing", "📖 Correlation Check"])

    with sub_tab1:
        st.markdown("#### Trailing Stop-Loss Calculator")
        st.markdown("A trailing stop moves with the price — locks in profits, limits losses.")

        col1, col2, col3 = st.columns(3)
        with col1:
            entry = st.number_input("Entry Price", value=100.0, step=1.0)
        with col2:
            current_price = st.number_input("Current Price", value=115.0, step=1.0)
        with col3:
            atr = st.number_input("ATR (Average True Range)", value=2.5, step=0.1)

        method = st.selectbox("Trailing Stop Method", ["ATR (2x ATR)", "Percentage (2%)", "Chandelier Exit"])
        trail_pct = {"ATR (2x ATR)": 2.0, "Percentage (2%)": 2.0, "Chandelier Exit": 3.0}

        if st.button("Calculate Trailing Stop"):
            method_key = "atr" if "ATR" in method else "percentage" if "Percentage" in method else "chandelier"
            trailing_sl = calculate_trailing_stop(
                entry, current_price, current_price * 1.1,
                atr, method=method_key, trail_pct=trail_pct[method]
            )

            gain_pct = ((current_price - entry) / entry) * 100
            locked_pct = ((current_price - trailing_sl) / entry) * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("💰 Unrealized Gain", format_percentage(gain_pct))
            with c2:
                st.metric("🛡️ Trailing Stop", format_currency(trailing_sl))
            with c3:
                st.metric("🔒 Locked Profit", format_percentage(locked_pct))

            st.success(f"At ₹{current_price:.2f}, your trailing stop is at **₹{trailing_sl:.2f}** — "
                      f"your profit is locked at {locked_pct:.1f}%")

    with sub_tab2:
        st.markdown("#### Smart Position Sizer")
        st.markdown("Adjusts for volatility, correlation, and current positions.")

        col1, col2 = st.columns(2)
        with col1:
            smart_capital = st.number_input("Smart Capital", value=100000.0, step=10000.0, key="smart_cap")
            smart_entry = st.number_input("Smart Entry Price", value=100.0, step=1.0, key="smart_entry")
            smart_sl = st.number_input("Smart Stop Loss", value=95.0, step=1.0, key="smart_sl")
        with col2:
            smart_risk = st.slider("Smart Risk Per Trade (%)", 0.5, 5.0, 2.0, key="smart_risk")
            smart_max = st.number_input("Smart Max Positions", min_value=1, value=5, key="smart_max")
            smart_current = st.number_input("Smart Current Open", min_value=0, value=0, key="smart_cur")

        if st.button("Calculate Smart Size"):
            smart_symbol_entry = st.text_input("Symbol (for volatility check)", value="AAPL",
                                              key="smart_sym") if False else "AAPL"
            smart_shares = smart_position_size(
                smart_capital, smart_risk, smart_entry, smart_sl,
                smart_symbol_entry, max_positions=smart_max, current_positions=smart_current
            )
            cost = smart_shares * smart_entry
            position_pct = (cost / smart_capital) * 100

            st.metric("📊 Recommended Shares", f"{smart_shares}")
            st.metric("💰 Total Cost", format_currency(cost))
            st.metric("📏 Position Size", f"{position_pct:.1f}% of capital")

    with sub_tab3:
        st.markdown("#### Portfolio Correlation Check")
        st.markdown("Are your positions too correlated? If they all move together, you have less diversification than you think.")

        if watchlist:
            symbols_list = [w["symbol"] for w in watchlist]
            corr_df, warnings = portfolio_correlation_check(symbols_list)

            if warnings:
                st.warning(f"⚠️ Found {len(warnings)} high correlations:")
                for w in warnings:
                    st.warning(w)
            else:
                st.success("✅ No high correlations found in your watchlist!")

            if corr_df is not None:
                st.markdown("#### Correlation Matrix")
                st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)
        else:
            st.info("Add stocks to your watchlist to check correlations.")

# ────────────────────────────────────────────────────────────────
# TAB 4: Portfolio Risk
# ────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📊 Portfolio Risk Assessment")

    holdings = get_portfolio()

    if holdings:
        # Get current prices
        current_prices = {}
        for h in holdings:
            try:
                df = get_stock_data(h["symbol"], period="5d")
                if not df.empty:
                    current_prices[h["symbol"]] = df["Close"].iloc[-1]
            except Exception:
                pass

        total_capital = sum(current_prices.get(h["symbol"], h["buy_price"]) * h["quantity"]
                           for h in holdings)

        # Value at Risk
        positions_for_var = [{"symbol": h["symbol"], "value": current_prices.get(h["symbol"], h["buy_price"]) * h["quantity"]}
                            for h in holdings]
        var_95 = calculate_portfolio_var(positions_for_var, confidence=0.95)

        # Position risk scoring
        risk_items = []
        for h in holdings:
            current = current_prices.get(h["symbol"], h["buy_price"])
            value = current * h["quantity"]
            score = score_position_risk(h["symbol"], h["buy_price"], h["stop_loss"] or h["buy_price"] * 0.95,
                                        value, total_capital)
            risk_items.append({
                "Symbol": h["symbol"],
                "Value": format_currency(value),
                "Risk Score": score,
                "Risk Label": get_risk_label(score),
                "Allocation%": round((value / total_capital) * 100, 1),
            })
            # Check concentration
            if (value / total_capital) * 100 > 20:
                st.warning(f"⚠️ **{h['symbol']}** is >20% of portfolio — high concentration risk")

        st.markdown("#### Position Risk Scores")
        st.dataframe(pd.DataFrame(risk_items), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 VaR (95%, 1-day)", f"{var_95}%",
                     help="Value at Risk: Maximum expected loss with 95% confidence")
        with col2:
            st.metric("🔢 Total Positions", f"{len(holdings)}")

    else:
        st.info("Add holdings to your portfolio to see risk assessment.")

    # Daily Loss Limit Check
    st.markdown("#### Daily Loss Limits")
    col1, col2, col3 = st.columns(3)
    with col1:
        daily_limit = st.number_input("Max Daily Loss (%)", value=5.0, step=0.5)
    with col2:
        max_dd = st.number_input("Max Drawdown (%)", value=15.0, step=2.5)
    with col3:
        st.markdown("---")
        st.info(f"⚠️ At **₹{total_capital:,.2f}**, your daily stop is at **₹{total_capital * (1-daily_limit/100):,.2f}**")

    st.markdown("---")
    st.warning("""
    ## 🚨 Risk Management Rules

    | Rule | Setting | When to Stop Trading |
    |------|---------|---------------------|
    | **Daily Loss Limit** | -5% of capital | Hit limit → Stop for the day |
    | **Weekly Loss Limit** | -10% of capital | Hit limit → Stop for the week |
    | **Max Drawdown** | -15% from peak | Hit limit → Review entire strategy |
    | **Position Concentration** | Max 20% per stock | Exceed → Rebalance |
    | **Max Positions** | 5-8 at a time | Over + → You can't watch them all |
    """)
