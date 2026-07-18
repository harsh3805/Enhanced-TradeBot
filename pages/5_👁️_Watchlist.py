"""
Page 5: Watchlist — Manage watched stocks with add/remove/group functionality.
"""
import streamlit as st
import pandas as pd
from core.data_fetcher import get_stock_data, search_stocks
from core.analyzer import calculate_all_indicators, get_latest_indicators
from core.strategies import combined_signal
from utils.database import add_to_watchlist, remove_from_watchlist, get_watchlist
from utils.helpers import signal_emoji, get_market_from_symbol
from utils.config import POPULAR_INDIAN as POPULAR_INDIAN_STOCKS

st.set_page_config(page_title="Watchlist", page_icon="👁️", layout="wide")
st.title("👁️ Watchlist Manager")

# ── Add Stock Form ──
st.markdown("### ➕ Add Stock to Watchlist")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    new_symbol = st.text_input("Stock Symbol", placeholder="AAPL / RELIANCE / TCS.NS")
with col2:
    new_market = st.selectbox("Market", ["US", "INDIA"])
with col3:
    new_sector = st.text_input("Sector (optional)", placeholder="Technology")

if st.button("➕ Add to Watchlist", type="primary"):
    if new_symbol:
        sym = new_symbol.upper()
        if new_market == "INDIA" and not sym.endswith((".NS", ".BO")):
            sym = f"{sym}.NS"
        try:
            add_to_watchlist(sym, new_market, new_sector)
            st.success(f"Added **{sym}** to watchlist!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ── Popular Stocks Quick Add ──
st.markdown("---")
st.markdown("### ⚡ Quick Add Popular Stocks")

st.markdown("**🇮🇳 Popular Indian Stocks**")
for sym in POPULAR_INDIAN_STOCKS[:10]:
    display_name = sym.replace(".NS", "")
    if st.button(f"+ {display_name}", key=f"add_{sym}", help=f"Add {sym} to watchlist"):
        add_to_watchlist(sym, "INDIA")
        st.success(f"Added {display_name}!")
        st.rerun()

st.markdown("---")

# ── Current Watchlist ──
watchlist = get_watchlist()

if watchlist:
    st.markdown(f"### 📋 Your Watchlist ({len(watchlist)} stocks)")

    # Group by market
    us_stocks = [w for w in watchlist if w["market"] == "US"]
    indian_stocks = [w for w in watchlist if w["market"] == "INDIA"]

    if us_stocks:
        st.markdown("#### 🇺🇸 US Stocks")
        with st.spinner("Loading US stocks..."):
            for w in us_stocks:
                sym = w["symbol"]
                try:
                    df = get_stock_data(sym, period="1mo")
                    if not df.empty and len(df) > 5:
                        df = calculate_all_indicators(df)
                        sig, conf, reason = combined_signal(df)
                        price = df["Close"].iloc[-1]
                        change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100 if len(df) > 1 else 0

                        cols = st.columns([2, 1, 1, 2, 1])
                        with cols[0]:
                            st.markdown(f"**{sym}**")
                        with cols[1]:
                            st.markdown(f"${price:.2f}")
                        with cols[2]:
                            color = "green" if change >= 0 else "red"
                            st.markdown(f":{color}[{change:+.2f}%]")
                        with cols[3]:
                            st.markdown(f"{signal_emoji(sig)} **{sig}** ({conf}%)")
                        with cols[4]:
                            if st.button("🗑️", key=f"rm_{sym}_{w['id']}"):
                                remove_from_watchlist(sym, "US")
                                st.rerun()
                except Exception:
                    cols = st.columns([2, 1, 1, 2, 1])
                    with cols[0]:
                        st.markdown(f"**{sym}**")
                    with cols[4]:
                        if st.button("🗑️", key=f"rm_{sym}_{w['id']}"):
                            remove_from_watchlist(sym, "US")
                            st.rerun()

    if indian_stocks:
        st.markdown("#### 🇮🇳 Indian Stocks")
        with st.spinner("Loading Indian stocks..."):
            for w in indian_stocks:
                sym = w["symbol"]
                try:
                    df = get_stock_data(sym, period="1mo")
                    if not df.empty and len(df) > 5:
                        df = calculate_all_indicators(df)
                        sig, conf, reason = combined_signal(df)
                        price = df["Close"].iloc[-1]
                        change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100 if len(df) > 1 else 0

                        cols = st.columns([2, 1, 1, 2, 1])
                        with cols[0]:
                            st.markdown(f"**{sym.replace('.NS', '')}**")
                        with cols[1]:
                            st.markdown(f"₹{price:.2f}")
                        with cols[2]:
                            color = "green" if change >= 0 else "red"
                            st.markdown(f":{color}[{change:+.2f}%]")
                        with cols[3]:
                            st.markdown(f"{signal_emoji(sig)} **{sig}** ({conf}%)")
                        with cols[4]:
                            if st.button("🗑️", key=f"rm_{sym}_{w['id']}"):
                                remove_from_watchlist(sym, "INDIA")
                                st.rerun()
                except Exception:
                    pass

    # Bulk actions
    st.markdown("---")
    if st.button("🗑️ Clear Entire Watchlist", type="secondary"):
        for w in watchlist:
            remove_from_watchlist(w["symbol"], w["market"])
        st.success("Watchlist cleared!")
        st.rerun()
else:
    st.info("Your watchlist is empty. Add stocks above to start tracking!")
