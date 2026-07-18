"""
Page 8: Stock Screener — Filter stocks by technical criteria.
"""
import streamlit as st
import pandas as pd
from core.screener import screen_stocks
from utils.config import POPULAR_INDIAN as POPULAR_INDIAN_STOCKS
from utils.helpers import signal_emoji

st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide")
st.title("🔍 Stock Screener")

st.markdown("Filter stocks based on technical indicators. Select your criteria and click **Screen**.")

# ── Filter Panel ──
with st.sidebar:
    st.markdown("### 🔧 Filter Criteria")

    # Stock universe
    universe = st.multiselect(
        "Stock Universe",
        ["Indian Popular", "Custom"],
        default=["Indian Popular"],
    )

    custom_symbols = ""
    if "Custom" in universe:
        custom_symbols = st.text_area("Custom Symbols (one per line)", value="RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK")

    st.markdown("---")
    st.markdown("### 📊 Indicator Filters")

    # RSI
    use_rsi = st.checkbox("Filter by RSI")
    rsi_min = st.slider("RSI Min", 0, 100, 30) if use_rsi else 0
    rsi_max = st.slider("RSI Max", 0, 100, 70) if use_rsi else 100

    # MACD
    macd_filter = st.selectbox("MACD Signal", ["Any", "Positive (Bullish)", "Negative (Bearish)"])

    # Price vs MA
    ma_filter = st.selectbox("Price vs Moving Average", ["Any", "Above SMA 20", "Below SMA 20", "Above SMA 50"])

    # Volume
    use_volume = st.checkbox("Filter by Minimum Volume")
    min_volume = st.number_input("Min Daily Volume", value=1000000, step=100000) if use_volume else 0

# ── Build symbols list ──
symbols = []
if "Indian Popular" in universe:
    symbols.extend([s.replace(".NS", "") for s in POPULAR_INDIAN_STOCKS])
if custom_symbols:
    symbols.extend([s.strip().upper() for s in custom_symbols.split("\n") if s.strip()])

# ── Build filters ──
filters = {}
if use_rsi:
    filters["rsi_min"] = rsi_min
    filters["rsi_max"] = rsi_max

if macd_filter == "Positive (Bullish)":
    filters["macd_positive"] = True
elif macd_filter == "Negative (Bearish)":
    filters["macd_negative"] = True

if ma_filter == "Above SMA 20":
    filters["price_above_sma20"] = True
elif ma_filter == "Below SMA 20":
    filters["price_below_sma20"] = True
elif ma_filter == "Above SMA 50":
    filters["price_above_sma50"] = True

if use_volume:
    filters["volume_min"] = min_volume

# ── Screen Button ──
st.markdown("---")
if st.button("🔍 Screen Stocks", type="primary", disabled=not symbols):
    with st.spinner(f"Screening {len(symbols)} stocks..."):
        results = screen_stocks(symbols, filters)

    if not results.empty:
        st.markdown(f"### 📊 Results: {len(results)} stocks match your criteria")

        # Display results
        st.dataframe(results, use_container_width=True, hide_index=True)

        # Summary stats
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_rsi = results["RSI"].mean() if "RSI" in results.columns else 0
            st.metric("Avg RSI", f"{avg_rsi:.1f}")
        with c2:
            bullish = len(results[results["MACD"] > 0]) if "MACD" in results.columns else 0
            st.metric("MACD Bullish", f"{bullish}/{len(results)}")
        with c3:
            avg_vol = results["Volume"].mean() if "Volume" in results.columns else 0
            st.metric("Avg Volume", f"{avg_vol:,.0f}")
    else:
        st.warning("No stocks match your criteria. Try relaxing the filters.")

elif not symbols:
    st.info("Select a stock universe or enter custom symbols to start screening.")

# ── Preset Screener Strategies ──
st.markdown("---")
st.markdown("### 🎯 Preset Screener Strategies")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟢 Oversold (RSI < 30)", use_container_width=True):
        st.session_state["preset"] = "oversold"
with col2:
    if st.button("🔴 Overbought (RSI > 70)", use_container_width=True):
        st.session_state["preset"] = "overbought"
with col3:
    if st.button("📈 MACD Bullish", use_container_width=True):
        st.session_state["preset"] = "macd_bullish"

preset = st.session_state.get("preset")
if preset:
    preset_filters = {}
    if preset == "oversold":
        preset_filters = {"rsi_max": 30}
        st.info("Screening for oversold stocks (RSI < 30)...")
    elif preset == "overbought":
        preset_filters = {"rsi_min": 70}
        st.info("Screening for overbought stocks (RSI > 70)...")
    elif preset == "macd_bullish":
        preset_filters = {"macd_positive": True}
        st.info("Screening for MACD bullish stocks...")

    if symbols:
        with st.spinner("Running preset screen..."):
            results = screen_stocks(symbols, preset_filters)
        if not results.empty:
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.warning("No stocks match this preset strategy.")
