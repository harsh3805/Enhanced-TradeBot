"""
Page 2: Technical Analysis — Interactive charts with indicators.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from core.data_fetcher import get_stock_data, get_stock_info
from core.analyzer import calculate_all_indicators, get_latest_indicators, get_trend, get_support_resistance
from utils.helpers import format_currency

st.set_page_config(page_title="Technical Analysis", page_icon="📈", layout="wide")
st.title("📈 Technical Analysis")

# ── Sidebar Controls ──
with st.sidebar:
    st.markdown("### Stock Selection")
    symbol = st.text_input("Enter Stock Symbol", value="AAPL", help="e.g., AAPL, RELIANCE.NS, MSFT")
    market = "INDIA"
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

    st.markdown("---")
    st.markdown("### Indicators")
    show_sma = st.checkbox("SMA (20/50/200)", value=True)
    show_ema = st.checkbox("EMA (12/26)", value=False)
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_volume = st.checkbox("Volume", value=True)
    show_vwap = st.checkbox("VWAP", value=False)
    show_atr = st.checkbox("ATR", value=False)

if market == "INDIA" and not symbol.endswith((".NS", ".BO")):
    symbol = f"{symbol}.NS"

# ── Fetch Data ──
with st.spinner(f"Loading data for {symbol}..."):
    df = get_stock_data(symbol, period=period)

if df.empty:
    st.error(f"Could not fetch data for **{symbol}**. Please check the symbol and try again.")
    st.stop()

df = calculate_all_indicators(df)
info = get_stock_info(symbol)
trend, confidence, trend_details = get_trend(df)
support, resistance = get_support_resistance(df)

# ── Company Info ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Price", format_currency(info.get("price", df["Close"].iloc[-1]), info.get("currency", "USD")))
with col2:
    st.metric("📊 Trend", f"{trend} ({confidence}%)")
with col3:
    st.metric("🏢 Sector", info.get("sector", "N/A"))
with col4:
    if info.get("pe_ratio"):
        st.metric("P/E Ratio", f"{info['pe_ratio']:.1f}")

st.markdown("---")

# ── Main Chart ──
num_indicators = sum([show_rsi, show_macd])
rows = 1 + num_indicators
row_heights = [0.6] + [0.2] * num_indicators

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=[""] * rows,
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#00D4AA", decreasing_line_color="#FF4757",
    ),
    row=1, col=1,
)

# Moving Averages
if show_sma:
    for col, color in [("SMA_20", "#FFD93D"), ("SMA_50", "#FF6B6B"), ("SMA_200", "#A855F7")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col.replace("_", " "),
                         line=dict(width=1, color=color)), row=1, col=1)

if show_ema:
    for col, color in [("EMA_12", "#00BFFF"), ("EMA_26", "#FF1493")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col.replace("_", " "),
                         line=dict(width=1, dash="dot", color=color)), row=1, col=1)

# Bollinger Bands
if show_bb and "BB_Upper" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                 line=dict(width=1, color="rgba(255,255,255,0.3)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                 line=dict(width=1, color="rgba(255,255,255,0.3)"),
                 fill="tonexty", fillcolor="rgba(255,255,255,0.05)"), row=1, col=1)

# VWAP
if show_vwap and "VWAP" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                 line=dict(width=1, color="#FF69B4", dash="dash")), row=1, col=1)

# Support/Resistance lines
for s in support:
    fig.add_hline(y=s, line_dash="dot", line_color="#00D4AA", opacity=0.4, row=1, col=1)
for r in resistance:
    fig.add_hline(y=r, line_dash="dot", line_color="#FF4757", opacity=0.4, row=1, col=1)

current_row = 2

# RSI
if show_rsi and "RSI" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                 line=dict(width=1.5, color="#00BFFF")), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4757", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00D4AA", opacity=0.5, row=current_row, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
    current_row += 1

# MACD
if show_macd and "MACD" in df.columns:
    colors = ["#00D4AA" if v >= 0 else "#FF4757" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
                 marker_color=colors, opacity=0.5), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                 line=dict(width=1.5, color="#00BFFF")), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                 line=dict(width=1, color="#FF6B6B")), row=current_row, col=1)
    fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    current_row += 1

# Volume
if show_volume:
    colors = ["#00D4AA" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#FF4757"
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                 marker_color=colors, opacity=0.4), row=1, col=1)

fig.update_layout(
    height=700,
    template="plotly_dark",
    showlegend=True,
    xaxis_rangeslider_visible=False,
    margin=dict(l=50, r=20, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
)

st.plotly_chart(fig, use_container_width=True)

# ── Indicator Values Table ──
st.markdown("### 📊 Latest Indicator Values")
indicators = get_latest_indicators(df)
cols = st.columns(4)
for i, (name, value) in enumerate(indicators.items()):
    with cols[i % 4]:
        st.metric(name, value)

# ── Trend & S/R ──
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### 📊 Trend Analysis")
    st.write(f"**Overall Trend:** {trend} ({confidence}% confidence)")
    if trend_details:
        st.write(f"- Bullish signals: {trend_details.get('bullish', 0)}")
        st.write(f"- Bearish signals: {trend_details.get('bearish', 0)}")

with col_b:
    st.markdown("### 🔑 Support & Resistance")
    if support:
        st.write("**Support Levels:**", ", ".join(f"₹{s}" if market == "INDIA" else f"${s}" for s in support))
    else:
        st.write("**Support:** No clear levels found")
    if resistance:
        st.write("**Resistance Levels:**", ", ".join(f"₹{r}" if market == "INDIA" else f"${r}" for r in resistance))
    else:
        st.write("**Resistance:** No clear levels found")
