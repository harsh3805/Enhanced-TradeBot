"""
Stock Trading Assistant — Main App
A real-world trading assistant for Indian and US stock markets.
"""
import streamlit as st
from utils.database import init_db

# Initialize database
init_db()

st.set_page_config(
    page_title="Stock Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4AA, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1A1F2E;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #2A2F3E;
    }
    .signal-buy { color: #00D4AA; font-weight: bold; }
    .signal-sell { color: #FF4757; font-weight: bold; }
    .signal-hold { color: #FFD93D; font-weight: bold; }
    div[data-testid="stMetric"] {
        background-color: #1A1F2E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2A2F3E;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">📈 Stock Trading Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered trading decisions for Indian markets</div>', unsafe_allow_html=True)

# Quick overview
col1, col2, col3 = st.columns(3)

st.markdown('**AI Trading Assistant — Indian Markets (NSE/BSE)**', unsafe_allow_html=True)
st.caption('Real-time analysis, automated BUY/SELL signals, risk management')
st.markdown("---")

st.markdown("---")

# Getting started guide
st.markdown("### 🚀 Getting Started")

st.markdown("""
1. **Add stocks to your Watchlist** — Go to the Watchlist page and add stocks you want to track
2. **View Technical Analysis** — Pick any stock and analyze it with 10+ indicators
3. **Check Trading Signals** — Get automated BUY/SELL/HOLD signals with confidence scores
4. **Track Your Portfolio** — Add your holdings and monitor P&L in real-time
5. **Manage Risk** — Use position sizing and stop-loss calculators
6. **Screener** — Find stocks matching your technical criteria
""")

st.markdown("### 📊 Supported Indicators")
ind_col1, ind_col2, ind_col3 = st.columns(3)
with ind_col1:
    st.markdown("- RSI (Relative Strength Index)\n- MACD\n- Bollinger Bands")
with ind_col2:
    st.markdown("- SMA / EMA (20, 50, 200)\n- ATR (Average True Range)\n- VWAP")
with ind_col3:
    st.markdown("- Stochastic RSI\n- ADX (Trend Strength)\n- Support/Resistance")

st.markdown("---")
st.markdown("*Built with Streamlit, yfinance, and Plotly. Not financial advice.*")
