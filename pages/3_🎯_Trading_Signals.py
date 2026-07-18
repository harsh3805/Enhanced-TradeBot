"""
Page 3: Trading Signals — Automated buy/sell/hold signals with confidence scores.
"""
import streamlit as st
import pandas as pd
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from core.strategies import combined_signal, rsi_strategy, macd_strategy, ma_crossover_strategy, bollinger_strategy, get_stop_loss_target
from utils.database import get_watchlist
from utils.helpers import signal_emoji, format_currency
from utils.config import POPULAR_INDIAN

st.set_page_config(page_title="Trading Signals", page_icon="🎯", layout="wide")
st.title("🎯 Trading Signals")

# ── Source Selection ──
source = st.radio("Stock Source", ["Watchlist", "Custom Input"], horizontal=True)

symbols = []
if source == "Watchlist":
    watchlist = get_watchlist()
    symbols = [w["symbol"] for w in watchlist]
    if not symbols:
        st.warning("Your watchlist is empty. Add stocks from the **Watchlist** page.")
else:
    custom = st.text_input("Enter symbols (comma-separated)", value="AAPL, RELIANCE.NS, MSFT, TCS.NS")
    symbols = [s.strip().upper() for s in custom.split(",") if s.strip()]

# ── Generate Signals ──
if symbols:
    if st.button("🎯 Generate Signals", type="primary"):
        results = []
        progress = st.progress(0)

        for i, sym in enumerate(symbols):
            progress.progress((i + 1) / len(symbols))
            try:
                df = get_stock_data(sym, period="6mo")
                if df.empty or len(df) < 30:
                    continue

                df = calculate_all_indicators(df)
                sig, conf, reason = combined_signal(df)
                sl, tp, rr = get_stop_loss_target(df, sig)

                # Individual strategy signals
                rsi_sig, rsi_conf, rsi_reason = rsi_strategy(df)
                macd_sig, macd_conf, macd_reason = macd_strategy(df)
                ma_sig, ma_conf, ma_reason = ma_crossover_strategy(df)
                bb_sig, bb_conf, bb_reason = bollinger_strategy(df)

                price = df["Close"].iloc[-1]

                results.append({
                    "Symbol": sym,
                    "Price": round(price, 2),
                    "Signal": f"{signal_emoji(sig)} {sig}",
                    "Confidence": f"{conf}%",
                    "Stop Loss": round(sl, 2) if sl else "—",
                    "Target": round(tp, 2) if tp else "—",
                    "R:R": rr if rr else "—",
                    "RSI": f"{rsi_sig}({rsi_conf}%)",
                    "MACD": f"{macd_sig}({macd_conf}%)",
                    "MA": f"{ma_sig}({ma_conf}%)",
                    "Bollinger": f"{bb_sig}({bb_conf}%)",
                })
            except Exception as e:
                st.warning(f"Error processing {sym}: {e}")
                continue

        progress.empty()

        if results:
            st.markdown("### 📊 Signal Summary")
            df_results = pd.DataFrame(results)

            # Color code the signal column
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Detailed breakdown for each stock
            st.markdown("---")
            st.markdown("### 🔍 Detailed Breakdown")

            for r in results:
                with st.expander(f"{signal_emoji(r['Signal'].split()[1])} {r['Symbol']} — {r['Signal']} ({r['Confidence']})"):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("RSI Signal", r["RSI"])
                    with c2:
                        st.metric("MACD Signal", r["MACD"])
                    with c3:
                        st.metric("MA Signal", r["MA"])
                    with c4:
                        st.metric("Bollinger Signal", r["Bollinger"])

                    if r["Stop Loss"] != "—":
                        st.info(f"💡 **Suggested:** Entry at ₹{r['Price'] if '.NS' in r['Symbol'] else r['Price']} | "
                                f"Stop Loss: {r['Stop Loss']} | Target: {r['Target']} | Risk:Reward = 1:{r['R:R']}")
        else:
            st.warning("No signals generated. Check if the symbols are valid.")
else:
    st.info("Select a stock source and click **Generate Signals** to get started.")

# ── Quick Signal Reference ──
st.markdown("---")
st.markdown("### 📖 Signal Strategy Guide")
with st.expander("How do the signals work?"):
    st.markdown("""
    **Combined Signal** weights 4 strategies equally (25% each):

    | Strategy | BUY Signal | SELL Signal |
    |----------|-----------|------------|
    | **RSI** | RSI < 30 (oversold) | RSI > 70 (overbought) |
    | **MACD** | MACD crosses above signal | MACD crosses below signal |
    | **MA Crossover** | SMA20 > SMA50 (Golden Cross) | SMA20 < SMA50 (Death Cross) |
    | **Bollinger** | Price at lower band | Price at upper band |

    **Confidence** ranges from 0-100% based on how strongly the indicators agree.
    - 80-100%: Strong signal — multiple indicators agree
    - 50-79%: Moderate signal — most indicators agree
    - 20-49%: Weak signal — mixed indicators
    - 0-19%: No clear signal — HOLD recommended
    """)
