"""
Page 3: Backtesting Engine — Prove strategies work (or don't) on historical data.
"""
import streamlit as st
import pandas as pd
from core.backtester import run_backtest, create_backtest_chart, run_multi_stock_backtest
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from utils.config import STRATEGY_WEIGHTS, POPULAR_INDIAN as POPULAR_INDIAN_ORIG
from utils.helpers import format_currency, format_percentage
from datetime import datetime

# Rename to avoid shadow conflict
POPULAR_INDIAN = [s.replace(".NS", "") for s in POPULAR_INDIAN_ORIG]

st.set_page_config(page_title="Backtester", page_icon="📉", layout="wide")
st.title("📉 Backtesting Engine")

st.markdown("""
> **Real trading rule:** *If you can't make money on historical data, you won't make money on live data.*
>
> This backtester simulates trades including slippage (0.05%), brokerage (0.03%), and STT (0.1%).
""")

tab1, tab2, tab3 = st.tabs(["🎯 Single Stock Backtest", "📊 Multi-Stock Comparison", "📖 How to Read Results"])

# ────────────────────────────────────────────────────────────────
# TAB 1: Single Stock Backtest
# ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Single Stock Backtest")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", help="e.g., AAPL, RELIANCE.NS")
    with col2:
        period = st.selectbox("Backtest Period", ["1y", "2y", "5y"], index=1,
                             help="Longer period = more reliable results")
    with col3:
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, step=10000)

    col_risk, col_mode = st.columns(2)
    with col_risk:
        risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5,
                                   help="% of capital risked per trade")
    with col_mode:
        strategy_preset = st.selectbox("Strategy", ["Combined (Default)", "RSI-Heavy", "MACD-Heavy", "MA-Heavy", "Bollinger-Heavy"])

    if st.button("▶️ Run Backtest", type="primary"):
        # Adjust weights based on preset
        weights = dict(STRATEGY_WEIGHTS)
        if strategy_preset == "RSI-Heavy":
            weights = {"rsi": 0.55, "macd": 0.15, "ma_crossover": 0.15, "bollinger": 0.15}
        elif strategy_preset == "MACD-Heavy":
            weights = {"rsi": 0.15, "macd": 0.55, "ma_crossover": 0.15, "bollinger": 0.15}
        elif strategy_preset == "MA-Heavy":
            weights = {"rsi": 0.15, "macd": 0.15, "ma_crossover": 0.55, "bollinger": 0.15}
        elif strategy_preset == "Bollinger-Heavy":
            weights = {"rsi": 0.15, "macd": 0.15, "ma_crossover": 0.15, "bollinger": 0.55}

        with st.spinner(f"Running backtest on {symbol} for {period}..."):
            result = run_backtest(symbol, period=period, initial_capital=initial_capital,
                                  risk_per_trade=risk_per_trade, custom_weights=weights)

        if "error" in result:
            st.error(result["error"])
        else:
            metrics = result["metrics"]

            # ── Key Metrics Row ──
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("📈 Total Return", format_percentage(metrics["total_return_pct"]),
                          delta=format_percentage(metrics["cagr"]) + " CAGR")
            with c2:
                st.metric("💵 Net Profit", format_currency(metrics["total_profit"]))
            with c3:
                st.metric("📊 Win Rate", f"{metrics['win_rate']}%",
                          delta=f"{metrics['winners']}W/{metrics['losers']}L")
            with c4:
                st.metric("⚡ Sharpe Ratio", metrics["sharpe_ratio"],
                          delta=f"Sortino: {metrics['sortino_ratio']}")
            with c5:
                st.metric("💧 Max Drawdown", f"{metrics['max_drawdown']}%",
                          delta=f"Calmar: {metrics['calmar_ratio']}")

            # ── Trade Stats ──
            with st.expander("📋 Detailed Trade Statistics", expanded=False):
                col_stats = st.columns(3)
                with col_stats[0]:
                    st.write("**Performance**")
                    st.write(f"- Total Trades: {metrics['total_trades']}")
                    st.write(f"- Winners: {metrics['winners']} | Losers: {metrics['losers']}")
                    st.write(f"- Win Rate: {metrics['win_rate']}%")
                    st.write(f"- Profit Factor: {metrics['profit_factor']}")
                    st.write(f"- Max Consecutive Wins: {metrics['max_consec_wins']}")
                with col_stats[1]:
                    st.write("**Trade Size**")
                    st.write(f"- Avg Win: {format_currency(metrics['avg_win'])}")
                    st.write(f"- Avg Loss: {format_currency(metrics['avg_loss'])}")
                    st.write(f"- Avg Holding: {metrics['avg_holding_days']} days")
                    st.write(f"- Total Commission: {format_currency(metrics['total_commission'])}")
                with col_stats[2]:
                    st.write("**Risk Metrics**")
                    st.write(f"- Sharpe Ratio: {metrics['sharpe_ratio']}")
                    st.write(f"- Sortino Ratio: {metrics['sortino_ratio']}")
                    st.write(f"- Max Drawdown: {metrics['max_drawdown']}%")
                    st.write(f"- Drawdown Duration: {metrics['max_dd_duration_days']} days")
                    st.write(f"- Calmar Ratio: {metrics['calmar_ratio']}")

            # ── Equity Curve ──
            st.markdown("### Equity Curve & Trade History")
            chart = create_backtest_chart(result)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            # ── Trade List ──
            if result["trades"]:
                st.markdown("### Trade Log")
                trades_df = pd.DataFrame(result["trades"])
                # Only show key columns
                display_cols = ["entry_date", "exit_date", "entry_price", "exit_price",
                                "shares", "net_pnl", "exit_reason", "holding_days"]
                display_df = trades_df[[c for c in display_cols if c in trades_df.columns]]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ── Verdict ──
            st.markdown("---")
            sharpe = metrics["sharpe_ratio"]
            win_rate = metrics["win_rate"]
            max_dd = metrics["max_drawdown"]
            total_return = metrics["total_return_pct"]

            if sharpe > 1.5 and win_rate > 50 and max_dd < 20 and total_return > 0:
                st.success(f"✅ **{symbol}: Strategy looks promising!** Sharpe {sharpe}, Win Rate {win_rate}%, DD {max_dd}%")
            elif sharpe > 0.8 and win_rate > 45:
                st.info(f"⚠️ **{symbol}: Mixed results.** Needs optimization. Sharpe {sharpe}")
            else:
                st.error(f"❌ **{symbol}: Strategy NOT profitable.** Sharpe {sharpe}, Return {total_return}%. Don't trade this.")

# ────────────────────────────────────────────────────────────────
# TAB 2: Multi-Stock Comparison
# ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Multi-Stock Backtest Comparison")
    st.markdown("Run the same strategy across multiple stocks to see where it works best.")

    multi_period = st.selectbox("Period", ["1y", "2y", "5y"], index=1, key="multi_period")
    multi_capital = st.number_input("Initial Capital", value=100000, key="multi_cap")

    candidate_pool = POPULAR_INDIAN
    selected = st.multiselect(
        f"Select stocks to backtest (from {len(candidate_pool)} popular stocks):",
        options=candidate_pool,
        default=candidate_pool[:5],
    )

    if st.button("▶️ Run Comparison", type="primary") and selected:
        symbols = [f"{s}.NS" if not s.endswith(".NS") else s for s in selected]

        with st.spinner(f"Running backtest across {len(symbols)} stocks..."):
            results = run_multi_stock_backtest(symbols, period=multi_period,
                                               initial_capital=multi_capital)

        if not results.empty:
            st.markdown("### 📊 Results by Stock")
            st.dataframe(results, use_container_width=True, hide_index=True)

            # Highlight best
            best = results.iloc[0]
            st.success(f"🥇 **Best performer: {best['symbol']}** — "
                       f"Sharpe {best['sharpe_ratio']}, Return {best['total_return_pct']:.1f}%, "
                       f"Win Rate {best['win_rate']}%")
        else:
            st.warning("No results. Check if the symbols are valid.")

# ────────────────────────────────────────────────────────────────
# TAB 3: Guide
# ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## 📖 Understanding Backtest Results

    ### Key Metrics

    | Metric | What It Tells You | Good | Bad |
    |--------|-------------------|------|-----|
    | **Sharpe Ratio** | Risk-adjusted return | >1.5 | <0.5 |
    | **Win Rate** | % of profitable trades | >55% | <40% |
    | **Profit Factor** | Gross profit / gross loss | >2.0 | <1.5 |
    | **Max Drawdown** | Worst peak-to-trough drop | <15% | >30% |
    | **Calmar Ratio** | CAGR / Max DD | >2.0 | <0.5 |
    | **Sortino Ratio** | Like Sharpe but ignores upside | >2.0 | <0.5 |

    ### Red Flags

    ⚠️ **High win rate + low profit factor** = Many small wins, one big loser
    ⚠️ **High Sharpe + only 1 year data** = Not tested through bear market
    ⚠️ **Max drawdown > 30%** = Would you have held through this?
    ⚠️ **< 20 trades total** = Not statistically significant

    ### Real Trading Rule
    > **If Sharpe < 1.0 → Don't trade it.**
    > **If Win Rate < 40% → You need very high risk-reward to compensate.**
    > **If Max DD > 25% → Your psychology will break before the strategy does.**
    """)
